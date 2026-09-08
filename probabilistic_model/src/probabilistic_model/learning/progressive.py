from __future__ import annotations

# %% imports
from collections import deque
import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Optional

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax.scipy.special import logsumexp as jax_logsumexp

from random_events.variable import Variable

from probabilistic_model.distributions.distributions import (
    DiscreteDistribution,
    SymbolicDistribution,
)
from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit as RxPc,
    SumUnit,
    Unit,
    LeafUnit,
    InnerUnit,
)


# %% pc-column
@dataclass
class PCColumn:
    """A single task-specific copy of a probabilistic circuit template."""

    task_id: str
    column_root: Unit
    variable_domain: Optional[tuple[Variable, ...]] = None
    task_context: dict[str, Any] = field(default_factory=dict)
    node_ids: set[int] = field(default_factory=set, init=False)
    """Reference ids (:attr:`Unit.index`) of every unit owned by this column."""
    leaf_ids: set[int] = field(default_factory=set, init=False)
    """Reference ids (:attr:`Unit.index`) of the leaf units owned by this column."""

    def __post_init__(self):
        if self.variable_domain is None:
            self.variable_domain = tuple(self.column_root.variables)
        elif not set(self.variable_domain).issubset(set(self.column_root.variables)):
            raise ValueError(
                "The variable domain must be compatible with the column's root circuit."
            )
        self._index_nodes()

    def _index_nodes(self) -> None:
        """
        Walk the column from :attr:`column_root` and record every unit's
        reference id in :attr:`node_ids`, and each leaf's id in
        :attr:`leaf_ids`.

        :attr:`column_root` must already carry its final, mounted
        :attr:`Unit.index` when this runs, since that id is what later
        identifies which column a unit belongs to.
        """
        queue: deque[Unit] = deque([self.column_root])
        while queue:
            node = queue.popleft()
            if node.index in self.node_ids:
                continue
            self.node_ids.add(node.index)
            if isinstance(node, LeafUnit):
                self.leaf_ids.add(node.index)
            for child in node.subcircuits:
                if not (isinstance(node, SumUnit) and isinstance(child, SumUnit)):
                    queue.append(child)


# %% column-mask
@dataclass(frozen=True)
class ColumnMask:
    """The units of a mounted progressive circuit that a learning step may update."""

    sum_nodes: frozenset[SumUnit]
    """Sum units whose mixture weights the maximization step may overwrite."""
    leaf_nodes: frozenset[LeafUnit]
    """Leaf units whose distribution parameters the maximization step may overwrite."""


# %% jax em numerics
@jax.jit
def _edge_responsibility(
    log_beta_parent: jax.Array,
    log_edge_weight: jax.Array,
    log_child_likelihood: jax.Array,
    log_parent_likelihood: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    Compute a sum edge's per-sample log-responsibility and its expected count.

    :param log_beta_parent: The parent unit's per-sample log-responsibility.
    :param log_edge_weight: The edge's log mixture weight.
    :param log_child_likelihood: The child unit's per-sample log-likelihood.
    :param log_parent_likelihood: The parent unit's per-sample log-likelihood.
    :return: The edge's per-sample log-responsibility and its expected count summed
        over samples.
    """
    log_gamma = (
        log_beta_parent + log_edge_weight + log_child_likelihood - log_parent_likelihood
    )
    log_gamma = jnp.where(jnp.isnan(log_gamma), -jnp.inf, log_gamma)
    expected_count = jnp.exp(jax_logsumexp(log_gamma))
    return log_gamma, expected_count


@jax.jit
def _normalized_log_weights(counts: jax.Array) -> jax.Array:
    """
    Normalize strictly positive expected counts into log mixture weights.

    :param counts: Smoothed expected counts per child.
    :return: The log of each count's share of the total, floored at ``1e-12``.
    """
    proportions = counts / jnp.sum(counts)
    return jnp.log(jnp.maximum(proportions, 1e-12))


@jax.jit
def _weighted_gaussian_parameters(
    values: jax.Array, weights: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """
    Compute the weighted mean and standard deviation of ``values``.

    :param values: Per-sample observations of the Gaussian's variable.
    :param weights: Non-negative posterior responsibilities per sample.
    :return: The weighted mean and standard deviation, with the variance floored at
        ``1e-6``.
    """
    total_weight = jnp.sum(weights)
    mean = jnp.sum(weights * values) / total_weight
    variance = jnp.sum(weights * (values - mean) ** 2) / total_weight
    return mean, jnp.sqrt(jnp.maximum(variance, 1e-6))


# %% progressive-pc
@dataclass
class ProgressivePC:
    """Parent-owned progressive circuit that manages task-specific columns."""

    ppc: RxPc = field(init=False)
    root_unit: Unit = field(init=False)
    template_pc: RxPc
    variable_domain: Optional[tuple[Variable, ...]] = None
    columns: list[PCColumn] = field(default_factory=list)
    task_context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.ppc = RxPc()
        self.root_unit = SumUnit(probabilistic_circuit=self.ppc)
        if self.template_pc is None:
            raise ValueError("template_pc must not be None.")
        if self.variable_domain is None:
            self.variable_domain = tuple(self.template_pc.variables)
        else:
            self.variable_domain = tuple(self.variable_domain)
        self._validate_template_domain()
        # Clear out Sum on Sums #TODO check if needed etc. wegen Cloumn to Column Swapping Detection
        self.template_pc.simplify()

    @classmethod
    def from_template(
        cls,
        template_pc: RxPc,
        variable_domain: Optional[Iterable[Variable]] = None,
        task_context: Optional[dict[str, Any]] = None,
    ) -> "ProgressivePC":
        return cls(
            template_pc=template_pc,
            variable_domain=(
                tuple(variable_domain) if variable_domain is not None else None
            ),
            task_context=task_context or {},
        )

    def _validate_template_domain(self):
        if not set(self.variable_domain).issubset(set(self.template_pc.variables)):
            raise ValueError(
                "The template variable domain must be compatible with the base circuit."
            )

    def iter_aligned_nodes(
        self, left: PCColumn, right: PCColumn
    ) -> Iterator[tuple[Unit, Unit]]:
        queue: deque[tuple[Unit, Unit]] = deque([(left.column_root, right.column_root)])
        visited: set[tuple[int, int]] = set()
        while queue:
            left_node, right_node = queue.popleft()
            pair_key = (left_node.index, right_node.index)
            if pair_key in visited:
                continue
            visited.add(pair_key)
            # Ensure node types match at the current position
            if type(left_node) is not type(right_node):
                raise ValueError(
                    f"Columns diverged: node types differ ({type(left_node).__name__} vs {type(right_node).__name__})."
                )
            yield left_node, right_node
            # Filter out cross-column lateral edges (SumUnit -> SumUnit)
            # TODO: Check wie ist mit LEAFS da die ja anders sein können oder Domain Postion gleich auch?
            left_children = [
                child
                for child in left_node.subcircuits
                if not (isinstance(left_node, SumUnit) and isinstance(child, SumUnit))
            ]
            right_children = [
                child
                for child in right_node.subcircuits
                if not (isinstance(right_node, SumUnit) and isinstance(child, SumUnit))
            ]
            if len(left_children) != len(right_children):
                raise ValueError(
                    f"Columns diverged: different number of children ({len(left_children)} vs {len(right_children)})."
                )
            for left_child, right_child in zip(left_children, right_children):
                queue.append((left_child, right_child))

    def add_column(
        self, task_id: str, task_context: Optional[dict[str, Any]] = None
    ) -> PCColumn:
        """Add a new task-specific column to the progressive circuit."""
        new_column_pc = copy.deepcopy(self.template_pc)
        new_column_pc_root = new_column_pc.root
        new_column_pc_index = new_column_pc_root.index
        mounted = self.ppc.mount(new_column_pc_root)
        true_column_root = mounted[new_column_pc_index]
        column = PCColumn(
            task_id=task_id,
            column_root=true_column_root,
            variable_domain=self.variable_domain,
            task_context=task_context or {},
        )
        self.columns.append(column)
        self.ppc.add_edge(self.root_unit, true_column_root, log_weight=0.0)
        self.root_unit.normalize()
        if len(self.columns) > 1:
            columns_iter = self.iter_aligned_nodes(self.columns[-2], self.columns[-1])
            for left_node, right_node in columns_iter:
                if isinstance(left_node, SumUnit) and isinstance(right_node, SumUnit):
                    self.ppc.add_edge(left_node, right_node, log_weight=0.0)
                    left_node.normalize()
        # self.ppc.is_decomposable()
        return column

    #

    # %% column-node-extraction
    def get_column_nodes(self, column: PCColumn) -> set[Unit]:
        """
        Resolve all internal units of the given column (excluding lateral subgraphs).

        :param column: The column whose internal units to collect.
        :return: A set of internal :class:`Unit` instances belonging to the column.
        """
        return {self.ppc.graph[node_id] for node_id in column.node_ids}

    # %% leaf-parameter-updates
    def _update_leaf_distribution(
        self,
        leaf: LeafUnit,
        data: npt.NDArray,
        weights: jax.Array,
        variable_to_index_map: dict[Variable, int],
    ) -> None:
        """
        Update the distribution parameters of a single leaf unit from weighted data.

        :param leaf: The leaf unit to update.
        :param data: The complete training dataset.
        :param weights: Non-negative posterior responsibilities per data row.
        :param variable_to_index_map: Map from Variable to column index in data.
        """
        total_weight = float(jnp.sum(weights))
        if total_weight <= 0.0:
            return

        distribution = leaf.distribution
        if distribution is None:
            return

        if isinstance(distribution, GaussianDistribution):
            var_idx = variable_to_index_map[distribution.variable]
            values = jnp.asarray(data[:, var_idx])
            mean, scale = _weighted_gaussian_parameters(values, weights)
            distribution.location = float(mean)
            distribution.scale = float(scale)

        elif isinstance(distribution, (DiscreteDistribution, SymbolicDistribution)):
            var_idx = variable_to_index_map[distribution.variable]
            values = data[:, var_idx]
            # Category keys are arbitrary Python hashes rather than fixed-width
            # numbers, so this membership mask is built on the numpy/Python side.
            weights_by_sample = np.asarray(weights)
            probabilities = distribution.probabilities
            for key in list(probabilities.keys()):
                mask = np.array([hash(v) == key for v in values])
                w_cat = float(np.sum(weights_by_sample[mask]))
                probabilities[key] = w_cat / total_weight
            distribution.normalize()

        elif hasattr(distribution, "fit"):
            var_indices = [variable_to_index_map[v] for v in distribution.variables]
            subset_data = data[:, var_indices]
            weights_by_sample = np.asarray(weights)
            try:
                distribution.fit(subset_data, weights=weights_by_sample)
            except TypeError:
                distribution.fit(subset_data)

    # %% column-mask
    def _column_mask(self, column: PCColumn) -> ColumnMask:
        """
        Build the mask of units that learning ``column`` is allowed to update.

        The mask always includes :attr:`root_unit`, since its mixture weights
        reflect every mounted column and adapt on every learning call, together
        with the column's own sum and leaf units. Every other column's units, and
        the lateral edges other columns own into this one, are excluded and stay
        frozen.

        :param column: The task column being learned.
        :return: The mask of updatable units.
        """
        column_nodes = self.get_column_nodes(column)
        sum_nodes = {self.root_unit} | {
            node for node in column_nodes if isinstance(node, SumUnit)
        }
        leaf_nodes = {node for node in column_nodes if isinstance(node, LeafUnit)}
        return ColumnMask(
            sum_nodes=frozenset(sum_nodes), leaf_nodes=frozenset(leaf_nodes)
        )

    # %% learning
    def learn(
        self,
        data: npt.NDArray,
        column: PCColumn,
        epochs: int = 1,
        smooth: float = 1e-8,
    ) -> list[float]:
        """
        Train parameters of the given column within the progressive circuit using EM.

        The E-step evaluates the complete circuit, so responsibilities flow through
        every mounted column, including frozen ones, and its numerics run on
        :mod:`jax`. The M-step masks the result down to :meth:`_column_mask` before
        writing updated weights and distribution parameters back onto the mounted
        units, so the target column is the only one, besides the shared root
        mixture, whose parameters change.

        :param data: Input data array with shape (N, num_variables).
        :param column: The specific task column to update.
        :param epochs: Number of learning iterations.
        :param smooth: Additive smoothing term for expected counts.
        :return: List of average log-likelihoods per epoch.
        """
        if column not in self.columns:
            raise ValueError(
                f"Column {column.task_id} is not registered in this ProgressivePC."
            )

        data = np.asarray(data)
        if len(data) == 0:
            return []

        mask = self._column_mask(column)
        variable_to_index_map = self.ppc.variable_to_index_map

        history_log_likelihoods: list[float] = []
        for _ in range(epochs):
            avg_ll, log_responsibilities, edge_expected_counts = self._expectation_step(
                data
            )
            history_log_likelihoods.append(avg_ll)
            self._maximization_step(
                mask,
                log_responsibilities,
                edge_expected_counts,
                data,
                variable_to_index_map,
                smooth,
            )

        return history_log_likelihoods

    def _expectation_step(
        self, data: npt.NDArray
    ) -> tuple[float, dict[int, jax.Array], dict[tuple[int, int], float]]:
        """
        Evaluate the complete circuit and propagate posterior responsibilities
        top-down through every mounted column.

        :param data: Input data array with shape (N, num_variables).
        :return: The average log-likelihood, each unit's log-responsibility per
            data row, and each sum edge's expected count.
        """
        num_samples = len(data)
        avg_log_likelihood = float(np.mean(self.ppc.log_likelihood(data)))

        log_responsibilities: dict[int, jax.Array] = {
            self.root_unit.index: jnp.zeros(num_samples)
        }
        edge_expected_counts: dict[tuple[int, int], float] = {}

        for layer in self.ppc.layers:
            for unit in layer:
                if unit.index not in log_responsibilities:
                    continue
                log_beta_u = log_responsibilities[unit.index]

                if isinstance(unit, SumUnit):
                    sum_val = jnp.asarray(unit.result_of_current_query)
                    for weight, child in unit.log_weighted_subcircuits:
                        child_val = jnp.asarray(child.result_of_current_query)
                        log_gamma, count = _edge_responsibility(
                            log_beta_u, weight, child_val, sum_val
                        )
                        edge_expected_counts[(unit.index, child.index)] = float(count)

                        if child.index not in log_responsibilities:
                            log_responsibilities[child.index] = log_gamma
                        else:
                            log_responsibilities[child.index] = jnp.logaddexp(
                                log_responsibilities[child.index], log_gamma
                            )

                elif isinstance(unit, InnerUnit):
                    for child in unit.subcircuits:
                        if child.index not in log_responsibilities:
                            log_responsibilities[child.index] = log_beta_u
                        else:
                            log_responsibilities[child.index] = jnp.logaddexp(
                                log_responsibilities[child.index], log_beta_u
                            )

        return avg_log_likelihood, log_responsibilities, edge_expected_counts

    def _maximization_step(
        self,
        mask: ColumnMask,
        log_responsibilities: dict[int, jax.Array],
        edge_expected_counts: dict[tuple[int, int], float],
        data: npt.NDArray,
        variable_to_index_map: dict[Variable, int],
        smooth: float,
    ) -> None:
        """
        Update the weights and leaf distributions of the masked units from the
        responsibilities computed by :meth:`_expectation_step`.

        Units outside ``mask`` are left untouched, which is what keeps every other
        column frozen; updated values are written back onto the same mounted
        :class:`Unit` instances so the circuit's node identity is preserved.

        :param mask: The units that may be updated.
        :param log_responsibilities: Each unit's log-responsibility per data row.
        :param edge_expected_counts: Each sum edge's expected count.
        :param data: The complete training dataset.
        :param variable_to_index_map: Map from Variable to column index in data.
        :param smooth: Additive smoothing term for expected counts.
        """
        for sum_node in mask.sum_nodes:
            subcircuits = sum_node.subcircuits
            if not subcircuits:
                continue
            counts = jnp.array(
                [
                    edge_expected_counts.get((sum_node.index, child.index), 0.0)
                    + smooth
                    for child in subcircuits
                ]
            )
            if float(jnp.sum(counts)) <= 0.0:
                continue
            new_log_weights = _normalized_log_weights(counts)
            for child, log_weight in zip(subcircuits, new_log_weights):
                self.ppc.add_edge(sum_node, child, log_weight=float(log_weight))
            sum_node.normalize()

        for leaf in mask.leaf_nodes:
            if leaf.index in log_responsibilities:
                weights = jnp.exp(log_responsibilities[leaf.index])
                self._update_leaf_distribution(
                    leaf, data, weights, variable_to_index_map
                )

from __future__ import annotations

# %% imports
from collections import deque
import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Optional

import numpy as np
import numpy.typing as npt

from random_events.variable import Variable

from probabilistic_model.distributions.distributions import (
    DiscreteDistribution,
    SymbolicDistribution,
)
from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit as RxPc, SumUnit, Unit, LeafUnit
)


# %% pc-column
@dataclass
class PCColumn:
    """A single task-specific copy of a probabilistic circuit template."""
    #TODO: List ids alle Nodes in Column bzw. min. LEAFS
    task_id: str
    column_root: Unit
    variable_domain: Optional[tuple[Variable, ...]] = None
    task_context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.variable_domain is None:
            self.variable_domain = tuple(self.column_root.variables)
        elif not set(self.variable_domain).issubset(set(self.column_root.variables)):
            raise ValueError(
                "The variable domain must be compatible with the column's root circuit."
            )


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

    def iter_aligned_nodes(self, left: PCColumn, right: PCColumn) -> Iterator[tuple[Unit, Unit]]:
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
            left_children = [ child for child in left_node.subcircuits
                if not (isinstance(left_node, SumUnit) and isinstance(child, SumUnit))
            ]
            right_children = [ child for child in right_node.subcircuits
                if not (isinstance(right_node, SumUnit) and isinstance(child, SumUnit))
            ]
            if len(left_children) != len(right_children):
                raise ValueError(
                    f"Columns diverged: different number of children ({len(left_children)} vs {len(right_children)})."
                )
            for left_child, right_child in zip(left_children, right_children):
                queue.append((left_child, right_child))

    def add_column(self, task_id: str, task_context: Optional[dict[str, Any]] = None) -> PCColumn:
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
                    self.ppc.add_edge(right_node, left_node, log_weight=0.0)
                    right_node.normalize()
        return column

    # %% column-node-extraction
    def get_column_nodes(self, column: PCColumn) -> set[Unit]:
        """
        Collect all internal units of the given column (excluding lateral subgraphs).

        :param column: The column whose internal units to collect.
        :return: A set of internal :class:`Unit` instances belonging to the column.
        """
        internal_nodes: set[Unit] = set()
        queue: deque[Unit] = deque([column.column_root])
        while queue:
            node = queue.popleft()
            if node in internal_nodes:
                continue
            internal_nodes.add(node)
            for child in node.subcircuits:
                if not (isinstance(node, SumUnit) and isinstance(child, SumUnit)):
                    queue.append(child)
        return internal_nodes

    # %% leaf-parameter-updates
    def _update_leaf_distribution(
        self,
        leaf: LeafUnit,
        data: npt.NDArray,
        weights: npt.NDArray,
        variable_to_index_map: dict[Variable, int],
    ) -> None:
        """
        Update the distribution parameters of a single leaf unit from weighted data.

        :param leaf: The leaf unit to update.
        :param data: The complete training dataset.
        :param weights: Non-negative posterior responsibilities per data row.
        :param variable_to_index_map: Map from Variable to column index in data.
        """
        total_weight = float(np.sum(weights))
        if total_weight <= 0.0:
            return

        distribution = leaf.distribution
        if distribution is None:
            return

        if isinstance(distribution, GaussianDistribution):
            var_idx = variable_to_index_map[distribution.variable]
            values = data[:, var_idx]
            mean = float(np.sum(weights * values) / total_weight)
            variance = float(np.sum(weights * ((values - mean) ** 2)) / total_weight)
            distribution.location = mean
            distribution.scale = float(np.sqrt(max(variance, 1e-6)))

        elif isinstance(distribution, (DiscreteDistribution, SymbolicDistribution)):
            var_idx = variable_to_index_map[distribution.variable]
            values = data[:, var_idx]
            probabilities = distribution.probabilities
            for key in list(probabilities.keys()):
                mask = np.array([hash(v) == key for v in values])
                w_cat = float(np.sum(weights[mask]))
                probabilities[key] = w_cat / total_weight
            distribution.normalize()

        elif hasattr(distribution, "fit"):
            var_indices = [variable_to_index_map[v] for v in distribution.variables]
            subset_data = data[:, var_indices]
            try:
                distribution.fit(subset_data, weights=weights)
            except TypeError:
                distribution.fit(subset_data)

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

        Forward pass evaluates the entire circuit across all mounted columns.
        Backward pass propagates responsibilities top-down, updating only the
        target column's internal nodes, its lateral cross-column connections,
        and the main root edges connecting columns.

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
        num_samples = len(data)
        if num_samples == 0:
            return []

        column_nodes = self.get_column_nodes(column)
        updatable_sum_nodes = {self.root_unit} | {
            node for node in column_nodes if isinstance(node, SumUnit)
        }
        updatable_leaf_nodes = {
            node for node in column_nodes if isinstance(node, LeafUnit)
        }
        variable_to_index_map = self.ppc.variable_to_index_map

        history_log_likelihoods: list[float] = []

        for _ in range(epochs):
            # 1. Forward pass: evaluate complete circuit on data
            ll = self.ppc.log_likelihood(data)
            avg_ll = float(np.mean(ll))
            history_log_likelihoods.append(avg_ll)

            # 2. Backward pass: compute posterior responsibilities top-down
            log_responsibilities: dict[int, np.ndarray] = {
                self.root_unit.index: np.zeros(num_samples)
            }
            edge_expected_counts: dict[tuple[int, int], float] = {}

            for layer in self.ppc.layers:
                for unit in layer:
                    if unit.index not in log_responsibilities:
                        continue
                    log_beta_u = log_responsibilities[unit.index]

                    if isinstance(unit, SumUnit):
                        sum_val = unit.result_of_current_query
                        for weight, child in unit.log_weighted_subcircuits:
                            child_val = child.result_of_current_query
                            log_gamma = log_beta_u + weight + child_val - sum_val
                            log_gamma = np.where(
                                np.isnan(log_gamma), -np.inf, log_gamma
                            )

                            max_log = np.max(log_gamma)
                            if max_log > -np.inf:
                                count = float(
                                    np.sum(np.exp(log_gamma - max_log))
                                ) * float(np.exp(max_log))
                            else:
                                count = 0.0
                            edge_expected_counts[(unit.index, child.index)] = count

                            if child.index not in log_responsibilities:
                                log_responsibilities[child.index] = log_gamma
                            else:
                                log_responsibilities[child.index] = np.logaddexp(
                                    log_responsibilities[child.index], log_gamma
                                )

                    elif isinstance(unit, InnerUnit):
                        for child in unit.subcircuits:
                            if child.index not in log_responsibilities:
                                log_responsibilities[child.index] = log_beta_u.copy()
                            else:
                                log_responsibilities[child.index] = np.logaddexp(
                                    log_responsibilities[child.index], log_beta_u
                                )

            # 3. Parameter Update: Update updatable sum units (root & target column)
            for sum_node in updatable_sum_nodes:
                subcircuits = sum_node.subcircuits
                if not subcircuits:
                    continue
                counts = np.array(
                    [
                        edge_expected_counts.get((sum_node.index, child.index), 0.0)
                        + smooth
                        for child in subcircuits
                    ]
                )
                total_count = np.sum(counts)
                if total_count > 0:
                    new_weights = counts / total_count
                    for child, weight in zip(subcircuits, new_weights):
                        self.ppc.add_edge(
                            sum_node,
                            child,
                            log_weight=float(np.log(max(weight, 1e-12))),
                        )
                    sum_node.normalize()

            # Update updatable leaf units in target column
            for leaf in updatable_leaf_nodes:
                if leaf.index in log_responsibilities:
                    weights = np.exp(log_responsibilities[leaf.index])
                    self._update_leaf_distribution(
                        leaf, data, weights, variable_to_index_map
                    )

        return history_log_likelihoods

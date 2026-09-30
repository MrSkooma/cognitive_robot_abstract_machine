from __future__ import annotations

# %% imports
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from jax.scipy.special import logsumexp as jax_logsumexp
from random_events.variable import Variable

from probabilistic_model.distributions.distributions import DiscreteDistribution
from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.exceptions import UnsupportedLeafDistributionError
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    InnerUnit,
    LeafUnit,
    SumUnit,
)
from probabilistic_model.probabilistic_circuit.rx.progressive import (
    CircuitColumn,
    ProgressiveProbabilisticCircuit,
)

# %% constants
MINIMUM_MIXTURE_PROPORTION = 1e-12
"""
Lower bound for a learned mixture weight before taking its logarithm.
"""

MINIMUM_GAUSSIAN_VARIANCE = 1e-6
"""
Lower bound for the variance of a learned Gaussian leaf.
"""

DEFAULT_SMOOTHING = 1e-8
"""
Default additive smoothing term for the expected counts of sum unit edges.
"""


# %% jax numerics
@jax.jit
def _edge_responsibility(
    log_parent_responsibility: jax.Array,
    log_edge_weight: jax.Array,
    log_child_likelihood: jax.Array,
    log_parent_likelihood: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """
    :return: The per-row log-responsibility of a sum unit edge and its expected count,
        summed over rows.
    """
    log_responsibility = (
        log_parent_responsibility
        + log_edge_weight
        + log_child_likelihood
        - log_parent_likelihood
    )
    log_responsibility = jnp.where(
        jnp.isnan(log_responsibility), -jnp.inf, log_responsibility
    )
    return log_responsibility, jnp.exp(jax_logsumexp(log_responsibility))


@jax.jit
def _normalized_log_weights(counts: jax.Array) -> jax.Array:
    """
    :param counts: Smoothed expected count of every child.
    :return: The log of every count's share, floored at
        :data:`MINIMUM_MIXTURE_PROPORTION`.
    """
    proportions = counts / jnp.sum(counts)
    return jnp.log(jnp.maximum(proportions, MINIMUM_MIXTURE_PROPORTION))


@jax.jit
def _weighted_gaussian_parameters(
    values: jax.Array, weights: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """
    :return: The weighted mean and standard deviation of ``values``, the variance
        floored at :data:`MINIMUM_GAUSSIAN_VARIANCE`.
    """
    total_weight = jnp.sum(weights)
    mean = jnp.sum(weights * values) / total_weight
    variance = jnp.sum(weights * (values - mean) ** 2) / total_weight
    return mean, jnp.sqrt(jnp.maximum(variance, MINIMUM_GAUSSIAN_VARIANCE))


# %% learning data structures
@dataclass(frozen=True)
class LearnableUnits:
    """
    The units whose parameters learning a column updates.
    """

    sum_units: frozenset[SumUnit]
    """
    Sum units whose weights are updated.
    """

    leaf_units: frozenset[LeafUnit]
    """
    Leaf units whose distributions are updated.
    """


@dataclass(frozen=True)
class Edge:
    """
    An edge, identified by the indices of its units.
    """

    parent_index: int
    """
    Index of the parent unit.
    """

    child_index: int
    """
    Index of the child unit.
    """


@dataclass
class ExpectationStepResult:
    """
    The posterior statistics of one expectation step.
    """

    average_log_likelihood: float
    """
    Average log-likelihood of the rows under the whole circuit.
    """

    log_responsibilities: dict[int, jax.Array]
    """
    Per-row log-responsibility of every reached unit, keyed by unit index.
    """

    edge_expected_counts: dict[Edge, float]
    """
    Expected count of every edge below a sum unit.
    """


# %% expectation maximization
@dataclass
class ProgressiveExpectationMaximization:
    """
    Expectation maximization for one column of a progressive probabilistic circuit.

    While the column is learned, the root gives it the whole weight, so it is trained as
    the only model of its task; it still reaches earlier columns through its edges to
    them. Only the units of the column are updated. Afterwards the root weights every
    column by its share of all rows the columns were learned from.

    .. warning::

        Later columns read from earlier ones, so learning an earlier column again also
        changes the later columns.
    """

    progressive_circuit: ProgressiveProbabilisticCircuit
    """
    The circuit whose columns are learned.
    """

    smoothing: float = DEFAULT_SMOOTHING
    """
    Added to the expected count of every edge below a sum unit.
    """

    def learn(
        self, data: npt.NDArray, column: CircuitColumn, epochs: int = 1
    ) -> list[float]:
        """
        Learn the parameters of a column.

        :param data: One row per sample, columns ordered like the circuit's variables.
        :param epochs: Number of expectation maximization iterations.
        :return: The average log-likelihood of the rows under the column before every
            iteration.
        :raises UnregisteredColumnError: If the column belongs to another progressive
            circuit.
        :raises IncompatibleVariableDomainError: If the variable domain of the column
            contains variables its root does not model.
        :raises UnsupportedVariableDomainChangeError: If a variable domain changed after
            the column was created.
        :raises UnsupportedLeafDistributionError: If the column has a leaf that cannot
            be learned.
        """
        self.progressive_circuit.validate_column(column)
        data = np.asarray(data)
        if len(data) == 0:
            return []

        learnable_units = self.learnable_units(column)
        self.progressive_circuit.restrict_root_to(column)
        history = []
        for _ in range(epochs):
            expectation = self._expectation_step(data)
            history.append(expectation.average_log_likelihood)
            self._maximization_step(learnable_units, expectation, data)
        column.sample_count += len(data)
        self.progressive_circuit.weight_root_by_sample_count()
        return history

    def learnable_units(self, column: CircuitColumn) -> LearnableUnits:
        """
        :return: The sum and leaf units of the column.
        :raises UnsupportedLeafDistributionError: If the column has a leaf that cannot
            be learned.
        """
        units = self.progressive_circuit.units_of(column)
        leaf_units = frozenset(unit for unit in units if isinstance(unit, LeafUnit))
        for leaf_unit in leaf_units:
            if not isinstance(
                leaf_unit.distribution, (GaussianDistribution, DiscreteDistribution)
            ):
                raise UnsupportedLeafDistributionError(leaf_unit.distribution)
        sum_units = frozenset(unit for unit in units if isinstance(unit, SumUnit))
        return LearnableUnits(sum_units=sum_units, leaf_units=leaf_units)

    def _expectation_step(self, data: npt.NDArray) -> ExpectationStepResult:
        """
        Evaluate the whole circuit and pass the responsibilities from the root down to
        every reached unit.
        """
        circuit = self.progressive_circuit.circuit
        average_log_likelihood = float(np.mean(circuit.log_likelihood(data)))
        log_responsibilities: dict[int, jax.Array] = {
            self.progressive_circuit.root.index: jnp.zeros(len(data))
        }
        edge_expected_counts: dict[Edge, float] = {}

        for layer in circuit.layers:
            for unit in layer:
                if unit.index not in log_responsibilities or not isinstance(
                    unit, InnerUnit
                ):
                    continue
                parent_responsibility = log_responsibilities[unit.index]
                if isinstance(unit, SumUnit):
                    parent_likelihood = jnp.asarray(unit.result_of_current_query)
                    for log_weight, child in unit.log_weighted_subcircuits:
                        child_responsibility, expected_count = _edge_responsibility(
                            parent_responsibility,
                            log_weight,
                            jnp.asarray(child.result_of_current_query),
                            parent_likelihood,
                        )
                        edge_expected_counts[Edge(unit.index, child.index)] = float(
                            expected_count
                        )
                        self._accumulate(
                            log_responsibilities, child.index, child_responsibility
                        )
                else:
                    for child in unit.subcircuits:
                        self._accumulate(
                            log_responsibilities, child.index, parent_responsibility
                        )

        return ExpectationStepResult(
            average_log_likelihood, log_responsibilities, edge_expected_counts
        )

    @staticmethod
    def _accumulate(
        log_responsibilities: dict[int, jax.Array],
        unit_index: int,
        log_responsibility: jax.Array,
    ) -> None:
        """
        Add a per-row log-responsibility to those already collected for a unit.
        """
        if unit_index in log_responsibilities:
            log_responsibility = jnp.logaddexp(
                log_responsibilities[unit_index], log_responsibility
            )
        log_responsibilities[unit_index] = log_responsibility

    def _maximization_step(
        self,
        learnable_units: LearnableUnits,
        expectation: ExpectationStepResult,
        data: npt.NDArray,
    ) -> None:
        """
        Update the weights and leaf distributions of the learnable units.
        """
        circuit = self.progressive_circuit.circuit
        for sum_unit in learnable_units.sum_units:
            subcircuits = sum_unit.subcircuits
            counts = jnp.array(
                [
                    expectation.edge_expected_counts.get(
                        Edge(sum_unit.index, child.index), 0.0
                    )
                    + self.smoothing
                    for child in subcircuits
                ]
            )
            if float(jnp.sum(counts)) <= 0.0:
                continue
            for child, log_weight in zip(subcircuits, _normalized_log_weights(counts)):
                circuit.add_edge(sum_unit, child, log_weight=float(log_weight))
            sum_unit.normalize()

        variable_to_index_map = circuit.variable_to_index_map
        for leaf_unit in learnable_units.leaf_units:
            weights = jnp.exp(expectation.log_responsibilities[leaf_unit.index])
            if float(jnp.sum(weights)) <= 0.0:
                continue
            self._update_leaf_distribution(
                leaf_unit, data, weights, variable_to_index_map
            )

    @staticmethod
    def _update_leaf_distribution(
        leaf_unit: LeafUnit,
        data: npt.NDArray,
        weights: jax.Array,
        variable_to_index_map: dict[Variable, int],
    ) -> None:
        """
        Fit the distribution of a leaf to the weighted rows, in place.

        :param weights: Non-negative responsibility of every row, with a positive total.
        :param variable_to_index_map: Column of every variable in ``data``.
        """
        distribution = leaf_unit.distribution
        values = data[:, variable_to_index_map[distribution.variable]]

        if isinstance(distribution, GaussianDistribution):
            mean, scale = _weighted_gaussian_parameters(jnp.asarray(values), weights)
            distribution.location = float(mean)
            distribution.scale = float(scale)
            return

        value_hashes = np.array([hash(value) for value in values])
        sample_weights = np.asarray(weights)
        probabilities = distribution.probabilities
        for key in list(probabilities.keys()):
            probabilities[key] = float(np.sum(sample_weights[value_hashes == key]))
        distribution.normalize()

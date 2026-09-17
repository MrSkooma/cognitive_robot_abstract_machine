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
    Compute the per-sample log-responsibility of a sum unit edge and its expected count.

    :param log_parent_responsibility: The per-sample log-responsibility of the parent.
    :param log_edge_weight: The log mixture weight of the edge.
    :param log_child_likelihood: The per-sample log-likelihood of the child.
    :param log_parent_likelihood: The per-sample log-likelihood of the parent.
    :return: The per-sample log-responsibility of the edge and its expected count summed
        over samples.
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
    Normalize positive expected counts into log mixture weights.

    :param counts: Smoothed expected counts per child.
    :return: The log of the share of each count, floored at
        :data:`MINIMUM_MIXTURE_PROPORTION`.
    """
    proportions = counts / jnp.sum(counts)
    return jnp.log(jnp.maximum(proportions, MINIMUM_MIXTURE_PROPORTION))


@jax.jit
def _weighted_gaussian_parameters(
    values: jax.Array, weights: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """
    Compute the weighted mean and standard deviation of ``values``.

    :param values: Per-sample observations of the variable.
    :param weights: Non-negative responsibilities per sample.
    :return: The weighted mean and standard deviation, with the variance floored at
        :data:`MINIMUM_GAUSSIAN_VARIANCE`.
    """
    total_weight = jnp.sum(weights)
    mean = jnp.sum(weights * values) / total_weight
    variance = jnp.sum(weights * (values - mean) ** 2) / total_weight
    return mean, jnp.sqrt(jnp.maximum(variance, MINIMUM_GAUSSIAN_VARIANCE))


# %% learning data structures
@dataclass(frozen=True)
class LearnableUnits:
    """
    The units of a progressive circuit whose parameters learning a column may update.
    """

    sum_units: frozenset[SumUnit]
    """
    Sum units whose mixture weights may be updated.
    """

    leaf_units: frozenset[LeafUnit]
    """
    Leaf units whose distribution parameters may be updated.
    """


@dataclass(frozen=True)
class Edge:
    """
    An edge of a circuit, identified by the indices of its units.
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
    Average log-likelihood of the data under the complete circuit.
    """

    log_responsibilities: dict[int, jax.Array]
    """
    Per-sample log-responsibility of every reached unit, keyed by unit index.
    """

    edge_expected_counts: dict[Edge, float]
    """
    Expected count of every sum unit edge.
    """


# %% expectation maximization
@dataclass
class ProgressiveExpectationMaximization:
    """
    Expectation maximization for a single column of a progressive probabilistic circuit.

    The expectation step evaluates the complete circuit, so responsibilities also flow
    through earlier columns. The maximization step only updates the units of the learned
    column; units of every other column stay unchanged. After learning, the root mixture
    weights every column by its share of all samples the columns were learned from.

    .. warning::

        Later columns read from earlier ones, so learning an earlier column again also
        changes the output of later columns.
    """

    progressive_circuit: ProgressiveProbabilisticCircuit
    """
    The progressive circuit whose columns are learned.
    """

    smoothing: float = DEFAULT_SMOOTHING
    """
    Additive smoothing term for the expected counts of sum unit edges.
    """

    def learn(
        self, data: npt.NDArray, column: CircuitColumn, epochs: int = 1
    ) -> list[float]:
        """
        Learn the parameters of a column from data.

        :param data: Data with one row per sample, ordered like the variables of the
            progressive circuit.
        :param column: The column to learn.
        :param epochs: Number of expectation maximization iterations.
        :return: The average log-likelihood of the data before each iteration.
        :raises UnregisteredColumnError: If the column does not belong to the
            progressive circuit.
        :raises IncompatibleVariableDomainError: If the domain of the column contains
            variables its root does not model.
        :raises UnsupportedVariableDomainChangeError: If the domain of the column or of
            the progressive circuit changed after the column was created.
        :raises UnsupportedLeafDistributionError: If the column contains a leaf that
            cannot be learned from weighted data.
        """
        self.progressive_circuit.validate_column(column)
        data = np.asarray(data)
        if len(data) == 0:
            return []

        learnable_units = self.learnable_units(column)
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
        :param column: The column to learn.
        :return: The sum and leaf units of the column.
        :raises UnsupportedLeafDistributionError: If the column contains a leaf that
            cannot be learned from weighted data.
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
        Evaluate the complete circuit and propagate the posterior responsibilities from
        the root to every reached unit.

        :param data: Data with one row per sample.
        :return: The posterior statistics.
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
        Add a log-responsibility to the responsibilities already collected for a unit.

        :param log_responsibilities: Collected log-responsibilities, keyed by unit index.
        :param unit_index: Index of the receiving unit.
        :param log_responsibility: The per-sample log-responsibility to add.
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
        Update the mixture weights and leaf distributions of the learnable units.

        :param learnable_units: The units that may be updated.
        :param expectation: The posterior statistics of the expectation step.
        :param data: Data with one row per sample.
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
        Fit the distribution of a leaf unit to weighted data in place.

        :param leaf_unit: The leaf unit to update.
        :param data: Data with one row per sample.
        :param weights: Positive total, non-negative responsibilities per sample.
        :param variable_to_index_map: Map from each variable to its column in the data.
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

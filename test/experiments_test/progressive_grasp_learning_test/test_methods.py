"""
The template circuit and the methods that learn a new grasp task.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from experiments.grasp_learning.exported_attempt import ValueRange
from experiments.progressive_grasp_learning.circuit_variables import (
    GraspOutcomeVariables,
    LiftOutcome,
)
from experiments.progressive_grasp_learning.exceptions import MissingEarlierTaskError
from experiments.progressive_grasp_learning.methods import (
    EarlierColumnUnchanged,
    EarlierTask,
    ExpectationMaximizationSettings,
    FixedScratch,
    LearningSettings,
    LiftRateOfTrainingAttempts,
    ProgressiveColumn,
    RatSpnTemplate,
    TunedScratch,
)
from experiments.progressive_grasp_learning.synthetic_tasks import BoxTask
from experiments.progressive_grasp_learning.task_attempts import (
    GraspNumber,
    ScaledAttempts,
)
from probabilistic_model.distributions.distributions import SymbolicDistribution
from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import LeafUnit

# %% fixtures

SEED = 7
"""
The seed of every template in these tests.
"""


@pytest.fixture
def variables() -> GraspOutcomeVariables:
    return GraspOutcomeVariables()


@pytest.fixture
def settings() -> LearningSettings:
    """
    Settings of a small template, learned briefly.
    """
    return LearningSettings(
        template=RatSpnTemplate(depth=1, units_per_region=2),
        expectation_maximization=ExpectationMaximizationSettings(iterations=5),
    )


@pytest.fixture
def earlier_attempts() -> ScaledAttempts:
    return BoxTask(
        box={GraspNumber.AZIMUTH: ValueRange(lowest=0.25, highest=0.75)}
    ).attempts(number_of_attempts=200, generator=np.random.default_rng(1))


@pytest.fixture
def new_attempts() -> ScaledAttempts:
    return BoxTask(
        box={GraspNumber.AZIMUTH: ValueRange(lowest=0.35, highest=0.85)}
    ).attempts(number_of_attempts=12, generator=np.random.default_rng(2))


@pytest.fixture
def probe_grasps() -> np.ndarray:
    return np.random.default_rng(4).uniform(size=(50, len(GraspNumber)))


# %% variables


def test_the_outcome_lands_in_the_outcome_variables_column(variables):
    grasps = np.full((1, len(GraspNumber)), 0.5)

    rows = variables.rows(grasps=grasps, outcomes=np.array([LiftOutcome.LIFTED]))

    outcome_column = [variable.name for variable in variables.variables].index(
        GraspOutcomeVariables.outcome_name
    )
    assert rows[0, outcome_column] == LiftOutcome.LIFTED


# %% the template


def test_the_same_seed_builds_the_same_template(settings, variables, probe_grasps):
    rows = variables.rows(
        grasps=probe_grasps, outcomes=np.zeros(len(probe_grasps), dtype=np.int64)
    )

    first, second = (
        settings.template.circuit(variables=variables, seed=SEED) for _ in range(2)
    )

    np.testing.assert_allclose(first.log_likelihood(rows), second.log_likelihood(rows))


def test_every_gaussian_leaf_starts_in_the_unit_interval_with_the_start_scale(
    settings, variables
):
    circuit = settings.template.circuit(variables=variables, seed=SEED)

    distributions = [
        unit.distribution
        for unit in circuit.nodes()
        if isinstance(unit, LeafUnit)
        and isinstance(unit.distribution, GaussianDistribution)
    ]

    assert all(0 <= distribution.location <= 1 for distribution in distributions)
    assert {distribution.scale for distribution in distributions} == {
        settings.template.start_scale
    }


# %% methods


def test_a_column_without_weight_on_the_earlier_task_learns_like_fixed_scratch(
    settings, variables, earlier_attempts, new_attempts, probe_grasps
):
    settings.earlier_column_share = 0.0
    earlier = EarlierTask.learned(
        attempts=earlier_attempts,
        settings=settings,
        expectation_maximization=settings.expectation_maximization,
        variables=variables,
        seed=SEED,
    )

    progressive, scratch = (
        method.model(training=new_attempts, earlier=earlier, seed=SEED)
        for method in [
            ProgressiveColumn(settings=settings, variables=variables),
            FixedScratch(settings=settings, variables=variables),
        ]
    )

    np.testing.assert_allclose(
        progressive.outcome_likelihoods(probe_grasps).log_joint,
        scratch.outcome_likelihoods(probe_grasps).log_joint,
    )


def test_learning_a_new_column_leaves_the_earlier_column_unchanged(
    settings, variables, earlier_attempts, new_attempts, probe_grasps
):
    earlier = EarlierTask.learned(
        attempts=earlier_attempts,
        settings=settings,
        expectation_maximization=settings.expectation_maximization,
        variables=variables,
        seed=SEED,
    )
    unchanged = EarlierColumnUnchanged(variables=variables)
    before = (
        unchanged.model(training=new_attempts, earlier=earlier, seed=SEED)
        .outcome_likelihoods(probe_grasps)
        .log_joint
    )

    ProgressiveColumn(settings=settings, variables=variables).model(
        training=new_attempts, earlier=earlier, seed=SEED
    )

    after = (
        unchanged.model(training=new_attempts, earlier=earlier, seed=SEED)
        .outcome_likelihoods(probe_grasps)
        .log_joint
    )
    np.testing.assert_allclose(after, before)


def test_tuned_scratch_learns_the_candidate_with_the_lowest_cross_validated_loss(
    settings, variables, earlier_attempts, new_attempts, probe_grasps
):
    candidates = [
        settings,
        LearningSettings(
            template=RatSpnTemplate(depth=1, units_per_region=5),
            expectation_maximization=ExpectationMaximizationSettings(iterations=5),
        ),
    ]
    tuned = TunedScratch(candidates=candidates, variables=variables)
    earlier = EarlierTask.learned(
        attempts=earlier_attempts,
        settings=settings,
        expectation_maximization=settings.expectation_maximization,
        variables=variables,
        seed=SEED,
    )
    losses = [
        tuned.cross_validated_loss(settings=candidate, training=new_attempts, seed=SEED)
        for candidate in candidates
    ]

    model = tuned.model(training=new_attempts, earlier=earlier, seed=SEED)

    best = FixedScratch(
        settings=candidates[int(np.argmin(losses))], variables=variables
    ).learned_column(training=new_attempts, seed=SEED)
    np.testing.assert_allclose(
        model.outcome_likelihoods(probe_grasps).log_joint,
        best.outcome_likelihoods(probe_grasps).log_joint,
    )


def test_the_outcome_pseudo_count_keeps_failure_possible_after_only_lifts(
    settings, variables, new_attempts
):
    settings.expectation_maximization.outcome_pseudo_count = 1.0
    only_lifts = new_attempts.rows(np.arange(len(new_attempts.lifted)))
    only_lifts.lifted = np.ones(len(only_lifts.lifted), dtype=bool)

    model = FixedScratch(settings=settings, variables=variables).learned_column(
        training=only_lifts, seed=SEED
    )

    failure_probabilities = [
        unit.distribution.probabilities[hash(LiftOutcome.NOT_LIFTED)]
        for unit in model.progressive_circuit.units_of(model.column)
        if isinstance(unit, LeafUnit)
        and isinstance(unit.distribution, SymbolicDistribution)
    ]
    assert min(failure_probabilities) > 0


def test_the_earlier_task_is_learned_with_its_own_expectation_maximization_settings(
    settings, variables, earlier_attempts, new_attempts, probe_grasps
):
    whole_pool = ExpectationMaximizationSettings(
        iterations=20, minimum_gaussian_variance=0.05**2
    )
    earlier = EarlierTask.learned(
        attempts=earlier_attempts,
        settings=settings,
        expectation_maximization=whole_pool,
        variables=variables,
        seed=SEED,
    )

    unchanged = EarlierColumnUnchanged(variables=variables).model(
        training=new_attempts, earlier=earlier, seed=SEED
    )

    learned_alone = FixedScratch(
        settings=dataclasses.replace(settings, expectation_maximization=whole_pool),
        variables=variables,
    ).learned_column(training=earlier_attempts, seed=SEED)
    np.testing.assert_allclose(
        unchanged.outcome_likelihoods(probe_grasps).log_joint,
        learned_alone.outcome_likelihoods(probe_grasps).log_joint,
    )


def test_the_lift_rate_of_the_training_attempts_is_predicted_for_every_grasp(
    new_attempts, probe_grasps
):
    outcome_pseudo_count = 1.0

    model = LiftRateOfTrainingAttempts(outcome_pseudo_count=outcome_pseudo_count).model(
        training=new_attempts, earlier=None, seed=SEED
    )

    lift_rate = (np.sum(new_attempts.lifted) + outcome_pseudo_count) / (
        len(new_attempts.lifted) + len(LiftOutcome) * outcome_pseudo_count
    )
    np.testing.assert_allclose(
        model.outcome_likelihoods(probe_grasps).lift_probabilities,
        np.full(len(probe_grasps), lift_rate),
    )


@pytest.mark.parametrize("method_type", [ProgressiveColumn, EarlierColumnUnchanged])
def test_a_method_reading_the_earlier_task_refuses_to_run_without_one(
    method_type, settings, variables, new_attempts
):
    method = (
        ProgressiveColumn(settings=settings, variables=variables)
        if method_type is ProgressiveColumn
        else EarlierColumnUnchanged(variables=variables)
    )

    with pytest.raises(MissingEarlierTaskError):
        method.model(training=new_attempts, earlier=None, seed=SEED)


def test_fixed_scratch_learns_without_an_earlier_task(
    settings, variables, new_attempts, probe_grasps
):
    model = FixedScratch(settings=settings, variables=variables).model(
        training=new_attempts, earlier=None, seed=SEED
    )

    assert model.outcome_likelihoods(probe_grasps).log_joint.shape == (
        len(probe_grasps),
        2,
    )

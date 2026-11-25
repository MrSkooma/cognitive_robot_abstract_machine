"""
The template circuit and the methods that learn a new grasp task.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.grasp_learning.exported_attempt import ValueRange
from experiments.progressive_grasp_learning.circuit_variables import (
    GraspOutcomeVariables,
    LiftOutcome,
)
from experiments.progressive_grasp_learning.methods import (
    EarlierColumnUnchanged,
    EarlierTask,
    FixedScratch,
    LearningSettings,
    ProgressiveColumn,
    RatSpnTemplate,
    TunedScratch,
)
from experiments.progressive_grasp_learning.synthetic_tasks import BoxTask
from experiments.progressive_grasp_learning.task_attempts import (
    GraspNumber,
    ScaledAttempts,
)
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
        template=RatSpnTemplate(depth=1, units_per_region=2), iterations=5
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
        attempts=earlier_attempts, settings=settings, variables=variables, seed=SEED
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
        attempts=earlier_attempts, settings=settings, variables=variables, seed=SEED
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
            template=RatSpnTemplate(depth=1, units_per_region=5), iterations=5
        ),
    ]
    tuned = TunedScratch(candidates=candidates, variables=variables)
    earlier = EarlierTask.learned(
        attempts=earlier_attempts, settings=settings, variables=variables, seed=SEED
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

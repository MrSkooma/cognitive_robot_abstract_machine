"""
Scoring predictions of lifting on a probe set.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.grasp_learning.exported_attempt import ValueRange
from experiments.progressive_grasp_learning.circuit_variables import LiftOutcome
from experiments.progressive_grasp_learning.exceptions import ZeroGraspDensityError
from experiments.progressive_grasp_learning.scoring import (
    OutcomeLikelihoods,
    ProbeScore,
)
from experiments.progressive_grasp_learning.synthetic_tasks import BoxTask
from experiments.progressive_grasp_learning.task_attempts import GraspNumber

# %% helpers


def likelihoods_of(lift_probabilities: np.ndarray) -> OutcomeLikelihoods:
    """
    :return: The likelihoods of a model that gives every grasp density 1 and the given
        probability of lifting.
    """
    with np.errstate(divide="ignore"):
        return OutcomeLikelihoods(
            log_joint=np.column_stack(
                [np.log(1 - lift_probabilities), np.log(lift_probabilities)]
            )
        )


def binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    """
    :return: The expected log loss of predicting each probability for an outcome drawn
        with it.
    """
    return -(
        probabilities * np.log(probabilities)
        + (1 - probabilities) * np.log(1 - probabilities)
    )


# %% skill


def test_the_true_model_reaches_the_skill_its_lift_probabilities_imply():
    task = BoxTask(box={GraspNumber.AZIMUTH: ValueRange(lowest=0.25, highest=0.75)})
    probe = task.attempts(number_of_attempts=40000, generator=np.random.default_rng(3))
    true_probabilities = task.lift_probabilities(probe.grasps)
    mean_lift_probability = np.mean(true_probabilities)

    score = ProbeScore(
        likelihoods=likelihoods_of(true_probabilities),
        lifted=probe.lifted,
        base_lift_rate=float(mean_lift_probability),
    )

    expected_skill = 1 - np.mean(binary_entropy(true_probabilities)) / binary_entropy(
        mean_lift_probability
    )
    assert score.skill == pytest.approx(expected_skill, abs=0.02)


def test_the_constant_prediction_of_the_base_rate_has_zero_skill():
    lifted = np.array([True, False, False, True, False])
    base_lift_rate = 0.4

    score = ProbeScore(
        likelihoods=likelihoods_of(np.full(len(lifted), base_lift_rate)),
        lifted=lifted,
        base_lift_rate=base_lift_rate,
    )

    assert score.skill == pytest.approx(0.0)


def test_a_confident_wrong_prediction_costs_the_clipped_loss():
    likelihoods = likelihoods_of(np.array([1.0]))

    losses = likelihoods.clipped_log_losses(np.array([False]))

    assert losses[0] == pytest.approx(-np.log(OutcomeLikelihoods.lowest_probability))


def test_a_grasp_without_density_is_refused():
    with pytest.raises(ZeroGraspDensityError):
        OutcomeLikelihoods(log_joint=np.array([[-np.inf, -np.inf]]))


# %% the two parts of the log-likelihood


def test_the_two_parts_add_up_to_the_joint_log_likelihood():
    log_joint = np.log(np.array([[0.2, 0.6], [0.5, 0.1], [0.3, 0.3]]))
    lifted = np.array([True, False, True])

    score = ProbeScore(
        likelihoods=OutcomeLikelihoods(log_joint=log_joint),
        lifted=lifted,
        base_lift_rate=0.5,
    )

    observed = np.where(
        lifted, log_joint[:, LiftOutcome.LIFTED], log_joint[:, LiftOutcome.NOT_LIFTED]
    )
    assert (
        score.average_log_likelihood_of_grasps
        + score.average_log_likelihood_of_outcomes
    ) == pytest.approx(np.mean(observed))

"""
Scoring a model's predictions of lifting on the probe set of a task.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from experiments.progressive_grasp_learning.circuit_variables import LiftOutcome
from experiments.progressive_grasp_learning.exceptions import ZeroGraspDensityError
from krrood.ormatic.utils import classproperty

# %% likelihoods of the outcomes


@dataclass(kw_only=True)
class OutcomeLikelihoods:
    """
    The joint log-likelihoods a model gives grasps together with each outcome.
    """

    log_joint: NDArray[np.float64]
    """
    One row per grasp, one column per :class:`LiftOutcome`: the log-likelihood of the
    grasp together with that outcome.
    """

    def __post_init__(self):
        """
        :raises ZeroGraspDensityError: If a grasp has zero density under both outcomes.
        """
        number_of_grasps = int(np.sum(np.isneginf(self.log_grasp_density)))
        if number_of_grasps > 0:
            raise ZeroGraspDensityError(number_of_grasps=number_of_grasps)

    @classproperty
    def lowest_probability(cls) -> float:
        """
        The lowest probability a prediction is clipped to before its logarithm.
        """
        return 0.01

    @property
    def log_grasp_density(self) -> NDArray[np.float64]:
        """
        The log-density of each grasp, whatever its outcome.
        """
        return np.logaddexp(
            self.log_joint[:, LiftOutcome.NOT_LIFTED],
            self.log_joint[:, LiftOutcome.LIFTED],
        )

    @property
    def lift_probabilities(self) -> NDArray[np.float64]:
        """
        The probability of lifting the object given each grasp.
        """
        return np.exp(self.log_joint[:, LiftOutcome.LIFTED] - self.log_grasp_density)

    def log_outcome_probabilities(
        self, lifted: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        """
        :param lifted: Whether each grasp lifted the object.
        :return: The log-probability the model gives each grasp's outcome, unclipped.
        """
        return (
            np.where(
                lifted,
                self.log_joint[:, LiftOutcome.LIFTED],
                self.log_joint[:, LiftOutcome.NOT_LIFTED],
            )
            - self.log_grasp_density
        )

    def clipped_log_losses(self, lifted: NDArray[np.bool_]) -> NDArray[np.float64]:
        """
        :param lifted: Whether each grasp lifted the object.
        :return: Each grasp's log loss of its outcome, with the probability of lifting
            clipped to :attr:`lowest_probability` and one minus it.
        """
        return self.clipped_log_losses_of(self.lift_probabilities, lifted)

    @classmethod
    def clipped_log_losses_of(
        cls, lift_probabilities: NDArray[np.float64], lifted: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        """
        :param lift_probabilities: The predicted probability of lifting for each grasp.
        :param lifted: Whether each grasp lifted the object.
        :return: Each grasp's log loss of its outcome, with the probability clipped to
            :attr:`lowest_probability` and one minus it.
        """
        clipped = np.clip(
            lift_probabilities, cls.lowest_probability, 1 - cls.lowest_probability
        )
        return -np.where(lifted, np.log(clipped), np.log(1 - clipped))


# %% the score on a probe set


@dataclass(kw_only=True)
class ProbeScore:
    """
    How well a model predicts lifting on the probe set of a task, against always
    predicting the lift rate of the task's training pool.
    """

    likelihoods: OutcomeLikelihoods
    """
    What the model gives the probe grasps.
    """

    lifted: NDArray[np.bool_]
    """
    Whether each probe grasp lifted the object.
    """

    base_lift_rate: float
    """
    The lift rate of the task's training pool, the constant prediction skill is measured
    against.
    """

    @property
    def attempt_losses(self) -> NDArray[np.float64]:
        """
        The model's clipped log loss of each probe attempt.
        """
        return self.likelihoods.clipped_log_losses(self.lifted)

    @property
    def base_attempt_losses(self) -> NDArray[np.float64]:
        """
        The clipped log loss of each probe attempt under the constant prediction.
        """
        return OutcomeLikelihoods.clipped_log_losses_of(
            np.full(len(self.lifted), self.base_lift_rate), self.lifted
        )

    @property
    def skill(self) -> float:
        """
        One minus the ratio of the model's mean loss to the constant prediction's: 0 is
        no better than the constant, 1 is perfect, below 0 is worse than the constant.
        """
        return float(
            1 - np.mean(self.attempt_losses) / np.mean(self.base_attempt_losses)
        )

    @property
    def average_log_likelihood_of_grasps(self) -> float:
        """
        The mean log-density of the probe grasps, the part of the joint log-likelihood
        that describes how grasps were sampled.
        """
        return float(np.mean(self.likelihoods.log_grasp_density))

    @property
    def average_log_likelihood_of_outcomes(self) -> float:
        """
        The mean log-probability of the probe outcomes given their grasps, unclipped,
        the part of the joint log-likelihood that describes which grasps lift.
        """
        return float(np.mean(self.likelihoods.log_outcome_probabilities(self.lifted)))

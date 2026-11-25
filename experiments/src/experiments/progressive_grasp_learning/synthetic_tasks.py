"""
Made-up grasp tasks whose lift probability is known, for checking the learning and the
scoring without a simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from experiments.grasp_learning.exported_attempt import ValueRange
from experiments.progressive_grasp_learning.task_attempts import (
    GraspNumber,
    ScaledAttempts,
)

# %% a box task


@dataclass(kw_only=True)
class BoxTask:
    """
    A made-up task over uniformly drawn scaled grasps: a grasp inside a box lifts the
    object with one probability, a grasp outside it with another.
    """

    box: dict[GraspNumber, ValueRange] = field(default_factory=dict)
    """
    The range of each number a grasp inside the box lies in; a number without a range
    does not restrict the box.
    """

    lift_probability_inside: float = 0.9
    """
    The probability that a grasp inside the box lifts the object.
    """

    lift_probability_outside: float = 0.05
    """
    The probability that a grasp outside the box lifts the object.
    """

    def lift_probabilities(self, grasps: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        :param grasps: One scaled grasp per row, numbers in the order of
            :class:`~experiments.progressive_grasp_learning.task_attempts.GraspNumber`.
        :return: The true probability that each grasp lifts the object.
        """
        grasp_numbers = list(GraspNumber)
        inside = np.ones(len(grasps), dtype=bool)
        for number, value_range in self.box.items():
            values = grasps[:, grasp_numbers.index(number)]
            inside &= (value_range.lowest <= values) & (values <= value_range.highest)
        return np.where(
            inside, self.lift_probability_inside, self.lift_probability_outside
        )

    def attempts(
        self, *, number_of_attempts: int, generator: np.random.Generator
    ) -> ScaledAttempts:
        """
        :param number_of_attempts: How many attempts are drawn.
        :param generator: Draws the grasps and their outcomes.
        :return: Attempts at grasps drawn uniformly from the unit cube, numbered from 0.
        """
        grasps = generator.uniform(size=(number_of_attempts, len(GraspNumber)))
        return ScaledAttempts(
            attempt_numbers=np.arange(number_of_attempts),
            grasps=grasps,
            lifted=generator.uniform(size=number_of_attempts)
            < self.lift_probabilities(grasps),
        )

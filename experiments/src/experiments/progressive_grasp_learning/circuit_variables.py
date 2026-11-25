"""
The variables of a circuit over a scaled grasp and its outcome, and the rows of data in
their order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray
from random_events.set import Set
from random_events.variable import Continuous, Symbolic
from sortedcontainers import SortedSet

from experiments.progressive_grasp_learning.task_attempts import GraspNumber
from krrood.ormatic.utils import classproperty

# %% the outcome


class LiftOutcome(IntEnum):
    """
    Whether an attempt lifted the object, as the outcome variable holds it.
    """

    NOT_LIFTED = 0
    """
    The attempt did not lift the object.
    """

    LIFTED = 1
    """
    The attempt lifted the object.
    """


# %% the variables


@dataclass(kw_only=True)
class GraspOutcomeVariables:
    """
    The variables of a circuit over the five scaled numbers of a grasp and whether it
    lifted the object.
    """

    variables: SortedSet = field(init=False)
    """
    The variables, in the order the circuit and every row of data hold them.
    """

    def __post_init__(self):
        self.variables = SortedSet(
            [Continuous(number.value) for number in GraspNumber]
            + [Symbolic(name=self.outcome_name, domain=Set.from_iterable(LiftOutcome))]
        )

    @classproperty
    def outcome_name(cls) -> str:
        """
        The name of the outcome variable.
        """
        return "lifted"

    def rows(
        self, *, grasps: NDArray[np.float64], outcomes: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        """
        :param grasps: One scaled grasp per row, numbers in the order of
            :class:`~experiments.progressive_grasp_learning.task_attempts.GraspNumber`.
        :param outcomes: The :class:`LiftOutcome` of each grasp.
        :return: The grasps with their outcomes, one row each, in the order of
            :attr:`variables`.
        """
        grasp_numbers = list(GraspNumber)
        columns = [
            (
                outcomes
                if variable.name == self.outcome_name
                else grasps[:, grasp_numbers.index(GraspNumber(variable.name))]
            )
            for variable in self.variables
        ]
        return np.column_stack(columns).astype(np.float64)

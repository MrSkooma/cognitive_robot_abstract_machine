"""
Exceptions of learning grasp tasks one after another.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import List

from krrood.exceptions import DataclassException

# %% scoring


@dataclass(kw_only=True)
class ZeroGraspDensityError(DataclassException):
    """
    Raised when a model gives a grasp zero density under both outcomes, so it predicts
    no probability of lifting for it.
    """

    number_of_grasps: int
    """
    How many grasps got zero density.
    """

    def error_message(self) -> str:
        return f"The model gives {self.number_of_grasps} grasps zero density."

    def suggest_correction(self) -> str:
        return (
            "Use a model whose leaves have support on the whole unit interval, such as "
            "Gaussian leaves."
        )


# %% methods


@dataclass(kw_only=True)
class MissingEarlierTaskError(DataclassException):
    """
    Raised when a method that learns from a task learned before is given none.
    """

    method_name: str
    """
    The name of the method's class.
    """

    def error_message(self) -> str:
        return f"{self.method_name} needs a task learned before the new one."

    def suggest_correction(self) -> str:
        return "Learn the earlier task first and pass it to the method."


# %% grids


@dataclass(kw_only=True)
class TooFewTrainingAttemptsError(DataclassException):
    """
    Raised when a grid asks to learn from more attempts than a task's training pool
    holds.
    """

    task_number: int
    """
    The number of the task.
    """

    number_of_training_attempts: int
    """
    The number of attempts asked for.
    """

    pool_size: int
    """
    The number of attempts in the task's training pool.
    """

    def error_message(self) -> str:
        return (
            f"Task {self.task_number} has {self.pool_size} attempts in its training "
            f"pool, fewer than the {self.number_of_training_attempts} asked for."
        )

    def suggest_correction(self) -> str:
        return "Collect more runs of the task or ask for fewer training attempts."


# %% statistics of grids


@dataclass(kw_only=True)
class NoMatchingFitsError(DataclassException):
    """
    Raised when a grid holds no fit of a method for the numbers of training attempts and
    the kind of task change asked for.
    """

    method_name: str
    """
    The name of the method asked for.
    """

    numbers_of_training_attempts: List[int]
    """
    The numbers of training attempts asked for.
    """

    def error_message(self) -> str:
        return (
            f"The grid holds no fit of {self.method_name} after "
            f"{self.numbers_of_training_attempts} training attempts."
        )

    def suggest_correction(self) -> str:
        return (
            "Ask for numbers of training attempts and a kind of task change the grid's "
            "definition holds."
        )

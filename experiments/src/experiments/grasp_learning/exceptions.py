"""
Exceptions of learning where grasps lift an object.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from krrood.exceptions import DataclassException

# %% exporting attempts


@dataclass(kw_only=True)
class GraspRangeNotAnIntervalError(DataclassException):
    """
    Raised when the statement of an object allows a number of the grasp from more than
    one interval, which a single range cannot describe.
    """

    variable_name: str
    """
    The name of the grasp's number.
    """

    number_of_intervals: int
    """
    How many intervals the statement allows it from.
    """

    def error_message(self) -> str:
        return (
            f"The statement allows {self.variable_name} from {self.number_of_intervals} "
            f"intervals, not from one."
        )

    def suggest_correction(self) -> str:
        return "Export the intervals of the statement instead of a single range."


# %% collecting runs


@dataclass(kw_only=True)
class RunNumberOutOfRangeError(DataclassException):
    """
    Raised when a run number falls outside the block of seeds its task has, so its seed
    would be the seed of a run of another task.
    """

    run_number: int
    """
    The run number given.
    """

    seeds_per_task: int
    """
    How many seeds each task has.
    """

    def error_message(self) -> str:
        return (
            f"Run number {self.run_number} is outside 0 to {self.seeds_per_task - 1}, "
            f"the run numbers a task has seeds for."
        )

    def suggest_correction(self) -> str:
        return f"Number the runs of a task from 0 to {self.seeds_per_task - 1}."


@dataclass(kw_only=True)
class CollectionRunExistsError(DataclassException):
    """
    Raised when a collection run would write a file an earlier run already wrote, which
    would mix the attempts of two runs.
    """

    path: Path
    """
    The file that exists already.
    """

    def error_message(self) -> str:
        return f"{self.path} exists already."

    def suggest_correction(self) -> str:
        return "Give the run another run number or another output directory."


@dataclass(kw_only=True)
class NoGraspReachesSurfaceError(DataclassException):
    """
    Raised when none of the grasps drawn for a run names a point the grasped part's
    surface reaches, so the run cannot store any further attempt.
    """

    number_of_grasps: int
    """
    How many grasps were drawn.
    """

    def error_message(self) -> str:
        return (
            f"None of the {self.number_of_grasps} grasps drawn names a point on the "
            f"grasped part's surface."
        )

    def suggest_correction(self) -> str:
        return "Check that the object's statement describes regions on its surface."

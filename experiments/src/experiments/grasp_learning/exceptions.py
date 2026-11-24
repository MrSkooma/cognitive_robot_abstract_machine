"""
Exceptions of learning where grasps lift an object.
"""

from __future__ import annotations

from dataclasses import dataclass

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

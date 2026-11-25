"""
Exceptions of learning grasp tasks one after another.
"""

from __future__ import annotations

from dataclasses import dataclass

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

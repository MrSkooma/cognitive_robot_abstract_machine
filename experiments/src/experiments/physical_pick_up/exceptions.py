from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Type

from krrood.exceptions import DataclassException
from semantic_digital_twin.semantic_annotations.mixins import PartWholeRelationship

# %% stored parts


@dataclass
class NoStoredPartsError(DataclassException):
    """
    Raised when the parts of a mesh are asked for that no stored answer was located in.
    """

    whole_type: Type[PartWholeRelationship]
    """
    The type of the annotation whose parts were asked for.
    """

    directory: Path
    """
    Where the stored answers are.
    """

    def error_message(self) -> str:
        return (
            f"No answer stored in {self.directory} was located in the mesh of the "
            f"{self.whole_type.__name__} asked about."
        )

    def suggest_correction(self) -> str:
        return (
            "Locate the parts of the object with the part finding server and store the "
            "answer together with the mesh it was located in."
        )


@dataclass(kw_only=True)
class NoRecordedPartsError(DataclassException):
    """
    Raised when the parts of a mesh are asked for that the part finding server was never
    asked about while its answers were recorded.
    """

    whole_type: Type[PartWholeRelationship]
    """
    The type of the annotation whose parts were asked for.
    """

    directory: Path
    """
    Where the recorded answers are.
    """

    def error_message(self) -> str:
        return (
            f"No answer recorded in {self.directory} belongs to the query about the "
            f"{self.whole_type.__name__}."
        )

    def suggest_correction(self) -> str:
        return (
            "Record the answers again with the part finding server, for instance after "
            "the mesh of the object changed."
        )


# %% grasping


@dataclass
class NoDefaultGraspError(DataclassException):
    """
    Raised when an attempt is to take an object by its default grasp, but no grasp is
    known to lift the object.
    """

    body_name: str
    """
    The name of the object's body.
    """

    def error_message(self) -> str:
        return (
            f"No grasp was given for '{self.body_name}', and its description knows "
            f"none that lifts it."
        )

    def suggest_correction(self) -> str:
        return "Give the attempt a grasp, for instance one drawn from the object's statement."

from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING
from krrood.exceptions import DataclassException

if TYPE_CHECKING:
    from random_events.variable import Variable

    from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import Unit
    from probabilistic_model.probabilistic_model import ProbabilisticModel


@dataclass
class IntractableError(DataclassException):
    """
    Exception raised when an inference is intractable for a model.

    For instance, the mode of a non-deterministic model.
    """

    model: ProbabilisticModel

    def error_message(self) -> str:
        return f"Inference is intractable for {self.model}."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class UndefinedOperationError(DataclassException):
    """
    Exception raised when an operation is not defined for a model.

    For instance, invoking the CDF of a model that contains symbolic variables.
    """

    model: ProbabilisticModel

    def error_message(self) -> str:
        return f"Operation is not defined for {self.model}."

    def suggest_correction(self) -> str:
        return ""


@dataclass
class ShapeMismatchError(DataclassException, ValueError):
    """
    Exception raised when the shape of two objects does not match.
    """

    received_shape: Any
    """
    The first object to compare.
    """

    expected_shape: Any
    """
    The second object to compare.
    """

    def error_message(self) -> str:
        return f"Expected shape {self.expected_shape}, received shape {self.received_shape}"

    def suggest_correction(self) -> str:
        return ""


@dataclass
class IncompatibleVariableDomainError(DataclassException):
    """
    Exception raised when a requested variable domain contains variables the circuit
    does not model.
    """

    variable_domain: tuple[Variable, ...]
    """
    The requested variable domain.
    """

    available_variables: tuple[Variable, ...]
    """
    The variables the circuit models.
    """

    def error_message(self) -> str:
        return (
            f"The variable domain {[variable.name for variable in self.variable_domain]} "
            f"is not a subset of the circuit's variables "
            f"{[variable.name for variable in self.available_variables]}."
        )

    def suggest_correction(self) -> str:
        return ""


@dataclass
class ColumnsDivergedError(DataclassException, ABC):
    """
    Exception raised when two columns of a progressive circuit do not share the same
    structure at an aligned position.
    """

    left: Unit
    """
    The unit of the first column at the diverging position.
    """

    right: Unit
    """
    The unit of the second column at the diverging position.
    """

    def suggest_correction(self) -> str:
        return "Build every column from the same template circuit."


@dataclass
class UnitTypeMismatchError(ColumnsDivergedError):
    """
    Exception raised when aligned units of two columns have different types.
    """

    def error_message(self) -> str:
        return (
            f"Aligned units have different types: {type(self.left).__name__} and "
            f"{type(self.right).__name__}."
        )


@dataclass
class ChildCountMismatchError(ColumnsDivergedError):
    """
    Exception raised when aligned units of two columns have a different number of
    children within their columns.
    """

    left_child_count: int
    """
    The number of children of :attr:`left` within its column.
    """

    right_child_count: int
    """
    The number of children of :attr:`right` within its column.
    """

    def error_message(self) -> str:
        return (
            f"Aligned units have a different number of children: "
            f"{self.left_child_count} and {self.right_child_count}."
        )


@dataclass
class ScopeMismatchError(ColumnsDivergedError):
    """
    Exception raised when aligned units of two columns model different variables.
    """

    def error_message(self) -> str:
        return (
            f"Aligned units model different variables: "
            f"{[variable.name for variable in self.left.variables]} and "
            f"{[variable.name for variable in self.right.variables]}."
        )


@dataclass
class UnregisteredColumnError(DataclassException):
    """
    Exception raised when a column is used with a progressive circuit it does not belong
    to.
    """

    task_id: str
    """
    The task identifier of the unregistered column.
    """

    def error_message(self) -> str:
        return f"Column {self.task_id} is not registered in this progressive circuit."

    def suggest_correction(self) -> str:
        return "Create columns with ProgressiveProbabilisticCircuit.add_column."


@dataclass
class UnsupportedLeafDistributionError(DataclassException):
    """
    Exception raised when a leaf distribution cannot be learned from weighted data.
    """

    distribution: Any
    """
    The distribution that cannot be learned.
    """

    def error_message(self) -> str:
        return (
            f"Leaf distributions of type {type(self.distribution).__name__} cannot be "
            f"learned from weighted data."
        )

    def suggest_correction(self) -> str:
        return "Use Gaussian, discrete or symbolic leaf distributions."


@dataclass
class UnsupportedVariableDomainChangeError(DataclassException):
    """
    Exception raised when the variable domain of a progressive circuit or one of its
    columns differs from the domain its existing columns were created with.
    """

    column_domain: tuple[Variable, ...]
    """
    The variable domain the existing columns were created with.
    """

    requested_domain: tuple[Variable, ...]
    """
    The changed variable domain.
    """

    def error_message(self) -> str:
        return (
            f"Changing the variable domain from "
            f"{[variable.name for variable in self.column_domain]} to "
            f"{[variable.name for variable in self.requested_domain]} after columns "
            f"were created is not supported."
        )

    def suggest_correction(self) -> str:
        return "Create a new progressive circuit for the changed variable domain."

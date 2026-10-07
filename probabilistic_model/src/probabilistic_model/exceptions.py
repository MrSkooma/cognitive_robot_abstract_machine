from __future__ import annotations
from abc import ABC
from dataclasses import dataclass
from typing import Any, Callable, List, Type, TYPE_CHECKING

from krrood.exceptions import DataclassException
from random_events.variable import Variable

if TYPE_CHECKING:
    from random_events.product_algebra import Event

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
class ProbabilisticCircuitRequiredError(DataclassException):
    """
    Exception raised when a distribution is asked for something whose answer only a
    probabilistic circuit can represent.
    """

    model: ProbabilisticModel
    """
    The distribution that was asked.
    """

    def error_message(self) -> str:
        return (
            f"{self.model} cannot answer this with a distribution of its own, since the "
            f"answer needs a probabilistic circuit."
        )

    def suggest_correction(self) -> str:
        return "Wrap the distribution into a probabilistic circuit and ask the circuit."


@dataclass
class EventIsNotABoxError(ProbabilisticCircuitRequiredError):
    """
    Exception raised when a model is asked to confine itself to something other than a
    single box.

    A box is one simple interval per variable. Anything wider leaves a shape no single
    truncated distribution describes.
    """

    event: Event
    """
    What it was asked to confine itself to.
    """

    def error_message(self) -> str:
        return f"{self.event} is not one box, so {self.model} cannot be confined to it."

    def suggest_correction(self) -> str:
        return (
            "Confine it to one simple interval per variable, or wrap the distribution "
            "into a probabilistic circuit and confine the circuit."
        )


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
class VariableNotInDistributionError(DataclassException):
    """
    Exception raised when a variable is named that a distribution is not over.

    A distribution's mean and covariance are laid out by its own variables, so a
    variable outside them has no index to be read from or written to.
    """

    variable: Variable
    """
    The variable that was named.
    """

    variables: List[Variable]
    """
    The variables the distribution is over.
    """

    def error_message(self) -> str:
        return (
            f"{self.variable} is not one of the variables "
            f"{[str(variable) for variable in self.variables]}."
        )

    def suggest_correction(self) -> str:
        return "Name one of the variables the distribution is over."


@dataclass
class NonContinuousVariableError(DataclassException, ValueError):
    """
    Exception raised when a model that only fits continuous variables is given others.
    """

    variables: List[Variable]
    """
    The variables that are not continuous.
    """

    def error_message(self) -> str:
        return (
            f"The variables {[str(variable) for variable in self.variables]} are not "
            f"continuous."
        )

    def suggest_correction(self) -> str:
        return "Fit them with a model that supports discrete variables."


@dataclass
class NoClosedFormError(DataclassException, NotImplementedError):
    """
    Exception raised when a query has no answer that the model or layer that was asked
    can represent in closed form.
    """

    asked_type: Type
    """
    The type of the model or layer that was asked.
    """

    query: Callable
    """
    The method that was queried.
    """

    def error_message(self) -> str:
        return f"{self.asked_type.__name__}.{self.query.__name__} has no closed form."

    def suggest_correction(self) -> str:
        return "Ask the query before truncating, or approximate it by sampling."


@dataclass
class InvalidMomentOrderError(DataclassException, ValueError):
    """
    Exception raised when a moment is asked for an order that is not a whole number or
    is negative.
    """

    order: Any
    """
    The order that was asked for.
    """

    def error_message(self) -> str:
        return f"There is no moment of order {self.order}."

    def suggest_correction(self) -> str:
        return "Ask for an order that is a whole number and not negative."


@dataclass
class ColumnsDivergedError(DataclassException, ABC):
    """
    Raised when two columns of a progressive circuit differ in structure at an aligned
    position.
    """

    left: Unit
    """
    The unit of the first column at that position.
    """

    right: Unit
    """
    The unit of the second column at that position.
    """

    def suggest_correction(self) -> str:
        return "Build every column from the same template circuit."


@dataclass
class UnitTypeMismatchError(ColumnsDivergedError):
    """
    Raised when aligned units of two columns have different types.
    """

    def error_message(self) -> str:
        return (
            f"Aligned units have different types: {type(self.left).__name__} and "
            f"{type(self.right).__name__}."
        )


@dataclass
class ChildCountMismatchError(ColumnsDivergedError):
    """
    Raised when aligned units of two columns have different numbers of children within
    their columns.
    """

    left_child_count: int
    """
    Number of children of :attr:`left` within its column.
    """

    right_child_count: int
    """
    Number of children of :attr:`right` within its column.
    """

    def error_message(self) -> str:
        return (
            f"Aligned units have a different number of children: "
            f"{self.left_child_count} and {self.right_child_count}."
        )


@dataclass
class ScopeMismatchError(ColumnsDivergedError):
    """
    Raised when aligned units of two columns model different variables.
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
    Raised when a column is used with a progressive circuit it does not belong to.
    """

    task_id: str
    """
    Task identifier of the column.
    """

    def error_message(self) -> str:
        return f"Column {self.task_id} is not registered in this progressive circuit."

    def suggest_correction(self) -> str:
        return "Create columns with ProgressiveProbabilisticCircuit.add_column."


@dataclass
class UnsupportedLeafDistributionError(DataclassException):
    """
    Raised when a leaf distribution cannot be learned from weighted rows.
    """

    distribution: ProbabilisticModel
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
    Raised when the variable domain of a progressive circuit or a column differs from
    the one its columns were created with.
    """

    column_domain: tuple[Variable, ...]
    """
    The variable domain the columns were created with.
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

from __future__ import annotations

# %% imports
import copy
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field

from random_events.variable import Variable
from typing_extensions import Any

from probabilistic_model.exceptions import (
    ChildCountMismatchError,
    IncompatibleVariableDomainError,
    ScopeMismatchError,
    UnitTypeMismatchError,
    UnregisteredColumnError,
    UnsupportedVariableDomainChangeError,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    SumUnit,
    Unit,
)


# %% circuit column
@dataclass
class CircuitColumn:
    """
    A task-specific copy of a template circuit inside a progressive circuit.

    .. note::

        The units of the column are recorded when it is created, so it must be created
        before any edges from it to other columns exist.
    """

    task_id: str
    """
    Identifier of the task this column represents.
    """

    root: Unit
    """
    Root unit of this column.
    """

    variable_domain: tuple[Variable, ...] | None = None
    """
    Variables this column models; defaults to the variables of :attr:`root`.
    """

    task_context: dict[str, Any] = field(default_factory=dict)
    """
    Metadata describing the task, kept alongside the column.
    """

    unit_indices: frozenset[int] = field(init=False)
    """
    Indices of every unit owned by this column.
    """

    def __post_init__(self):
        if self.variable_domain is None:
            self.variable_domain = tuple(self.root.variables)
        self.validate_variable_domain()
        descendants = self.root.probabilistic_circuit.descendants(self.root)
        self.unit_indices = frozenset(
            [self.root.index] + [unit.index for unit in descendants]
        )

    def validate_variable_domain(self) -> None:
        """
        Check that :attr:`variable_domain` only contains variables of :attr:`root`.

        :raises IncompatibleVariableDomainError: If it contains other variables.
        """
        available_variables = tuple(self.root.variables)
        if not set(self.variable_domain).issubset(available_variables):
            raise IncompatibleVariableDomainError(
                tuple(self.variable_domain), available_variables
            )

    def contains(self, unit: Unit) -> bool:
        """
        :param unit: A unit of the circuit this column belongs to.
        :return: Whether the unit is owned by this column.
        """
        return unit.index in self.unit_indices

    def children_of(self, unit: Unit) -> list[Unit]:
        """
        :param unit: A unit owned by this column.
        :return: The children of the unit that are owned by this column, excluding edges
            into other columns.
        """
        return [child for child in unit.subcircuits if self.contains(child)]


# %% aligned units
@dataclass(frozen=True)
class AlignedUnits:
    """
    Two units at the same structural position in two columns.
    """

    left: Unit
    """
    The unit of the first column.
    """

    right: Unit
    """
    The unit of the second column.
    """


# %% progressive probabilistic circuit
@dataclass
class ProgressiveProbabilisticCircuit:
    """
    A probabilistic circuit that learns tasks one after another in separate columns,
    following the idea of progressive neural networks.

    Every column is a copy of :attr:`template`. Each sum unit of a new column receives
    the aligned sum units of every earlier column as additional children, so the new
    column can reuse what earlier columns learned while their own units stay unchanged.
    """

    template: ProbabilisticCircuit
    """
    Circuit copied to create the initial structure and parameters of each column.
    """

    variable_domain: tuple[Variable, ...] | None = None
    """
    Variables every column models; defaults to the variables of :attr:`template`.
    """

    task_context: dict[str, Any] = field(default_factory=dict)
    """
    Metadata describing this progressive circuit as a whole.
    """

    columns: list[CircuitColumn] = field(default_factory=list, init=False)
    """
    Columns of this progressive circuit, oldest first.
    """

    circuit: ProbabilisticCircuit = field(init=False)
    """
    The circuit holding :attr:`root` and every column.
    """

    root: SumUnit = field(init=False)
    """
    Sum unit mixing over the roots of every column.
    """

    def __post_init__(self):
        if self.variable_domain is None:
            self.variable_domain = tuple(self.template.variables)
        self.validate_variable_domain()
        self.circuit = ProbabilisticCircuit()
        self.root = SumUnit(probabilistic_circuit=self.circuit)

    def validate_variable_domain(self) -> None:
        """
        Check that :attr:`variable_domain` only contains variables of :attr:`template`
        and still matches the domain of every existing column.

        :raises IncompatibleVariableDomainError: If it contains other variables.
        :raises UnsupportedVariableDomainChangeError: If it changed after columns were
            created.
        """
        available_variables = tuple(self.template.variables)
        if not set(self.variable_domain).issubset(available_variables):
            raise IncompatibleVariableDomainError(
                tuple(self.variable_domain), available_variables
            )
        for column in self.columns:
            if set(column.variable_domain) != set(self.variable_domain):
                raise UnsupportedVariableDomainChangeError(
                    tuple(column.variable_domain), tuple(self.variable_domain)
                )

    def validate_column(self, column: CircuitColumn) -> None:
        """
        Check that a column belongs to this progressive circuit and that its variable
        domain is still usable.

        :param column: The column to check.
        :raises UnregisteredColumnError: If the column does not belong to this
            progressive circuit.
        :raises IncompatibleVariableDomainError: If the domain of the column contains
            variables its root does not model.
        :raises UnsupportedVariableDomainChangeError: If the domain of the column
            differs from :attr:`variable_domain`.
        """
        if column not in self.columns:
            raise UnregisteredColumnError(column.task_id)
        column.validate_variable_domain()
        if set(column.variable_domain) != set(self.variable_domain):
            raise UnsupportedVariableDomainChangeError(
                tuple(self.variable_domain), tuple(column.variable_domain)
            )

    def add_column(
        self, task_id: str, task_context: dict[str, Any] | None = None
    ) -> CircuitColumn:
        """
        Add a column for a new task, connected to every earlier column.

        :param task_id: Identifier of the new task.
        :param task_context: Metadata describing the new task.
        :return: The new column.
        :raises IncompatibleVariableDomainError: If :attr:`variable_domain` contains
            variables the template does not model.
        :raises UnsupportedVariableDomainChangeError: If :attr:`variable_domain` changed
            after columns were created.
        """
        self.validate_variable_domain()
        template_copy = copy.deepcopy(self.template)
        mounted_units = self.circuit.mount(template_copy.root)
        column = CircuitColumn(
            task_id=task_id,
            root=mounted_units[template_copy.root.index],
            variable_domain=self.variable_domain,
            task_context=task_context or {},
        )
        for earlier_column in self.columns:
            self._connect_to_earlier_column(column, earlier_column)
        self.columns.append(column)
        self.root.add_subcircuit(column.root, log_weight=0.0)
        self.root.normalize()
        return column

    def _connect_to_earlier_column(
        self, column: CircuitColumn, earlier_column: CircuitColumn
    ) -> None:
        """
        Add every sum unit of an earlier column as a child of the aligned sum unit of
        ``column`` and renormalize the receiving sum units.

        :param column: The column receiving the new edges.
        :param earlier_column: The column whose sum units become children.
        """
        aligned_sum_units = [
            aligned
            for aligned in self.aligned_units(column, earlier_column)
            if isinstance(aligned.left, SumUnit)
        ]
        for aligned in aligned_sum_units:
            self.circuit.add_edge(aligned.left, aligned.right, log_weight=0.0)
            aligned.left.normalize()

    def aligned_units(
        self, left: CircuitColumn, right: CircuitColumn
    ) -> Iterator[AlignedUnits]:
        """
        Walk two columns in parallel and yield the units at matching positions.

        Edges between columns are not followed.

        :param left: The first column.
        :param right: The second column.
        :return: The aligned units, starting with the roots of both columns.
        :raises ColumnsDivergedError: If the columns differ in structure.
        """
        queue: deque[AlignedUnits] = deque([AlignedUnits(left.root, right.root)])
        visited: set[AlignedUnits] = set()
        while queue:
            aligned = queue.popleft()
            if aligned in visited:
                continue
            visited.add(aligned)
            if type(aligned.left) is not type(aligned.right):
                raise UnitTypeMismatchError(aligned.left, aligned.right)
            if tuple(aligned.left.variables) != tuple(aligned.right.variables):
                raise ScopeMismatchError(aligned.left, aligned.right)
            yield aligned
            left_children = left.children_of(aligned.left)
            right_children = right.children_of(aligned.right)
            if len(left_children) != len(right_children):
                raise ChildCountMismatchError(
                    aligned.left,
                    aligned.right,
                    len(left_children),
                    len(right_children),
                )
            queue.extend(
                AlignedUnits(left_child, right_child)
                for left_child, right_child in zip(left_children, right_children)
            )

    def units_of(self, column: CircuitColumn) -> set[Unit]:
        """
        :param column: A column of this progressive circuit.
        :return: Every unit owned by the column.
        """
        return {self.circuit.graph[index] for index in column.unit_indices}

from __future__ import annotations

# %% imports
import copy
import math
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field

from random_events.variable import Variable

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
    The copy of the template that models one task inside a progressive circuit.

    .. note::

        The column records its units when it is created, so it must be created before
        edges from it to other columns exist.
    """

    task_id: str
    """
    Identifier of the task.
    """

    root: Unit
    """
    Root unit of the column.
    """

    variable_domain: tuple[Variable, ...] | None = None
    """
    Variables of the task, a subset of the variables of :attr:`root`; defaults to all of
    them.

    The column itself always models every variable of :attr:`root`.
    """

    unit_indices: frozenset[int] = field(init=False)
    """
    Indices of the units the column owns.
    """

    sample_count: int = field(default=0, init=False)
    """
    Number of rows the column was learned from, summed over every call of :meth:`~probab
    ilistic_model.learning.progressive.ProgressiveExpectationMaximization.learn`.
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
        :raises IncompatibleVariableDomainError: If :attr:`variable_domain` contains
            variables :attr:`root` does not model.
        """
        available_variables = tuple(self.root.variables)
        if not set(self.variable_domain).issubset(available_variables):
            raise IncompatibleVariableDomainError(
                tuple(self.variable_domain), available_variables
            )

    def contains(self, unit: Unit) -> bool:
        """
        :return: Whether the column owns the unit.
        """
        return unit.index in self.unit_indices

    def children_of(self, unit: Unit) -> list[Unit]:
        """
        :param unit: A unit the column owns.
        :return: The children of the unit that the column owns, without those in other
            columns.
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
    A probabilistic circuit that learns tasks one after another, one column per task, as
    progressive neural networks do.

    Every column is a copy of :attr:`template`. Each sum unit of a new column also mixes
    the aligned sum units of every earlier column, so the new column can reuse them
    while they stay unchanged.
    """

    template: ProbabilisticCircuit
    """
    Circuit every column is copied from, structure and initial parameters.
    """

    variable_domain: tuple[Variable, ...] | None = None
    """
    Variables of the tasks, a subset of the variables of :attr:`template`; defaults to
    all of them.

    Every column still models every variable of :attr:`template`.
    """

    columns: list[CircuitColumn] = field(default_factory=list, init=False)
    """
    The columns, oldest first.
    """

    circuit: ProbabilisticCircuit = field(init=False)
    """
    The circuit holding :attr:`root` and every column.
    """

    root: SumUnit = field(init=False)
    """
    Sum unit mixing the roots of all columns.
    """

    def __post_init__(self):
        if self.variable_domain is None:
            self.variable_domain = tuple(self.template.variables)
        self.validate_variable_domain()
        self.circuit = ProbabilisticCircuit()
        self.root = SumUnit(probabilistic_circuit=self.circuit)

    def validate_variable_domain(self) -> None:
        """
        :raises IncompatibleVariableDomainError: If :attr:`variable_domain` contains
            variables :attr:`template` does not model.
        :raises UnsupportedVariableDomainChangeError: If :attr:`variable_domain` changed
            after columns were created.
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
        :raises UnregisteredColumnError: If the column belongs to another progressive
            circuit.
        :raises IncompatibleVariableDomainError: If the variable domain of the column
            contains variables its root does not model.
        :raises UnsupportedVariableDomainChangeError: If the variable domain of the
            column differs from :attr:`variable_domain`.
        """
        if column not in self.columns:
            raise UnregisteredColumnError(column.task_id)
        column.validate_variable_domain()
        if set(column.variable_domain) != set(self.variable_domain):
            raise UnsupportedVariableDomainChangeError(
                tuple(self.variable_domain), tuple(column.variable_domain)
            )

    def add_column(self, task_id: str) -> CircuitColumn:
        """
        Add a column for a new task, connected to every earlier column.

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
        Make every sum unit of ``earlier_column`` a child of the aligned sum unit of
        ``column``.
        """
        aligned_sum_units = [
            aligned
            for aligned in self.aligned_units(column, earlier_column)
            if isinstance(aligned.left, SumUnit)
        ]
        for aligned in aligned_sum_units:
            self.circuit.add_edge(aligned.left, aligned.right, log_weight=0.0)
            aligned.left.normalize()

    def restrict_root_to(self, column: CircuitColumn) -> None:
        """
        Give the whole weight of :attr:`root` to one column.

        :raises UnregisteredColumnError: If the column belongs to another progressive
            circuit.
        """
        if column not in self.columns:
            raise UnregisteredColumnError(column.task_id)
        for other_column in self.columns:
            self.circuit.add_edge(
                self.root,
                other_column.root,
                log_weight=0.0 if other_column is column else -math.inf,
            )

    def weight_root_by_sample_count(self) -> None:
        """
        Weight every column below :attr:`root` by its share of all rows the columns were
        learned from; nothing changes before any column was learned.
        """
        total_sample_count = sum(column.sample_count for column in self.columns)
        if total_sample_count == 0:
            return
        for column in self.columns:
            log_weight = (
                math.log(column.sample_count / total_sample_count)
                if column.sample_count > 0
                else -math.inf
            )
            self.circuit.add_edge(self.root, column.root, log_weight=log_weight)

    def aligned_units(
        self, left: CircuitColumn, right: CircuitColumn
    ) -> Iterator[AlignedUnits]:
        """
        Walk two columns in parallel, without following edges between columns.

        :return: The units at matching positions, starting with the roots.
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
        :return: The units the column owns.
        """
        return {self.circuit.graph[index] for index in column.unit_indices}

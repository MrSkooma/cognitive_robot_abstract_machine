from enum import IntEnum

import numpy as np
import pytest
from random_events.interval import closed
from random_events.set import Set
from random_events.variable import Continuous, Symbolic

from probabilistic_model.distributions.distributions import SymbolicDistribution
from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.exceptions import (
    ChildCountMismatchError,
    IncompatibleVariableDomainError,
    ScopeMismatchError,
    UnitTypeMismatchError,
    UnregisteredColumnError,
    UnsupportedLeafDistributionError,
    UnsupportedVariableDomainChangeError,
)
from probabilistic_model.learning.progressive import ProgressiveExpectationMaximization
from probabilistic_model.probabilistic_circuit.rx.helper import fully_factorized
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    LeafUnit,
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)
from probabilistic_model.probabilistic_circuit.rx.progressive import (
    AlignedUnits,
    CircuitColumn,
    ProgressiveProbabilisticCircuit,
)
from probabilistic_model.utils import MissingDict


# %% helpers
class Color(IntEnum):
    """
    Symbols of a symbolic test variable.
    """

    RED = 0
    BLUE = 1


def _make_factorized_template(*variable_names: str) -> ProbabilisticCircuit:
    """
    A product of one standard Gaussian leaf per variable.
    """
    return fully_factorized([Continuous(name) for name in variable_names])


def _make_mixture_template(*variable_names: str) -> ProbabilisticCircuit:
    """
    A uniform mixture of two factorized Gaussian products.
    """
    circuit = ProbabilisticCircuit()
    sum_root = SumUnit(probabilistic_circuit=circuit)
    for _ in range(2):
        component = _make_factorized_template(*variable_names)
        mounted_units = circuit.mount(component.root)
        sum_root.add_subcircuit(mounted_units[component.root.index], log_weight=0.0)
    sum_root.normalize()
    return circuit


def _gaussian_leaves(
    progressive_circuit: ProgressiveProbabilisticCircuit, column: CircuitColumn
) -> list[LeafUnit]:
    """
    The Gaussian leaf units of a column.
    """
    return [
        unit
        for unit in progressive_circuit.units_of(column)
        if isinstance(unit, LeafUnit)
        and isinstance(unit.distribution, GaussianDistribution)
    ]


def _sum_unit_weights(
    progressive_circuit: ProgressiveProbabilisticCircuit, column: CircuitColumn
) -> dict[int, list[float]]:
    """
    The log weights of every sum unit of a column, keyed by unit index.
    """
    return {
        unit.index: [float(weight) for weight, _ in unit.log_weighted_subcircuits]
        for unit in progressive_circuit.units_of(column)
        if isinstance(unit, SumUnit)
    }


def _column_log_likelihood(
    progressive_circuit: ProgressiveProbabilisticCircuit,
    column: CircuitColumn,
    data: np.ndarray,
) -> np.ndarray:
    """
    The per-sample log-likelihood computed at the root of a column.
    """
    progressive_circuit.circuit.log_likelihood(data)
    return np.array(column.root.result_of_current_query)


# %% circuit column
def test_column_defaults_variable_domain_to_root_variables():
    template = _make_factorized_template("x", "y")

    column = CircuitColumn(task_id="task-a", root=template.root)

    assert column.variable_domain == tuple(template.variables)


def test_column_rejects_variable_domain_outside_root_circuit():
    template = _make_factorized_template("x", "y")
    invalid_domain = tuple(template.variables) + (Continuous("z"),)

    with pytest.raises(IncompatibleVariableDomainError):
        CircuitColumn(
            task_id="task-a", root=template.root, variable_domain=invalid_domain
        )


def test_column_owns_every_unit_of_its_root():
    template = _make_mixture_template("x", "y")

    column = CircuitColumn(task_id="task-a", root=template.root)

    assert column.unit_indices == frozenset(unit.index for unit in template.nodes())


# %% progressive circuit initialization
def test_progressive_circuit_uses_template_domain_by_default():
    template = _make_factorized_template("x", "y")

    progressive_circuit = ProgressiveProbabilisticCircuit(template=template)

    assert progressive_circuit.template is template
    assert progressive_circuit.variable_domain == tuple(template.variables)
    assert progressive_circuit.columns == []


def test_progressive_circuit_rejects_variable_domain_outside_template():
    template = _make_factorized_template("x", "y")
    invalid_domain = tuple(template.variables) + (Continuous("z"),)

    with pytest.raises(IncompatibleVariableDomainError):
        ProgressiveProbabilisticCircuit(
            template=template, variable_domain=invalid_domain
        )


def test_progressive_circuit_leaves_template_unchanged():
    template = ProbabilisticCircuit()
    outer_sum = SumUnit(probabilistic_circuit=template)
    inner_mixture = _make_mixture_template("x", "y")
    mounted_units = template.mount(inner_mixture.root)
    outer_sum.add_subcircuit(mounted_units[inner_mixture.root.index], log_weight=0.0)
    unit_count = len(template.nodes())

    progressive_circuit = ProgressiveProbabilisticCircuit(template=template)
    progressive_circuit.add_column(task_id="task-a")

    assert len(template.nodes()) == unit_count


# %% adding columns
def test_add_column_registers_column_below_normalized_root():
    progressive_circuit = ProgressiveProbabilisticCircuit(
        template=_make_factorized_template("x", "y")
    )

    column = progressive_circuit.add_column(
        task_id="task-a", task_context={"dataset": "d1"}
    )

    assert progressive_circuit.columns == [column]
    assert column.task_context == {"dataset": "d1"}
    assert progressive_circuit.circuit.root is progressive_circuit.root
    assert progressive_circuit.circuit.has_edge(progressive_circuit.root, column.root)
    assert progressive_circuit.root.is_normalized()


def test_new_column_reads_from_earlier_column():
    progressive_circuit = ProgressiveProbabilisticCircuit(
        template=_make_mixture_template("x", "y")
    )
    earlier_column = progressive_circuit.add_column(task_id="task-a")

    column = progressive_circuit.add_column(task_id="task-b")

    assert progressive_circuit.circuit.has_edge(column.root, earlier_column.root)
    assert not progressive_circuit.circuit.has_edge(earlier_column.root, column.root)
    assert column.root.is_normalized()


def test_new_column_reads_from_every_earlier_column():
    progressive_circuit = ProgressiveProbabilisticCircuit(
        template=_make_mixture_template("x", "y")
    )
    earlier_columns = [
        progressive_circuit.add_column(task_id="task-a"),
        progressive_circuit.add_column(task_id="task-b"),
    ]

    column = progressive_circuit.add_column(task_id="task-c")

    template_child_count = len(progressive_circuit.template.root.subcircuits)
    assert len(column.root.subcircuits) == template_child_count + len(earlier_columns)
    for earlier_column in earlier_columns:
        assert progressive_circuit.circuit.has_edge(column.root, earlier_column.root)


def test_adding_column_leaves_earlier_column_unchanged():
    progressive_circuit = ProgressiveProbabilisticCircuit(
        template=_make_mixture_template("x", "y")
    )
    earlier_column = progressive_circuit.add_column(task_id="task-a")
    weights_before = _sum_unit_weights(progressive_circuit, earlier_column)

    progressive_circuit.add_column(task_id="task-b")

    assert _sum_unit_weights(progressive_circuit, earlier_column) == weights_before


# %% variable domain changes
def test_add_column_uses_changed_variable_domain_before_first_column():
    progressive_circuit = ProgressiveProbabilisticCircuit(
        template=_make_factorized_template("x", "y")
    )
    [x, _] = progressive_circuit.template.variables
    progressive_circuit.variable_domain = (x,)

    column = progressive_circuit.add_column(task_id="task-a")

    assert column.variable_domain == progressive_circuit.variable_domain


def test_add_column_rejects_changed_variable_domain_outside_template():
    progressive_circuit = ProgressiveProbabilisticCircuit(
        template=_make_factorized_template("x", "y")
    )
    progressive_circuit.variable_domain = progressive_circuit.variable_domain + (
        Continuous("z"),
    )

    with pytest.raises(IncompatibleVariableDomainError):
        progressive_circuit.add_column(task_id="task-a")


def test_add_column_rejects_variable_domain_change_after_first_column():
    progressive_circuit = ProgressiveProbabilisticCircuit(
        template=_make_factorized_template("x", "y")
    )
    progressive_circuit.add_column(task_id="task-a")
    [x, _] = progressive_circuit.template.variables
    progressive_circuit.variable_domain = (x,)

    with pytest.raises(UnsupportedVariableDomainChangeError):
        progressive_circuit.add_column(task_id="task-b")


def test_learn_rejects_column_variable_domain_outside_its_root():
    progressive_circuit = ProgressiveProbabilisticCircuit(
        template=_make_factorized_template("x")
    )
    column = progressive_circuit.add_column(task_id="task-a")
    column.variable_domain = column.variable_domain + (Continuous("z"),)

    with pytest.raises(IncompatibleVariableDomainError):
        ProgressiveExpectationMaximization(progressive_circuit).learn(
            np.array([[1.0]]), column
        )


def test_learn_rejects_changed_column_variable_domain():
    progressive_circuit = ProgressiveProbabilisticCircuit(
        template=_make_factorized_template("x", "y")
    )
    column = progressive_circuit.add_column(task_id="task-a")
    [x, _] = progressive_circuit.template.variables
    column.variable_domain = (x,)

    with pytest.raises(UnsupportedVariableDomainChangeError):
        ProgressiveExpectationMaximization(progressive_circuit).learn(
            np.array([[1.0, 1.0]]), column
        )


# %% aligned units
def test_aligned_units_pair_every_unit_of_both_columns():
    progressive_circuit = ProgressiveProbabilisticCircuit(
        template=_make_mixture_template("x", "y")
    )
    earlier_column = progressive_circuit.add_column(task_id="task-a")
    column = progressive_circuit.add_column(task_id="task-b")

    aligned_units = list(progressive_circuit.aligned_units(column, earlier_column))

    assert aligned_units[0] == AlignedUnits(column.root, earlier_column.root)
    assert {aligned.left for aligned in aligned_units} == progressive_circuit.units_of(
        column
    )
    assert {aligned.right for aligned in aligned_units} == progressive_circuit.units_of(
        earlier_column
    )


def test_aligned_units_rejects_different_unit_types():
    product_template = _make_factorized_template("x")
    sum_template = _make_mixture_template("x")
    progressive_circuit = ProgressiveProbabilisticCircuit(template=product_template)

    with pytest.raises(UnitTypeMismatchError):
        list(
            progressive_circuit.aligned_units(
                CircuitColumn(task_id="task-a", root=product_template.root),
                CircuitColumn(task_id="task-b", root=sum_template.root),
            )
        )


def test_aligned_units_rejects_different_scopes():
    left_template = _make_factorized_template("x", "y")
    right_template = _make_factorized_template("x", "z")
    progressive_circuit = ProgressiveProbabilisticCircuit(template=left_template)

    with pytest.raises(ScopeMismatchError):
        list(
            progressive_circuit.aligned_units(
                CircuitColumn(task_id="task-a", root=left_template.root),
                CircuitColumn(task_id="task-b", root=right_template.root),
            )
        )


def test_aligned_units_rejects_different_child_counts():
    two_component_template = _make_mixture_template("x")
    three_component_template = _make_mixture_template("x")
    three_component_root = three_component_template.root
    extra_component = _make_factorized_template("x")
    mounted_units = three_component_template.mount(extra_component.root)
    three_component_root.add_subcircuit(
        mounted_units[extra_component.root.index], log_weight=0.0
    )
    progressive_circuit = ProgressiveProbabilisticCircuit(
        template=two_component_template
    )

    with pytest.raises(ChildCountMismatchError):
        list(
            progressive_circuit.aligned_units(
                CircuitColumn(task_id="task-a", root=two_component_template.root),
                CircuitColumn(task_id="task-b", root=three_component_root),
            )
        )


# %% column units
def test_units_of_columns_are_disjoint():
    progressive_circuit = ProgressiveProbabilisticCircuit(
        template=_make_mixture_template("x", "y")
    )
    earlier_column = progressive_circuit.add_column(task_id="task-a")
    column = progressive_circuit.add_column(task_id="task-b")

    assert progressive_circuit.units_of(column).isdisjoint(
        progressive_circuit.units_of(earlier_column)
    )


# %% learning
def test_learnable_units_are_root_and_column_units():
    progressive_circuit = ProgressiveProbabilisticCircuit(
        template=_make_mixture_template("x", "y")
    )
    progressive_circuit.add_column(task_id="task-a")
    column = progressive_circuit.add_column(task_id="task-b")
    column_units = progressive_circuit.units_of(column)

    learnable_units = ProgressiveExpectationMaximization(
        progressive_circuit
    ).learnable_units(column)

    assert learnable_units.sum_units == {progressive_circuit.root} | {
        unit for unit in column_units if isinstance(unit, SumUnit)
    }
    assert learnable_units.leaf_units == {
        unit for unit in column_units if isinstance(unit, LeafUnit)
    }


def test_learn_fits_gaussian_leaf_to_data_of_single_column():
    progressive_circuit = ProgressiveProbabilisticCircuit(
        template=_make_factorized_template("x")
    )
    column = progressive_circuit.add_column(task_id="task-a")
    data = np.random.default_rng(42).normal(loc=5.0, scale=0.5, size=(100, 1))

    ProgressiveExpectationMaximization(progressive_circuit).learn(data, column)

    [gaussian_leaf] = _gaussian_leaves(progressive_circuit, column)
    assert gaussian_leaf.distribution.location == pytest.approx(np.mean(data))
    assert gaussian_leaf.distribution.scale == pytest.approx(np.std(data))


def test_learn_fits_symbolic_leaf_to_data_of_single_column():
    color = Symbolic("color", domain=Set.from_iterable(Color))
    template = ProbabilisticCircuit()
    leaf(
        SymbolicDistribution(
            variable=color,
            probabilities=MissingDict(
                float, {hash(Color.RED): 0.5, hash(Color.BLUE): 0.5}
            ),
        ),
        template,
    )
    progressive_circuit = ProgressiveProbabilisticCircuit(template=template)
    column = progressive_circuit.add_column(task_id="task-a")
    data = np.array([[Color.RED], [Color.RED], [Color.BLUE], [Color.RED]])

    ProgressiveExpectationMaximization(progressive_circuit).learn(data, column)

    [symbolic_leaf] = progressive_circuit.units_of(column)
    assert symbolic_leaf.distribution.probabilities[hash(Color.RED)] == pytest.approx(
        np.mean(data[:, 0] == Color.RED)
    )


def test_learn_increases_likelihood():
    progressive_circuit = ProgressiveProbabilisticCircuit(
        template=_make_mixture_template("x", "y")
    )
    column = progressive_circuit.add_column(task_id="task-a")
    data = np.random.default_rng(42).normal(loc=5.0, scale=0.5, size=(100, 2))

    history = ProgressiveExpectationMaximization(progressive_circuit).learn(
        data, column, epochs=5
    )

    assert len(history) == 5
    assert history[-1] > history[0]


def test_learning_new_column_leaves_earlier_column_unchanged():
    progressive_circuit = ProgressiveProbabilisticCircuit(
        template=_make_mixture_template("x", "y")
    )
    learner = ProgressiveExpectationMaximization(progressive_circuit)
    random = np.random.default_rng(0)
    earlier_data = random.normal(loc=0.0, scale=1.0, size=(50, 2))
    earlier_column = progressive_circuit.add_column(task_id="task-a")
    learner.learn(earlier_data, earlier_column, epochs=3)
    likelihood_before = _column_log_likelihood(
        progressive_circuit, earlier_column, earlier_data
    )
    weights_before = _sum_unit_weights(progressive_circuit, earlier_column)

    column = progressive_circuit.add_column(task_id="task-b")
    learner.learn(random.normal(loc=10.0, scale=0.2, size=(50, 2)), column, epochs=3)

    np.testing.assert_array_equal(
        _column_log_likelihood(progressive_circuit, earlier_column, earlier_data),
        likelihood_before,
    )
    assert _sum_unit_weights(progressive_circuit, earlier_column) == weights_before


def test_learning_new_column_reuses_earlier_column_that_explains_data():
    progressive_circuit = ProgressiveProbabilisticCircuit(
        template=_make_mixture_template("x", "y")
    )
    learner = ProgressiveExpectationMaximization(progressive_circuit)
    data = np.random.default_rng(0).normal(loc=5.0, scale=0.5, size=(100, 2))
    earlier_column = progressive_circuit.add_column(task_id="task-a")
    learner.learn(data, earlier_column, epochs=5)
    column = progressive_circuit.add_column(task_id="task-b")
    weight_before = progressive_circuit.circuit.graph.get_edge_data(
        column.root.index, earlier_column.root.index
    )

    learner.learn(data, column, epochs=1)

    assert (
        progressive_circuit.circuit.graph.get_edge_data(
            column.root.index, earlier_column.root.index
        )
        > weight_before
    )


def test_learn_rejects_unregistered_column():
    template = _make_factorized_template("x")
    progressive_circuit = ProgressiveProbabilisticCircuit(template=template)
    unregistered_column = CircuitColumn(task_id="ghost", root=template.root)

    with pytest.raises(UnregisteredColumnError):
        ProgressiveExpectationMaximization(progressive_circuit).learn(
            np.array([[1.0]]), unregistered_column
        )


def test_learn_rejects_unsupported_leaf_distribution():
    template = ProbabilisticCircuit()
    leaf(
        UniformDistribution(
            variable=Continuous("x"), interval=closed(0.0, 1.0).simple_sets[0]
        ),
        template,
    )
    progressive_circuit = ProgressiveProbabilisticCircuit(template=template)
    column = progressive_circuit.add_column(task_id="task-a")

    with pytest.raises(UnsupportedLeafDistributionError):
        ProgressiveExpectationMaximization(progressive_circuit).learn(
            np.array([[0.5]]), column
        )


def test_learn_returns_empty_history_for_empty_data():
    progressive_circuit = ProgressiveProbabilisticCircuit(
        template=_make_factorized_template("x")
    )
    column = progressive_circuit.add_column(task_id="task-a")

    history = ProgressiveExpectationMaximization(progressive_circuit).learn(
        np.empty((0, 1)), column
    )

    assert history == []


# %% structural properties of layouts
def _make_indicator_variable(name: str, branch_count: int) -> Symbolic:
    """
    A symbolic variable with one unique domain element per mixture branch.
    """
    return Symbolic(
        name, domain=Set.from_iterable([f"{name}-{i}" for i in range(branch_count)])
    )


def _make_indicator_mixture(
    branch_variable: Symbolic, continuous_variables: list[Continuous]
) -> ProbabilisticCircuit:
    """
    A mixture where every branch also carries a unique value of ``branch_variable``, so
    sibling branches never overlap in support and the mixture is deterministic.
    """
    circuit = ProbabilisticCircuit()
    sum_root = SumUnit(probabilistic_circuit=circuit)
    for element in branch_variable.domain.simple_sets:
        product = ProductUnit(probabilistic_circuit=circuit)
        for variable in continuous_variables:
            product.add_subcircuit(
                leaf(
                    GaussianDistribution(variable=variable, location=0.0, scale=1.0),
                    circuit,
                )
            )
        indicator = SymbolicDistribution(
            variable=branch_variable,
            probabilities=MissingDict(float, {hash(element): 1.0}),
        )
        product.add_subcircuit(leaf(indicator, circuit))
        sum_root.add_subcircuit(product, log_weight=0.0)
    sum_root.normalize()
    return circuit


def _make_wide_layout() -> ProbabilisticCircuit:
    """
    A single 5-branch mixture spread over 6 continuous variables.
    """
    branch = _make_indicator_variable("wide_branch", branch_count=5)
    variables = [Continuous(f"wide_x{i}") for i in range(6)]
    return _make_indicator_mixture(branch, variables)


def _make_nested_layout() -> ProbabilisticCircuit:
    """
    A mixture-of-mixtures: one outer branch nests another indicator mixture.
    """
    circuit = ProbabilisticCircuit()
    top_branch = Symbolic(
        "nested_top", domain=Set.from_iterable(["outer-0", "outer-1"])
    )
    outer_variables = [Continuous(f"nested_x{i}") for i in range(3)]
    inner_branch = _make_indicator_variable("nested_inner", branch_count=3)
    inner_variables = [Continuous(f"nested_y{i}") for i in range(3)]

    sum_root = SumUnit(probabilistic_circuit=circuit)
    products = []
    for top_value in ["outer-0", "outer-1"]:
        product = ProductUnit(probabilistic_circuit=circuit)
        for variable in outer_variables:
            product.add_subcircuit(
                leaf(
                    GaussianDistribution(variable=variable, location=0.0, scale=1.0),
                    circuit,
                )
            )
        product.add_subcircuit(
            leaf(
                SymbolicDistribution(
                    variable=top_branch,
                    probabilities=MissingDict(float, {hash(top_value): 1.0}),
                ),
                circuit,
            )
        )
        sum_root.add_subcircuit(product, log_weight=0.0)
        products.append(product)

    inner_mixture = _make_indicator_mixture(inner_branch, inner_variables)
    mounted_units = circuit.mount(inner_mixture.root)
    products[1].add_subcircuit(mounted_units[inner_mixture.root.index])
    sum_root.normalize()
    return circuit


def _make_split_layout() -> ProbabilisticCircuit:
    """
    A product of two independent indicator mixtures over disjoint variables.
    """
    circuit = ProbabilisticCircuit()
    root = ProductUnit(probabilistic_circuit=circuit)
    for name, branch_count in [("split_left", 2), ("split_right", 3)]:
        mixture = _make_indicator_mixture(
            _make_indicator_variable(name, branch_count=branch_count),
            [Continuous(f"{name}_x{i}") for i in range(3)],
        )
        mounted_units = circuit.mount(mixture.root)
        root.add_subcircuit(mounted_units[mixture.root.index])
    return circuit


def _any_sum_unit(
    progressive_circuit: ProgressiveProbabilisticCircuit, column: CircuitColumn
) -> SumUnit:
    """
    Any one sum unit belonging to the given column.
    """
    return next(
        unit
        for unit in progressive_circuit.units_of(column)
        if isinstance(unit, SumUnit)
    )


@pytest.mark.parametrize(
    "make_layout",
    [_make_wide_layout, _make_nested_layout, _make_split_layout],
    ids=["wide", "nested", "split"],
)
def test_edges_between_columns_preserve_decomposability_but_break_determinism(
    make_layout,
):
    template = make_layout()
    assert template.is_decomposable()
    assert template.is_deterministic()

    progressive_circuit = ProgressiveProbabilisticCircuit(template=template)
    progressive_circuit.add_column(task_id="task-a")
    assert progressive_circuit.circuit.is_decomposable()
    assert progressive_circuit.circuit.is_deterministic()

    progressive_circuit.add_column(task_id="task-b")
    progressive_circuit.add_column(task_id="task-c")

    # Edges between columns connect sum units of equal scope and never touch the
    # children of a product unit.
    assert progressive_circuit.circuit.is_decomposable()
    # A sum unit reading from an earlier column gains a child with the same support as
    # its own branches.
    assert not progressive_circuit.circuit.is_deterministic()
    oldest_sum_unit = _any_sum_unit(progressive_circuit, progressive_circuit.columns[0])
    newest_sum_unit = _any_sum_unit(
        progressive_circuit, progressive_circuit.columns[-1]
    )
    assert oldest_sum_unit.is_deterministic()
    assert not newest_sum_unit.is_deterministic()

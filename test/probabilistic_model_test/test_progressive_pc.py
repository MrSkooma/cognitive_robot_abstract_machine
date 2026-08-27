import numpy as np
import pytest
from random_events.variable import Continuous

from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.learning.progressive import PCColumn, ProgressivePC
from probabilistic_model.probabilistic_circuit.rx.helper import fully_factorized
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    LeafUnit,
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
)


# %% helpers
def _make_template_pc(*variable_names: str) -> ProbabilisticCircuit:
    return fully_factorized([Continuous(name) for name in variable_names])


def _make_mixture_template(*variable_names: str) -> ProbabilisticCircuit:
    pc = ProbabilisticCircuit()
    sum_root = SumUnit(probabilistic_circuit=pc)
    prod1 = fully_factorized([Continuous(name) for name in variable_names])
    prod2 = fully_factorized([Continuous(name) for name in variable_names])
    m1 = pc.mount(prod1.root)
    m2 = pc.mount(prod2.root)
    sum_root.add_subcircuit(m1[prod1.root.index], log_weight=0.0)
    sum_root.add_subcircuit(m2[prod2.root.index], log_weight=0.0)
    sum_root.normalize()
    return pc


# %% pc-column
def test_pc_column_initialization_sets_attributes():
    template = _make_template_pc("x", "y")
    variable_domain = tuple(template.variables)

    column = PCColumn(
        task_id="task-a",
        column_root=template.root,
        variable_domain=variable_domain,
        task_context={"step": 1},
    )

    assert column.task_id == "task-a"
    assert column.column_root is template.root
    assert column.variable_domain == variable_domain
    assert column.task_context == {"step": 1}


def test_pc_column_defaults_variable_domain_to_root_variables():
    template = _make_template_pc("x", "y")

    column = PCColumn(
        task_id="task-a",
        column_root=template.root,
    )

    assert column.variable_domain == tuple(template.variables)


def test_pc_column_rejects_variable_domains_outside_root_circuit():
    template = _make_template_pc("x", "y")
    invalid_domain = tuple(template.variables) + (Continuous("z"),)

    with pytest.raises(ValueError, match="compatible"):
        PCColumn(
            task_id="task-a",
            column_root=template.root,
            variable_domain=invalid_domain,
        )


# %% progressive-pc initialization
def test_progressive_pc_from_template_uses_template_domain_by_default():
    template = _make_template_pc("x", "y")

    progressive_pc = ProgressivePC.from_template(template_pc=template)

    assert progressive_pc.template_pc is template
    assert progressive_pc.variable_domain == tuple(template.variables)
    assert progressive_pc.columns == []
    assert isinstance(progressive_pc.ppc.root, SumUnit)


def test_progressive_pc_rejects_none_template():
    with pytest.raises(ValueError, match="template_pc must not be None"):
        ProgressivePC(template_pc=None)


def test_progressive_pc_rejects_variable_domain_outside_template():
    template = _make_template_pc("x", "y")
    invalid_domain = tuple(template.variables) + (Continuous("z"),)

    with pytest.raises(ValueError, match="compatible"):
        ProgressivePC(
            template_pc=template,
            variable_domain=invalid_domain,
        )


# %% adding columns
def test_add_column_registers_column_and_connects_to_root():
    template = _make_template_pc("x", "y")
    progressive_pc = ProgressivePC(template_pc=template)

    column = progressive_pc.add_column(task_id="task-a", task_context={"dataset": "d1"})

    assert progressive_pc.columns == [column]
    assert column.task_id == "task-a"
    assert column.task_context == {"dataset": "d1"}
    assert progressive_pc.ppc.has_edge(progressive_pc.ppc.root, column.column_root)
    assert progressive_pc.ppc.root.is_normalized()


def test_add_multiple_columns_creates_lateral_connections_for_sum_nodes():
    template = _make_mixture_template("x", "y")
    progressive_pc = ProgressivePC(template_pc=template)

    column1 = progressive_pc.add_column(task_id="task-a")
    column2 = progressive_pc.add_column(task_id="task-b")

    assert progressive_pc.columns == [column1, column2]
    assert progressive_pc.ppc.has_edge(progressive_pc.ppc.root, column1.column_root)
    assert progressive_pc.ppc.has_edge(progressive_pc.ppc.root, column2.column_root)
    assert progressive_pc.ppc.has_edge(column2.column_root, column1.column_root)
    assert column2.column_root.is_normalized()


# %% aligned node iteration
def test_iter_aligned_nodes_returns_matching_node_pairs():
    template = _make_mixture_template("x", "y")
    progressive_pc = ProgressivePC(template_pc=template)
    left_column = progressive_pc.add_column(task_id="task-a")
    right_column = progressive_pc.add_column(task_id="task-b")

    aligned_nodes = list(progressive_pc.iter_aligned_nodes(left_column, right_column))

    assert len(aligned_nodes) > 0
    assert all(type(left) is type(right) for left, right in aligned_nodes)
    assert (left_column.column_root, right_column.column_root) in aligned_nodes


def test_iter_aligned_nodes_skips_lateral_sum_to_sum_connections():
    template = _make_mixture_template("x", "y")
    progressive_pc = ProgressivePC(template_pc=template)
    left_column = progressive_pc.add_column(task_id="task-a")
    right_column = progressive_pc.add_column(task_id="task-b")

    assert progressive_pc.ppc.has_edge(
        right_column.column_root, left_column.column_root
    )

    aligned_nodes = list(progressive_pc.iter_aligned_nodes(left_column, right_column))

    left_visited = [left for left, _ in aligned_nodes]
    right_visited = [right for _, right in aligned_nodes]
    assert left_column.column_root in left_visited
    assert right_column.column_root in right_visited


def test_iter_aligned_nodes_rejects_columns_with_different_node_types():
    prod_template = fully_factorized([Continuous("x")])
    sum_template = _make_mixture_template("x")

    left_column = PCColumn(task_id="task-a", column_root=prod_template.root)
    right_column = PCColumn(task_id="task-b", column_root=sum_template.root)

    progressive_pc = ProgressivePC(template_pc=prod_template)

    with pytest.raises(ValueError, match="Columns diverged: node types differ"):
        list(progressive_pc.iter_aligned_nodes(left_column, right_column))


def test_iter_aligned_nodes_rejects_columns_with_different_branching():
    template_two_children = fully_factorized([Continuous("x"), Continuous("y")])
    template_three_children = fully_factorized(
        [Continuous("x"), Continuous("y"), Continuous("z")]
    )

    left_column = PCColumn(task_id="task-a", column_root=template_two_children.root)
    right_column = PCColumn(task_id="task-b", column_root=template_three_children.root)

    progressive_pc = ProgressivePC(template_pc=template_two_children)

    with pytest.raises(
        ValueError, match="Columns diverged: different number of children"
    ):
        list(progressive_pc.iter_aligned_nodes(left_column, right_column))


# %% column node extraction
def test_get_column_nodes_excludes_lateral_subgraphs():
    template = _make_mixture_template("x", "y")
    progressive_pc = ProgressivePC(template_pc=template)
    column1 = progressive_pc.add_column(task_id="task-a")
    column2 = progressive_pc.add_column(task_id="task-b")

    nodes_col1 = progressive_pc.get_column_nodes(column1)
    nodes_col2 = progressive_pc.get_column_nodes(column2)

    assert column1.column_root in nodes_col1
    assert column2.column_root in nodes_col2
    # Column 2's internal nodes must not overlap with Column 1's nodes
    assert nodes_col1.isdisjoint(nodes_col2)


# %% learning
def test_learn_updates_column_parameters_and_increases_likelihood():
    template = _make_mixture_template("x", "y")
    progressive_pc = ProgressivePC(template_pc=template)
    column = progressive_pc.add_column(task_id="task-a")

    np.random.seed(42)
    data = np.random.normal(loc=5.0, scale=0.5, size=(100, 2))

    history = progressive_pc.learn(data=data, column=column, epochs=5)

    assert len(history) == 5
    # Likelihood should improve or stay non-decreasing
    assert history[-1] > history[0]

    # Leaf distributions should have updated toward mean ~ 5.0
    col_leaves = [
        node
        for node in progressive_pc.get_column_nodes(column)
        if isinstance(node, LeafUnit)
        and isinstance(node.distribution, GaussianDistribution)
    ]
    for leaf in col_leaves:
        assert 4.0 <= leaf.distribution.location <= 6.0


def test_learn_freezes_previous_columns_when_learning_new_column():
    template = _make_mixture_template("x", "y")
    progressive_pc = ProgressivePC(template_pc=template)

    column1 = progressive_pc.add_column(task_id="task-a")
    column2 = progressive_pc.add_column(task_id="task-b")

    # Record column 1 leaves parameters before training column 2
    col1_leaves = [
        node
        for node in progressive_pc.get_column_nodes(column1)
        if isinstance(node, LeafUnit)
        and isinstance(node.distribution, GaussianDistribution)
    ]
    col1_initial_locations = [leaf.distribution.location for leaf in col1_leaves]
    col1_initial_scales = [leaf.distribution.scale for leaf in col1_leaves]

    # Train on column 2 only with data centered at 10.0
    np.random.seed(42)
    data_col2 = np.random.normal(loc=10.0, scale=0.2, size=(50, 2))
    progressive_pc.learn(data=data_col2, column=column2, epochs=3)

    # Column 1 parameters must remain completely unchanged
    col1_after_locations = [leaf.distribution.location for leaf in col1_leaves]
    col1_after_scales = [leaf.distribution.scale for leaf in col1_leaves]

    assert col1_initial_locations == col1_after_locations
    assert col1_initial_scales == col1_after_scales

    # Column 2 parameters must have adapted toward ~10.0
    col2_leaves = [
        node
        for node in progressive_pc.get_column_nodes(column2)
        if isinstance(node, LeafUnit)
        and isinstance(node.distribution, GaussianDistribution)
    ]
    for leaf in col2_leaves:
        assert 8.0 <= leaf.distribution.location <= 12.0


def test_learn_rejects_unregistered_column():
    template = _make_template_pc("x")
    progressive_pc = ProgressivePC(template_pc=template)
    unregistered = PCColumn(task_id="ghost", column_root=template.root)

    with pytest.raises(ValueError, match="not registered"):
        progressive_pc.learn(data=np.array([[1.0]]), column=unregistered)


def test_learn_handles_empty_data():
    template = _make_template_pc("x")
    progressive_pc = ProgressivePC(template_pc=template)
    column = progressive_pc.add_column(task_id="task-a")

    history = progressive_pc.learn(data=np.empty((0, 1)), column=column)
    assert history == []

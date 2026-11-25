"""
Splitting the variables of a region graph into regions down to single variables.
"""

from __future__ import annotations

import random

import pytest
from random_events.variable import Continuous
from sortedcontainers import SortedSet

from probabilistic_model.learning.region_graph.region_graph import RegionGraph

# %% fixtures


@pytest.fixture
def region_graph_split_to_single_variables() -> RegionGraph:
    """
    A region graph over six variables, split twice, so that some regions reach a single
    variable before the last split.
    """
    random.seed(1)
    return RegionGraph(
        SortedSet([Continuous(str(index)) for index in range(6)]),
        partitions=2,
        depth=2,
        repetitions=2,
    ).create_random_region_graph()


# %% partitions


def test_every_partition_splits_into_regions(region_graph_split_to_single_variables):
    empty_partitions = [
        partition
        for partition in region_graph_split_to_single_variables.partition_nodes()
        if region_graph_split_to_single_variables.out_degree(partition) == 0
    ]

    assert empty_partitions == []


def test_a_circuit_is_built_over_every_variable(region_graph_split_to_single_variables):
    circuit = region_graph_split_to_single_variables.as_probabilistic_circuit(
        input_units=2, sum_units=2
    )

    assert circuit.variables == region_graph_split_to_single_variables.variables

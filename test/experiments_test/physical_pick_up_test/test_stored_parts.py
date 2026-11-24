from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from krrood.adapters.json_serializer import to_json

from experiments.physical_pick_up.exceptions import (
    NoRecordedPartsError,
    NoStoredPartsError,
)
from experiments.physical_pick_up.stored_parts import (
    AnswerStoringLocator,
    RecordedAnswerLocator,
    StoredPart,
    StoredPartLocator,
)
from semantic_digital_twin.adapters.part_finding.server import PartFindingCache
from semantic_digital_twin.part_finding.locating import LocatedPart, PartQuery
from semantic_digital_twin.semantic_annotations.semantic_annotations import Mug

# %% fixtures


@pytest.fixture
def mesh() -> trimesh.Trimesh:
    return trimesh.creation.box(extents=(0.1, 0.1, 0.1))


@pytest.fixture
def directory(tmp_path: Path, mesh: trimesh.Trimesh) -> Path:
    """
    Holds one stored answer: the faces 0 to 3 of :func:`mesh` are a handle.
    """
    np.savez_compressed(
        tmp_path / "mug_0.npz", vertices=mesh.vertices, faces=mesh.faces
    )
    (tmp_path / "mug_0.json").write_text(
        json.dumps(
            {
                "object": "mug_0",
                "parts": [
                    to_json(
                        StoredPart(
                            relationship="handle", faces=[0, 1, 2, 3], confidence=0.9
                        )
                    )
                ],
            }
        )
    )
    return tmp_path


# %% answering from stored answers


def test_a_query_of_a_stored_mesh_is_answered_with_the_stored_parts(directory, mesh):
    locator = StoredPartLocator(directory=directory)

    [part] = locator.locate(PartQuery.from_mesh(mesh, Mug, ["handle", "rim"]))

    assert part.relationship_name == "handle"
    np.testing.assert_array_equal(part.faces, [0, 1, 2, 3])
    assert part.confidence == 0.9


def test_a_stored_mesh_is_answered_also_once_it_is_turned(directory, mesh):
    locator = StoredPartLocator(directory=directory)
    turned = mesh.copy().apply_transform(
        trimesh.transformations.rotation_matrix(0.7, [0.0, 0.0, 1.0])
    )

    [part] = locator.locate(PartQuery.from_mesh(turned, Mug, ["handle"]))

    np.testing.assert_array_equal(part.faces, [0, 1, 2, 3])


def test_a_stored_mesh_is_answered_also_once_its_vertices_are_rounded(directory, mesh):
    locator = StoredPartLocator(directory=directory)
    rounded = trimesh.Trimesh(
        vertices=mesh.vertices.astype(np.float32), faces=mesh.faces, process=False
    )

    [part] = locator.locate(PartQuery.from_mesh(rounded, Mug, ["handle"]))

    np.testing.assert_array_equal(part.faces, [0, 1, 2, 3])


def test_a_stored_mesh_is_answered_also_once_its_duplicate_vertices_are_merged(
    directory, mesh
):
    locator = StoredPartLocator(directory=directory)
    unmerged = trimesh.Trimesh(
        vertices=mesh.vertices[mesh.faces].reshape(-1, 3),
        faces=np.arange(3 * len(mesh.faces)).reshape(-1, 3),
        process=False,
    )

    [part] = locator.locate(PartQuery.from_mesh(unmerged, Mug, ["handle"]))

    np.testing.assert_array_equal(part.faces, [0, 1, 2, 3])


def test_only_the_parts_asked_for_are_answered(directory, mesh):
    locator = StoredPartLocator(directory=directory)

    assert locator.locate(PartQuery.from_mesh(mesh, Mug, ["rim"])) == []


def test_a_mesh_without_a_stored_answer_cannot_be_answered(directory):
    locator = StoredPartLocator(directory=directory)
    other_mesh = trimesh.creation.box(extents=(0.2, 0.1, 0.1))

    with pytest.raises(NoStoredPartsError):
        locator.locate(PartQuery.from_mesh(other_mesh, Mug, ["handle"]))


def test_a_mesh_without_a_stored_answer_has_no_parts_unless_an_answer_is_required(
    directory,
):
    locator = StoredPartLocator(directory=directory, require_stored_answer=False)
    other_mesh = trimesh.creation.box(extents=(0.2, 0.1, 0.1))

    assert locator.locate(PartQuery.from_mesh(other_mesh, Mug, ["handle"])) == []


# %% storing answers


def test_a_stored_answer_is_answered_again_without_the_locator_that_found_it(
    directory, mesh, tmp_path
):
    query = PartQuery.from_mesh(mesh, Mug, ["handle"])
    storing_locator = AnswerStoringLocator(
        answering_locator=StoredPartLocator(directory=directory),
        directory=tmp_path / "stored",
        object_name="mug",
    )
    [found] = storing_locator.locate(query)

    [answered] = StoredPartLocator(directory=tmp_path / "stored").locate(query)

    assert answered.relationship_name == found.relationship_name
    np.testing.assert_array_equal(answered.faces, found.faces)
    assert answered.confidence == found.confidence


def test_a_recorded_query_is_answered_with_its_recorded_parts(mesh, tmp_path):
    query = PartQuery.from_mesh(mesh, Mug, ["handle"])
    recorded_answers = PartFindingCache(directory=tmp_path)
    recorded_answers.write(
        recorded_answers.key(query, locator=""),
        [
            LocatedPart(
                relationship_name="handle",
                faces=np.array([4, 5], dtype=np.int64),
                confidence=0.8,
            )
        ],
    )

    [part] = RecordedAnswerLocator(recorded_answers=recorded_answers).locate(query)

    assert part.relationship_name == "handle"
    np.testing.assert_array_equal(part.faces, [4, 5])
    assert part.confidence == 0.8


def test_a_query_never_recorded_cannot_be_answered(mesh, tmp_path):
    locator = RecordedAnswerLocator(
        recorded_answers=PartFindingCache(directory=tmp_path)
    )

    with pytest.raises(NoRecordedPartsError):
        locator.locate(PartQuery.from_mesh(mesh, Mug, ["handle"]))

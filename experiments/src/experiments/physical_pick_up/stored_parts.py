"""
Answering part queries with the parts a part finding server located before, so that an
experiment splits its objects as the server would without asking it again.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Dict, List

from krrood.adapters.json_serializer import from_json, to_json

from experiments.physical_pick_up.exceptions import (
    NoRecordedPartsError,
    NoStoredPartsError,
)
from semantic_digital_twin.adapters.part_finding.server import PartFindingCache
from semantic_digital_twin.part_finding.locating import (
    LocatedPart,
    PartLocator,
    PartQuery,
)

# %% the stored answers


@dataclass(kw_only=True)
class StoredPart:
    """
    One part a part finding server located, as an answer file stores it.
    """

    relationship: str
    """
    The part-whole relationship the part fills.
    """

    faces: List[int]
    """
    The faces of the part, as indices into the mesh the answer was located in.
    """

    confidence: float
    """
    How sure the server is of the part.
    """

    @classmethod
    def from_located_part(cls, part: LocatedPart) -> StoredPart:
        """
        :param part: A part as a locator answers it.
        :return: The part as an answer file stores it.
        """
        return cls(
            relationship=part.relationship_name,
            faces=[int(face) for face in part.faces],
            confidence=float(part.confidence),
        )

    def located_part(self) -> LocatedPart:
        """
        :return: The part as a locator answers it.
        """
        return LocatedPart(
            relationship_name=self.relationship,
            faces=np.asarray(self.faces, dtype=np.int64),
            confidence=self.confidence,
        )


# %% answering from the stored answers


@dataclass
class StoredAnswer:
    """
    The parts stored for one mesh, with what the mesh is recognized by.
    """

    edge_lengths: NDArray[np.float64]
    """
    The length of every edge of every face of the mesh, which stays the same when the
    mesh is moved or turned.
    """

    parts: List[StoredPart]
    """
    The parts located in the mesh.
    """


@dataclass(kw_only=True)
class StoredPartLocator(PartLocator):
    """
    Answers a query with the parts stored for the mesh it asks about.

    The directory holds, for every object, ``<object>.json``, the server's answer, whose
    ``parts`` are :class:`StoredPart` serialized by
    :mod:`krrood.adapters.json_serializer`, and ``<object>.npz``, the ``vertices`` and
    ``faces`` of the mesh the answer was located in. A query is answered for a mesh
    whose faces, in the same order, have edges as long, so also once the mesh is moved
    or turned or its duplicate vertices are merged.
    """

    directory: Path
    """
    Where the stored answers are.
    """

    require_stored_answer: bool = True
    """
    Whether a mesh needs a stored answer; one that does not is taken to have no parts.
    """

    edge_length_tolerance: float = 1e-5
    """
    How much the edges of a queried mesh may differ in length from those of a stored
    one, in meters, for the two to be the same mesh.
    """

    answers_by_face_count: Dict[int, List[StoredAnswer]] = field(init=False, repr=False)
    """
    The stored answers, by the number of faces of the mesh they were located in.
    """

    def __post_init__(self) -> None:
        self.answers_by_face_count = {}
        for mesh_file in sorted(self.directory.glob("*.npz")):
            answer_file = mesh_file.with_suffix(".json")
            if not answer_file.is_file():
                continue
            mesh = np.load(mesh_file)
            answer = json.loads(answer_file.read_text())
            self.answers_by_face_count.setdefault(len(mesh["faces"]), []).append(
                StoredAnswer(
                    edge_lengths=self.edge_lengths(
                        vertices=mesh["vertices"], faces=mesh["faces"]
                    ),
                    parts=[from_json(part) for part in answer["parts"]],
                )
            )

    def locate(self, query: PartQuery) -> list[LocatedPart]:
        """
        :raises NoStoredPartsError: If no answer is stored for the query's mesh and
            :attr:`require_stored_answer` says it has to be.
        """
        edge_lengths = self.edge_lengths(vertices=query.vertices, faces=query.faces)
        answers = [
            answer
            for answer in self.answers_by_face_count.get(len(query.faces), [])
            if np.allclose(
                answer.edge_lengths,
                edge_lengths,
                rtol=0.0,
                atol=self.edge_length_tolerance,
            )
        ]
        if not answers and not self.require_stored_answer:
            return []
        if not answers:
            raise NoStoredPartsError(
                whole_type=query.whole_type, directory=self.directory
            )
        return [
            part.located_part()
            for part in answers[0].parts
            if part.relationship in query.relationship_names
        ]

    @staticmethod
    def edge_lengths(
        *, vertices: NDArray[np.float64], faces: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        """
        :param vertices: The vertices of a mesh.
        :param faces: Its faces.
        :return: The length of every edge of every face, face by face.
        """
        corners = np.asarray(vertices, dtype=np.float64)[np.asarray(faces)]
        return np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2)


# %% storing answers


@dataclass(kw_only=True)
class RecordedAnswerLocator(PartLocator):
    """
    Answers a query with the parts the part finding server answered it with while its
    answers were recorded, one file per query, as :class:`PartFindingCache` keeps them.
    """

    recorded_answers: PartFindingCache
    """
    The recorded answers.
    """

    def locate(self, query: PartQuery) -> list[LocatedPart]:
        """
        :raises NoRecordedPartsError: If the query was never recorded.
        """
        parts = self.recorded_answers.read(self.recorded_answers.key(query, locator=""))
        if parts is None:
            raise NoRecordedPartsError(
                whole_type=query.whole_type, directory=self.recorded_answers.directory
            )
        return parts


@dataclass(kw_only=True)
class AnswerStoringLocator(PartLocator):
    """
    Answers a query with the parts another locator finds, and stores every answer with
    its mesh in :attr:`directory`, so that a :class:`StoredPartLocator` answers the same
    queries without that locator.
    """

    answering_locator: PartLocator
    """
    The locator whose answers are stored.
    """

    directory: Path
    """
    Where the answers are stored.
    """

    object_name: str
    """
    The name of the object whose parts are asked for; every answer file begins with it.
    """

    stored_answer_count: int = field(default=0, init=False)
    """
    How many answers were stored so far.
    """

    def locate(self, query: PartQuery) -> list[LocatedPart]:
        parts = self.answering_locator.locate(query)
        stem = f"{self.object_name}_{self.stored_answer_count}"
        self.directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.directory / f"{stem}.npz", vertices=query.vertices, faces=query.faces
        )
        answer = {
            "object": stem,
            "parts": [to_json(StoredPart.from_located_part(part)) for part in parts],
        }
        (self.directory / f"{stem}.json").write_text(json.dumps(answer))
        self.stored_answer_count += 1
        return parts

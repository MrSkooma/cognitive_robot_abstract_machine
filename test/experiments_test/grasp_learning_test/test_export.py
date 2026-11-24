"""
Exporting stored grasp attempts to a file of JSON lines and reading them back.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest

from .test_grasp_learning import database, milk_scene, synthetic_attempts, task_of

from experiments.grasp_learning.export import AttemptExport
from experiments.grasp_learning.exported_attempt import ExportFile
from semantic_digital_twin.robots.tracy import Tracy

# %% fixtures


@pytest.fixture
def export(milk_scene, database) -> AttemptExport:
    """
    An export of a database holding five attempts of the scene's robot on the milk.
    """
    database.add_attempts(synthetic_attempts(milk_scene, 5))
    return AttemptExport(scene=milk_scene, database=database, commit="test-commit")


# %% the exported attempts


def test_every_stored_attempt_is_exported_with_its_grasp_and_result(
    export, milk_scene, database
):
    stored = database.attempts(task_of(milk_scene))

    exported = export.exported_attempts()

    assert [attempt.grasp.height for attempt in exported] == [
        attempt.grasp.height for attempt in stored
    ]
    assert [attempt.result.lifted for attempt in exported] == [
        attempt.grasp.result.lifted for attempt in stored
    ]


def test_every_exported_grasp_lies_in_its_ranges(export):
    for attempt in export.exported_attempts():
        ranges = asdict(attempt.grasp_ranges)
        for name, value in asdict(attempt.grasp).items():
            assert ranges[name]["lowest"] <= value < ranges[name]["highest"]


def test_attempts_of_another_robot_are_not_exported(milk_scene, database):
    attempts = synthetic_attempts(milk_scene, 2)
    attempts[0].robot = Tracy
    database.add_attempts(attempts)

    exported = AttemptExport(
        scene=milk_scene, database=database, commit="test-commit"
    ).exported_attempts()

    assert [attempt.robot for attempt in exported] == [type(milk_scene.robot).__name__]


# %% the file


def test_an_export_file_reads_back_the_attempts_written(export, tmp_path):
    attempts = export.exported_attempts()
    export_file = ExportFile(path=tmp_path / "attempts.jsonl")

    export_file.write(attempts=attempts)

    assert export_file.read() == attempts

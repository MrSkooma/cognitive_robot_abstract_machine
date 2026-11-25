"""
Collecting numbered runs of grasp attempts: their seeds, their files and their manifest.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from .test_grasp_learning import (
    milk_scene,
    simulates_physics,
    synthetic_attempts,
    task_of,
)

from experiments.grasp_learning import collection
from experiments.grasp_learning.collection import CollectionRun, NumberedRun
from experiments.grasp_learning.collection_files import (
    CollectionDirectory,
    CollectionFileSuffix,
    CollectionManifest,
)
from experiments.grasp_learning.database import GraspDatabase
from experiments.grasp_learning.exceptions import (
    CollectionRunExistsError,
    RunNumberOutOfRangeError,
)
from experiments.grasp_learning.exported_attempt import ExportFile

# %% fixtures


@pytest.fixture
def run(milk_scene, tmp_path) -> CollectionRun:
    """
    A run of the scene's robot on the milk, writing into a fresh directory.
    """
    return CollectionRun(
        scene=milk_scene,
        numbered_run=NumberedRun(task_number=3, run_number=1),
        output_directory=tmp_path / "run",
        commit="test-commit",
        number_of_attempts=2,
    )


@pytest.fixture
def run_database(run) -> GraspDatabase:
    """
    The run's own database, holding six synthetic attempts.
    """
    run.output_directory.mkdir(parents=True)
    run_database = GraspDatabase.connect(
        f"sqlite:///{run.path(CollectionFileSuffix.DATABASE)}"
    )
    run_database.add_attempts(synthetic_attempts(run.scene, 6))
    return run_database


# %% seeds


def test_runs_of_different_tasks_never_share_a_seed():
    seeds = [
        NumberedRun(task_number=task_number, run_number=run_number).seed
        for task_number in range(1, 15)
        for run_number in range(NumberedRun.seeds_per_task)
    ]

    assert len(set(seeds)) == len(seeds)


@pytest.mark.parametrize("run_number", [-1, NumberedRun.seeds_per_task])
def test_a_run_number_outside_its_tasks_block_of_seeds_is_refused(run_number):
    with pytest.raises(RunNumberOutOfRangeError):
        NumberedRun(task_number=1, run_number=run_number)


# %% files of a run


def test_a_run_refuses_to_write_over_an_earlier_run(run):
    run.output_directory.mkdir(parents=True)
    run.path(CollectionFileSuffix.DATABASE).touch()

    with pytest.raises(CollectionRunExistsError):
        run.collect()


def test_the_exported_attempts_are_the_stored_attempts(run, run_database):
    stored = run_database.attempts(task_of(run.scene))

    run.write_results(
        database=run_database,
        number_of_grasps_missing_the_surface=0,
        stored_part_files=[],
        started=datetime.now(timezone.utc),
    )

    exported = ExportFile(path=run.path(CollectionFileSuffix.ATTEMPTS)).read()
    assert [attempt.grasp.height for attempt in exported] == [
        attempt.grasp.height for attempt in stored
    ]


def test_the_manifest_counts_the_stored_and_the_lifting_attempts(run, run_database):
    stored = run_database.attempts(task_of(run.scene))

    manifest = run.write_results(
        database=run_database,
        number_of_grasps_missing_the_surface=0,
        stored_part_files=[],
        started=datetime.now(timezone.utc),
    )

    assert manifest.number_of_attempts == len(stored)
    assert manifest.number_of_lifting_attempts == sum(
        attempt.grasp.result.lifted for attempt in stored
    )


def test_the_manifest_reads_back_as_written(run, run_database):
    manifest = run.write_results(
        database=run_database,
        number_of_grasps_missing_the_surface=3,
        stored_part_files=["bowl.json"],
        started=datetime.now(timezone.utc),
    )

    assert (
        CollectionManifest.read(path=run.path(CollectionFileSuffix.MANIFEST))
        == manifest
    )


def test_the_manifest_type_lives_outside_the_module_run_as_a_command():
    assert CollectionManifest.__module__ != collection.__name__


def test_the_files_of_the_stored_parts_directory_are_listed(run, tmp_path):
    run.stored_parts_directory = tmp_path / "parts"
    run.stored_parts_directory.mkdir()
    (run.stored_parts_directory / "cup.json").touch()
    (run.stored_parts_directory / "bowl.json").touch()

    assert run.stored_part_files() == ["bowl.json", "cup.json"]


def test_an_empty_stored_parts_directory_lists_no_files(run, tmp_path):
    run.stored_parts_directory = tmp_path / "parts"
    run.stored_parts_directory.mkdir()

    assert run.stored_part_files() == []


def test_the_runs_of_a_task_are_found_in_run_number_order(milk_scene, tmp_path):
    for run_number in [2, 1]:
        run = CollectionRun(
            scene=milk_scene,
            numbered_run=NumberedRun(task_number=3, run_number=run_number),
            output_directory=tmp_path,
            commit="test-commit",
        )
        run_database = GraspDatabase.connect(
            f"sqlite:///{run.path(CollectionFileSuffix.DATABASE)}"
        )
        run_database.add_attempts(synthetic_attempts(milk_scene, 2))
        run.write_results(
            database=run_database,
            number_of_grasps_missing_the_surface=0,
            stored_part_files=[],
            started=datetime.now(timezone.utc),
        )

    runs = CollectionDirectory(path=tmp_path).runs_of_task(3)

    assert [run.manifest.run_number for run in runs] == [1, 2]
    assert [run.attempts_path.exists() for run in runs] == [True, True]


# %% repeating a run


@simulates_physics
def test_the_same_numbered_run_draws_the_same_grasps_again(milk_scene, tmp_path):
    numbered_run = NumberedRun(task_number=3, run_number=1)
    runs = [
        CollectionRun(
            scene=milk_scene,
            numbered_run=numbered_run,
            output_directory=tmp_path / directory,
            commit="test-commit",
            number_of_attempts=2,
        )
        for directory in ["first", "second"]
    ]

    for repeated_run in runs:
        repeated_run.collect()

    first, second = (
        ExportFile(path=repeated_run.path(CollectionFileSuffix.ATTEMPTS)).read()
        for repeated_run in runs
    )
    assert [attempt.grasp for attempt in first] == [attempt.grasp for attempt in second]

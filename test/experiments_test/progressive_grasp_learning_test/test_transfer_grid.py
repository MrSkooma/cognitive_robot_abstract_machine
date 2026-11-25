"""
Pairing tasks and running a grid of fits over their attempts.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys

import numpy as np
import pytest

from .test_task_attempts import exported_attempt, middle_grasp, rim_statement

from experiments.grasp_learning.collection_files import (
    CollectionDirectory,
    CollectionFileSuffix,
    CollectionManifest,
)
from experiments.grasp_learning.exported_attempt import ExportFile, ValueRange
from experiments.progressive_grasp_learning.exceptions import (
    TooFewTrainingAttemptsError,
)
from experiments.progressive_grasp_learning.methods import (
    LearningSettings,
    RatSpnTemplate,
)
from experiments.progressive_grasp_learning import run_grid
from experiments.progressive_grasp_learning.synthetic_tasks import BoxTask
from experiments.progressive_grasp_learning.task_attempts import GraspNumber
from experiments.progressive_grasp_learning.transfer_grid import (
    ComparedMethod,
    FitResult,
    GridDefinition,
    GridFile,
    TaskChange,
    TaskIdentity,
    TaskPair,
    TransferGrid,
)
from krrood.adapters.json_serializer import from_json, to_json

# %% fixtures

PR2_BOWL = TaskIdentity(task_number=1, robot="PR2", object_name="bowl")
PR2_MUG = TaskIdentity(task_number=7, robot="PR2", object_name="mug")
TRACY_BOWL = TaskIdentity(task_number=4, robot="Tracy", object_name="bowl")


def write_box_task(
    directory: CollectionDirectory, task: TaskIdentity, *, number_of_attempts: int
) -> None:
    """
    Write one finished collection run of a task whose grasps lift exactly when their
    scaled height lies in a box that depends on the task number.
    """
    statement = rim_statement()
    box = BoxTask(
        box={GraspNumber.HEIGHT: ValueRange(lowest=0.2, highest=0.6)},
        lift_probability_inside=0.9,
        lift_probability_outside=0.1,
    )
    scaled = box.attempts(
        number_of_attempts=number_of_attempts,
        generator=np.random.default_rng(task.task_number),
    )
    attempts = []
    for grasp_row, lifted in zip(scaled.grasps, scaled.lifted):
        grasp = middle_grasp(statement)
        for number, value in zip(GraspNumber, grasp_row):
            value_range = number.range_of(statement)
            scaled_value = value_range.lowest + value * (
                value_range.highest - value_range.lowest
            )
            match number:
                case GraspNumber.AZIMUTH:
                    grasp.azimuth = scaled_value
                case GraspNumber.HEIGHT:
                    grasp.height = scaled_value
                case GraspNumber.DEPTH:
                    grasp.depth = scaled_value
                case GraspNumber.PITCH:
                    grasp.pitch = scaled_value
                case GraspNumber.ROLL:
                    grasp.roll = scaled_value
        attempts.append(
            exported_attempt(statement=statement, grasp=grasp, lifted=bool(lifted))
        )
    name = f"{task.robot.lower()}-{task.object_name}-task{task.task_number}-run1"
    ExportFile(path=directory.path / f"{name}{CollectionFileSuffix.ATTEMPTS}").write(
        attempts=attempts
    )
    CollectionManifest(
        robot=task.robot,
        object_name=task.object_name,
        task_number=task.task_number,
        run_number=1,
        seed=1000 * task.task_number + 1,
        number_of_attempts=len(attempts),
        number_of_lifting_attempts=sum(attempt.result.lifted for attempt in attempts),
        number_of_grasps_missing_the_surface=0,
        stored_parts_directory=None,
        stored_part_files=[],
        commit="test-commit",
        started="2026-11-25T00:00:00+00:00",
        wall_seconds=1.0,
    ).write(path=directory.path / f"{name}{CollectionFileSuffix.MANIFEST}")


@pytest.fixture
def definition() -> GridDefinition:
    """
    A small grid over three tasks, with every method.
    """
    small = LearningSettings(
        template=RatSpnTemplate(depth=1, units_per_region=2), iterations=3
    )
    return GridDefinition(
        task_numbers=[
            PR2_BOWL.task_number,
            PR2_MUG.task_number,
            TRACY_BOWL.task_number,
        ],
        scaling_task_numbers=[PR2_BOWL.task_number],
        numbers_of_training_attempts=[3, 6],
        number_of_repetitions=2,
        settings=small,
        scratch_candidates=[small],
        methods=list(ComparedMethod),
    )


@pytest.fixture
def directory(tmp_path) -> CollectionDirectory:
    """
    A collection directory holding 40 attempts of each of the three tasks.
    """
    directory = CollectionDirectory(path=tmp_path / "collection")
    directory.path.mkdir()
    for task in [PR2_BOWL, PR2_MUG, TRACY_BOWL]:
        write_box_task(directory, task, number_of_attempts=40)
    return directory


# %% pairs


def test_ordered_pairs_change_exactly_one_of_robot_and_object():
    pairs = TaskPair.ordered_pairs([PR2_BOWL, PR2_MUG, TRACY_BOWL])

    assert {(pair.earlier, pair.new, pair.change) for pair in pairs} == {
        (PR2_BOWL, PR2_MUG, TaskChange.OBJECT),
        (PR2_MUG, PR2_BOWL, TaskChange.OBJECT),
        (PR2_BOWL, TRACY_BOWL, TaskChange.ROBOT),
        (TRACY_BOWL, PR2_BOWL, TaskChange.ROBOT),
    }


# %% running a grid


def test_a_grid_writes_one_fit_per_method_pairing_size_and_repetition(
    definition, directory, tmp_path
):
    grid = TransferGrid(definition=definition, directory=directory)

    number_of_fits = grid.run(output_directory=tmp_path / "grid", number_of_workers=2)

    pairs = TaskPair.ordered_pairs([PR2_BOWL, PR2_MUG, TRACY_BOWL])
    new_tasks = {pair.new for pair in pairs}
    methods_reading_earlier = [
        method
        for method in definition.methods
        if method.transfer_method(definition).reads_earlier_task
    ]
    pairings = len(pairs) * len(methods_reading_earlier) + len(new_tasks) * (
        len(definition.methods) - len(methods_reading_earlier)
    )
    assert (
        number_of_fits
        == pairings
        * len(definition.numbers_of_training_attempts)
        * definition.number_of_repetitions
    )


def test_the_fits_file_reads_back_the_written_results(definition, directory, tmp_path):
    output_directory = tmp_path / "grid"
    number_of_fits = TransferGrid(definition=definition, directory=directory).run(
        output_directory=output_directory, number_of_workers=1
    )

    results = [
        from_json(json.loads(line))
        for line in (output_directory / GridFile.FITS).read_text().splitlines()
    ]

    assert len(results) == number_of_fits
    assert all(isinstance(result, FitResult) for result in results)
    assert all(math.isfinite(result.skill) for result in results)


def test_methods_that_ignore_the_earlier_task_name_none(
    definition, directory, tmp_path
):
    output_directory = tmp_path / "grid"
    TransferGrid(definition=definition, directory=directory).run(
        output_directory=output_directory, number_of_workers=1
    )

    results = [
        from_json(json.loads(line))
        for line in (output_directory / GridFile.FITS).read_text().splitlines()
    ]

    assert {
        result.earlier_task_number
        for result in results
        if result.method in (ComparedMethod.FIXED_SCRATCH, ComparedMethod.TUNED_SCRATCH)
    } == {None}


def test_asking_for_more_attempts_than_the_pool_holds_is_refused(definition, directory):
    definition.numbers_of_training_attempts = [1000]
    grid = TransferGrid(definition=definition, directory=directory)

    with pytest.raises(TooFewTrainingAttemptsError):
        grid.units(grid.scaled_tasks())[0].run()


# %% the command line


def test_the_fits_the_command_writes_read_back_as_results(
    definition, directory, tmp_path
):
    definition_path = tmp_path / "definition.json"
    definition_path.write_text(json.dumps(to_json(definition)))
    output_directory = tmp_path / "grid"

    subprocess.run(
        [
            sys.executable,
            "-m",
            run_grid.__name__,
            "--definition",
            str(definition_path),
            "--collection-directory",
            str(directory.path),
            "--output-directory",
            str(output_directory),
        ],
        check=True,
        capture_output=True,
    )

    results = [
        from_json(json.loads(line))
        for line in (output_directory / GridFile.FITS).read_text().splitlines()
    ]
    assert all(isinstance(result, FitResult) for result in results)

"""
Reading the attempts of a task, scaling their grasps and splitting them into a training
pool and a probe set.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from typing_extensions import List

from experiments.grasp_learning.collection_files import (
    CollectionDirectory,
    CollectionFileSuffix,
    CollectionManifest,
)
from experiments.grasp_learning.exported_attempt import (
    ExportFile,
    ExportedAttempt,
    ExportedGrasp,
    ExportedGraspRanges,
    ExportedPlacement,
    ExportedResult,
    ExportedRotation,
    ExportedVector,
    ValueRange,
)
from experiments.progressive_grasp_learning.task_attempts import (
    GraspNumber,
    GraspScaling,
    ScaledAttempts,
    TaskAttempts,
)

# %% exported attempts without a simulation


def rim_statement() -> ExportedGraspRanges:
    """
    Ranges shaped like a rim statement: narrow pitch and roll.
    """
    return ExportedGraspRanges(
        azimuth=ValueRange(lowest=0.0, highest=2 * math.pi),
        height=ValueRange(lowest=0.78, highest=0.93),
        depth=ValueRange(lowest=0.001, highest=0.003),
        pitch=ValueRange(lowest=0.0, highest=0.5),
        roll=ValueRange(lowest=-0.39, highest=0.39),
    )


def body_statement() -> ExportedGraspRanges:
    """
    Ranges shaped like the statement of a box's body: wide pitch and roll.
    """
    return ExportedGraspRanges(
        azimuth=ValueRange(lowest=0.0, highest=2 * math.pi),
        height=ValueRange(lowest=0.05, highest=0.95),
        depth=ValueRange(lowest=0.0, highest=0.03),
        pitch=ValueRange(lowest=0.0, highest=1.2),
        roll=ValueRange(lowest=-math.pi / 4, highest=math.pi / 4),
    )


def exported_attempt(
    *,
    statement: ExportedGraspRanges,
    grasp: ExportedGrasp,
    lifted: bool = False,
    attempt_number: int = 1,
) -> ExportedAttempt:
    """
    :return: An attempt with the given grasp, drawn from the given statement.
    """
    zero = ExportedVector(x=0.0, y=0.0, z=0.0)
    no_rotation = ExportedRotation(axis=ExportedVector(x=0.0, y=0.0, z=1.0), angle=0.0)
    return ExportedAttempt(
        attempt_number=attempt_number,
        robot="PR2",
        object_name="bowl",
        source="STATED_REGIONS",
        placement=ExportedPlacement(x=0.0, y=0.0, yaw=0.0),
        grasp=grasp,
        grasp_ranges=statement,
        result=ExportedResult(
            lifted=lifted,
            object_raised=lifted,
            motion_completed=True,
            object_displacement=zero,
            object_rotation=no_rotation,
            translational_slip=zero,
            rotational_slip=no_rotation,
        ),
        commit="test-commit",
    )


def middle_grasp(statement: ExportedGraspRanges) -> ExportedGrasp:
    """
    :return: The grasp in the middle of every range of the statement.
    """
    middle = {
        number: (number.range_of(statement).lowest + number.range_of(statement).highest)
        / 2
        for number in GraspNumber
    }
    return ExportedGrasp(**middle)


def write_run(
    directory: CollectionDirectory,
    *,
    task_number: int,
    run_number: int,
    attempts: List[ExportedAttempt],
) -> None:
    """
    Write the files a finished collection run leaves behind.
    """
    name = f"pr2-bowl-task{task_number}-run{run_number}"
    ExportFile(path=directory.path / f"{name}{CollectionFileSuffix.ATTEMPTS}").write(
        attempts=attempts
    )
    CollectionManifest(
        robot="PR2",
        object_name="bowl",
        task_number=task_number,
        run_number=run_number,
        seed=1000 * task_number + run_number,
        number_of_attempts=len(attempts),
        number_of_lifting_attempts=sum(attempt.result.lifted for attempt in attempts),
        number_of_grasps_missing_the_surface=0,
        stored_parts_directory=None,
        stored_part_files=[],
        commit="test-commit",
        started="2026-11-25T00:00:00+00:00",
        wall_seconds=1.0,
    ).write(path=directory.path / f"{name}{CollectionFileSuffix.MANIFEST}")


# %% scaling


def test_a_statements_bounds_map_to_zero_and_one():
    statement = rim_statement()
    grasp = middle_grasp(statement)
    grasp.height = statement.height.lowest
    grasp.depth = statement.depth.highest

    scaled = GraspScaling().scaled_grasp(
        exported_attempt(statement=statement, grasp=grasp)
    )

    assert scaled[list(GraspNumber).index(GraspNumber.HEIGHT)] == pytest.approx(0.0)
    assert scaled[list(GraspNumber).index(GraspNumber.DEPTH)] == pytest.approx(1.0)


def test_the_shared_range_of_an_angle_is_its_widest_over_the_statements():
    scaling = GraspScaling.angles_shared_by([rim_statement(), body_statement()])

    assert scaling.shared_ranges == {
        GraspNumber.AZIMUTH: body_statement().azimuth,
        GraspNumber.PITCH: body_statement().pitch,
        GraspNumber.ROLL: body_statement().roll,
    }


def test_an_angle_scales_alike_in_tasks_with_different_statements():
    scaling = GraspScaling.angles_shared_by([rim_statement(), body_statement()])
    pitch = list(GraspNumber).index(GraspNumber.PITCH)
    scaled = []
    for statement in [rim_statement(), body_statement()]:
        grasp = middle_grasp(statement)
        grasp.pitch = 0.4
        scaled.append(
            scaling.scaled_grasp(exported_attempt(statement=statement, grasp=grasp))[
                pitch
            ]
        )

    assert scaled[0] == pytest.approx(scaled[1])


# %% the training pool and the probe set


def scaled_attempts(number_of_attempts: int) -> ScaledAttempts:
    """
    :return: Attempts numbered from 0, every third one lifting.
    """
    return ScaledAttempts(
        attempt_numbers=np.arange(number_of_attempts),
        grasps=np.zeros((number_of_attempts, len(GraspNumber))),
        lifted=np.arange(number_of_attempts) % 3 == 0,
    )


def test_the_probe_set_holds_the_attempts_the_split_rule_names():
    attempts = scaled_attempts(10)

    probe_numbers = attempts.probe_set().attempt_numbers

    assert probe_numbers.tolist() == [
        number
        for number in range(10)
        if number % ScaledAttempts.split_modulus in ScaledAttempts.probe_remainders
    ]


def test_pool_and_probe_set_together_hold_every_attempt_once():
    attempts = scaled_attempts(1000)

    numbers = np.concatenate(
        [
            attempts.training_pool().attempt_numbers,
            attempts.probe_set().attempt_numbers,
        ]
    )

    assert sorted(numbers.tolist()) == list(range(1000))


# %% reading the runs of a task


def test_attempts_are_numbered_in_run_order(tmp_path):
    directory = CollectionDirectory(path=tmp_path)
    statement = rim_statement()
    heights = {1: [0.80, 0.81], 2: [0.90, 0.91]}
    for run_number in [2, 1]:
        attempts = []
        for height in heights[run_number]:
            grasp = middle_grasp(statement)
            grasp.height = height
            attempts.append(exported_attempt(statement=statement, grasp=grasp))
        write_run(directory, task_number=4, run_number=run_number, attempts=attempts)

    task = TaskAttempts.read(directory=directory, task_number=4)

    assert [attempt.grasp.height for attempt in task.attempts] == (
        heights[1] + heights[2]
    )


def test_runs_of_other_tasks_are_not_read(tmp_path):
    directory = CollectionDirectory(path=tmp_path)
    statement = rim_statement()
    for task_number in [4, 5]:
        write_run(
            directory,
            task_number=task_number,
            run_number=1,
            attempts=[
                exported_attempt(statement=statement, grasp=middle_grasp(statement))
            ],
        )

    task = TaskAttempts.read(directory=directory, task_number=4)

    assert len(task.attempts) == 1

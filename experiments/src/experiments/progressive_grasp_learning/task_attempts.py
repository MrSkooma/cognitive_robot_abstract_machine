"""
The attempts of one grasp task, read from the task's collection runs, numbered in the
order they were stored, scaled onto the unit interval and split into a training pool and
a probe set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Iterable, List

from experiments.grasp_learning.collection_files import CollectionDirectory
from experiments.grasp_learning.exported_attempt import (
    ExportFile,
    ExportedAttempt,
    ExportedGrasp,
    ExportedGraspRanges,
    ValueRange,
)
from krrood.ormatic.utils import classproperty

# %% the numbers of a grasp


class GraspNumber(StrEnum):
    """
    The five numbers of a surface grasp, in the order a scaled grasp holds them.
    """

    AZIMUTH = "azimuth"
    """
    The direction around the object's vertical axis.
    """

    HEIGHT = "height"
    """
    The height of the grasped point on the object.
    """

    DEPTH = "depth"
    """
    How far behind the surface point the gripper closes.
    """

    PITCH = "pitch"
    """
    The tilt of the approach away from straight down.
    """

    ROLL = "roll"
    """
    The turn of the closing direction about the approach.
    """

    @property
    def is_angle(self) -> bool:
        """
        Whether the number is an angle, which means the same gripper orientation on
        every object.
        """
        return self in (GraspNumber.AZIMUTH, GraspNumber.PITCH, GraspNumber.ROLL)

    def value_of(self, grasp: ExportedGrasp) -> float:
        """
        :param grasp: An exported grasp.
        :return: This number of the grasp.
        """
        match self:
            case GraspNumber.AZIMUTH:
                return grasp.azimuth
            case GraspNumber.HEIGHT:
                return grasp.height
            case GraspNumber.DEPTH:
                return grasp.depth
            case GraspNumber.PITCH:
                return grasp.pitch
            case GraspNumber.ROLL:
                return grasp.roll

    def range_of(self, ranges: ExportedGraspRanges) -> ValueRange:
        """
        :param ranges: The ranges a statement draws a grasp's numbers from.
        :return: The range of this number.
        """
        match self:
            case GraspNumber.AZIMUTH:
                return ranges.azimuth
            case GraspNumber.HEIGHT:
                return ranges.height
            case GraspNumber.DEPTH:
                return ranges.depth
            case GraspNumber.PITCH:
                return ranges.pitch
            case GraspNumber.ROLL:
                return ranges.roll


# %% scaling


@dataclass(kw_only=True)
class GraspScaling:
    """
    Maps each number of a grasp linearly onto 0 to 1: by the range in
    :attr:`shared_ranges` where it holds one for the number, otherwise by the range of
    the statement the attempt was drawn from.
    """

    shared_ranges: dict[GraspNumber, ValueRange] = field(default_factory=dict)
    """
    The ranges every task shares, by the number they scale.
    """

    @classmethod
    def angles_shared_by(
        cls, statements: Iterable[ExportedGraspRanges]
    ) -> GraspScaling:
        """
        :param statements: The ranges of the statements of every task compared.
        :return: A scaling that maps each angle by its widest range over the statements
            and the other numbers by each attempt's own statement.
        """
        statements = list(statements)
        shared_ranges = {}
        for number in GraspNumber:
            if not number.is_angle:
                continue
            ranges = [number.range_of(statement) for statement in statements]
            shared_ranges[number] = ValueRange(
                lowest=min(value_range.lowest for value_range in ranges),
                highest=max(value_range.highest for value_range in ranges),
            )
        return cls(shared_ranges=shared_ranges)

    def scaled_grasp(self, attempt: ExportedAttempt) -> NDArray[np.float64]:
        """
        :param attempt: An exported attempt.
        :return: Its grasp's numbers in the order of :class:`GraspNumber`, each mapped
            onto 0 to 1.
        """
        scaled = []
        for number in GraspNumber:
            value_range = self.shared_ranges.get(
                number, number.range_of(attempt.grasp_ranges)
            )
            scaled.append(
                (number.value_of(attempt.grasp) - value_range.lowest)
                / (value_range.highest - value_range.lowest)
            )
        return np.array(scaled)


# %% scaled attempts


@dataclass(kw_only=True)
class ScaledAttempts:
    """
    Attempts of one task as arrays: their numbers, their scaled grasps and whether each
    lifted the object.
    """

    attempt_numbers: NDArray[np.int64]
    """
    The number of each attempt within its task.
    """

    grasps: NDArray[np.float64]
    """
    One row per attempt, the grasp's numbers in the order of :class:`GraspNumber`.
    """

    lifted: NDArray[np.bool_]
    """
    Whether each attempt lifted the object.
    """

    @classproperty
    def split_modulus(cls) -> int:
        """
        The number an attempt's number is divided by to decide its side of the split.
        """
        return 5

    @classproperty
    def probe_remainders(cls) -> tuple[int, ...]:
        """
        The remainders that put an attempt into the probe set.
        """
        return (3, 4)

    @property
    def lift_rate(self) -> float:
        """
        The share of the attempts that lifted the object.
        """
        return float(np.mean(self.lifted))

    def rows(self, selected: NDArray[np.bool_] | NDArray[np.int64]) -> ScaledAttempts:
        """
        :param selected: A mask over the attempts, or the positions of the attempts.
        :return: The selected attempts.
        """
        return ScaledAttempts(
            attempt_numbers=self.attempt_numbers[selected],
            grasps=self.grasps[selected],
            lifted=self.lifted[selected],
        )

    def probe_set(self) -> ScaledAttempts:
        """
        :return: The attempts held back for scoring models, never learned from.
        """
        return self.rows(
            np.isin(self.attempt_numbers % self.split_modulus, self.probe_remainders)
        )

    def training_pool(self) -> ScaledAttempts:
        """
        :return: The attempts models learn from.
        """
        return self.rows(
            ~np.isin(self.attempt_numbers % self.split_modulus, self.probe_remainders)
        )


# %% the attempts of a task


@dataclass(kw_only=True)
class TaskAttempts:
    """
    Every collected attempt of one task, numbered from 0 in the order of the runs and,
    within a run, in the order they were stored.
    """

    task_number: int
    """
    The number of the task.
    """

    attempts: List[ExportedAttempt]
    """
    The attempts; an attempt's position is its number.
    """

    @classmethod
    def read(cls, *, directory: CollectionDirectory, task_number: int) -> TaskAttempts:
        """
        :param directory: Where the task's collection runs wrote their files.
        :param task_number: The number of the task.
        :return: The attempts of every finished run of the task.
        """
        return cls(
            task_number=task_number,
            attempts=[
                attempt
                for run in directory.runs_of_task(task_number)
                for attempt in ExportFile(path=run.attempts_path).read()
            ],
        )

    def scaled(self, scaling: GraspScaling) -> ScaledAttempts:
        """
        :param scaling: How the grasps' numbers are mapped onto 0 to 1.
        :return: The attempts as arrays.
        """
        return ScaledAttempts(
            attempt_numbers=np.arange(len(self.attempts)),
            grasps=np.array(
                [scaling.scaled_grasp(attempt) for attempt in self.attempts]
            ),
            lifted=np.array([attempt.result.lifted for attempt in self.attempts]),
        )

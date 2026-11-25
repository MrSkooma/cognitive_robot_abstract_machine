"""
Learning every new task of a grid from a few of its attempts, by every compared method,
after each task it is paired with, and scoring each model on the new task's probe set.
"""

from __future__ import annotations

import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
from typing_extensions import List

from experiments.grasp_learning.collection_files import CollectionDirectory
from experiments.progressive_grasp_learning.circuit_variables import (
    GraspOutcomeVariables,
)
from experiments.progressive_grasp_learning.exceptions import (
    TooFewTrainingAttemptsError,
)
from experiments.progressive_grasp_learning.methods import (
    EarlierColumnUnchanged,
    EarlierTask,
    FixedScratch,
    LearningSettings,
    ProgressiveColumn,
    TransferMethod,
    TunedScratch,
)
from experiments.progressive_grasp_learning.scoring import (
    OutcomeLikelihoods,
    ProbeScore,
)
from experiments.progressive_grasp_learning.task_attempts import (
    GraspScaling,
    ScaledAttempts,
    TaskAttempts,
)
from krrood.adapters.json_serializer import to_json

# %% compared methods


class ComparedMethod(StrEnum):
    """
    The methods a grid compares.
    """

    PROGRESSIVE_COLUMN = "progressive column"
    """
    A new column beside the earlier task's frozen column.
    """

    TUNED_SCRATCH = "tuned scratch"
    """
    A single column whose settings are chosen by cross-validation on the new attempts.
    """

    FIXED_SCRATCH = "fixed scratch"
    """
    A single column with the grid's settings.
    """

    EARLIER_COLUMN_UNCHANGED = "earlier column unchanged"
    """
    The earlier task's column, not learned on the new attempts.
    """

    def transfer_method(self, definition: GridDefinition) -> TransferMethod:
        """
        :param definition: The grid the method runs in.
        :return: The method, set up with the grid's settings.
        """
        match self:
            case ComparedMethod.PROGRESSIVE_COLUMN:
                return ProgressiveColumn(settings=definition.settings)
            case ComparedMethod.TUNED_SCRATCH:
                return TunedScratch(candidates=definition.scratch_candidates)
            case ComparedMethod.FIXED_SCRATCH:
                return FixedScratch(settings=definition.settings)
            case ComparedMethod.EARLIER_COLUMN_UNCHANGED:
                return EarlierColumnUnchanged()


# %% tasks and pairs


class TaskChange(StrEnum):
    """
    What differs between an earlier task and a new one.
    """

    OBJECT = "object"
    """
    The same robot grasps another object.
    """

    ROBOT = "robot"
    """
    Another robot grasps the same object.
    """


@dataclass(frozen=True, kw_only=True)
class TaskIdentity:
    """
    The number of a task and the robot and object it is about.
    """

    task_number: int
    """
    The number of the task.
    """

    robot: str
    """
    The name of the robot's class.
    """

    object_name: str
    """
    The name of the object's body.
    """

    @classmethod
    def of_task(
        cls, *, directory: CollectionDirectory, task_number: int
    ) -> TaskIdentity:
        """
        :param directory: Where the task's runs wrote their files.
        :param task_number: The number of the task.
        :return: The identity the task's first run recorded.
        """
        manifest = directory.runs_of_task(task_number)[0].manifest
        return cls(
            task_number=task_number,
            robot=manifest.robot,
            object_name=manifest.object_name,
        )


@dataclass(frozen=True, kw_only=True)
class TaskPair:
    """
    An earlier task and a new task learned after it.
    """

    earlier: TaskIdentity
    """
    The task learned first.
    """

    new: TaskIdentity
    """
    The task learned from few attempts after it.
    """

    @property
    def change(self) -> TaskChange | None:
        """
        What differs between the two tasks; none if both robot and object differ, or
        neither does.
        """
        same_robot = self.earlier.robot == self.new.robot
        same_object = self.earlier.object_name == self.new.object_name
        if same_robot and not same_object:
            return TaskChange.OBJECT
        if same_object and not same_robot:
            return TaskChange.ROBOT
        return None

    @classmethod
    def ordered_pairs(cls, tasks: List[TaskIdentity]) -> List[TaskPair]:
        """
        :param tasks: The tasks of a grid.
        :return: Every ordered pair of them that differs in exactly one of robot and
            object.
        """
        pairs = [
            cls(earlier=earlier, new=new)
            for earlier in tasks
            for new in tasks
            if earlier != new
        ]
        return [pair for pair in pairs if pair.change is not None]


# %% the definition of a grid


@dataclass(kw_only=True)
class GridDefinition:
    """
    Which tasks, numbers of attempts, repetitions, settings and methods a grid runs.
    """

    task_numbers: List[int]
    """
    The tasks learned after one another in every ordered pair of them.
    """

    scaling_task_numbers: List[int]
    """
    The tasks whose statements set the ranges the angles of every task are scaled by.
    """

    numbers_of_training_attempts: List[int]
    """
    How many attempts of the new task each model learns from.
    """

    number_of_repetitions: int
    """
    How often every fit is repeated with another subset of attempts and another
    template.
    """

    settings: LearningSettings
    """
    How the earlier task, the progressive column and fixed scratch are learned.
    """

    scratch_candidates: List[LearningSettings]
    """
    The settings tuned scratch chooses from.
    """

    methods: List[ComparedMethod]
    """
    The methods compared.
    """

    seed: int = 0
    """
    Seeds every repetition's template and subsets of attempts.
    """

    def template_seed(self, repetition: int) -> int:
        """
        :param repetition: The number of a repetition.
        :return: The seed of the templates of that repetition.
        """
        return int(
            np.random.default_rng([self.seed, repetition]).integers(
                np.iinfo(np.uint32).max
            )
        )

    def subset_order(
        self, *, task_number: int, repetition: int, number_of_attempts: int
    ) -> np.ndarray:
        """
        :param task_number: The number of the new task.
        :param repetition: The number of a repetition.
        :param number_of_attempts: The size of the task's training pool.
        :return: The order in which the pool's attempts enter the training subsets, so
            every smaller subset lies inside every larger one.
        """
        return np.random.default_rng([self.seed, task_number, repetition]).permutation(
            number_of_attempts
        )


# %% results


@dataclass(kw_only=True)
class FitResult:
    """
    One model of a new task, learned by one method from some of its attempts, scored on
    its probe set.
    """

    method: ComparedMethod
    """
    The method that learned the model.
    """

    change: TaskChange | None
    """
    What differs between the earlier and the new task; none for a method that does not
    read the earlier task.
    """

    earlier_task_number: int | None
    """
    The task learned before; none for a method that does not read it.
    """

    new_task_number: int
    """
    The task the model is of.
    """

    number_of_training_attempts: int
    """
    How many attempts of the new task the model learned from.
    """

    repetition: int
    """
    The repetition the fit belongs to.
    """

    skill: float
    """
    The skill on the probe set.
    """

    average_log_likelihood_of_grasps: float
    """
    The mean log-density of the probe grasps.
    """

    average_log_likelihood_of_outcomes: float
    """
    The mean log-probability of the probe outcomes given their grasps.
    """

    fit_seconds: float
    """
    How long learning the model took, on the clock.
    """

    probe_attempt_losses: List[float]
    """
    The clipped log loss of each probe attempt, in the order of the probe set.
    """


@dataclass(kw_only=True)
class ProbeBaseline:
    """
    The constant prediction every model of a task is scored against.
    """

    task_number: int
    """
    The number of the task.
    """

    base_lift_rate: float
    """
    The lift rate of the task's training pool.
    """

    base_attempt_losses: List[float]
    """
    The clipped log loss of each probe attempt under the constant prediction, in the
    order of the probe set.
    """


class GridFile(StrEnum):
    """
    The files a grid writes into its output directory.
    """

    DEFINITION = "definition.json"
    """
    The grid's definition.
    """

    FITS = "fits.jsonl"
    """
    One :class:`FitResult` per line.
    """

    PROBES = "probes.jsonl"
    """
    One :class:`ProbeBaseline` per line.
    """


# %% the work of one new task in one repetition


@dataclass(kw_only=True)
class PairedEarlierTask:
    """
    A task paired with the new task as the one learned before it.
    """

    pair: TaskPair
    """
    The pair of the earlier and the new task.
    """

    attempts: ScaledAttempts
    """
    The attempts of the earlier task.
    """


@dataclass(kw_only=True)
class GridUnit:
    """
    Every fit of one new task in one repetition: each method, each number of training
    attempts, after each earlier task paired with it.
    """

    definition: GridDefinition
    """
    The grid the unit belongs to.
    """

    new_task_number: int
    """
    The number of the new task.
    """

    new_attempts: ScaledAttempts
    """
    The attempts of the new task.
    """

    earlier_tasks: List[PairedEarlierTask]
    """
    The tasks learned before the new one, at least one.
    """

    repetition: int
    """
    The repetition the unit belongs to.
    """

    def run(self) -> List[FitResult]:
        """
        :return: The result of every fit of the unit.
        :raises TooFewTrainingAttemptsError: If a number of training attempts exceeds
            the new task's training pool.
        """
        pool = self.new_attempts.training_pool()
        largest = max(self.definition.numbers_of_training_attempts)
        if largest > len(pool.lifted):
            raise TooFewTrainingAttemptsError(
                task_number=self.new_task_number,
                number_of_training_attempts=largest,
                pool_size=len(pool.lifted),
            )
        order = self.definition.subset_order(
            task_number=self.new_task_number,
            repetition=self.repetition,
            number_of_attempts=len(pool.lifted),
        )
        seed = self.definition.template_seed(self.repetition)
        methods = {
            compared_method: compared_method.transfer_method(self.definition)
            for compared_method in self.definition.methods
        }
        learned_earlier_tasks = []
        if any(method.reads_earlier_task for method in methods.values()):
            variables = GraspOutcomeVariables()
            learned_earlier_tasks = [
                (
                    paired.pair,
                    EarlierTask.learned(
                        attempts=paired.attempts.training_pool(),
                        settings=self.definition.settings,
                        variables=variables,
                        seed=seed,
                    ),
                )
                for paired in self.earlier_tasks
            ]
        results = []
        for compared_method, method in methods.items():
            pairings = (
                learned_earlier_tasks if method.reads_earlier_task else [(None, None)]
            )
            for pair, earlier in pairings:
                for (
                    number_of_training_attempts
                ) in self.definition.numbers_of_training_attempts:
                    training = pool.rows(order[:number_of_training_attempts])
                    started = time.perf_counter()
                    model = method.model(training=training, earlier=earlier, seed=seed)
                    fit_seconds = time.perf_counter() - started
                    results.append(
                        self.fit_result(
                            compared_method=compared_method,
                            pair=pair,
                            number_of_training_attempts=number_of_training_attempts,
                            likelihoods=model.outcome_likelihoods(
                                self.new_attempts.probe_set().grasps
                            ),
                            fit_seconds=fit_seconds,
                        )
                    )
        return results

    def fit_result(
        self,
        *,
        compared_method: ComparedMethod,
        pair: TaskPair | None,
        number_of_training_attempts: int,
        likelihoods: OutcomeLikelihoods,
        fit_seconds: float,
    ) -> FitResult:
        """
        :param compared_method: The method that learned the model.
        :param pair: The pair of earlier and new task; none for a method that does not
            read the earlier task.
        :param number_of_training_attempts: How many attempts the model learned from.
        :param likelihoods: What the model gives the probe grasps.
        :param fit_seconds: How long learning took.
        :return: The model's result on the new task's probe set.
        """
        score = ProbeScore(
            likelihoods=likelihoods,
            lifted=self.new_attempts.probe_set().lifted,
            base_lift_rate=self.new_attempts.training_pool().lift_rate,
        )
        return FitResult(
            method=compared_method,
            change=None if pair is None else pair.change,
            earlier_task_number=None if pair is None else pair.earlier.task_number,
            new_task_number=self.new_task_number,
            number_of_training_attempts=number_of_training_attempts,
            repetition=self.repetition,
            skill=score.skill,
            average_log_likelihood_of_grasps=score.average_log_likelihood_of_grasps,
            average_log_likelihood_of_outcomes=score.average_log_likelihood_of_outcomes,
            fit_seconds=fit_seconds,
            probe_attempt_losses=score.attempt_losses.tolist(),
        )


# %% the grid


@dataclass(kw_only=True)
class TransferGrid:
    """
    A grid of fits over the collected attempts of its tasks.
    """

    definition: GridDefinition
    """
    What the grid runs.
    """

    directory: CollectionDirectory
    """
    Where the tasks' collection runs wrote their files.
    """

    output_directory: Path
    """
    Where the grid's files go.
    """

    def scaled_tasks(self) -> dict[int, ScaledAttempts]:
        """
        :return: The attempts of every task of the grid by task number, scaled with the
            angle ranges of the scaling tasks' statements.
        """
        scaling = GraspScaling.angles_shared_by(
            attempt.grasp_ranges
            for task_number in self.definition.scaling_task_numbers
            for attempt in TaskAttempts.read(
                directory=self.directory, task_number=task_number
            ).attempts
        )
        return {
            task_number: TaskAttempts.read(
                directory=self.directory, task_number=task_number
            ).scaled(scaling)
            for task_number in self.definition.task_numbers
        }

    def units(self, scaled_tasks: dict[int, ScaledAttempts]) -> List[GridUnit]:
        """
        :param scaled_tasks: The attempts of every task of the grid by task number.
        :return: One unit per new task that has an earlier task, and per repetition.
        """
        identities = [
            TaskIdentity.of_task(directory=self.directory, task_number=task_number)
            for task_number in self.definition.task_numbers
        ]
        pairs = TaskPair.ordered_pairs(identities)
        units = []
        for new in identities:
            earlier_tasks = [
                PairedEarlierTask(
                    pair=pair, attempts=scaled_tasks[pair.earlier.task_number]
                )
                for pair in pairs
                if pair.new == new
            ]
            if not earlier_tasks:
                continue
            units.extend(
                GridUnit(
                    definition=self.definition,
                    new_task_number=new.task_number,
                    new_attempts=scaled_tasks[new.task_number],
                    earlier_tasks=earlier_tasks,
                    repetition=repetition,
                )
                for repetition in range(self.definition.number_of_repetitions)
            )
        return units

    def prepare(self) -> List[GridUnit]:
        """
        Write the definition and the probe baselines into :attr:`output_directory`.

        :return: The units of the grid.
        :raises FileExistsError: If the output directory exists already.
        """
        self.output_directory.mkdir(parents=True, exist_ok=False)
        (self.output_directory / GridFile.DEFINITION).write_text(
            json.dumps(to_json(self.definition), indent=2) + "\n"
        )
        scaled_tasks = self.scaled_tasks()
        with (self.output_directory / GridFile.PROBES).open("w") as probes:
            for task_number, attempts in scaled_tasks.items():
                probe = attempts.probe_set()
                baseline = ProbeBaseline(
                    task_number=task_number,
                    base_lift_rate=attempts.training_pool().lift_rate,
                    base_attempt_losses=OutcomeLikelihoods.clipped_log_losses_of(
                        np.full(len(probe.lifted), attempts.training_pool().lift_rate),
                        probe.lifted,
                    ).tolist(),
                )
                probes.write(json.dumps(to_json(baseline)) + "\n")
        return self.units(scaled_tasks)


# %% running grids


@dataclass(kw_only=True)
class GridRunner:
    """
    Runs the units of several grids in one pool of processes and writes each grid's
    fits into its own output directory.
    """

    number_of_workers: int = 1
    """
    How many processes run units at once.
    """

    def run(self, grids: List[TransferGrid]) -> int:
        """
        :param grids: The grids, each with an output directory that does not exist yet.
        :return: The number of fits written.
        """
        number_of_fits = 0
        with ProcessPoolExecutor(
            max_workers=self.number_of_workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
            fits_paths = {}
            for grid in grids:
                for unit in grid.prepare():
                    fits_paths[executor.submit(unit.run)] = (
                        grid.output_directory / GridFile.FITS
                    )
            for future in as_completed(fits_paths):
                with fits_paths[future].open("a") as fits:
                    for result in future.result():
                        fits.write(json.dumps(to_json(result)) + "\n")
                        number_of_fits += 1
        return number_of_fits

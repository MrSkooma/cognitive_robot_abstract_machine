"""
Skill curves and gains of one method over another, read from the files of a grid, with
bootstrap intervals over repetitions and probe attempts together, and the Holm
correction for testing several gains at once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from typing_extensions import List

from experiments.progressive_grasp_learning.transfer_grid import (
    ComparedMethod,
    FitResult,
    GridDefinition,
    GridFile,
    ProbeBaseline,
    TaskChange,
)
from krrood.adapters.json_serializer import from_json

# %% bootstrap resamples


@dataclass(kw_only=True)
class Resample:
    """
    One draw of the bootstrap: how often each repetition and each probe attempt of each
    task is drawn.
    """

    repetition_counts: NDArray[np.float64]
    """
    How often each repetition is drawn, by repetition number.
    """

    probe_attempt_counts: dict[int, NDArray[np.float64]]
    """
    How often each probe attempt is drawn, by task number and then in the order of the
    task's probe set.
    """

    @classmethod
    def everything_once(
        cls, *, number_of_repetitions: int, probe_sizes: dict[int, int]
    ) -> Resample:
        """
        :param number_of_repetitions: The number of repetitions of the grid.
        :param probe_sizes: The number of probe attempts by task number.
        :return: The draw that holds every repetition and probe attempt once, which
            gives the estimate itself.
        """
        return cls(
            repetition_counts=np.ones(number_of_repetitions),
            probe_attempt_counts={
                task_number: np.ones(size) for task_number, size in probe_sizes.items()
            },
        )

    @classmethod
    def drawn(
        cls,
        *,
        number_of_repetitions: int,
        probe_sizes: dict[int, int],
        generator: np.random.Generator,
    ) -> Resample:
        """
        :param number_of_repetitions: The number of repetitions of the grid.
        :param probe_sizes: The number of probe attempts by task number.
        :param generator: Draws the resample.
        :return: A draw with replacement of as many repetitions and, for each task, as
            many probe attempts as there are.
        """
        return cls(
            repetition_counts=generator.multinomial(
                number_of_repetitions,
                np.full(number_of_repetitions, 1 / number_of_repetitions),
            ).astype(np.float64),
            probe_attempt_counts={
                task_number: generator.multinomial(
                    size, np.full(size, 1 / size)
                ).astype(np.float64)
                for task_number, size in probe_sizes.items()
            },
        )


# %% paired fits


@dataclass(kw_only=True)
class PairedTaskFits:
    """
    The fits of one new task paired with the fits they are compared with, one row per
    pairing.
    """

    task_number: int
    """
    The number of the new task.
    """

    base_attempt_losses: NDArray[np.float64]
    """
    The loss of each probe attempt under the constant prediction skill is measured
    against.
    """

    repetitions: NDArray[np.int64]
    """
    The repetition of each row.
    """

    attempt_losses: NDArray[np.float64]
    """
    The loss of each probe attempt (column) under each row's fit.
    """

    reference_attempt_losses: NDArray[np.float64]
    """
    The loss of each probe attempt (column) under the fit each row is compared with.
    """

    def skill_gains(
        self, probe_attempt_counts: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        :param probe_attempt_counts: How often each probe attempt is drawn.
        :return: Each row's skill minus the skill of the fit it is compared with, on the
            drawn probe attempts.
        """
        return (
            self.reference_attempt_losses @ probe_attempt_counts
            - self.attempt_losses @ probe_attempt_counts
        ) / (self.base_attempt_losses @ probe_attempt_counts)


@dataclass(kw_only=True)
class BootstrapEstimate:
    """
    A mean skill gain with its 95 % bootstrap interval and two-sided p-value.
    """

    estimate: float
    """
    The gain on every repetition and probe attempt.
    """

    lower: float
    """
    The 2.5th percentile of the resampled gains.
    """

    upper: float
    """
    The 97.5th percentile of the resampled gains.
    """

    p_value: float
    """
    The two-sided bootstrap p-value of a gain of zero.
    """


@dataclass(kw_only=True)
class SkillGain:
    """
    The skill of fits minus the skill of the fits they are paired with, averaged over
    every pairing with equal weight.
    """

    paired_task_fits: List[PairedTaskFits]
    """
    The pairings, grouped by new task.
    """

    number_of_repetitions: int
    """
    The number of repetitions of the grid.
    """

    def value(self, resample: Resample) -> float:
        """
        :param resample: The repetitions and probe attempts drawn.
        :return: The mean gain over the drawn repetitions and probe attempts.
        """
        gains = np.concatenate(
            [
                task_fits.skill_gains(
                    resample.probe_attempt_counts[task_fits.task_number]
                )
                for task_fits in self.paired_task_fits
            ]
        )
        weights = np.concatenate(
            [
                resample.repetition_counts[task_fits.repetitions]
                for task_fits in self.paired_task_fits
            ]
        )
        return float(np.sum(gains * weights) / np.sum(weights))

    def bootstrapped(
        self, *, generator: np.random.Generator, number_of_resamples: int = 2000
    ) -> BootstrapEstimate:
        """
        :param generator: Draws the resamples.
        :param number_of_resamples: How many resamples are drawn.
        :return: The gain with its interval and p-value.
        """
        probe_sizes = {
            task_fits.task_number: len(task_fits.base_attempt_losses)
            for task_fits in self.paired_task_fits
        }
        resampled = np.array(
            [
                self.value(
                    Resample.drawn(
                        number_of_repetitions=self.number_of_repetitions,
                        probe_sizes=probe_sizes,
                        generator=generator,
                    )
                )
                for _ in range(number_of_resamples)
            ]
        )
        lower, upper = np.percentile(resampled, [2.5, 97.5])
        one_sided = min(np.sum(resampled <= 0), np.sum(resampled >= 0))
        return BootstrapEstimate(
            estimate=self.value(
                Resample.everything_once(
                    number_of_repetitions=self.number_of_repetitions,
                    probe_sizes=probe_sizes,
                )
            ),
            lower=float(lower),
            upper=float(upper),
            p_value=float(min(1.0, 2 * (one_sided + 1) / (number_of_resamples + 1))),
        )


# %% the files of a grid


@dataclass(kw_only=True)
class GridResults:
    """
    What a grid wrote: its definition, every fit, and the constant prediction of each
    task.
    """

    definition: GridDefinition
    """
    The grid's definition.
    """

    fits: List[FitResult]
    """
    Every fit of the grid.
    """

    baselines: dict[int, ProbeBaseline]
    """
    The constant prediction of each task, by task number.
    """

    @classmethod
    def read(cls, output_directory: Path) -> GridResults:
        """
        :param output_directory: The directory a grid wrote its files into.
        :return: The grid's results.
        """
        baselines = [
            from_json(json.loads(line))
            for line in (output_directory / GridFile.PROBES).read_text().splitlines()
        ]
        return cls(
            definition=from_json(
                json.loads((output_directory / GridFile.DEFINITION).read_text())
            ),
            fits=[
                from_json(json.loads(line))
                for line in (output_directory / GridFile.FITS).read_text().splitlines()
            ],
            baselines={baseline.task_number: baseline for baseline in baselines},
        )

    def skill(
        self,
        *,
        method: ComparedMethod,
        number_of_training_attempts: int,
        change: TaskChange | None = None,
    ) -> SkillGain:
        """
        :param method: The method.
        :param number_of_training_attempts: How many attempts the fits learned from.
        :param change: Keeps only the pairs with this change; none keeps every fit.
        :return: The method's skill, as its gain over the constant prediction.
        """
        fits = [
            fit
            for fit in self.fits
            if fit.method == method
            and fit.number_of_training_attempts == number_of_training_attempts
            and (change is None or fit.change == change)
        ]
        return SkillGain(
            paired_task_fits=[
                PairedTaskFits(
                    task_number=task_number,
                    base_attempt_losses=np.array(
                        self.baselines[task_number].base_attempt_losses
                    ),
                    repetitions=np.array([fit.repetition for fit in task_fits]),
                    attempt_losses=np.array(
                        [fit.probe_attempt_losses for fit in task_fits]
                    ),
                    reference_attempt_losses=np.tile(
                        self.baselines[task_number].base_attempt_losses,
                        (len(task_fits), 1),
                    ),
                )
                for task_number, task_fits in self.fits_by_new_task(fits).items()
            ],
            number_of_repetitions=self.definition.number_of_repetitions,
        )

    def gain(
        self,
        *,
        method: ComparedMethod,
        reference_method: ComparedMethod,
        change: TaskChange,
        numbers_of_training_attempts: List[int],
    ) -> SkillGain:
        """
        :param method: The method whose gain is measured; it reads the earlier task.
        :param reference_method: The method it is compared with, fit by fit.
        :param change: Keeps only the pairs with this change.
        :param numbers_of_training_attempts: The numbers of training attempts averaged
            over.
        :return: The method's skill minus the reference method's, paired by task pair,
            number of training attempts and repetition.
        """
        reference_reads_earlier_task = reference_method.transfer_method(
            self.definition
        ).reads_earlier_task
        references = {
            self.pairing_key(fit, earlier_task_number=fit.earlier_task_number): fit
            for fit in self.fits
            if fit.method == reference_method
        }
        fits = [
            fit
            for fit in self.fits
            if fit.method == method
            and fit.change == change
            and fit.number_of_training_attempts in numbers_of_training_attempts
        ]
        paired_task_fits = []
        for task_number, task_fits in self.fits_by_new_task(fits).items():
            paired_references = [
                references[
                    self.pairing_key(
                        fit,
                        earlier_task_number=(
                            fit.earlier_task_number
                            if reference_reads_earlier_task
                            else None
                        ),
                    )
                ]
                for fit in task_fits
            ]
            paired_task_fits.append(
                PairedTaskFits(
                    task_number=task_number,
                    base_attempt_losses=np.array(
                        self.baselines[task_number].base_attempt_losses
                    ),
                    repetitions=np.array([fit.repetition for fit in task_fits]),
                    attempt_losses=np.array(
                        [fit.probe_attempt_losses for fit in task_fits]
                    ),
                    reference_attempt_losses=np.array(
                        [fit.probe_attempt_losses for fit in paired_references]
                    ),
                )
            )
        return SkillGain(
            paired_task_fits=paired_task_fits,
            number_of_repetitions=self.definition.number_of_repetitions,
        )

    @staticmethod
    def pairing_key(
        fit: FitResult, *, earlier_task_number: int | None
    ) -> tuple[int | None, int, int, int]:
        """
        :param fit: A fit.
        :param earlier_task_number: The earlier task the fit is paired under; none for a
            fit of a method that does not read it.
        :return: What a fit and the fit it is compared with share.
        """
        return (
            earlier_task_number,
            fit.new_task_number,
            fit.number_of_training_attempts,
            fit.repetition,
        )

    @staticmethod
    def fits_by_new_task(fits: List[FitResult]) -> dict[int, List[FitResult]]:
        """
        :param fits: Fits of any tasks.
        :return: The fits grouped by new task number.
        """
        grouped = {}
        for fit in fits:
            grouped.setdefault(fit.new_task_number, []).append(fit)
        return grouped


# %% several tests at once


@dataclass(kw_only=True)
class HolmCorrection:
    """
    Holm's step-down correction, which keeps the chance of any false rejection among
    several tests at most :attr:`family_error_rate`.
    """

    family_error_rate: float = 0.05
    """
    The highest chance of rejecting any true null hypothesis of the family.
    """

    def rejected(self, p_values: List[float]) -> List[bool]:
        """
        :param p_values: The p-values of the tests.
        :return: For each test, in the given order, whether its null hypothesis is
            rejected.
        """
        number_of_tests = len(p_values)
        rejected = [False] * number_of_tests
        for rank, index in enumerate(np.argsort(p_values)):
            if p_values[index] > self.family_error_rate / (number_of_tests - rank):
                break
            rejected[index] = True
        return rejected

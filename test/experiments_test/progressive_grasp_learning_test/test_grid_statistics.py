"""
Skill curves, gains between methods with bootstrap intervals, and the Holm correction.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from .test_transfer_grid import definition, directory

from experiments.progressive_grasp_learning import analyse_grid
from experiments.progressive_grasp_learning.exceptions import NoMatchingFitsError
from experiments.progressive_grasp_learning.grid_statistics import (
    GridAnalysis,
    GridResults,
    HolmCorrection,
    Resample,
    TransferEndpoints,
)
from experiments.progressive_grasp_learning.transfer_grid import (
    ComparedMethod,
    GridRunner,
    TaskChange,
    TransferGrid,
)
from krrood.adapters.json_serializer import from_json

# %% fixtures


@pytest.fixture
def results(definition, directory, tmp_path) -> GridResults:
    """
    The results of a small grid over three box tasks, with every method.
    """
    output_directory = tmp_path / "grid"
    GridRunner(number_of_workers=2).run(
        [
            TransferGrid(
                definition=definition,
                directory=directory,
                output_directory=output_directory,
            )
        ]
    )
    return GridResults.read(output_directory)


def everything_once(results: GridResults) -> Resample:
    """
    :return: The resample that holds every repetition and probe attempt of the grid once.
    """
    return Resample.everything_once(
        number_of_repetitions=results.definition.number_of_repetitions,
        probe_sizes={
            task_number: len(baseline.base_attempt_losses)
            for task_number, baseline in results.baselines.items()
        },
    )


# %% estimates


def test_the_skill_of_a_method_is_the_mean_skill_of_its_fits(results):
    number_of_training_attempts = results.definition.numbers_of_training_attempts[0]

    skill = results.skill(
        method=ComparedMethod.FIXED_SCRATCH,
        number_of_training_attempts=number_of_training_attempts,
    ).value(everything_once(results))

    assert skill == pytest.approx(
        np.mean(
            [
                fit.skill
                for fit in results.fits
                if fit.method == ComparedMethod.FIXED_SCRATCH
                and fit.number_of_training_attempts == number_of_training_attempts
            ]
        )
    )


def test_a_gain_is_the_mean_skill_difference_of_paired_fits(results):
    numbers_of_training_attempts = results.definition.numbers_of_training_attempts

    gain = results.gain(
        method=ComparedMethod.PROGRESSIVE_COLUMN,
        reference_method=ComparedMethod.TUNED_SCRATCH,
        change=TaskChange.OBJECT,
        numbers_of_training_attempts=numbers_of_training_attempts,
    ).value(everything_once(results))

    scratch_skills = {
        (
            fit.new_task_number,
            fit.number_of_training_attempts,
            fit.repetition,
        ): fit.skill
        for fit in results.fits
        if fit.method == ComparedMethod.TUNED_SCRATCH
    }
    assert gain == pytest.approx(
        np.mean(
            [
                fit.skill
                - scratch_skills[
                    (
                        fit.new_task_number,
                        fit.number_of_training_attempts,
                        fit.repetition,
                    )
                ]
                for fit in results.fits
                if fit.method == ComparedMethod.PROGRESSIVE_COLUMN
                and fit.change == TaskChange.OBJECT
            ]
        )
    )


def test_a_reference_that_reads_the_earlier_task_is_paired_by_task_pair(results):
    numbers_of_training_attempts = results.definition.numbers_of_training_attempts

    gain = results.gain(
        method=ComparedMethod.PROGRESSIVE_COLUMN,
        reference_method=ComparedMethod.EARLIER_COLUMN_UNCHANGED,
        change=TaskChange.ROBOT,
        numbers_of_training_attempts=numbers_of_training_attempts,
    ).value(everything_once(results))

    unchanged_skills = {
        (
            fit.earlier_task_number,
            fit.new_task_number,
            fit.number_of_training_attempts,
            fit.repetition,
        ): fit.skill
        for fit in results.fits
        if fit.method == ComparedMethod.EARLIER_COLUMN_UNCHANGED
    }
    assert gain == pytest.approx(
        np.mean(
            [
                fit.skill
                - unchanged_skills[
                    (
                        fit.earlier_task_number,
                        fit.new_task_number,
                        fit.number_of_training_attempts,
                        fit.repetition,
                    )
                ]
                for fit in results.fits
                if fit.method == ComparedMethod.PROGRESSIVE_COLUMN
                and fit.change == TaskChange.ROBOT
            ]
        )
    )


def test_a_method_has_no_gain_over_itself_in_any_resample(results):
    gain = results.gain(
        method=ComparedMethod.PROGRESSIVE_COLUMN,
        reference_method=ComparedMethod.PROGRESSIVE_COLUMN,
        change=TaskChange.OBJECT,
        numbers_of_training_attempts=results.definition.numbers_of_training_attempts,
    )

    estimate = gain.bootstrapped(
        generator=np.random.default_rng(0), number_of_resamples=50
    )

    assert (estimate.estimate, estimate.lower, estimate.upper, estimate.p_value) == (
        0.0,
        0.0,
        0.0,
        1.0,
    )


def test_asking_for_a_number_of_training_attempts_the_grid_lacks_is_refused(results):
    missing = max(results.definition.numbers_of_training_attempts) + 1

    with pytest.raises(NoMatchingFitsError):
        results.gain(
            method=ComparedMethod.PROGRESSIVE_COLUMN,
            reference_method=ComparedMethod.TUNED_SCRATCH,
            change=TaskChange.OBJECT,
            numbers_of_training_attempts=[missing],
        )


# %% several tests at once


@pytest.mark.parametrize(
    "p_values, rejected",
    [
        ([0.04, 0.01], [True, True]),
        ([0.03, 0.04], [False, False]),
        ([0.02, 0.06], [True, False]),
        ([0.06, 0.02], [False, True]),
    ],
)
def test_holm_compares_the_smallest_p_value_with_the_strictest_threshold(
    p_values, rejected
):
    assert HolmCorrection(family_error_rate=0.05).rejected(p_values) == rejected


# %% the primary endpoints


def test_each_endpoint_is_the_gain_over_the_pairs_of_its_change(results):
    endpoints = TransferEndpoints(
        method=ComparedMethod.PROGRESSIVE_COLUMN,
        reference_method=ComparedMethod.TUNED_SCRATCH,
        numbers_of_training_attempts=results.definition.numbers_of_training_attempts,
        number_of_resamples=50,
    )

    evaluated = endpoints.evaluated(
        grid_results=results, generator=np.random.default_rng(0)
    )

    assert [(endpoint.change, endpoint.gain.estimate) for endpoint in evaluated] == [
        (
            change,
            pytest.approx(
                results.gain(
                    method=endpoints.method,
                    reference_method=endpoints.reference_method,
                    change=change,
                    numbers_of_training_attempts=endpoints.numbers_of_training_attempts,
                ).value(everything_once(results))
            ),
        )
        for change in TaskChange
    ]


def test_the_endpoints_are_decided_together_by_holms_correction(results):
    endpoints = TransferEndpoints(
        method=ComparedMethod.PROGRESSIVE_COLUMN,
        reference_method=ComparedMethod.TUNED_SCRATCH,
        numbers_of_training_attempts=results.definition.numbers_of_training_attempts,
        number_of_resamples=50,
    )

    evaluated = endpoints.evaluated(
        grid_results=results, generator=np.random.default_rng(0)
    )

    assert [
        endpoint.zero_gain_rejected for endpoint in evaluated
    ] == endpoints.holm_correction.rejected(
        [endpoint.gain.p_value for endpoint in evaluated]
    )


# %% the command line


def test_the_analysis_the_command_writes_reads_back(definition, directory, tmp_path):
    grid_directory = tmp_path / "grid"
    GridRunner().run(
        [
            TransferGrid(
                definition=definition,
                directory=directory,
                output_directory=grid_directory,
            )
        ]
    )
    analysis_path = tmp_path / "analysis.json"

    subprocess.run(
        [
            sys.executable,
            "-m",
            analyse_grid.__name__,
            "--grid-directory",
            str(grid_directory),
            "--analysis",
            str(analysis_path),
            "--resamples",
            "20",
            "--numbers-of-training-attempts",
            *(str(number) for number in definition.numbers_of_training_attempts),
        ],
        check=True,
        capture_output=True,
    )

    analysis = from_json(json.loads(analysis_path.read_text()))
    assert isinstance(analysis, GridAnalysis)
    assert [endpoint.change for endpoint in analysis.endpoints] == list(TaskChange)

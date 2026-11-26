"""
Analysing the files of a finished grid from the command line: the primary endpoints and
every method's skill curve, printed and written as JSON.

Run it with ``python -m experiments.progressive_grasp_learning.analyse_grid``; see
``--help``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.progressive_grasp_learning.grid_statistics import (
    GridAnalysis,
    GridResults,
    TransferEndpoints,
)
from experiments.progressive_grasp_learning.transfer_grid import ComparedMethod
from krrood.adapters.json_serializer import to_json

# %% the command line


def main() -> None:
    """
    Analyse a grid, print its endpoints and skill curves, and write the analysis.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-directory",
        type=Path,
        required=True,
        help="the directory a grid wrote its files into",
    )
    parser.add_argument(
        "--analysis",
        type=Path,
        required=True,
        help="the JSON file the analysis is written to",
    )
    parser.add_argument(
        "--numbers-of-training-attempts",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="the numbers of training attempts each endpoint averages over",
    )
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    analysis = GridAnalysis.of_grid(
        grid_results=GridResults.read(arguments.grid_directory),
        endpoints=TransferEndpoints(
            method=ComparedMethod.PROGRESSIVE_COLUMN,
            reference_method=ComparedMethod.TUNED_SCRATCH,
            numbers_of_training_attempts=arguments.numbers_of_training_attempts,
            number_of_resamples=arguments.resamples,
        ),
        generator=np.random.default_rng(arguments.seed),
    )
    arguments.analysis.write_text(json.dumps(to_json(analysis), indent=2) + "\n")
    for endpoint in analysis.endpoints:
        gain = endpoint.gain
        print(
            f"endpoint {endpoint.change.value}: {gain.estimate:+.3f} "
            f"[{gain.lower:+.3f}, {gain.upper:+.3f}], p = {gain.p_value:.4f}, "
            f"zero gain rejected: {endpoint.zero_gain_rejected}"
        )
    for point in analysis.skill_curves:
        change = "all" if point.change is None else point.change.value
        skill = point.skill
        print(
            f"{point.method.value:<36} {change:<7} k={point.number_of_training_attempts:<4} "
            f"{skill.estimate:+.3f} [{skill.lower:+.3f}, {skill.upper:+.3f}]"
        )


if __name__ == "__main__":
    main()

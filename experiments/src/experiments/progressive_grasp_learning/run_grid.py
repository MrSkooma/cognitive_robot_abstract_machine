"""
Running the grids definition files describe, from the command line.

Run it with ``python -m experiments.progressive_grasp_learning.run_grid``; see
``--help``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.grasp_learning.collection_files import CollectionDirectory
from experiments.progressive_grasp_learning.transfer_grid import (
    GridRunner,
    TransferGrid,
)
from krrood.adapters.json_serializer import from_json

# %% the command line


def main() -> None:
    """
    Run the grids the definition files describe and report how many fits they wrote.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--definitions",
        type=Path,
        nargs="+",
        required=True,
        help="JSON files each holding a grid's definition",
    )
    parser.add_argument(
        "--collection-directory",
        type=Path,
        required=True,
        help="where the tasks' collection runs wrote their files",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="the directory each grid's files go to, in a new directory named after "
        "its definition file",
    )
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args()
    grids = [
        TransferGrid(
            definition=from_json(json.loads(definition_path.read_text())),
            directory=CollectionDirectory(path=arguments.collection_directory),
            output_directory=arguments.output_directory / definition_path.stem,
        )
        for definition_path in arguments.definitions
    ]
    number_of_fits = GridRunner(number_of_workers=arguments.workers).run(grids)
    print(f"{number_of_fits} fits written to {arguments.output_directory}")


if __name__ == "__main__":
    main()

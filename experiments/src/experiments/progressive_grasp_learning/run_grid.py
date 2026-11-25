"""
Running the grid a definition file describes, from the command line.

Run it with ``python -m experiments.progressive_grasp_learning.run_grid``; see
``--help``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.grasp_learning.collection_files import CollectionDirectory
from experiments.progressive_grasp_learning.transfer_grid import TransferGrid
from krrood.adapters.json_serializer import from_json

# %% the command line


def main() -> None:
    """
    Run the grid a definition file describes and report how many fits it wrote.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--definition",
        type=Path,
        required=True,
        help="a JSON file holding the grid's definition",
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
        help="a new directory the grid's files go to",
    )
    parser.add_argument("--workers", type=int, default=1)
    arguments = parser.parse_args()
    grid = TransferGrid(
        definition=from_json(json.loads(arguments.definition.read_text())),
        directory=CollectionDirectory(path=arguments.collection_directory),
    )
    number_of_fits = grid.run(
        output_directory=arguments.output_directory,
        number_of_workers=arguments.workers,
    )
    print(f"{number_of_fits} fits written to {arguments.output_directory}")


if __name__ == "__main__":
    main()

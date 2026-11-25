"""
The files a collection run writes and the manifest describing it, read without the
simulation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from typing_extensions import List

from krrood.adapters.json_serializer import from_json, to_json

# %% the files of a run


class CollectionFileSuffix(StrEnum):
    """
    The endings of the files a collection run writes, after the run's name.
    """

    DATABASE = ".sqlite"
    """
    The SQLite database the attempts are stored in while they are tried.
    """

    ATTEMPTS = ".jsonl"
    """
    The attempts, exported as JSON lines.
    """

    MANIFEST = ".manifest.json"
    """
    The manifest describing the run.
    """


# %% the manifest


@dataclass(kw_only=True)
class CollectionManifest:
    """
    What a collection run did, written beside its attempts.
    """

    robot: str
    """
    The name of the robot's class.
    """

    object_name: str
    """
    The name of the object's body.
    """

    task_number: int
    """
    The number of the task the run belongs to.
    """

    run_number: int
    """
    The number of the run within its task.
    """

    seed: int
    """
    The seed of where the object stood and of which grasps were drawn.
    """

    number_of_attempts: int
    """
    How many attempts the run stored.
    """

    number_of_lifting_attempts: int
    """
    How many of them lifted the object.
    """

    number_of_grasps_missing_the_surface: int
    """
    How many drawn grasps named a point the grasped part's surface does not reach and
    were therefore not tried.
    """

    stored_parts_directory: str | None
    """
    The directory of parts located before that answered the object's part queries, if
    one was given.
    """

    stored_part_files: list[str]
    """
    The files that directory held when the run started.
    """

    commit: str
    """
    The commit of the code that collected the run.
    """

    started: str
    """
    When the run started, in ISO 8601 and UTC.
    """

    wall_seconds: float
    """
    How long the run took on the clock, in seconds, building the scene excluded.
    """

    def write(self, *, path: Path) -> None:
        """
        Write the manifest as JSON, replacing what the file held.

        :param path: The file the manifest goes to.
        """
        path.write_text(json.dumps(to_json(self), indent=2) + "\n")

    @classmethod
    def read(cls, *, path: Path) -> CollectionManifest:
        """
        :param path: A file a manifest was written to.
        :return: The manifest.
        """
        return from_json(json.loads(path.read_text()))


# %% a directory of runs


@dataclass(kw_only=True)
class CollectedRun:
    """
    A finished collection run: its manifest and the file of its attempts.
    """

    manifest: CollectionManifest
    """
    What the run did.
    """

    attempts_path: Path
    """
    The file of the run's exported attempts.
    """


@dataclass(kw_only=True)
class CollectionDirectory:
    """
    A directory collection runs wrote their files to.
    """

    path: Path
    """
    Where the directory is.
    """

    def runs_of_task(self, task_number: int) -> List[CollectedRun]:
        """
        :param task_number: The number of a task.
        :return: The finished runs of that task, ordered by run number.
        """
        runs = []
        for manifest_path in self.path.glob(f"*{CollectionFileSuffix.MANIFEST}"):
            manifest = CollectionManifest.read(path=manifest_path)
            if manifest.task_number != task_number:
                continue
            name = manifest_path.name.removesuffix(CollectionFileSuffix.MANIFEST)
            runs.append(
                CollectedRun(
                    manifest=manifest,
                    attempts_path=self.path / f"{name}{CollectionFileSuffix.ATTEMPTS}",
                )
            )
        return sorted(runs, key=lambda run: run.manifest.run_number)

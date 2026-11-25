"""
Collecting the attempts of one robot on one object as one numbered run: grasps drawn
uniformly from the object's statement are tried, stored in a database of the run's own,
exported to JSON lines and described by a manifest.

Run it with ``python -m experiments.grasp_learning.collection``; see ``--help``. One
process collects one run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.grasp_learning.collection_files import (
    CollectionFileSuffix,
    CollectionManifest,
)
from experiments.grasp_learning.database import GraspDatabase
from experiments.grasp_learning.exceptions import (
    CollectionRunExistsError,
    NoGraspReachesSurfaceError,
    RunNumberOutOfRangeError,
)
from experiments.grasp_learning.export import AttemptExport
from experiments.grasp_learning.exported_attempt import ExportFile
from experiments.grasp_learning.pipeline import GraspLearningPipeline
from experiments.grasp_learning.records import GraspSource
from experiments.physical_pick_up.pick_up_experiment import PickUpExperiment
from experiments.physical_pick_up.scene import PickUpScene, PickUpSceneChoice
from krrood.ormatic.utils import classproperty
from krrood.parametrization.model_registries import UniformPriorRegistry

# %% the number of a run and its seed


@dataclass(frozen=True, kw_only=True)
class NumberedRun:
    """
    A run of a task, numbered so that every run of every task has a seed of its own.
    """

    task_number: int
    """
    The number of the task the run belongs to.
    """

    run_number: int
    """
    The number of the run within its task.
    """

    def __post_init__(self):
        """
        :raises RunNumberOutOfRangeError: If the run number falls outside the block of
            seeds of its task.
        """
        if not 0 <= self.run_number < self.seeds_per_task:
            raise RunNumberOutOfRangeError(
                run_number=self.run_number, seeds_per_task=self.seeds_per_task
            )

    @classproperty
    def seeds_per_task(cls) -> int:
        """
        The size of the block of seeds each task's runs draw their seeds from.
        """
        return 1000

    @property
    def seed(self) -> int:
        """
        The seed of where the object stands and of which grasps are drawn.
        """
        return self.seeds_per_task * self.task_number + self.run_number


# %% the run


@dataclass(kw_only=True)
class CollectionRun:
    """
    One run of a robot's attempts at grasps on an object, drawn uniformly from the
    object's statement, with every file it writes named after the run.
    """

    scene: PickUpScene
    """
    The robot and the object.
    """

    numbered_run: NumberedRun
    """
    The task and run the attempts belong to.
    """

    output_directory: Path
    """
    The directory the run's files are written to.
    """

    commit: str
    """
    The commit of the code that collects the run.
    """

    number_of_attempts: int = 200
    """
    How many attempts the run stores.
    """

    stored_parts_directory: Path | None = None
    """
    The directory of parts located before that answers the object's part queries, if one
    is given.
    """

    @property
    def name(self) -> str:
        """
        The name every file of the run starts with.
        """
        return (
            f"{type(self.scene.robot).__name__.lower()}"
            f"-{self.scene.graspable.root.name.name}"
            f"-task{self.numbered_run.task_number}"
            f"-run{self.numbered_run.run_number}"
        )

    def path(self, suffix: CollectionFileSuffix) -> Path:
        """
        :param suffix: The kind of file.
        :return: Where the run writes that file.
        """
        return self.output_directory / f"{self.name}{suffix}"

    def collect(self) -> CollectionManifest:
        """
        Try the run's attempts, store them, export them and write the manifest.

        :return: The manifest written.
        :raises CollectionRunExistsError: If a file of the run exists already.
        :raises NoGraspReachesSurfaceError: If no drawn grasp reaches the grasped part's
            surface.
        """
        for suffix in CollectionFileSuffix:
            if self.path(suffix).exists():
                raise CollectionRunExistsError(path=self.path(suffix))
        self.output_directory.mkdir(parents=True, exist_ok=True)
        stored_part_files = self.stored_part_files()
        started = datetime.now(timezone.utc)
        database = GraspDatabase.connect(
            f"sqlite:///{self.path(CollectionFileSuffix.DATABASE)}"
        )
        number_of_grasps_missing_the_surface = self.attempt(database=database)
        return self.write_results(
            database=database,
            number_of_grasps_missing_the_surface=number_of_grasps_missing_the_surface,
            stored_part_files=stored_part_files,
            started=started,
        )

    def stored_part_files(self) -> list[str]:
        """
        :return: The names of the files in :attr:`stored_parts_directory`, sorted; none
            if no directory is given.
        """
        if self.stored_parts_directory is None:
            return []
        return sorted(path.name for path in self.stored_parts_directory.iterdir())

    def attempt(self, *, database: GraspDatabase) -> int:
        """
        Try grasps until :attr:`number_of_attempts` attempts are stored, drawing grasps
        again in place of those that miss the grasped part's surface.

        Numpy's global generator, which draws the grasps, is seeded with the run's seed,
        so the same run draws the same grasps again.

        :param database: Where the attempts are stored.
        :return: How many drawn grasps missed the grasped part's surface.
        :raises NoGraspReachesSurfaceError: If no grasp of one drawing reaches the
            surface.
        """
        np.random.seed(self.numbered_run.seed)
        pipeline = GraspLearningPipeline(
            experiment=PickUpExperiment(
                scene=self.scene, time_limit=GraspLearningPipeline.attempt_time_limit
            ),
            database=database,
            seed=self.numbered_run.seed,
        )
        number_stored = 0
        number_of_grasps_missing_the_surface = 0
        while number_stored < self.number_of_attempts:
            number_drawn = self.number_of_attempts - number_stored
            attempts = pipeline.attempt(
                GraspSource.STATED_REGIONS, number_drawn, UniformPriorRegistry()
            )
            if not attempts:
                raise NoGraspReachesSurfaceError(number_of_grasps=number_drawn)
            number_stored += len(attempts)
            number_of_grasps_missing_the_surface += number_drawn - len(attempts)
        return number_of_grasps_missing_the_surface

    def write_results(
        self,
        *,
        database: GraspDatabase,
        number_of_grasps_missing_the_surface: int,
        stored_part_files: list[str],
        started: datetime,
    ) -> CollectionManifest:
        """
        Export the stored attempts and write the manifest.

        :param database: Where the run's attempts are stored.
        :param number_of_grasps_missing_the_surface: How many drawn grasps were not
            tried because they missed the grasped part's surface.
        :param stored_part_files: The files the stored-parts directory held when the run
            started.
        :param started: When the run started.
        :return: The manifest written.
        """
        attempts = AttemptExport(
            scene=self.scene, database=database, commit=self.commit
        ).exported_attempts()
        ExportFile(path=self.path(CollectionFileSuffix.ATTEMPTS)).write(
            attempts=attempts
        )
        manifest = CollectionManifest(
            robot=type(self.scene.robot).__name__,
            object_name=self.scene.graspable.root.name.name,
            task_number=self.numbered_run.task_number,
            run_number=self.numbered_run.run_number,
            seed=self.numbered_run.seed,
            number_of_attempts=len(attempts),
            number_of_lifting_attempts=sum(
                attempt.result.lifted for attempt in attempts
            ),
            number_of_grasps_missing_the_surface=number_of_grasps_missing_the_surface,
            stored_parts_directory=(
                None
                if self.stored_parts_directory is None
                else str(self.stored_parts_directory)
            ),
            stored_part_files=stored_part_files,
            commit=self.commit,
            started=started.isoformat(),
            wall_seconds=(datetime.now(timezone.utc) - started).total_seconds(),
        )
        manifest.write(path=self.path(CollectionFileSuffix.MANIFEST))
        return manifest


# %% the command line


def main() -> None:
    """
    Collect one run of the chosen robot on the chosen object and report how many of its
    attempts lifted the object.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    scene_choice = PickUpSceneChoice(parser)
    scene_choice.add_arguments()
    parser.add_argument("--task-number", type=int, required=True)
    parser.add_argument("--run-number", type=int, required=True)
    parser.add_argument(
        "--attempts",
        type=int,
        default=CollectionRun.number_of_attempts,
        help="attempts the run stores",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="the directory the run's database, attempts and manifest go to",
    )
    arguments = parser.parse_args()
    run = CollectionRun(
        scene=scene_choice.scene(arguments),
        numbered_run=NumberedRun(
            task_number=arguments.task_number, run_number=arguments.run_number
        ),
        output_directory=arguments.output_directory,
        commit=AttemptExport.checked_out_commit(),
        number_of_attempts=arguments.attempts,
        stored_parts_directory=arguments.stored_parts,
    )
    manifest = run.collect()
    print(
        f"{run.name}: {manifest.number_of_lifting_attempts} of "
        f"{manifest.number_of_attempts} attempts lifted the object"
    )


if __name__ == "__main__":
    main()

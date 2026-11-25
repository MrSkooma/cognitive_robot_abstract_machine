"""
Exporting the stored attempts of one robot on one object to a file of JSON lines, which
the learning side reads without the simulation.

Run it with ``python -m experiments.grasp_learning.export``; see ``--help``.
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from typing_extensions import List

from experiments.grasp_learning.database import GraspDatabase
from experiments.grasp_learning.exceptions import GraspRangeNotAnIntervalError
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
from experiments.grasp_learning.grasp_variables import SurfaceGraspVariablePrefix
from experiments.grasp_learning.records import GraspAttempt, GraspLearningTask
from experiments.physical_pick_up.scene import PickUpScene, PickUpSceneChoice
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.spatial_types.spatial_types import AxisAngle

# %% exporting


@dataclass(kw_only=True)
class AttemptExport:
    """
    The stored attempts of the robot of a scene on the object in it, as exported
    attempts.
    """

    scene: PickUpScene
    """
    The robot and the object whose attempts are exported.
    """

    database: GraspDatabase
    """
    Where the attempts are stored.
    """

    commit: str
    """
    The commit of the code that exports them.
    """

    @staticmethod
    def checked_out_commit() -> str:
        """
        :return: The commit of this checkout, marked ``-dirty`` when it has uncommitted
            changes.
        """
        return subprocess.run(
            ["git", "describe", "--always", "--dirty", "--abbrev=40"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    @cached_property
    def grasp_ranges(self) -> ExportedGraspRanges:
        """
        The ranges the statement of the object draws the grasp's numbers from.

        :raises GraspRangeNotAnIntervalError: If the statement allows a number from more
            than one interval.
        """
        statement = self.scene.graspable.grasped_part().surface_grasp_statement()
        event = UnderspecifiedParameters(
            statement
        ).truncation_assignments_from_where_conditions
        ranges = {}
        for variable in event.variables:
            intervals = event.marginal([variable]).simple_sets[0][variable].simple_sets
            if len(intervals) != 1:
                raise GraspRangeNotAnIntervalError(
                    variable_name=variable.name, number_of_intervals=len(intervals)
                )
            name = variable.name.removeprefix(SurfaceGraspVariablePrefix.GRASP)
            ranges[name] = ValueRange(
                lowest=float(intervals[0].lower), highest=float(intervals[0].upper)
            )
        return ExportedGraspRanges(**ranges)

    def exported_attempts(self) -> List[ExportedAttempt]:
        """
        :return: Every stored attempt of the scene's robot on its object, in the order
            they were stored.
        """
        task = GraspLearningTask.of_graspable(
            self.scene.graspable, type(self.scene.arm.end_effector)
        )
        robot_type = type(self.scene.robot)
        return [
            self.exported_attempt(attempt_number=number, attempt=attempt)
            for number, attempt in self.database.numbered_attempts(task).items()
            if attempt.robot is robot_type
        ]

    def exported_attempt(
        self, *, attempt_number: int, attempt: GraspAttempt
    ) -> ExportedAttempt:
        """
        :param attempt_number: The number the attempt was stored under.
        :param attempt: A stored attempt, tried already.
        :return: The attempt as an export file holds it.
        """
        grasp = attempt.grasp
        result = grasp.result
        return ExportedAttempt(
            attempt_number=attempt_number,
            robot=attempt.robot.__name__,
            object_name=attempt.object_name,
            source=str(attempt.source),
            placement=ExportedPlacement(
                x=attempt.placement.x,
                y=attempt.placement.y,
                yaw=attempt.placement.yaw,
            ),
            grasp=ExportedGrasp(
                azimuth=grasp.azimuth,
                height=grasp.height,
                depth=grasp.depth,
                pitch=grasp.pitch,
                roll=grasp.roll,
            ),
            grasp_ranges=self.grasp_ranges,
            result=ExportedResult(
                lifted=result.lifted,
                object_raised=result.object_raised,
                motion_completed=result.motion_completed,
                object_displacement=self.exported_vector(
                    vector=result.object_displacement
                ),
                object_rotation=self.exported_rotation(rotation=result.object_rotation),
                translational_slip=self.exported_vector(
                    vector=result.translational_slip
                ),
                rotational_slip=self.exported_rotation(rotation=result.rotational_slip),
            ),
            commit=self.commit,
        )

    @staticmethod
    def exported_vector(*, vector: Vector3) -> ExportedVector:
        """
        :param vector: A vector of the simulation.
        :return: Its three components.
        """
        x, y, z = (float(component) for component in vector.to_np()[:3])
        return ExportedVector(x=x, y=y, z=z)

    @classmethod
    def exported_rotation(cls, *, rotation: AxisAngle) -> ExportedRotation:
        """
        :param rotation: A rotation of the simulation.
        :return: Its axis and angle.
        """
        return ExportedRotation(
            axis=cls.exported_vector(vector=rotation.axis), angle=float(rotation.angle)
        )


# %% the command line


def main() -> None:
    """
    Export the stored attempts of the chosen robot on the chosen object.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    scene_choice = PickUpSceneChoice(parser)
    scene_choice.add_arguments()
    parser.add_argument(
        "--output", type=Path, required=True, help="the file the attempts go to"
    )
    arguments = parser.parse_args()
    export = AttemptExport(
        scene=scene_choice.scene(arguments),
        database=GraspDatabase.from_environment(),
        commit=AttemptExport.checked_out_commit(),
    )
    attempts = export.exported_attempts()
    ExportFile(path=arguments.output).write(attempts=attempts)
    print(f"{len(attempts)} attempts written to {arguments.output}")


if __name__ == "__main__":
    main()

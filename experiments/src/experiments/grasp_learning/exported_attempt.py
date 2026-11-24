"""
One grasp attempt as an export file holds it, one JSON line per attempt, written and
read with :mod:`krrood.adapters.json_serializer`.

The records hold plain numbers only, so that the learning side reads them without the
simulation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import List

from krrood.adapters.json_serializer import from_json, to_json

# %% numbers of space and rotation


@dataclass(kw_only=True)
class ExportedVector:
    """
    A vector along the axes the field holding it names.
    """

    x: float
    """
    The component along the first axis.
    """

    y: float
    """
    The component along the second axis.
    """

    z: float
    """
    The component along the third axis.
    """


@dataclass(kw_only=True)
class ExportedRotation:
    """
    A rotation by an angle about an axis.
    """

    axis: ExportedVector
    """
    The unit axis rotated about, along the axes the field holding the rotation names.
    """

    angle: float
    """
    The angle rotated by, in radians.
    """


# %% the grasp


@dataclass(kw_only=True)
class ExportedGrasp:
    """
    The five numbers of a surface grasp, named as on
    :class:`~semantic_digital_twin.grasping.surface_grasp.SurfaceGrasp`.
    """

    azimuth: float
    """
    The direction around the object's vertical axis from which the point is found, in
    radians.
    """

    height: float
    """
    The height of the point, as a fraction of the object's bounding box from its bottom
    to its top.
    """

    depth: float
    """
    How far beyond the surface point, along the line of sight, the gripper's center
    closes, in meters.
    """

    pitch: float
    """
    How far the direction of approach is tilted from straight down towards the line of
    sight, in radians.
    """

    roll: float
    """
    How far the closing direction is turned about the direction of approach, in radians.
    """


@dataclass(kw_only=True)
class ValueRange:
    """
    The values a grasp number was drawn from, from :attr:`lowest` up to but not
    including :attr:`highest`.
    """

    lowest: float
    """
    The smallest value allowed.
    """

    highest: float
    """
    The bound every value lies below.
    """


@dataclass(kw_only=True)
class ExportedGraspRanges:
    """
    The range each number of :class:`ExportedGrasp` was drawn from, as the statement of
    the object states it.
    """

    azimuth: ValueRange
    """
    The range of :attr:`ExportedGrasp.azimuth`.
    """

    height: ValueRange
    """
    The range of :attr:`ExportedGrasp.height`.
    """

    depth: ValueRange
    """
    The range of :attr:`ExportedGrasp.depth`.
    """

    pitch: ValueRange
    """
    The range of :attr:`ExportedGrasp.pitch`.
    """

    roll: ValueRange
    """
    The range of :attr:`ExportedGrasp.roll`.
    """


# %% the attempt


@dataclass(kw_only=True)
class ExportedPlacement:
    """
    Where the object stood for the attempt.
    """

    x: float
    """
    The x-coordinate of the middle of the object's footprint, in meters.
    """

    y: float
    """
    The y-coordinate of the middle of the object's footprint, in meters.
    """

    yaw: float
    """
    How far the object is turned about the vertical axis from how it is modeled, in
    radians.
    """


@dataclass(kw_only=True)
class ExportedResult:
    """
    What happened to the object, as
    :class:`~semantic_digital_twin.grasping.surface_grasp.GraspResult` records it.
    """

    lifted: bool
    """
    Whether the object ended up clear of where it rested and the robot went through
    every step of the pick-up.
    """

    object_raised: bool
    """
    Whether the object ended up clear of where it rested.
    """

    motion_completed: bool
    """
    Whether the robot went through every step of the pick-up within its time limit.
    """

    object_displacement: ExportedVector
    """
    How far and in which direction the object moved from where it rested to where it was
    held at the end, in meters, along the world's axes.
    """

    object_rotation: ExportedRotation
    """
    How the object turned from where it rested to where it was held at the end, its axis
    along the world's axes.
    """

    translational_slip: ExportedVector
    """
    How far and in which direction the object moved relative to the grasp while it was
    lifted and held, in meters, along the grasp's axes.
    """

    rotational_slip: ExportedRotation
    """
    How the object turned relative to the grasp while it was lifted and held, its axis
    along the grasp's axes.
    """


@dataclass(kw_only=True)
class ExportedAttempt:
    """
    One attempt of a robot to pick an object up by a surface grasp.
    """

    attempt_number: int
    """
    The number the attempt was stored under, unique within its database.
    """

    robot: str
    """
    The name of the robot's type.
    """

    object_name: str
    """
    The name of the object's body.
    """

    source: str
    """
    Where the grasp was drawn from.
    """

    placement: ExportedPlacement
    """
    Where the object stood.
    """

    grasp: ExportedGrasp
    """
    The grasp tried.
    """

    grasp_ranges: ExportedGraspRanges
    """
    The ranges the grasp's numbers were drawn from.
    """

    result: ExportedResult
    """
    What happened to the object.
    """

    commit: str
    """
    The commit of the code that exported the attempt.
    """


# %% the file


@dataclass(kw_only=True)
class ExportFile:
    """
    A file of exported attempts, one JSON line each.
    """

    path: Path
    """
    Where the file is.
    """

    def write(self, *, attempts: List[ExportedAttempt]) -> None:
        """
        Write the attempts, replacing what the file held.

        :param attempts: The attempts, in the order they are written.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            "".join(json.dumps(to_json(attempt)) + "\n" for attempt in attempts)
        )

    def read(self) -> List[ExportedAttempt]:
        """
        :return: The attempts, in the order they were written.
        """
        return [
            from_json(json.loads(line))
            for line in self.path.read_text().splitlines()
            if line
        ]

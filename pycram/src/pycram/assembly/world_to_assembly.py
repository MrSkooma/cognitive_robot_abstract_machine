

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body

from .assembly_plan import AssemblyPlan, AssemblyStep
from ..datastructures.pose import PoseStamped


def _default_body_filter(body: Body) -> bool:
    """
    Default filter for bodies: include bodies that have collision geometry.

    Bodies without collision geometry are typically reference frames or
    abstract points that cannot be grasped.

    :param body: The body to check.
    :return: True if the body should be included in the assembly plan.
    """
    return len(body.collision.shapes) > 0


@dataclass
class WorldToAssemblyPlan:
    """
    Converts a World into an AssemblyPlan.

    Traverses the world's kinematic structure in topological order,
    computes world-frame poses for each body, and creates AssemblySteps.

    :ivar body_filter: Optional filter to exclude certain bodies.
        Defaults to excluding bodies without collision geometry.
    """

    body_filter: Optional[Callable[[Body], bool]] = field(default=None)


    def convert(self, world: World) -> AssemblyPlan:
        """
        Convert a World into an AssemblyPlan.

        Iterates over bodies in topological order (parents before children),
        skips the root body, and creates AssemblySteps with world-frame poses.

        :param world: The World to convert.
        :return: An AssemblyPlan containing steps for each valid body.
        """
        assembly_plan = AssemblyPlan()

        filter_fn = (
            self.body_filter if self.body_filter is not None else _default_body_filter
        )

        bodies_sorted = world.bodies_topologically_sorted

        order = 0
        for body in bodies_sorted:
            if body == world.root:
                continue

            if not filter_fn(body):
                continue

            world_T_body = world.compute_forward_kinematics(world.root, body)
            target_pose = PoseStamped.from_matrix(world_T_body.to_np(), world.root)

            step = AssemblyStep(body=body, target_pose=target_pose, order=order)
            assembly_plan.add_step(step)
            order += 1

        return assembly_plan

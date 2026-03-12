from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Callable, Iterator, Dict, Any, Optional, Type, Union, TYPE_CHECKING

from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.world_description.world_entity import Body

from ..datastructures.enums import Arms, ApproachDirection, VerticalAlignment
from ..datastructures.grasp import GraspDescription
from ..language import SequentialPlan
from ..datastructures.pose import PoseStamped
from ..designators.location_designator import CostmapLocation
from ..robot_plans import (
    PickUpActionDescription,
    NavigateActionDescription,
    PlaceActionDescription,
    ParkArmsActionDescription,
)

if TYPE_CHECKING:
    from ..datastructures.pose import PoseStamped
    from ..datastructures.dataclasses import Context
    from ..robot_plans.actions.base import ActionDescription
    from ..language import LanguagePlan


def _default_grasp() -> GraspDescription:
    """
    Default grasp: front approach with no vertical alignment.

    :return: A default GraspDescription.
    """
    return GraspDescription(
        ApproachDirection.FRONT, VerticalAlignment.NoAlignment, False
    )


@dataclass
class AssemblyStep:
    """
    Represents a single assembly step: placing a body at a target pose.

    :ivar body: The body to be placed.
    :ivar target_pose: The target pose in world frame where the body should be placed.
    :ivar order: The order of this step in the assembly sequence (topological order).
    :ivar grasp_description: Optional grasp description for this step.
    :ivar action_type: Optional action type to use (default: TransportAction).
    :ivar custom_params: Additional parameters for the action.
    """

    body: Body
    target_pose: PoseStamped
    order: int
    grasp_description: Optional[GraspDescription] = None
    action_type: Optional[Type[ActionDescription]] = None
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"AssemblyStep(body={self.body.name}, order={self.order})"


@dataclass
class AssemblyPlan:
    """
    A collection of AssemblySteps representing a complete assembly plan.

    Provides iteration, filtering, configuration, and execution methods.
    Supports customization of grasp, action type, and parameters per step or globally.
    """

    steps: List[AssemblyStep] = field(default_factory=list)
    """The list of assembly steps in order."""

    start_point: Optional[PoseStamped] = None
    """Starting point for assembly. If None, defaults to first object's location."""

    default_arm: Arms = field(default=Arms.RIGHT)
    """Default arm to use for manipulation."""

    default_grasp: Optional[Union[GraspDescription, Callable[[Body], GraspDescription]]] = None
    """Default grasp or resolver function. Can be a GraspDescription or callable(Body) -> GraspDescription."""

    park_between_steps: bool = True
    """Whether to park arms between steps."""

    skip_navigation: bool = False
    """Whether to skip navigation actions (for stationary robots)."""

    def __iter__(self) -> Iterator[AssemblyStep]:
        """Iterate over assembly steps in order."""
        return iter(self.steps)

    def __len__(self) -> int:
        """Return the number of assembly steps."""
        return len(self.steps)

    def __getitem__(self, index: int) -> AssemblyStep:
        """Get an assembly step by index."""
        return self.steps[index]

    def add_step(self, step: AssemblyStep) -> None:
        """Add a step to the assembly plan."""
        self.steps.append(step)

    def _find_step(self, body: Body) -> Optional[AssemblyStep]:
        """Find a step by body."""
        for step in self.steps:
            if step.body == body:
                return step
        return None

    def set_grasp(self, body: Body, grasp: GraspDescription) -> AssemblyPlan:
        """
        Set grasp description for a specific body.

        :param body: The body to configure.
        :param grasp: The grasp description to use.
        :return: Self for chaining.
        """
        step = self._find_step(body)
        if step:
            step.grasp_description = grasp
        return self

    def set_action(self, body: Body, action_type: Type[ActionDescription]) -> AssemblyPlan:
        """
        Set action type for a specific body.

        :param body: The body to configure.
        :param action_type: The action type to use (e.g., TransportAction).
        :return: Self for chaining.
        """
        step = self._find_step(body)
        if step:
            step.action_type = action_type
        return self

    def set_params(self, body: Body, **kwargs: Any) -> AssemblyPlan:
        """
        Set custom parameters for a specific body's action.

        :param body: The body to configure.
        :param kwargs: Additional parameters for the action.
        :return: Self for chaining.
        """
        step = self._find_step(body)
        if step:
            step.custom_params.update(kwargs)
        return self

    def set_start_point(self, pose: PoseStamped) -> AssemblyPlan:
        """
        Set the starting point for assembly.

        :param pose: The pose where the robot should start.
        :return: Self for chaining.
        """
        self.start_point = pose
        return self

    def filter(self, predicate: Callable[[AssemblyStep], bool]) -> AssemblyPlan:
        """
        Filter assembly steps based on a predicate.

        :param predicate: A callable that takes an AssemblyStep and returns True to keep it.
        :return: A new AssemblyPlan containing only steps that match the predicate.
        """
        filtered_steps = [step for step in self.steps if predicate(step)]
        return AssemblyPlan(
            steps=filtered_steps,
            start_point=self.start_point,
            default_arm=self.default_arm,
            default_grasp=self.default_grasp,
            park_between_steps=self.park_between_steps,
            skip_navigation=self.skip_navigation,
        )

    def reorder(self, key: Callable[[AssemblyStep], int]) -> AssemblyPlan:
        """
        Reorder assembly steps based on a key function.

        :param key: A callable that takes an AssemblyStep and returns a sort key.
        :return: A new AssemblyPlan with steps sorted by the key.
        """
        sorted_steps = sorted(self.steps, key=key)
        for i, step in enumerate(sorted_steps):
            step.order = i
        return AssemblyPlan(
            steps=sorted_steps,
            start_point=self.start_point,
            default_arm=self.default_arm,
            default_grasp=self.default_grasp,
            park_between_steps=self.park_between_steps,
            skip_navigation=self.skip_navigation,
        )

    def _resolve_grasp(self, step: AssemblyStep) -> GraspDescription:
        """
        Resolve the grasp for a step, using step override, default, or fallback.
        """
        if step.grasp_description is not None:
            return step.grasp_description
        
        if self.default_grasp is not None:
            if callable(self.default_grasp):
                return self.default_grasp(step.body)
            return self.default_grasp
        return _default_grasp()

    def _default_transport_actions(
            self,
            step: AssemblyStep,
            arm: Arms,
            grasp: GraspDescription,
            robot_view: AbstractRobot,
    ) -> List[ActionDescription]:
        """
        Creates the default sequence of actions for transporting an object (PickUp → Navigate → Place).

        :param step: The assembly step to create actions for.
        :param arm: The arm to use for manipulation.
        :param grasp: The grasp description to use.
        :param robot_view: The robot for reachability calculations.
        :return: List of action descriptions for the transport sequence.
        """
        actions = []

        pickup = PickUpActionDescription(
            object_designator=step.body,
            arm=arm,
            grasp_description=grasp,
        )
        actions.append(pickup)

        # Skip navigation for stationary robots
        if not self.skip_navigation and robot_view is not None:
            navigate_location = CostmapLocation(
                target=step.target_pose,
                reachable_arm=arm,
                reachable_for=robot_view,
                grasp_descriptions=grasp,
            )
            navigate = NavigateActionDescription(
                target_location=navigate_location,
                keep_joint_states=True,
            )
            actions.append(navigate)

        place = PlaceActionDescription(
            object_designator=step.body,
            target_location=step.target_pose,
            arm=arm,
        )
        actions.append(place)

        return actions

    def to_plan(
        self,
        plan_type: Type[LanguagePlan],
        context: Context,
        robot_view: AbstractRobot = None,
    ) -> LanguagePlan:
        """
        Convert this AssemblyPlan into a PyCRAM plan.

        Generates actions for each step using the configured action type
        (default: TransportAction pattern with PickUp → Navigate → Place).

        :param plan_type: The plan type to create (e.g., SequentialPlan).
        :param context: The PyCRAM context for plan execution.
        :param robot_view: The robot for reachability calculations.
        :return: A plan of the specified type containing all actions.
        """
        action_descriptions: List[Any] = []

        # Navigate to start point if set
        if self.start_point is not None:
            navigate_start = NavigateActionDescription(
                target_location=self.start_point,
                keep_joint_states=False,
            )
            action_descriptions.append(navigate_start)

        for step in self.steps:
            grasp = self._resolve_grasp(step)
            arm = self.default_arm

            if step.action_type is not None:
                action = step.action_type.description(
                    object_designator=step.body,
                    target_location=step.target_pose,
                    arm=arm,
                    **step.custom_params,
                )
                action_descriptions.append(action)
            else:
                transport_actions = self._default_transport_actions(
                    step=step,
                    arm=arm,
                    grasp=grasp,
                    robot_view=robot_view,
                )
                action_descriptions.extend(transport_actions)

            if self.park_between_steps:
                park = ParkArmsActionDescription(Arms.BOTH)
                action_descriptions.append(park)

        return plan_type(context, *action_descriptions)

    def __repr__(self) -> str:
        return f"AssemblyPlan(steps={len(self.steps)})"

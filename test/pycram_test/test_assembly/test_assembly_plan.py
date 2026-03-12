"""
Tests for the assembly planning module.

Tests WorldToAssemblyPlan and AssemblyPlan converters with fluent API.
"""

from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import TriangleMesh
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Point3
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName

from pycram.assembly.assembly_plan import AssemblyStep, AssemblyPlan
from pycram.assembly.world_to_assembly import WorldToAssemblyPlan
from pycram.datastructures.enums import Arms
from pycram.language import SequentialPlan
from pycram.robot_plans import PickUpAction, NavigateAction, PlaceAction, ParkArmsAction


def create_simple_mesh():
    """Create a simple cube mesh for testing."""
    import trimesh

    return trimesh.creation.box(extents=[0.1, 0.1, 0.1])


def create_test_world_with_bodies(num_bodies: int = 3) -> World:
    """
    Create a test World with a root and multiple child bodies.

    :param num_bodies: Number of child bodies to create.
    :return: A World with root and child bodies connected via FixedConnections.
    """
    world = World()

    with world.modify_world():
        # Create root body (automatically becomes root as it has no parent)
        root = Body(name=PrefixedName(name="root"))
        world.add_body(root)

        # Create child bodies with collision geometry
        for i in range(num_bodies):
            mesh = create_simple_mesh()
            collision = TriangleMesh(mesh=mesh)
            collision_collection = ShapeCollection(shapes=[collision])

            body = Body(
                name=PrefixedName(name=f"part_{i}"),
                collision=collision_collection,
            )

            # Create transform: offset each body along x-axis
            transform = HomogeneousTransformationMatrix.from_point_rotation_matrix(
                Point3(x=float(i) * 0.2, y=0.0, z=0.1)
            )

            # Create fixed connection to root
            connection = FixedConnection(
                name=PrefixedName(name=f"root_to_part_{i}"),
                parent=root,
                child=body,
                parent_T_connection_expression=transform,
            )

            world.add_connection(connection)

    return world


def create_test_world_with_shapeless_body() -> World:
    """
    Create a test World with one body that has no collision geometry.
    """
    world = World()

    with world.modify_world():
        # Create root body (automatically becomes root as it has no parent)
        root = Body(name=PrefixedName(name="root"))
        world.add_body(root)

        # Create body with collision geometry
        mesh = create_simple_mesh()
        collision = TriangleMesh(mesh=mesh)
        collision_collection = ShapeCollection(shapes=[collision])
        body_with_mesh = Body(
            name=PrefixedName(name="part_with_mesh"),
            collision=collision_collection,
        )

        # Create body without collision geometry (shapeless)
        shapeless_body = Body(
            name=PrefixedName(name="shapeless_part"),
            collision=ShapeCollection(shapes=[]),
        )

        # Create transforms
        transform1 = HomogeneousTransformationMatrix.from_point_rotation_matrix(
            Point3(x=0.1, y=0.0, z=0.1)
        )
        transform2 = HomogeneousTransformationMatrix.from_point_rotation_matrix(
            Point3(x=0.2, y=0.0, z=0.1)
        )

        # Create connections
        connection1 = FixedConnection(
            name=PrefixedName(name="root_to_mesh"),
            parent=root,
            child=body_with_mesh,
            parent_T_connection_expression=transform1,
        )
        connection2 = FixedConnection(
            name=PrefixedName(name="root_to_shapeless"),
            parent=root,
            child=shapeless_body,
            parent_T_connection_expression=transform2,
        )

        world.add_connection(connection1)
        world.add_connection(connection2)

    return world


class TestAssemblyStep:
    """Tests for AssemblyStep dataclass."""

    def test_assembly_step_creation(self):
        """Test creating an AssemblyStep."""
        world = create_test_world_with_bodies(1)
        body = [b for b in world.bodies if b != world.root][0]

        from pycram.datastructures.pose import PoseStamped

        pose = PoseStamped.from_list([0.1, 0.0, 0.1], [0, 0, 0, 1], world.root)

        step = AssemblyStep(body=body, target_pose=pose, order=0)

        assert step.body == body
        assert step.order == 0
        assert step.target_pose == pose

    def test_assembly_step_repr(self):
        """Test string representation of AssemblyStep."""
        world = create_test_world_with_bodies(1)
        body = [b for b in world.bodies if b != world.root][0]

        from pycram.datastructures.pose import PoseStamped

        pose = PoseStamped.from_list([0.1, 0.0, 0.1], [0, 0, 0, 1], world.root)

        step = AssemblyStep(body=body, target_pose=pose, order=0)

        assert "AssemblyStep" in repr(step)
        assert "order=0" in repr(step)


class TestAssemblyPlan:
    """Tests for AssemblyPlan container."""

    def test_assembly_plan_creation(self):
        """Test creating an empty AssemblyPlan."""
        plan = AssemblyPlan()
        assert len(plan) == 0
        assert list(plan) == []

    def test_assembly_plan_add_step(self):
        """Test adding steps to AssemblyPlan."""
        world = create_test_world_with_bodies(2)
        bodies = [b for b in world.bodies if b != world.root]

        from pycram.datastructures.pose import PoseStamped

        plan = AssemblyPlan()
        for i, body in enumerate(bodies):
            pose = PoseStamped.from_list(
                [float(i) * 0.2, 0.0, 0.1], [0, 0, 0, 1], world.root
            )
            step = AssemblyStep(body=body, target_pose=pose, order=i)
            plan.add_step(step)

        assert len(plan) == 2
        assert plan[0].order == 0
        assert plan[1].order == 1

    def test_assembly_plan_filter(self):
        """Test filtering AssemblyPlan."""
        world = create_test_world_with_bodies(3)
        bodies = [b for b in world.bodies if b != world.root]

        from pycram.datastructures.pose import PoseStamped

        plan = AssemblyPlan()
        for i, body in enumerate(bodies):
            pose = PoseStamped.from_list(
                [float(i) * 0.2, 0.0, 0.1], [0, 0, 0, 1], world.root
            )
            step = AssemblyStep(body=body, target_pose=pose, order=i)
            plan.add_step(step)

        # Filter to only include steps with order > 0
        filtered = plan.filter(lambda s: s.order > 0)

        assert len(filtered) == 2
        assert all(s.order > 0 for s in filtered)

    def test_assembly_plan_iteration(self):
        """Test iterating over AssemblyPlan."""
        world = create_test_world_with_bodies(2)
        bodies = [b for b in world.bodies if b != world.root]

        from pycram.datastructures.pose import PoseStamped

        plan = AssemblyPlan()
        for i, body in enumerate(bodies):
            pose = PoseStamped.from_list(
                [float(i) * 0.2, 0.0, 0.1], [0, 0, 0, 1], world.root
            )
            step = AssemblyStep(body=body, target_pose=pose, order=i)
            plan.add_step(step)

        collected = list(plan)
        assert len(collected) == 2


class TestWorldToAssemblyPlan:
    """Tests for WorldToAssemblyPlan converter."""

    def test_convert_simple_world(self):
        """Test converting a simple world to assembly plan."""
        world = create_test_world_with_bodies(3)

        converter = WorldToAssemblyPlan()
        assembly_plan = converter.convert(world)

        # Should have 3 steps (excluding root)
        assert len(assembly_plan) == 3

        # Steps should be in topological order
        for i, step in enumerate(assembly_plan):
            assert step.order == i

    def test_convert_skips_root(self):
        """Test that conversion skips the root body."""
        world = create_test_world_with_bodies(2)

        converter = WorldToAssemblyPlan()
        assembly_plan = converter.convert(world)

        # Root should not be in the plan
        body_names = [step.body.name.name for step in assembly_plan]
        assert "root" not in body_names

    def test_convert_skips_shapeless_bodies(self):
        """Test that conversion skips bodies without collision geometry."""
        world = create_test_world_with_shapeless_body()

        converter = WorldToAssemblyPlan()
        assembly_plan = converter.convert(world)

        # Should only have 1 step (the body with mesh)
        assert len(assembly_plan) == 1
        assert assembly_plan[0].body.name.name == "part_with_mesh"

    def test_convert_with_custom_filter(self):
        """Test conversion with custom body filter."""
        world = create_test_world_with_bodies(3)

        # Filter to only include bodies with "0" in name
        custom_filter = lambda b: "0" in b.name.name

        converter = WorldToAssemblyPlan(body_filter=custom_filter)
        assembly_plan = converter.convert(world)

        # Should only have 1 step
        assert len(assembly_plan) == 1
        assert "0" in assembly_plan[0].body.name.name

    def test_convert_world_frame_poses(self):
        """Test that poses are in world frame."""
        world = create_test_world_with_bodies(2)

        converter = WorldToAssemblyPlan()
        assembly_plan = converter.convert(world)

        # All poses should have world.root as frame
        for step in assembly_plan:
            assert step.target_pose.frame_id == world.root


class TestAssemblyPlanToPlan:
    """Tests for AssemblyPlan.to_plan() method."""

    def test_generate_empty_plan(self, simple_pr2_world_setup):
        """Test generating from empty assembly plan."""
        world, robot_view, context = simple_pr2_world_setup

        assembly_plan = AssemblyPlan()

        sequential_plan = assembly_plan.to_plan(SequentialPlan, context)

        # Should create an empty sequential plan
        assert sequential_plan is not None

    def test_generate_action_sequence(self, simple_pr2_world_setup):
        """Test that correct action sequence is generated."""
        world, robot_view, context = simple_pr2_world_setup
        test_world = create_test_world_with_bodies(1)

        converter = WorldToAssemblyPlan()
        assembly_plan = converter.convert(test_world)

        assembly_plan.park_between_steps = True
        sequential_plan = assembly_plan.to_plan(SequentialPlan, context)

        # Get action nodes - check child nodes of root
        action_nodes = [
            node for node in sequential_plan.nodes if node != sequential_plan.root
        ]

        # For 1 body with park_between_steps=True (no robot_view, so no Navigate):
        # PickUp, Place, ParkArms = 3 actions
        assert len(action_nodes) == 3

        # Check action types in order
        action_types = [node.designator_type for node in action_nodes]
        assert action_types[0] == PickUpAction
        assert action_types[1] == PlaceAction
        assert action_types[2] == ParkArmsAction

    def test_generate_without_park_arms(self, simple_pr2_world_setup):
        """Test generating without ParkArms between steps."""
        world, robot_view, context = simple_pr2_world_setup
        test_world = create_test_world_with_bodies(1)

        converter = WorldToAssemblyPlan()
        assembly_plan = converter.convert(test_world)

        assembly_plan.park_between_steps = False
        sequential_plan = assembly_plan.to_plan(SequentialPlan, context)

        # Get action nodes
        action_nodes = [
            node for node in sequential_plan.nodes if node != sequential_plan.root
        ]

        # For 1 body without park (no robot_view, so no Navigate): PickUp, Place = 2 actions
        assert len(action_nodes) == 2

        action_types = [node.designator_type for node in action_nodes]
        assert ParkArmsAction not in action_types

    def test_generate_uses_specified_arm(self, simple_pr2_world_setup):
        """Test that specified arm is used."""
        world, robot_view, context = simple_pr2_world_setup
        test_world = create_test_world_with_bodies(1)

        converter = WorldToAssemblyPlan()
        assembly_plan = converter.convert(test_world)

        assembly_plan.default_arm = Arms.LEFT
        sequential_plan = assembly_plan.to_plan(SequentialPlan, context)

        # Check that actions use left arm
        action_nodes = [
            node for node in sequential_plan.nodes if node != sequential_plan.root
        ]
        pickup_nodes = [
            node for node in action_nodes if node.designator_type == PickUpAction
        ]
        assert len(pickup_nodes) == 1
        assert pickup_nodes[0].kwargs.get("arm") == Arms.LEFT


class TestAssemblyPlanConfiguration:
    """Tests for the configuration options on AssemblyPlan."""

    def test_configuration_chaining(self, simple_pr2_world_setup):
        """Test that configuration can be set and used correctly."""
        world, robot_view, context = simple_pr2_world_setup
        test_world = create_test_world_with_bodies(2)

        converter = WorldToAssemblyPlan()
        assembly_plan = converter.convert(test_world)

        # Set configuration via direct assignment
        assembly_plan.default_arm = Arms.RIGHT
        assembly_plan.park_between_steps = True

        sequential_plan = assembly_plan.to_plan(SequentialPlan, context)

        assert sequential_plan is not None

        # Should have actions for 2 bodies
        # 2 * (PickUp + Place + ParkArms) = 6 actions (no Navigate when robot_view=None)
        action_nodes = [
            node for node in sequential_plan.nodes if node != sequential_plan.root
        ]
        assert len(action_nodes) == 6

    def test_filter_with_to_plan(self, simple_pr2_world_setup):
        """Test filter combined with to_plan."""
        world, robot_view, context = simple_pr2_world_setup
        test_world = create_test_world_with_bodies(3)

        converter = WorldToAssemblyPlan()
        assembly_plan = converter.convert(test_world)

        # Filter to only include first body
        filtered_plan = assembly_plan.filter(lambda s: "part_0" in s.body.name.name)
        filtered_plan.park_between_steps = False
        sequential_plan = filtered_plan.to_plan(SequentialPlan, context)

        # Should have actions for 1 body only
        # 1 * (PickUp + Place) = 2 actions (no Navigate when robot_view=None, no ParkArms)
        action_nodes = [
            node for node in sequential_plan.nodes if node != sequential_plan.root
        ]
        assert len(action_nodes) == 2

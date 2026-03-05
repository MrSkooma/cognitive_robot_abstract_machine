from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional
import re

import numpy as np
import trimesh

from .pipeline import Step
from ..world import World
from ..world_description.world_entity import Body
from ..world_description.connections import FixedConnection
from ..world_description.shape_collection import ShapeCollection
from ..world_description.geometry import TriangleMesh, Scale
from ..spatial_types import HomogeneousTransformationMatrix
from ..datastructures.prefixed_name import PrefixedName


class RootNodeNotFoundError(ValueError):
    """Exception raised when the root node cannot be found in the parsed world elements."""

    pass


@dataclass
class NodeProcessingResult:
    """Result of processing a single node in the scene graph."""

    body: Body
    """The Body object created from the node's geometry."""

    visited_nodes: Set[str]
    """Set of node names that were visited during processing."""

    children_to_visit: Set[str]
    """Set of child node names that need to be processed next."""


@dataclass
class GLTFLoader(Step):
    """Load GLTF/GLB files into a World.

    This loader parses GLTF/GLB files (including FreeCAD exports) and creates
    Body objects with FixedConnection relationships matching the scene hierarchy.

    Features:
    - Handles FreeCAD naming conventions (e.g., Bolt_001, Bolt_002 are fused)
    - Applies node transformations correctly
    - Skips non-geometry nodes while preserving hierarchy
    - Creates proper parent-child connections

    Example:
        >>> world = World()
        >>> loader = GLTFLoader(file_path="model.gltf")
        >>> world = loader.apply(world)

    Limitations:
    - Only creates FixedConnection (no joints/articulations)
    - Does not handle GLTF extensions for physics/joints

    Attributes:
        file_path: Path to the GLTF/GLB file
        scene: The loaded trimesh Scene (set after _apply is called)

    Raises:
        ValueError: If the file cannot be loaded or parsed.
    """

    file_path: str
    """Path to the GLTF/GLB file."""

    scene: Optional[trimesh.Scene] = field(default=None, init=False)
    """The loaded trimesh Scene (set after _apply is called)."""

    def _get_root_node(self) -> str:
        """
        Get the single root node of the scene graph.

        :return: The name of the root node.
        :raises RootNodeNotFoundError: If no root node exists or multiple root nodes are found.
        """
        base_frame = self.scene.graph.base_frame
        root_children = self.scene.graph.transforms.children.get(base_frame, [])
        if not root_children:
            raise RootNodeNotFoundError("No root node found in the scene graph.")

        if len(root_children) > 1:
            raise RootNodeNotFoundError(
                f"Multiple root nodes found in scene: {root_children}. "
                "The scene must have a single root node."
            )
        return root_children[0]

    def _get_relative_transform(
        self, parent_node: str, child_node: str
    ) -> HomogeneousTransformationMatrix:
        """
        Get the relative transform from parent to child node.

        Computes the transform that converts from parent frame to child frame.

        :param parent_node: Name of the parent node.
        :param child_node: Name of the child node.
        :return: The relative transformation matrix from parent to child.
        """
        parent_transform, _ = self.scene.graph.get(parent_node)
        child_transform, _ = self.scene.graph.get(child_node)

        # Compute relative transform: parent_inv @ child
        parent_inv = np.linalg.inv(parent_transform)
        relative = parent_inv @ child_transform

        return HomogeneousTransformationMatrix(relative)

    def _trimesh_to_body(self, mesh: trimesh.Trimesh, name: str) -> Body:
        """
        Convert a trimesh.Trimesh to a Body object.

        :param mesh: The trimesh mesh to convert.
        :param name: The name for the resulting Body.
        :return: A Body object with the mesh as both collision and visual geometry.
        """
        # Create TriangleMesh geometry from trimesh
        triangle_mesh = TriangleMesh(
            mesh=mesh,
            origin=HomogeneousTransformationMatrix.from_xyz_rpy(),  # Identity transform
            scale=Scale(1.0, 1.0, 1.0),  # No scaling
        )

        # Create ShapeCollection for collision and visual
        shape_collection = ShapeCollection([triangle_mesh])

        # Create Body
        body = Body(
            name=PrefixedName(name),
            collision=shape_collection,
            visual=shape_collection,  # Use same for both collision and visual
        )

        return body

    def _extract_base_name(self, node_name: str) -> str:
        """
        Extract the prefix before the first underscore.

        This handles FreeCAD suffixes (e.g., _001, _002) and trimesh-generated
        hex hash suffixes (e.g., _3a7ad2). For example:

        - ``Bolt_001_abc123`` → ``Bolt``
        - ``Scale060_1_00e71f`` → ``Scale060``

        :param node_name: The original node name with potential suffix.
        :return: The base name (text before the first underscore).
        """
        match = re.match(r"^([^_]+)", str(node_name))
        return match.group(1) if match else str(node_name)

    def _collect_matching_children(
        self, node: str, base_name: str, object_nodes: Set[str]
    ) -> Tuple[Set[str], Set[str]]:
        """
        Collect children that match base_name and those that don't.

        :param node: The parent node to collect children from.
        :param base_name: The base name to match against.
        :param object_nodes: Set of already processed object nodes to exclude.
        :return: A tuple of (matching children, non-matching children).
        """
        matching = set()
        non_matching = set()
        for child in self.scene.graph.transforms.children.get(node, []):
            if child in object_nodes:
                continue
            if self._extract_base_name(child) == base_name:
                matching.add(child)
            else:
                non_matching.add(child)
        return matching, non_matching

    def _grouping_similar_meshes(self, base_node: str) -> Tuple[Set[str], Set[str]]:
        """
        Group meshes with similar names (e.g., Bolt_001, Bolt_002 -> Bolt).

        This method recursively collects all nodes that share the same base name
        as the given node, which is useful for FreeCAD exports where parts are
        split into multiple numbered meshes.

        :param base_node: The starting node to group from.
        :return: A tuple of (grouped object nodes, non-matching child nodes to visit).
        """
        base_name = self._extract_base_name(base_node)
        object_nodes = {base_node}
        new_object_nodes = set()
        to_search = [base_node]
        max_iterations = 10000

        for _ in range(max_iterations):
            if not to_search:
                break
            node = to_search.pop()
            matching, non_matching = self._collect_matching_children(
                node, base_name, object_nodes
            )
            object_nodes.update(matching)
            to_search.extend(matching)
            new_object_nodes.update(non_matching)
        else:
            print(
                f"Warning: Hit max iterations in _grouping_similar_meshes for {base_node}"
            )

        return object_nodes, new_object_nodes

    def _fusion_meshes(self, object_nodes: Set[str]) -> trimesh.Trimesh:
        """
        Fuse multiple mesh nodes into a single mesh.

        Applies the world transform to each mesh before concatenating them.

        :param object_nodes: Set of node names to fuse.
        :return: A single concatenated mesh, or empty Trimesh if no geometry found.
        """
        meshes: List[trimesh.Trimesh] = []
        for node in object_nodes:
            transform, geometry_name = self.scene.graph.get(node)
            if geometry_name is None:
                continue
            geometry = self.scene.geometry.get(geometry_name)
            if geometry is None:
                continue
            mesh = geometry.copy()
            mesh.apply_transform(transform)
            meshes.append(mesh)
        if meshes:
            return trimesh.util.concatenate(meshes)  # type: ignore[return-value]
        return trimesh.Trimesh()  # Empty mesh if no geometry found

    def _add_child_connection(
        self,
        world: World,
        parent_node: str,
        child_node: str,
        world_elements: Dict[str, Body],
    ) -> None:
        """
        Add a child body and its connection to the world.

        Creates a :class:`FixedConnection` between the parent and child bodies
        with the appropriate relative transform.

        :param world: The world to add the connection to.
        :param parent_node: The name of the parent node.
        :param child_node: The name of the child node.
        :param world_elements: Dictionary mapping node names to Body objects.
        """
        if child_node not in world_elements or parent_node not in world_elements:
            return
        parent_body = world_elements[parent_node]
        child_body = world_elements[child_node]
        world.add_kinematic_structure_entity(child_body)
        relative_transform = self._get_relative_transform(parent_node, child_node)
        conn = FixedConnection(
            parent=parent_body,
            child=child_body,
            parent_T_connection_expression=relative_transform,
            name=PrefixedName(f"{parent_node}_{child_node}"),
        )
        world.add_connection(conn)

    def _build_world_from_elements(
        self,
        world_elements: Dict[str, Body],
        connection: Dict[str, List[str]],
        world: World,
    ) -> World:
        """
        Build the world from parsed elements and their connections.

        :param world_elements: Dictionary mapping node names to Body objects.
        :param connection: Dictionary mapping parent node names to list of child node names.
        :param world: The world to add entities to.
        :return: The modified world.
        :raises RootNodeNotFoundError: If the root node is not found in world_elements.
        """
        object_root = self._get_root_node()
        if object_root not in world_elements:
            raise RootNodeNotFoundError(
                f"Root node '{object_root}' not found in world_elements"
            )

        object_root_body = world_elements[object_root]
        world.add_kinematic_structure_entity(object_root_body)

        # Connect root to world root if exists
        if world.root is not None and world.root != object_root_body:
            root_transform, _ = self.scene.graph.get(object_root)
            conn = FixedConnection(
                parent=world.root,
                child=object_root_body,
                parent_T_connection_expression=HomogeneousTransformationMatrix(
                    root_transform
                ),
                name=PrefixedName(f"object_root_{object_root}"),
            )
            world.add_connection(conn)

        # Add all child connections via BFS
        to_add_nodes = [object_root]
        while to_add_nodes:
            node = to_add_nodes.pop()
            for child in connection.get(node, []):
                to_add_nodes.append(child)
                self._add_child_connection(world, node, child, world_elements)

        return world

    def _create_empty_body(self, name: str) -> Body:
        """
        Create an empty body with no geometry.

        :param name: The name for the Body.
        :return: A Body object with empty collision and visual shape collections.
        """
        return Body(
            name=PrefixedName(name),
            collision=ShapeCollection([]),
            visual=ShapeCollection([]),
        )

    def _process_node(
        self, node: str, body_parent: str, visited_nodes: Set[str]
    ) -> Tuple[Optional[NodeProcessingResult], Set[Tuple[str, str]]]:
        """
        Process any node (geometry or non-geometry) and return result.

        Groups similar meshes, fuses them, and creates a Body object if geometry exists.
        Otherwise, passes children through to the body_parent.

        :param node: The node name to process.
        :param body_parent: The parent body name for connections.
        :param visited_nodes: Set of already visited node names.
        :return: Tuple of (NodeProcessingResult if body created, children to visit with parents).
        """
        object_nodes, remaining_children = self._grouping_similar_meshes(node)
        mesh = self._fusion_meshes(object_nodes)

        children_raw = self.scene.graph.transforms.children.get(node, [])
        non_matching = {
            c for c in children_raw if c not in object_nodes and c not in visited_nodes
        }

        if len(mesh.vertices) > 0:
            base_name = self._extract_base_name(node)
            body = self._trimesh_to_body(mesh, base_name)
            result = NodeProcessingResult(
                body=body,
                visited_nodes=object_nodes,
                children_to_visit=remaining_children.difference(visited_nodes),
            )
            children_to_visit = {(c, node) for c in result.children_to_visit} | {
                (c, body_parent) for c in non_matching
            }
            return result, children_to_visit

        # No geometry - pass children through to current body_parent
        children_to_visit = {
            (c, body_parent) for c in children_raw if c not in visited_nodes
        }
        return None, children_to_visit

    def _process_root_node(
        self, root: str
    ) -> Tuple[Body, Set[str], Set[Tuple[str, str]]]:
        """
        Process the root node of the scene graph.

        Creates the root Body object and identifies children to visit.

        :param root: The root node name.
        :return: A tuple of (root body, visited nodes, children to visit with their body parents).
        """
        result, children_to_visit = self._process_node(root, root, set())

        if result is not None and len(result.body.visual.shapes) > 0:
            return result.body, result.visited_nodes, children_to_visit

        # Root has no geometry - create empty body
        body = self._create_empty_body(root)
        children_raw = self.scene.graph.transforms.children.get(root, [])
        return body, {root}, {(child, root) for child in children_raw}

    def _add_body_to_world_elements(
        self,
        world_elements: Dict[str, Body],
        base_name_to_node: Dict[str, str],
        node: str,
        body: Body,
    ) -> None:
        """
        Add a body to world_elements, merging meshes if the base name already exists.

        If a body with the same base name already exists, fuses the new body's
        meshes into the existing body.

        :param world_elements: Dictionary mapping node names to Body objects.
        :param base_name_to_node: Dictionary mapping base names to their first node key.
        :param node: The node name to use as the key.
        :param body: The Body object to add.
        """
        base_name = str(body.name)

        existing_node = base_name_to_node.get(base_name)

        if existing_node is not None:
            # Merge meshes from new body into existing body
            existing_body = world_elements[existing_node]
            merged_shapes = (
                list(existing_body.visual.shapes) + list(body.visual.shapes)
            )
            merged_collision = (
                list(existing_body.collision.shapes) + list(body.collision.shapes)
            )
            existing_body.visual = ShapeCollection(merged_shapes)
            existing_body.collision = ShapeCollection(merged_collision)
            # Map the new node to the existing body for connection tracking
            world_elements[node] = existing_body
        else:
            base_name_to_node[base_name] = node
            world_elements[node] = body

    def _create_world_objects(self, world: World) -> World:
        """
        Parse the scene graph and create world objects with connections.

        This method traverses the scene graph, groups similar meshes (e.g., Bolt_001, Bolt_002),
        fuses them and creates Body objects with parent-child connections.

        Non-geometry nodes (like transforms/sketches) are skipped but their children
        are still processed with the correct parent body.

        :param world: The world to populate with objects.
        :return: The modified world with all bodies and connections added.
        """
        root = self._get_root_node()
        world_elements: Dict[str, Body] = {}
        base_name_to_node: Dict[str, str] = {}
        connection: Dict[str, List[str]] = {}
        visited_nodes: Set[str] = set()

        # Process root
        root_body, root_visited, to_visit_new_node = self._process_root_node(root)
        world_elements[root] = root_body
        base_name_to_node[str(root_body.name)] = root
        visited_nodes = visited_nodes.union(root_visited)

        while to_visit_new_node:
            node, body_parent = to_visit_new_node.pop()

            if node in visited_nodes:
                continue

            result, children_to_visit = self._process_node(node, body_parent, visited_nodes)

            if result is not None:
                # Node created a body
                self._add_body_to_world_elements(
                    world_elements, base_name_to_node, node, result.body
                )
                visited_nodes.update(result.visited_nodes | {node})
                connection.setdefault(body_parent, []).append(node)
                connection[node] = []
                to_visit_new_node.update(children_to_visit)
            else:
                # No geometry - just pass through to children
                visited_nodes.add(node)
                to_visit_new_node.update(children_to_visit)

        return self._build_world_from_elements(world_elements, connection, world)

    def _apply(self, world: World) -> World:
        """
        Load GLTF/GLB file and create world objects.

        :param world: The world to populate with loaded objects.
        :return: The modified world with bodies and connections from the GLTF file.
        :raises ValueError: If the file cannot be loaded.
        """
        try:
            self.scene = trimesh.load(self.file_path)  # type: ignore[assignment]
        except Exception as e:
            raise ValueError(f"Failed to load file '{self.file_path}': {e}") from e

        # Handle case where trimesh loads a single mesh instead of a Scene
        if isinstance(self.scene, trimesh.Trimesh):
            mesh = self.scene
            self.scene = trimesh.Scene()
            self.scene.add_geometry(mesh, node_name="root", geom_name="root_geom")

        if len(self.scene.geometry) == 0:
            root = self._get_root_node()
            world.add_kinematic_structure_entity(self._create_empty_body(root))
            return world

        return self._create_world_objects(world)

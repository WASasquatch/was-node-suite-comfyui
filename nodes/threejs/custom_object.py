"""A scene-graph object built by hand-written JavaScript."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_GEOMETRY, THREE_MATERIAL, THREE_OBJECT
from ...modules.threejs.spec import compact_deps, create_spec

REQUIRES = "threejs"

DEFAULT_BODY = "const group = new THREE.Group();\nif (object1) group.add(object1);\nreturn group;"


class ThreeCustomObject(io.ComfyNode):
    """Build an object from a JavaScript body."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeCustomObject",
            display_name="Three Custom Object",
            search_aliases=[
                "WASThreeCustomObject",
                "Three Custom Object",
                "custom object",
                "javascript",
                "instanced",
            ],
            category="WAS Suite/Three",
            description=(
                "Build an object from a short JavaScript body for what wiring cannot express, "
                "such as a hundred scattered copies, an instanced crowd or a point cloud. Wired "
                "inputs arrive in the code by socket name. The code runs in the browser with the "
                "reach of any extension, so only load a workflow carrying it from a trusted "
                "source."
            ),
            inputs=[
                io.String.Input(
                    "javascript",
                    default=DEFAULT_BODY,
                    multiline=True,
                    tooltip=(
                        "A body returning an Object3D, as "
                        "`return new THREE.InstancedMesh(geometry1, material1, 100);`. `THREE` "
                        "is in scope."
                    ),
                ),
                THREE_GEOMETRY.Input(
                    "geometry1",
                    optional=True,
                    tooltip="A shape reachable in the body as `geometry1`.",
                ),
                THREE_GEOMETRY.Input(
                    "geometry2",
                    optional=True,
                    tooltip="A shape reachable in the body as `geometry2`.",
                ),
                THREE_MATERIAL.Input(
                    "material1",
                    optional=True,
                    tooltip="A surface reachable in the body as `material1`.",
                ),
                THREE_MATERIAL.Input(
                    "material2",
                    optional=True,
                    tooltip="A surface reachable in the body as `material2`.",
                ),
                THREE_OBJECT.Input(
                    "object1",
                    optional=True,
                    tooltip="An object reachable in the body as `object1`, to add or to place.",
                ),
                THREE_OBJECT.Input(
                    "object2",
                    optional=True,
                    tooltip="An object reachable in the body as `object2`, to add or to place.",
                ),
            ],
            outputs=[
                THREE_OBJECT.Output(
                    display_name="object",
                    tooltip="The object the code returned, for Three Group or Three Scene.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        javascript,
        geometry1=None,
        geometry2=None,
        material1=None,
        material2=None,
        object1=None,
        object2=None,
    ) -> io.NodeOutput:
        """Carry the code and its inputs to the browser."""
        return io.NodeOutput(
            create_spec(
                "object",
                "CustomObject",
                params={"javascript": javascript},
                deps=compact_deps(
                    geometry1=geometry1,
                    geometry2=geometry2,
                    material1=material1,
                    material2=material2,
                    object1=object1,
                    object2=object2,
                ),
            )
        )

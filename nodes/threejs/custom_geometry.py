"""A shape built by hand-written JavaScript."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_GEOMETRY
from ...modules.threejs.spec import create_spec

REQUIRES = "threejs"

DEFAULT_BODY = "return new THREE.TorusKnotGeometry(1, 0.3, 128, 24);"


class ThreeCustomGeometry(io.ComfyNode):
    """Build a geometry from a JavaScript body."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeCustomGeometry",
            display_name="Three Custom Geometry",
            search_aliases=[
                "WASThreeCustomGeometry",
                "Three Custom Geometry",
                "custom geometry",
                "javascript",
                "torus knot",
            ],
            category="WAS Suite/Three",
            description=(
                "Make any shape the pack has no node for, such as a torus knot, a lathe or a "
                "tube, from a short JavaScript body that returns one, for Three Mesh. The code "
                "runs in the browser with the reach of any extension, so only load a workflow "
                "carrying it from a trusted source."
            ),
            inputs=[
                io.String.Input(
                    "javascript",
                    default=DEFAULT_BODY,
                    multiline=True,
                    tooltip=(
                        "A body returning a geometry, as "
                        "`return new THREE.LatheGeometry(points);`. `THREE` is in scope."
                    ),
                ),
            ],
            outputs=[
                THREE_GEOMETRY.Output(
                    display_name="geometry",
                    tooltip="The shape the code returned, for the geometry socket on Three Mesh.",
                ),
            ],
        )

    @classmethod
    def execute(cls, javascript) -> io.NodeOutput:
        """Carry the code to the browser."""
        return io.NodeOutput(
            create_spec("geometry", "CustomGeometry", params={"javascript": javascript})
        )

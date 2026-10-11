"""A rectangular box shape for a Three.js mesh."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_GEOMETRY
from ...modules.threejs.spec import create_spec

REQUIRES = "threejs"


class ThreeBoxGeometry(io.ComfyNode):
    """Build a box geometry descriptor."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeBoxGeometry",
            display_name="Three Box Geometry",
            search_aliases=[
                "WASThreeBoxGeometry",
                "Three Box Geometry",
                "box",
                "cube",
                "geometry",
            ],
            category="WAS Suite/Three",
            description=(
                "A rectangular box, sized in scene units and centred on its own origin, for "
                "Three Mesh together with a material. Segment counts subdivide each face, which "
                "matters only where a shader or a displacement map needs vertices to move."
            ),
            inputs=[
                io.Float.Input(
                    "width",
                    default=1.0,
                    min=0.0001,
                    max=100000.0,
                    step=0.01,
                    tooltip="Size along X in scene units. 1.0 is a unit cube, 2.0 twice as wide.",
                ),
                io.Float.Input(
                    "height",
                    default=1.0,
                    min=0.0001,
                    max=100000.0,
                    step=0.01,
                    tooltip="Size along Y in scene units. 1.0 is a unit cube, 0.1 a flat slab.",
                ),
                io.Float.Input(
                    "depth",
                    default=1.0,
                    min=0.0001,
                    max=100000.0,
                    step=0.01,
                    tooltip="Size along Z in scene units. 1.0 is a unit cube, 0.1 a thin panel.",
                ),
                io.Int.Input(
                    "width_segments",
                    default=1,
                    min=1,
                    max=1024,
                    tooltip="How many divisions across X. 1 leaves flat faces, 32 gives a shader room to bend them.",
                ),
                io.Int.Input(
                    "height_segments",
                    default=1,
                    min=1,
                    max=1024,
                    tooltip="How many divisions across Y. 1 leaves flat faces, 32 gives a shader room to bend them.",
                ),
                io.Int.Input(
                    "depth_segments",
                    default=1,
                    min=1,
                    max=1024,
                    tooltip="How many divisions across Z. 1 leaves flat faces, 32 gives a shader room to bend them.",
                ),
            ],
            outputs=[
                THREE_GEOMETRY.Output(
                    display_name="geometry",
                    tooltip="The box shape, for the geometry socket on Three Mesh.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls, width, height, depth, width_segments, height_segments, depth_segments
    ) -> io.NodeOutput:
        """Describe the box."""
        return io.NodeOutput(
            create_spec(
                "geometry",
                "BoxGeometry",
                params={
                    "args": [
                        float(width),
                        float(height),
                        float(depth),
                        int(width_segments),
                        int(height_segments),
                        int(depth_segments),
                    ]
                },
            )
        )

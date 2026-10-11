"""A flat rectangle for a Three.js mesh."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_GEOMETRY
from ...modules.threejs.spec import create_spec

REQUIRES = "threejs"


class ThreePlaneGeometry(io.ComfyNode):
    """Build a plane geometry descriptor."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreePlaneGeometry",
            display_name="Three Plane Geometry",
            search_aliases=[
                "WASThreePlaneGeometry",
                "Three Plane Geometry",
                "plane",
                "quad",
                "floor",
            ],
            category="WAS Suite/Three",
            description=(
                "A flat rectangle facing +Z, the usual carrier for a texture and, laid flat with "
                "Three Transform Object, the usual ground plane. Only one side is drawn unless "
                "the material's side is set to 'double'."
            ),
            inputs=[
                io.Float.Input(
                    "width",
                    default=1.0,
                    min=0.0001,
                    max=100000.0,
                    step=0.01,
                    tooltip="Size along X in scene units. 1.0 is a unit card, 20.0 a floor.",
                ),
                io.Float.Input(
                    "height",
                    default=1.0,
                    min=0.0001,
                    max=100000.0,
                    step=0.01,
                    tooltip="Size along Y in scene units. 1.0 is square with a width of 1.0.",
                ),
                io.Int.Input(
                    "width_segments",
                    default=1,
                    min=1,
                    max=4096,
                    tooltip="Divisions across X. 1 is a flat card, 64 gives a shader room to bend it.",
                ),
                io.Int.Input(
                    "height_segments",
                    default=1,
                    min=1,
                    max=4096,
                    tooltip="Divisions across Y. 1 is a flat card, 64 gives a shader room to bend it.",
                ),
            ],
            outputs=[
                THREE_GEOMETRY.Output(
                    display_name="geometry",
                    tooltip="The plane shape, for the geometry socket on Three Mesh.",
                ),
            ],
        )

    @classmethod
    def execute(cls, width, height, width_segments, height_segments) -> io.NodeOutput:
        """Describe the plane."""
        return io.NodeOutput(
            create_spec(
                "geometry",
                "PlaneGeometry",
                params={
                    "args": [
                        float(width),
                        float(height),
                        int(width_segments),
                        int(height_segments),
                    ]
                },
            )
        )

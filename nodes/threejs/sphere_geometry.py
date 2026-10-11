"""A sphere shape for a Three.js mesh."""

from __future__ import annotations

import math

from comfy_api.latest import io

from ...modules.compat.types import THREE_GEOMETRY
from ...modules.threejs.spec import create_spec

REQUIRES = "threejs"


class ThreeSphereGeometry(io.ComfyNode):
    """Build a sphere geometry descriptor."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeSphereGeometry",
            display_name="Three Sphere Geometry",
            search_aliases=[
                "WASThreeSphereGeometry",
                "Three Sphere Geometry",
                "sphere",
                "ball",
                "geometry",
            ],
            category="WAS Suite/Three",
            description=(
                "A sphere centred on its own origin, for Three Mesh. A shorter theta length "
                "gives a dome and a shorter phi length gives a wedge; 32 by 16 segments is "
                "smooth at normal sizes."
            ),
            inputs=[
                io.Float.Input(
                    "radius",
                    default=1.0,
                    min=0.0001,
                    max=100000.0,
                    step=0.01,
                    tooltip="Radius in scene units. 1.0 gives a sphere two units across.",
                ),
                io.Int.Input(
                    "width_segments",
                    default=32,
                    min=3,
                    max=2048,
                    tooltip="Divisions around the equator. 32 looks round, 8 reads as a faceted gem.",
                ),
                io.Int.Input(
                    "height_segments",
                    default=16,
                    min=2,
                    max=2048,
                    tooltip="Divisions from pole to pole. 16 looks round, 4 reads as a faceted gem.",
                ),
                io.Float.Input(
                    "phi_start",
                    default=0.0,
                    min=-360.0,
                    max=360.0,
                    step=0.1,
                    tooltip="Where the sweep around the equator begins, in degrees. 0.0 starts at the front.",
                ),
                io.Float.Input(
                    "phi_length",
                    default=360.0,
                    min=0.0,
                    max=360.0,
                    step=0.1,
                    tooltip="How far it sweeps around, in degrees. 360.0 closes it, 180.0 gives a half.",
                ),
                io.Float.Input(
                    "theta_start",
                    default=0.0,
                    min=0.0,
                    max=180.0,
                    step=0.1,
                    tooltip="Where the sweep from the top begins, in degrees. 0.0 starts at the north pole.",
                ),
                io.Float.Input(
                    "theta_length",
                    default=180.0,
                    min=0.0,
                    max=180.0,
                    step=0.1,
                    tooltip="How far it sweeps down, in degrees. 180.0 reaches the south pole, 90.0 gives a dome.",
                ),
            ],
            outputs=[
                THREE_GEOMETRY.Output(
                    display_name="geometry",
                    tooltip="The sphere shape, for the geometry socket on Three Mesh.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        radius,
        width_segments,
        height_segments,
        phi_start,
        phi_length,
        theta_start,
        theta_length,
    ) -> io.NodeOutput:
        """Describe the sphere."""
        return io.NodeOutput(
            create_spec(
                "geometry",
                "SphereGeometry",
                params={
                    "args": [
                        float(radius),
                        int(width_segments),
                        int(height_segments),
                        math.radians(float(phi_start)),
                        math.radians(float(phi_length)),
                        math.radians(float(theta_start)),
                        math.radians(float(theta_length)),
                    ]
                },
            )
        )

"""A light source for a Three.js scene."""

from __future__ import annotations

import math

from comfy_api.latest import io

from ...modules.compat.types import THREE_OBJECT
from ...modules.threejs.spec import create_spec

REQUIRES = "threejs"

#: Kinds of light the node offers, in the order the menu lists them.
LIGHT_TYPES = ("ambient", "hemisphere", "directional", "point", "spot")


class ThreeLight(io.ComfyNode):
    """Build a light descriptor."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeLight",
            display_name="Three Light",
            search_aliases=[
                "WASThreeLight",
                "Three Light",
                "light",
                "lamp",
                "directional",
                "point light",
            ],
            category="WAS Suite/Three",
            description=(
                "One light, wired into a group or straight into the scene; a standard or "
                "physical material is black without one. Choose sunlight, a bulb, a spotlight "
                "cone, an even ambient lift or a sky-to-ground fade; only the sun, bulb and "
                "spot cast shadows."
            ),
            inputs=[
                io.Combo.Input(
                    "light_type",
                    options=list(LIGHT_TYPES),
                    default="directional",
                    tooltip=(
                        "Which kind of light. 'directional' is sun, 'point' a bulb, 'spot' a "
                        "cone, 'ambient' a flat lift, 'hemisphere' a sky colour fading to "
                        "ground_color."
                    ),
                ),
                io.String.Input(
                    "color",
                    default="#ffffff",
                    multiline=False,
                    tooltip=(
                        "Light colour as hexadecimal. #ffffff is neutral, #ffd8a8 warm, "
                        "#a8c8ff cool daylight."
                    ),
                ),
                io.String.Input(
                    "ground_color",
                    default="#404040",
                    multiline=False,
                    tooltip=(
                        "The upward bounce colour, used by 'hemisphere' alone. #404040 reads as "
                        "grey ground."
                    ),
                ),
                io.Float.Input(
                    "intensity",
                    default=2.0,
                    min=0.0,
                    max=100000.0,
                    step=0.01,
                    tooltip=(
                        "How bright the light is. 2.0 suits a key light, 0.3 a fill, 0 turns it "
                        "off."
                    ),
                ),
                io.Float.Input(
                    "position_x",
                    default=3.0,
                    min=-100000.0,
                    max=100000.0,
                    step=0.01,
                    tooltip=(
                        "Position along X in scene units. 3.0 puts a key light to the right; for a "
                        "directional light this is a direction."
                    ),
                ),
                io.Float.Input(
                    "position_y",
                    default=5.0,
                    min=-100000.0,
                    max=100000.0,
                    step=0.01,
                    tooltip="Position along Y in scene units. 5.0 puts a key light above the subject.",
                ),
                io.Float.Input(
                    "position_z",
                    default=4.0,
                    min=-100000.0,
                    max=100000.0,
                    step=0.01,
                    tooltip="Position along Z in scene units. 4.0 puts a key light in front of the subject.",
                ),
                io.Float.Input(
                    "distance",
                    default=0.0,
                    min=0.0,
                    max=1000000.0,
                    step=0.01,
                    tooltip=(
                        "How far a point or spot light reaches before it is fully dark. 0.0 "
                        "means no limit."
                    ),
                ),
                io.Float.Input(
                    "decay",
                    default=2.0,
                    min=0.0,
                    max=100.0,
                    step=0.01,
                    tooltip=(
                        "How fast a point or spot light falls off. 2.0 is real light, 0.0 does "
                        "not fall off at all."
                    ),
                ),
                io.Float.Input(
                    "angle",
                    default=45.0,
                    min=0.1,
                    max=179.0,
                    step=0.1,
                    tooltip=(
                        "Width of a spot light's cone in degrees. 45.0 is a broad pool, 10.0 a "
                        "tight beam."
                    ),
                ),
                io.Float.Input(
                    "penumbra",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How soft a spot light's edge is. 0.0 is a hard rim, 1.0 fades across "
                        "the whole cone."
                    ),
                ),
                io.Boolean.Input(
                    "cast_shadow",
                    default=True,
                    tooltip=(
                        "Whether this light throws shadows. Ignored by 'ambient' and "
                        "'hemisphere', which cast none."
                    ),
                ),
            ],
            outputs=[
                THREE_OBJECT.Output(
                    display_name="light",
                    tooltip="The light, for Three Group or the root socket on Three Scene.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        light_type,
        color,
        ground_color,
        intensity,
        position_x,
        position_y,
        position_z,
        distance,
        decay,
        angle,
        penumbra,
        cast_shadow,
    ) -> io.NodeOutput:
        """Describe the light."""
        return io.NodeOutput(
            create_spec(
                "object",
                "Light",
                params={
                    "lightType": light_type,
                    "color": color,
                    "groundColor": ground_color,
                    "intensity": float(intensity),
                    "position": [float(position_x), float(position_y), float(position_z)],
                    "distance": float(distance),
                    "decay": float(decay),
                    "angle": math.radians(float(angle)),
                    "penumbra": float(penumbra),
                    "castShadow": bool(cast_shadow),
                },
            )
        )

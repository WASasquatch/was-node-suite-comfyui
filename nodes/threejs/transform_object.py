"""Move, turn and scale an object without changing it."""

from __future__ import annotations

import math

from comfy_api.latest import io

from ...modules.compat.types import THREE_OBJECT
from ...modules.threejs.spec import create_spec, require_spec

REQUIRES = "threejs"


class ThreeTransformObject(io.ComfyNode):
    """Wrap an object in a placed parent."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeTransformObject",
            display_name="Three Transform Object",
            search_aliases=[
                "WASThreeTransformObject",
                "Three Transform Object",
                "transform",
                "position",
                "rotate",
                "scale",
            ],
            category="WAS Suite/Three",
            description=(
                "Place an object in the scene by wrapping it in a parent that carries position, "
                "rotation and scale, so one mesh can be placed in several spots. Rotation is in "
                "degrees, X then Y then Z; -90 on X lays a plane flat as a floor."
            ),
            inputs=[
                THREE_OBJECT.Input(
                    "object",
                    tooltip="The object to place. It is wrapped rather than changed.",
                ),
                io.Float.Input(
                    "position_x",
                    default=0.0,
                    min=-100000.0,
                    max=100000.0,
                    step=0.01,
                    tooltip="Move along X in scene units. 0.0 leaves it at the origin.",
                ),
                io.Float.Input(
                    "position_y",
                    default=0.0,
                    min=-100000.0,
                    max=100000.0,
                    step=0.01,
                    tooltip="Move along Y in scene units. 0.5 lifts a unit cube to sit on the ground.",
                ),
                io.Float.Input(
                    "position_z",
                    default=0.0,
                    min=-100000.0,
                    max=100000.0,
                    step=0.01,
                    tooltip="Move along Z in scene units. 0.0 leaves it put; positive moves toward the camera.",
                ),
                io.Float.Input(
                    "rotation_x",
                    default=0.0,
                    min=-36000.0,
                    max=36000.0,
                    step=0.1,
                    tooltip="Turn around X in degrees. -90.0 lays an upright plane flat as a floor.",
                ),
                io.Float.Input(
                    "rotation_y",
                    default=0.0,
                    min=-36000.0,
                    max=36000.0,
                    step=0.1,
                    tooltip="Turn around Y in degrees. 45.0 swings the object to face the corner.",
                ),
                io.Float.Input(
                    "rotation_z",
                    default=0.0,
                    min=-36000.0,
                    max=36000.0,
                    step=0.1,
                    tooltip="Turn around Z in degrees. 180.0 stands the object on its head.",
                ),
                io.Float.Input(
                    "scale_x",
                    default=1.0,
                    min=-10000.0,
                    max=10000.0,
                    step=0.01,
                    tooltip="Stretch along X. 1.0 leaves it, 2.0 doubles it, -1.0 mirrors it.",
                ),
                io.Float.Input(
                    "scale_y",
                    default=1.0,
                    min=-10000.0,
                    max=10000.0,
                    step=0.01,
                    tooltip="Stretch along Y. 1.0 leaves it, 0.5 halves its height.",
                ),
                io.Float.Input(
                    "scale_z",
                    default=1.0,
                    min=-10000.0,
                    max=10000.0,
                    step=0.01,
                    tooltip="Stretch along Z. 1.0 leaves it, 0.1 flattens it to a slab.",
                ),
                io.String.Input(
                    "name",
                    default="Transform",
                    multiline=False,
                    tooltip="Label carried into the scene graph, such as 'pedestal'. Custom code finds it by name.",
                ),
                io.Boolean.Input(
                    "visible",
                    default=True,
                    tooltip="`true` draws the object and anything under it; `false` hides all of it.",
                ),
            ],
            outputs=[
                THREE_OBJECT.Output(
                    display_name="object",
                    tooltip="The placed object, for Three Group or the root socket on Three Scene.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        object,
        position_x,
        position_y,
        position_z,
        rotation_x,
        rotation_y,
        rotation_z,
        scale_x,
        scale_y,
        scale_z,
        name,
        visible,
    ) -> io.NodeOutput:
        """Wrap and place the object.

        Raises:
            ValueError: The input is not an object descriptor.
        """
        require_spec(object, "object")
        return io.NodeOutput(
            create_spec(
                "object",
                "Group",
                params={
                    "name": name,
                    "visible": bool(visible),
                    "position": [float(position_x), float(position_y), float(position_z)],
                    "rotation": [
                        math.radians(float(rotation_x)),
                        math.radians(float(rotation_y)),
                        math.radians(float(rotation_z)),
                    ],
                    "scale": [float(scale_x), float(scale_y), float(scale_z)],
                },
                children=[object],
            )
        )

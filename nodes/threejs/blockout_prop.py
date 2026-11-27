"""A grey-box shape standing on the ground, for blocking out a shot."""

from __future__ import annotations

import math
import re

from comfy_api.latest import io

from ...modules.compat.types import THREE_OBJECT
from ...modules.threejs.spec import create_spec, require_spec

REQUIRES = "threejs"

#: Shapes in the order the menu lists them.
SHAPES = (
    "box",
    "ground",
    "person",
    "ramp",
    "stairs",
    "arch",
    "cylinder",
    "capsule",
    "sphere",
    "cone",
)

#: Marks a chain of props, so the next prop adds to it rather than nesting it.
CHAIN_META = {"blockout": True}

COLOUR = re.compile(r"^#?([0-9a-fA-F]{6})$")


class ThreeBlockoutProp(io.ComfyNode):
    """Place one blockout shape on the ground."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeBlockoutProp",
            display_name="Three Blockout Prop",
            search_aliases=[
                "WASThreeBlockoutProp",
                "Three Blockout Prop",
                "blockout",
                "grey box",
                "prop",
                "mannequin",
                "stairs",
                "ramp",
            ],
            category="WAS Suite/Three",
            description=(
                "Stand a plain shape on the ground to block out a shot: a box, a floor, a person, "
                "a ramp, stairs, an arch, a cylinder, a capsule, a sphere or a cone. Chain props "
                "one into the next to build a set, then wire the last into Three Scene."
            ),
            inputs=[
                io.Combo.Input(
                    "shape",
                    options=list(SHAPES),
                    default="box",
                    tooltip=(
                        "`ground` = a flat floor; `person` = a standing figure whose nose faces "
                        "forward; `ramp` and `stairs` rise toward the back; the others are the "
                        "plain solids."
                    ),
                ),
                io.Float.Input(
                    "width",
                    default=1.0,
                    min=0.001,
                    max=100000.0,
                    step=0.1,
                    tooltip="Size along X, in scene units. 1.0 = one metre; a person ignores it.",
                ),
                io.Float.Input(
                    "height",
                    default=1.0,
                    min=0.001,
                    max=100000.0,
                    step=0.1,
                    tooltip="Size upward, in scene units. 1.8 = a standing adult; a ground ignores it.",
                ),
                io.Float.Input(
                    "depth",
                    default=1.0,
                    min=0.001,
                    max=100000.0,
                    step=0.1,
                    tooltip="Size along Z, front to back, in scene units. 1.0 = one metre; 0.2 = a thin wall; a person ignores it.",
                ),
                io.Float.Input(
                    "position_x",
                    default=0.0,
                    min=-100000.0,
                    max=100000.0,
                    step=0.1,
                    tooltip="Where its centre stands along X. 0.0 = the origin; positive is right of a camera at +Z.",
                ),
                io.Float.Input(
                    "position_z",
                    default=0.0,
                    min=-100000.0,
                    max=100000.0,
                    step=0.1,
                    tooltip="Where its centre stands along Z. 0.0 = the origin; positive is toward a camera at +Z.",
                ),
                io.Float.Input(
                    "turn",
                    default=0.0,
                    min=-36000.0,
                    max=36000.0,
                    step=1.0,
                    tooltip="Degrees it turns about the upward axis. 0.0 faces +Z; 90.0 faces +X; 180.0 faces away.",
                ),
                io.String.Input(
                    "colour",
                    default="#a0a0a0",
                    tooltip="Its colour as #rrggbb. #a0a0a0 = blockout grey; #c8a070 = a warm wood.",
                ),
                io.Float.Input(
                    "elevation",
                    default=0.0,
                    min=-100000.0,
                    max=100000.0,
                    step=0.1,
                    tooltip="How far its base sits above the ground, in scene units. 0.0 = on the ground.",
                    advanced=True,
                ),
                io.Int.Input(
                    "steps",
                    default=6,
                    min=1,
                    max=200,
                    tooltip="Number of steps when shape is stairs. 6 = a short flight; other shapes ignore it.",
                    advanced=True,
                ),
                THREE_OBJECT.Input(
                    "objects",
                    optional=True,
                    tooltip=(
                        "Props and objects placed before this one, from another Three Blockout Prop "
                        "or any Three object. This prop is added beside them."
                    ),
                ),
            ],
            outputs=[
                THREE_OBJECT.Output(
                    display_name="objects",
                    tooltip=(
                        "This prop with everything wired into objects, for Three Scene, Three Group "
                        "or the next Three Blockout Prop."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        shape,
        width,
        height,
        depth,
        position_x,
        position_z,
        turn,
        colour,
        elevation=0.0,
        steps=6,
        objects=None,
    ) -> io.NodeOutput:
        """Describe the prop, gathered with the props before it.

        Raises:
            ValueError: The shape is unknown, the colour is not #rrggbb, or objects is not an object.
        """
        if shape not in SHAPES:
            raise ValueError(f"Three Blockout Prop has no shape {shape!r}. Choose one of: {', '.join(SHAPES)}.")
        found = COLOUR.match(str(colour).strip())
        if not found:
            raise ValueError(
                f"Three Blockout Prop was given the colour {colour!r}. Write it as #rrggbb, such as #a0a0a0."
            )
        prop = create_spec(
            "object",
            "BlockoutProp",
            params={
                "shape": shape,
                "size": [float(width), float(height), float(depth)],
                "position": [float(position_x), float(elevation), float(position_z)],
                "turn": math.radians(float(turn)),
                "colour": "#" + found.group(1).lower(),
                "steps": int(steps),
            },
        )
        children = []
        if objects is not None:
            require_spec(objects, "object")
            chained = objects.get("type") == "Group" and objects.get("meta") == CHAIN_META
            children = list(objects.get("children") or []) if chained else [objects]
        children.append(prop)
        return io.NodeOutput(
            create_spec(
                "object",
                "Group",
                params={
                    "name": "Blockout",
                    "visible": True,
                    "position": [0.0, 0.0, 0.0],
                    "rotation": [0.0, 0.0, 0.0],
                    "scale": [1.0, 1.0, 1.0],
                },
                children=children,
                meta=CHAIN_META,
            )
        )

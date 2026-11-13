"""Keep an object standing on the ground beneath it."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_OBJECT
from ...modules.threejs.spec import create_spec, require_spec

REQUIRES = "threejs"


class ThreeFollowGround(io.ComfyNode):
    """Set an object's height from the ground under it, every frame."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeFollowGround",
            display_name="Three Follow Ground",
            search_aliases=[
                "WASThreeFollowGround",
                "Three Follow Ground",
                "terrain",
                "snap to ground",
                "walk on ground",
                "floor",
            ],
            category="WAS Suite/Three",
            description=(
                "Keep an object standing on uneven ground as it moves, so a character walking a "
                "path over rocks and dips neither floats nor sinks. Wire the walker into object "
                "and the floor model into ground; the height under it is read every frame."
            ),
            inputs=[
                THREE_OBJECT.Input(
                    "object",
                    tooltip="The object to keep on the ground, such as a walking character. It is wrapped rather than changed.",
                ),
                THREE_OBJECT.Input(
                    "ground",
                    tooltip=(
                        "The floor to stand on, such as a terrain or cave model. Wire the same "
                        "object into Three Group or Three Scene as well, so it is in the scene to "
                        "be found."
                    ),
                ),
                io.Float.Input(
                    "offset",
                    default=0.0,
                    min=-100.0,
                    max=100.0,
                    step=0.01,
                    tooltip="Height above the ground. 0.0 stands on it; 0.05 lifts the object a little; -0.05 sinks it into soft ground.",
                ),
                io.Float.Input(
                    "footprint",
                    default=0.7,
                    min=0.05,
                    max=20.0,
                    step=0.05,
                    tooltip=(
                        "Width of the patch of ground the height is read from. 0.7 suits a person "
                        "and steps over pits and pebbles; 0.2 follows every bump; 3.0 suits a "
                        "vehicle."
                    ),
                ),
            ],
            outputs=[
                THREE_OBJECT.Output(
                    display_name="grounded",
                    tooltip="The object kept on the ground, for Three Group or the root socket on Three Scene.",
                ),
            ],
        )

    @classmethod
    def execute(cls, object, ground, offset, footprint) -> io.NodeOutput:
        """Describe the grounding.

        Raises:
            ValueError: An input is not an object descriptor.
        """
        require_spec(object, "object")
        require_spec(ground, "object")
        return io.NodeOutput(
            create_spec(
                "object",
                "FollowGround",
                params={
                    "groundId": ground.get("id", ""),
                    "offset": float(offset),
                    "footprint": max(float(footprint), 0.05),
                },
                children=[object],
            )
        )

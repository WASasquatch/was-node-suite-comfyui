"""How a camera follows or aims at an object in the scene."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_OBJECT, THREE_TRACK
from ...modules.threejs.spec import create_spec, require_spec

REQUIRES = "threejs"

#: What the tracker does, in the order the menu lists them.
MODES = ("aim", "follow", "aim and follow")


class ThreeTrack(io.ComfyNode):
    """Point a camera at an object, or carry it along."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeTrack",
            display_name="Three Track",
            search_aliases=[
                "WASThreeTrack",
                "Three Track",
                "look at",
                "follow",
                "target",
                "constraint",
            ],
            category="WAS Suite/Three",
            description=(
                "Aim a camera at an object, carry the camera along with it, or both, so a "
                "moving subject stays framed. Wire the object in and the result into a camera's "
                "track socket; it follows the object already in the scene, motion included, "
                "without copying it."
            ),
            inputs=[
                THREE_OBJECT.Input(
                    "target",
                    tooltip=(
                        "The object to track. Wire the same object into Three Group or Three "
                        "Scene as well, so it is actually in the scene to be found."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=list(MODES),
                    default="aim",
                    tooltip=(
                        "`aim` turns to face the object and stays put. `follow` moves with it "
                        "and keeps facing the way it was. `aim and follow` does both, which is "
                        "a camera rigged to the subject."
                    ),
                ),
                io.Float.Input(
                    "offset_x",
                    default=0.0,
                    min=-10000.0,
                    max=10000.0,
                    step=0.1,
                    tooltip="Where the camera sits relative to the object, across. 0.0 is level with it.",
                ),
                io.Float.Input(
                    "offset_y",
                    default=0.0,
                    min=-10000.0,
                    max=10000.0,
                    step=0.1,
                    tooltip="Where it sits above the object. 2.0 looks down on it, -2.0 up at it.",
                ),
                io.Float.Input(
                    "offset_z",
                    default=0.0,
                    min=-10000.0,
                    max=10000.0,
                    step=0.1,
                    tooltip="How far back it sits. 8.0 trails the object by eight units.",
                ),
                io.Float.Input(
                    "aim_offset_y",
                    default=0.0,
                    min=-10000.0,
                    max=10000.0,
                    step=0.1,
                    tooltip=(
                        "Raises the point it aims at, above the object's own middle. 1.5 aims "
                        "at the head of a figure whose middle is at the waist."
                    ),
                ),
                io.Float.Input(
                    "damping",
                    default=0.0,
                    min=0.0,
                    max=0.99,
                    step=0.01,
                    tooltip=(
                        "How much the camera lags behind. 0.0 is locked to the object, 0.85 "
                        "drifts after it and smooths a jittery subject."
                    ),
                ),
            ],
            outputs=[
                THREE_TRACK.Output(
                    display_name="track",
                    tooltip="The tracking, for a camera's track socket.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls, target, mode, offset_x, offset_y, offset_z, aim_offset_y, damping
    ) -> io.NodeOutput:
        """Describe the tracking.

        Raises:
            ValueError: The target is not an object descriptor.
        """
        require_spec(target, "object")
        return io.NodeOutput(
            create_spec(
                "track",
                "Track",
                params={
                    "mode": mode,
                    "offset": [float(offset_x), float(offset_y), float(offset_z)],
                    "aimOffsetY": float(aim_offset_y),
                    "damping": float(damping),
                    # The id alone is what finds the object; the descriptor is not rebuilt.
                    "targetId": target.get("id", ""),
                },
            )
        )

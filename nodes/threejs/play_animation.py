"""Play an animation clip a model file was saved with."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_OBJECT
from ...modules.threejs.spec import create_spec, require_spec

REQUIRES = "threejs"

#: How the clip is timed, in the order the menu lists them.
UNITS = ("per second", "per capture", "per timeline")

#: What happens at the end of the clip, in the order the menu lists them.
LOOPS = ("repeat", "once", "ping pong")

#: What becomes of a clip that moves the model across the ground, in menu order.
ROOT_MOTION = ("in place", "travel", "as authored")

#: How the feet are posed, in menu order.
FEET = ("as authored", "flat when planted")


class ThreePlayAnimation(io.ComfyNode):
    """Run a model's own animation."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreePlayAnimation",
            display_name="Three Play Animation",
            search_aliases=[
                "WASThreePlayAnimation",
                "Three Play Animation",
                "animation",
                "clip",
                "skinning",
                "rig",
            ],
            category="WAS Suite/Three",
            description=(
                "Play a clip saved inside a .glb, .gltf, .dae or .fbx model, so a rigged "
                "character walks rather than standing in its bind pose. The clip runs at its "
                "authored speed or spread across the capture or the timeline. A walk can stay "
                "on the spot, so a path or a turn moves it instead."
            ),
            inputs=[
                THREE_OBJECT.Input(
                    "object",
                    tooltip="The loaded model to animate, from Three Load Model.",
                ),
                io.String.Input(
                    "clip",
                    default="",
                    multiline=False,
                    tooltip=(
                        "Which clip to play, by name as in `Walk`, or by number as in `0`. "
                        "Empty plays the first one. A name the model does not carry is an "
                        "error naming every clip it does."
                    ),
                ),
                io.Combo.Input(
                    "units",
                    options=list(UNITS),
                    default="per timeline",
                    tooltip=(
                        "'per timeline' fits `speed` passes of the clip across Three App's "
                        "loop_seconds, so the frame rate only samples it. 'per second' runs the "
                        "clip at the speed it was authored at instead. 'per capture' fits the "
                        "passes across the frames actually taken."
                    ),
                ),
                io.Float.Input(
                    "speed",
                    default=1.0,
                    min=-100.0,
                    max=100.0,
                    step=0.05,
                    tooltip=(
                        "On 'per second', the playback rate: 1.0 is the authored speed, 0.5 "
                        "half of it. On the two spread units, how many passes fill the span: "
                        "1.0 is one, 2.0 is two. Negative runs it backwards."
                    ),
                ),
                io.Float.Input(
                    "offset",
                    default=0.0,
                    min=-3600.0,
                    max=3600.0,
                    step=0.05,
                    tooltip=(
                        "Seconds into the clip the run begins at. 0.0 starts at the beginning; "
                        "give copies 0.0, 0.4 and 0.8 so a crowd does not march in step."
                    ),
                ),
                io.Combo.Input(
                    "loop",
                    options=list(LOOPS),
                    default="repeat",
                    tooltip=(
                        "`repeat` runs the clip again from the start, `once` holds the last "
                        "pose, `ping pong` runs it backwards and forwards."
                    ),
                ),
                io.Combo.Input(
                    "root_motion",
                    options=list(ROOT_MOTION),
                    default="in place",
                    tooltip=(
                        "For a clip that moves the model across the ground. `in place` walks "
                        "on the spot, keeping the sway and the rise and fall. `travel` carries "
                        "on from where each pass ended. `as authored` jumps back to the start "
                        "of every pass."
                    ),
                    optional=True,
                ),
                io.Combo.Input(
                    "feet",
                    options=list(FEET),
                    default="as authored",
                    tooltip=(
                        "`flat when planted` turns each foot and toe so a standing foot lies "
                        "as it does in the model's rest pose, for a clip made on other feet, "
                        "such as a walk on a character in heeled boots. `as authored` plays "
                        "the feet as saved."
                    ),
                    optional=True,
                ),
            ],
            outputs=[
                THREE_OBJECT.Output(
                    display_name="animated",
                    tooltip="The model with its clip running, for Three Group or Three Scene.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        object,
        clip,
        units,
        speed,
        offset,
        loop,
        root_motion="in place",
        feet="as authored",
    ) -> io.NodeOutput:
        """Describe the clip to play.

        Raises:
            ValueError: The input is not an object descriptor.
        """
        require_spec(object, "object")
        return io.NodeOutput(
            create_spec(
                "object",
                "ClipGroup",
                params={
                    "clip": str(clip).strip(),
                    "units": units,
                    "speed": float(speed),
                    "offset": float(offset),
                    "loop": loop,
                    "root_motion": root_motion if root_motion in ROOT_MOTION else "in place",
                    "feet": feet if feet in FEET else "as authored",
                },
                children=[object],
            )
        )

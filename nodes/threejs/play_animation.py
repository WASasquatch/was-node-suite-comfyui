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
                "authored speed, across the captured frames, or across Three App's "
                "loop_seconds, as `units` chooses."
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
            ],
            outputs=[
                THREE_OBJECT.Output(
                    display_name="animated",
                    tooltip="The model with its clip running, for Three Group or Three Scene.",
                ),
            ],
        )

    @classmethod
    def execute(cls, object, clip, units, speed, offset, loop) -> io.NodeOutput:
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
                },
                children=[object],
            )
        )

"""Steadying a per-frame effect along the motion of the clip it was applied to."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import MOTION

logger = log.get_logger("nodes.video_temporal_consistency")

NODE_NAME = "Video Temporal Consistency"

#: What both sides of the comparison on the node are written under in the temp folder.
PREFIX = "was.temporal_consistency"


class VideoTemporalConsistency(io.ComfyNode):
    """Carry a per-frame effect along a clip's motion so it stops flickering."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASVideoTemporalConsistency",
            display_name=NODE_NAME,
            search_aliases=[
                "WASVideoTemporalConsistency", NODE_NAME,
                "Image Temporal Consistency",
                "temporal consistency",
                "deflicker effect",
                "stabilize style",
                "flicker",
            ],
            category="WAS Suite/Animation",
            is_output_node=True,
            description=(
                "Stop an effect applied frame by frame from flickering: a style filter, a "
                "grade or an upscaler run on each frame of a clip. What the effect changed is "
                "carried along the clip's own motion from the frames around it, so it sticks "
                "to the surfaces it was painted on while the clip itself moves as it did."
            ),
            inputs=[
                io.MultiType.Input(
                    "source",
                    [io.Image, io.Video],
                    tooltip=(
                        "The clip before the effect, as IMAGE frames or a VIDEO, which the "
                        "motion is read from. A VIDEO also gives the video output its frame "
                        "rate and audio."
                    ),
                ),
                io.Image.Input(
                    "processed",
                    tooltip=(
                        "The same frames after the effect, as many as source and at any size, "
                        "such as an upscaler's output."
                    ),
                ),
                io.Float.Input(
                    "strength",
                    default=0.8,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How much of each frame's effect comes from its neighbours: 0 = "
                        "processed as it is; 0.8 = default; 1.0 = holds the effect until a "
                        "surface changes."
                    ),
                ),
                MOTION.Input(
                    "motion",
                    optional=True,
                    tooltip=(
                        "Motion from Video Motion, measured from source. Left empty, it is "
                        "measured here at 768 px on the long side."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="The processed frames, held steady: same size and count as processed.",
                ),
                io.Video.Output(
                    display_name="video",
                    tooltip=(
                        "The same frames as a clip, at source's frame rate and with its audio "
                        "when source is a VIDEO, otherwise at 24 fps with none."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, source, processed, strength=0.8, motion=None) -> io.NodeOutput:
        """Steady the effect and write both sides for the comparison on the node.

        Raises:
            ValueError: The two batches hold different numbers of frames, or fewer than two,
                or the motion was measured from another clip.
        """
        import comfy.model_management

        from ...modules.image import temporal_consistency
        from ...modules.media import clip as clips

        clip = clips.open_clip(source, NODE_NAME)
        count = int(clip.frames.shape[0])
        if int(processed.shape[0]) != count:
            raise ValueError(
                f"{NODE_NAME} was given {count} source frame(s) and {int(processed.shape[0])} "
                f"processed frame(s). Connect the same frames before and after the effect."
            )
        if count < 2:
            raise ValueError(
                f"{NODE_NAME} works across frames and the clip holds {count}. Connect a clip "
                f"of two frames or more."
            )
        device = comfy.model_management.get_torch_device()
        step = clips.progress(3 * count - 1)
        motion = clips.motion_for(clip, motion, NODE_NAME, device, step)
        steadied = temporal_consistency.steady(
            clip.frames, processed, motion, float(strength), device=device, progress=step
        )
        logger.info("steadied %d frame(s) at strength %g", count, float(strength))
        answer = clips.rebuild(clip, steadied)
        ui = clips.compare(clips.rebuild(clip, processed, audio=None), answer, PREFIX)
        return io.NodeOutput(steadied, answer, ui=ui)

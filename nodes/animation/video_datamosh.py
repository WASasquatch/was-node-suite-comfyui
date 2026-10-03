"""Datamosh: a clip's pixels pushed along its own motion instead of being replaced."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import MOTION
from ...modules.image import motion_effects

logger = log.get_logger("nodes.video_datamosh")

NODE_NAME = "Video Datamosh"

#: What both sides of the comparison on the node are written under in the temp folder.
PREFIX = "was.datamosh"


class VideoDatamosh(io.ComfyNode):
    """Drag one frame's pixels along the motion of the frames after it."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASVideoDatamosh",
            display_name=NODE_NAME,
            search_aliases=[
                "WASVideoDatamosh", NODE_NAME,
                "datamosh",
                "glitch",
                "pixel melt",
                "i-frame removal",
                "compression glitch",
            ],
            category="WAS Suite/Animation",
            is_output_node=True,
            description=(
                "The datamosh glitch of a video with its keyframes removed: the picture already "
                "on screen is dragged along the clip's motion in blocks instead of being "
                "replaced, so across a cut the old shot is pushed around by the new one while "
                "its moving parts bloom through. Lower residual melts any shot."
            ),
            inputs=[
                io.Video.Input("video", tooltip="The clip, at least two frames; one with a cut moshes across it."),
                io.Int.Input(
                    "start_frame",
                    default=0,
                    min=0,
                    max=100000,
                    tooltip="First frame moshed, counting from 0; the frames before it play clean.",
                ),
                io.Float.Input(
                    "amplify",
                    default=1.0,
                    min=0.0,
                    max=8.0,
                    step=0.05,
                    tooltip="How far the motion is pushed: 1 = as it moved; 2 = twice as far, for blooms; 0 = frozen.",
                ),
                io.Float.Input(
                    "residual",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Share of each frame's own change carried in with the motion: 1 = a "
                        "shot holds together and moshes across cuts; 0.5 = smears; 0 = melts."
                    ),
                ),
                io.Float.Input(
                    "refresh",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Share of each real frame mixed back in: 0 = full mosh; 0.1 = slowly recovers; 1 = no mosh.",
                ),
                io.Int.Input(
                    "keyframe_every",
                    default=0,
                    min=0,
                    max=10000,
                    tooltip="Frames between clean frames, counted from start_frame: 0 = never; 24 = once a second at 24 fps.",
                ),
                io.Int.Input(
                    "block_size",
                    default=16,
                    min=0,
                    max=128,
                    tooltip="Side of the blocks the picture moves in: 16 = codec-like; 4 = finer; 0 = smooth per pixel.",
                ),
                io.Combo.Input(
                    "at_cuts",
                    options=list(motion_effects.AT_CUTS),
                    default="carry",
                    tooltip="'carry' pushes the old shot around with the new one's motion; 'reset' starts clean at every cut.",
                ),
                MOTION.Input(
                    "motion",
                    optional=True,
                    tooltip=(
                        "Motion from Video Motion, measured from this same clip. Left empty, it "
                        "is measured here at 768 px on the long side."
                    ),
                ),
            ],
            outputs=[
                io.Video.Output(
                    display_name="video",
                    tooltip="The moshed clip: same size, length, frame rate and audio.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        video,
        start_frame=0,
        amplify=1.0,
        residual=1.0,
        refresh=0.0,
        keyframe_every=0,
        block_size=16,
        at_cuts="carry",
        motion=None,
    ) -> io.NodeOutput:
        """Mosh the clip and write both sides for the comparison on the node.

        Raises:
            ValueError: The clip holds fewer than two frames, or the motion was measured from
                another clip.
        """
        import comfy.model_management

        from ...modules.media import clip as clips

        source = clips.open_clip(video, NODE_NAME)
        count = int(source.frames.shape[0])
        if count < 2:
            raise ValueError(
                f"{NODE_NAME} follows motion between frames and the clip holds {count}. "
                f"Connect a clip of two frames or more."
            )
        device = comfy.model_management.get_torch_device()
        step = clips.progress(2 * count - 1)
        motion = clips.motion_for(source, motion, NODE_NAME, device, step)
        moshed = motion_effects.datamosh(
            source.frames, motion, int(start_frame), float(amplify), float(refresh),
            int(keyframe_every), int(block_size), str(at_cuts), float(residual), device, step,
        )
        logger.info("moshed %s from frame %d", motion.describe(), int(start_frame))
        answer = clips.rebuild(source, moshed, alpha=source.alpha)
        return io.NodeOutput(answer, ui=clips.compare(video, answer, PREFIX))

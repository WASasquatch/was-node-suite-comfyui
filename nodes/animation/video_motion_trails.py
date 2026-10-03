"""Fading trails left behind everything that moves in a clip."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import MOTION
from ...modules.image import motion_effects

logger = log.get_logger("nodes.video_motion_trails")

NODE_NAME = "Video Motion Trails"

#: What both sides of the comparison on the node are written under in the temp folder.
PREFIX = "was.motion_trails"


class VideoMotionTrails(io.ComfyNode):
    """Leave echoes or a smear behind whatever moves."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASVideoMotionTrails",
            display_name=NODE_NAME,
            search_aliases=[
                "WASVideoMotionTrails", NODE_NAME,
                "echo",
                "ghosting",
                "light trails",
                "afterimage",
                "onion skin",
            ],
            category="WAS Suite/Animation",
            is_output_node=True,
            description=(
                "Leave a fading trail behind everything that moves: a continuous smear, or "
                "echoes spaced a few frames apart. The background's own motion, a pan and its "
                "parallax, leaves none, so a moving shot trails only its subjects. Audio and "
                "frame rate carry through."
            ),
            inputs=[
                io.Video.Input("video", tooltip="The clip, at least two frames."),
                io.Float.Input(
                    "length",
                    default=8.0,
                    min=0.5,
                    max=240.0,
                    step=0.5,
                    tooltip="Frames a trail takes to fade: 3 = a short tail; 8 = default; 30 = long streaks.",
                ),
                io.Int.Input(
                    "spacing",
                    default=1,
                    min=1,
                    max=120,
                    tooltip="Frames between echoes: 1 = a continuous smear; 4 = separate copies.",
                ),
                io.Float.Input(
                    "opacity",
                    default=0.6,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="How strongly the trail shows: 0 = none; 0.6 = default; 1 = solid.",
                ),
                io.Float.Input(
                    "threshold",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=0.1,
                    tooltip="Speed, in pixels per frame, that leaves a trail: 0.5 = slight motion too; 4 = only fast action.",
                ),
                io.Combo.Input(
                    "blend",
                    options=list(motion_effects.BLENDS),
                    default="normal",
                    tooltip=(
                        "How the trail is laid over the frame: 'normal' paints it; 'lighten' "
                        "keeps only what is brighter, for light trails; 'add' glows."
                    ),
                ),
                io.Boolean.Input(
                    "ignore_camera",
                    default=True,
                    tooltip=(
                        "`true` = a pan, its parallax included, leaves no trail; `false` = "
                        "everything the camera sweeps past trails."
                    ),
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
                    tooltip="The clip with its trails: same size, length, frame rate and audio.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        video,
        length=8.0,
        spacing=1,
        opacity=0.6,
        threshold=1.0,
        blend="normal",
        ignore_camera=True,
        motion=None,
    ) -> io.NodeOutput:
        """Draw the trails and write both sides for the comparison on the node.

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
        trailed = motion_effects.trails(
            source.frames, motion, float(length), int(spacing), float(opacity),
            float(threshold), str(blend), bool(ignore_camera), device, step,
        )
        logger.info("trailed %s", motion.describe())
        answer = clips.rebuild(source, trailed, alpha=source.alpha)
        return io.NodeOutput(answer, ui=clips.compare(video, answer, PREFIX))

"""A mask of whatever moves in a clip, apart from the camera's own movement."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import MOTION

logger = log.get_logger("nodes.video_motion_mask")

NODE_NAME = "Video Motion Mask"

#: What both sides of the comparison on the node are written under in the temp folder.
PREFIX = "was.motion_mask"

#: Colour the moving area is tinted on the node's comparison.
TINT = (1.0, 0.25, 0.2)


class VideoMotionMask(io.ComfyNode):
    """Mask every pixel that moves faster than a threshold, frame by frame."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASVideoMotionMask",
            display_name=NODE_NAME,
            search_aliases=[
                "WASVideoMotionMask", NODE_NAME,
                "moving mask",
                "motion segmentation",
                "motion detection",
                "foreground from motion",
            ],
            category="WAS Suite/Animation",
            is_output_node=True,
            description=(
                "Mask whatever moves in a clip, frame by frame, with the background's own "
                "motion taken away first, the camera's pan and the parallax it causes, so only "
                "the subjects light up. For feeding SAM, inpainting or any effect that should "
                "touch only what moves."
            ),
            inputs=[
                io.Video.Input("video", tooltip="The clip, at least two frames."),
                io.Float.Input(
                    "threshold",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=0.1,
                    tooltip=(
                        "Speed, in pixels per frame, that counts as moving: 0.5 = catches "
                        "slight motion; 1.0 = default; 4.0 = only fast action."
                    ),
                ),
                io.Boolean.Input(
                    "ignore_camera",
                    default=True,
                    tooltip=(
                        "`true` = the background's motion, a pan and its parallax included, is "
                        "taken away, so a moving shot masks only its subjects; `false` = "
                        "everything the camera sweeps past counts."
                    ),
                ),
                io.Int.Input(
                    "grow",
                    default=4,
                    min=0,
                    max=256,
                    tooltip="Pixels the mask is widened by: 0 = as measured; 4 = default; 16 = generous.",
                ),
                io.Float.Input(
                    "feather",
                    default=2.0,
                    min=0.0,
                    max=64.0,
                    step=0.5,
                    tooltip="Softness of the mask's edge in pixels: 0 = hard; 2 = default; 8 = soft.",
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
                io.Mask.Output(
                    display_name="mask",
                    tooltip="White where something moves, one mask per frame at the clip's size.",
                ),
                io.Video.Output(
                    display_name="video",
                    tooltip=(
                        "The clip with what moves tinted red, for saving or showing elsewhere: "
                        "same size, length, frame rate and audio."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, video, threshold=1.0, ignore_camera=True, grow=4, feather=2.0, motion=None) -> io.NodeOutput:
        """Build the mask, tint the clip with it, and write both sides for the comparison.

        Raises:
            ValueError: The clip holds fewer than two frames, or the motion was measured from
                another clip.
        """
        import torch

        import comfy.model_management

        from ...modules.image import motion_effects
        from ...modules.media import clip as clips

        source = clips.open_clip(video, NODE_NAME)
        count, height, width = (int(v) for v in source.frames.shape[:3])
        if count < 2:
            raise ValueError(
                f"{NODE_NAME} looks for motion between frames and the clip holds {count}. "
                f"Connect a clip of two frames or more."
            )
        device = comfy.model_management.get_torch_device()
        step = clips.progress(2 * count - 1)
        motion = clips.motion_for(source, motion, NODE_NAME, device, step)

        masks = torch.empty(count, height, width)
        tinted = torch.empty_like(source.frames[..., :3])
        tint = torch.tensor(TINT, device=device).view(1, 3, 1, 1)
        for index in range(count):
            mask = motion_effects.moving_mask(
                motion, index, (height, width), float(threshold), bool(ignore_camera),
                int(grow), float(feather), device,
            )
            masks[index] = mask[0, 0].cpu()
            frame = source.frames[index:index + 1, ..., :3].to(device=device, dtype=torch.float32)
            frame = frame.permute(0, 3, 1, 2)
            shaded = frame + 0.6 * mask * (tint - frame)
            tinted[index] = shaded[0].permute(1, 2, 0).to(tinted.dtype).to(tinted.device)
            step(1)

        covered = float(masks.mean())
        logger.info("masked %s, %.1f%% moving", motion.describe(), 100.0 * covered)
        shown = clips.rebuild(source, tinted)
        return io.NodeOutput(masks, shown, ui=clips.compare(video, shown, PREFIX))

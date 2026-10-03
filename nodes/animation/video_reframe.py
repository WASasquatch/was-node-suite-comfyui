"""Reframing a clip to another aspect, following its subject."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import MOTION
from ...modules.media import reframe

logger = log.get_logger("nodes.video_reframe")

NODE_NAME = "Video Reframe"

#: What both sides of the comparison on the node are written under in the temp folder.
PREFIX = "was.reframe"

#: Long side, in pixels, cuts are looked for at when no motion is wired.
CUT_SIDE = 384


class VideoReframe(io.ComfyNode):
    """Crop a clip to another aspect, the window following the subject."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASVideoReframe",
            display_name=NODE_NAME,
            search_aliases=[
                "WASVideoReframe", NODE_NAME,
                "auto reframe",
                "vertical video",
                "9:16",
                "crop to aspect",
                "follow subject",
            ],
            category="WAS Suite/Animation",
            is_output_node=True,
            description=(
                "Crop a clip to another aspect, such as 9:16 for a phone, with the window "
                "following the subject a mask marks along a smoothed path, each scene on its "
                "own. Without a mask the window stays centred. The crop keeps the source's own "
                "pixels, so a 9:16 crop of a 1216x688 clip is 386x688."
            ),
            inputs=[
                io.Video.Input("video", tooltip="The clip to reframe."),
                io.Combo.Input(
                    "aspect",
                    options=list(reframe.ASPECTS),
                    default="9:16",
                    tooltip="Width to height of the crop: '9:16' = phone; '4:5' = portrait post; '1:1' = square.",
                ),
                io.Float.Input(
                    "smoothing",
                    default=0.5,
                    min=0.0,
                    max=10.0,
                    step=0.05,
                    tooltip="Seconds of movement averaged in the path: 0 = locked to the subject; 0.5 = default; 2 = a slow, steady follow.",
                ),
                io.Float.Input(
                    "zoom",
                    default=1.0,
                    min=1.0,
                    max=4.0,
                    step=0.05,
                    tooltip="How much tighter than the largest crop that fits: 1.0 = full height; 1.5 = closer.",
                ),
                io.Combo.Input(
                    "follow",
                    options=list(reframe.FOLLOW),
                    default="largest",
                    tooltip=(
                        "What of the mask the window follows: 'largest' = its biggest region, "
                        "held from frame to frame, for one subject among several; 'everything' = "
                        "the centre of all of it."
                    ),
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip=(
                        "The subject per frame, white on the subject, such as from SAM 3, or one "
                        "mask for the whole clip. A frame where it is empty keeps the last "
                        "position."
                    ),
                ),
                MOTION.Input(
                    "motion",
                    optional=True,
                    tooltip=(
                        "Motion from Video Motion, measured from this same clip at any size, whose cuts reset "
                        "the path. Left empty, cuts are found here."
                    ),
                ),
            ],
            outputs=[
                io.Video.Output(
                    display_name="video",
                    tooltip="The reframed clip at the crop's size, with the source's frame rate and audio.",
                ),
                io.Video.Output(
                    display_name="preview",
                    tooltip=(
                        "The source at its own size with the window outlined and the rest dimmed, "
                        "for showing where the frame travels."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, video, aspect="9:16", smoothing=0.5, zoom=1.0, follow="largest", mask=None, motion=None) -> io.NodeOutput:
        """Reframe the clip and write both sides for the comparison on the node.

        Raises:
            ValueError: The mask holds a different number of frames than the clip, or the motion
                was measured from another clip.
        """
        import torch

        import comfy.model_management

        from ...modules.image import motion as motion_field
        from ...modules.media import clip as clips

        source = clips.open_clip(video, NODE_NAME)
        count, height, width = (int(v) for v in source.frames.shape[:3])
        if mask is not None and int(mask.shape[0]) not in (1, count):
            raise ValueError(
                f"The mask batch holds {int(mask.shape[0])} frame(s) and the clip {count}. "
                f"Connect one mask per frame of the clip, or a single one for all of it."
            )
        device = comfy.model_management.get_torch_device()
        step = clips.progress(2 * count - 1)
        if count > 1:
            if motion is None:
                motion = motion_field.measure(source.frames, CUT_SIDE, device, step)
            else:
                motion = clips.motion_for(source, motion, NODE_NAME, device, step)
            cuts = motion.cuts()
        else:
            cuts = []
        crop = reframe.crop_size(height, width, reframe.ASPECTS[str(aspect)], float(zoom))
        path = reframe.follow(mask, count, (height, width), crop, cuts, float(smoothing) * float(source.rate), str(follow))

        framed = torch.empty((count, crop[0], crop[1], source.frames.shape[-1]), dtype=source.frames.dtype)
        shown = torch.empty_like(source.frames)
        alpha = torch.empty((count, crop[0], crop[1]), dtype=source.alpha.dtype) if source.alpha is not None else None
        for index, (left, top) in enumerate(path):
            planes = source.frames[index:index + 1].to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)
            framed[index] = reframe.cropped(planes, left, top, crop)[0].permute(1, 2, 0).to(framed.dtype).cpu()
            shown[index] = reframe.marked(planes, left, top, crop)[0].permute(1, 2, 0).to(shown.dtype).cpu()
            if alpha is not None:
                plane = source.alpha[index:index + 1].to(device=device, dtype=torch.float32).unsqueeze(1)
                alpha[index] = reframe.cropped(plane, left, top, crop)[0, 0].to(alpha.dtype).cpu()
            step(1)

        logger.info("reframed %d frame(s) to %dx%d", count, crop[1], crop[0])
        answer = clips.rebuild(source, framed, alpha=alpha)
        preview = clips.rebuild(source, shown)
        return io.NodeOutput(answer, preview, ui=clips.compare(video, preview, PREFIX))

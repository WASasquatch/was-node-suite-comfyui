"""Film motion blur added to a video along the motion measured between its own frames."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LIST, MOTION
from ...modules.image import motion_blur

logger = log.get_logger("nodes.video_motion_blur")

NODE_NAME = "Video Motion Blur"

#: What both sides of the comparison on the node are written under in the temp folder.
PREFIX = "was.motion_blur"


class VideoMotionBlur(io.ComfyNode):
    """Blur a video along its own motion, as a film shutter records it."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASVideoMotionBlur",
            display_name=NODE_NAME,
            search_aliases=[
                "WASVideoMotionBlur", NODE_NAME,
                "motion blur",
                "vector motion blur",
                "shutter angle",
                "180 degree shutter",
                "film blur",
                "speed ramp",
                "optical flow",
            ],
            category="WAS Suite/Animation",
            is_output_node=True,
            description=(
                "Add the motion blur a film camera records, drawn along the motion measured "
                "between the video's own frames, so a crisp or strobing clip moves like "
                "filmed footage. A subject mask or a depth batch keeps a subject in front "
                "of what streaks past behind it. Audio and frame rate carry through."
            ),
            inputs=[
                io.Video.Input(
                    "video",
                    tooltip=(
                        "The clip to blur, at least two frames. Frame rate and audio pass "
                        "through unchanged."
                    ),
                ),
                io.Float.Input(
                    "shutter_angle",
                    default=180.0,
                    min=0.0,
                    max=motion_blur.MAX_SHUTTER,
                    step=1.0,
                    tooltip=(
                        "Exposure per frame in degrees: 0 = no blur; 90 = crisp action; "
                        "180 = the film standard; 360 = the whole frame interval; above 360 = "
                        "exaggerated streaks."
                    ),
                ),
                io.Int.Input(
                    "samples",
                    default=32,
                    min=4,
                    max=motion_blur.MAX_SAMPLES,
                    tooltip=(
                        "Points read along each pixel's path: 16 = quick look; 32 = smooth "
                        "streaks up to about 30 px; 96 = long streaks without stepping."
                    ),
                ),
                io.Combo.Input(
                    "blur_layers",
                    options=list(motion_blur.LAYERS),
                    default="all",
                    tooltip=(
                        "With a mask wired: 'all' blurs everything; 'background' keeps the "
                        "subject sharp; 'subject' blurs only the subject. Without a mask "
                        "everything is blurred."
                    ),
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip=(
                        "Subject mask per frame, white on the subject, such as from SAM 3. "
                        "The subject stays in front: background streaks pass behind it and "
                        "its own blur spreads over the background."
                    ),
                ),
                io.Image.Input(
                    "depth",
                    optional=True,
                    tooltip=(
                        "Depth per frame, white nearest, such as from Depth Anything. Decides "
                        "which surface passes in front where two motions meet."
                    ),
                ),
                MOTION.Input(
                    "motion",
                    optional=True,
                    tooltip=(
                        "Motion from Video Motion, measured from this same clip, so one "
                        "measurement serves several nodes. Left empty, it is measured here at "
                        "768 px on the long side."
                    ),
                ),
                LIST.Input(
                    "shutter_curve",
                    optional=True,
                    tooltip=(
                        "Shutter angles across the clip for a speed ramp, such as the values of "
                        "Curve to Numbers: [90, 360, 90] opens up mid-clip. Stretched to the "
                        "clip's length; replaces shutter_angle."
                    ),
                ),
            ],
            outputs=[
                io.Video.Output(
                    display_name="video",
                    tooltip="The blurred clip: same size, length, frame rate and audio.",
                ),
                io.Image.Output(
                    display_name="motion_preview",
                    tooltip=(
                        "The blur each frame was given, at the size motion was measured: hue "
                        "is direction, brightness is length, black is none."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        video,
        shutter_angle=180.0,
        samples=32,
        blur_layers="all",
        mask=None,
        depth=None,
        motion=None,
        shutter_curve=None,
    ) -> io.NodeOutput:
        """Blur the clip and write both sides for the comparison on the node.

        Raises:
            ValueError: The clip holds fewer than two frames, a mask or depth batch holds a
                different number of frames than the clip, the motion was measured from
                another clip, or the shutter curve holds something other than numbers.
        """
        import torch

        import comfy.model_management

        from ...modules.media import clip as clips

        source = clips.open_clip(video, NODE_NAME)
        count = int(source.frames.shape[0])
        if count < 2:
            raise ValueError(
                f"{NODE_NAME} measures motion between frames and the clip holds {count}. "
                f"Connect a clip of two frames or more."
            )
        for name, batch in (("mask", mask), ("depth", depth)):
            if batch is not None and int(batch.shape[0]) not in (1, count):
                raise ValueError(
                    f"The {name} batch holds {int(batch.shape[0])} frame(s) and the clip "
                    f"{count}. Connect one {name} per frame of the clip, or a single one "
                    f"for all of it."
                )
        shutter = float(shutter_angle)
        if shutter_curve is not None:
            try:
                shutter = [float(value) for value in shutter_curve]
            except (TypeError, ValueError):
                raise ValueError(
                    f"{NODE_NAME}'s shutter_curve holds something other than numbers. Connect "
                    f"the values of Curve to Numbers, or a list of angles in degrees."
                ) from None
            if not shutter:
                raise ValueError(
                    f"{NODE_NAME}'s shutter_curve is empty. Connect at least one angle, or "
                    f"disconnect it to use shutter_angle."
                )

        frames = source.frames
        if source.alpha is not None:
            frames = torch.cat([frames, source.alpha.to(frames.dtype).unsqueeze(-1)], -1)

        device = comfy.model_management.get_torch_device()
        step = clips.progress(2 * count - 1)
        motion = clips.motion_for(source, motion, NODE_NAME, device, step)
        blurred, pictures = motion_blur.blur_frames(
            frames,
            shutter=shutter,
            samples=int(samples),
            mask=mask,
            depth=depth,
            layers=str(blur_layers),
            device=device,
            progress=step,
            motion=motion,
        )
        alpha = None
        if source.alpha is not None:
            blurred, alpha = blurred[..., :-1].contiguous(), blurred[..., -1].contiguous()

        answer = clips.rebuild(source, blurred, alpha=alpha)
        logger.info("blurred %s at %s", motion.describe(), (
            f"{shutter:g} degrees" if isinstance(shutter, float) else f"{len(shutter)} curve point(s)"
        ))
        return io.NodeOutput(answer, pictures, ui=clips.compare(video, answer, PREFIX))

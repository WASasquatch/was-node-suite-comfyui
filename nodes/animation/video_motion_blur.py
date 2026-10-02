"""Film motion blur added to a video along the motion measured between its own frames."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.image import motion_blur

logger = log.get_logger("nodes.video_motion_blur")

NODE_NAME = "Video Motion Blur"

#: What each side of the comparison on the node is written under in the temp folder.
PREFIX_BEFORE = "was.motion_blur.before"
PREFIX_AFTER = "was.motion_blur.after"


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
                "optical flow",
            ],
            category="WAS Suite/Animation",
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
                io.Int.Input(
                    "motion_resolution",
                    default=motion_blur.MOTION_SIDE,
                    min=0,
                    max=4096,
                    tooltip=(
                        "Long side, in pixels, the motion is measured at: 0 = the video's "
                        "own size; 512 = fastest; 768 = default; 1280 = small, fine motion."
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
            ],
            outputs=[
                io.Video.Output(
                    display_name="video",
                    tooltip="The blurred clip: same size, length, frame rate and audio.",
                ),
                io.Image.Output(
                    display_name="motion",
                    tooltip=(
                        "The measured motion per frame at motion_resolution: hue is "
                        "direction, brightness is blur length, black is still."
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
        motion_resolution=motion_blur.MOTION_SIDE,
        blur_layers="all",
        mask=None,
        depth=None,
    ) -> io.NodeOutput:
        """Blur the clip and write both sides for the comparison on the node.

        Raises:
            ValueError: The clip holds fewer than two frames, or a mask or depth batch holds
                a different number of frames than the clip.
        """
        import torch

        import comfy.model_management
        import comfy.utils
        from comfy_api.latest import InputImpl, Types

        from ...modules.media.temp_video import to_temp

        parts = video.get_components()
        frames = parts.images
        count = int(frames.shape[0])
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

        alpha = parts.alpha
        source = frames
        if alpha is not None:
            source = torch.cat([frames, alpha.to(frames.dtype).unsqueeze(-1)], -1)

        bar = comfy.utils.ProgressBar(2 * count - 1)

        def advance(steps):
            comfy.model_management.throw_exception_if_processing_interrupted()
            bar.update(steps)

        blurred, motion = motion_blur.blur_frames(
            source,
            shutter=float(shutter_angle),
            samples=int(samples),
            motion_side=int(motion_resolution),
            mask=mask,
            depth=depth,
            layers=str(blur_layers),
            device=comfy.model_management.get_torch_device(),
            progress=advance,
        )
        blurred = blurred.clamp_(0.0, 1.0)
        if alpha is not None:
            blurred, alpha = blurred[..., :-1].contiguous(), blurred[..., -1].contiguous()

        answer = InputImpl.VideoFromComponents(
            Types.VideoComponents(
                images=blurred,
                frame_rate=parts.frame_rate,
                audio=parts.audio,
                metadata=parts.metadata,
                alpha=alpha,
            ),
            bit_depth=video.get_bit_depth(),
            color_space=video.get_color_space(),
        )
        logger.info(
            "blurred %d frame(s) at a %g degree shutter, motion measured at %s",
            count, float(shutter_angle), "x".join(str(v) for v in motion.shape[1:3]),
        )

        sides = {"a_video": [], "b_video": []}
        for key, clip, prefix in (("a_video", video, PREFIX_BEFORE), ("b_video", answer, PREFIX_AFTER)):
            written = to_temp(clip, prefix)
            if written is not None:
                sides[key].append(written)
        return io.NodeOutput(answer, motion, ui=sides)

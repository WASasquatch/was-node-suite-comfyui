"""Measuring a clip's motion once, for every node that works along it."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import MOTION, MOTION_MODEL
from ...modules.image import motion as motion_field

logger = log.get_logger("nodes.video_motion")

NODE_NAME = "Video Motion"

#: What both sides of the comparison on the node are written under in the temp folder.
PREFIX = "was.motion"


class VideoMotion(io.ComfyNode):
    """Measure the motion between every pair of neighbouring frames in a clip."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASVideoMotion",
            display_name=NODE_NAME,
            search_aliases=[
                "WASVideoMotion", NODE_NAME,
                "optical flow",
                "motion vectors",
                "motion estimation",
                "flow field",
            ],
            category="WAS Suite/Animation",
            is_output_node=True,
            description=(
                "Measure how every pixel moves from frame to frame of a clip, once, so Video "
                "Motion Blur, Video Stabilize, Video Split Scenes and the other motion nodes "
                "share one measurement instead of each taking their own. With a SEA-RAFT or "
                "FlowSeek network on motion_model, that network measures it. The panel plays "
                "the clip beside its motion."
            ),
            inputs=[
                io.MultiType.Input(
                    "video",
                    [io.Video, io.Image],
                    tooltip="The clip, as a VIDEO or a batch of IMAGE frames, at least two frames.",
                ),
                io.Int.Input(
                    "motion_resolution",
                    default=motion_field.MOTION_SIDE,
                    min=0,
                    max=4096,
                    tooltip=(
                        "Long side, in pixels, the motion is measured at: 0 = the clip's own "
                        "size; 512 = fastest; 768 = default; 1280 = small, fine motion."
                    ),
                ),
                MOTION_MODEL.Input(
                    "motion_model",
                    optional=True,
                    tooltip=(
                        "A SEA-RAFT or FlowSeek network from Video Motion Model Loader to measure "
                        "the motion with. Empty = the built-in texture flow."
                    ),
                ),
            ],
            outputs=[
                MOTION.Output(
                    display_name="motion",
                    tooltip=(
                        "The measured motion, for any motion node fed this same clip, at any size. Holds "
                        "about 3 MB per frame at 768 px."
                    ),
                ),
                io.Image.Output(
                    display_name="motion_preview",
                    tooltip=(
                        "The motion per frame at the measured size: hue is direction, "
                        "brightness is speed, black is still."
                    ),
                ),
                io.Video.Output(
                    display_name="motion_video",
                    tooltip=(
                        "motion_preview as a clip at the clip's frame rate and with its audio, "
                        "for saving or showing elsewhere."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls, video, motion_resolution=motion_field.MOTION_SIDE, motion_model=None
    ) -> io.NodeOutput:
        """Measure the clip's motion and draw it.

        Raises:
            ValueError: The clip holds fewer than two frames.
        """
        import torch

        import comfy.model_management

        from ...modules.image import optical_flow
        from ...modules.media import clip as clips

        source = clips.open_clip(video, NODE_NAME)
        count = int(source.frames.shape[0])
        if count < 2:
            raise ValueError(
                f"{NODE_NAME} measures motion between frames and the clip holds {count}. "
                f"Connect a clip of two frames or more."
            )
        device = comfy.model_management.get_torch_device()
        step = clips.progress(2 * count - 1)
        motion = motion_field.measure(
            source.frames, int(motion_resolution), device, step, network=motion_model
        )

        height, width = motion.size
        pictures = torch.zeros(count, height, width, 3)
        for index in range(count):
            a, _, _ = motion.paths(index, device)
            pictures[index] = motion_field.visualise(a).cpu()
            step(1)

        cuts = sum(motion.cuts())
        logger.info("measured %s, %d cut(s)", motion.describe(), cuts)
        small = torch.empty(count, height, width, 3)
        for start in range(0, count, motion_field.PAIR_BATCH):
            chunk = source.frames[start:start + motion_field.PAIR_BATCH, ..., :3].to(device)
            resized = optical_flow.resize(chunk.permute(0, 3, 1, 2).float(), height, width)
            small[start:start + motion_field.PAIR_BATCH] = resized.permute(0, 2, 3, 1).cpu()
        before = clips.rebuild(source, small, audio=None)
        shown = clips.rebuild(source, pictures)
        return io.NodeOutput(motion, pictures, shown, ui=clips.compare(before, shown, PREFIX))

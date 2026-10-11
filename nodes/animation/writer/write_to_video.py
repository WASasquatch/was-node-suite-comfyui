"""Append images to a video file that grows across prompts."""

from __future__ import annotations

import os

from comfy_api.latest import io

from ....modules.io import rooted
from ....modules import log
from ....modules.convert.tensors import tensor2pil
from ....modules.interface import file_report
from ....modules.util import sandbox

logger = log.get_logger("nodes.animation.writer")


def rescale_image(image, max_dimension: int):
    """Shrink an image until neither edge is longer than ``max_dimension``.

    Args:
        image: Source PIL image.
        max_dimension: Longest edge in pixels. An image already within it is returned
            untouched rather than scaled up.

    Returns:
        The image, resampled with Lanczos when it was too large. Each edge is truncated to
        a whole pixel, so the result can be a pixel short of the requested size.
    """
    from PIL import Image

    width, height = image.size
    if width > max_dimension or height > max_dimension:
        scaling_factor = max(width, height) / max_dimension
        new_width = int(width / scaling_factor)
        new_height = int(height / scaling_factor)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image


class WriteToVideo(io.ComfyNode):
    """Add every image in the batch to one video file, creating it on the first run."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        from ....modules.media import video

        return io.Schema(
            node_id="Write to Video",
            display_name="Write to Video",
            search_aliases=["Write to Video", "append video", "mp4 writer", "video writer"],
            category="WAS Suite/Animation",
            description=(
                "Append the images to a video that keeps growing across prompts, fading in "
                "from the last frame already in the file. The images pass through unchanged, "
                "with the path of the video."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The images to append. A batch is appended one image at a time, in "
                        "batch order, each faded in from the one before it."
                    ),
                ),
                io.Int.Input(
                    "transition_frames",
                    default=30,
                    min=0,
                    max=120,
                    step=1,
                    tooltip=(
                        "How many blended frames are drawn between the last frame in the file "
                        "and the incoming image. 0 cuts straight to it; anything above 60 is "
                        "treated as 60. Ignored the first time, when there is nothing to fade "
                        "from."
                    ),
                ),
                io.Float.Input(
                    "image_delay_sec",
                    default=2.5,
                    min=0.1,
                    max=60000.0,
                    step=0.1,
                    tooltip=(
                        "How long each appended image is held once the fade into it has "
                        "finished, in seconds. The fraction is dropped, so 2.5 holds for 2 "
                        "seconds and anything under 1 second is held for a single frame."
                    ),
                ),
                io.Int.Input(
                    "fps",
                    default=30,
                    min=1,
                    max=60.0,
                    step=1,
                    tooltip=(
                        "Frames per second of a newly created video, which also decides how "
                        "many frames a held image lasts. Appending to a video that already "
                        "exists keeps that file's own rate instead."
                    ),
                ),
                io.Int.Input(
                    "max_size",
                    default=512,
                    min=128,
                    max=1920,
                    step=1,
                    tooltip=(
                        "Longest edge, in pixels, each image is scaled to before it is "
                        "encoded. 512 turns a 1024x768 image into 512x384. Frame size is "
                        "fixed by the first image, so later images are fitted to it."
                    ),
                ),
                io.Combo.Input(
                    "root",
                    options=rooted.options(),
                    tooltip=(
                        "Which folder the file lands in: ComfyUI's own 'output' or 'temp', "
                        "or any folder added under paths.allow_write in config.yaml, listed "
                        "by its own name. filename names the part below it, so "
                        "'[time(%Y-%m-%d)]/clip' files each day's under a dated folder."
                    ),
                ),
                io.String.Input(
                    "filename",
                    default="comfy_writer",
                    multiline=False,
                    tooltip=(
                        "Name of the video, without an extension, the codec picks one, such "
                        "as '.mp4' or '.mkv'. Every prompt using the same name appends to the "
                        "same file, so change the name, or put a token such as "
                        "[time(%Y-%m-%d)] in it, to start a new clip."
                    ),
                ),
                io.Combo.Input(
                    "codec",
                    options=video.codec_options(),
                    tooltip=(
                        "`AVC1` = .mp4, plays anywhere; `H264` = the same in .mkv; `MP4V` = "
                        "older, larger, always available; `FFV1` = lossless .mkv, very large; "
                        "`H265`, `HEVC`, `AV01` = smaller, slower; `VP90` = .webm; `PRORES` = "
                        ".mov for editing. One this machine lacks is refused."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="IMAGE_PASS",
                    tooltip="The images exactly as they arrived, so this node can sit mid-chain.",
                ),
                io.String.Output(
                    display_name="filepath_text",
                    tooltip="Full path of the video that was written, extension included.",
                ),
                io.String.Output(
                    display_name="filename_text",
                    tooltip=(
                        "The filename widget as it was typed, with no folder and no extension, "
                        "for feeding a caption or a log."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        transition_frames=30,
        image_delay_sec=2.5,
        fps=30,
        max_size=512,
        root=rooted.DEFAULT,
        filename="comfy_writer",
        codec="AVC1",
    ) -> io.NodeOutput:
        import comfy.utils

        from ....modules.media import video


        transition_frames = min(max(transition_frames, 0), 60)
        fps = min(max(fps, 1), 60)

        video.require_codec(codec)
        below, _, leaf = (filename or "").replace("\\", "/").rpartition("/")
        target = sandbox.resolve_write_file(rooted.destination(root, below), leaf)
        os.makedirs(target.parent, exist_ok=True)

        writer = video.VideoWriter(
            int(transition_frames),
            int(fps),
            int(image_delay_sec),
            max_size=max_size,
            codec=codec,
        )

        written = ""
        progress = comfy.utils.ProgressBar(len(image))
        for frame in image:
            written = writer.write(rescale_image(tensor2pil(frame), max_size), str(target))
            progress.update(1)

        logger.info("appended %s image(s) to %s", len(image), written)
        file_report.publish(
            [written] if written else [],
            intended=1,
            folder=str(target.parent),
            facts={"codec": codec, "appended": f"{len(image)} frame(s)"},
        )
        return io.NodeOutput(image, written, filename)

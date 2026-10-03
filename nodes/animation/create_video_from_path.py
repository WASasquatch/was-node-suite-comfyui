"""Render a folder of images into one video file."""

from __future__ import annotations

import os

from comfy_api.latest import io

from ...modules.io import rooted
from ...modules import log
from ...modules.util import sandbox

logger = log.get_logger("nodes.animation")


class CreateVideoFromPath(io.ComfyNode):
    """Encode every image in a folder into a video, fading between them."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        from ...modules.media import video

        return io.Schema(
            node_id="Create Video from Path",
            display_name="Create Video from Path",
            search_aliases=[
                "Create Video from Path",
                "images to video",
                "slideshow",
                "mp4 from folder",
            ],
            category="WAS Suite/Animation",
            description=(
                "Turn a folder of images into a video, holding each image and cross-fading "
                "into the next. The codec decides how the file is compressed and what "
                "extension it gets: 'AVC1' and 'H264' are the same widely playable codec in "
                "an .mp4 and an .mkv, 'MP4V' is older and larger but always available, 'FFV1' "
                "is lossless so the file is very large, 'H265', 'HEVC' and 'AV01' make "
                "smaller files and take longer to encode, 'VP90' writes a .webm and 'PRORES' "
                "a .mov for editing."
            ),
            inputs=[
                io.Int.Input(
                    "transition_frames",
                    default=30,
                    min=0,
                    max=120,
                    step=1,
                    tooltip=(
                        "How many blended frames are drawn between one image and the next. 0 "
                        "cuts straight from each image to the following one; anything above 60 "
                        "is treated as 60."
                    ),
                ),
                io.Float.Input(
                    "image_delay_sec",
                    default=2.5,
                    min=0.01,
                    max=60000.0,
                    step=0.01,
                    tooltip=(
                        "How long each image is held, in seconds. The fraction is dropped, so "
                        "2.5 holds for 2 seconds and anything under 1 holds for no time at "
                        "all, leaving only the fades."
                    ),
                ),
                io.Int.Input(
                    "fps",
                    default=30,
                    min=1,
                    max=60.0,
                    step=1,
                    tooltip=(
                        "Frames per second of the finished video, which also decides how many "
                        "frames a held image lasts: 2 seconds at 30 fps is 60 frames."
                    ),
                ),
                io.Int.Input(
                    "max_size",
                    default=512,
                    min=128,
                    max=1920,
                    step=1,
                    tooltip=(
                        "Longest edge, in pixels, the frames are scaled to. 512 turns a "
                        "1024x768 image into 512x384. The first image decides the frame size "
                        "and the others are fitted to it."
                    ),
                ),
                io.Combo.Input(
                    "input_root",
                    options=rooted.read_options(),
                    tooltip=(
                        "Which folder the source images are in: ComfyUI's own 'input', "
                        "'output' or 'temp', or any folder added under paths.allow_read in "
                        "config.yaml, listed by its own name. input_folder names the part "
                        "below it."
                    ),
                ),
                io.String.Input(
                    "input_folder",
                    default="",
                    multiline=False,
                    tooltip=(
                        "Folder below the input root holding the frames, such as "
                        "'plates/shot_01'. Empty reads the root itself."
                    ),
                ),
                io.Combo.Input(
                    "output_root",
                    options=rooted.options(),
                    tooltip=(
                        "Which folder the file lands in: ComfyUI's own 'output' or 'temp', "
                        "or any folder added under paths.allow_write in config.yaml, listed "
                        "by its own name. output_folder names the part below it."
                    ),
                ),
                io.String.Input(
                    "output_folder",
                    default="",
                    multiline=False,
                    tooltip=(
                        "Folder below the output root the file is written to, created if it "
                        "is not there. Tokens expand, so '[time(%Y-%m-%d)]' files each day's "
                        "under a dated folder."
                    ),
                ),
                io.String.Input(
                    "filename",
                    default="comfy_video",
                    multiline=False,
                    tooltip=(
                        "Name of the video, without an extension, the codec picks one, such "
                        "as '.mp4' or '.mkv'. An existing file with the same name is "
                        "overwritten, so put a token such as [time(%H-%M-%S)] in the name to "
                        "keep every run."
                    ),
                ),
                io.Combo.Input(
                    "codec",
                    options=video.codec_options(),
                    tooltip=(
                        "How the video is compressed, which also picks the file extension. A "
                        "codec this machine cannot encode is reported before anything is "
                        "written."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="filepath_text",
                    tooltip=(
                        "Full path of the video that was written, extension included. Empty "
                        "when the input folder held no images."
                    ),
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
    def fingerprint_inputs(cls, **kwargs) -> float:
        """NaN never equals itself, so the folder is re-encoded on every prompt."""
        return float("NaN")

    @classmethod
    def execute(
        cls,
        transition_frames=30,
        image_delay_sec=2.5,
        fps=30,
        max_size=512,
        input_root=rooted.READ_DEFAULT,
        input_folder="",
        output_root=rooted.DEFAULT,
        output_folder="",
        filename="comfy_video",
        codec="AVC1",
    ) -> io.NodeOutput:
        from ...modules.media import video


        transition_frames = min(max(transition_frames, 0), 60)
        fps = min(max(fps, 1), 60)

        video.require_codec(codec)
        source = rooted.source(input_root, input_folder)
        target = sandbox.resolve_write_file(
            rooted.destination(output_root, output_folder), filename
        )
        os.makedirs(target.parent, exist_ok=True)

        writer = video.VideoWriter(
            int(transition_frames), int(fps), int(image_delay_sec), max_size, codec
        )
        written = writer.create_video(str(source), str(target))

        if written:
            logger.info("video written to %s", written)
        return io.NodeOutput(written, filename)

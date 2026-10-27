"""Decode a video into numbered image files."""

from __future__ import annotations

from pathlib import Path

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import NUMBER
from ...modules.io import picker, rooted
from ...modules.media import reader
from ...modules.util import sandbox

logger = log.get_logger("nodes.animation")

#: What the video menu lists, and what it says when there is nothing to list.
NO_VIDEOS = "no video files found"


def video_options() -> list[str]:
    """The menu's entries, or a line saying there are none."""
    return picker.labels(reader.VIDEO_EXTENSIONS) or [NO_VIDEOS]


def video_path(video: str) -> str:
    """The video one menu entry names, as a path, or an empty string."""
    entry = str(video or "").strip()
    if not entry or entry == NO_VIDEOS:
        return ""
    return picker.resolve(entry, reader.VIDEO_EXTENSIONS) or ""




class VideoDumpFrames(io.ComfyNode):
    """Write every frame of a video to its own image file.

    Frames are numbered from 0 in decode order and named ``<prefix><number>.<extension>``.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Video Dump Frames",
            display_name="Video Dump Frames",
            search_aliases=[
                "Video Dump Frames",
                "extract frames",
                "video to images",
                "frame dump",
            ],
            category="WAS Suite/Animation",
            description="Save every frame of a video file as a numbered image.",
            inputs=[
                picker.path_input(io.Combo.Input(
                    "video",
                    options=video_options(),
                    tooltip=(
                        "Which video to read. The menu lists every container in ComfyUI's "
                        "input, output and temp folders and in any folder added under "
                        "paths.allow_read: .mp4, .mkv, .webm, .mov and the rest ffmpeg opens."
                    ),
                )),
                io.Combo.Input(
                    "root",
                    options=rooted.options(),
                    tooltip=(
                        "Which folder the files land in: ComfyUI's own 'output' or 'temp', "
                        "or any folder added under paths.allow_write in config.yaml, listed "
                        "by its own name. The name below it says the rest."
                    ),
                ),
                io.String.Input(
                    "folder",
                    default="frames",
                    multiline=False,
                    tooltip=(
                        "Folder below the root that the stills are written into, created if "
                        "it is not there. Tokens expand, so '[time(%Y-%m-%d)]/frames' files "
                        "each day's under a dated folder."
                    ),
                ),
                io.String.Input(
                    "prefix",
                    default="frame_",
                    multiline=False,
                    tooltip=(
                        "Text before the frame number, so 'frame_' gives 'frame_0000.png'. "
                        "Tokens are expanded here too. Leave it empty to name the files by "
                        "number alone."
                    ),
                ),
                io.Int.Input(
                    "filenumber_digits",
                    default=4,
                    min=-1,
                    max=8,
                    step=1,
                    tooltip=(
                        "How many digits the frame number is padded to: 4 gives 'frame_0001', "
                        "which keeps the files in order when they are listed. 0 or -1 writes "
                        "the bare number, so 'frame_10' sorts before 'frame_2'."
                    ),
                ),
                io.Combo.Input(
                    "extension",
                    options=["png", "jpg", "gif", "tiff"],
                    tooltip=(
                        "Image format for each frame. `png` is lossless and the safest choice; "
                        "`jpg` is much smaller but loses detail; `tiff` is lossless and large; "
                        "`gif` drops each frame to 256 colours."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="folder_written",
                    tooltip=(
                        "Full path of the folder the frames were written to, for feeding an "
                        "image batch loader."
                    ),
                ),
                NUMBER.Output(
                    display_name="processed_count",
                    tooltip=(
                        "How many frames were written, for the NUMBER inputs of the suite's "
                        "own maths nodes."
                    ),
                ),
                io.Float.Output(
                    display_name="processed_count_float",
                    tooltip="The same count as a decimal, for example 250.0.",
                ),
                io.Int.Output(
                    display_name="processed_count_int",
                    tooltip="The same count as a whole number, for a core INT input.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> float:
        """NaN never equals itself, so the video is decoded again on every prompt."""
        return float("NaN")

    @classmethod
    def execute(
        cls,
        video,
        root=rooted.DEFAULT,
        folder="frames",
        prefix="frame_",
        filenumber_digits=4,
        extension="png",
    ) -> io.NodeOutput:
        from ...modules.media import frames

        found = video_path(video)
        if not found:
            raise ValueError(
                f"`{video}` names no video that is there. Pick another from the menu, or add "
                f"its folder to paths.allow_read in config.yaml"
            )
        source = Path(found)
        destination = rooted.destination(root, folder)

        processed = frames.extract(
            str(source),
            str(destination),
            prefix,
            extension,
            int(filenumber_digits),
        )

        logger.info("wrote %s frame(s) from %s to %s", processed, source, destination)
        return io.NodeOutput(str(destination), processed, float(processed), int(processed))

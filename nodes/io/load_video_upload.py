"""Load a video picked from ComfyUI's input, output or temp folder."""

from __future__ import annotations

import os

from comfy_api.latest import io

from ...modules import log
from ...modules.compat import limits
from ...modules.compat.types import WAS_VIDEO_METADATA
from ...modules.image import sizing
from ...modules.media import reader, sampling
from .load_video import load

logger = log.get_logger("nodes.io")


class LoadVideoUpload(io.ComfyNode):
    """Load a video chosen in ComfyUI's input folder."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLoadVideoUpload",
            display_name="Load Video (Upload)",
            search_aliases=[
                "WASLoadVideoUpload",
                "Load Video (Upload)",
                "upload video",
            ],
            category="WAS Suite/IO",
            description=(
                "Load a video and hand on the video itself, its frames as an image batch, its "
                "sound and how long it is. Upload with the button and play it back on the "
                "node. Frames can be thinned by count and strategy and brought to one size."
            ),
            inputs=[
                io.Combo.Input(
                    "file",
                    options=reader.video_labels(),
                    upload=io.UploadType.video,
                    tooltip=(
                        "Which video to read, from ComfyUI's input folder. The button below "
                        "uploads one and selects it, and the player shows what is selected."
                    ),
                ),
                io.Int.Input(
                    "num_frames",
                    advanced=True,
                    default=0,
                    min=0,
                    max=reader.MAX_FRAMES,
                    tooltip=(
                        f"How many frames to keep, chosen by strategy. `0` = the whole clip, "
                        f"up to the {reader.MAX_FRAMES} ceiling; `16` = a short sample. A clip "
                        f"too long to fit in memory stops with the count to set."
                    ),
                ),
                io.Combo.Input(
                    "strategy",
                    advanced=True,
                    options=list(sampling.STRATEGIES),
                    default="uniform",
                    tooltip=(
                        "How num_frames are chosen. uniform = evenly spaced; head = first; "
                        "center = middle; tail = last; random = a seeded pick; every_nth = "
                        "every nth. uniform gives a contact sheet of a whole clip, head "
                        "gives a run that plays."
                    ),
                ),
                io.Int.Input(
                    "nth",
                    advanced=True,
                    default=1,
                    min=1,
                    max=limits.max_resolution(),
                    tooltip=(
                        "Step between the frames the strategy may choose from. 1 uses every "
                        "frame; 2 thins to every other one first, so `head` takes the opening "
                        "of the clip on alternate frames. It applies to every strategy."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    advanced=True,
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=io.ControlAfterGenerate.fixed,
                    tooltip=(
                        "Seed for random, so a re-run keeps the same frames. Ignored by the "
                        "other strategies. Any whole number; `0` is as good a seed as any. "
                        "Left on `fixed`, a re-run is served from the cache instead of the "
                        "file being read again."
                    ),
                ),
                io.Float.Input(
                    "target_fps",
                    advanced=True,
                    default=0.0,
                    min=0.0,
                    max=reader.MAX_RATE,
                    step=0.01,
                    tooltip=(
                        "Rate the frames come out at. 0 keeps the file's own. A lower rate "
                        "drops frames and a higher one repeats them, so the clip runs for "
                        "the same time either way. Set it to match a model that wants 8 or "
                        "16 fps."
                    ),
                ),
                io.Int.Input(
                    "start",
                    advanced=True,
                    default=0,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    optional=True,
                    tooltip=(
                        "First frame to consider, counting from 0 through the file's own "
                        "frames. Negative counts back from the end, so -60 starts sixty "
                        "frames before it."
                    ),
                ),
                io.Int.Input(
                    "end",
                    advanced=True,
                    default=-1,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    optional=True,
                    tooltip=(
                        "Last frame to consider, inclusive. -1 is the final frame, which is "
                        "the whole clip together with a start of 0."
                    ),
                ),
                io.Combo.Input(
                    "resize_mode",
                    advanced=True,
                    options=list(sizing.MODES),
                    default=sizing.FIT_AND_PAD,
                    tooltip=(
                        "How each frame meets the size below. `fit and pad` keeps the whole "
                        "frame and pads the rest, `fill and crop` fills the size and trims "
                        "the overhang, `stretch` distorts to fit, `crop or pad` never "
                        "resamples."
                    ),
                ),
                io.Int.Input(
                    "width",
                    advanced=True,
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=8,
                    tooltip=(
                        "Width every frame is brought to. 0 takes the width the file was "
                        "encoded at, which is what loads a clip at its own size."
                    ),
                ),
                io.Int.Input(
                    "height",
                    advanced=True,
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=8,
                    tooltip=(
                        "Height every frame is brought to. 0 takes the height the file was "
                        "encoded at."
                    ),
                ),
                io.Int.Input(
                    "max_size",
                    advanced=True,
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=8,
                    optional=True,
                    tooltip=(
                        "Longest edge the derived size is held to, keeping the aspect. Only "
                        "read when width and height are 0, which is where a 4K clip would "
                        "otherwise fill memory. 0 lifts the cap."
                    ),
                ),
                io.Combo.Input(
                    "interpolation",
                    advanced=True,
                    options=list(sizing.FILTER_NAMES),
                    default=sizing.DEFAULT_FILTER,
                    optional=True,
                    tooltip="Resampling filter. `lanczos` is the sharpest for a downscale.",
                ),
                io.Combo.Input(
                    "align",
                    advanced=True,
                    options=list(sizing.ALIGNMENT_NAMES),
                    default=sizing.DEFAULT_ALIGNMENT,
                    optional=True,
                    tooltip=(
                        "Which part of a frame survives a crop, and which side carries the "
                        "wider bar of a pad."
                    ),
                ),
                io.String.Input(
                    "pad_color",
                    advanced=True,
                    default="#000000",
                    optional=True,
                    tooltip="Fill for space a frame does not cover. Any Pillow colour.",
                ),
                io.Combo.Input(
                    "channels",
                    advanced=True,
                    options=list(sizing.CHANNELS),
                    default="RGB",
                    optional=True,
                    tooltip=(
                        "Channels the image batch carries. `RGBA` keeps the pad transparent. "
                        "The video output is always colour, since a video carries no "
                        "transparency."
                    ),
                ),
                io.String.Input(
                    "path",
                    optional=True,
                    force_input=True,
                    tooltip=(
                        "A path wired in as text, such as `E:\\clips\\take_01.mp4`, read in place of the "
                        "file menu. Only a file in a folder that menu reads is opened."
                    ),
                ),
            ],
            outputs=[
                io.Video.Output(
                    display_name="video",
                    tooltip=(
                        "The frames that were kept, with their sound, as a video at the rate "
                        "below. A whole clip too long for one batch is read from the file as "
                        "it is used. Wire it into Save Video, or into any node taking a VIDEO."
                    ),
                ),
                io.Image.Output(
                    display_name="images",
                    tooltip=(
                        "The same frames as one image batch, in playback order, every one at "
                        "the same size. Blocked where they would not fit in memory; set "
                        "num_frames or max_size in the advanced inputs to bring them down."
                    ),
                ),
                io.Audio.Output(
                    display_name="audio",
                    tooltip=(
                        "The sound playing under the frames that were kept, from where they "
                        "start and for as long as they run. Empty when the file is silent, "
                        "so read has_audio before wiring this into a save node."
                    ),
                ),
                WAS_VIDEO_METADATA.Output(
                    display_name="metadata",
                    tooltip=(
                        "What this read measured: the rate, the frame count, the size, the "
                        "duration, the bit depth and whether there is sound, beside the same "
                        "figures for the file itself. Wire it into Video Metadata to read any "
                        "of them as a number."
                    ),
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls, file, num_frames=0, strategy="uniform", nth=1, seed=0, target_fps=0.0,
        resize_mode=sizing.FIT_AND_PAD, width=0, height=0, start=0, end=-1, max_size=0,
        interpolation=sizing.DEFAULT_FILTER, align=sizing.DEFAULT_ALIGNMENT,
        pad_color="#000000", channels="RGB", path=None,
    ):
        """When the chosen file was last written, so an edit is read again."""
        # An empty name resolves to the input folder itself, which exists, so it is refused
        # before the folder is asked about it.
        if str(path or "").strip():
            file = path
        chosen = (file or "").strip()
        if not chosen:
            return float("NaN")
        try:
            return os.path.getmtime(reader.input_path(file))
        except Exception:
            return float("NaN")

    @classmethod
    def validate_inputs(cls, file=None):
        """Accept any ``file``; ``execute`` reports one that names nothing there.

        Args:
            file: The stored menu value, which a wired ``path`` replaces.

        Returns:
            True, always.
        """
        return True

    @classmethod
    def execute(
        cls, file, num_frames=0, strategy="uniform", nth=1, seed=0, target_fps=0.0,
        resize_mode=sizing.FIT_AND_PAD, width=0, height=0, start=0, end=-1, max_size=0,
        interpolation=sizing.DEFAULT_FILTER, align=sizing.DEFAULT_ALIGNMENT,
        pad_color="#000000", channels="RGB", path=None,
    ) -> io.NodeOutput:
        """Read the chosen video and hand on its frames, sound and measurements.

        Raises:
            DependencyError: PyAV is not installed.
            PathNotAllowed: The file resolved outside every permitted read root.
            ValueError: Nothing was chosen, or no frame could be decoded.
        """
        chosen = reader.input_path(path if str(path or "").strip() else file)
        return load(
            chosen, num_frames, strategy, nth, seed, target_fps, resize_mode, width, height,
            start, end, max_size, interpolation, align, pad_color, channels,
        )

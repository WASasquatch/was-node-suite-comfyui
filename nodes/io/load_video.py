"""Load a video from ComfyUI's input folder as a video, a frame batch and its sound."""

from __future__ import annotations

import os

from comfy_api.latest import io

from ...modules import log
from ...modules.compat import limits
from ...modules.compat.types import WAS_VIDEO_METADATA
from ...modules.image import sizing
from ...modules.media import reader, sampling
from ...modules.util import sandbox

logger = log.get_logger("nodes.io")


def load(
    path: str,
    num_frames: int = 0,
    strategy: str = "uniform",
    nth: int = 1,
    seed: int = 0,
    target_fps: float = 0.0,
    resize_mode: str = sizing.FIT_AND_PAD,
    width: int = 0,
    height: int = 0,
    start: int = 0,
    end: int = -1,
    max_size: int = 1024,
    interpolation: str = sizing.DEFAULT_FILTER,
    align: str = sizing.DEFAULT_ALIGNMENT,
    pad_color: str = "#000000",
    channels: str = "RGB",
) -> io.NodeOutput:
    """Read one video file and answer the four outputs both loaders publish.

    Args:
        path: The video file, already resolved inside a permitted read root.
        num_frames: How many frames to keep, 0 for every frame in the range.
        strategy: One of :data:`modules.media.sampling.STRATEGIES`.
        nth: Step between kept frames, read only by ``every_nth``.
        seed: Seed for ``random``.
        target_fps: Rate the frames are answered at, 0 to keep the file's own.
        resize_mode: One of :data:`modules.image.sizing.MODES`.
        width: Width every frame is brought to, 0 for the file's own.
        height: Height every frame is brought to, 0 for the file's own.
        start: First frame to consider, counting from 0.
        end: Last frame to consider, inclusive.
        max_size: Longest edge a derived size is held to, 0 for none.
        interpolation: A name from :data:`modules.image.sizing.FILTER_NAMES`.
        align: A name from :data:`modules.image.sizing.ALIGNMENT_NAMES`.
        pad_color: Fill for space a frame does not cover.
        channels: ``"RGB"`` or ``"RGBA"``.

    Returns:
        The video, the frame batch, the audio, and what the read measured.

    Raises:
        DependencyError: PyAV is not installed.
        ValueError: The file holds no video stream, no frame could be decoded, or the frames
            asked for do not fit in memory.
    """
    clip = reader.read(
        path,
        start=start,
        end=end,
        num_frames=num_frames,
        strategy=strategy,
        nth=nth,
        seed=seed,
        target_fps=target_fps,
        resize_mode=resize_mode,
        width=width,
        height=height,
        max_size=max_size,
        interpolation=interpolation,
        align=align,
        pad_color=pad_color,
        channels=channels,
    )

    images = clip.images
    frames = int(images.shape[0])
    frame_height, frame_width = int(images.shape[1]), int(images.shape[2])
    duration = frames / clip.fps if clip.fps > 0 else 0.0
    logger.info(
        "loaded %d of %d frame(s) from %s at %dx%d, %.6g fps, %s, %.2f s of %s",
        frames, clip.source.frame_count, os.path.basename(path), frame_width, frame_height,
        clip.fps,
        sampling.describe(strategy, nth) if num_frames else "every frame in the range",
        duration, "sound" if clip.audio is not None else "silence",
    )
    return io.NodeOutput(
        reader.to_video(images, clip.fps, clip.audio, clip.source.bit_depth),
        images,
        clip.audio,
        {
            "fps": float(clip.fps),
            "frame_count": frames,
            "duration": float(duration),
            "width": frame_width,
            "height": frame_height,
            "has_audio": clip.audio is not None,
            "bit_depth": int(clip.source.bit_depth),
            "source_fps": float(clip.source.fps),
            "source_frame_count": int(clip.source.frame_count),
            "source_duration": float(clip.source.duration),
            "source_width": int(clip.source.width),
            "source_height": int(clip.source.height),
            "filename": os.path.basename(path),
        },
    )


class LoadVideo(io.ComfyNode):
    """Load a video from ComfyUI's input folder, with the pack's selection and sizing surface."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLoadVideo",
            display_name="Load Video (Advanced)",
            search_aliases=[
                "WASLoadVideo",
                "Load Video",
                "open video",
                "video file",
                "video to images",
                "mp4",
                "frames from video",
            ],
            category="WAS Suite/IO",
            description=(
                "Load a video from ComfyUI's input folder and hand on everything in it at "
                "once: the video itself, its frames as an image batch, its sound, and how "
                "long it is. Upload a file with the button on the node and play it back "
                "there. Frames are chosen with the same range and strategy controls the "
                "frame samplers use, and brought to one size the same way the image loaders "
                "do it. 16 frames are taken unless told otherwise, since a clip can hold "
                "thousands and a batch is one tensor in memory."
            ),
            inputs=[
                io.Combo.Input(
                    "file",
                    options=reader.video_labels(),
                    upload=io.UploadType.video,
                    tooltip=(
                        "Which video to read. Each entry carries the folder it sits in: "
                        "`clip.mp4 [input]`, `render.mp4 [output]`, `scratch.mp4 [temp]`. "
                        "The button below uploads one into input and selects it, and the "
                        "player shows what is selected."
                    ),
                ),
                io.Int.Input(
                    "num_frames",
                    default=0,
                    min=0,
                    max=reader.MAX_FRAMES,
                    tooltip=(
                        f"How many frames to keep, chosen by the strategy below. `0` = the "
                        f"whole clip, up to the {reader.MAX_FRAMES} ceiling; `16` = a short "
                        f"sample. A batch is one tensor, so a long clip at full size stops "
                        f"with the count to set rather than running out of memory."
                    ),
                ),
                io.Combo.Input(
                    "strategy",
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
                    default=1024,
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
                    options=list(sizing.FILTER_NAMES),
                    default=sizing.DEFAULT_FILTER,
                    optional=True,
                    tooltip="Resampling filter. `lanczos` is the sharpest for a downscale.",
                ),
                io.Combo.Input(
                    "align",
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
                    default="#000000",
                    optional=True,
                    tooltip="Fill for space a frame does not cover. Any Pillow colour.",
                ),
                io.Combo.Input(
                    "channels",
                    options=list(sizing.CHANNELS),
                    default="RGB",
                    optional=True,
                    tooltip=(
                        "Channels the image batch carries. `RGBA` keeps the pad transparent. "
                        "The video output is always colour, since a video carries no "
                        "transparency."
                    ),
                ),
            ],
            outputs=[
                io.Video.Output(
                    display_name="video",
                    tooltip=(
                        "The frames that were kept, with their sound, as a video at the rate "
                        "below. Wire it into Save Video, or into any node taking a VIDEO."
                    ),
                ),
                io.Image.Output(
                    display_name="images",
                    tooltip=(
                        "The same frames as one image batch, in playback order, every one at "
                        "the same size."
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
        resize_mode=sizing.FIT_AND_PAD, width=0, height=0, start=0, end=-1, max_size=1024,
        interpolation=sizing.DEFAULT_FILTER, align=sizing.DEFAULT_ALIGNMENT,
        pad_color="#000000", channels="RGB",
    ):
        """When the chosen file was last written, so an edited video is read again."""
        import folder_paths

        # An empty name resolves to the input folder itself, which exists, so it is refused
        # before the folder is asked about it.
        chosen = (file or "").strip()
        if not chosen or not folder_paths.exists_annotated_filepath(chosen):
            return float("NaN")
        # Its modification time rather than its digest: a video is large enough that
        # hashing it would cost more than the read the fingerprint is there to avoid.
        return os.path.getmtime(reader.input_path(file))

    @classmethod
    def validate_inputs(cls, file):
        """Whether the chosen file is still in one of ComfyUI's own folders."""
        import folder_paths

        if not (file or "").strip():
            return "no video was chosen. Pick one from the list, or upload one with the button"
        if sandbox.names_another_host(file):
            return "a path naming another machine is not read"
        if not folder_paths.exists_annotated_filepath(file):
            return (
                f"`{file}` is not in ComfyUI's input, output or temp folder. Pick "
                f"another, or upload it again"
            )
        return True

    @classmethod
    def execute(
        cls, file, num_frames=0, strategy="uniform", nth=1, seed=0, target_fps=0.0,
        resize_mode=sizing.FIT_AND_PAD, width=0, height=0, start=0, end=-1, max_size=1024,
        interpolation=sizing.DEFAULT_FILTER, align=sizing.DEFAULT_ALIGNMENT,
        pad_color="#000000", channels="RGB",
    ) -> io.NodeOutput:
        """Read the chosen video and hand on its frames, its sound and what it measures.

        Raises:
            DependencyError: PyAV is not installed.
            PathNotAllowed: The chosen file resolved outside every permitted read root.
            ValueError: Nothing was chosen, the file holds no video stream, or no frame
                could be decoded.
        """
        return load(
            reader.input_path(file), num_frames, strategy, nth, seed, target_fps,
            resize_mode, width, height, start, end, max_size, interpolation, align,
            pad_color, channels,
        )

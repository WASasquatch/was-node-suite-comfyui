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


def _streamed(path: str, start: int, end: int, num_frames: int, target_fps: float, width: int,
              height: int, max_size: int, channels: str, refusal: str) -> io.NodeOutput | None:
    """The four outputs of a read too large for one batch, where it keeps whole frames of the file.

    Args:
        path: The video file.
        start: First frame to consider.
        end: Last frame to consider, inclusive.
        num_frames: Frames asked for, 0 for every frame in the range.
        target_fps: Rate asked for, 0 for the file's own.
        width: Width asked for, 0 for the file's own.
        height: Height asked for, 0 for the file's own.
        max_size: Longest edge asked for, 0 for none.
        channels: ``"RGB"`` or ``"RGBA"``.
        refusal: Why the frames could not be loaded as one batch.

    Returns:
        The file as a video read as it plays, a blocked frame batch, the sound and the figures,
        or ``None`` where the read trims, retimes or resizes the frames.
    """
    from comfy_api.latest import InputImpl
    from comfy_execution.graph_utils import ExecutionBlocker

    source = reader.probe(path)
    first, stop = sampling.slice_bounds(source.frame_count, start, end)
    count = stop - first
    whole = (
        (not num_frames or num_frames >= count)
        and (not target_fps or abs(float(target_fps) - source.fps) < 1e-3)
        and not width and not height
        and (not max_size or max_size >= max(source.width, source.height))
        and channels == "RGB" and source.fps > 0 and count > 0
    )
    if not whole:
        return None
    begin = first / source.fps
    seconds = count / source.fps
    trimmed = first > 0 or stop < source.frame_count
    video = InputImpl.VideoFromFile(path, start_time=begin, duration=seconds if trimmed else 0)
    audio = reader.audio_span(path, begin, seconds) if source.has_audio else None
    blocked = ExecutionBlocker(
        f"{refusal}. The video output carries every frame, read from the file as it is used."
    )
    logger.info(
        "%s holds %d frame(s) at %dx%d, too many for one batch; the video output reads them "
        "from the file and the images output is blocked",
        os.path.basename(path), count, source.width, source.height,
    )
    return io.NodeOutput(
        video,
        blocked,
        audio,
        {
            "fps": float(source.fps),
            "frame_count": count,
            "duration": float(seconds),
            "width": int(source.width),
            "height": int(source.height),
            "has_audio": audio is not None,
            "bit_depth": int(source.bit_depth),
            "source_fps": float(source.fps),
            "source_frame_count": int(source.frame_count),
            "source_duration": float(source.duration),
            "source_width": int(source.width),
            "source_height": int(source.height),
            "filename": os.path.basename(path),
        },
    )


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
    max_size: int = 0,
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
            asked for do not fit in memory and are trimmed, retimed or resized.
    """
    try:
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
    except reader.BatchTooLarge as refusal:
        streamed = _streamed(path, start, end, num_frames, target_fps, width, height, max_size,
                             channels, str(refusal))
        if streamed is None:
            raise
        return streamed

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
                    advanced=True,
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
        pad_color="#000000", channels="RGB",
    ):
        """When the chosen file was last written, so an edited video is read again."""
        # An empty name resolves to the input folder itself, which exists, so it is refused
        # before the folder is asked about it.
        chosen = (file or "").strip()
        if not chosen or sandbox.annotated_path(chosen) is None:
            return float("NaN")
        # Its modification time rather than its digest: a video is large enough that
        # hashing it would cost more than the read the fingerprint is there to avoid.
        return os.path.getmtime(reader.input_path(file))

    @classmethod
    def validate_inputs(cls, file):
        """Whether the chosen file is still in one of ComfyUI's own folders."""
        if not (file or "").strip():
            return "no video was chosen. Pick one from the list, or upload one with the button"
        if sandbox.names_another_host(file):
            return "a path naming another machine is not read"
        if sandbox.annotated_path(file) is None:
            return (
                f"`{file}` is not in ComfyUI's input, output or temp folder. Pick "
                f"another, or upload it again"
            )
        return True

    @classmethod
    def execute(
        cls, file, num_frames=0, strategy="uniform", nth=1, seed=0, target_fps=0.0,
        resize_mode=sizing.FIT_AND_PAD, width=0, height=0, start=0, end=-1, max_size=0,
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

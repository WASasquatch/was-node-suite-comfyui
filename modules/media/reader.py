"""Reading a video with PyAV: what its header says, and the frames and audio inside it.

Frames come back as one ``IMAGE`` batch, every frame at one size. Audio is
``{"waveform", "sample_rate"}``, the waveform shaped ``(1, channels, samples)``.
"""

from __future__ import annotations

import os
import threading
import time
from fractions import Fraction
from typing import NamedTuple

import numpy as np
import torch
from PIL import Image

from .. import deps, log
from ..convert.tensors import stack_images
from ..image import sizing
from ..image.draw import parse_color
from ..util import sandbox
from . import sampling

__all__ = [
    "Clip",
    "VIDEO_EXTENSIONS",
    "DEFAULT_RATE",
    "FALLBACK_PAD",
    "LISTING_TTL",
    "MAX_BATCH_PIXELS",
    "MAX_FRAMES",
    "MAX_RATE",
    "Metadata",
    "frame_size",
    "input_path",
    "input_videos",
    "video_labels",
    "probe",
    "read",
    "to_video",
]

logger = log.get_logger("media.reader")

#: Container extensions a video menu offers, and the ones a download may keep. libavformat
#: reads a file by its content rather than by its name, so this is for menus, not decoding.
VIDEO_EXTENSIONS = (
    ".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".gif",
)

#: Most frames one read answers. The frames become a single tensor, so the ceiling is what
#: keeps a feature-length file from being read into memory whole.
MAX_FRAMES = 4096

#: How many pixels one read may answer in total, counting every frame it keeps. A colour
#: batch costs twelve bytes a pixel as float32, and the stack that builds it holds the
#: frames twice over, so this is around five gigabytes at the moment the batch is assembled.
MAX_BATCH_PIXELS = 192 * 1024 * 1024

#: Frame rate a stream is read at when its header names none.
DEFAULT_RATE = 30.0

#: Highest rate ``target_fps`` accepts, which covers every consumer capture format.
MAX_RATE = 240.0

#: Fill for space a frame does not cover, when one cannot be read from the widget.
FALLBACK_PAD = (0, 0, 0, 255)

#: Seconds the input directory's video listing is reused for. A burst of ``/object_info``
#: requests costs one listing, and a freshly uploaded file appears within it.
LISTING_TTL = 5.0

#: Serializes the listing below, which is read from ComfyUI's server thread for a combo and
#: from the prompt thread for a node.
_listing_lock = threading.Lock()

#: ``(monotonic stamp, file names)`` of the last input directory listing.
_listing: tuple[float, tuple[str, ...]] = (0.0, ())


#: Bits per colour component assumed where a stream's format does not name one, matching
#: `VideoInput.get_bit_depth` in comfy_api.
DEFAULT_BIT_DEPTH = 8


class Metadata(NamedTuple):
    """What a video file's header says, before anything is decoded.

    Attributes:
        fps: Frames per second.
        width: Frame width in pixels.
        height: Frame height in pixels.
        frame_count: Frames the file holds.
        duration: Seconds the file runs for.
        has_audio: Whether the file carries an audio stream.
        bit_depth: Bits per colour component the stream is encoded at, 8 where the format
            does not say.
    """

    fps: float
    width: int
    height: int
    frame_count: int
    duration: float
    has_audio: bool
    bit_depth: int = DEFAULT_BIT_DEPTH


class Clip(NamedTuple):
    """The frames and audio one read answered.

    Attributes:
        images: An ``IMAGE`` batch, ``(frames, height, width, channels)``, in playback order.
        audio: ``{"waveform": (1, channels, samples), "sample_rate": int}``, or ``None``.
        fps: Rate the frames are answered at.
        indices: Which source frame each image is, in the order they are answered.
        source: What the file's header said.
    """

    images: torch.Tensor
    audio: dict | None
    fps: float
    indices: list[int]
    source: Metadata


def input_videos() -> list[str]:
    """The video files sitting in ComfyUI's input directory, in name order.

    Returns:
        File names, memoized for :data:`LISTING_TTL` seconds. Empty outside ComfyUI and
        where the directory cannot be read.
    """
    global _listing
    with _listing_lock:
        stamp, names = _listing
        now = time.monotonic()
        if stamp and now - stamp < LISTING_TTL:
            return list(names)
        names = tuple(_scan_input())
        _listing = (now, names)
        return list(names)


def video_labels() -> list[str]:
    """Every video under ComfyUI's own directories, as the labels a widget stores.

    Returns:
        ``<relative path> [input]``, ``[output]`` or ``[temp]`` per file, in the listing's
        own order. Empty outside ComfyUI and where no root can be read.
    """
    try:
        import folder_paths
    except ImportError:
        return []
    from ..util import file_listing

    try:
        entries = file_listing.view(tags=file_listing.ROOTS)
    except Exception as error:
        logger.debug("the file listing could not be read: %s", error)
        return list(input_videos())
    names = [entry.relative for entry in entries]
    keep = set(folder_paths.filter_files_content_types(names, ["video"]))
    return [entry.label for entry in entries if entry.relative in keep]


def input_path(name: str) -> str:
    """The file a video widget names, resolved inside a permitted read root.

    Args:
        name: The widget's value, either a bare name in the input folder or one carrying
            its folder as `clip.mp4 [output]`.

    Returns:
        The absolute path of the file.

    Raises:
        PathNotAllowed: It resolved outside every permitted read root.
        ValueError: The widget is empty, or names a file that is not there.
    """
    import folder_paths

    chosen = (name or "").strip()
    if not chosen:
        raise ValueError(
            "no video was chosen. Pick one from the file list, or put a video in ComfyUI's "
            "input, output or temp folder and use the upload button on the node"
        )
    if not folder_paths.exists_annotated_filepath(chosen):
        raise ValueError(
            f"`{chosen}` is not in ComfyUI's input, output or temp folder any more. Pick "
            f"another from the file list, or upload it again"
        )
    return str(sandbox.resolve_read(folder_paths.get_annotated_filepath(chosen)))


def probe(path: str) -> Metadata:
    """Read a video file's header without decoding any of it.

    Args:
        path: Video file to open.

    Returns:
        What the header says.

    Raises:
        DependencyError: PyAV is not installed.
        ValueError: The file holds no video stream.
    """
    av = deps.require("av")

    name = str(path)
    with av.open(name, mode="r") as container:
        return _describe(container, _video_stream(container, name), name)


def read(
    path: str,
    start: int = 0,
    end: int = -1,
    num_frames: int = 0,
    strategy: str = "uniform",
    nth: int = 1,
    seed: int = 0,
    target_fps: float = 0.0,
    resize_mode: str = sizing.FIT_AND_PAD,
    width: int = 0,
    height: int = 0,
    max_size: int = 0,
    interpolation: str = sizing.DEFAULT_FILTER,
    align: str = sizing.DEFAULT_ALIGNMENT,
    pad_color: str = "#000000",
    channels: str = "RGB",
    limit: int = MAX_FRAMES,
) -> Clip:
    """Decode the frames a selection keeps, at one size, with the audio playing under them.

    Args:
        path: Video file to open.
        start: First frame to consider, counting from 0. Negative counts back from the end.
        end: Last frame to consider, inclusive. -1 is the final frame.
        num_frames: How many frames to keep out of that range, 0 for all of them up to
            ``limit``.
        strategy: One of :data:`modules.media.sampling.STRATEGIES`.
        nth: Step between kept frames, read only by ``every_nth``.
        seed: Seed for ``random``.
        target_fps: Rate the frames are answered at, 0 to keep the file's own. A lower rate
            drops frames and a higher one repeats them, so the range plays for as long
            either way.
        resize_mode: One of :data:`modules.image.sizing.MODES`.
        width: Width every frame is brought to, 0 for the size it was encoded at.
        height: Height every frame is brought to, 0 for the size it was encoded at.
        max_size: Longest edge a derived size is held to, read only where both sides were
            derived. 0 for no cap.
        interpolation: A name from :data:`modules.image.sizing.FILTER_NAMES`.
        align: A name from :data:`modules.image.sizing.ALIGNMENT_NAMES`.
        pad_color: Fill for space a frame does not cover, in any Pillow spelling.
        channels: ``"RGB"`` or ``"RGBA"``.
        limit: Most frames the read answers.

    Returns:
        The frames as one batch, the audio covering what they play for, and the file's own
        header.

    Raises:
        DependencyError: PyAV is not installed.
        ValueError: The file holds no video stream, no frame at all, no frame that could be
            decoded, or more frames than one batch will hold.
    """
    av = deps.require("av")

    name = str(path)
    pad = parse_color(pad_color, FALLBACK_PAD)
    with av.open(name, mode="r") as container:
        stream = _video_stream(container, name)
        source = _describe(container, stream, name)
        if source.frame_count <= 0:
            raise ValueError(
                f"`{name}` reports no frames, so there is nothing to read. A container "
                f"written by an interrupted encode reads this way"
            )
        if source.width <= 0 or source.height <= 0:
            raise ValueError(
                f"`{name}` reports a frame size of {source.width}x{source.height}, so its "
                f"frames cannot be brought to a size. Re-encode it and read it again"
            )

        first, stop = sampling.slice_bounds(source.frame_count, start, end)
        window = _retimed(list(range(first, stop)), source.fps, target_fps)
        if num_frames:
            picked = sampling.frame_indices(len(window), num_frames, strategy, nth, seed)
            chosen = [window[index] for index in picked]
        else:
            chosen = window
        chosen = chosen[: max(1, int(limit))]

        target = frame_size((source.width, source.height), width, height, max_size)
        _affordable(len(chosen), target)
        decoded = _decoded(
            container, stream, set(chosen), target, resize_mode, interpolation, align, pad,
            channels,
        )
        indices = [number for number in chosen if number in decoded]
        if not indices:
            raise ValueError(
                f"none of the {len(chosen)} frame(s) asked for could be decoded from "
                f"`{name}`. The file may be truncated or its stream damaged"
            )
        if len(indices) < len(chosen):
            # A header whose frame count is higher than what the stream actually decodes.
            logger.warning(
                "%s ended after %d of the %d frame(s) asked for; the rest were dropped",
                os.path.basename(name), len(indices), len(chosen),
            )

        rate = float(target_fps) if target_fps > 0 else source.fps
        audio = None
        if source.has_audio and rate > 0:
            audio = _audio(name, min(indices) / source.fps, len(indices) / rate)

    return Clip(_stacked(decoded, indices, target), audio, rate, indices, source)


def frame_size(source: tuple[int, int], width: int, height: int, cap: int) -> tuple[int, int]:
    """The size every frame is brought to.

    Args:
        source: ``(width, height)`` of the frames as they were decoded.
        width: Requested width, 0 to take it from the frame.
        height: Requested height, 0 to take it from the frame.
        cap: Longest edge the derived size is held to, 0 for none. Read only where both
            sides were derived, since an explicit size is what was asked for.

    Returns:
        ``(width, height)``, never below 1 on either side.
    """
    wide, high = max(1, int(source[0])), max(1, int(source[1]))
    if width and height:
        return max(1, int(width)), max(1, int(height))
    if width:
        return max(1, int(width)), max(1, round(high * width / wide))
    if height:
        return max(1, round(wide * height / high)), max(1, int(height))
    if cap and max(wide, high) > cap:
        scale = cap / max(wide, high)
        return max(1, round(wide * scale)), max(1, round(high * scale))
    return wide, high


def to_video(
    images: torch.Tensor,
    fps: float,
    audio: dict | None = None,
    bit_depth: int = DEFAULT_BIT_DEPTH,
):
    """One ``VIDEO`` carrying an image batch, its rate and its sound.

    Args:
        images: An ``IMAGE`` batch, ``(frames, height, width, channels)``. A fourth channel
            is dropped, since a video carries no transparency.
        fps: Frames per second the batch plays at.
        audio: The ``AUDIO`` to carry, or ``None`` for a silent video.
        bit_depth: Bits per colour component to report, which is what a save node encodes
            at. Left at the default a ten bit source would be written back as eight.

    Returns:
        A ComfyUI video built from those components.
    """
    from comfy_api.latest import InputImpl, Types

    colour = images[..., :3] if images.shape[-1] > 3 else images
    return InputImpl.VideoFromComponents(
        Types.VideoComponents(
            images=colour,
            audio=audio,
            frame_rate=Fraction(max(fps, 1e-6)).limit_denominator(100000),
        ),
        bit_depth=max(int(bit_depth), DEFAULT_BIT_DEPTH),
    )


# ---------------------------------------------------------------------- internals


def _scan_input() -> list[str]:
    """Every file in ComfyUI's input directory that carries a video mime type, sorted."""
    try:
        import folder_paths
    except ImportError:
        return []
    directory = folder_paths.get_input_directory()
    try:
        found = [
            name
            for name in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, name))
        ]
    except OSError as error:
        logger.debug("the input directory could not be listed: %s", error)
        return []
    return sorted(folder_paths.filter_files_content_types(found, ["video"]))


def _video_stream(container, path: str):
    """The container's first video stream, set up for threaded decoding.

    Args:
        container: An open av container.
        path: The file it was opened from, named in the message.

    Returns:
        The stream.

    Raises:
        ValueError: The file holds no video stream.
    """
    stream = next((entry for entry in container.streams if entry.type == "video"), None)
    if stream is None:
        raise ValueError(
            f"`{path}` holds no video stream, so there are no frames to read. A sound-only "
            f"file, or one whose extension does not match what is inside it, reads this way"
        )
    stream.thread_type = "AUTO"
    return stream


def _describe(container, stream, path: str) -> Metadata:
    """What a container and its video stream report about themselves.

    Args:
        container: An open av container.
        stream: Its video stream.
        path: The file it was opened from, read a second time to count packets where the
            header names neither a frame count nor a duration.

    Returns:
        The header's rate, size, frame count, duration and whether there is sound. A frame
        count the header leaves unset is derived from the duration, and failing that by
        counting packets.
    """
    av = deps.require("av")

    fps = float(Fraction(stream.average_rate)) if stream.average_rate else DEFAULT_RATE
    fps = fps if fps > 0 else DEFAULT_RATE
    duration = float(container.duration / av.time_base) if container.duration else 0.0
    count = int(stream.frames or 0)
    if count <= 0 and duration > 0:
        count = int(round(duration * fps))
    if count <= 0:
        count = _counted(path, stream.index)
    if duration <= 0 and count > 0:
        duration = count / fps
    return Metadata(
        fps=fps,
        width=int(stream.width or 0),
        height=int(stream.height or 0),
        frame_count=count,
        duration=duration,
        has_audio=bool(container.streams.audio),
        bit_depth=_bit_depth(stream),
    )


def _bit_depth(stream) -> int:
    """Bits per colour component one video stream is encoded at.

    Args:
        stream: An av video stream.

    Returns:
        The widest component's depth, and :data:`DEFAULT_BIT_DEPTH` where the format names
        no components. This is what ``comfy_api``'s own reader answers for the same file.
    """
    components = getattr(getattr(stream, "format", None), "components", None)
    if not components:
        return DEFAULT_BIT_DEPTH
    try:
        return max(int(component.bits) for component in components)
    except (TypeError, ValueError):
        return DEFAULT_BIT_DEPTH


def _counted(path: str, index: int) -> int:
    """How many packets one video stream holds, counted in a container of its own.

    Args:
        path: The video file.
        index: Which of its streams to count.

    Returns:
        The packet count, which is the frame count for every codec that carries one frame
        per packet.
    """
    av = deps.require("av")

    count = 0
    # A container of its own, so nothing the caller is reading has to be rewound.
    with av.open(str(path), mode="r") as container:
        for packet in container.demux(container.streams[index]):
            if packet.size:
                count += 1
    return count


def _retimed(frames: list[int], source_fps: float, target_fps: float) -> list[int]:
    """Which source frame each frame of a rate change shows.

    Args:
        frames: Source frame numbers, in playback order.
        source_fps: The file's own rate.
        target_fps: Rate wanted, 0 to keep the file's own.

    Returns:
        Frame numbers, dropped where the target rate is lower and repeated where it is
        higher, so the run plays for as long either way.
    """
    if target_fps <= 0 or source_fps <= 0 or not frames:
        return frames
    count = max(1, round(len(frames) * target_fps / source_fps))
    step = source_fps / target_fps
    return [frames[min(len(frames) - 1, int(index * step))] for index in range(count)]


def _affordable(frames: int, target: tuple[int, int]) -> None:
    """Refuse a batch too large to hold, before any of it is decoded.

    Args:
        frames: How many frames the read would answer.
        target: ``(width, height)`` every one of them is brought to.

    Raises:
        ValueError: The batch would hold more than :data:`MAX_BATCH_PIXELS` pixels.
    """
    pixels = frames * target[0] * target[1]
    if pixels <= MAX_BATCH_PIXELS:
        return
    allowed = max(1, MAX_BATCH_PIXELS // (target[0] * target[1]))
    raise ValueError(
        f"{frames} frame(s) at {target[0]}x{target[1]} come to about "
        f"{pixels * 12 / 1024 ** 3:.1f} GiB as one batch, which is more than one load will "
        f"hold in memory. Set num_frames to {allowed} or fewer, lower target_fps, or bring "
        f"the frames down with max_size, width and height"
    )


def _decoded(
    container, stream, wanted: set[int], target: tuple[int, int], resize_mode: str,
    interpolation: str, align: str, pad: tuple[int, int, int, int], channels: str,
) -> dict[int, Image.Image]:
    """Decode the listed frames of a video stream, each one brought to a single size.

    Args:
        container: An open av container, positioned at the start.
        stream: Its video stream.
        wanted: Frame numbers to keep, counting from 0.
        target: ``(width, height)`` every frame is brought to.
        resize_mode: One of :data:`modules.image.sizing.MODES`.
        interpolation: A name from :data:`modules.image.sizing.FILTER_NAMES`.
        align: A name from :data:`modules.image.sizing.ALIGNMENT_NAMES`.
        pad: ``(red, green, blue, alpha)`` filling space a frame does not cover.
        channels: ``"RGB"`` or ``"RGBA"``.

    Returns:
        ``{frame number: image}`` for every frame that decoded.
    """
    last = max(wanted)
    kept: dict[int, Image.Image] = {}
    # One pass in presentation order, stopping at the last frame asked for. Seeking per
    # frame would be slower on a long clip than reading through it once. Each frame is
    # brought to size as it arrives, so a 4K source never holds more than one 4K image.
    for number, frame in enumerate(container.decode(stream)):
        if number in wanted:
            image = Image.fromarray(frame.to_ndarray(format="rgb24"))
            kept[number] = sizing.as_channels(
                sizing.fit(image, target[0], target[1], resize_mode, interpolation, align, pad),
                channels,
            )
        if number >= last:
            break
    return kept


def _stacked(
    decoded: dict[int, Image.Image], indices: list[int], target: tuple[int, int]
) -> torch.Tensor:
    """Assemble the decoded frames into one ``IMAGE`` batch, in playback order.

    Args:
        decoded: ``{frame number: image}``, every image already at ``target``.
        indices: Frame numbers to stack, in the order they play.
        target: ``(width, height)`` they were brought to, named in the message.

    Returns:
        A float32 tensor shaped ``(frames, height, width, channels)``.

    Raises:
        ValueError: The batch did not fit in memory.
    """
    try:
        return stack_images([decoded[number] for number in indices])
    except (MemoryError, ArithmeticError, RuntimeError) as short:
        need = len(indices) * target[0] * target[1] * 4 * 4 / (1024 ** 3)
        raise ValueError(
            f"{len(indices)} frame(s) at {target[0]}x{target[1]} need about {need:.1f} GiB "
            f"as one batch and would not fit ({short}). Keep fewer frames with num_frames, "
            f"or set a smaller width and height"
        ) from short


def _audio(path: str, begin: float, seconds: float) -> dict | None:
    """Decode the sound playing over one span of a video.

    Args:
        path: The video file, opened again so the audio pass starts at the beginning.
        begin: Where the span starts, in seconds from the start of the file.
        seconds: How long the span runs for.

    Returns:
        ``{"waveform": (1, channels, samples), "sample_rate": int}``, or ``None`` where
        there is no decodable audio stream and where the span holds no samples.
    """
    av = deps.require("av")

    blocks: list[np.ndarray] = []
    head: float | None = None
    done = False
    with av.open(str(path), mode="r") as container:
        # A stream FFmpeg has no decoder for carries no codec context, and decoding its
        # packets takes the process down with it.
        stream = next(
            (entry for entry in container.streams.audio if entry.codec_context is not None),
            None,
        )
        rate = int(stream.sample_rate or 0) if stream is not None else 0
        if not rate or seconds <= 0:
            return None

        finish = begin + seconds
        resampler = av.audio.resampler.AudioResampler(format="fltp")
        try:
            for frame in container.decode(stream):
                for block in resampler.resample(frame):
                    when = float(block.time) if block.time is not None else None
                    if when is not None:
                        if when + block.samples / rate <= begin:
                            continue
                        if when >= finish:
                            done = True
                            break
                        if head is None:
                            head = when
                    blocks.append(block.to_ndarray())
                if done:
                    break
            if not done:
                blocks += [block.to_ndarray() for block in resampler.resample(None)]
        except av.error.FFmpegError as error:
            logger.warning("the audio stream stopped decoding, keeping what was read: %s", error)

    if not blocks:
        return None
    data = np.concatenate(blocks, axis=1)
    # The first block kept can start before the span does, so its own time decides how many
    # samples are trimmed off the front.
    started = begin if head is None else head
    offset = max(0, int(round((begin - started) * rate)))
    data = data[:, offset : offset + max(1, int(round(seconds * rate)))]
    if data.shape[1] == 0:
        return None
    waveform = torch.from_numpy(np.ascontiguousarray(data)).unsqueeze(0).float()
    return {"waveform": waveform, "sample_rate": rate}

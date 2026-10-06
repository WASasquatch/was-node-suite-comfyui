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
from ..convert.tensors import pil2tensor
from ..image import scratch, sizing
from ..image.draw import parse_color
from ..util import file_listing, sandbox
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
    "audio_length",
    "audio_span",
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
#: batch costs twelve bytes a pixel as float32.
MAX_BATCH_PIXELS = 192 * 1024 * 1024

#: Closing sentence of the refusal for a batch that does not fit.
SMALLER = "Keep fewer frames with num_frames, or set a smaller width and height."

#: Frame rate a stream is read at when its header names none.
DEFAULT_RATE = 30.0

#: Highest rate ``target_fps`` accepts, which covers every consumer capture format.
MAX_RATE = 240.0

#: Fill for space a frame does not cover, when one cannot be read from the widget.
FALLBACK_PAD = (0, 0, 0, 255)

#: Seconds the input directory's video listing is reused for at least. A burst of
#: ``/object_info`` requests costs one listing, and a freshly uploaded file appears within it.
LISTING_TTL = file_listing.LISTING_TTL

#: Serializes the listing below, which is read from ComfyUI's server thread for a combo and
#: from the prompt thread for a node.
_listing_lock = threading.Lock()

#: ``(monotonic time it is reused until, file names)`` of the last input directory listing.
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
        File names, memoized for at least :data:`LISTING_TTL` seconds. Empty outside
        ComfyUI and where the directory cannot be read.
    """
    global _listing
    with _listing_lock:
        until, names = _listing
        now = time.monotonic()
        if until and now < until:
            return list(names)
        names = tuple(_scan_input())
        _listing = (file_listing.reuse_until(now, time.monotonic()), names)
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

    try:
        # One sample name per suffix the walk holds, so the menu's limit counts videos alone.
        suffixes = {os.path.splitext(entry.relative)[1].lower() for entry in file_listing.scan()}
        samples = [f"video{suffix}" for suffix in sorted(suffixes) if suffix]
        videos = folder_paths.filter_files_content_types(samples, ["video"])
        if not videos:
            return []
        entries = file_listing.view(
            [os.path.splitext(name)[1] for name in videos], tags=file_listing.ROOTS
        )
    except Exception as error:
        logger.debug("the file listing could not be read: %s", error)
        return list(input_videos())
    return [entry.label for entry in entries]


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
    chosen = (name or "").strip()
    if not chosen:
        raise ValueError(
            "no video was chosen. Pick one from the file list, or put a video in ComfyUI's "
            "input, output or temp folder and use the upload button on the node"
        )
    if sandbox.names_another_host(chosen):
        raise ValueError(f"`{chosen}` names another machine, which a video is not read from")
    found = sandbox.annotated_path(chosen)
    if found is None:
        raise ValueError(
            f"`{chosen}` is not in ComfyUI's input, output or temp folder any more. Pick "
            f"another from the file list, or upload it again"
        )
    return str(sandbox.resolve_read(found))


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
        batch = _allocated(len(chosen), target, 4 if channels == "RGBA" else 3)
        decoded = _decoded(
            container, stream, chosen, batch, target, resize_mode, interpolation, align, pad,
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
            audio = audio_span(name, min(indices) / source.fps, len(indices) / rate)

    return Clip(_kept(batch, chosen, decoded), audio, rate, indices, source)


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


def _allocated(frames: int, target: tuple[int, int], channels: int) -> torch.Tensor:
    """An unfilled ``IMAGE`` batch for a read.

    Args:
        frames: Frames it holds.
        target: ``(width, height)`` of every frame.
        channels: 3 or 4.

    Returns:
        A float32 tensor shaped ``(frames, height, width, channels)``, in memory or in a
        scratch file.

    Raises:
        ValueError: Neither memory nor a scratch drive has room for it.
    """
    try:
        return scratch.allocate((frames, target[1], target[0], channels), advice=SMALLER)
    except MemoryError as short:
        raise ValueError(str(short)) from short


def _decoded(
    container, stream, chosen: list[int], batch: torch.Tensor, target: tuple[int, int],
    resize_mode: str, interpolation: str, align: str, pad: tuple[int, int, int, int],
    channels: str,
) -> set[int]:
    """Decode the listed frames of a video stream into their slots of a batch.

    Args:
        container: An open av container, positioned at the start.
        stream: Its video stream.
        chosen: Frame number each slot of ``batch`` holds, counting from 0. A number may
            fill several slots.
        batch: ``(len(chosen), height, width, channels)`` float32 tensor, written in place.
        target: ``(width, height)`` every frame is brought to.
        resize_mode: One of :data:`modules.image.sizing.MODES`.
        interpolation: A name from :data:`modules.image.sizing.FILTER_NAMES`.
        align: A name from :data:`modules.image.sizing.ALIGNMENT_NAMES`.
        pad: ``(red, green, blue, alpha)`` filling space a frame does not cover.
        channels: ``"RGB"`` or ``"RGBA"``.

    Returns:
        The frame numbers that decoded. The slots of every other number are left unwritten.
    """
    slots: dict[int, list[int]] = {}
    for position, number in enumerate(chosen):
        slots.setdefault(number, []).append(position)
    last = max(slots)
    decoded: set[int] = set()
    # One pass in presentation order, stopping at the last frame asked for.
    for number, frame in enumerate(container.decode(stream)):
        if number in slots:
            image = Image.fromarray(frame.to_ndarray(format="rgb24"))
            plane = pil2tensor(sizing.as_channels(
                sizing.fit(image, target[0], target[1], resize_mode, interpolation, align, pad),
                channels,
            ))[0]
            for position in slots[number]:
                batch[position].copy_(plane)
            decoded.add(number)
        if number >= last:
            break
    return decoded


def _kept(batch: torch.Tensor, chosen: list[int], decoded: set[int]) -> torch.Tensor:
    """The batch without the slots of frames that did not decode, in playback order.

    Args:
        batch: The batch :func:`_decoded` filled.
        chosen: Frame number each slot holds.
        decoded: The frame numbers that decoded.

    Returns:
        ``batch`` itself where every frame decoded, otherwise a new batch of the slots that
        were written.

    Raises:
        ValueError: Neither memory nor a scratch drive has room for the smaller batch.
    """
    if all(number in decoded for number in chosen):
        return batch
    filled = [
        batch[position:position + 1]
        for position, number in enumerate(chosen) if number in decoded
    ]
    try:
        return scratch.join(filled, advice=SMALLER)
    except MemoryError as short:
        raise ValueError(str(short)) from short


def audio_length(path: str) -> float:
    """How long the sound in a file runs for, from its header.

    Args:
        path: The file to open, a video or a sound file.

    Returns:
        The length in seconds, ``0.0`` where the file carries no audio stream or states no
        length.

    Raises:
        DependencyError: PyAV is not installed.
    """
    av = deps.require("av")

    with av.open(str(path), mode="r") as container:
        stream = next(
            (entry for entry in container.streams.audio if entry.codec_context is not None),
            None,
        )
        if stream is None:
            return 0.0
        if stream.duration is not None and stream.time_base is not None:
            return max(0.0, float(stream.duration * stream.time_base))
        if container.duration is not None:
            return max(0.0, float(container.duration) / 1_000_000.0)
    return 0.0


def audio_span(path: str, begin: float, seconds: float) -> dict | None:
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

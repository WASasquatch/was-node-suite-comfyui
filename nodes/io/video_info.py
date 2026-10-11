"""Read a video's size, length, rate, colour depth and sound from its header."""

from __future__ import annotations

from comfy_api.latest import VideoInput, io

from ...modules import deps, log

logger = log.get_logger("nodes.io")


def figure(video, method: str, fallback):
    """One figure a video reports about itself.

    Args:
        video: The video to ask.
        method: Name of the method answering the figure.
        fallback: Answer for a figure the video does not report.

    Returns:
        What the method answered, as the same type as ``fallback``.
    """
    read = getattr(video, method, None)
    if read is None:
        return fallback
    try:
        return type(fallback)(read())
    except Exception as error:
        logger.warning("a video could not report %s, reading it as %r (%s)",
                       method, fallback, error)
        return fallback


def has_sound(video) -> bool:
    """Whether a video carries an audio track that can be decoded.

    Args:
        video: The video to inspect.

    Returns:
        True where an audio stream is present and FFmpeg holds a decoder for it.

    Raises:
        DependencyError: PyAV is not installed.
    """
    streamable = getattr(type(video), "get_stream_source", None)
    # True where the class reads its source as it stands rather than encoding one first.
    if streamable is not None and streamable is not VideoInput.get_stream_source:
        av = deps.require("av")
        with av.open(video.get_stream_source(), mode="r") as container:
            # A stream FFmpeg holds no decoder for carries no codec context.
            return any(
                stream.codec_context is not None for stream in container.streams.audio
            )
    audio = video.get_components().audio
    waveform = audio.get("waveform") if audio else None
    return waveform is not None and waveform.numel() > 0


def describe(
    width: int,
    height: int,
    duration: float,
    frame_count: int,
    fps: float,
    bit_depth: int,
    has_audio: bool,
) -> str:
    """Every measurement written out as one line.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.
        duration: Seconds the video runs for.
        frame_count: Frames the video holds.
        fps: Frames per second.
        bit_depth: Bits per colour component.
        has_audio: Whether there is a sound track.

    Returns:
        Text shaped ``1920x1080, 240 frames, 10.00s at 24 fps, 8-bit, with sound``.
    """
    sound = "with sound" if has_audio else "silent"
    return (
        f"{width}x{height}, {frame_count} frames, {duration:.2f}s at {fps:.6g} fps, "
        f"{bit_depth}-bit, {sound}"
    )


class VideoInfo(io.ComfyNode):
    """Measure any video and answer its figures as separate numbers."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASVideoInfo",
            display_name="Video Info",
            search_aliases=[
                "WASVideoInfo",
                "Video Info",
                "video width",
                "video height",
                "video duration",
                "video fps",
                "video frame count",
                "video bit depth",
                "probe video",
            ],
            category="WAS Suite/IO",
            description=(
                "Measure any VIDEO and hand on its width, height, duration, frame count, "
                "frame rate, bit depth and whether it has sound. Use it to size a resize, "
                "tell a sampler how many frames are coming, or check for sound before wiring "
                "audio. A long file measures as fast as a short one."
            ),
            inputs=[
                io.Video.Input(
                    "video",
                    tooltip=(
                        "The video to measure. Anything on a VIDEO wire: a file that was "
                        "loaded, a video built from frames, or a trimmed one. A trim is "
                        "honoured, so the figures describe what plays rather than the file."
                    ),
                ),
            ],
            outputs=[
                io.Int.Output(
                    display_name="width",
                    tooltip="Frame width in pixels. 1920 for HD, 3840 for 4K.",
                ),
                io.Int.Output(
                    display_name="height",
                    tooltip="Frame height in pixels. 1080 for HD, 2160 for 4K.",
                ),
                io.Float.Output(
                    display_name="duration",
                    tooltip=(
                        "Seconds the video plays for. 10.0 for 240 frames at 24 fps. 0.0 "
                        "where the header names no length and none could be worked out from "
                        "the frame count and the rate."
                    ),
                ),
                io.Int.Output(
                    display_name="frame_count",
                    tooltip=(
                        "How many frames play. 240 for ten seconds at 24 fps. Taken from "
                        "the header, or worked out from the duration and the rate where the "
                        "header does not say. Feed it to a sampler that needs its length."
                    ),
                ),
                io.Float.Output(
                    display_name="fps",
                    tooltip=(
                        "Frames per second. 24 for film, 25 or 30 for broadcast, 29.97 for "
                        "NTSC. 0.0 where the header names no rate. Feed it to a save node so "
                        "a render plays at the speed it was shot."
                    ),
                ),
                io.Int.Output(
                    display_name="bit_depth",
                    tooltip=(
                        "Bits per colour component. 8 for most footage, 10 for HDR and "
                        "higher end capture. Feed it to a save node so a 10-bit source is "
                        "written back at 10 bits instead of being flattened to 8."
                    ),
                ),
                io.Boolean.Output(
                    display_name="has_audio",
                    tooltip=(
                        "True when there is a sound track that can be decoded. False for a "
                        "silent file, and for one whose track FFmpeg has no decoder for. "
                        "Check it before wiring an audio socket into a save."
                    ),
                ),
                io.String.Output(
                    display_name="summary",
                    tooltip=(
                        "All of the above on one line, as `1920x1080, 240 frames, 10.00s at "
                        "24 fps, 8-bit, with sound`. Wire it to a text preview, or into a "
                        "filename prefix to stamp a render with what it came from."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, video) -> io.NodeOutput:
        """Measure the video and answer each figure.

        Args:
            video: The video to read.

        Returns:
            The size, the duration, the frame count, the rate, the bit depth, whether there
            is sound, and the seven of them written out as one line.

        Raises:
            DependencyError: PyAV is not installed.
            ValueError: Nothing arrived on the video input, or the source holds no video
                stream.
        """
        if video is None:
            raise ValueError(
                "no video arrived on the video input. Wire a video loader, a node that "
                "builds a video from frames, or anything else answering a VIDEO into it"
            )

        width, height = video.get_dimensions()
        duration = figure(video, "get_duration", 0.0)
        fps = figure(video, "get_frame_rate", 0.0)
        frame_count = figure(video, "get_frame_count", 0)
        bit_depth = figure(video, "get_bit_depth", 8)

        # Each figure fills in from the other two where the header left it unreported.
        if fps <= 0 and duration > 0 and frame_count > 0:
            fps = frame_count / duration
        if frame_count <= 0 and fps > 0 and duration > 0:
            frame_count = int(round(duration * fps))
        if duration <= 0 and fps > 0 and frame_count > 0:
            duration = frame_count / fps

        has_audio = has_sound(video)
        summary = describe(
            int(width), int(height), float(duration), int(frame_count), float(fps),
            int(bit_depth), has_audio,
        )
        logger.info("read a video: %s", summary)
        return io.NodeOutput(
            int(width),
            int(height),
            float(duration),
            int(frame_count),
            float(fps),
            int(bit_depth),
            bool(has_audio),
            summary,
        )

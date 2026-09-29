"""A song's beats, downbeats, tempo and loudness, for timing a video to it."""

from __future__ import annotations

from comfy_api.latest import io, ui

from ...modules.compat.types import BEATS
from ...modules.media import beats as beat_analysis


class AudioBeats(io.ComfyNode):
    """Find a song's beats and loudness at a video frame rate."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioBeats",
            display_name="Audio Beats",
            search_aliases=[
                "WASAudioBeats",
                "Audio Beats",
                "beat detection",
                "tempo",
                "bpm",
                "audio reactive",
            ],
            category="WAS Suite/Animation",
            description=(
                "Find a song's beats, the first beat of each bar, its tempo and how loud it "
                "is at every video frame, for timing a video to the music. The audio output "
                "is the stretch analysed. The strip on the node shows loudness, beats and bar "
                "lines over time."
            ),
            inputs=[
                io.Audio.Input("audio", tooltip="The song, from Load Audio."),
                io.Float.Input(
                    "fps",
                    default=24.0,
                    min=1.0,
                    max=240.0,
                    step=0.001,
                    tooltip="Video frames per second, as `24` for MiniMax H3 or `16` for Wan.",
                ),
                io.Int.Input(
                    "beats_per_bar",
                    default=4,
                    min=1,
                    max=16,
                    tooltip="Beats in one bar, as `4` for most music or `3` for a waltz.",
                ),
                io.Float.Input(
                    "start",
                    default=0.0,
                    min=0.0,
                    max=86400.0,
                    step=0.01,
                    tooltip="Seconds into the song the video starts at, as `0.0` or `31.5`.",
                ),
                io.Float.Input(
                    "length",
                    default=0.0,
                    min=0.0,
                    max=86400.0,
                    step=0.01,
                    tooltip="Seconds of song the video covers, as `48.0`, or `0.0` for the rest.",
                ),
            ],
            outputs=[
                BEATS.Output(display_name="beats", tooltip="The beats, times and per-frame loudness together."),
                io.Audio.Output(
                    display_name="audio",
                    tooltip="The stretch analysed, from start for length seconds.",
                ),
                io.Float.Output(display_name="tempo", tooltip="Beats per minute."),
                io.Float.Output(
                    display_name="bar_seconds",
                    tooltip="Seconds in one bar, for lining segment lengths up with bars.",
                ),
                io.Int.Output(display_name="beat_count", tooltip="Beats found."),
                io.Image.Output(display_name="plot", tooltip="The strip drawn on the node."),
                io.String.Output(
                    display_name="report",
                    tooltip="Tempo, counts and the first beat and bar times.",
                ),
            ],
        )

    @classmethod
    def execute(cls, audio, fps, beats_per_bar, start, length) -> io.NodeOutput:
        """Cut the song to the video's stretch and analyse it.

        Raises:
            ValueError: The stretch holds no audio.
        """
        stretch = beat_analysis.trimmed(audio, start, length)
        if stretch["waveform"].shape[-1] < stretch["sample_rate"]:
            raise ValueError(
                f"Audio Beats has less than a second of audio from {start}s. Lower start, "
                f"or raise length"
            )
        result = beat_analysis.analyse(stretch, float(fps), int(beats_per_bar))
        picture = beat_analysis.plot(result)
        report = (
            f"{result.tempo:.2f} BPM, {len(result.times)} beats and {len(result.downbeats)} "
            f"bars over {result.duration:.2f}s, one bar {result.bar_seconds:.3f}s; first "
            f"beats {', '.join(f'{t:.2f}' for t in result.times[:4])}s; first bars "
            f"{', '.join(f'{t:.2f}' for t in result.downbeats[:3])}s"
        )
        return io.NodeOutput(
            result, stretch, float(result.tempo), float(result.bar_seconds),
            len(result.times), picture, report, ui=ui.PreviewImage(picture, cls=cls),
        )

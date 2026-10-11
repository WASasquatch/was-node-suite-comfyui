"""Read an AUDIO's length, shape and level as separate numbers."""

from __future__ import annotations

import math

import torch
from comfy_api.latest import io, ui


def sample_rate_of(audio) -> int:
    """The rate a clip was sampled at.

    Args:
        audio: An ``AUDIO`` mapping, or anything that is not a mapping.

    Returns:
        Samples per second, or 0 where none was recorded or it is not a positive whole
        number.
    """
    if not isinstance(audio, dict):
        return 0
    try:
        rate = int(audio.get("sample_rate", 0))
    except (TypeError, ValueError, OverflowError):
        return 0
    return rate if rate > 0 else 0


def dimensions(waveform) -> tuple[int, int, int]:
    """Batch, channel and sample counts of a waveform.

    Args:
        waveform: A tensor shaped ``(batch, channels, samples)``, or anything else.

    Returns:
        ``(batch_size, channels, samples)``, zeroes where the shape cannot be read.
    """
    try:
        sizes = [int(size) for size in getattr(waveform, "shape", None)]
    except (TypeError, ValueError):
        return 0, 0, 0
    if len(sizes) == 3:
        return sizes[0], sizes[1], sizes[2]
    # A waveform missing its batch axis, or both leading axes, counts as one of each.
    if len(sizes) == 2:
        return 1, sizes[0], sizes[1]
    if len(sizes) == 1:
        return 1, 1, sizes[0]
    return 0, 0, 0


def levels(waveform) -> tuple[float, float]:
    """Peak and root mean square level of a waveform.

    Args:
        waveform: A tensor holding sample values.

    Returns:
        ``(peak, rms)``, zeroes where there are no samples, the values are not real
        numbers, or a result is not finite.
    """
    if not isinstance(waveform, torch.Tensor) or waveform.numel() == 0:
        return 0.0, 0.0
    if waveform.is_complex():
        return 0.0, 0.0
    try:
        flat = waveform.detach().reshape(-1).cpu().to(torch.float64)
        peak = float(flat.abs().max())
        rms = float(flat.square().mean().sqrt())
    except (RuntimeError, TypeError, ValueError):
        return 0.0, 0.0
    if not math.isfinite(peak) or not math.isfinite(rms):
        return 0.0, 0.0
    return peak, rms


def reading(audio, silence_threshold: float) -> tuple:
    """Every figure an AUDIO carries.

    Args:
        audio: An ``AUDIO`` mapping of ``waveform`` and ``sample_rate``, or anything else.
        silence_threshold: Loudest sample that still counts as silence, 0.0 to 1.0.

    Returns:
        ``(duration, sample_rate, channels, samples, batch_size, is_silent, peak, rms,
        summary)``, zeroed where nothing readable arrived.
    """
    rate = sample_rate_of(audio)
    waveform = audio.get("waveform") if isinstance(audio, dict) else None
    batch_size, channels, samples = dimensions(waveform)
    peak, rms = levels(waveform)
    duration = samples / rate if rate and samples else 0.0
    silent = peak <= silence_threshold

    if not rate and not samples:
        summary = "no audio"
    else:
        summary = (
            f"{rate} Hz, {channels} ch, {samples} samples, {duration:.3f} s, "
            f"batch {batch_size}, peak {peak:.4g}, rms {rms:.4g}"
        )
        if silent:
            summary += ", silent"
    return duration, rate, channels, samples, batch_size, silent, peak, rms, summary


class AudioMetadata(io.ComfyNode):
    """Split an AUDIO's length, shape and level into the numbers a graph wires."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioMetadata",
            display_name="Audio Metadata",
            search_aliases=[
                "WASAudioMetadata",
                "Audio Metadata",
                "audio duration",
                "sample rate",
                "audio channels",
                "audio peak",
                "audio rms",
                "silent audio",
            ],
            category="WAS Suite/IO",
            description=(
                "Read how long an AUDIO plays, its sample rate, channels, samples, clip count "
                "and peak and average level, as separate numbers and one line of text drawn "
                "on the node. Missing or unreadable sound reads as zeroes with is_silent "
                "true, so a graph can branch on silence before an encoder."
            ),
            inputs=[
                io.Audio.Input(
                    "audio",
                    tooltip=(
                        "The sound to measure, from Load Video (Advanced), Load Video "
                        "(Upload) or any node with an AUDIO output. A file with no sound "
                        "track answers nothing on that socket, which reads here as zeroes "
                        "rather than an error."
                    ),
                ),
                io.Float.Input(
                    "silence_threshold",
                    default=0.0, min=0.0, max=1.0, step=0.0001, round=0.00001,
                    tooltip=(
                        "Loudest sample that still counts as silence, on a 0.0 to 1.0 scale. "
                        "0.0 = only a track silent to the last bit; 0.001 = about -60 dBFS, "
                        "which also catches a noise floor or a dither tail."
                    ),
                ),
            ],
            outputs=[
                io.Float.Output(
                    display_name="duration",
                    tooltip=(
                        "Seconds one clip plays for, which is samples divided by "
                        "sample_rate. 0.0 where there is no sound. Feed it to a frame count "
                        "so a render comes out the length of the music."
                    ),
                ),
                io.Int.Output(
                    display_name="sample_rate",
                    tooltip=(
                        "Samples per second the clip was recorded at: 44100 for CD audio, "
                        "48000 for most video. 0 where nothing readable arrived. Compare "
                        "two of them before mixing clips that would not line up."
                    ),
                ),
                io.Int.Output(
                    display_name="channels",
                    tooltip=(
                        "How many channels one clip holds: 1 = mono, 2 = stereo. 0 where "
                        "nothing readable arrived."
                    ),
                ),
                io.Int.Output(
                    display_name="samples",
                    tooltip=(
                        "How many samples each channel of one clip holds. At 48000 Hz, "
                        "480000 samples is ten seconds."
                    ),
                ),
                io.Int.Output(
                    display_name="batch_size",
                    tooltip=(
                        "How many clips are stacked in this AUDIO. A loader answers 1; more "
                        "than that comes from a node that stacked several together."
                    ),
                ),
                io.Boolean.Output(
                    display_name="is_silent",
                    tooltip=(
                        "True when the loudest sample sits at or below silence_threshold, "
                        "and true for a missing or unreadable AUDIO. Wire it into a switch "
                        "to skip an encode that would only write a silent track."
                    ),
                ),
                io.Float.Output(
                    display_name="peak",
                    tooltip=(
                        "The loudest single sample anywhere in the batch, as a distance "
                        "from zero: 0.0 = silence, 1.0 = full scale, above 1.0 = clipped "
                        "when it is written out. Use it to catch a take that needs the "
                        "level pulled down."
                    ),
                ),
                io.Float.Output(
                    display_name="rms",
                    tooltip=(
                        "The average level across the whole batch, on the same 0.0 to 1.0 "
                        "scale as peak. It tracks how loud something sounds far better than "
                        "peak does: music mastered loud sits near 0.2, a quiet dialogue "
                        "take nearer 0.02."
                    ),
                ),
                io.String.Output(
                    display_name="summary",
                    tooltip=(
                        "The whole reading as one line, `44100 Hz, 2 ch, 132300 samples, "
                        "3.000 s, batch 1, peak 0.813, rms 0.204`, with `, silent` on the "
                        "end where it is. `no audio` where nothing readable arrived. The "
                        "same text is drawn on the node."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, audio=None, silence_threshold: float = 0.0) -> io.NodeOutput:
        """Answer every figure the audio carries.

        Args:
            audio: The sound to measure.
            silence_threshold: Loudest sample that still counts as silence.

        Returns:
            The nine figures, and the summary line drawn on the node.
        """
        duration, rate, channels, samples, batch_size, silent, peak, rms, summary = reading(
            audio, silence_threshold
        )
        return io.NodeOutput(
            duration, rate, channels, samples, batch_size, silent, peak, rms, summary,
            ui=ui.PreviewText(summary),
        )

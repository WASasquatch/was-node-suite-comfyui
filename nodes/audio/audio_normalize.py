"""Bring a clip to a target loudness, peak or RMS level, with a true peak ceiling."""

from __future__ import annotations

import math

import torch
from comfy_api.latest import io

from ...modules.audio import clip, loudness
from ...modules.interface.audio_sides import before_after

REQUIRES = "audio"

#: The levels a clip can be brought to, in menu order.
MODES = ("loudness", "peak", "true peak", "rms")


def _level(wave: torch.Tensor, rate: int, mode: str) -> tuple[float, float]:
    """The level ``mode`` measures on a ``(channels, samples)`` waveform, and its true peak, in dB."""
    figures = loudness.measure(wave, rate)
    level = {
        "loudness": figures["integrated"],
        "peak": figures["sample_peak"],
        "true peak": figures["true_peak"],
        "rms": figures["rms"],
    }[mode]
    return level, figures["true_peak"]


class AudioNormalize(io.ComfyNode):
    """Set a clip's level by measurement rather than by ear."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioNormalize",
            display_name="Audio Normalize",
            search_aliases=[
                "WASAudioNormalize", "Audio Normalize", "normalize audio", "loudness normalize",
                "lufs", "peak normalize", "audio gain",
            ],
            category="WAS Suite/Audio",
            description=(
                "Bring a clip to a target loudness in LUFS, a peak or an RMS level, never "
                "past a true peak ceiling, so clips from different sources play at one level. "
                "The node plays the clip before and after."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to level."),
                io.Combo.Input(
                    "mode", options=list(MODES), default="loudness",
                    tooltip="`loudness` = integrated LUFS, how loud it sounds; `peak` = loudest sample; "
                    "`true peak` = loudest point between samples; `rms` = average level.",
                ),
                io.Float.Input(
                    "target", default=-14.0, min=-70.0, max=0.0, step=0.1, round=0.01,
                    tooltip="Level to reach, in LUFS or dB. -14 = streaming music; -16 = podcasts; "
                    "-23 = broadcast; -1 = peak just under full scale.",
                ),
                io.Float.Input(
                    "ceiling", default=-1.0, min=-30.0, max=0.0, step=0.1, round=0.01,
                    tooltip="Highest true peak allowed in dBTP; the gain is lowered to stay under it. "
                    "-1 = safe for lossy encoding; 0 = full scale.",
                ),
                io.Boolean.Input(
                    "linked", default=False,
                    tooltip="`false` = each clip of a batch is levelled on its own; `true` = one gain "
                    "for the whole batch, measured on all of it.",
                ),
            ],
            outputs=[
                io.Audio.Output(display_name="audio", tooltip="The levelled clip."),
                io.Float.Output(display_name="gain_db", tooltip="dB of gain applied to the first clip."),
                io.Float.Output(display_name="measured", tooltip="The first clip's level before, in LUFS or dB."),
            ],
        )

    @classmethod
    def execute(cls, audio, mode="loudness", target=-14.0, ceiling=-1.0, linked=False) -> io.NodeOutput:
        """Measure and level every clip.

        Raises:
            ValueError: ``mode`` is not one this node measures.
        """
        if mode not in MODES:
            raise ValueError(f"mode {mode!r} is not one of {', '.join(MODES)}.")
        wave = clip.waveform_of(audio)
        rate = clip.rate_of(audio)
        groups = [wave] if linked else [wave[index:index + 1] for index in range(int(wave.shape[0]))]
        levelled, gains, measured = [], [], []
        for group in groups:
            flat = group.permute(1, 0, 2).reshape(int(group.shape[1]), -1)
            level, peak = _level(flat, rate, mode)
            gain = 0.0
            if level > loudness.FLOOR:
                gain = float(target) - level
                if peak > loudness.FLOOR and peak + gain > float(ceiling):
                    gain = float(ceiling) - peak
            gains.append(gain)
            measured.append(level)
            levelled.append(group * (10.0 ** (gain / 20.0)))
        out = torch.cat(levelled, dim=0)
        facts = {
            "normalize": {
                "mode": mode, "target": float(target), "ceiling": float(ceiling),
                "gain_db": [round(g, 3) for g in gains],
                "measured": [round(m, 3) if math.isfinite(m) else None for m in measured],
            },
        }
        return io.NodeOutput(
            clip.make(out, rate), gains[0], measured[0],
            ui=before_after(wave, rate, out, rate, facts),
        )

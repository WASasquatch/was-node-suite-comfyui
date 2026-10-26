"""Sharpen or soften the attack of every hit in a clip, and lengthen or shorten its sustain."""

from __future__ import annotations

import numpy as np
import torch
from comfy_api.latest import io

from ...modules.audio import clip, dynamics
from ...modules.interface.audio_sides import before_after

REQUIRES = "audio"

#: Most dB the shaper adds or takes away at any sample.
MAX_SHAPE_DB = 24.0


def shaped_db(signal: np.ndarray, rate: int, attack: float, sustain: float, speed: float) -> np.ndarray:
    """Gain of a transient shaper at every sample.

    Args:
        signal: ``(channels, samples)``.
        rate: Samples per second.
        attack: ``-1`` to ``1``: how far transients are pushed down or up.
        sustain: ``-1`` to ``1``: how far tails are pushed down or up.
        speed: Multiplies every envelope time; below 1 is snappier.

    Returns:
        Gains in dB, one per sample.
    """
    level = dynamics.detected(signal, rate)
    fast = dynamics.to_db(dynamics.smoothed(level, rate, 0.001 * speed, 0.05 * speed))
    slow = dynamics.to_db(dynamics.smoothed(level, rate, 0.025 * speed, 0.05 * speed))
    short = dynamics.to_db(dynamics.smoothed(level, rate, 0.001 * speed, 0.03 * speed))
    long = dynamics.to_db(dynamics.smoothed(level, rate, 0.001 * speed, 0.3 * speed))
    onset = np.clip(fast - slow, 0.0, MAX_SHAPE_DB)
    tail = np.clip(long - short, 0.0, MAX_SHAPE_DB)
    return np.clip(float(attack) * onset + float(sustain) * tail, -MAX_SHAPE_DB, MAX_SHAPE_DB)


class AudioTransientShaper(io.ComfyNode):
    """Shape attack and sustain without a threshold."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioTransientShaper",
            display_name="Audio Transient Shaper",
            search_aliases=[
                "WASAudioTransientShaper", "Audio Transient Shaper", "transient designer", "punch",
                "attack sustain", "drum shaper",
            ],
            category="WAS Suite/Audio/Dynamics",
            description=(
                "Make every hit in a clip punchier or softer and its ring longer or shorter, "
                "whatever its level, such as tightening a drum loop or taming a plucked guitar. "
                "No threshold to set."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to shape."),
                io.Float.Input(
                    "attack", default=0.5, min=-1.0, max=1.0, step=0.01,
                    tooltip="`1` = hits jump out; `0` = unchanged; `-1` = hits softened, as if "
                    "further away.",
                ),
                io.Float.Input(
                    "sustain", default=0.0, min=-1.0, max=1.0, step=0.01,
                    tooltip="`1` = tails and room ring longer; `0` = unchanged; `-1` = tails cut "
                    "short, dry and tight.",
                ),
                io.Float.Input(
                    "speed", default=1.0, min=0.25, max=4.0, step=0.05,
                    tooltip="Multiplies the shaper's timing. `0.5` = snappy, for hats and clicks; "
                    "`1` = drums; `2` = slow, for pianos and pads.",
                ),
                io.Float.Input(
                    "mix", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="`1` = fully shaped; `0` = the input unchanged.",
                ),
                io.Float.Input(
                    "output_db", default=0.0, min=-24.0, max=24.0, step=0.1, round=0.01,
                    tooltip="dB applied to the result. `0` = unchanged.",
                ),
            ],
            outputs=[io.Audio.Output(display_name="audio", tooltip="The shaped clip.")],
        )

    @classmethod
    def execute(cls, audio, attack=0.5, sustain=0.0, speed=1.0, mix=1.0, output_db=0.0) -> io.NodeOutput:
        """Shape every clip of the batch."""
        wave = clip.waveform_of(audio)
        rate = clip.rate_of(audio)
        results, first_gain = [], None
        for index in range(int(wave.shape[0])):
            signal = wave[index].double().numpy()
            gain = shaped_db(signal, rate, attack, sustain, speed)
            wet = dynamics.applied(signal, gain)
            out = (signal * (1.0 - mix) + wet * mix) * 10.0 ** (float(output_db) / 20.0)
            results.append(torch.from_numpy(out.astype(np.float32)))
            if first_gain is None:
                first_gain = gain
        result = torch.stack(results, dim=0)
        facts = {"dynamics": dynamics.summary("transient", first_gain, rate, attack=attack, sustain=sustain)}
        return io.NodeOutput(clip.make(result, rate), ui=before_after(wave, rate, result, rate, facts))

"""Limit how fast a clip's waveform may move, rounding off its edges into a soft, slewed distortion."""

from __future__ import annotations

import numpy as np
import torch
from comfy_api.latest import io

from ...modules.audio import clip, colour
from ...modules.interface.audio_sides import before_after

REQUIRES = "audio"


class AudioSlew(io.ComfyNode):
    """A slew limiter with separate rise and fall."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioSlew",
            display_name="Audio Slew",
            search_aliases=[
                "WASAudioSlew", "Audio Slew", "slew limiter", "slew", "rate limiter", "lag",
                "smoother", "wet slew",
            ],
            category="WAS Suite/Audio/Colour",
            description=(
                "Limit how fast a clip's waveform may rise and fall, so sharp edges round off into "
                "a soft, lopsided distortion and high frequencies dull, the slew of an analog "
                "circuit pushed too hard. Mix sets how much of the slewed signal is heard."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to slew."),
                io.Combo.Input(
                    "mode", options=list(colour.SLEW_MODES), default="limit",
                    tooltip="`limit` = steps capped at a fixed rate, triangles and grit; `smooth` = "
                    "eases towards the input, a lopsided low pass.",
                ),
                io.Float.Input(
                    "rise_ms", default=0.5, min=0.001, max=100.0, step=0.001, round=0.0001,
                    tooltip="Milliseconds a full swing from bottom to top takes. `0.05` = barely "
                    "touched; `0.5` = rounded; `5` = heavy and muffled.",
                ),
                io.Float.Input(
                    "fall_ms", default=0.5, min=0.001, max=100.0, step=0.001, round=0.0001,
                    tooltip="Milliseconds a full swing down takes. Equal to rise_ms = even; different "
                    "= lopsided, which adds even harmonics.",
                ),
                io.Float.Input(
                    "mix", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="`1` = fully slewed; `0.5` = blended; `0` = the input unchanged.",
                ),
                io.Float.Input(
                    "output_db", default=0.0, min=-24.0, max=24.0, step=0.1, round=0.01,
                    tooltip="dB applied to the result. `0` = unchanged.",
                ),
            ],
            outputs=[io.Audio.Output(display_name="audio", tooltip="The slewed clip.")],
        )

    @classmethod
    def execute(cls, audio, mode="limit", rise_ms=0.5, fall_ms=0.5, mix=1.0, output_db=0.0) -> io.NodeOutput:
        """Slew every clip of the batch."""
        wave = clip.waveform_of(audio)
        rate = clip.rate_of(audio)
        level = 10.0 ** (float(output_db) / 20.0)
        results = []
        for index in range(int(wave.shape[0])):
            signal = wave[index].double().numpy()
            wet = colour.slewed(signal, rate, rise_ms / 1000.0, fall_ms / 1000.0, mode)
            out = (signal * (1.0 - mix) + wet * mix) * level
            results.append(torch.from_numpy(out.astype(np.float32)))
        result = torch.stack(results, dim=0)
        facts = {"slew": {"mode": mode, "rise_ms": float(rise_ms), "fall_ms": float(fall_ms)}}
        return io.NodeOutput(clip.make(result, rate), ui=before_after(wave, rate, result, rate, facts))

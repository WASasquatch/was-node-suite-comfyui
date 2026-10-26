"""Ring modulate a clip, tremolo it with a carrier, or shift every frequency in it by a set amount."""

from __future__ import annotations

import numpy as np
import torch
from comfy_api.latest import io

from ...modules.audio import clip, colour, modulation
from ...modules.interface.audio_sides import before_after

REQUIRES = "audio"


class AudioRingMod(io.ComfyNode):
    """A ring modulator and frequency shifter."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioRingMod",
            display_name="Audio Ring Mod",
            search_aliases=[
                "WASAudioRingMod", "Audio Ring Mod", "ring modulator", "frequency shifter", "robot voice",
                "dalek", "bode shifter", "amplitude modulation",
            ],
            category="WAS Suite/Audio/Colour",
            description=(
                "Multiply a clip by a sine carrier for metallic, robotic ring modulation, or "
                "shift every frequency in it by a fixed number of hertz for inharmonic, "
                "barber-pole effects. The carrier can wander with an LFO."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to modulate."),
                io.Combo.Input(
                    "mode", options=list(colour.RING_MODES), default="ring",
                    tooltip="`ring` = sum and difference tones, bell and robot; `amplitude` = "
                    "tremolo at audio rates; `frequency shift` = every partial moved by frequency hertz.",
                ),
                io.Float.Input(
                    "frequency", default=300.0, min=-5000.0, max=5000.0, step=0.1,
                    tooltip="Carrier hertz, or the shift. `30` = rough buzz; `300` = robot voice; "
                    "`-50` shifted = everything slightly flat and detuned.",
                ),
                io.Float.Input(
                    "depth", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="`amplitude` only: how deep the modulation is. `1` = full; `0.3` = shimmer.",
                ),
                io.Float.Input(
                    "lfo_rate", default=0.2, min=0.01, max=20.0, step=0.01,
                    tooltip="Hertz of the LFO moving the carrier. `0.2` = slow drift; `5` = warble.",
                ),
                io.Float.Input(
                    "lfo_depth", default=0.0, min=0.0, max=2000.0, step=0.1,
                    tooltip="Hertz the carrier wanders either way. `0` = steady; `100` = sci-fi sweep.",
                ),
                io.Float.Input(
                    "mix", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="`1` = fully modulated; `0.5` = blended; `0` = the input unchanged.",
                ),
                io.Float.Input(
                    "output_db", default=0.0, min=-24.0, max=24.0, step=0.1, round=0.01,
                    tooltip="dB applied to the result. `0` = unchanged.",
                ),
            ],
            outputs=[io.Audio.Output(display_name="audio", tooltip="The modulated clip.")],
        )

    @classmethod
    def execute(cls, audio, mode="ring", frequency=300.0, depth=1.0, lfo_rate=0.2, lfo_depth=0.0, mix=1.0,
                output_db=0.0) -> io.NodeOutput:
        """Modulate every clip of the batch."""
        wave = clip.waveform_of(audio)
        rate = clip.rate_of(audio)
        level = 10.0 ** (float(output_db) / 20.0)
        length = int(wave.shape[-1])
        carrier = float(frequency) + float(lfo_depth) * modulation.lfo(length, rate, lfo_rate, "sine")
        if mode != "frequency shift":
            carrier = np.abs(carrier)
        results = []
        for index in range(int(wave.shape[0])):
            signal = wave[index].double().numpy()
            wet = colour.ringed(signal, rate, mode, carrier, depth)
            out = (signal * (1.0 - mix) + wet * mix) * level
            results.append(torch.from_numpy(out.astype(np.float32)))
        result = torch.stack(results, dim=0)
        facts = {"ring": {"mode": mode, "frequency": float(frequency), "lfo_depth": float(lfo_depth)}}
        return io.NodeOutput(clip.make(result, rate), ui=before_after(wave, rate, result, rate, facts))

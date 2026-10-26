"""Crush a clip to fewer bits and a lower sample rate, freely or in the voice of a classic sampler."""

from __future__ import annotations

import numpy as np
import torch
from comfy_api.latest import io

from ...modules.audio import clip, colour, filters
from ...modules.interface.audio_sides import before_after

REQUIRES = "audio"


class AudioBitcrusher(io.ComfyNode):
    """A bit depth and sample rate reducer with sampler voicings."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioBitcrusher",
            display_name="Audio Bitcrusher",
            search_aliases=[
                "WASAudioBitcrusher", "Audio Bitcrusher", "bitcrusher", "crusher", "sample rate reducer",
                "decimator", "SP-1200", "MPC60", "lo-fi drums", "8-bit",
            ],
            category="WAS Suite/Audio/Colour",
            description=(
                "Reduce a clip to fewer bits and a lower sample rate for grit and aliasing, freely "
                "or as an SP-1200, MPC60, S950, 8-bit console, 4-bit chip or telephone, each with "
                "its own filtering."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to crush."),
                io.Combo.Input(
                    "voicing", options=list(colour.CRUSH_VOICINGS), default="custom",
                    tooltip="`custom` = bits and rate as set; `SP-1200` = 12-bit 26 kHz, unfiltered "
                    "crunch; `MPC60` = 12-bit 40 kHz, smoother; `S950` = 12-bit 32 kHz, dark; `8-bit "
                    "console` and `4-bit chip` = game sounds.",
                ),
                io.Float.Input(
                    "bits", default=8.0, min=1.0, max=24.0, step=0.1,
                    tooltip="Bits per sample on `custom`. `16` = clean; `8` = gritty; `4` = harsh; "
                    "`1` = a square wave of the sound.",
                ),
                io.Float.Input(
                    "sample_rate", default=11025.0, min=200.0, max=192000.0, step=1.0,
                    tooltip="Hertz samples are held at on `custom`. `22050` = dull; `8000` = phone; "
                    "`2000` = ringing aliases. At or above the clip's rate nothing is held.",
                ),
                io.Float.Input(
                    "jitter", default=0.0, min=0.0, max=1.0, step=0.01,
                    tooltip="How unevenly samples are held. `0` = steady; `0.3` = a worn clock; `1` = "
                    "noisy and broken.",
                ),
                io.Combo.Input(
                    "dither", options=list(colour.DITHERS), default="none",
                    tooltip="`none` = hard steps, most grit; `tpdf` = noise instead of distortion on "
                    "quiet parts; `noise shaped` = that noise pushed up high, quieter to the ear.",
                ),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xFFFFFFFF, control_after_generate=io.ControlAfterGenerate.fixed,
                    tooltip="Seed of the jitter and dither, such as `0`. The same seed gives the same result.",
                ),
                io.Float.Input(
                    "mix", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="`1` = fully crushed; `0.5` = crunch under the clean sound; `0` = the "
                    "input unchanged.",
                ),
                io.Float.Input(
                    "output_db", default=0.0, min=-24.0, max=24.0, step=0.1, round=0.01,
                    tooltip="dB applied to the result. `0` = unchanged.",
                ),
            ],
            outputs=[io.Audio.Output(display_name="audio", tooltip="The crushed clip.")],
        )

    @classmethod
    def execute(cls, audio, voicing="custom", bits=8.0, sample_rate=11025.0, jitter=0.0, dither="none", seed=0,
                mix=1.0, output_db=0.0) -> io.NodeOutput:
        """Crush every clip of the batch."""
        from scipy.signal import sosfilt

        wave = clip.waveform_of(audio)
        rate = clip.rate_of(audio)
        preset = colour.CRUSH_VOICINGS.get(voicing)
        before = after = None
        if preset is not None:
            bits, sample_rate, before_hz, after_hz = preset
            before = filters.cut("high cut", before_hz, 24, rate) if before_hz and before_hz < 0.45 * rate else None
            after = filters.cut("high cut", after_hz, 12, rate) if after_hz and after_hz < 0.45 * rate else None
            if voicing == "telephone":
                before = np.vstack([before, filters.cut("low cut", 300.0, 12, rate)]) if before is not None else None
        level = 10.0 ** (float(output_db) / 20.0)
        results = []
        for index in range(int(wave.shape[0])):
            signal = wave[index].double().numpy()
            fed = sosfilt(before, signal, axis=-1) if before is not None else signal
            if voicing == "telephone":
                fed = np.sign(fed) * np.log1p(255.0 * np.abs(np.clip(fed, -1.0, 1.0))) / np.log1p(255.0)
            wet = colour.crushed(fed, rate, bits, sample_rate, jitter, dither, seed + index)
            if voicing == "telephone":
                wet = np.sign(wet) * (np.power(256.0, np.abs(wet)) - 1.0) / 255.0
            if after is not None:
                wet = sosfilt(after, wet, axis=-1)
            out = (signal * (1.0 - mix) + wet * mix) * level
            results.append(torch.from_numpy(out.astype(np.float32)))
        result = torch.stack(results, dim=0)
        facts = {"crusher": {"voicing": voicing, "bits": float(bits), "sample_rate": float(min(sample_rate, rate))}}
        return io.NodeOutput(clip.make(result, rate), ui=before_after(wave, rate, result, rate, facts))

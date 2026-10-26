"""Shape a clip's tone with up to 16 parametric bands, dragged into place over its spectrum."""

from __future__ import annotations

import numpy as np
import torch
from comfy_api.latest import io

from ...modules.audio import clip, equalizer
from ...modules.interface.audio_sides import before_after

REQUIRES = "audio"

#: The bands a fresh node starts with, all flat.
DEFAULT_BANDS = "off low cut 30 12; low shelf 100 0 0.7; bell 400 0 1; bell 2500 0 1; high shelf 8000 0 0.7; off high cut 18000 12"


class AudioEqualizer(io.ComfyNode):
    """A parametric equalizer."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioEqualizer",
            display_name="Audio Equalizer",
            search_aliases=[
                "WASAudioEqualizer", "Audio Equalizer", "eq", "parametric eq", "equaliser",
                "low cut", "high shelf", "tone",
            ],
            category="WAS Suite/Audio/Filter",
            description=(
                "Boost or cut parts of a clip's tone with up to 16 bells, shelves, cuts and "
                "notches, dragged into place on a curve drawn over the clip's spectrum before "
                "and after."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to equalise."),
                io.String.Input(
                    "bands", default=DEFAULT_BANDS,
                    tooltip="Bands as `kind hertz gain Q`, cuts as `kind hertz slope`, separated by "
                    "`;`: `low cut 30 24; bell 2500 -3 1.2`. `off` bypasses one. Drag them on the "
                    "curve instead.",
                ),
                io.Float.Input(
                    "output_db", default=0.0, min=-24.0, max=24.0, step=0.1, round=0.01,
                    tooltip="dB applied to the result. `0` = unchanged.",
                ),
            ],
            outputs=[io.Audio.Output(display_name="audio", tooltip="The equalised clip.")],
        )

    @classmethod
    def execute(cls, audio, bands=DEFAULT_BANDS, output_db=0.0) -> io.NodeOutput:
        """Equalise every clip of the batch.

        Raises:
            ValueError: ``bands`` holds an entry that cannot be read.
        """
        from scipy.signal import sosfilt

        wave = clip.waveform_of(audio)
        rate = clip.rate_of(audio)
        parsed = equalizer.parse_bands(bands)
        # A bell or a shelf at 0 dB is left out of the cascade, so a flat EQ is exact.
        active = [band for band in parsed if band.kind in equalizer.CUTS or abs(band.gain) > 1e-9
                  or band.kind in ("notch", "band pass")]
        chain = equalizer.sections(active, rate)
        level = 10.0 ** (float(output_db) / 20.0)
        results = []
        for index in range(int(wave.shape[0])):
            signal = wave[index].double().numpy()
            out = sosfilt(chain, signal, axis=-1) if len(chain) else signal
            results.append(torch.from_numpy((out * level).astype(np.float32)))
        result = torch.stack(results, dim=0)
        freqs, before = equalizer.spectrum(wave[0].double().numpy(), rate)
        _, after = equalizer.spectrum(result[0].double().numpy(), rate)
        facts = {"eq": {
            "rate": rate, "output_db": float(output_db), "bands": [band.as_dict() for band in parsed],
            "spectrum": {"freqs": freqs, "in": before, "out": after},
        }}
        return io.NodeOutput(clip.make(result, rate), ui=before_after(wave, rate, result, rate, facts))

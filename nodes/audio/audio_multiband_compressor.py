"""Compress the lows, mids and highs of a clip separately, split by Linkwitz-Riley crossovers."""

from __future__ import annotations

import numpy as np
import torch
from comfy_api.latest import io

from ...modules.audio import clip, dynamics, filters
from ...modules.interface.audio_sides import before_after

REQUIRES = "audio"

#: The bands, low to high.
BANDS = ("low", "mid", "high")


class AudioMultibandCompressor(io.ComfyNode):
    """A three band compressor."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioMultibandCompressor",
            display_name="Audio Multiband Compressor",
            search_aliases=[
                "WASAudioMultibandCompressor", "Audio Multiband Compressor", "multiband",
                "multiband compressor", "mastering", "three band compressor",
            ],
            category="WAS Suite/Audio/Dynamics",
            description=(
                "Compress the lows, mids and highs of a clip each on their own, so a boomy bass "
                "or a harsh top is held without pumping the rest, such as glueing a mix before "
                "the limiter."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to compress."),
                io.Float.Input(
                    "low_split", default=200.0, min=40.0, max=1000.0, step=5.0,
                    tooltip="Hertz between the low and mid bands. `120` = kick and bass only; `200` "
                    "= typical; `400` = includes low mids.",
                ),
                io.Float.Input(
                    "high_split", default=3000.0, min=1000.0, max=12000.0, step=50.0,
                    tooltip="Hertz between the mid and high bands. `2500` = presence and up; `6000` "
                    "= air and sibilance only.",
                ),
                io.Float.Input("low_threshold_db", default=-20.0, min=-60.0, max=0.0, step=0.5,
                               tooltip="dBFS the low band starts compressing at, such as `-20`."),
                io.Float.Input("low_ratio", default=3.0, min=1.0, max=20.0, step=0.1,
                               tooltip="Ratio of the low band. `1` = off; `3` = firm."),
                io.Float.Input("low_gain_db", default=0.0, min=-24.0, max=24.0, step=0.1,
                               tooltip="dB added to the low band after compression. `0` = none."),
                io.Float.Input("mid_threshold_db", default=-20.0, min=-60.0, max=0.0, step=0.5,
                               tooltip="dBFS the mid band starts compressing at, such as `-20`."),
                io.Float.Input("mid_ratio", default=2.0, min=1.0, max=20.0, step=0.1,
                               tooltip="Ratio of the mid band. `1` = off; `2` = gentle."),
                io.Float.Input("mid_gain_db", default=0.0, min=-24.0, max=24.0, step=0.1,
                               tooltip="dB added to the mid band after compression. `0` = none."),
                io.Float.Input("high_threshold_db", default=-24.0, min=-60.0, max=0.0, step=0.5,
                               tooltip="dBFS the high band starts compressing at, such as `-24`."),
                io.Float.Input("high_ratio", default=2.5, min=1.0, max=20.0, step=0.1,
                               tooltip="Ratio of the high band. `1` = off; `2.5` = tames harshness."),
                io.Float.Input("high_gain_db", default=0.0, min=-24.0, max=24.0, step=0.1,
                               tooltip="dB added to the high band after compression. `0` = none."),
                io.Float.Input(
                    "attack_ms", default=15.0, min=0.1, max=500.0, step=0.1,
                    tooltip="Milliseconds every band takes to come down. `5` = tight; `30` = keeps "
                    "transients.",
                ),
                io.Float.Input(
                    "release_ms", default=150.0, min=5.0, max=5000.0, step=1.0,
                    tooltip="Milliseconds every band takes to recover. `80` = lively; `300` = glued.",
                ),
                io.Float.Input(
                    "mix", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="`1` = fully compressed; `0` = the input unchanged.",
                ),
                io.Float.Input(
                    "output_db", default=0.0, min=-24.0, max=24.0, step=0.1, round=0.01,
                    tooltip="dB applied to the result. `0` = unchanged.",
                ),
            ],
            outputs=[io.Audio.Output(display_name="audio", tooltip="The compressed clip.")],
        )

    @classmethod
    def execute(cls, audio, low_split=200.0, high_split=3000.0, low_threshold_db=-20.0, low_ratio=3.0,
                low_gain_db=0.0, mid_threshold_db=-20.0, mid_ratio=2.0, mid_gain_db=0.0,
                high_threshold_db=-24.0, high_ratio=2.5, high_gain_db=0.0, attack_ms=15.0,
                release_ms=150.0, mix=1.0, output_db=0.0) -> io.NodeOutput:
        """Compress every clip of the batch, band by band."""
        from scipy.signal import sosfilt

        wave = clip.waveform_of(audio)
        rate = clip.rate_of(audio)
        upper = min(float(high_split), 0.45 * rate)
        lower = min(float(low_split), upper * 0.8)
        low_a, high_a = filters.crossover(lower, rate)
        low_b, high_b = filters.crossover(upper, rate)
        settings = {
            "low": (low_threshold_db, low_ratio, low_gain_db),
            "mid": (mid_threshold_db, mid_ratio, mid_gain_db),
            "high": (high_threshold_db, high_ratio, high_gain_db),
        }
        results, first = [], None
        for index in range(int(wave.shape[0])):
            signal = wave[index].double().numpy()
            low = sosfilt(low_a, signal, axis=-1)
            rest = sosfilt(high_a, signal, axis=-1)
            # The low band through both halves of the second crossover, matching the upper bands' phase.
            low = sosfilt(high_b, low, axis=-1) + sosfilt(low_b, low, axis=-1)
            bands = {"low": low, "mid": sosfilt(low_b, rest, axis=-1), "high": sosfilt(high_b, rest, axis=-1)}
            wet = np.zeros_like(signal)
            gains = {}
            for name in BANDS:
                threshold, ratio, makeup = settings[name]
                level = dynamics.smoothed(dynamics.detected(bands[name], rate), rate, attack_ms / 1000.0, release_ms / 1000.0)
                gain = dynamics.compressed_db(dynamics.to_db(level), threshold, ratio, 6.0)
                wet += dynamics.applied(bands[name], gain + float(makeup))
                gains[name] = gain
            out = (signal * (1.0 - mix) + wet * mix) * 10.0 ** (float(output_db) / 20.0)
            results.append(torch.from_numpy(out.astype(np.float32)))
            if first is None:
                first = gains
        result = torch.stack(results, dim=0)
        deepest = np.minimum(np.minimum(first["low"], first["mid"]), first["high"])
        facts = {"dynamics": dynamics.summary(
            "multiband", deepest, rate, low_split=lower, high_split=upper,
            bands=[{"band": name, "threshold": float(settings[name][0]), "ratio": float(settings[name][1]),
                    "max_gr": round(float(-first[name].min()), 3)} for name in BANDS],
        )}
        return io.NodeOutput(clip.make(result, rate), ui=before_after(wave, rate, result, rate, facts))

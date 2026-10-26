"""Even out a clip's level, or duck it under a sidechain, with a soft-knee compressor."""

from __future__ import annotations

import numpy as np
import torch
from comfy_api.latest import io

from ...modules.audio import clip, dynamics, filters
from ...modules.interface.audio_sides import before_after

REQUIRES = "audio"


class AudioCompressor(io.ComfyNode):
    """A feed-forward compressor with sidechain, knee, lookahead and parallel mix."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioCompressor",
            display_name="Audio Compressor",
            search_aliases=[
                "WASAudioCompressor", "Audio Compressor", "compressor", "ducking", "sidechain",
                "parallel compression", "dynamics",
            ],
            category="WAS Suite/Audio/Dynamics",
            description=(
                "Turn loud passages down so a clip sits at an even level, or duck it under "
                "another clip wired into sidechain, such as music under a voice. Threshold "
                "and ratio drag on the curve drawn on the node."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to compress."),
                io.Float.Input(
                    "threshold_db", default=-18.0, min=-60.0, max=0.0, step=0.1, round=0.01,
                    tooltip="Level in dBFS above which the gain comes down. -18 = moderate; -30 = "
                    "most of the clip; -6 = only the loudest peaks.",
                ),
                io.Float.Input(
                    "ratio", default=4.0, min=1.0, max=30.0, step=0.1, round=0.01,
                    tooltip="dB in per dB out above the threshold. `1` = off; `2` = gentle; `4` = "
                    "firm; `10` and up = close to limiting.",
                ),
                io.Float.Input(
                    "knee_db", default=6.0, min=0.0, max=24.0, step=0.5,
                    tooltip="dB around the threshold the ratio eases in over. `0` = hard knee; `6` = "
                    "smooth; `12` = very gradual.",
                ),
                io.Float.Input(
                    "attack_ms", default=10.0, min=0.05, max=500.0, step=0.05, round=0.01,
                    tooltip="Milliseconds the gain takes to come down. `1` = catches transients; "
                    "`10` = keeps the punch; `50` = slow, lets drums through.",
                ),
                io.Float.Input(
                    "release_ms", default=120.0, min=5.0, max=5000.0, step=1.0,
                    tooltip="Milliseconds the gain takes to recover. `50` = fast and lively; `120` = "
                    "natural; `500` = smooth, for vocals and buses.",
                ),
                io.Combo.Input(
                    "detector", options=list(dynamics.DETECTORS), default="peak",
                    tooltip="`peak` = reacts to every sample, tighter control; `rms` = reacts to the "
                    "average level over 10 ms, smoother and closer to what the ear hears.",
                ),
                io.Float.Input(
                    "lookahead_ms", default=0.0, min=0.0, max=20.0, step=0.1,
                    tooltip="Milliseconds the gain starts coming down before a peak arrives. `0` = "
                    "off; `5` = catches sharp transients without clicks.",
                ),
                io.Float.Input(
                    "makeup_db", default=0.0, min=-24.0, max=24.0, step=0.1, round=0.01,
                    tooltip="dB added after compression to bring the level back up. `0` = none.",
                ),
                io.Boolean.Input(
                    "auto_makeup", default=False,
                    tooltip="`true` = add the gain the threshold and ratio take off at full scale, "
                    "halved, on top of makeup_db; `false` = makeup_db only.",
                ),
                io.Float.Input(
                    "sidechain_cut_hz", default=0.0, min=0.0, max=1000.0, step=1.0,
                    tooltip="Hertz below which the detector ignores the signal, so bass does not "
                    "pump the gain. `0` = off; `100` = typical for a mix.",
                ),
                io.Float.Input(
                    "mix", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="`1` = fully compressed; `0.5` = parallel compression, punch with "
                    "the dynamics kept; `0` = the input unchanged.",
                ),
                io.Float.Input(
                    "output_db", default=0.0, min=-24.0, max=24.0, step=0.1, round=0.01,
                    tooltip="dB applied to the result. `0` = unchanged.",
                ),
                io.Audio.Input(
                    "sidechain", optional=True,
                    tooltip="Another clip whose level drives the gain, such as a voice ducking "
                    "music. Empty = the clip drives itself.",
                ),
            ],
            outputs=[
                io.Audio.Output(display_name="audio", tooltip="The compressed clip."),
                io.Float.Output(display_name="max_reduction", tooltip="Most dB of gain reduction on the first clip."),
            ],
        )

    @classmethod
    def execute(cls, audio, threshold_db=-18.0, ratio=4.0, knee_db=6.0, attack_ms=10.0, release_ms=120.0,
                detector="peak", lookahead_ms=0.0, makeup_db=0.0, auto_makeup=False, sidechain_cut_hz=0.0,
                mix=1.0, output_db=0.0, sidechain=None) -> io.NodeOutput:
        """Compress every clip of the batch."""
        from scipy.signal import sosfilt

        wave = clip.waveform_of(audio)
        rate = clip.rate_of(audio)
        batch, _, length = (int(size) for size in wave.shape)
        makeup = float(makeup_db)
        if auto_makeup:
            makeup += -float(threshold_db) * (1.0 - 1.0 / max(1.0, float(ratio))) / 2.0
        cut = filters.cut("low cut", sidechain_cut_hz, 12, rate) if sidechain_cut_hz > 0 else None
        results, first_gain = [], None
        for index in range(batch):
            signal = wave[index].double().numpy()
            key = dynamics.sidechain(sidechain, rate, length, batch, index)
            key = signal if key is None else key
            if cut is not None:
                key = sosfilt(cut, key, axis=-1)
            level = dynamics.smoothed(dynamics.detected(key, rate, detector), rate, attack_ms / 1000.0, release_ms / 1000.0)
            gain = dynamics.compressed_db(dynamics.to_db(level), threshold_db, ratio, knee_db)
            gain = dynamics.looked_ahead(gain, clip.samples_at(lookahead_ms / 1000.0, rate))
            wet = dynamics.applied(signal, gain + makeup)
            out = (signal * (1.0 - mix) + wet * mix) * 10.0 ** (float(output_db) / 20.0)
            results.append(torch.from_numpy(out.astype(np.float32)))
            if first_gain is None:
                first_gain = gain
        result = torch.stack(results, dim=0)
        facts = {"dynamics": dynamics.summary(
            "compressor", first_gain, rate, threshold=threshold_db, ratio=ratio, knee=knee_db,
            makeup=makeup, sidechained=sidechain is not None,
        )}
        return io.NodeOutput(
            clip.make(result, rate), facts["dynamics"]["max_gr"],
            ui=before_after(wave, rate, result, rate, facts),
        )

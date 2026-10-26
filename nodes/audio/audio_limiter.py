"""Hold a clip under a ceiling with a lookahead brickwall limiter that reads peaks between samples."""

from __future__ import annotations

import numpy as np
import torch
from comfy_api.latest import io

from ...modules.audio import clip, dynamics
from ...modules.interface.audio_sides import before_after

REQUIRES = "audio"


class AudioLimiter(io.ComfyNode):
    """A true peak brickwall limiter."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioLimiter",
            display_name="Audio Limiter",
            search_aliases=[
                "WASAudioLimiter", "Audio Limiter", "limiter", "brickwall", "true peak limiter",
                "maximizer", "mastering limiter",
            ],
            category="WAS Suite/Audio/Dynamics",
            description=(
                "Keep every peak at or under a ceiling, reading the peaks between samples, so "
                "a clip can be made louder without clipping. The final stage of a master, "
                "before Audio Normalize or after it."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to limit."),
                io.Float.Input(
                    "gain_db", default=0.0, min=0.0, max=24.0, step=0.1, round=0.01,
                    tooltip="dB pushed into the limiter, which is how much louder the result gets. "
                    "`0` = only catch overs; `6` = noticeably louder.",
                ),
                io.Float.Input(
                    "ceiling_db", default=-1.0, min=-24.0, max=0.0, step=0.1, round=0.01,
                    tooltip="Highest level let through. `-1` = safe for lossy encoding; `-0.3` = "
                    "lossless delivery; `0` = full scale.",
                ),
                io.Float.Input(
                    "release_ms", default=100.0, min=1.0, max=2000.0, step=1.0,
                    tooltip="Milliseconds the gain takes to recover. `20` = loud and dense; `100` = "
                    "natural; `500` = transparent on sparse material.",
                ),
                io.Float.Input(
                    "lookahead_ms", default=5.0, min=0.5, max=20.0, step=0.1,
                    tooltip="Milliseconds the gain starts falling before a peak. `5` = clean; `1` = "
                    "sharper transients, more distortion on bass.",
                ),
                io.Boolean.Input(
                    "true_peak", default=True,
                    tooltip="`true` = reads peaks between samples, four times oversampled, so the "
                    "ceiling holds after encoding; `false` = sample peaks only.",
                ),
            ],
            outputs=[
                io.Audio.Output(display_name="audio", tooltip="The limited clip."),
                io.Float.Output(display_name="max_reduction", tooltip="Most dB of gain reduction on the first clip."),
            ],
        )

    @classmethod
    def execute(cls, audio, gain_db=0.0, ceiling_db=-1.0, release_ms=100.0, lookahead_ms=5.0,
                true_peak=True) -> io.NodeOutput:
        """Limit every clip of the batch."""
        wave = clip.waveform_of(audio)
        rate = clip.rate_of(audio)
        drive = 10.0 ** (float(gain_db) / 20.0)
        results, first_gain = [], None
        for index in range(int(wave.shape[0])):
            signal = wave[index].double().numpy() * drive
            gain = dynamics.limited(signal, rate, ceiling_db, release_ms / 1000.0, lookahead_ms / 1000.0, bool(true_peak))
            results.append(torch.from_numpy(dynamics.applied(signal, gain).astype(np.float32)))
            if first_gain is None:
                first_gain = gain
        result = torch.stack(results, dim=0)
        facts = {"dynamics": dynamics.summary(
            "limiter", first_gain, rate, ceiling=ceiling_db, drive=gain_db, true_peak=bool(true_peak),
        )}
        return io.NodeOutput(
            clip.make(result, rate), facts["dynamics"]["max_gr"],
            ui=before_after(wave, rate, result, rate, facts),
        )

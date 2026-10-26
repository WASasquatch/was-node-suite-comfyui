"""Warm or wreck a clip with tube, tape, transistor, diode, clip, fold and rectify saturation curves."""

from __future__ import annotations

import math

import numpy as np
import torch
from comfy_api.latest import io

from ...modules.audio import clip, colour, filters
from ...modules.interface.audio_sides import before_after

REQUIRES = "audio"

#: Oversampling choices, in menu order.
OVERSAMPLING = ("4x", "2x", "1x", "8x")


class AudioSaturator(io.ComfyNode):
    """A waveshaping saturator, oversampled."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioSaturator",
            display_name="Audio Saturator",
            search_aliases=[
                "WASAudioSaturator", "Audio Saturator", "saturation", "distortion", "overdrive",
                "tube", "tape saturation", "wavefolder", "fuzz", "clipper",
            ],
            category="WAS Suite/Audio/Colour",
            description=(
                "Push a clip into a saturation curve, from gentle tube and tape warmth to "
                "transistor fuzz, hard clipping and wavefolding, oversampled so the harmonics "
                "stay clean. The node draws the curve it applies."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to saturate."),
                io.Combo.Input(
                    "mode", options=list(colour.SATURATIONS), default="tube",
                    tooltip="`tube` and `tape` = warm and soft; `transistor` = harder fuzz; `diode` = "
                    "lopsided bite; `hard clip` = square edges; `foldback` and `wavefolder` = metallic "
                    "folds; `rectify` = octave up.",
                ),
                io.Float.Input(
                    "drive_db", default=12.0, min=0.0, max=48.0, step=0.5,
                    tooltip="dB pushed into the curve. `3` = subtle warmth; `12` = clearly driven; "
                    "`30` and up = destroyed.",
                ),
                io.Float.Input(
                    "bias", default=0.0, min=-1.0, max=1.0, step=0.01,
                    tooltip="Offsets the curve so it breaks up unevenly, adding even harmonics. `0` = "
                    "symmetric; `0.2` = classic tube warmth.",
                ),
                io.Float.Input(
                    "tone", default=0.0, min=-1.0, max=1.0, step=0.01,
                    tooltip="Tilts the result around 1 kHz, 6 dB at the extremes. `-1` = darker; `0` "
                    "= flat; `1` = brighter.",
                ),
                io.Combo.Input(
                    "oversample", options=list(OVERSAMPLING), default="4x",
                    tooltip="`4x` = clean; `8x` = cleanest for heavy drive and folding; `1x` = fastest, "
                    "harsh aliasing like a cheap pedal.",
                ),
                io.Boolean.Input(
                    "auto_gain", default=True,
                    tooltip="`true` = the result is brought back to the input's average level, so "
                    "drive changes tone, not loudness; `false` = louder as it saturates.",
                ),
                io.Float.Input(
                    "mix", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="`1` = fully saturated; `0.3` = parallel drive, body kept; `0` = the input "
                    "unchanged.",
                ),
                io.Float.Input(
                    "output_db", default=0.0, min=-24.0, max=24.0, step=0.1, round=0.01,
                    tooltip="dB applied to the result. `0` = unchanged.",
                ),
            ],
            outputs=[io.Audio.Output(display_name="audio", tooltip="The saturated clip.")],
        )

    @classmethod
    def execute(cls, audio, mode="tube", drive_db=12.0, bias=0.0, tone=0.0, oversample="4x", auto_gain=True,
                mix=1.0, output_db=0.0) -> io.NodeOutput:
        """Saturate every clip of the batch."""
        from scipy.signal import lfilter, resample_poly, sosfilt

        wave = clip.waveform_of(audio)
        rate = clip.rate_of(audio)
        factor = {"1x": 1, "2x": 2, "4x": 4, "8x": 8}.get(oversample, 4)
        drive = 10.0 ** (float(drive_db) / 20.0)
        level = 10.0 ** (float(output_db) / 20.0)
        tilt = None
        if abs(tone) > 1e-6:
            tilt = np.vstack([
                filters.biquad("low shelf", 1000.0, 0.5, -6.0 * tone, rate),
                filters.biquad("high shelf", 1000.0, 0.5, 6.0 * tone, rate),
            ])
        dc_pole = math.exp(-2.0 * math.pi * 10.0 / rate)
        results = []
        for index in range(int(wave.shape[0])):
            signal = wave[index].double().numpy()
            length = signal.shape[-1]
            up = resample_poly(signal * drive, factor, 1, axis=-1) if factor > 1 else signal * drive
            shaped = colour.saturate(up, mode, bias)
            wet = resample_poly(shaped, 1, factor, axis=-1)[:, :length] if factor > 1 else shaped
            wet = lfilter([1.0, -1.0], [1.0, -dc_pole], wet, axis=-1)
            if tilt is not None:
                wet = sosfilt(tilt, wet, axis=-1)
            if auto_gain:
                rms_in = float(np.sqrt(np.mean(signal * signal)))
                rms_out = float(np.sqrt(np.mean(wet * wet)))
                if rms_out > 1e-12 and rms_in > 1e-12:
                    wet = wet * (rms_in / rms_out)
            out = (signal * (1.0 - mix) + wet * mix) * level
            results.append(torch.from_numpy(out.astype(np.float32)))
        result = torch.stack(results, dim=0)
        facts = {"saturator": {
            "mode": mode, "drive_db": float(drive_db), "bias": float(bias),
            "curve": colour.transfer_curve(mode, drive, bias),
        }}
        return io.NodeOutput(clip.make(result, rate), ui=before_after(wave, rate, result, rate, facts))

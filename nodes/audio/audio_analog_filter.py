"""Run a clip through a saturating analog filter voicing, its cutoff moved by an envelope and an LFO."""

from __future__ import annotations

import numpy as np
import torch
from comfy_api.latest import io

from ...modules.audio import analog, clip, dynamics, modulation
from ...modules.interface.audio_sides import before_after

REQUIRES = "audio"

#: Octaves the envelope moves the cutoff at full amount.
ENVELOPE_OCTAVES = 6.0

#: Oversampling choices, in menu order.
OVERSAMPLING = ("2x", "1x", "4x")

#: Most points of the cutoff curve handed to the panel.
CURVE_POINTS = 600


class AudioAnalogFilter(io.ComfyNode):
    """A resonant analog filter with drive, envelope follower and LFO."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioAnalogFilter",
            display_name="Audio Analog Filter",
            search_aliases=[
                "WASAudioAnalogFilter", "Audio Analog Filter", "moog ladder", "diode ladder", "303",
                "ms-20", "sem filter", "resonant filter", "auto wah", "envelope filter", "filter sweep",
            ],
            category="WAS Suite/Audio/Filter",
            description=(
                "Run a clip through a resonant analog filter, a Moog-style ladder, a 303-style "
                "diode, an MS-20 or an SEM, driven into its saturation, with the cutoff swept by "
                "the clip's own envelope for auto-wah or by an LFO."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to filter."),
                io.Combo.Input(
                    "model", options=list(analog.MODELS), default="ladder",
                    tooltip="`ladder` = warm 24 dB Moog style; `diode` = squelchy 303 style; `ms-20` "
                    "= screaming 12 dB; `sem` = smooth 12 dB with every response.",
                ),
                io.Combo.Input(
                    "mode", options=list(analog.MODES), default="low pass",
                    tooltip="`low pass` = darker; `high pass` = thinner; `band pass` = telephone "
                    "and wah; `notch` = phasey hole, `sem` only. Falls back to `low pass` where a "
                    "voicing has no such response.",
                ),
                io.Float.Input(
                    "cutoff", default=1200.0, min=20.0, max=20000.0, step=1.0,
                    tooltip="Hertz the filter bends at. `200` = muffled; `1200` = warm; `5000` = "
                    "bright. Drag on the curve.",
                ),
                io.Float.Input(
                    "resonance", default=0.3, min=0.0, max=1.0, step=0.01,
                    tooltip="Peak at the cutoff. `0` = none; `0.6` = vocal and squelchy; `0.95` and "
                    "up = rings and self-oscillates.",
                ),
                io.Float.Input(
                    "drive_db", default=6.0, min=0.0, max=36.0, step=0.5,
                    tooltip="dB pushed into the filter's saturation. `0` = clean; `6` = warm; `24` = "
                    "growling. The level is brought back down after.",
                ),
                io.Float.Input(
                    "envelope", default=0.0, min=-1.0, max=1.0, step=0.01,
                    tooltip="How far the clip's own level moves the cutoff, up to 6 octaves. `0.5` = "
                    "auto-wah that opens on every hit; `-0.5` = closes on hits.",
                ),
                io.Float.Input(
                    "lfo_depth", default=0.0, min=0.0, max=4.0, step=0.01,
                    tooltip="Octaves the LFO sweeps the cutoff either way. `0` = off; `1` = classic "
                    "sweep; `3` = dramatic.",
                ),
                io.Float.Input(
                    "lfo_rate", default=0.5, min=0.01, max=40.0, step=0.01,
                    tooltip="LFO cycles per second, when bpm is 0. `0.25` = slow sweep; `6` = "
                    "wobble.", advanced=True,
                ),
                io.Combo.Input(
                    "lfo_shape", options=list(modulation.LFO_SHAPES), default="sine",
                    tooltip="`sine` = smooth; `triangle` = even; `square` = jumps; `saw up` and `saw "
                    "down` = ramps; `random steps` = sample and hold.", advanced=True,
                ),
                io.Float.Input(
                    "bpm", default=0.0, min=0.0, max=400.0, step=0.1,
                    tooltip="Tempo the LFO locks to, one cycle per division. `0` = use lfo_rate.",
                    advanced=True,
                ),
                io.Combo.Input(
                    "division", options=list(modulation.DIVISIONS), default="1/4",
                    tooltip="Note length of one LFO cycle at bpm: `1/4` = a beat; `1/8d` = dotted "
                    "eighth; `1/16t` = triplet sixteenth.", advanced=True,
                ),
                io.Float.Input(
                    "attack_ms", default=5.0, min=0.1, max=1000.0, step=0.1,
                    tooltip="Milliseconds the envelope takes to rise. `2` = snappy wah; `50` = slow "
                    "bloom.", advanced=True,
                ),
                io.Float.Input(
                    "release_ms", default=150.0, min=1.0, max=5000.0, step=1.0,
                    tooltip="Milliseconds the envelope takes to fall. `80` = quick; `400` = lazy.",
                    advanced=True,
                ),
                io.Combo.Input(
                    "oversample", options=list(OVERSAMPLING), default="2x",
                    tooltip="`2x` = clean at high cutoffs and drive; `1x` = half the time, may alias; "
                    "`4x` = cleanest, slowest.", advanced=True,
                ),
                io.Float.Input(
                    "mix", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="`1` = fully filtered; `0` = the input unchanged.",
                ),
                io.Float.Input(
                    "output_db", default=0.0, min=-24.0, max=24.0, step=0.1, round=0.01,
                    tooltip="dB applied to the result. `0` = unchanged.",
                ),
            ],
            outputs=[io.Audio.Output(display_name="audio", tooltip="The filtered clip.")],
        )

    @classmethod
    def execute(cls, audio, model="ladder", mode="low pass", cutoff=1200.0, resonance=0.3, drive_db=6.0,
                envelope=0.0, lfo_depth=0.0, lfo_rate=0.5, lfo_shape="sine", bpm=0.0, division="1/4",
                attack_ms=5.0, release_ms=150.0, oversample="2x", mix=1.0, output_db=0.0) -> io.NodeOutput:
        """Filter every clip of the batch."""
        from scipy.signal import resample_poly

        wave = clip.waveform_of(audio)
        rate = clip.rate_of(audio)
        factor = {"1x": 1, "2x": 2, "4x": 4}.get(oversample, 2)
        inner = rate * factor
        drive = 10.0 ** (float(drive_db) / 20.0)
        level = 10.0 ** (float(output_db) / 20.0)
        speed = modulation.rate_of_division(division, bpm) if bpm > 0 else float(lfo_rate)
        results, first_curve = [], None
        for index in range(int(wave.shape[0])):
            signal = wave[index].double().numpy()
            length = signal.shape[-1]
            octaves = np.zeros(length)
            if envelope:
                follow = dynamics.to_db(modulation.envelope(signal, rate, attack_ms / 1000.0, release_ms / 1000.0))
                octaves += float(envelope) * ENVELOPE_OCTAVES * np.clip((follow + 60.0) / 60.0, 0.0, 1.0)
            if lfo_depth:
                octaves += float(lfo_depth) * modulation.lfo(length, rate, speed, lfo_shape)
            hertz = np.clip(float(cutoff) * 2.0 ** octaves, 20.0, 0.45 * inner)
            driven = signal * drive
            if factor > 1:
                driven = resample_poly(driven, factor, 1, axis=-1)
                steps = np.repeat(hertz, factor)[: driven.shape[-1]]
            else:
                steps = hertz
            wet = analog.filtered(driven, analog.coefficient(steps, inner, model), resonance, model, mode)
            if factor > 1:
                wet = resample_poly(wet, 1, factor, axis=-1)[:, :length]
            wet = wet / max(1.0, drive ** 0.5)
            out = (signal * (1.0 - mix) + wet * mix) * level
            results.append(torch.from_numpy(out.astype(np.float32)))
            if first_curve is None:
                first_curve = hertz
        result = torch.stack(results, dim=0)
        edges = np.linspace(0, first_curve.size, min(CURVE_POINTS, first_curve.size) + 1).astype(np.int64)
        trace = [round(float(np.exp(np.log(first_curve[a:b]).mean())), 1) for a, b in zip(edges[:-1], edges[1:]) if b > a]
        facts = {"filter": {
            "model": model, "mode": mode if mode in analog.modes_of(model) else "low pass",
            "cutoff": float(cutoff), "resonance": float(resonance), "drive_db": float(drive_db),
            "modes": list(analog.modes_of(model)),
            "cutoff_curve": {"offset": 0.0, "step": first_curve.size / max(1, len(trace)) / rate, "values": trace},
        }}
        return io.NodeOutput(clip.make(result, rate), ui=before_after(wave, rate, result, rate, facts))

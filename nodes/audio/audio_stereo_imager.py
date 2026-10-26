"""Widen or narrow a clip's stereo image, fold its bass to mono, and spread a mono clip with a short delay."""

from __future__ import annotations

import math

import numpy as np
import torch
from comfy_api.latest import io

from ...modules.audio import clip, delays, filters
from ...modules.interface.audio_sides import before_after

REQUIRES = "audio"

#: Which channel a Haas delay lands on.
SIDES = ("right", "left")


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    """The correlation of two channels, 1 for mono and -1 for opposite polarity."""
    a, b = left - left.mean(), right - right.mean()
    spread = math.sqrt(float(np.dot(a, a)) * float(np.dot(b, b)))
    return round(float(np.dot(a, b)) / spread, 4) if spread > 0 else 1.0


def _width(left: np.ndarray, right: np.ndarray) -> float:
    """Side level over mid level, 0 for mono."""
    mid, side = (left + right) / 2.0, (left - right) / 2.0
    mid_rms = math.sqrt(float(np.mean(mid * mid)))
    return round(math.sqrt(float(np.mean(side * side))) / mid_rms, 4) if mid_rms > 0 else 0.0


class AudioStereoImager(io.ComfyNode):
    """A mid and side stereo width tool."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioStereoImager",
            display_name="Audio Stereo Imager",
            search_aliases=[
                "WASAudioStereoImager", "Audio Stereo Imager", "stereo width", "stereo widener",
                "mono bass", "mid side", "haas", "vectorscope",
            ],
            category="WAS Suite/Audio",
            description=(
                "Widen or narrow a clip's stereo image, fold the bass to mono so it stays tight on "
                "every system, and spread a mono clip with a short Haas delay. The node draws a "
                "vectorscope and the correlation."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to place. A mono clip comes out as stereo."),
                io.Float.Input(
                    "width", default=1.0, min=0.0, max=2.0, step=0.01,
                    tooltip="`0` = mono; `1` = unchanged; `1.5` = wider; `2` = very wide, check it in "
                    "mono.",
                ),
                io.Float.Input(
                    "mono_below_hz", default=0.0, min=0.0, max=500.0, step=1.0,
                    tooltip="Hertz below which everything is folded to mono. `0` = off; `120` = tight "
                    "bass for a mix; `200` = vinyl safe.",
                ),
                io.Float.Input(
                    "haas_ms", default=0.0, min=0.0, max=30.0, step=0.1,
                    tooltip="Milliseconds one channel is delayed, which spreads a mono clip. `0` = "
                    "off; `12` = wide; past `25` it starts to sound like an echo.",
                ),
                io.Combo.Input(
                    "haas_side", options=list(SIDES), default="right",
                    tooltip="`right` = the right channel is delayed, the sound leans left; `left` = "
                    "the reverse.",
                ),
                io.Float.Input(
                    "balance", default=0.0, min=-1.0, max=1.0, step=0.01,
                    tooltip="`-1` = left only; `0` = centred; `1` = right only.",
                ),
                io.Float.Input(
                    "output_db", default=0.0, min=-24.0, max=24.0, step=0.1, round=0.01,
                    tooltip="dB applied to the result. `0` = unchanged.",
                ),
            ],
            outputs=[
                io.Audio.Output(display_name="audio", tooltip="The placed clip, in stereo."),
                io.Float.Output(display_name="correlation", tooltip="Correlation of the first clip's channels after, 1 = mono, 0 = wide, below 0 = cancels in mono."),
            ],
        )

    @classmethod
    def execute(cls, audio, width=1.0, mono_below_hz=0.0, haas_ms=0.0, haas_side="right", balance=0.0,
                output_db=0.0) -> io.NodeOutput:
        """Place every clip of the batch."""
        from scipy.signal import sosfilt

        wave = clip.waveform_of(audio)
        rate = clip.rate_of(audio)
        stereo = clip.with_channels(wave, 2)
        level = 10.0 ** (float(output_db) / 20.0)
        split = filters.crossover(mono_below_hz, rate) if mono_below_hz > 0 else None
        pan = max(-1.0, min(1.0, float(balance)))
        left_gain = math.cos(max(pan, 0.0) * math.pi / 2.0)
        right_gain = math.cos(max(-pan, 0.0) * math.pi / 2.0)
        results, facts = [], None
        for index in range(int(stereo.shape[0])):
            pair = stereo[index].double().numpy()
            out = pair.copy()
            if abs(float(width) - 1.0) > 1e-9 or split is not None:
                mid, side = (pair[0] + pair[1]) / 2.0, (pair[0] - pair[1]) / 2.0
                side = side * float(width)
                if split is not None:
                    low, high = split
                    side = sosfilt(high, side)
                    mid = sosfilt(low, mid) + sosfilt(high, mid)
                out = np.stack([mid + side, mid - side])
            if haas_ms > 0:
                lane = 1 if haas_side == "right" else 0
                out[lane] = delays.read_delayed(out[lane:lane + 1], haas_ms * rate / 1000.0)[0]
            out[0] *= left_gain
            out[1] *= right_gain
            out *= level
            results.append(torch.from_numpy(out.astype(np.float32)))
            if facts is None:
                facts = {"stereo": {
                    "correlation_in": _correlation(pair[0], pair[1]), "correlation_out": _correlation(out[0], out[1]),
                    "width_in": _width(pair[0], pair[1]), "width_out": _width(out[0], out[1]),
                }}
        result = torch.stack(results, dim=0)
        return io.NodeOutput(
            clip.make(result, rate), facts["stereo"]["correlation_out"],
            ui=before_after(wave, rate, result, rate, facts),
        )

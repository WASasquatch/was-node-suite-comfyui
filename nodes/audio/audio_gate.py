"""Silence or push down the quiet parts of a clip with a noise gate or a downward expander."""

from __future__ import annotations

import numpy as np
import torch
from comfy_api.latest import io

from ...modules.audio import clip, dynamics, filters
from ...modules.interface.audio_sides import before_after

REQUIRES = "audio"

#: How the quiet parts are treated, in menu order.
MODES = ("gate", "expander")


class AudioGate(io.ComfyNode):
    """A noise gate with hysteresis and hold, or a soft downward expander."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioGate",
            display_name="Audio Gate",
            search_aliases=[
                "WASAudioGate", "Audio Gate", "noise gate", "expander", "downward expander",
                "remove background noise", "gate drums",
            ],
            category="WAS Suite/Audio/Dynamics",
            description=(
                "Shut out the quiet parts of a clip, such as room noise between words or bleed "
                "between drum hits, with a gate that opens and closes or an expander that eases "
                "them down. A sidechain input lets another clip open it."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to gate."),
                io.Combo.Input(
                    "mode", options=list(MODES), default="gate",
                    tooltip="`gate` = opens and closes, with hold and hysteresis; `expander` = turns "
                    "quiet parts down smoothly by ratio, gentler on breaths and tails.",
                ),
                io.Float.Input(
                    "threshold_db", default=-40.0, min=-90.0, max=0.0, step=0.5,
                    tooltip="Level in dBFS the gate opens at, or below which the expander turns down. "
                    "`-40` = speech in a quiet room; `-25` = noisy room.",
                ),
                io.Float.Input(
                    "depth_db", default=40.0, min=0.0, max=90.0, step=0.5,
                    tooltip="dB the quiet parts are turned down by. `10` = gentle; `40` = near "
                    "silence; `90` = off entirely.",
                ),
                io.Float.Input(
                    "ratio", default=3.0, min=1.0, max=20.0, step=0.1,
                    tooltip="Expander only: dB of fall per dB under the threshold. `2` = gentle; `4` "
                    "= firm.",
                ),
                io.Float.Input(
                    "hysteresis_db", default=4.0, min=0.0, max=20.0, step=0.5,
                    tooltip="Gate only: dB under the threshold the level falls before the gate closes, "
                    "so it does not chatter. `4` = typical.",
                ),
                io.Float.Input(
                    "attack_ms", default=1.0, min=0.01, max=100.0, step=0.01,
                    tooltip="Milliseconds to open. `0.1` = snaps open for drums; `5` = soft for voice.",
                ),
                io.Float.Input(
                    "hold_ms", default=50.0, min=0.0, max=2000.0, step=1.0,
                    tooltip="Gate only: milliseconds it stays open after the level drops. `50` = "
                    "speech; `200` = keeps a decaying note.",
                ),
                io.Float.Input(
                    "release_ms", default=150.0, min=1.0, max=5000.0, step=1.0,
                    tooltip="Milliseconds to close. `50` = tight; `300` = natural tails.",
                ),
                io.Float.Input(
                    "lookahead_ms", default=2.0, min=0.0, max=10.0, step=0.1,
                    tooltip="Milliseconds the gate opens ahead of the sound, so the first transient "
                    "is not cut. `0` = off.",
                ),
                io.Float.Input(
                    "sidechain_cut_hz", default=0.0, min=0.0, max=2000.0, step=1.0,
                    tooltip="Hertz below which the detector ignores the signal, so rumble does not "
                    "open it. `0` = off; `150` = typical.",
                ),
                io.Float.Input(
                    "output_db", default=0.0, min=-24.0, max=24.0, step=0.1, round=0.01,
                    tooltip="dB applied to the result. `0` = unchanged.",
                ),
                io.Audio.Input(
                    "sidechain", optional=True,
                    tooltip="Another clip whose level opens the gate, such as a kick opening a bass "
                    "line. Empty = the clip opens itself.",
                ),
            ],
            outputs=[
                io.Audio.Output(display_name="audio", tooltip="The gated clip."),
                io.Float.Output(display_name="open_share", tooltip="Share of the first clip the gate was open for, 0 to 1."),
            ],
        )

    @classmethod
    def execute(cls, audio, mode="gate", threshold_db=-40.0, depth_db=40.0, ratio=3.0, hysteresis_db=4.0,
                attack_ms=1.0, hold_ms=50.0, release_ms=150.0, lookahead_ms=2.0, sidechain_cut_hz=0.0,
                output_db=0.0, sidechain=None) -> io.NodeOutput:
        """Gate every clip of the batch."""
        from scipy.signal import sosfilt

        wave = clip.waveform_of(audio)
        rate = clip.rate_of(audio)
        batch, _, length = (int(size) for size in wave.shape)
        cut = filters.cut("low cut", sidechain_cut_hz, 12, rate) if sidechain_cut_hz > 0 else None
        ahead = clip.samples_at(lookahead_ms / 1000.0, rate)
        results, first_gain = [], None
        for index in range(batch):
            signal = wave[index].double().numpy()
            key = dynamics.sidechain(sidechain, rate, length, batch, index)
            key = signal if key is None else key
            if cut is not None:
                key = sosfilt(cut, key, axis=-1)
            if mode == "expander":
                level = dynamics.smoothed(dynamics.detected(key, rate, "rms"), rate, attack_ms / 1000.0, release_ms / 1000.0)
                gain = dynamics.expanded_db(dynamics.to_db(level), threshold_db, ratio, depth_db, 6.0)
            else:
                gain = dynamics.gated_db(
                    dynamics.to_db(dynamics.detected(key, rate)), rate, threshold_db, hysteresis_db,
                    depth_db, attack_ms / 1000.0, hold_ms / 1000.0, release_ms / 1000.0,
                )
            if ahead:
                early = np.concatenate([gain[ahead:], np.full(min(ahead, gain.size), gain[-1])])[: gain.size]
                gain = np.maximum(gain, early)
            out = dynamics.applied(signal, gain) * 10.0 ** (float(output_db) / 20.0)
            results.append(torch.from_numpy(out.astype(np.float32)))
            if first_gain is None:
                first_gain = gain
        result = torch.stack(results, dim=0)
        open_share = float(np.mean(first_gain > -1.0)) if first_gain.size else 0.0
        facts = {"dynamics": dynamics.summary(
            "gate" if mode != "expander" else "expander", first_gain, rate,
            threshold=threshold_db, depth=depth_db, ratio=ratio, hysteresis=hysteresis_db,
            open_share=round(open_share, 4), sidechained=sidechain is not None,
        )}
        return io.NodeOutput(
            clip.make(result, rate), open_share,
            ui=before_after(wave, rate, result, rate, facts),
        )

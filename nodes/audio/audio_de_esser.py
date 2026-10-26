"""Tame harsh s, sh and t sounds in a voice by turning down only the sibilant band when it flares."""

from __future__ import annotations

import numpy as np
import torch
from comfy_api.latest import io

from ...modules.audio import clip, dynamics, filters
from ...modules.interface.audio_sides import before_after

REQUIRES = "audio"

#: How the reduction is applied, in menu order.
MODES = ("split band", "wideband", "listen")


class AudioDeEsser(io.ComfyNode):
    """A sibilance compressor keyed on a high band."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioDeEsser",
            display_name="Audio De-Esser",
            search_aliases=[
                "WASAudioDeEsser", "Audio De-Esser", "de-esser", "deesser", "sibilance",
                "harsh s", "vocal",
            ],
            category="WAS Suite/Audio/Dynamics",
            description=(
                "Turn down harsh s, sh and t sounds in a voice only when they flare, leaving the "
                "rest untouched, such as smoothing a bright text-to-speech take. Listen mode "
                "plays the band being caught, to tune the frequency."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The voice to de-ess."),
                io.Float.Input(
                    "frequency", default=6000.0, min=2000.0, max=16000.0, step=50.0,
                    tooltip="Hertz the sibilance band starts at. `5000` = deeper voices; `6000` = "
                    "most voices; `8000` = bright female voices and TTS.",
                ),
                io.Float.Input(
                    "threshold_db", default=-30.0, min=-60.0, max=0.0, step=0.5,
                    tooltip="Level in dBFS of the band above which it is turned down. Lower catches "
                    "more: `-40` = heavy; `-20` = only the worst.",
                ),
                io.Float.Input(
                    "range_db", default=8.0, min=0.0, max=24.0, step=0.5,
                    tooltip="Most dB the band is turned down. `4` = subtle; `8` = typical; `16` = "
                    "lisping.",
                ),
                io.Combo.Input(
                    "mode", options=list(MODES), default="split band",
                    tooltip="`split band` = only the high band goes down; `wideband` = the whole "
                    "voice dips, more natural on light sibilance; `listen` = output the band caught.",
                ),
                io.Float.Input(
                    "output_db", default=0.0, min=-24.0, max=24.0, step=0.1, round=0.01,
                    tooltip="dB applied to the result. `0` = unchanged.",
                ),
            ],
            outputs=[
                io.Audio.Output(display_name="audio", tooltip="The de-essed voice, or the band caught in listen mode."),
                io.Float.Output(display_name="max_reduction", tooltip="Most dB the band was turned down on the first clip."),
            ],
        )

    @classmethod
    def execute(cls, audio, frequency=6000.0, threshold_db=-30.0, range_db=8.0, mode="split band",
                output_db=0.0) -> io.NodeOutput:
        """De-ess every clip of the batch."""
        from scipy.signal import sosfilt

        wave = clip.waveform_of(audio)
        rate = clip.rate_of(audio)
        low, high = filters.crossover(min(frequency, 0.45 * rate), rate)
        results, first_gain = [], None
        for index in range(int(wave.shape[0])):
            signal = wave[index].double().numpy()
            band = sosfilt(high, signal, axis=-1)
            level = dynamics.smoothed(dynamics.detected(band, rate), rate, 0.0005, 0.06)
            gain = np.maximum(dynamics.compressed_db(dynamics.to_db(level), threshold_db, 8.0, 4.0), -abs(range_db))
            gain = dynamics.looked_ahead(gain, clip.samples_at(0.002, rate))
            if mode == "listen":
                out = band
            elif mode == "wideband":
                out = dynamics.applied(signal, gain)
            else:
                out = sosfilt(low, signal, axis=-1) + dynamics.applied(band, gain)
            out = out * 10.0 ** (float(output_db) / 20.0)
            results.append(torch.from_numpy(out.astype(np.float32)))
            if first_gain is None:
                first_gain = gain
        result = torch.stack(results, dim=0)
        facts = {"dynamics": dynamics.summary(
            "de-esser", first_gain, rate, threshold=threshold_db, depth=range_db, frequency=frequency, mode=mode,
        )}
        return io.NodeOutput(
            clip.make(result, rate), facts["dynamics"]["max_gr"],
            ui=before_after(wave, rate, result, rate, facts),
        )

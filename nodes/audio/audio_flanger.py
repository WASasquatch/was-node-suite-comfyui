"""Sweep a short delayed copy against a clip: a classic or a through-zero flanger."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.audio import clip, modulation
from ...modules.audio import modulation_effects as effects
from ...modules.interface.audio_sides import before_after

REQUIRES = "audio"


class AudioFlanger(io.ComfyNode):
    """A flanger with feedback, tempo sync and a through-zero mode."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioFlanger",
            display_name="Audio Flanger",
            search_aliases=[
                "WASAudioFlanger", "Audio Flanger", "flanger", "flange", "through-zero", "tape flange",
                "jet", "comb sweep", "modulation",
            ],
            category="WAS Suite/Audio/Modulation",
            description=(
                "Sweep a very short delayed copy against a clip for the jet-plane whoosh of a "
                "flanger, with feedback for a metallic ring and a through-zero mode that cancels "
                "like two tapes running in sync. The sweep follows a rate or a tempo."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to flange."),
                io.Combo.Input(
                    "mode", options=list(effects.FLANGER_MODES), default="classic",
                    tooltip="`classic` = a delayed copy sweeps against the clip, the jet whoosh; "
                    "`through-zero` = an inverted copy sweeps past the clip itself, cancelling at each "
                    "crossing like two tapes in sync.",
                ),
                io.Float.Input(
                    "rate", default=0.25, min=0.01, max=10.0, step=0.01, round=0.001,
                    tooltip="Hertz of the sweep. `0.1` = one pass every 10 seconds; `0.25` = typical; "
                    "`2` = fast. Unused when bpm is above 0.",
                ),
                io.Float.Input(
                    "depth", default=0.7, min=0.0, max=1.0, step=0.01,
                    tooltip="Share of manual swept either side. `0` = a fixed comb; `0.5` = from half to "
                    "one and a half times manual; `1` = from 0 to twice manual.",
                ),
                io.Float.Input(
                    "manual", default=2.0, min=0.1, max=10.0, step=0.01, round=0.001,
                    tooltip="Milliseconds of the centre delay. `0.5` = a high, bright sweep; `2` = "
                    "typical; `8` = low, close to a chorus.",
                ),
                io.Float.Input(
                    "feedback", default=0.5, min=-0.95, max=0.95, step=0.01,
                    tooltip="Share of the delayed copy fed back. `0` = gentle; `0.7` = a metallic ring; "
                    "`-0.7` = hollow and nasal. The average level is held as it rises.",
                ),
                io.Float.Input(
                    "spread", default=0.0, min=0.0, max=0.5, step=0.01,
                    tooltip="Share of a cycle the right channel's sweep runs ahead of the left. `0` = "
                    "the same on both sides; `0.25` = swirling; `0.5` = opposite. A mono clip stays mono.",
                ),
                io.Combo.Input(
                    "shape", options=list(modulation.LFO_SHAPES), default="triangle",
                    tooltip="`triangle` = an even sweep up and down; `sine` = lingers at the ends; `saw "
                    "up` and `saw down` = one way, then back; `random smooth` = wandering. Jumps are "
                    "smoothed over 2 ms.",
                ),
                io.Float.Input(
                    "bpm", default=0.0, min=0.0, max=999.0, step=0.1, round=0.01,
                    tooltip="Tempo the sweep follows. `0` = use rate; `120` = one sweep per division at "
                    "120 BPM.",
                ),
                io.Combo.Input(
                    "division", options=list(modulation.DIVISIONS), default="1/1",
                    tooltip="Note length of one sweep when bpm is above 0. `1/4` = a beat; `1/1` = a bar "
                    "of 4/4; `d` = dotted; `t` = triplet.",
                ),
                io.Float.Input(
                    "mix", default=0.5, min=0.0, max=1.0, step=0.01,
                    tooltip="`0` = the input unchanged; `0.5` = equal blend, the deepest notches; `1` = "
                    "the delayed copy only, a pitch wobble.",
                ),
                io.Float.Input(
                    "output_db", default=0.0, min=-24.0, max=24.0, step=0.1, round=0.01,
                    tooltip="dB applied to the result. `0` = unchanged.",
                ),
            ],
            outputs=[io.Audio.Output(display_name="audio", tooltip="The flanged clip, as long as the input.")],
        )

    @classmethod
    def execute(cls, audio, mode="classic", rate=0.25, depth=0.7, manual=2.0, feedback=0.5, spread=0.0,
                shape="triangle", bpm=0.0, division="1/1", mix=0.5, output_db=0.0) -> io.NodeOutput:
        """Flange every clip of the batch.

        Raises:
            ValueError: ``mode`` or ``shape`` is not one this node runs, or ``division`` is not a
                note length.
        """
        if mode not in effects.FLANGER_MODES:
            raise ValueError(f"mode {mode!r} is not one of {', '.join(effects.FLANGER_MODES)}.")
        if shape not in modulation.LFO_SHAPES:
            raise ValueError(f"shape {shape!r} is not one of {', '.join(modulation.LFO_SHAPES)}.")
        wave = clip.waveform_of(audio)
        sample_rate = clip.rate_of(audio)
        sweep_hz = effects.lfo_rate(rate, bpm, division)

        def effect(signal):
            return effects.flanger(signal, sample_rate, mode, sweep_hz, shape, depth, manual, feedback, spread)

        out = effects.applied(wave, mix, output_db, effect)
        facts = {"flanger": {
            "mode": mode, "rate_hz": round(sweep_hz, 6), "synced": float(bpm) > 0.0, "bpm": float(bpm),
            "division": division, "shape": shape, "cycle": effects.cycle(shape),
            "steps": effects.random_steps(shape, sweep_hz, wave.shape[-1] / sample_rate), "depth": float(depth),
            "manual_ms": float(manual), "feedback": float(feedback), "spread": float(spread),
            "mix": float(mix), "output_db": float(output_db),
            **effects.flanger_figures(mode, depth, manual),
        }}
        return io.NodeOutput(clip.make(out, sample_rate), ui=before_after(wave, sample_rate, out, sample_rate, facts))

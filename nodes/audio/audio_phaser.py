"""Sweep notches through a clip with a chain of allpass stages: a phaser."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.audio import clip, modulation
from ...modules.audio import modulation_effects as effects
from ...modules.interface.audio_sides import before_after

REQUIRES = "audio"


class AudioPhaser(io.ComfyNode):
    """A phaser of 2 to 12 allpass stages with feedback and tempo sync."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioPhaser",
            display_name="Audio Phaser",
            search_aliases=[
                "WASAudioPhaser", "Audio Phaser", "phaser", "phase shifter", "allpass", "swirl",
                "modulation",
            ],
            category="WAS Suite/Audio/Modulation",
            description=(
                "Sweep notches through a clip's spectrum with a chain of allpass stages, for the "
                "swirling phase sound of guitars, keys and drums. The sweep runs between two "
                "frequencies on a rate or a tempo."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to phase."),
                io.Int.Input(
                    "stages", default=4, min=2, max=12, step=2,
                    tooltip="Allpass stages; every 2 make one notch. `2` = subtle; `4` = classic; `8` = "
                    "deep and vocal; `12` = dense.",
                ),
                io.Float.Input(
                    "rate", default=0.5, min=0.01, max=10.0, step=0.01, round=0.001,
                    tooltip="Hertz of the sweep. `0.1` = a slow swirl; `0.5` = typical; `4` = a fast "
                    "warble. Unused when bpm is above 0.",
                ),
                io.Float.Input(
                    "low_hz", default=200.0, min=20.0, max=20000.0, step=1.0,
                    tooltip="Hertz the sweep reaches at its lowest. `100` = deep; `300` = typical. "
                    "Swapped with high_hz when above it.",
                ),
                io.Float.Input(
                    "high_hz", default=2000.0, min=20.0, max=20000.0, step=1.0,
                    tooltip="Hertz the sweep reaches at its highest. `1500` = mellow; `4000` = bright; "
                    "`10000` = airy. Held below half the sample rate.",
                ),
                io.Float.Input(
                    "feedback", default=0.3, min=-0.95, max=0.95, step=0.01,
                    tooltip="Share of the last stage fed back into the first. `0` = smooth; `0.7` = sharp "
                    "resonant peaks; negative = peaks moved between the notches. The average level is held.",
                ),
                io.Float.Input(
                    "spread", default=0.0, min=0.0, max=0.5, step=0.01,
                    tooltip="Share of a cycle the right channel's sweep runs ahead of the left. `0` = "
                    "the same on both sides; `0.25` = swirling; `0.5` = opposite. A mono clip stays mono.",
                ),
                io.Combo.Input(
                    "shape", options=list(modulation.LFO_SHAPES), default="sine",
                    tooltip="`sine` = a smooth sweep; `triangle` = even up and down; `saw up` and `saw "
                    "down` = one way, then back; `random smooth` = wandering. Jumps are smoothed over 2 ms.",
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
                    tooltip="`0` = the input unchanged; `0.5` = equal blend, the classic full notches; "
                    "`1` = the phase-shifted copy only, close to the input.",
                ),
                io.Float.Input(
                    "output_db", default=0.0, min=-24.0, max=24.0, step=0.1, round=0.01,
                    tooltip="dB applied to the result. `0` = unchanged.",
                ),
            ],
            outputs=[io.Audio.Output(display_name="audio", tooltip="The phased clip, as long as the input.")],
        )

    @classmethod
    def execute(cls, audio, stages=4, rate=0.5, low_hz=200.0, high_hz=2000.0, feedback=0.3, spread=0.0,
                shape="sine", bpm=0.0, division="1/1", mix=0.5, output_db=0.0) -> io.NodeOutput:
        """Phase every clip of the batch.

        Raises:
            ValueError: ``shape`` is not one this node runs, or ``division`` is not a note length.
        """
        if shape not in modulation.LFO_SHAPES:
            raise ValueError(f"shape {shape!r} is not one of {', '.join(modulation.LFO_SHAPES)}.")
        wave = clip.waveform_of(audio)
        sample_rate = clip.rate_of(audio)
        sweep_hz = effects.lfo_rate(rate, bpm, division)

        def effect(signal):
            return effects.phaser(signal, sample_rate, stages, sweep_hz, low_hz, high_hz, feedback, spread, shape)

        out = effects.applied(wave, mix, output_db, effect)
        facts = {"phaser": {
            "stages": int(stages), "rate_hz": round(sweep_hz, 6), "synced": float(bpm) > 0.0,
            "bpm": float(bpm), "division": division, "shape": shape, "cycle": effects.cycle(shape),
            "steps": effects.random_steps(shape, sweep_hz, wave.shape[-1] / sample_rate),
            "feedback": float(feedback), "spread": float(spread), "mix": float(mix),
            "output_db": float(output_db),
            **effects.phaser_figures(stages, low_hz, high_hz, sample_rate),
        }}
        return io.NodeOutput(clip.make(out, sample_rate), ui=before_after(wave, sample_rate, out, sample_rate, facts))

"""Pulse a clip's volume, or move it between the speakers: a tremolo and an auto-pan."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.audio import clip, modulation
from ...modules.audio import modulation_effects as effects
from ...modules.interface.audio_sides import before_after

REQUIRES = "audio"


class AudioTremolo(io.ComfyNode):
    """A tremolo and a constant power auto-pan with tempo sync."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioTremolo",
            display_name="Audio Tremolo",
            search_aliases=[
                "WASAudioTremolo", "Audio Tremolo", "tremolo", "auto-pan", "autopan", "pan",
                "volume pulse", "chop", "modulation",
            ],
            category="WAS Suite/Audio/Modulation",
            description=(
                "Pulse a clip's volume, or move it from side to side, at a steady rate or in time "
                "with a tempo. A mono clip auto-panned comes out stereo. The node plays the clip "
                "before and after."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to pulse or pan."),
                io.Combo.Input(
                    "mode", options=list(effects.TREMOLO_MODES), default="tremolo",
                    tooltip="`tremolo` = the volume pulses; `auto-pan` = the sound moves between left and "
                    "right at constant power, and a mono clip comes out stereo.",
                ),
                io.Float.Input(
                    "rate", default=4.0, min=0.01, max=40.0, step=0.01, round=0.001,
                    tooltip="Hertz of the pulse. `2` = a slow throb; `5` = a classic amp tremolo; `12` = "
                    "a fast flutter. Unused when bpm is above 0.",
                ),
                io.Float.Input(
                    "depth", default=0.5, min=0.0, max=1.0, step=0.01,
                    tooltip="`0` = no change; `0.5` = dips to half volume; `1` = dips to silence, or pans "
                    "fully from side to side in `auto-pan`.",
                ),
                io.Combo.Input(
                    "shape", options=list(modulation.LFO_SHAPES), default="sine",
                    tooltip="`sine` = a smooth pulse; `triangle` = even ramps; `square` = on and off, a "
                    "chop; `saw up` = swells; `saw down` = decays; random shapes wander. Jumps are "
                    "smoothed over 2 ms.",
                ),
                io.Float.Input(
                    "stereo_phase", default=0.0, min=0.0, max=0.5, step=0.01,
                    tooltip="Share of a cycle the right channel's pulse runs ahead of the left in "
                    "`tremolo`. `0` = together; `0.5` = alternating sides. `auto-pan` ignores it.",
                ),
                io.Float.Input(
                    "bpm", default=0.0, min=0.0, max=999.0, step=0.1, round=0.01,
                    tooltip="Tempo the pulse follows. `0` = use rate; `120` = one pulse per division at "
                    "120 BPM.",
                ),
                io.Combo.Input(
                    "division", options=list(modulation.DIVISIONS), default="1/8",
                    tooltip="Note length of one pulse when bpm is above 0. `1/4` = a beat; `1/8` = two "
                    "per beat; `d` = dotted; `t` = triplet.",
                ),
                io.Float.Input(
                    "mix", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="`0` = the input unchanged; `0.5` = half the depth; `1` = the full effect.",
                ),
                io.Float.Input(
                    "output_db", default=0.0, min=-24.0, max=24.0, step=0.1, round=0.01,
                    tooltip="dB applied to the result. `0` = unchanged. A full `auto-pan` raises the near "
                    "side 3 dB.",
                ),
            ],
            outputs=[io.Audio.Output(
                display_name="audio",
                tooltip="The pulsed or panned clip, as long as the input; stereo after `auto-pan`.",
            )],
        )

    @classmethod
    def execute(cls, audio, mode="tremolo", rate=4.0, depth=0.5, shape="sine", stereo_phase=0.0, bpm=0.0,
                division="1/8", mix=1.0, output_db=0.0) -> io.NodeOutput:
        """Pulse or pan every clip of the batch.

        Raises:
            ValueError: ``mode`` or ``shape`` is not one this node runs, ``division`` is not a note
                length, or a mono clip made stereo would be too large.
        """
        if mode not in effects.TREMOLO_MODES:
            raise ValueError(f"mode {mode!r} is not one of {', '.join(effects.TREMOLO_MODES)}.")
        if shape not in modulation.LFO_SHAPES:
            raise ValueError(f"shape {shape!r} is not one of {', '.join(modulation.LFO_SHAPES)}.")
        wave = clip.waveform_of(audio)
        sample_rate = clip.rate_of(audio)
        pulse_hz = effects.lfo_rate(rate, bpm, division)
        source = wave
        if mode == "auto-pan" and int(wave.shape[1]) == 1:
            clip.check_size(int(wave.shape[0]), 2, int(wave.shape[2]), "audio")
            source = clip.with_channels(wave, 2)

        def effect(signal):
            return effects.tremolo(signal, sample_rate, mode, pulse_hz, depth, shape, stereo_phase)

        out = effects.applied(source, mix, output_db, effect)
        facts = {"tremolo": {
            "mode": mode, "rate_hz": round(pulse_hz, 6), "synced": float(bpm) > 0.0, "bpm": float(bpm),
            "division": division, "shape": shape, "cycle": effects.cycle(shape),
            "steps": effects.random_steps(shape, pulse_hz, wave.shape[-1] / sample_rate), "depth": float(depth),
            "stereo_phase": float(stereo_phase) if mode == "tremolo" else 0.0, "mix": float(mix),
            "output_db": float(output_db),
            **effects.tremolo_figures(mode, depth, mix),
        }}
        return io.NodeOutput(clip.make(out, sample_rate), ui=before_after(wave, sample_rate, out, sample_rate, facts))

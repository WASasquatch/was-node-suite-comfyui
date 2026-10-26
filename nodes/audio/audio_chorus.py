"""Thicken a clip with drifting delayed copies: a chorus, a string ensemble or a vibrato."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.audio import clip
from ...modules.audio import modulation_effects as effects
from ...modules.interface.audio_sides import before_after

REQUIRES = "audio"


class AudioChorus(io.ComfyNode):
    """A multi-voice chorus, a three-phase ensemble and a vibrato."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioChorus",
            display_name="Audio Chorus",
            search_aliases=[
                "WASAudioChorus", "Audio Chorus", "chorus", "ensemble", "string ensemble", "vibrato",
                "doubling", "modulation",
            ],
            category="WAS Suite/Audio/Modulation",
            description=(
                "Thicken a voice, guitar or pad with slightly delayed copies of itself that drift "
                "in time: a multi-voice chorus, the lush three-phase string ensemble, or a pitch "
                "vibrato. The node plays the clip before and after."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to chorus."),
                io.Combo.Input(
                    "mode", options=list(effects.CHORUS_MODES), default="chorus",
                    tooltip="`chorus` = several drifting copies, wider and thicker; `ensemble` = three "
                    "copies on a slow and a fast wobble, the string machine sound; `vibrato` = one "
                    "moving copy, a pitch wobble, best at mix 1.",
                ),
                io.Float.Input(
                    "rate", default=0.8, min=0.01, max=10.0, step=0.01, round=0.001,
                    tooltip="Hertz the copies drift at. `0.3` = slow and lush; `0.8` = typical; `3` = "
                    "fast and seasick. `ensemble` adds a wobble 8 times faster.",
                ),
                io.Float.Input(
                    "depth", default=2.0, min=0.0, max=10.0, step=0.01, round=0.001,
                    tooltip="Milliseconds the delay swings either side of `delay`. `0.5` = subtle; `2` = "
                    "typical; `5` = a strong pitch wobble.",
                ),
                io.Float.Input(
                    "delay", default=15.0, min=1.0, max=40.0, step=0.1, round=0.01,
                    tooltip="Milliseconds the copies run behind the clip. `7` = tight; `15` = typical; "
                    "`30` = close to a double. `vibrato` ignores it and stays in time.",
                ),
                io.Int.Input(
                    "voices", default=3, min=2, max=6,
                    tooltip="Copies in `chorus` mode, spread in delay and timing. `2` = light; `3` = "
                    "typical; `6` = dense. `ensemble` uses 3 and `vibrato` 1.",
                ),
                io.Float.Input(
                    "spread", default=0.25, min=0.0, max=0.5, step=0.01,
                    tooltip="Share of a cycle the right channel's drift runs ahead of the left. `0` = "
                    "the same on both sides; `0.25` = wide; `0.5` = widest. A mono clip stays mono.",
                ),
                io.Float.Input(
                    "feedback", default=0.0, min=-0.5, max=0.5, step=0.01,
                    tooltip="Share of the copies fed back in `chorus` mode, for a metallic edge. `0` = "
                    "off; `0.3` = noticeable; negative = hollow. The other modes ignore it.",
                ),
                io.Float.Input(
                    "mix", default=0.5, min=0.0, max=1.0, step=0.01,
                    tooltip="`0` = the input unchanged; `0.5` = equal blend, the classic chorus; `1` = "
                    "the copies only, which `vibrato` wants.",
                ),
                io.Float.Input(
                    "output_db", default=0.0, min=-24.0, max=24.0, step=0.1, round=0.01,
                    tooltip="dB applied to the result. `0` = unchanged.",
                ),
            ],
            outputs=[io.Audio.Output(display_name="audio", tooltip="The chorused clip, as long as the input.")],
        )

    @classmethod
    def execute(cls, audio, mode="chorus", rate=0.8, depth=2.0, delay=15.0, voices=3, spread=0.25,
                feedback=0.0, mix=0.5, output_db=0.0) -> io.NodeOutput:
        """Chorus every clip of the batch.

        Raises:
            ValueError: ``mode`` is not one this node runs.
        """
        if mode not in effects.CHORUS_MODES:
            raise ValueError(f"mode {mode!r} is not one of {', '.join(effects.CHORUS_MODES)}.")
        wave = clip.waveform_of(audio)
        sample_rate = clip.rate_of(audio)

        def effect(signal):
            return effects.chorus(signal, sample_rate, mode, rate, depth, delay, voices, spread, feedback)

        out = effects.applied(wave, mix, output_db, effect)
        facts = {"chorus": {
            "mode": mode, "rate_hz": round(float(rate), 4), "depth_ms": float(depth), "delay_ms": float(delay),
            "spread": float(spread), "feedback": float(feedback) if mode == "chorus" else 0.0,
            "mix": float(mix), "output_db": float(output_db),
            **effects.chorus_figures(sample_rate, mode, rate, depth, delay, voices),
        }}
        return io.NodeOutput(clip.make(out, sample_rate), ui=before_after(wave, sample_rate, out, sample_rate, facts))

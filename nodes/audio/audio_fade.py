"""Fade a clip in at its start and out at its end."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.audio import clip, curves, edit
from ...modules.interface.audio_sides import before_after

REQUIRES = "audio"


class AudioFade(io.ComfyNode):
    """Fade the ends of a clip."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioFade",
            display_name="Audio Fade",
            search_aliases=["WASAudioFade", "Audio Fade", "fade in", "fade out", "audio fade in out"],
            category="WAS Suite/Audio",
            description=(
                "Fade a clip up from silence at its start and down to silence at its end, each "
                "on its own curve. The fades are drawn on the waveform and drag to length."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to fade."),
                io.Float.Input(
                    "fade_in", default=1.0, min=0.0, max=clip.MAX_SECONDS, step=0.01, round=0.0001,
                    tooltip="Seconds the start rises over. 0 = no fade in.",
                ),
                io.Float.Input(
                    "fade_out", default=1.0, min=0.0, max=clip.MAX_SECONDS, step=0.01, round=0.0001,
                    tooltip="Seconds the end falls over. 0 = no fade out.",
                ),
                io.Combo.Input(
                    "curve_in", options=list(curves.CURVES), default="s-curve",
                    tooltip="Shape of the fade in. `equal power` = steady level; `linear` = straight ramp; "
                    "`s-curve` = soft ends; `exponential` = slow start; `logarithmic` = fast start.",
                ),
                io.Combo.Input(
                    "curve_out", options=list(curves.CURVES), default="s-curve",
                    tooltip="Shape of the fade out, read from the end: `exponential` falls fast at "
                    "first, `logarithmic` holds the level longer, `s-curve` is soft at both ends.",
                ),
            ],
            outputs=[io.Audio.Output(display_name="audio", tooltip="The faded clip.")],
        )

    @classmethod
    def execute(cls, audio, fade_in=1.0, fade_out=1.0, curve_in="s-curve", curve_out="s-curve") -> io.NodeOutput:
        """Apply both fades."""
        wave = clip.waveform_of(audio)
        rate = clip.rate_of(audio)
        out = edit.faded(wave, clip.samples_at(fade_in, rate), clip.samples_at(fade_out, rate), curve_in, curve_out)
        facts = {"fade": {"fade_in": float(fade_in), "fade_out": float(fade_out),
                          "curve_in": curve_in, "curve_out": curve_out}}
        return io.NodeOutput(clip.make(out, rate), ui=before_after(wave, rate, out, rate, facts))

"""Hand one clip over to the next across an overlap: a crossfade, a dip, a cut, a sweep or a DJ move."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.audio import clip, curves, transition
from ...modules.interface.audio_sides import figures, panel_ui

REQUIRES = "audio"


class AudioTransition(io.ComfyNode):
    """Join two clips through a transition."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioTransition",
            display_name="Audio Transition",
            search_aliases=[
                "WASAudioTransition", "Audio Transition", "crossfade", "dj transition",
                "filter sweep", "echo out", "tape stop", "backspin",
            ],
            category="WAS Suite/Audio",
            description=(
                "Join two clips across an overlap: a crossfade, a dip to silence, a clean cut, "
                "a filter sweep, an echo out, a tape stop or a backspin. The node draws the "
                "seam and both fades on the joined waveform."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("a", tooltip="The clip that plays first and hands over."),
                io.Audio.Input("b", tooltip="The clip that follows. Brought to a's sample rate."),
                io.Float.Input(
                    "duration", default=2.0, min=0.0, max=600.0, step=0.01, round=0.0001,
                    tooltip="Seconds the two clips overlap. 2 = a DJ blend; 0.05 = a quick "
                    "splice; 0 = butt join. Held to the shorter clip. Drag the seam's edge.",
                ),
                io.Combo.Input(
                    "style", options=list(transition.STYLES), default="crossfade",
                    tooltip="`crossfade` = blend; `dip to silence` = out, then in; `cut` = switch "
                    "halfway; `filter sweep` = a darkens as b's bass arrives; `echo out` = a "
                    "repeats away; `tape stop` = a slows to a halt; `backspin` = a spins back.",
                ),
                io.Combo.Input(
                    "curve", options=list(curves.CURVES), default=curves.DEFAULT_CURVE,
                    tooltip="Shape of the fades. `equal power` = steady level; `linear` = straight "
                    "ramp; `s-curve` = soft ends; `exponential` = slow start; `logarithmic` = fast start.",
                ),
            ],
            outputs=[
                io.Audio.Output(display_name="audio", tooltip="a, the transition, then b."),
                io.Float.Output(display_name="seam_start", tooltip="Seconds into the result the overlap starts."),
                io.Float.Output(display_name="seam_end", tooltip="Seconds into the result the overlap ends."),
            ],
        )

    @classmethod
    def execute(cls, a, b, duration=2.0, style="crossfade", curve=curves.DEFAULT_CURVE) -> io.NodeOutput:
        """Join ``a`` and ``b``.

        Raises:
            ValueError: The batches cannot be paired.
        """
        (wave_a, wave_b), rate, _ = clip.matched([a, b], ["a", "b"])
        batch_a, batch_b = int(wave_a.shape[0]), int(wave_b.shape[0])
        if batch_a != batch_b:
            if min(batch_a, batch_b) != 1:
                raise ValueError(f"a holds {batch_a} clips and b {batch_b}; they pair one to one, or one to many.")
            batch = max(batch_a, batch_b)
            wave_a, wave_b = wave_a.expand(batch, -1, -1), wave_b.expand(batch, -1, -1)
        out, seam_start, seam_end = transition.transitioned(
            wave_a, wave_b, rate, clip.samples_at(duration, rate), style, curve,
        )
        facts = {
            "out": figures(out, rate),
            "transition": {
                "style": style, "curve": curve,
                "seam": [seam_start / rate, seam_end / rate],
                "a_duration": clip.duration(wave_a, rate), "b_duration": clip.duration(wave_b, rate),
            },
        }
        return io.NodeOutput(
            clip.make(out, rate), seam_start / rate, seam_end / rate,
            ui=panel_ui([("out", out, rate)], facts),
        )

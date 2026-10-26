"""Trim a clip, cut regions out of it and fade its ends, edited on its waveform."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.audio import clip, curves, edit
from ...modules.interface.audio_sides import before_after

REQUIRES = "audio"


class AudioEditor(io.ComfyNode):
    """Trim, cut and fade one clip."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioEditor",
            display_name="Audio Editor",
            search_aliases=[
                "WASAudioEditor", "Audio Editor", "trim audio", "cut audio", "snip audio",
                "audio fade", "waveform editor",
            ],
            category="WAS Suite/Audio",
            description=(
                "Trim a clip, cut regions out of it and fade its ends, by dragging on its "
                "waveform. Joins left by a cut are crossfaded so they do not click, and the "
                "edited clip plays on the node beside the source."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to edit."),
                io.Float.Input(
                    "start", default=0.0, min=0.0, max=clip.MAX_SECONDS, step=0.001, round=0.0001,
                    tooltip="Seconds into the clip the kept part starts. 0 = the beginning. Set "
                    "by dragging the left edge on the waveform.",
                ),
                io.Float.Input(
                    "end", default=0.0, min=0.0, max=clip.MAX_SECONDS, step=0.001, round=0.0001,
                    tooltip="Seconds into the clip the kept part ends. 0 = the end. Set by "
                    "dragging the right edge on the waveform.",
                ),
                io.String.Input(
                    "cuts", default="",
                    tooltip="Regions taken out, in seconds of the source: `1.2-2.5, 4:00-4:10.5`. "
                    "Empty = no cuts. Select on the waveform and press Cut, or Delete.",
                ),
                io.Float.Input(
                    "fade_in", default=0.0, min=0.0, max=clip.MAX_SECONDS, step=0.01, round=0.0001,
                    tooltip="Seconds the start fades up over. 0 = no fade. Drag the handle at the "
                    "kept part's top left corner.",
                ),
                io.Float.Input(
                    "fade_out", default=0.0, min=0.0, max=clip.MAX_SECONDS, step=0.01, round=0.0001,
                    tooltip="Seconds the end fades down over. 0 = no fade. Drag the handle at the "
                    "kept part's top right corner.",
                ),
                io.Combo.Input(
                    "curve", options=list(curves.CURVES), default=curves.DEFAULT_CURVE,
                    tooltip="Shape of the fades and cut joins. `equal power` = steady level through a "
                    "join; `linear` = straight ramp; `s-curve` = soft ends; `exponential` = slow "
                    "start; `logarithmic` = fast start.",
                ),
                io.Float.Input(
                    "crossfade", default=0.01, min=0.0, max=10.0, step=0.001, round=0.0001,
                    tooltip="Seconds each join left by a cut overlaps by. 0.01 = a click-free "
                    "splice; 0.5 = a slow blend; 0 = a hard join that may click.",
                ),
            ],
            outputs=[
                io.Audio.Output(display_name="audio", tooltip="The edited clip."),
                io.Float.Output(display_name="duration", tooltip="Seconds the edited clip plays for."),
            ],
        )

    @classmethod
    def execute(cls, audio, start=0.0, end=0.0, cuts="", fade_in=0.0, fade_out=0.0,
                curve=curves.DEFAULT_CURVE, crossfade=0.01) -> io.NodeOutput:
        """Apply the trim, the cuts and the fades.

        Raises:
            ValueError: The trim keeps nothing, or ``cuts`` is not a list of regions.
        """
        wave = clip.waveform_of(audio)
        rate = clip.rate_of(audio)
        length = int(wave.shape[-1])
        total = length / rate
        if float(start) >= total:
            raise ValueError(f"start is {start:.3f} s, at or past the end of the {total:.3f} s clip.")
        if float(end) > 0 and float(end) <= float(start):
            raise ValueError(f"end ({end:.3f} s) is at or before start ({start:.3f} s); 0 keeps to the end.")
        first, last = edit.span_of(rate, length, start, end)
        regions = edit.parse_regions(cuts, "cuts")
        spans = []
        for region_start, region_end in regions:
            lo, hi = edit.span_of(rate, length, region_start, region_end)
            lo, hi = max(lo, first), min(hi, last)
            if hi > lo:
                spans.append((lo - first, hi - first))
        kept = wave[..., first:last]
        edited = edit.without(kept, spans, clip.samples_at(crossfade, rate), curve) if spans else kept.clone()
        edited = edit.faded(edited, clip.samples_at(fade_in, rate), clip.samples_at(fade_out, rate), curve, curve)
        facts = {
            "edit": {
                "start": first / rate, "end": last / rate,
                "cuts": [[a / rate + first / rate, b / rate + first / rate] for a, b in spans],
                "fade_in": float(fade_in), "fade_out": float(fade_out), "curve": curve,
            },
        }
        return io.NodeOutput(
            clip.make(edited, rate), clip.duration(edited, rate),
            ui=before_after(wave, rate, edited, rate, facts),
        )

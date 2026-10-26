"""Play a clip several times over, or until it fills a length, with crossfaded seams."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.audio import clip, curves, edit
from ...modules.interface.audio_sides import before_after

REQUIRES = "audio"


class AudioLoop(io.ComfyNode):
    """Repeat a clip."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioLoop",
            display_name="Audio Loop",
            search_aliases=["WASAudioLoop", "Audio Loop", "repeat audio", "loop audio", "extend audio"],
            category="WAS Suite/Audio",
            description=(
                "Play a clip several times over, or as often as it takes to fill a length, "
                "with each seam crossfaded, such as stretching a music bed under a longer video."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to repeat."),
                io.Int.Input(
                    "times", default=2, min=1, max=10000,
                    tooltip="Plays in all: 2 = the clip twice. Ignored while duration is above 0.",
                ),
                io.Float.Input(
                    "duration", default=0.0, min=0.0, max=clip.MAX_SECONDS, step=0.01, round=0.0001,
                    tooltip="Seconds to fill, repeating as often as needed and cutting the last play "
                    "short. 0 = use times.",
                ),
                io.Float.Input(
                    "crossfade", default=0.05, min=0.0, max=60.0, step=0.01, round=0.0001,
                    tooltip="Seconds each seam overlaps by. 0.05 = click-free; 2 = a smooth blend "
                    "for ambience; 0 = hard seam.",
                ),
                io.Combo.Input(
                    "curve", options=list(curves.CURVES), default=curves.DEFAULT_CURVE,
                    tooltip="Shape of the seams. `equal power` = steady level; `linear` = straight ramp; "
                    "`s-curve` = soft ends; `exponential` = slow start; `logarithmic` = fast start.",
                ),
            ],
            outputs=[
                io.Audio.Output(display_name="audio", tooltip="The repeated clip."),
                io.Float.Output(display_name="duration", tooltip="Seconds the result plays for."),
            ],
        )

    @classmethod
    def execute(cls, audio, times=2, duration=0.0, crossfade=0.05, curve=curves.DEFAULT_CURVE) -> io.NodeOutput:
        """Repeat the clip."""
        wave = clip.waveform_of(audio)
        rate = clip.rate_of(audio)
        out = edit.looped(wave, int(times), clip.samples_at(duration, rate), clip.samples_at(crossfade, rate), curve)
        length = int(wave.shape[-1])
        overlap = min(clip.samples_at(crossfade, rate), length // 2)
        pitch = max(1, length - overlap)
        seams = [index * pitch / rate for index in range(1, int(out.shape[-1]) // pitch + 1)
                 if index * pitch < int(out.shape[-1]) - overlap]
        facts = {"loop": {"seams": seams[:2000]}}
        return io.NodeOutput(
            clip.make(out, rate), clip.duration(out, rate),
            ui=before_after(wave, rate, out, rate, facts),
        )

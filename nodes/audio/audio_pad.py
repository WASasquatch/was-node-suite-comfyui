"""Add silence before and after a clip, or place it at a time inside a set length."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.audio import clip, edit
from ...modules.interface.audio_sides import before_after

REQUIRES = "audio"


class AudioPad(io.ComfyNode):
    """Pad a clip with silence."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioPad",
            display_name="Audio Pad",
            search_aliases=[
                "WASAudioPad", "Audio Pad", "add silence", "pad audio", "offset audio",
                "delay start", "audio to length",
            ],
            category="WAS Suite/Audio",
            description=(
                "Add silence before and after a clip, or set it to an exact length with its "
                "start placed at a time, such as lining a sound effect up with a frame of video."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to pad."),
                io.Float.Input(
                    "before", default=0.0, min=0.0, max=clip.MAX_SECONDS, step=0.01, round=0.0001,
                    tooltip="Seconds of silence placed before the clip, such as `0.5`, which is where the clip now starts.",
                ),
                io.Float.Input(
                    "after", default=0.0, min=0.0, max=clip.MAX_SECONDS, step=0.01, round=0.0001,
                    tooltip="Seconds of silence placed after the clip. Ignored while length is above 0.",
                ),
                io.Float.Input(
                    "length", default=0.0, min=0.0, max=clip.MAX_SECONDS, step=0.01, round=0.0001,
                    tooltip="Seconds the result lasts exactly: silence fills what is left, and a clip "
                    "that runs past the end is cut. 0 = before + clip + after.",
                ),
            ],
            outputs=[
                io.Audio.Output(display_name="audio", tooltip="The padded clip."),
                io.Float.Output(display_name="duration", tooltip="Seconds the result plays for."),
            ],
        )

    @classmethod
    def execute(cls, audio, before=0.0, after=0.0, length=0.0) -> io.NodeOutput:
        """Pad the clip."""
        wave = clip.waveform_of(audio)
        rate = clip.rate_of(audio)
        lead = clip.samples_at(before, rate)
        if float(length) > 0:
            total = max(1, clip.samples_at(length, rate))
            out = edit.padded(wave, lead, max(0, total - lead - int(wave.shape[-1])))[..., :total].contiguous()
        else:
            out = edit.padded(wave, lead, clip.samples_at(after, rate))
        facts = {"pad": {"start": lead / rate}}
        return io.NodeOutput(
            clip.make(out, rate), clip.duration(out, rate),
            ui=before_after(wave, rate, out, rate, facts),
        )

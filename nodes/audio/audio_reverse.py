"""Play a clip backwards."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.audio import clip
from ...modules.interface.audio_sides import before_after

REQUIRES = "audio"


class AudioReverse(io.ComfyNode):
    """Reverse a clip."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioReverse",
            display_name="Audio Reverse",
            search_aliases=["WASAudioReverse", "Audio Reverse", "reverse audio", "backwards audio"],
            category="WAS Suite/Audio",
            description=(
                "Play a clip backwards, for reverse swells into a cymbal hit or a reverb tail "
                "that rises into a sound."
            ),
            is_output_node=True,
            inputs=[io.Audio.Input("audio", tooltip="The clip to reverse.")],
            outputs=[io.Audio.Output(display_name="audio", tooltip="The clip, last sample first.")],
        )

    @classmethod
    def execute(cls, audio) -> io.NodeOutput:
        """Reverse the clip."""
        wave = clip.waveform_of(audio)
        rate = clip.rate_of(audio)
        out = wave.flip(-1).contiguous()
        return io.NodeOutput(clip.make(out, rate), ui=before_after(wave, rate, out, rate))

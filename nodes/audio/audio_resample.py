"""Convert a clip to another sample rate."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.audio import clip
from ...modules.audio.resample import resampled
from ...modules.interface.audio_sides import before_after

REQUIRES = "audio"

#: Rates offered, in menu order; ``custom`` reads the rate widget.
RATES = ("48000", "44100", "96000", "32000", "24000", "22050", "16000", "8000", "custom")


class AudioResample(io.ComfyNode):
    """Change a clip's sample rate."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioResample",
            display_name="Audio Resample",
            search_aliases=[
                "WASAudioResample", "Audio Resample", "sample rate", "convert sample rate",
                "resample audio", "48khz", "44.1khz",
            ],
            category="WAS Suite/Audio",
            description=(
                "Convert a clip to another sample rate without changing its pitch or length, "
                "band-limited so nothing aliases, such as bringing a 24 kHz voice to a 48 kHz "
                "video track."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to convert."),
                io.Combo.Input(
                    "sample_rate", options=list(RATES), default="48000",
                    tooltip="Samples per second wanted: 48000 = video; 44100 = CD and music; "
                    "16000 = speech models. custom = the rate widget.",
                ),
                io.Int.Input(
                    "rate", default=48000, min=clip.MIN_RATE, max=clip.MAX_RATE,
                    tooltip="Samples per second while sample_rate is custom, 1000 to 384000.",
                ),
            ],
            outputs=[io.Audio.Output(display_name="audio", tooltip="The clip at the new rate.")],
        )

    @classmethod
    def execute(cls, audio, sample_rate="48000", rate=48000) -> io.NodeOutput:
        """Resample the clip."""
        wave = clip.waveform_of(audio)
        source = clip.rate_of(audio)
        target = int(rate) if sample_rate == "custom" else int(sample_rate)
        clip.check_size(int(wave.shape[0]), int(wave.shape[1]),
                        int(round(int(wave.shape[-1]) * target / source)), "sample_rate")
        out = resampled(wave, source, target)
        return io.NodeOutput(clip.make(out, target), ui=before_after(wave, source, out, target))

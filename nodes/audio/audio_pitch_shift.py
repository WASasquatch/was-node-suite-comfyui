"""Move a clip up or down in pitch without changing its length, keeping a voice's formants."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.audio import clip, pitch
from ...modules.interface.audio_sides import before_after

REQUIRES = "audio"


class AudioPitchShift(io.ComfyNode):
    """A pitch shifter with formant correction."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioPitchShift",
            display_name="Audio Pitch Shift",
            search_aliases=[
                "WASAudioPitchShift", "Audio Pitch Shift", "pitch shift", "transpose", "key change",
                "formant", "chipmunk", "deep voice",
            ],
            category="WAS Suite/Audio",
            description=(
                "Move a clip up or down in pitch by semitones while it keeps its length, such as "
                "matching a song's key or deepening a voice. Keeping formants stops a shifted "
                "voice sounding like a chipmunk or a giant."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to shift."),
                io.Float.Input(
                    "semitones", default=0.0, min=-24.0, max=24.0, step=0.01,
                    tooltip="Shift in semitones. `12` = an octave up; `-5` = a fourth down; `0.5` = a "
                    "quarter tone; `0` = unchanged.",
                ),
                io.Boolean.Input(
                    "keep_formants", default=True,
                    tooltip="`true` = the voice keeps its size and character; `false` = formants move "
                    "with the pitch, the chipmunk and giant effects.",
                ),
                io.Float.Input(
                    "mix", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="`1` = shifted only; `0.5` = harmony with the original; `0` = the input "
                    "unchanged.",
                ),
                io.Float.Input(
                    "output_db", default=0.0, min=-24.0, max=24.0, step=0.1, round=0.01,
                    tooltip="dB applied to the result. `0` = unchanged.",
                ),
            ],
            outputs=[io.Audio.Output(display_name="audio", tooltip="The shifted clip, as long as the input.")],
        )

    @classmethod
    def execute(cls, audio, semitones=0.0, keep_formants=True, mix=1.0, output_db=0.0) -> io.NodeOutput:
        """Shift every clip of the batch.

        Raises:
            ValueError: The clip is too long to shift in one pass.
        """
        wave = clip.waveform_of(audio)
        rate = clip.rate_of(audio)
        wet = pitch.shifted(wave, rate, semitones, bool(keep_formants))
        result = (wave * (1.0 - mix) + wet * mix) * 10.0 ** (float(output_db) / 20.0)
        facts = {"pitch": {"semitones": float(semitones), "keep_formants": bool(keep_formants)}}
        return io.NodeOutput(clip.make(result, rate), ui=before_after(wave, rate, result, rate, facts))

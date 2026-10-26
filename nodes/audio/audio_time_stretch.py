"""Make a clip play faster or slower, keeping its pitch or letting the pitch follow like tape."""

from __future__ import annotations

import numpy as np
from comfy_api.latest import io

from ...modules.audio import clip
from ...modules.audio.resample import read_at
from ...modules.interface.audio_sides import before_after

REQUIRES = "audio"

#: Fastest and slowest speeds offered.
MIN_SPEED = 0.25
MAX_SPEED = 4.0

#: How the speed changes the sound, in menu order.
METHODS = ("keep pitch", "tape")

#: Most samples, across channels, keep pitch stretches in one pass.
MAX_VOCODED = 1 << 26


class AudioTimeStretch(io.ComfyNode):
    """Change a clip's speed."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioTimeStretch",
            display_name="Audio Time Stretch",
            search_aliases=[
                "WASAudioTimeStretch", "Audio Time Stretch", "time stretch", "change tempo",
                "speed up audio", "slow down audio", "varispeed", "fit audio to duration",
            ],
            category="WAS Suite/Audio",
            description=(
                "Make a clip faster or slower by a factor or to an exact length, keeping its "
                "pitch or letting the pitch follow like tape, such as fitting music to a video. "
                "The node plays both lengths."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to stretch."),
                io.Float.Input(
                    "speed", default=1.0, min=MIN_SPEED, max=MAX_SPEED, step=0.01, round=0.0001,
                    tooltip="1 = unchanged; 2 = twice as fast, half as long; 0.5 = half speed. "
                    "Ignored while duration is above 0.",
                ),
                io.Float.Input(
                    "duration", default=0.0, min=0.0, max=clip.MAX_SECONDS, step=0.01, round=0.0001,
                    tooltip="Seconds the result lasts exactly, such as a video's length. 0 = use speed.",
                ),
                io.Combo.Input(
                    "method", options=list(METHODS), default="keep pitch",
                    tooltip="`keep pitch` = same key, new tempo; `tape` = pitch rises and falls with "
                    "the speed, like a record played at the wrong speed.",
                ),
            ],
            outputs=[
                io.Audio.Output(display_name="audio", tooltip="The stretched clip."),
                io.Float.Output(display_name="speed", tooltip="The speed applied: input length over output length."),
            ],
        )

    @classmethod
    def execute(cls, audio, speed=1.0, duration=0.0, method="keep pitch") -> io.NodeOutput:
        """Stretch the clip.

        Raises:
            ValueError: ``duration`` asks for a speed outside 0.25 to 4, or keep pitch is asked
                of more than :data:`MAX_VOCODED` samples.
        """
        wave = clip.waveform_of(audio)
        rate = clip.rate_of(audio)
        length = int(wave.shape[-1])
        if float(duration) > 0:
            factor = (length / rate) / float(duration)
            if not MIN_SPEED <= factor <= MAX_SPEED:
                raise ValueError(
                    f"stretching {length / rate:.3f} s to {duration:.3f} s needs a speed of {factor:.3f}, "
                    f"outside {MIN_SPEED} to {MAX_SPEED}. Stretch in two passes, or loop or trim first."
                )
        else:
            factor = float(speed)
        out_length = max(1, int(round(length / factor)))
        clip.check_size(int(wave.shape[0]), int(wave.shape[1]), out_length, "speed or duration")
        if abs(factor - 1.0) < 1e-9:
            out = wave.clone()
        elif method == "tape":
            out = read_at(wave, np.arange(out_length, dtype=np.float64) * factor, factor)
        else:
            from ...modules.media.audio_timeline import retimed

            held = int(wave.shape[0]) * int(wave.shape[1]) * max(length, out_length)
            if held > MAX_VOCODED:
                raise ValueError(
                    f"keep pitch reads {held:,} samples across this clip's channels, past the "
                    f"{MAX_VOCODED:,} it stretches in one pass, about 11 minutes of 48 kHz stereo. "
                    f"Use method tape, or split the clip with Audio Split first."
                )
            out = retimed(clip.make(wave, rate), out_length / rate, lambda at: at * factor, 1.0)["waveform"]
            out = out[..., :out_length].to(wave.dtype).contiguous()
        facts = {"stretch": {"speed": factor, "method": method}}
        return io.NodeOutput(clip.make(out, rate), factor, ui=before_after(wave, rate, out, rate, facts))

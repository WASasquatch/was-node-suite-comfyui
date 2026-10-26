"""Repeat a clip as echoes: a clean digital delay, a tape echo, a bucket brigade or a ping-pong."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ...modules.audio import clip, delay_effects, modulation
from ...modules.interface.audio_sides import before_after

REQUIRES = "audio"


class AudioDelay(io.ComfyNode):
    """A feedback delay in four voicings, set in milliseconds or synced to a tempo."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioDelay",
            display_name="Audio Delay",
            search_aliases=[
                "WASAudioDelay", "Audio Delay", "delay", "echo", "tape echo", "slapback", "ping pong",
                "bucket brigade", "analog delay",
            ],
            category="WAS Suite/Audio/Space",
            description=(
                "Repeat a clip as echoes that fade away: clean and exact, a wobbling tape echo, a dark "
                "bucket brigade, or bouncing between left and right, timed in milliseconds or to a tempo. "
                "The node plays the clip before and after and marks where the repeats land."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to echo."),
                io.Combo.Input(
                    "mode", options=list(delay_effects.MODES), default="digital",
                    tooltip="`digital` = clean, exact repeats; `tape` = wow, flutter and saturation, each "
                    "repeat duller; `BBD` = dark, gritty analog repeats; `ping-pong` = repeats alternate "
                    "left and right, a mono clip becomes stereo.",
                ),
                io.Float.Input(
                    "time_ms", default=350.0, min=1.0, max=4000.0, step=1.0, round=0.01,
                    tooltip="Milliseconds between repeats when bpm is 0. `80` = slapback; `350` = classic "
                    "echo; `1000` = long and spacious.",
                ),
                io.Float.Input(
                    "bpm", default=0.0, min=0.0, max=400.0, step=0.1, round=0.01,
                    tooltip="Tempo the repeats lock to, one per division. `0` = use time_ms; `120` with "
                    "`1/8` = 250 ms.",
                ),
                io.Combo.Input(
                    "division", options=list(modulation.DIVISIONS), default="1/8",
                    tooltip="Note length between repeats when bpm is above 0. `1/4` = a beat; `1/8d` = "
                    "dotted eighth; `1/8t` = triplet eighth.",
                ),
                io.Float.Input(
                    "feedback", default=0.4, min=0.0, max=0.98, step=0.01,
                    tooltip="Share of each repeat fed into the next. `0` = one repeat; `0.4` = a few; "
                    "`0.9` = long trails; `0.98` = near endless.",
                ),
                io.Float.Input(
                    "low_cut", default=0.0, min=0.0, max=2000.0, step=1.0,
                    tooltip="Hertz cut below on every repeat, so each is thinner. `0` = off; `150` = keeps "
                    "the bass clean; `500` = telephone thin.",
                ),
                io.Float.Input(
                    "high_cut", default=0.0, min=0.0, max=24000.0, step=10.0,
                    tooltip="Hertz cut above on every repeat, so each is darker. `0` = off; `6000` = softer "
                    "trails; `2500` = muffled.",
                ),
                io.Float.Input(
                    "drive", default=0.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Saturation on every repeat. `0` = clean (`tape` keeps a little); `0.3` = warm; "
                    "`1` = crushed and compressed.",
                ),
                io.Float.Input(
                    "duck", default=0.0, min=0.0, max=1.0, step=0.01,
                    tooltip="How far the repeats dip while the dry clip is loud. `0` = off; `0.5` = half; `1` "
                    "= the repeats only come up in the gaps.",
                    advanced=True,
                ),
                io.Float.Input(
                    "width", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Stereo spread of the repeats. `0` = centred; `1` = as the clip, or hard left "
                    "and right in `ping-pong`.",
                    advanced=True,
                ),
                io.Float.Input(
                    "mix", default=0.35, min=0.0, max=1.0, step=0.01,
                    tooltip="`0` = the input unchanged; `0.35` = echoes behind the clip; `1` = the repeats "
                    "only.",
                ),
                io.Float.Input(
                    "output_db", default=0.0, min=-24.0, max=24.0, step=0.1, round=0.01,
                    tooltip="dB applied to the result. `0` = unchanged.",
                ),
                io.Float.Input(
                    "tail", default=-1.0, min=-1.0, max=60.0, step=0.1, round=0.01,
                    tooltip="Seconds added past the input for the repeats to fade. `-1` = until they fall "
                    "60 dB, at most 30; `0` = none, cut at the input's end.",
                    advanced=True,
                ),
            ],
            outputs=[io.Audio.Output(display_name="audio", tooltip="The echoed clip, longer by the tail.")],
        )

    @classmethod
    def execute(cls, audio, mode="digital", time_ms=350.0, bpm=0.0, division="1/8", feedback=0.4, low_cut=0.0,
                high_cut=0.0, drive=0.0, duck=0.0, width=1.0, mix=0.35, output_db=0.0, tail=-1.0) -> io.NodeOutput:
        """Echo every clip of the batch.

        Raises:
            ValueError: ``mode`` is not one this node runs, the delay time is out of range, or the tail
                would make the clip too long.
        """
        if mode not in delay_effects.MODES:
            raise ValueError(f"mode {mode!r} is not one of {', '.join(delay_effects.MODES)}.")
        wave = clip.waveform_of(audio)
        rate = clip.rate_of(audio)
        batch, channels, length = (int(dim) for dim in wave.shape)
        seconds = delay_effects.delay_seconds(time_ms, bpm, division)
        extra = delay_effects.auto_tail(seconds, feedback) if float(tail) < 0.0 else float(tail)
        total = length + clip.samples_at(extra, rate)
        out_channels = max(2, channels) if mode == "ping-pong" else channels
        clip.check_size(batch, out_channels, total, "tail")
        results = []
        for index in range(batch):
            dry = wave[index].numpy()
            wet = None
            if float(mix) > 0.0:
                wet = delay_effects.delayed(dry, rate, mode, seconds, feedback, low_cut, high_cut, drive, width, total)
                delay_effects.ducked(wet, dry, rate, duck)
            results.append(torch.from_numpy(delay_effects.blended(dry, wet, mix, output_db, total, out_channels)))
        out = torch.stack(results, dim=0)
        times, levels = delay_effects.repeat_figures(seconds, feedback)
        facts = {"delay": {
            "mode": mode, "time": round(seconds, 6), "feedback": float(feedback), "repeats": times,
            "levels": levels,
        }}
        if mode == "ping-pong":
            facts["delay"]["sides"] = ["left" if number % 2 == 0 else "right" for number in range(len(times))]
        return io.NodeOutput(clip.make(out, rate), ui=before_after(wave, rate, out, rate, facts))

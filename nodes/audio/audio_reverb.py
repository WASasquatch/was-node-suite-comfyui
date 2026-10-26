"""Place a clip in a room, hall, plate, chamber or spring tank with an algorithmic reverb."""

from __future__ import annotations

import numpy as np
import torch
from comfy_api.latest import io

from ...modules.audio import clip, delay_effects, reverb
from ...modules.interface.audio_sides import before_after

REQUIRES = "audio"


class AudioReverb(io.ComfyNode):
    """A feedback delay network reverb with five rooms and a measured decay."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioReverb",
            display_name="Audio Reverb",
            search_aliases=[
                "WASAudioReverb", "Audio Reverb", "reverb", "room", "hall", "plate reverb", "chamber",
                "spring reverb", "ambience", "space",
            ],
            category="WAS Suite/Audio/Space",
            description=(
                "Put a dry voice, instrument or effect in a room, hall, plate, chamber or spring tank, "
                "with the decay time set in seconds and the tail left to ring out. The node plays the "
                "clip before and after and draws the reverb's measured decay."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to reverberate."),
                io.Combo.Input(
                    "room", options=list(reverb.ROOMS), default="hall",
                    tooltip="`room` = small, close walls; `hall` = large and smooth; `plate` = dense, bright, "
                    "no early echoes; `chamber` = warm and thick; `spring` = the boing of a guitar amp tank.",
                ),
                io.Float.Input(
                    "decay", default=2.0, min=0.1, max=20.0, step=0.05, round=0.001,
                    tooltip="Seconds the tail takes to fall 60 dB around 1 kHz. `0.5` = small room; `2` = "
                    "hall; `6` = cathedral.",
                ),
                io.Float.Input(
                    "size", default=1.0, min=0.25, max=2.0, step=0.01, round=0.001,
                    tooltip="Scale on the room's dimensions. `1` = as designed; `0.5` = half, denser and "
                    "tighter; `2` = double, sparser echoes.",
                ),
                io.Float.Input(
                    "pre_delay", default=20.0, min=0.0, max=500.0, step=1.0, round=0.1,
                    tooltip="Milliseconds before the reverb answers. `0` = at once; `20` = natural; `80` = "
                    "keeps a voice clear of its tail.",
                ),
                io.Float.Input(
                    "damping", default=0.5, min=0.0, max=1.0, step=0.01,
                    tooltip="How much faster the highs die than the decay. `0` = as long, bright; `0.5` = "
                    "about half as long; `1` = a tenth as long, dark.",
                ),
                io.Float.Input(
                    "modulation", default=0.3, min=0.0, max=1.0, step=0.01,
                    tooltip="Slow movement in the tail that smooths metallic ringing. `0` = static; `0.3` = "
                    "subtle; `1` = lush chorus. `spring` ignores it.",
                    advanced=True,
                ),
                io.Float.Input(
                    "low_cut", default=80.0, min=0.0, max=2000.0, step=1.0,
                    tooltip="Hertz below which the reverb is cut. `0` = off; `80` = clears rumble; `300` = "
                    "thin, keeps a mix clean.",
                ),
                io.Float.Input(
                    "high_cut", default=10000.0, min=0.0, max=24000.0, step=10.0,
                    tooltip="Hertz above which the reverb is cut. `0` = off; `10000` = natural; `4000` = dark "
                    "and vintage.",
                ),
                io.Float.Input(
                    "width", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="Stereo spread of the reverb. `0` = the same on every channel; `1` = fully "
                    "decorrelated. A mono clip stays mono.",
                    advanced=True,
                ),
                io.Float.Input(
                    "mix", default=0.3, min=0.0, max=1.0, step=0.01,
                    tooltip="`0` = the input unchanged; `0.3` = a reverb on a dry track; `1` = the reverb "
                    "only, for a send.",
                ),
                io.Float.Input(
                    "output_db", default=0.0, min=-24.0, max=24.0, step=0.1, round=0.01,
                    tooltip="dB applied to the result. `0` = unchanged.",
                ),
                io.Float.Input(
                    "tail", default=-1.0, min=-1.0, max=60.0, step=0.1, round=0.01,
                    tooltip="Seconds added past the input for the reverb to ring out. `-1` = decay plus "
                    "pre_delay, at most 30; `0` = none, cut at the input's end.",
                    advanced=True,
                ),
            ],
            outputs=[io.Audio.Output(display_name="audio", tooltip="The reverberated clip, longer by the tail.")],
        )

    @classmethod
    def execute(cls, audio, room="hall", decay=2.0, size=1.0, pre_delay=20.0, damping=0.5, modulation=0.3,
                low_cut=80.0, high_cut=10000.0, width=1.0, mix=0.3, output_db=0.0, tail=-1.0) -> io.NodeOutput:
        """Reverberate every clip of the batch.

        Raises:
            ValueError: ``room`` is not one this node builds, or the tail would make the clip too long.
        """
        wave = clip.waveform_of(audio)
        rate = clip.rate_of(audio)
        batch, channels, length = (int(dim) for dim in wave.shape)
        offset = clip.samples_at(float(pre_delay) / 1000.0, rate)
        seconds = reverb.auto_tail(decay, float(pre_delay) / 1000.0) if float(tail) < 0.0 else float(tail)
        total = length + clip.samples_at(seconds, rate)
        clip.check_size(batch, channels, total, "tail")
        space = reverb.Reverb(room, rate, channels, decay, size, damping, modulation)
        sections = delay_effects.cut_sections(rate, low_cut, high_cut)
        response = delay_effects.filtered(space.impulse_response.copy(), sections)
        figures = reverb.decay_figures(delay_effects.widened(response, width), rate)
        results = []
        for index in range(batch):
            dry = wave[index].numpy()
            wet = None
            if float(mix) > 0.0:
                source = np.zeros((channels, total), dtype=np.float32)
                span = max(0, min(length, total - offset))
                source[:, offset:offset + span] = dry[:, :span]
                wet = delay_effects.widened(delay_effects.filtered(space.render(source), sections), width)
            results.append(torch.from_numpy(delay_effects.blended(dry, wet, mix, output_db, total, channels)))
        out = torch.stack(results, dim=0)
        facts = {"reverb": {
            "room": room, "target": float(decay), "rt60": figures["rt60"], "band": figures["band"],
            "decay": figures["decay"],
        }}
        return io.NodeOutput(clip.make(out, rate), ui=before_after(wave, rate, out, rate, facts))

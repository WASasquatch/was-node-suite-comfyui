"""Make a clip sound played back from worn tape, vinyl, AM radio, a telephone or VHS."""

from __future__ import annotations

import numpy as np
import torch
from comfy_api.latest import io

from ...modules.audio import clip, lofi
from ...modules.interface.audio_sides import before_after

REQUIRES = "audio"


class AudioLoFi(io.ComfyNode):
    """A worn media emulation."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioLoFi",
            display_name="Audio Lo-Fi",
            search_aliases=[
                "WASAudioLoFi", "Audio Lo-Fi", "lofi", "lo-fi", "vinyl", "cassette", "tape wow flutter",
                "old radio", "telephone voice", "vhs",
            ],
            category="WAS Suite/Audio/Colour",
            description=(
                "Play a clip back from a worn cassette, vinyl record, AM radio, telephone, VHS or "
                "tired tape: pitch wobble, hiss, crackle, a narrowed band and saturation, each "
                "scaled by its own amount and by age."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to age."),
                io.Combo.Input(
                    "medium", options=list(lofi.MEDIA), default="cassette",
                    tooltip="`cassette` = hiss and wow; `vinyl` = crackle and rumble; `AM radio` and "
                    "`telephone` = narrow mono; `VHS` = wobbly and dull; `worn tape` = everything, "
                    "heavily.",
                ),
                io.Float.Input(
                    "age", default=0.5, min=0.0, max=1.0, step=0.01,
                    tooltip="Scales every amount below. `0` = nearly new; `0.5` = used; `1` = falling "
                    "apart.",
                ),
                io.Float.Input("wow", default=0.5, min=0.0, max=1.0, step=0.01,
                               tooltip="Slow pitch drift. `0` = steady; `1` = seasick."),
                io.Float.Input("flutter", default=0.5, min=0.0, max=1.0, step=0.01,
                               tooltip="Fast pitch flutter. `0` = steady; `1` = warbling."),
                io.Float.Input("noise", default=0.5, min=0.0, max=1.0, step=0.01,
                               tooltip="Hiss and rumble. `0` = silent background; `1` = loud."),
                io.Float.Input("crackle", default=0.5, min=0.0, max=1.0, step=0.01,
                               tooltip="Clicks and pops, mostly on `vinyl`. `0` = none; `1` = dusty."),
                io.Float.Input("bandwidth", default=0.3, min=0.0, max=1.0, step=0.01,
                               tooltip="Narrows the band past the medium's own. `0` = as the medium; `1` = "
                               "muffled and thin."),
                io.Float.Input("saturation", default=0.5, min=0.0, max=1.0, step=0.01,
                               tooltip="How hard the medium is driven. `0` = clean; `1` = squashed."),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xFFFFFFFF, control_after_generate=io.ControlAfterGenerate.fixed,
                    tooltip="Seed of the wobble, noise and crackle, such as `0`. The same seed gives the same wear.",
                ),
                io.Float.Input(
                    "mix", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="`1` = fully worn; `0` = the input unchanged.",
                ),
                io.Float.Input(
                    "output_db", default=0.0, min=-24.0, max=24.0, step=0.1, round=0.01,
                    tooltip="dB applied to the result. `0` = unchanged.",
                ),
            ],
            outputs=[io.Audio.Output(display_name="audio", tooltip="The worn clip.")],
        )

    @classmethod
    def execute(cls, audio, medium="cassette", age=0.5, wow=0.5, flutter=0.5, noise=0.5, crackle=0.5,
                bandwidth=0.3, saturation=0.5, seed=0, mix=1.0, output_db=0.0) -> io.NodeOutput:
        """Age every clip of the batch."""
        wave = clip.waveform_of(audio)
        rate = clip.rate_of(audio)
        level = 10.0 ** (float(output_db) / 20.0)
        results = []
        for index in range(int(wave.shape[0])):
            signal = wave[index].double().numpy()
            wet = lofi.worn(signal, rate, medium, age, wow, flutter, noise, crackle, bandwidth, saturation, seed + index)
            out = (signal * (1.0 - mix) + wet * mix) * level
            results.append(torch.from_numpy(out.astype(np.float32)))
        result = torch.stack(results, dim=0)
        facts = {"lofi": {"medium": medium, "age": float(age)}}
        return io.NodeOutput(clip.make(result, rate), ui=before_after(wave, rate, result, rate, facts))

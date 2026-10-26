"""Place a clip in a recorded space by convolving it with an impulse response."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ...modules.audio import clip, delay_effects, reverb
from ...modules.interface.audio_sides import before_after

REQUIRES = "audio"


class AudioConvolutionReverb(io.ComfyNode):
    """A convolution reverb reading its impulse response from a second clip."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioConvolutionReverb",
            display_name="Audio Convolution Reverb",
            search_aliases=[
                "WASAudioConvolutionReverb", "Audio Convolution Reverb", "convolution", "impulse response",
                "IR reverb", "reverb", "cabinet", "space",
            ],
            category="WAS Suite/Audio/Space",
            description=(
                "Put a clip in a real space, or through a speaker cabinet, by convolving it with an "
                "impulse response recorded there and loaded with Load Audio. The node plays the clip "
                "before and after and draws the impulse response's measured decay."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to reverberate."),
                io.Audio.Input(
                    "impulse",
                    tooltip="The impulse response, at most 30 s after length and stretch. Mono = the same "
                    "space on every channel; stereo on a mono clip = a stereo result. A batch pairs clip by clip.",
                ),
                io.Float.Input(
                    "pre_delay", default=0.0, min=0.0, max=500.0, step=1.0, round=0.1,
                    tooltip="Milliseconds the reverb starts late. `0` = as recorded; `30` = keeps a voice "
                    "clear of its tail.",
                ),
                io.Float.Input(
                    "length", default=0.0, min=0.0, max=reverb.MAX_IMPULSE_SECONDS, step=0.01, round=0.001,
                    tooltip="Seconds of the impulse response kept, faded at the end. `0` = all of it; `1.5` "
                    "= a shorter decay from the same space.",
                ),
                io.Float.Input(
                    "stretch", default=1.0, min=0.5, max=2.0, step=0.01, round=0.001,
                    tooltip="Factor on the impulse response's duration. `1` = as recorded; `0.5` = a smaller, "
                    "brighter space; `2` = a larger, darker one.",
                ),
                io.Float.Input(
                    "low_cut", default=0.0, min=0.0, max=2000.0, step=1.0,
                    tooltip="Hertz below which the reverb is cut. `0` = off; `100` = clears boom; `300` = thin.",
                ),
                io.Float.Input(
                    "high_cut", default=0.0, min=0.0, max=24000.0, step=10.0,
                    tooltip="Hertz above which the reverb is cut. `0` = off; `8000` = softer; `4000` = dark.",
                ),
                io.Boolean.Input(
                    "normalise", default=True,
                    tooltip="`true` = the impulse response is brought to unit energy, so mix sets a sane "
                    "level; `false` = used at its recorded level.",
                ),
                io.Float.Input(
                    "mix", default=0.3, min=0.0, max=1.0, step=0.01,
                    tooltip="`0` = the input unchanged; `0.3` = a reverb on a dry track; `1` = the reverb "
                    "only, for a send or a cabinet.",
                ),
                io.Float.Input(
                    "output_db", default=0.0, min=-24.0, max=24.0, step=0.1, round=0.01,
                    tooltip="dB applied to the result. `0` = unchanged.",
                ),
                io.Float.Input(
                    "tail", default=-1.0, min=-1.0, max=60.0, step=0.1, round=0.01,
                    tooltip="Seconds added past the input for the reverb to ring out. `-1` = the impulse "
                    "response's length plus pre_delay; `0` = none, cut at the input's end.",
                    advanced=True,
                ),
            ],
            outputs=[io.Audio.Output(display_name="audio", tooltip="The reverberated clip, longer by the tail.")],
        )

    @classmethod
    def execute(cls, audio, impulse, pre_delay=0.0, length=0.0, stretch=1.0, low_cut=0.0, high_cut=0.0,
                normalise=True, mix=0.3, output_db=0.0, tail=-1.0) -> io.NodeOutput:
        """Convolve every clip of the batch with its impulse response.

        Raises:
            ValueError: The impulse response runs past 30 s, or the tail would make the clip too long.
        """
        wave = clip.waveform_of(audio)
        rate = clip.rate_of(audio)
        response = clip.waveform_of(impulse, "impulse")
        response_rate = clip.rate_of(impulse, "impulse")
        batch, channels, samples = (int(dim) for dim in wave.shape)
        channels = max(channels, int(response.shape[1]))
        sections = delay_effects.cut_sections(rate, low_cut, high_cut)
        prepared = reverb.prepared_impulse(response, response_rate, rate, channels, length, stretch, normalise,
                                           sections)
        taps = int(prepared.shape[-1])
        offset = clip.samples_at(float(pre_delay) / 1000.0, rate)
        seconds = (taps + offset) / float(rate) if float(tail) < 0.0 else float(tail)
        total = samples + clip.samples_at(seconds, rate)
        clip.check_size(batch, channels, total, "tail")
        results = []
        for index in range(batch):
            dry = clip.with_channels(wave[index:index + 1], channels)[0].numpy()
            wet = None
            if float(mix) > 0.0:
                wet = reverb.convolved(dry, prepared[min(index, prepared.shape[0] - 1)], offset, total)
            results.append(torch.from_numpy(delay_effects.blended(dry, wet, mix, output_db, total, channels)))
        out = torch.stack(results, dim=0)
        figures = reverb.decay_figures(prepared[0], rate, noisy=True)
        facts = {"convolution": {
            "impulse_seconds": round(taps / float(rate), 6), "rt60": figures["rt60"], "band": figures["band"],
            "decay": figures["decay"],
        }}
        return io.NodeOutput(clip.make(out, rate), ui=before_after(wave, rate, out, rate, facts))

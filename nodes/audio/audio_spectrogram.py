"""Draw a clip's spectrum over time as a picture."""

from __future__ import annotations

import torch
from comfy_api.latest import io, ui

from ...modules.audio import clip, spectrogram

REQUIRES = "audio"


class AudioSpectrogram(io.ComfyNode):
    """A spectrogram picture of each clip."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAudioSpectrogram",
            display_name="Audio Spectrogram",
            search_aliases=[
                "WASAudioSpectrogram", "Audio Spectrogram", "spectrogram", "spectrum", "audio to image",
                "frequency plot",
            ],
            category="WAS Suite/Audio",
            description=(
                "Draw a clip's spectrum over time as an IMAGE, high frequencies at the top, to "
                "see hum, noise, cut-offs and harmonics, or to feed the sound to an image model."
            ),
            is_output_node=True,
            inputs=[
                io.Audio.Input("audio", tooltip="The clip to draw. Every clip of a batch gets a picture."),
                io.Int.Input("width", default=1024, min=64, max=8192, step=8, tooltip="Picture width in pixels, such as `1024`; time runs left to right."),
                io.Int.Input("height", default=512, min=64, max=4096, step=8, tooltip="Picture height in pixels, such as `512`; frequency runs bottom to top."),
                io.Combo.Input(
                    "scale", options=list(spectrogram.SCALES), default="log",
                    tooltip="`log` = octaves evenly spaced, music reads best; `mel` = hearing's own "
                    "scale, speech reads best; `linear` = hertz evenly spaced, hum stands out.",
                ),
                io.Combo.Input(
                    "fft_size", options=list(spectrogram.FFT_SIZES), default="2048",
                    tooltip="Window in samples: 4096 = finer pitch, blurred timing; 512 = sharp "
                    "timing, coarse pitch; 2048 = between.",
                ),
                io.Float.Input(
                    "floor_db", default=-90.0, min=-160.0, max=-10.0, step=1.0,
                    tooltip="Level drawn darkest, in dB below a full-scale sine. -90 = quiet detail; "
                    "-60 = only the loud parts.",
                ),
                io.Combo.Input(
                    "colours", options=list(spectrogram.COLOURS), default="magma",
                    tooltip="Colour map: `magma` and `inferno` run black to pale yellow, `viridis` "
                    "purple to yellow, `greyscale` black to white.",
                ),
            ],
            outputs=[io.Image.Output(display_name="image", tooltip="One spectrogram picture per clip.")],
        )

    @classmethod
    def execute(cls, audio, width=1024, height=512, scale="log", fft_size="2048", floor_db=-90.0,
                colours="magma") -> io.NodeOutput:
        """Draw every clip."""
        wave = clip.waveform_of(audio)
        rate = clip.rate_of(audio)
        fft = int(fft_size)
        from ...modules.image import sizing

        sizing.check_canvas(int(width), int(height), "The spectrogram", "Lower width or height.")
        count = int(wave.shape[0])
        if count > sizing.batch_limit(int(width), int(height)):
            raise ValueError(
                f"{count} spectrograms of {width}x{height} are more than one batch holds. Lower "
                f"width or height, or split the batch."
            )
        pictures = [
            spectrogram.spectrogram(wave[index], rate, int(width), int(height), fft, scale, floor_db, colours)
            for index in range(int(wave.shape[0]))
        ]
        image = torch.stack(pictures, dim=0)
        return io.NodeOutput(image, ui=ui.PreviewImage(image, cls=cls))

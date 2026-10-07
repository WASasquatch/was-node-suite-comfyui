"""Starting a Kandinsky 6 clip from a picture."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.limits import max_resolution
from ...modules.model.kandinsky6 import conditioning

NODE_NAME = "Kandinsky 6 Image To Video"


class Kandinsky6ImageToVideo(io.ComfyNode):
    """Condition a Kandinsky 6 clip on a start image and answer its empty latent."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASKandinsky6ImageToVideo",
            display_name=NODE_NAME,
            search_aliases=[
                "WASKandinsky6ImageToVideo", NODE_NAME,
                "kandinsky i2v",
                "k6 image to video",
                "image to video audio",
                "start image",
            ],
            category="WAS Suite/Latent/Video",
            description=(
                "Start a Kandinsky 6 clip from a picture: the clip opens on it and moves on from "
                "there, with sound. Answers the conditioning and the empty latent for KSampler; "
                "without a start image it is plain text-to-video."
            ),
            inputs=[
                io.Conditioning.Input("positive", tooltip="The prompt, from CLIP Text Encode or Kandinsky 6 Text Encode."),
                io.Conditioning.Input("negative", tooltip="The negative prompt, from CLIP Text Encode or Kandinsky 6 Text Encode."),
                io.Vae.Input("vae", tooltip="The HunyuanVideo VAE, the same one VAE Decode uses."),
                io.Int.Input(
                    "width", default=864, min=16, max=max_resolution(), step=16,
                    tooltip="Clip width in pixels, a multiple of 16. The start image is resized and center-cropped to it.",
                ),
                io.Int.Input(
                    "height", default=480, min=16, max=max_resolution(), step=16,
                    tooltip="Clip height in pixels, a multiple of 16.",
                ),
                io.Int.Input(
                    "length", default=121, min=1, max=max_resolution(), step=4,
                    tooltip="Frames, 4n + 1: 121 = 5 seconds at 24 fps; 241 = 10 seconds.",
                ),
                io.Float.Input(
                    "fps", default=24.0, min=1.0, max=120.0, step=1.0,
                    tooltip=(
                        "Frame rate the clip plays at, which sets how much sound is generated. "
                        "24 = Kandinsky 6's own; give Create Video the same."
                    ),
                ),
                io.Int.Input(
                    "batch_size", default=1, min=1, max=4096,
                    tooltip="Clips generated at once: 1 = one clip; 2 = two clips, each from its own noise.",
                ),
                io.Image.Input(
                    "start_image", optional=True,
                    tooltip="The picture the clip opens on; the first of a batch is used. Empty = text-to-video.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive", tooltip="The prompt with the start image, for KSampler."),
                io.Conditioning.Output(display_name="negative", tooltip="The negative prompt with the start image, for KSampler."),
                io.Latent.Output(display_name="latent", tooltip="Video and audio latents of one duration, for KSampler."),
            ],
        )

    @classmethod
    def execute(cls, positive, negative, vae, width, height, length, fps, batch_size,
                start_image=None) -> io.NodeOutput:
        latent = conditioning.empty_latent(width, height, length, fps, batch_size)
        if start_image is not None:
            positive, negative = conditioning.with_reference(positive, negative, vae, start_image, width, height)
        return io.NodeOutput(positive, negative, latent)

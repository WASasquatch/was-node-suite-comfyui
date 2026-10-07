"""An empty Kandinsky 6 video and audio latent."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.limits import max_resolution
from ...modules.model.kandinsky6 import conditioning

NODE_NAME = "Empty Kandinsky 6 Latent"


class EmptyKandinsky6Latent(io.ComfyNode):
    """Start a Kandinsky 6 clip from empty video and audio latents."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASEmptyKandinsky6Latent",
            display_name=NODE_NAME,
            search_aliases=[
                "WASEmptyKandinsky6Latent", NODE_NAME,
                "kandinsky latent",
                "k6 latent",
                "text to video audio",
                "empty video audio latent",
            ],
            category="WAS Suite/Latent/Video",
            description=(
                "Start a Kandinsky 6 text-to-video clip: empty video and sound of one duration, "
                "which KSampler generates together. VAE Decode turns the result into frames and "
                "VAE Decode Audio into its soundtrack."
            ),
            inputs=[
                io.Int.Input(
                    "width", default=864, min=16, max=max_resolution(), step=16,
                    tooltip="Clip width in pixels, a multiple of 16. 864 x 480 = the size Kandinsky 6 was trained at.",
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
            ],
            outputs=[
                io.Latent.Output(
                    tooltip="Video and audio latents of one duration, for KSampler.",
                ),
            ],
        )

    @classmethod
    def execute(cls, width, height, length, fps, batch_size) -> io.NodeOutput:
        return io.NodeOutput(conditioning.empty_latent(width, height, length, fps, batch_size))

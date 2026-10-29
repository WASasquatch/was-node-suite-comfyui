"""A MiniMax H3 latent from upscaled frames and the sound of the clip they came from."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.interface import run_result
from ...modules.latent import h3_extend

NODE_NAME = "H3 Hires Latent"

#: The pixel multiple the H3 video VAE encodes.
SIDE_MULTIPLE = 16


class H3HiresLatent(io.ComfyNode):
    """Encode upscaled frames into an H3 latent that keeps the first pass's sound."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASH3HiresLatent",
            display_name=NODE_NAME,
            search_aliases=[
                "WASH3HiresLatent", NODE_NAME,
                "hires",
                "hires fix",
                "second pass",
                "upscale",
                "pika h3 flex",
                "minimax h3",
                "video to video",
            ],
            category="WAS Suite/Latent/Video",
            description=(
                "Encode the upscaled frames of a MiniMax H3 clip into a latent that keeps the clip's "
                "sound, for a second pass at the larger size. Sample it from a low sigma, as Pika "
                "H3 Flex Sampling's 100, 40, 0, with conditioning at that size."
            ),
            inputs=[
                io.Latent.Input(
                    "latent",
                    tooltip="The first pass's video and audio latent, from its sampler. Its sound is kept unchanged.",
                ),
                io.Image.Input(
                    "images",
                    tooltip=(
                        "Every frame of the first pass after upscaling, as 124 for 5 seconds. Width "
                        "and height are multiples of 16, as 1344x768."
                    ),
                ),
                io.Vae.Input(
                    "vae",
                    tooltip="The MiniMax H3 video VAE, as `minimax_h3_video_vae_int8_convrot`.",
                ),
            ],
            outputs=[
                io.Latent.Output(
                    display_name="latent",
                    tooltip="Video and audio latent at the frames' size, the first pass's sound in it.",
                ),
            ],
        )

    @classmethod
    def execute(cls, latent, images, vae) -> io.NodeOutput:
        """Encode the frames beside the first pass's sound.

        Raises:
            ValueError: The latent is not an H3 video and audio latent, the frame count differs
                from the latent's, or a side is not a multiple of 16.
        """
        import comfy.model_management

        video, audio = h3_extend.split(latent)
        frames = h3_extend.frames_for(int(video.shape[2]))
        if int(images.shape[0]) != frames:
            raise ValueError(
                f"the latent holds {frames} frames and {int(images.shape[0])} images arrived. "
                f"Upscale every frame VAE Decode returns for it")
        height, width = int(images.shape[1]), int(images.shape[2])
        if height % SIDE_MULTIPLE or width % SIDE_MULTIPLE:
            raise ValueError(
                f"the frames are {width}x{height}. Upscale to a width and height that are "
                f"multiples of {SIDE_MULTIPLE}, as 1344x768")
        encoded = vae.encode(images[..., :3])
        device = comfy.model_management.intermediate_device()
        run_result.publish(
            summary=f"{width}x{height}, {frames} frames, sound kept",
            counts={"width": width, "height": height, "frames": frames},
        )
        return io.NodeOutput(h3_extend.join(encoded.to(device), audio.to(device)))

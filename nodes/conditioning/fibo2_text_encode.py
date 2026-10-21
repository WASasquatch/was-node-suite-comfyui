"""Encoding a Fibo 2.0 caption, with the images an edit changes."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.limits import max_resolution
from ...modules.interface import run_result
from ...modules.model.fibo2 import conditioning

NODE_NAME = "Text Encode Fibo 2"

#: The image sockets, in order.
IMAGE_SLOTS = ("image_1", "image_2", "image_3", "image_4", "image_5")


class TextEncodeFibo2(io.ComfyNode):
    """Encode a Fibo 2.0 caption, and the images it edits, into conditioning and a latent."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASTextEncodeFibo2",
            display_name=NODE_NAME,
            search_aliases=[
                "WASTextEncodeFibo2", NODE_NAME,
                "fibo", "fibo 2", "bria", "vgl", "image edit", "json prompt",
            ],
            category="WAS Suite/Conditioning",
            description=(
                "Turn a Fibo 2.0 caption into conditioning, with up to five images to edit. Hands on "
                "the prompt and negative for the sampler, and an empty latent of about one megapixel in "
                "image_1's shape, or 1024 square to generate."
            ),
            inputs=[
                io.Clip.Input(
                    "clip",
                    tooltip="qwen3vl_4b_bf16.safetensors from Load CLIP, with type fibo2 where ComfyUI offers it, otherwise krea2.",
                ),
                io.String.Input(
                    "prompt", multiline=True,
                    tooltip=(
                        "A JSON caption, as Generate Fibo 2 Prompt writes, or plain text. To edit: the "
                        "instruction, as 'make the sky stormy', or a caption with an edit_instruction key."
                    ),
                ),
                io.String.Input(
                    "negative_prompt", multiline=True, default="", advanced=True,
                    tooltip=(
                        "What guidance steers away from, as a JSON caption such as "
                        "{\"style_medium\":\"cartoon\"}. Empty = Bria's own: its detailed negative caption to "
                        "generate, the empty caption to edit."
                    ),
                ),
                io.Vae.Input(
                    "vae", optional=True,
                    tooltip="flux2-vae.safetensors from Load VAE. Needed to edit.",
                ),
                io.Image.Input(
                    "image_1", optional=True,
                    tooltip="An image to edit. A batch counts as several images; five at most in all.",
                ),
                io.Image.Input("image_2", optional=True, tooltip="Another image the edit reads, as image_2 in the prompt."),
                io.Image.Input("image_3", optional=True, tooltip="Another image the edit reads, as image_3 in the prompt."),
                io.Image.Input("image_4", optional=True, tooltip="Another image the edit reads, as image_4 in the prompt."),
                io.Image.Input("image_5", optional=True, tooltip="Another image the edit reads, as image_5 in the prompt."),
                io.Mask.Input(
                    "mask", optional=True,
                    tooltip="White where image_1 is repainted, the rest kept. Needs exactly one image.",
                ),
                io.Int.Input(
                    "width", default=0, min=0, max=max_resolution(), step=16,
                    tooltip=(
                        "0 = 1024 to generate, or about one megapixel in image_1's shape to edit, the area "
                        "Fibo 2.0 was trained at. Multiples of 16."
                    ),
                ),
                io.Int.Input(
                    "height", default=0, min=0, max=max_resolution(), step=16,
                    tooltip=(
                        "0 = 1024 to generate, or about one megapixel in image_1's shape to edit, the area "
                        "Fibo 2.0 was trained at. Multiples of 16."
                    ),
                ),
                io.Int.Input(
                    "batch_size", default=1, min=1, max=4096,
                    tooltip="Images sampled per run from the one prompt: 1 = one image, 4 = four variations.",
                ),
                io.Boolean.Input(
                    "text_images", default=True, advanced=True,
                    tooltip=(
                        "true = Qwen3-VL also sees each text_render text drawn as a picture, as in "
                        "training, for cleaner lettering; false = the words alone."
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive", tooltip="The prompt, for the sampler's positive."),
                io.Conditioning.Output(display_name="negative", tooltip="The negative, for the sampler's negative."),
                io.Latent.Output(display_name="latent", tooltip="An empty latent at the image size, to sample into."),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, width, height, batch_size, negative_prompt="", vae=None, image_1=None,
                image_2=None, image_3=None, image_4=None, image_5=None, mask=None, text_images=True) -> io.NodeOutput:
        images = [image_1, image_2, image_3, image_4, image_5]
        positive, negative, latent = conditioning.conditioning(
            clip, prompt, negative_prompt, images=images, mask=mask, vae=vae, width=width, height=height,
            batch_size=batch_size, draw_text=text_images,
        )
        samples = latent["samples"]
        size = f"{samples.shape[-1] * 16}x{samples.shape[-2] * 16}"
        references = positive[0][1].get("reference_latents") or []
        tokens = [entry[0].shape[1] for entry in positive]
        facts = {"mode": "edit" if references else "generate", "size": size}
        if references:
            facts["images"] = ", ".join(f"{ref.shape[-1] * 16}x{ref.shape[-2] * 16}" for ref in references)
        if len(positive) > 1:
            facts["text images"] = f"shown from sigma {conditioning.prompts.TEXT_IMAGE_SIGMA} down"
        run_result.publish(
            summary=f"{facts['mode'].capitalize()} at {size}, {max(tokens)} prompt tokens",
            counts={"prompt tokens": max(tokens), "negative tokens": int(negative[0][0].shape[1])},
            facts=facts,
        )
        return io.NodeOutput(positive, negative, latent)

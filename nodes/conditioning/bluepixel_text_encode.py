"""Encoding a BluePixel edit: the instruction, and the picture it changes."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.interface import size_report
from ...modules.model.bluepixel import edit, model as bluepixel

NODE_NAME = "Text Encode BluePixel"

#: Width of the FLUX.2 Klein 9B text features: three Qwen3 8B layers of 4096.
TEXT_WIDTH = 12288


class TextEncodeBluePixel(io.ComfyNode):
    """Encode a BluePixel edit instruction, with the picture it edits, into conditioning and a latent."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASTextEncodeBluePixel",
            display_name=NODE_NAME,
            search_aliases=[
                "WASTextEncodeBluePixel", NODE_NAME,
                "bluepixel", "blue pixel", "bluepixel edit", "4k image edit", "instruction edit",
            ],
            category="WAS Suite/Conditioning",
            description=(
                "Turn a BluePixel edit instruction and the picture it changes into conditioning for a "
                "CFG Guider. Hands on the prompt and negative, both carrying the picture, an empty "
                "latent of the edit's size, and the picture at that size."
            ),
            inputs=[
                io.Clip.Input(
                    "clip",
                    tooltip="qwen_3_8b.safetensors or qwen_3_8b_fp8mixed.safetensors from Load CLIP with type flux2.",
                ),
                io.String.Input(
                    "prompt", multiline=True,
                    tooltip=(
                        "The edit in English, as 'Change the gray sweater to burgundy red. Preserve the "
                        "person, pose and background.'"
                    ),
                ),
                io.Vae.Input("vae", tooltip="DC-AE f64c128 from BluePixel VAE Loader."),
                io.Image.Input(
                    "image",
                    tooltip="The picture to edit. A batch edits each picture with the same instruction.",
                ),
                io.Int.Input(
                    "resolution", default=4096, min=0, max=8192, step=64,
                    tooltip=(
                        "The edit's pixel count as a square side: 4096 = 16.8 MP, as released, where faces "
                        "hold best; 2048 = 4.2 MP; 0 = the picture's own. The shape follows the picture in "
                        "steps of 64."
                    ),
                ),
                io.String.Input(
                    "negative_prompt", multiline=True, default="", advanced=True,
                    tooltip="What guidance steers away from, as 'blurry, oversaturated'. Empty = BluePixel's own, the empty instruction.",
                ),
            ],
            outputs=[
                io.Conditioning.Output(
                    "positive", tooltip="The instruction with the picture as its reference, for positive on CFG Guider.",
                ),
                io.Conditioning.Output(
                    "negative", tooltip="The negative with the same reference, for negative on CFG Guider at cfg 3.",
                ),
                io.Latent.Output(
                    "latent", tooltip="An empty latent of the edit's size, for SamplerCustomAdvanced and BluePixel Scheduler.",
                ),
                io.Image.Output(
                    "image", tooltip="The picture brought to the edit's size, the before for a comparison.",
                ),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, vae, image, resolution, negative_prompt="") -> io.NodeOutput:
        """Encode both prompts, size and encode the picture, and attach it to both.

        Raises:
            ValueError: The CLIP is not FLUX.2 Klein 9B's text encoder, or the VAE is not DC-AE f64c128.
        """
        import torch

        if getattr(vae, "latent_channels", None) != bluepixel.CHANNELS or vae.spacial_compression_encode() != bluepixel.COMPRESSION:
            raise ValueError(
                f"{NODE_NAME} needs DC-AE f64c128, which encodes {bluepixel.CHANNELS} channels at a "
                f"{bluepixel.COMPRESSION}th of the size. Load it with BluePixel VAE Loader."
            )
        height, width = int(image.shape[1]), int(image.shape[2])
        target = int(resolution) * int(resolution)
        out_w, out_h = edit.edit_size(width, height, target)
        sized = edit.resize(image, out_w, out_h)
        reference = vae.encode(sized)

        positive = _encoded(clip, prompt)
        negative = _encoded(clip, negative_prompt)
        positive = _with_reference(positive, reference)
        negative = _with_reference(negative, reference)

        tokens_w, tokens_h = out_w // bluepixel.COMPRESSION, out_h // bluepixel.COMPRESSION
        latent = torch.zeros(
            (sized.shape[0], bluepixel.CHANNELS, tokens_h, tokens_w),
            device=reference.device, dtype=reference.dtype,
        )
        size_report.publish(
            image, sized, action="resized for the edit",
            facts={"tokens": f"{tokens_w * tokens_h:,} edited + {tokens_w * tokens_h:,} reference"},
        )
        return io.NodeOutput(positive, negative, {"samples": latent}, sized)


def _encoded(clip, text: str):
    """``text`` encoded the way CLIP Text Encode encodes it, checked to be FLUX.2 Klein 9B's features."""
    conditioning = clip.encode_from_tokens_scheduled(clip.tokenize(text))
    width = int(conditioning[0][0].shape[-1])
    if width != TEXT_WIDTH:
        raise ValueError(
            f"{NODE_NAME} needs FLUX.2 Klein 9B's text features, {TEXT_WIDTH} wide, and this CLIP gives "
            f"{width}. Load qwen_3_8b.safetensors with Load CLIP and set its type to flux2."
        )
    return conditioning


def _with_reference(conditioning, reference):
    """Each entry of ``conditioning`` with ``reference`` appended to its reference latents."""
    out = []
    for cond, options in conditioning:
        options = options.copy()
        options["reference_latents"] = list(options.get("reference_latents") or []) + [reference]
        out.append([cond, options])
    return out

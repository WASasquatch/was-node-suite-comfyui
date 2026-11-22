"""MiniMax H3 Image conditioning with picture rows added for photographic detail."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.limits import max_resolution
from ...modules.interface import run_result
from ...modules.sampling import h3_visrow

NODE_NAME = "H3 Image VisRow Conditioning"

#: The reference picture sockets, in order: ``image_1`` is Picture 1 in the prompt.
IMAGE_SLOTS = tuple(f"image_{index}" for index in range(1, 10))


class H3ImageVisRowConditioning(io.ComfyNode):
    """Encode a MiniMax H3 Image prompt and its reference pictures, then add picture rows."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASH3ImageVisRowConditioning",
            display_name=NODE_NAME,
            search_aliases=[
                "WASH3ImageVisRowConditioning", NODE_NAME,
                "visrow", "vision rows", "minimax h3 image", "h3 image conditioning",
                "blank picture", "photographic detail", "skin texture",
            ],
            category="WAS Suite/Conditioning",
            description=(
                "Encode a MiniMax H3 Image prompt, with up to nine reference pictures, and add picture "
                "rows that make skin, fur and stone more photographic. Strength 1 matches reading the "
                "prompt beside a blank picture, while Picture 1 stays free for a real reference."
            ),
            inputs=[
                io.Clip.Input("clip", tooltip="The MiniMax H3 Image text encoder, from Load CLIP with type minimax."),
                io.Vae.Input(
                    "vae", optional=True,
                    tooltip="The MiniMax H3 Image VAE. With it the reference pictures are also anchored as latents.",
                ),
                io.String.Input(
                    "prompt", multiline=True,
                    tooltip="What to make, or the edit, naming references as <Picture 1>, <Picture 2>: `Put the hat from <Picture 2> on the man in <Picture 1>`.",
                ),
                io.Int.Input(
                    "width", default=0, min=0, max=max_resolution(), step=32,
                    tooltip="Width of the picture: `0` = image_1's width after its reference scaling, or 2048 without references.",
                ),
                io.Int.Input(
                    "height", default=0, min=0, max=max_resolution(), step=32,
                    tooltip="Height of the picture: `0` = image_1's height after its reference scaling, or 2048 without references.",
                ),
                io.Float.Input(
                    "strength", default=1.0, min=0.0, max=4.0, step=0.05,
                    tooltip=(
                        "Picture rows added, in pictures' worth: `1` = 1026 rows, as a blank picture adds; "
                        "`2` = crisper still; `0` = MiniMax H3 Image Conditioning unchanged."
                    ),
                ),
                io.Combo.Input(
                    "placement",
                    options=list(h3_visrow.PLACEMENTS),
                    default=h3_visrow.AFTER,
                    advanced=True,
                    tooltip="`after prompt` = the rows follow the prompt; `before prompt` = between the reference pictures and the prompt.",
                ),
                io.Image.Input("image_1", optional=True, tooltip="Picture 1 in the prompt, and the picture an edit changes."),
                io.Image.Input("image_2", optional=True, tooltip="Picture 2 in the prompt."),
                io.Image.Input("image_3", optional=True, tooltip="Picture 3 in the prompt."),
                io.Image.Input("image_4", optional=True, tooltip="Picture 4 in the prompt."),
                io.Image.Input("image_5", optional=True, tooltip="Picture 5 in the prompt."),
                io.Image.Input("image_6", optional=True, tooltip="Picture 6 in the prompt."),
                io.Image.Input("image_7", optional=True, tooltip="Picture 7 in the prompt."),
                io.Image.Input("image_8", optional=True, tooltip="Picture 8 in the prompt."),
                io.Image.Input("image_9", optional=True, tooltip="Picture 9 in the prompt."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive", tooltip="The prompt with its picture rows, for the guider or sampler."),
                io.Latent.Output(display_name="latent", tooltip="An empty latent at the picture size, to sample into."),
            ],
        )

    @classmethod
    def execute(cls, clip, prompt, width, height, strength, placement=h3_visrow.AFTER, vae=None,
                image_1=None, image_2=None, image_3=None, image_4=None, image_5=None, image_6=None,
                image_7=None, image_8=None, image_9=None) -> io.NodeOutput:
        """Encode through MiniMax H3 Image Conditioning, then add the picture rows.

        Raises:
            ValueError: The prompt is empty, the placement is unknown, or the text encoder is not MiniMax H3 Image's.
        """
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageConditioning

        if not str(prompt or "").strip():
            raise ValueError(f"{NODE_NAME} has no prompt. Type what to make.")
        if placement not in h3_visrow.PLACEMENTS:
            raise ValueError(f"{NODE_NAME} has no placement {placement!r}. Choose {' or '.join(h3_visrow.PLACEMENTS)}.")
        slots = (image_1, image_2, image_3, image_4, image_5, image_6, image_7, image_8, image_9)
        images = {name: image for name, image in zip(IMAGE_SLOTS, slots) if image is not None}
        core = MiniMaxH3ImageConditioning.execute(
            clip=clip, prompt=str(prompt), width=width, height=height, vae=vae, images=images or None
        )
        positive, latent = core.args[0], core.args[1]
        count = h3_visrow.row_count(strength)
        if count:
            positive = h3_visrow.add_rows(positive, h3_visrow.picture_row(clip), count, placement)

        samples = latent["samples"]
        size = f"{samples.shape[-1] * 16}x{samples.shape[-2] * 16}"
        tags = positive[0][1].get("minimax_token_tags")
        vision = 0 if tags is None else int((tags.reshape(-1) == 0).sum())
        facts = {"mode": "edit" if images else "generate", "size": size}
        if count:
            facts["picture rows"] = f"{count}, {placement}"
        if images:
            facts["references"] = ", ".join(name for name in IMAGE_SLOTS if name in images)
        run_result.publish(
            summary=f"{facts['mode'].capitalize()} at {size}, {count} picture rows",
            counts={
                "prompt rows": int(positive[0][0].shape[1]) - vision,
                "picture rows": count,
                "reference rows": vision - count,
            },
            facts=facts,
        )
        return io.NodeOutput(positive, latent)

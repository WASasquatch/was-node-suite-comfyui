"""Retouch the skin of a portrait: blotchy tone evened, small spots healed, texture kept."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.image import retouch

logger = log.get_logger("nodes.image.develop")


class ImagePortraitRetouch(io.ComfyNode):
    """Even blotchy skin tone and heal small spots inside a skin mask, keeping pore texture."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImagePortraitRetouch",
            display_name="Image Portrait Retouch",
            search_aliases=[
                "WASImagePortraitRetouch", "Image Portrait Retouch", "retouch", "skin retouch",
                "blemish", "spot healing", "acne", "frequency separation", "even skin tone", "redness",
            ],
            category="WAS Suite/Image/Develop",
            description=(
                "Retouch the skin inside a skin mask: blotchy redness and patchy tone evened, small "
                "spots healed with nearby skin, pore texture kept, everything outside the mask "
                "untouched. It leaves large acne fields, wrinkles, deep shadows and face shape alone. "
                "On video the evening holds steady but healed spots flicker, so heal stills."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The portraits to retouch, as picture codes. A batch is retouched frame by frame; an alpha channel passes through.",
                ),
                io.Mask.Input(
                    "mask",
                    tooltip="The skin to retouch: white = retouched, black = left exactly as it was. Wire Image Skin Mask or Image Simple Skin Mask; mouth = 'remove lips' keeps lip colour whole. One mask per frame or one for all.",
                ),
                io.Float.Input(
                    "even_tone", default=0.5, min=0.0, max=1.0, step=0.01,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="Blotchy redness and patchy colour evened toward the skin around them. 0 = off; 0.5 = calmer, still natural; 1 = most blotches, flush and dark marks gone. Shading deeper than a blotch, such as cheekbones and the nose's side, stays.",
                ),
                io.Float.Input(
                    "texture", default=1.0, min=0.0, max=1.0, step=0.01,
                    display_mode=io.NumberDisplay.slider,
                    tooltip="Fine texture kept: pores, fine lines, grain. 1 = all of it; 0.5 = a little softer; 0 = fine texture at about half its strength on smooth skin. Deep lines and feature edges stay crisp at every setting.",
                ),
                io.Combo.Input(
                    "blemishes", options=list(retouch.BLEMISHES), default="red spots",
                    tooltip="'red spots' = heal small red spots such as pimples, keeping freckles and moles; 'all spots' = small dark spots, freckles and moles too; 'off' = no healing. Each spot is replaced by nearby skin, texture and all.",
                ),
                io.Float.Input(
                    "blemish_size", default=0.03, min=0.005, max=0.08, step=0.005,
                    tooltip="Largest spot healed, as a share of the skin region's size, the square root of its area. 0.03 on a face mask 600 px across heals spots up to about 18 px; 0.06 = twice as wide.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="The retouched pictures. Every pixel outside the mask is the input's own.",
                ),
                io.Mask.Output(
                    display_name="healed",
                    tooltip="Where spots were healed: 1 = replaced by nearby skin, 0 = untouched by healing, soft at each edge. Preview it to check every heal.",
                ),
            ],
        )

    @classmethod
    def execute(cls, images, mask, even_tone=0.5, texture=1.0, blemishes="red spots", blemish_size=0.03) -> io.NodeOutput:
        """Retouch every frame and answer the batch with a mask of healed spots.

        Raises:
            ValueError: The pictures carry fewer than three channels, or ``blemishes`` names no
                option the node offers.
            MemoryError: Neither free memory nor a scratch drive can hold the result.
        """
        if blemishes not in retouch.BLEMISHES:
            raise ValueError(
                f"Image Portrait Retouch has no blemishes option {blemishes!r}. "
                f"Choose one of: {', '.join(retouch.BLEMISHES)}."
            )
        if images.ndim != 4 or int(images.shape[-1]) < 3:
            raise ValueError(
                f"Image Portrait Retouch needs colour pictures with 3 or 4 channels, and this batch "
                f"is shaped {tuple(images.shape)}. Convert it with Images to RGB first."
            )
        if float(mask.max()) <= 0.0:
            logger.warning(
                "Image Portrait Retouch: the mask holds no skin, so the pictures pass through "
                "unchanged. Wire Image Skin Mask or Image Simple Skin Mask into mask."
            )
        advance = None
        try:
            import comfy.model_management
            import comfy.utils

            bar = comfy.utils.ProgressBar(int(images.shape[0]))

            def advance(count):
                comfy.model_management.throw_exception_if_processing_interrupted()
                bar.update(count)
        except ImportError:
            pass
        result, healed, count = retouch.retouch(
            images, mask, float(even_tone), float(texture), blemishes, float(blemish_size), advance,
        )
        logger.info(
            "Image Portrait Retouch retouched %d frame(s) and healed %d spot(s)",
            int(images.shape[0]), count,
        )
        return io.NodeOutput(result, healed)

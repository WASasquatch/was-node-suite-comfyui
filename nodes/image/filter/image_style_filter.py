"""Photo-app and modern colour grades."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.convert.tensors import filtered_planes
from ....modules.interface import preview

#: The style menu, in alphabetical order. Every entry but :data:`SPARKLE_STYLE` names a
#: grade in :data:`modules.image.recipes.RECIPES`.
STYLES = [
    "1977",
    "aden",
    "bleach bypass",
    "brannan",
    "brooklyn",
    "clarendon",
    "clean punch",
    "cross process",
    "earlybird",
    "faded film",
    "fairy tale",
    "film noir",
    "gingham",
    "golden hour",
    "hudson",
    "inkwell",
    "kelvin",
    "lark",
    "lofi",
    "maven",
    "mayfair",
    "moody blue",
    "moon",
    "nashville",
    "neon night",
    "perpetua",
    "reyes",
    "rise",
    "slumber",
    "soft portrait",
    "stinson",
    "teal and orange",
    "toaster",
    "valencia",
    "walden",
    "willow",
    "xpro2",
]

#: The one style with no recipe behind it. Drawn by the pack's own glitter pass.
SPARKLE_STYLE = "fairy tale"

#: The slot the ungraded picture and the graded one are filed under, on the input side and the
#: output side of the node.
PAIR_SLOT = "image"


class ImageStyleFilter(io.ComfyNode):
    """Apply one of 37 photo-app and modern colour treatments to an image."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Style Filter",
            display_name="Image Style Filter",
            search_aliases=[
                "Image Style Filter",
                "instagram",
                "pilgram",
                "photo filter",
                "colour grade",
                "look",
            ],
            category="WAS Suite/Image/Filter",
            description=(
                "Recolour an image with one of 37 looks, from familiar photo-app filters to "
                "modern grades such as `teal and orange` and `bleach bypass`. A contact sheet "
                "on the node shows every look over the picture, and a click on a tile picks it."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="The image to treat. A batch is handled one image at a time.",
                ),
                io.Combo.Input(
                    "style",
                    options=STYLES,
                    tooltip=(
                        "Which of the 37 looks to apply. `inkwell`, `moon` = black and white; "
                        "`kelvin`, `golden hour` = warmest; `moody blue` = coolest; `bleach "
                        "bypass` = hardest; `fairy tale` = random glitter, not a grade, "
                        "different every run."
                    ),
                ),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    tooltip=(
                        "How far towards the look the result sits. 1.0 is the whole look, "
                        "0.5 half of it, 0.0 the image unchanged. Ignored by `fairy tale`."
                    ),
                ),
                io.Boolean.Input(
                    "contact_sheet",
                    default=True,
                    tooltip=(
                        "`true` draws every style over this picture on the node, so one run "
                        "shows all 37 and a cell can be clicked to pick it. `false` draws the "
                        "picture beside the graded result with the difference between them, "
                        "which is quicker on a large frame."
                    ),
                ),
                io.Boolean.Input(
                    "use_gpu",
                    default=True,
                    tooltip=(
                        "`true` = grades on the graphics card, several times quicker on a large "
                        "frame; `false` = on the processor. A card that refuses the work falls "
                        "back to the processor."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip=(
                        "The styled images, one for each that went in and the same size as the "
                        "source. Every image in a batch gets the same style, and under `fairy "
                        "tale` each one gets its own glitter."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls, image, style, strength=1.0, contact_sheet=True, use_gpu=True
    ) -> io.NodeOutput:
        """Grade every image in the batch.

        Raises:
            ValueError: The style is not one of :data:`STYLES`.
        """
        from ....modules.image import recipes

        if style == SPARKLE_STYLE:
            from ....modules.image.filters import sparkle

            result = filtered_planes(image, sparkle)
        elif not recipes.known(style):
            raise ValueError(
                f"{style!r} is not a look this node has. Choose one of the {len(STYLES)} "
                f"names in the style menu, such as 'golden hour' or 'clarendon'. A value "
                f"arriving on the style socket has to match one of them exactly, in lower "
                f"case."
            )
        else:
            result = recipes.apply(image, style, float(strength), bool(use_gpu))

        # Publishing changes nothing this returns, and does nothing at all while no browser
        # is connected.
        if contact_sheet:
            preview.publish_frames(cls.sheet(image, bool(use_gpu)))
        else:
            preview.publish(image, slot=PAIR_SLOT)
            preview.publish_output(result, slot=PAIR_SLOT)

        return io.NodeOutput(result)

    #: Longest side of a contact sheet tile, in pixels.
    TILE = 128

    @classmethod
    def sheet(cls, image, use_gpu: bool = True):
        """Every style drawn over one picture, small, in menu order.

        Args:
            image: The batch to draw from. Only its first picture is used.
            use_gpu: Whether to grade on ComfyUI's compute device.

        Returns:
            A ``(1 + len(STYLES), height, width, 3)`` batch: the picture untouched, then one
            tile per style.
        """
        import torch

        from ....modules.image import recipes

        first = image[:1]
        height, width = first.shape[1], first.shape[2]
        longest = max(height, width)
        if longest > cls.TILE:
            scale = cls.TILE / longest
            first = torch.nn.functional.interpolate(
                first.permute(0, 3, 1, 2),
                size=(max(1, round(height * scale)), max(1, round(width * scale))),
                mode="area",
            ).permute(0, 2, 3, 1)

        tiles = [first]
        for name in STYLES:
            # The glitter is drawn from an unseeded generator, so no tile of it is true.
            tiles.append(first if name == SPARKLE_STYLE else recipes.apply(first, name, 1.0, use_gpu))
        return torch.cat(tiles, dim=0)

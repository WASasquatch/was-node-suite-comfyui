"""Background removal."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import REMBG_MODEL
from ...modules.image import cutout

REQUIRES = "preprocessors"

logger = log.get_logger("nodes.ai.image_rembg")

#: Strings that stand for false once folded to lower case.
FALSE_WORDS = frozenset({"false", "none", "decimal(0)", "fraction(0,1)", "set()", "range(0)"})

#: Strings that stand for false exactly as written: the repr of every falsey builtin.
FALSE_LITERALS = frozenset({"0", "0.0", "0j", "''", '""', "()", "[]", "{}"})


def as_bool(value) -> bool:
    """Read a widget value as a boolean, including a string spelling of one.

    Args:
        value: The widget value.

    Returns:
        The value as a boolean.
    """
    if type(value) is str:
        value = value.strip()
        return not (value.lower() in FALSE_WORDS or value in FALSE_LITERALS)
    return bool(value)


class ImageRembg(io.ComfyNode):
    """Cut the subject out of every image in a batch."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Rembg (Remove Background)",
            display_name="Image Remove Background",
            search_aliases=[
                "Image Rembg (Remove Background)",
                "Image Remove Background",
                "remove background",
                "cutout",
                "birefnet",
                "ben2",
            ],
            category="WAS Suite/Image/AI",
            description=(
                "Remove the background from an image, leaving the subject on transparency or "
                "on a flat colour, and hand on the cutout as a mask for further compositing. "
                "The cutout network comes from Image Remove Background Model Loader."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The images to cut out. Every image in the batch is processed.",
                ),
                REMBG_MODEL.Input(
                    "rembg_model",
                    tooltip=(
                        "The cutout network, from Image Remove Background Model Loader, "
                        "which is where the choice of model is made. One loader can feed "
                        "several nodes so the network is built once."
                    ),
                ),
                io.Boolean.Input(
                    "transparency",
                    default=True,
                    tooltip=(
                        "On, the background becomes transparent and the result carries an "
                        "alpha channel. Off, the result is a plain colour image and the "
                        "background is whatever background_color says, or black when that is "
                        "`none`."
                    ),
                ),
                io.Boolean.Input(
                    "post_processing",
                    default=False,
                    tooltip=(
                        "Tidy the cutout by removing stray specks and filling pinholes. Helps "
                        "on a busy background and can nibble at thin details such as stray "
                        "hairs."
                    ),
                ),
                io.Boolean.Input(
                    "only_mask",
                    default=False,
                    tooltip=(
                        "Return the cutout shape itself as a greyscale image, white where the "
                        "subject is, instead of the subject's pixels. Useful as a mask for "
                        "another node. The result is always three channels, so transparency "
                        "and background_color do nothing while this is on."
                    ),
                ),
                io.Boolean.Input(
                    "alpha_matting",
                    default=False,
                    tooltip=(
                        "Refine the edge with alpha matting, which recovers soft detail such "
                        "as hair and fur. Noticeably slower, and it is what the three "
                        "alpha_matting values below control."
                    ),
                ),
                io.Int.Input(
                    "alpha_matting_foreground_threshold",
                    default=240,
                    min=0,
                    max=255,
                    tooltip=(
                        "How certain a pixel has to be to count as definitely the subject, "
                        "from 0 to 255. Lower takes in more of the edge as subject; the "
                        "default 240 keeps only the most confident core."
                    ),
                ),
                io.Int.Input(
                    "alpha_matting_background_threshold",
                    default=10,
                    min=0,
                    max=255,
                    tooltip=(
                        "How certain a pixel has to be to count as definitely background, "
                        "from 0 to 255. Higher discards more of the edge; the default 10 "
                        "leaves everything between the two thresholds for the matting to "
                        "decide."
                    ),
                ),
                io.Int.Input(
                    "alpha_matting_erode_size",
                    default=10,
                    min=0,
                    max=255,
                    tooltip=(
                        "How far in from the edge, in pixels, the uncertain band is grown "
                        "before matting. Larger values give the matting more room to work and "
                        "soften the edge; 0 leaves the band as the thresholds drew it."
                    ),
                ),
                io.Combo.Input(
                    "background_color",
                    default="none",
                    options=["none", "black", "white", "magenta", "chroma green", "chroma blue"],
                    tooltip=(
                        "What fills the background: `none` = left empty; `chroma green`, "
                        "`chroma blue` = keying colours; `magenta` = shows leftovers clearly. "
                        "With transparency on, the fill only shows once the alpha channel is "
                        "discarded."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip=(
                        "The cut-out images, as a batch the same length as the input. Four "
                        "channels when transparency is on, three when it is off or when "
                        "only_mask is on."
                    ),
                ),
                io.Mask.Output(
                    display_name="mask",
                    tooltip=(
                        "The cutout as a mask, one per image: 1.0 = subject kept, 0.0 = "
                        "background removed, soft values on the edge. The same values as the "
                        "alpha of images, after post_processing and alpha_matting. Mask Invert "
                        "turns it into the background."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        images,
        rembg_model,
        transparency,
        post_processing,
        only_mask,
        alpha_matting,
        alpha_matting_foreground_threshold,
        alpha_matting_background_threshold,
        alpha_matting_erode_size,
        background_color,
    ) -> io.NodeOutput:
        """Cut the subject out of every frame.

        Raises:
            ArithmeticError: alpha_matting was asked for and its solve did not settle.
        """
        transparency = as_bool(transparency)
        alpha_matting = as_bool(alpha_matting)
        post_processing = as_bool(post_processing)
        only_mask = as_bool(only_mask)

        logger.info(
            "Removing the background from %s image(s) with %s.",
            len(images),
            rembg_model.name,
        )
        mattes = cutout.mattes(rembg_model, images)
        if alpha_matting:
            mattes = cutout.refine(
                images,
                mattes,
                float(alpha_matting_foreground_threshold) / 255.0,
                float(alpha_matting_background_threshold) / 255.0,
                int(alpha_matting_erode_size),
            )
        # The one matte both outputs carry.
        shape = (cutout.tidy(mattes) if post_processing else mattes).clamp(0.0, 1.0)
        mask = shape.float().contiguous()
        if only_mask:
            shown = shape.unsqueeze(-1).expand(-1, -1, -1, 3)
            return io.NodeOutput(shown.contiguous(), mask)
        return io.NodeOutput(
            cutout.compose(images, shape, transparency, background_color, False),
            mask,
        )

"""Lens-style colour fringing."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.convert.tensors import filtered_planes


def lingrad(size, direction: str, white_ratio: int):
    """Draw a ramp that is black except for a white band along one edge.

    Args:
        size: ``(width, height)`` in pixels.
        direction: ``'vertical'`` puts the band along the bottom edge, ``'horizontal'``
            along the right edge. Any other value leaves the image black.
        white_ratio: Width of the ramp in pixels. 0 leaves the image entirely black.

    Returns:
        A mode ``L`` PIL image of ``size``.
    """
    import numpy as np
    from PIL import Image

    width, height = int(size[0]), int(size[1])
    image = np.zeros((height, width), dtype=np.uint8)
    if direction in ('vertical', 'horizontal'):
        length = height if direction == 'vertical' else width
        black_end = length - int(white_ratio)
        positions = np.arange(length, dtype=np.int64)
        lit = positions > black_end
        ramp = np.zeros(length, dtype=np.uint8)
        if lit.any():
            ramp[lit] = (((positions[lit] - black_end) / (length - black_end)) * 255).astype(
                np.int64
            )
        image[:] = ramp[:, None] if direction == 'vertical' else ramp[None, :]

    return Image.fromarray(image)


def create_fade_mask(size, fade_radius: int):
    """Build a mask that is white in the middle and falls to black at all four edges.

    Args:
        size: ``(width, height)`` in pixels.
        fade_radius: Half-width of the fade in pixels. 0 gives a mask that is white
            everywhere, so the effect reaches the edges of the frame.

    Returns:
        A mode ``L`` PIL image of ``size``.
    """
    from PIL import Image, ImageChops, ImageOps

    mask = Image.new("L", size, 255)

    left = ImageOps.invert(lingrad(size, 'horizontal', int(fade_radius * 2)))
    right = left.copy().transpose(Image.FLIP_LEFT_RIGHT)
    top = ImageOps.invert(lingrad(size, 'vertical', int(fade_radius * 2)))
    bottom = top.copy().transpose(Image.FLIP_TOP_BOTTOM)

    mask = ImageChops.multiply(mask, left)
    mask = ImageChops.multiply(mask, right)
    mask = ImageChops.multiply(mask, top)
    mask = ImageChops.multiply(mask, bottom)
    mask = ImageChops.multiply(mask, mask)

    return mask


def apply_chromatic_aberration(img, r_offset: int, g_offset: int, b_offset: int,
                               fade_radius: int, intensity: float = 1.0, fade_mask=None):
    """Shift an image's colour channels apart, fading the effect out towards the edges.

    Args:
        img: Source PIL image, three-channel.
        r_offset: Horizontal shift of the red channel in pixels.
        g_offset: Vertical shift of the green channel in pixels.
        b_offset: Vertical shift of the blue channel in pixels.
        fade_radius: Half-width in pixels of the fade back to the unshifted image at the
            frame edges.
        intensity: How much of the shifted image is mixed in, 0.0 to 1.0.
        fade_mask: The mask :func:`create_fade_mask` answers for this size and radius, or
            None to build it here. A mask of another size is ignored.

    Returns:
        An ``RGB`` PIL image the same size as the source.

    Raises:
        ValueError: The image does not have exactly three channels, so the split does not
            unpack into three names.
    """
    from PIL import Image, ImageChops

    r, g, b = img.split()

    r_offset_img = ImageChops.offset(r, r_offset, 0)
    g_offset_img = ImageChops.offset(g, 0, g_offset)
    b_offset_img = ImageChops.offset(b, 0, b_offset)

    merged = Image.merge("RGB", (r_offset_img, g_offset_img, b_offset_img))
    strength = min(1.0, max(0.0, float(intensity)))
    if strength < 1.0:
        merged = Image.blend(img.convert("RGB"), merged, strength)
    if fade_mask is None or fade_mask.size != img.size:
        fade_mask = create_fade_mask(img.size, fade_radius)

    return Image.composite(merged, img, fade_mask).convert("RGB")


class ImageChromaticAberration(io.ComfyNode):
    """Separate an image's red, green and blue channels to fake lens colour fringing."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Chromatic Aberration",
            display_name="Image Chromatic Aberration",
            search_aliases=[
                "Image Chromatic Aberration",
                "chromatic aberration",
                "colour fringe",
                "rgb shift",
                "lens",
            ],
            category="WAS Suite/Image/Filter",
            description=(
                "Nudge the red, green and blue channels apart so edges pick up coloured "
                "fringes, the way a cheap lens does. The effect fades out towards the edges "
                "of the frame."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="The image to shift. A batch is handled one image at a time.",
                ),
                io.Int.Input(
                    "red_offset",
                    default=2,
                    min=-255,
                    max=255,
                    step=1,
                    tooltip=(
                        "How far the red channel moves sideways, in pixels. Positive is right, "
                        "negative is left, 0 leaves it in place. 2 is a subtle fringe, 20 is "
                        "obvious."
                    ),
                ),
                io.Int.Input(
                    "green_offset",
                    default=-1,
                    min=-255,
                    max=255,
                    step=1,
                    tooltip=(
                        "How far the green channel moves vertically, in pixels. Positive is "
                        "down, negative is up, 0 leaves it in place."
                    ),
                ),
                io.Int.Input(
                    "blue_offset",
                    default=1,
                    min=-255,
                    max=255,
                    step=1,
                    tooltip=(
                        "How far the blue channel moves vertically, in pixels, on the same "
                        "positive-is-down reading as green_offset. Giving green and blue "
                        "opposite signs is what produces the classic red-and-cyan fringe."
                    ),
                ),
                io.Float.Input(
                    "intensity",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How much of the shifted result is mixed back over the original. "
                        "1.0 = the offsets in full, 0.5 = half the fringing, 0.0 = the "
                        "picture unchanged. Use it to dial one setting rather than "
                        "rebalancing all three offsets."
                    ),
                ),
                io.Int.Input(
                    "fade_radius",
                    default=12,
                    min=0,
                    max=1024,
                    step=1,
                    tooltip=(
                        "How far in from each edge the effect fades back to the untouched "
                        "image, in pixels. 0 applies the shift right to the border, which "
                        "exposes the wrapped-around strip; 12 hides it; large values confine the "
                        "fringing to the centre of the frame."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(tooltip="The image with its colour channels shifted apart."),
            ],
        )

    @classmethod
    def execute(cls, image, red_offset, green_offset, blue_offset, intensity,
                fade_radius) -> io.NodeOutput:
        mask = None
        if getattr(image, "ndim", 0) == 4:
            mask = create_fade_mask((int(image.shape[2]), int(image.shape[1])), fade_radius)
        return io.NodeOutput(filtered_planes(
            image,
            lambda plane: apply_chromatic_aberration(
                plane, red_offset, green_offset, blue_offset, fade_radius, intensity,
                fade_mask=mask,
            ),
        ))

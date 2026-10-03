"""Photographic film grain."""

from __future__ import annotations

import random

from comfy_api.latest import io

from ....modules.convert.tensors import filtered_planes


def apply_film_grain(img, density: float = 0.1, intensity: float = 1.0,
                     highlights: float = 1.0, supersample_factor: int = 4):
    """Blend randomly speckled grain into an image.

    Each call seeds its grain from the global :mod:`random` state.

    Args:
        img: Source PIL image.
        density: Share of the enlarged pixels to speckle, 0.0 to 1.0. The same pixel can be
            drawn more than once, so the share actually covered is a little lower.
        intensity: Blend weight between the source and the grain layer.
        highlights: Brightness multiplier applied to the finished image.
        supersample_factor: How many times larger the grain is generated before being
            scaled down. Cost grows with the square of this.

    Returns:
        A PIL image the same size and mode as the source.
    """
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter

    img_gray = img.convert('L')
    original_size = img.size
    # Speckling at the enlarged size is what makes the grain finer than one pixel once it
    # is scaled back down.
    img_gray = img_gray.resize(
        ((img.size[0] * supersample_factor), (img.size[1] * supersample_factor)),
        Image.Resampling.BILINEAR,
    )
    pixels = np.array(img_gray)
    count = int(density * pixels.size)
    generator = np.random.default_rng(random.getrandbits(64))
    flat = pixels.reshape(-1)
    flat[generator.integers(0, flat.size, count)] = generator.integers(0, 256, count, dtype=np.uint8)
    img_gray = Image.fromarray(pixels)

    img_noise = img_gray.convert('RGB')
    img_noise = img_noise.filter(ImageFilter.GaussianBlur(radius=0.125))
    img_noise = img_noise.resize(original_size, Image.Resampling.LANCZOS)
    img_noise = img_noise.filter(ImageFilter.EDGE_ENHANCE_MORE)
    img_final = Image.blend(img, img_noise, intensity)

    return ImageEnhance.Brightness(img_final).enhance(highlights)


class ImageFilmGrain(io.ComfyNode):
    """Blend randomly speckled photographic grain into an image."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Film Grain",
            display_name="Image Film Grain",
            search_aliases=["Image Film Grain", "film grain", "grain", "noise", "analog", "35mm"],
            category="WAS Suite/Image/Filter",
            description=(
                "Lay photographic grain over an image, the speckle a film negative has. The "
                "grain is random on every run, so the same inputs do not give the same "
                "speckle twice."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The image to add grain to. A batch is handled one image at a time, and "
                        "each image draws its own grain rather than sharing one layer."
                    ),
                ),
                io.Float.Input(
                    "density",
                    default=1.0,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How much of the frame is speckled. 1.0 covers it evenly, 0.5 leaves "
                        "gaps of untouched image between the grains, 0.01 gives sparse specks."
                    ),
                ),
                io.Float.Input(
                    "intensity",
                    default=1.0,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How strongly the grain layer replaces the picture. 0.01 is barely "
                        "there, 0.3 is a realistic amount, 1.0 discards the original colours "
                        "and leaves the grey grain layer alone."
                    ),
                ),
                io.Float.Input(
                    "highlights",
                    default=1.0,
                    min=0.01,
                    max=255.0,
                    step=0.01,
                    tooltip=(
                        "Brightness multiplier applied at the end, to win back the light the "
                        "grain absorbs. 1.0 leaves it alone, 1.2 lifts it slightly, and large "
                        "values blow the image out to white."
                    ),
                ),
                io.Int.Input(
                    "supersample_factor",
                    default=4,
                    min=1,
                    max=8,
                    step=1,
                    tooltip=(
                        "How much larger the grain is drawn before being scaled down, which "
                        "decides how fine it is. 1 gives coarse one-pixel grain, 4 is a fine "
                        "realistic grain, 8 is finer still. Cost grows with the square of this, "
                        "so 8 is 64 times the work of 1."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(tooltip="The image with grain blended into it."),
            ],
        )

    @classmethod
    def execute(cls, image, density, intensity, highlights, supersample_factor) -> io.NodeOutput:
        return io.NodeOutput(filtered_planes(
            image,
            lambda plane: apply_film_grain(
                plane, density, intensity, highlights, supersample_factor
            ),
        ))

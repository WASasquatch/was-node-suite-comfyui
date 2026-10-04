"""Scramble an image into noise that keeps its palette."""

from __future__ import annotations

import random

import numpy as np
from comfy_api.latest import io

from .....modules.convert.tensors import pil2tensor, tensor2pil


class ImageToNoise(io.ComfyNode):
    """Turn an image into four-channel noise built from its own colours."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image to Noise",
            display_name="Image to Noise",
            search_aliases=[
                "Image to Noise",
                "scramble",
                "shuffle pixels",
                "palette noise",
            ],
            category="WAS Suite/Image/Filter",
            description=(
                "Shuffle an image's pixels into noise that keeps its palette. Handy as a "
                "starting texture for image-to-image work, or as grain that matches the "
                "colours of a shot."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The image whose colours the noise is built from. A batch is "
                        "scrambled frame by frame."
                    ),
                ),
                io.Int.Input(
                    "num_colors",
                    default=16,
                    min=2,
                    max=256,
                    step=2,
                    tooltip=(
                        "How many colours the image is reduced to before its pixels are "
                        "shuffled. 2 gives noise in two tones, 16 keeps the broad palette, "
                        "256 keeps almost every shade of the original."
                    ),
                ),
                io.Int.Input(
                    "black_mix",
                    default=0,
                    min=0,
                    max=20,
                    step=1,
                    tooltip=(
                        "How many passes of random black pixels are laid over the noise. "
                        "Each pass blacks out about half of what is left, so 1 is a coarse "
                        "half-and-half speckle and 4 is nearly black. 0 adds none. Every "
                        "pass draws once per pixel, so high values on a large image are slow."
                    ),
                ),
                io.Float.Input(
                    "gaussian_mix",
                    default=0.0,
                    min=0,
                    max=1024,
                    step=0.1,
                    tooltip=(
                        "Radius of a blur mixed back into the noise, in pixels, which softens "
                        "it into clumps instead of single dots. 0.0 skips it; 2.0 gives soft "
                        "grain. Most of the original noise is kept whatever the radius."
                    ),
                ),
                io.Float.Input(
                    "brightness",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    tooltip=(
                        "Brightness of the result. 1.0 leaves it as it is, 0.5 halves it, "
                        "0.0 is black and 2.0 is twice as bright."
                    ),
                ),
                io.Combo.Input(
                    "output_mode",
                    options=["batch", "list"],
                    tooltip=(
                        "`batch` emits one image holding every scrambled frame, which is what "
                        "the image nodes expect. `list` emits the frames as a list on the same "
                        "socket instead, for a downstream node that reads one."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=18446744073709551615,
                    tooltip=(
                        "Which shuffle is used. The same seed and the same input always give "
                        "the same noise; change it to scramble differently. Any whole number; "
                        "`0` is as good a seed as any."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="image",
                    tooltip=(
                        "The scrambled noise, one frame per input frame, with an alpha "
                        "channel taken from the shuffle."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls, images, num_colors, black_mix, gaussian_mix, brightness, output_mode, seed
    ) -> io.NodeOutput:
        from .....modules.image import scratch

        noise_images = (
            pil2tensor(
                _image_to_noise(
                    tensor2pil(image),
                    num_colors=num_colors,
                    black_mix=black_mix,
                    brightness=brightness,
                    gaussian_mix=gaussian_mix,
                    seed=seed,
                )
            )
            for image in images
        )

        if output_mode == "list":
            return io.NodeOutput(list(noise_images))
        batch = None
        for index, frame in enumerate(noise_images):
            if batch is None:
                batch = scratch.allocate(
                    (len(images),) + tuple(frame.shape[1:]),
                    node="Image to Noise",
                    advice="Passing fewer or smaller frames also fits it.",
                )
            batch[index] = frame[0]
        if batch is None:
            raise ValueError(
                "Image to Noise was handed an empty image batch. Wire in at least one image."
            )
        return io.NodeOutput(batch)


def _image_to_noise(
    image,
    num_colors: int,
    black_mix: int,
    brightness: float,
    gaussian_mix: float,
    seed: int,
):
    """Scramble one image into noise.

    Every call reseeds the shared ``random`` module, so each frame of a batch is shuffled
    the same way.

    Args:
        image: Source image.
        num_colors: Palette size the image is quantised to first.
        black_mix: Passes of random black pixels laid over the result.
        brightness: Brightness multiplier, 1.0 leaving the result unchanged.
        gaussian_mix: Blur radius in pixels mixed back in, 0 skipping the blur.
        seed: Seed for the shuffle and the black speckle.

    Returns:
        An ``RGBA`` image the size of ``image``.
    """
    from PIL import Image, ImageEnhance, ImageFilter

    random.seed(int(seed))
    image = image.quantize(colors=num_colors)
    image = image.convert("RGBA")
    pixel_data = list(image.getdata())
    random.shuffle(pixel_data)
    randomized_image = Image.new("RGBA", image.size)
    randomized_image.putdata(pixel_data)

    width, height = image.size
    randomized_image = Image.alpha_composite(
        randomized_image, _black_speckle(width, height, black_mix)
    )

    enhancer = ImageEnhance.Brightness(randomized_image)
    randomized_image = enhancer.enhance(brightness)

    if gaussian_mix > 0:
        original_noise = randomized_image.copy()
        randomized_gaussian = randomized_image.filter(ImageFilter.GaussianBlur(radius=gaussian_mix))
        randomized_image = Image.blend(randomized_image, randomized_gaussian, 0.65)
        randomized_image = Image.blend(randomized_image, original_noise, 0.25)

    return randomized_image


def _black_speckle(width: int, height: int, passes: int):
    """An overlay of opaque black pixels chosen by coin flip.

    Args:
        width: Overlay width in pixels.
        height: Overlay height in pixels.
        passes: How many times to flip for every pixel. 0 gives a fully transparent
            overlay, which composites as a no-op.

    Returns:
        An ``RGBA`` image, black where a flip came up heads and transparent elsewhere.
    """
    from PIL import Image

    opaque = np.zeros((height, width), dtype=bool)
    for _ in range(passes):
        for x in range(width):
            for y in range(height):
                if random.randint(0, 1) == 1:
                    opaque[y, x] = True

    speckle = np.zeros((height, width, 4), dtype=np.uint8)
    speckle[..., 3] = np.where(opaque, 255, 0)
    return Image.fromarray(speckle)

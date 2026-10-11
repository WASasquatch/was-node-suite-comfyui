"""Render spectrally shaped random noise as an image."""

from __future__ import annotations

import numpy as np
from comfy_api.latest import io

from .....modules.convert.tensors import pil2tensor
from .....modules.log import get_logger

logger = get_logger("image.power_noise")

#: Largest value ``numpy.random.seed`` accepts is ``2 ** 32 - 1``. A seed above this is
#: folded into 0-4294967293 rather than rejected, so the 64-bit seed widget every other
#: node carries stays usable here.
MAX_SEED = 4294967294

#: Noise types built by shaping a grey noise spectrum. ``pink`` multiplies the spectrum by
#: the radial frequency, ``blue`` divides by it and ``green`` divides by its square root,
#: which respectively bias the result towards coarse blobs, fine grain, and a mid-frequency
#: band between the two.
SHAPED = ("pink", "blue", "green")

#: What the ``mix`` type blends, in order. Only as many entries as there are masks are
#: reached, and there are three masks, so ``green`` and ``blue`` do not contribute.
MIX_TYPES = ("white", "grey", "pink", "green", "blue")

#: How many blue noise masks ``mix`` blends through.
MIX_MASKS = 3


class ImagePowerNoise(io.ComfyNode):
    """Generate noise with a chosen frequency distribution."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Power Noise",
            display_name="Image Power Noise",
            search_aliases=[
                "Image Power Noise",
                "power noise",
                "white noise",
                "blue noise",
                "pink noise",
                "grain",
            ],
            category="WAS Suite/Image/Generate",
            description=(
                "Generate a noise image with a chosen grain size: flat white noise, soft "
                "pink noise, fine blue noise, or a blend of several. Useful as film grain, "
                "as a dither pattern, or as the starting texture for an image-to-image pass."
            ),
            inputs=[
                io.Int.Input(
                    "width",
                    default=512,
                    min=64,
                    max=4096,
                    step=1,
                    tooltip="Width of the generated image, in pixels.",
                ),
                io.Int.Input(
                    "height",
                    default=512,
                    min=64,
                    max=4096,
                    step=1,
                    tooltip="Height of the generated image, in pixels.",
                ),
                io.Float.Input(
                    "frequency",
                    default=0.5,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    tooltip=(
                        "The frequency shaping is worked out from the image size alone, so "
                        "changing this value does not change the image; use noise_type to "
                        "choose the grain size."
                    ),
                ),
                io.Float.Input(
                    "attenuation",
                    default=0.5,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    tooltip=(
                        "Spread of the random draw behind `grey`, `pink`, `blue`, `green` and "
                        "`mix`. The result is stretched to black-to-white, so this changes the "
                        "grain's character, not its brightness. 0.0 = a blank image; `white` "
                        "ignores it."
                    ),
                ),
                io.Combo.Input(
                    "noise_type",
                    options=["grey", "white", "pink", "blue", "green", "mix"],
                    tooltip=(
                        "`white` = uniform per-pixel static; `grey` = bell-curved per-pixel "
                        "static; `pink` = large soft blobs; `blue` = fine grain that dithers "
                        "without clumps; `green` = between pink and blue; `mix` = white, grey "
                        "and pink through blue noise masks, cloudier and patchier."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=18446744073709551615,
                    tooltip=(
                        "Which draw is used. The same seed always gives the same image, so "
                        "change it for a different one. Values above 4294967294 are folded "
                        "down into range and the log says what was used instead."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="image",
                    tooltip="The noise, as a greyscale image with all three channels equal.",
                ),
            ],
        )

    @classmethod
    def execute(cls, width, height, frequency, attenuation, noise_type, seed) -> io.NodeOutput:
        return io.NodeOutput(
            pil2tensor(_power_noise(width, height, frequency, attenuation, noise_type, seed))
        )


def _power_noise(
    width: int,
    height: int,
    frequency: float,
    attenuation: float,
    noise_type: str,
    seed: int,
):
    """Draw one noise field and render it as an image.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        frequency: Accepted and unused. The spectral shaping is derived from the image
            dimensions, so no noise type reads this value.
        attenuation: Standard deviation of the normal draw behind every type but ``white``.
        noise_type: One of ``grey``, ``white``, ``pink``, ``blue``, ``green``, ``mix``.
        seed: Seed for the draw. Folded into the range ``numpy.random.seed`` accepts.

    Returns:
        An ``RGB`` image of ``(width, height)``.

    Raises:
        ValueError: ``noise_type`` is not one of the six.
    """
    if seed > MAX_SEED:
        seed = _shorten_to_range(seed, 0, MAX_SEED - 1)
        logger.warning("Seed too large for power noise; rescaled to: %s", seed)

    np.random.seed(seed)

    if noise_type == "white":
        noise = _white_noise(width, height)
    elif noise_type == "grey":
        noise = _grey_noise(width, height, attenuation)
    elif noise_type in SHAPED:
        noise = _shaped_noise(width, height, attenuation, noise_type)
    elif noise_type == "mix":
        masks = [
            _seeded_mask(width, height, attenuation, seed + index)
            for index in range(MIX_MASKS)
        ]
        noise = _blend_noise(width, height, masks, MIX_TYPES, attenuation)
    else:
        raise ValueError(f"Unsupported noise type `{noise_type}`")

    if noise_type != "mix":
        noise = _stretched(noise)

    return _grey_image(noise).convert("RGB")


def _shorten_to_range(value: int, min_value: int, max_value: int) -> int:
    """Wrap ``value`` into ``[min_value, max_value]``.

    Args:
        value: Value to fold.
        min_value: Lowest value of the range.
        max_value: Highest value of the range, included.

    Returns:
        ``value`` reduced modulo the length of the range and offset back onto it.
    """
    range_length = max_value - min_value + 1
    return ((value - min_value) % range_length) + min_value


def _white_noise(width: int, height: int):
    """A uniform draw over ``[0, 1)``, shaped ``(height, width)``."""
    return np.random.random((height, width))


def _grey_noise(width: int, height: int, attenuation: float):
    """A zero-mean normal draw of standard deviation ``attenuation``, as ``(height, width)``."""
    return np.random.normal(0, attenuation, (height, width))


def _shaped_noise(width: int, height: int, attenuation: float, noise_type: str):
    """Grey noise reweighted by radial frequency.

    Args:
        width: Field width.
        height: Field height.
        attenuation: Standard deviation of the underlying grey draw.
        noise_type: ``pink`` to multiply by the frequency, ``blue`` to divide by it, or
            ``green`` to divide by its square root.

    Returns:
        The real part of the reweighted field, as ``(height, width)``.
    """
    noise = _grey_noise(width, height, attenuation)
    scale = 1.0 / (width * height)

    fy = np.fft.fftfreq(height)[:, np.newaxis] ** 2
    fx = np.fft.fftfreq(width) ** 2
    power = np.sqrt(fy + fx)
    power[0, 0] = 1

    spectrum = np.fft.fft2(noise)
    if noise_type == "pink":
        spectrum = spectrum * power
    elif noise_type == "blue":
        spectrum = spectrum / power
    else:
        spectrum = spectrum / np.sqrt(power)

    noise = np.fft.ifft2(spectrum)
    noise *= scale / noise.std()
    return np.real(noise)


def _seeded_mask(width: int, height: int, attenuation: float, seed: int):
    """One blue noise field drawn from its own seed, for the ``mix`` blend.

    Args:
        width: Field width.
        height: Field height.
        attenuation: Standard deviation of the underlying grey draw.
        seed: Seed applied to the global numpy generator before the draw.

    Returns:
        A blue noise field as ``(height, width)``.
    """
    np.random.seed(seed)
    return _shaped_noise(width, height, attenuation, "blue")


def _blend_noise(width: int, height: int, masks, noise_types, attenuation: float):
    """Composite one noise type per mask into a single field.

    Args:
        width: Field width.
        height: Field height.
        masks: Fields used as blend factors, brightest where the new noise shows through.
        noise_types: Type to draw for each mask, in the same order.
        attenuation: Standard deviation passed to each draw that takes one.

    Returns:
        The composited image as a uint8 ``(height, width)`` array.
    """
    from PIL import Image

    blended = Image.new("L", (width, height), color=0)
    for mask, noise_type in zip(masks, noise_types):
        if noise_type == "white":
            noise = _white_noise(width, height)
        elif noise_type == "grey":
            noise = _grey_noise(width, height, attenuation)
        else:
            noise = _shaped_noise(width, height, attenuation, noise_type)
        blended = Image.composite(
            blended, _grey_image(_stretched(noise)), _grey_image(_stretched(mask))
        )

    return np.asarray(blended)


def _stretched(noise):
    """Scale a field so its lowest value sits at 0 and its highest at 255.

    Args:
        noise: Any real-valued field.

    Returns:
        The scaled field, as floats. A field with no range to stretch gives NaN, which
        casts to 0.
    """
    return 255 * (noise - np.min(noise)) / (np.max(noise) - np.min(noise))


def _grey_image(noise):
    """Wrap a field already scaled to 0-255 as a greyscale image.

    Args:
        noise: Field shaped ``(height, width)``, truncated to 8 bits as it is.

    Returns:
        A PIL image in mode ``L``.
    """
    from PIL import Image

    return Image.fromarray(noise.astype(np.uint8))

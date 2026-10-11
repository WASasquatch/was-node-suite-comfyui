"""Sine-wave remapping of every channel value."""

from __future__ import annotations

import numpy as np
from comfy_api.latest import io

from ....modules.convert.tensors import filtered_planes
from ....modules.interface import preview


def sine(x, freq: float, amp: float):
    """Sample a sine wave of ``freq`` cycles per unit, scaled to ``amp``.

    Args:
        x: Value or array in ``[0, 1]``.
        freq: Cycles per unit of ``x``.
        amp: Peak height of the wave.

    Returns:
        Values in ``[-amp, amp]``.
    """
    return amp * np.sin(2 * np.pi * freq * x)


def nova_sine(image, amplitude: float, frequency: float):
    """Replace every channel value with a sine wave sampled at that value.

    Args:
        image: Source PIL image.
        amplitude: Peak height of the wave, 0.0 to 1.0, where 1.0 is the full
            black-to-white range.
        frequency: Cycles of the wave across the 0-1 brightness range. Clamped to half the
            image width, past which the bands are finer than the pixels that would show
            them.

    Returns:
        A uint8 array shaped like the source image's own array.
    """
    img_array = np.array(image)

    max_freq = image.width / 2
    if frequency > max_freq:
        frequency = max_freq

    scaled = sine(img_array / 255, frequency, amplitude) * 255

    # Truncated towards zero and wrapped into 0-255 the way a store into a uint8 array
    # wraps a negative value, which is where the banding comes from.
    return (scaled.astype(np.int64) % 256).astype(np.uint8)


class ImageNovaFilter(io.ComfyNode):
    """Remap an image's tones through a sine wave, producing hard psychedelic bands."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Nova Filter",
            display_name="Image Nova Filter",
            search_aliases=[
                "Image Nova Filter",
                "nova",
                "solarize",
                "posterize",
                "psychedelic",
                "sine",
            ],
            category="WAS Suite/Image/Filter",
            description=(
                "Push an image's brightness through a sine wave, which turns smooth "
                "gradients into hard bands of colour. A solarising effect rather than a "
                "photographic one; the shapes stay recognisable but the tones do not."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="The image to remap. A batch is handled one image at a time.",
                ),
                io.Float.Input(
                    "amplitude",
                    default=0.1,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    tooltip=(
                        "How far the wave swings, where 1.0 is the whole black-to-white range. "
                        "0.0 gives a black image, 0.1 keeps the result dark with faint bands, "
                        "1.0 gives full-strength colour bands."
                    ),
                ),
                io.Float.Input(
                    "frequency",
                    default=3.14,
                    min=0.0,
                    max=100.0,
                    step=0.001,
                    tooltip=(
                        "How many bands the wave lays across the brightness range. 0.0 = a "
                        "black image; 1.0 = one broad sweep; 3.14 = a handful of bands; larger "
                        "= many fine ones. Capped at half the image width."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(tooltip="The remapped image, the same size as the source."),
            ],
        )

    @classmethod
    def execute(cls, image, amplitude, frequency) -> io.NodeOutput:
        # The wave is sampled at each pixel's own brightness, so the image that arrives here
        # is what an interface previews the curve over, and its width is what caps the
        # frequency. Publishing it changes nothing this returns, and does nothing at all
        # while no browser is connected.
        preview.publish(image)
        return io.NodeOutput(filtered_planes(
            image, lambda plane: nova_sine(plane, amplitude, frequency)
        ))

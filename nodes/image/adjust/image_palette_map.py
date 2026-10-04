"""Repaint an image in the colours of a palette."""

from __future__ import annotations

import numpy as np
import torch
from comfy_api.latest import io

from ....modules.compat.types import LIST
from ....modules.convert.tensors import image_planes
from ....modules.image import dynamic, palette_map

#: The palette a fresh node arrives with: a five-stop dark-teal to warm-cream ramp, which
#: shows what `Luminance Ramp` does to a greyscale plate without anything being typed.
DEFAULT_PALETTE = "#10141f\n#2a4a5e\n#6f8f8c\n#c9b48a\n#f4ead8"


class ImagePaletteMap(io.ComfyNode):
    """Repaint an image in a palette's colours, by nearest match or as a gradient."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImagePaletteMap",
            display_name="Image Palette Map",
            search_aliases=[
                "WASImagePaletteMap", "Image Palette Map",
                "colorize",
                "colourise",
                "gradient map",
                "duotone",
                "quantize",
                "palette",
                "posterize",
                "false colour",
            ],
            category="WAS Suite/Image/Adjustment",
            description=(
                (
                    (
                        "Repaint an image in a palette's colours, either by matching each "
                        "pixel to its closest colour or by mapping brightness along the "
                        "palette as a gradient. Matching is measured in Oklab and can be "
                        "dithered. `Perceptual` keeps the picture's own hues as closely as the "
                        "palette allows; `Luminance Ramp` throws the original colour away, "
                        "which colourises a greyscale plate or grades a depth map, and reads "
                        "the palette sorted dark to light. Several colours can share a palette "
                        "line separated by commas, a comma inside brackets belongs to its "
                        "colour, PIL's other spellings such as 'hsl(30, 100%, 50%)' read, and "
                        "a line that is not a colour is skipped. On dither, `none` leaves flat "
                        "bands, `FloydSteinberg` flickers on a sequence where `Bayer` stays "
                        "put, and `Luminance Ramp` reads neither. Leave normalize off for "
                        "anything animated, or an exposure change between frames shifts the "
                        "colours."
                    )
                )
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The images to repaint. Each one is handled on its own, so a "
                        "sequence keeps a consistent look."
                    ),
                ),
                io.String.Input(
                    "palette",
                    multiline=True,
                    default=DEFAULT_PALETTE,
                    tooltip=(
                        "The palette, one colour per line: '#ff8800', '#f80', 'orange', "
                        "'rgb(255, 136, 0)'. Ignored when color_palettes is connected."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=list(palette_map.MODES),
                    tooltip=(
                        "`Perceptual` snaps every pixel to its nearest palette colour, for a "
                        "fixed set of inks. `Luminance Ramp` places pixels along the palette "
                        "by brightness."
                    ),
                ),
                io.Combo.Input(
                    "dither",
                    options=list(palette_map.DITHERS),
                    tooltip=(
                        "How the error left by matching is spread, in `Perceptual` mode. "
                        "`FloydSteinberg` gives a fine organic stipple; `Bayer` a fixed 8x8 "
                        "crosshatch."
                    ),
                ),
                io.Boolean.Input(
                    "smooth",
                    default=True,
                    tooltip=(
                        "Whether `Luminance Ramp` blends between neighbouring palette "
                        "colours. On, the palette becomes a continuous gradient and shading "
                        "survives. Off, every pixel snaps to one of the palette's colours, "
                        "which posterises the picture into exactly that many bands, the "
                        "poster or screen-print look. Not read by `Perceptual`."
                    ),
                ),
                io.Boolean.Input(
                    "normalize",
                    default=False,
                    tooltip=(
                        "Whether `Luminance Ramp` stretches the palette across each image's "
                        "darkest and lightest values. Off by default, which reads brightness "
                        "absolutely and keeps a sequence stable."
                    ),
                ),
                io.Boolean.Input(
                    "reverse",
                    default=False,
                    tooltip=(
                        "Whether the palette is read in the opposite direction. This flips a "
                        "`Luminance Ramp` end to end, so a dark-to-light palette renders the "
                        "picture as a negative. It changes nothing visible in `Perceptual` "
                        "mode, where a colour is chosen by distance rather than by position."
                    ),
                ),
                io.Float.Input(
                    "blend",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How much of the repainted image replaces the original. 1.0 is the "
                        "palette alone. Lower values let the original show through, which is "
                        "how a colourised plate is tied back to its own hues, around 0.6 to "
                        "0.8 grades an image without it looking like a filter."
                    ),
                ),
                LIST.Input(
                    "color_palettes",
                    optional=True,
                    extra_dict={"forceInput": True},
                    tooltip=(
                        "One palette per image, as Image Color Palette emits. Connected, it "
                        "replaces the text box. Fewer palettes than images is fine: the list "
                        "repeats, so one palette covers a whole batch."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="The repainted images, at their original size.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        images,
        palette=DEFAULT_PALETTE,
        mode="Perceptual",
        dither="none",
        smooth=True,
        normalize=False,
        reverse=False,
        blend=1.0,
        color_palettes=None,
    ) -> io.NodeOutput:
        """Repaint every image of the batch.

        Raises:
            ValueError: The batch holds no images.
            MemoryError: Neither free memory nor a scratch drive can hold the result.
        """
        from ....modules.image import scratch

        sources = cls.palettes(palette, color_palettes)

        scale = dynamic.peak(images) if dynamic.carries(images) else 1.0
        planes = image_planes(images)
        repainted = None
        for index, plane in enumerate(planes):
            folded = dynamic.fold(plane, scale)
            colors = palette_map.parse_palette(sources[index % len(sources)])
            pixels = folded.images[..., :3].detach().cpu().numpy().astype(np.float32)
            mapped = palette_map.apply_palette(
                pixels,
                colors,
                mode=mode,
                dither=dither,
                smooth=smooth,
                reverse=reverse,
                blend=blend,
                normalize=normalize,
            )
            frame = torch.from_numpy(mapped)
            if repainted is None:
                repainted = scratch.allocate(
                    (len(planes),) + tuple(frame.shape),
                    frame.dtype,
                    node="Image Palette Map",
                    advice="Passing fewer frames also fits it.",
                )
            repainted[index] = frame
            if folded.scale != 1.0:
                repainted[index].mul_(folded.scale)

        if repainted is None:
            raise ValueError(
                "Image Palette Map was given no images. Connect an image or a batch of them."
            )
        return io.NodeOutput(repainted)

    @staticmethod
    def palettes(typed: str, connected) -> list:
        """The palette source for each image, in batch order.

        Args:
            typed: The text box, one colour per line.
            connected: The ``color_palettes`` socket, or ``None``.

        Returns:
            A list of palette sources, each acceptable to
            :func:`modules.image.palette_map.parse_palette`. Holds one entry when the text
            box is the source.
        """
        if not connected:
            return [typed]
        entries = list(connected) if isinstance(connected, (list, tuple)) else [connected]
        return entries or [typed]

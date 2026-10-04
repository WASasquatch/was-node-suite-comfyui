"""Reduce images to low-resolution, few-colour pixel art."""

from __future__ import annotations

import math

import torch
from PIL import Image
from comfy_api.latest import io

from ....modules.image import dynamic
from ....modules import log
from ....modules.compat.types import LIST
from ....modules.convert.tensors import pil2tensor, tensor2pil
from ....modules.image import palette_map
from ....modules.image.palette import compute_device, kmeans, pixels_of

logger = log.get_logger("nodes.image.process")

#: Refinement passes the colour clustering is allowed when nothing else says.
KMEANS_MAX_ITER = 100

#: Seed for the clustering, so the same image always reduces to the same colours.
KMEANS_RANDOM_STATE = 42

#: Luminance weights for the tonal palette comparisons, applied in this order.
LUMA = (0.299, 0.587, 0.114)

#: Grey the tonal sort measures a palette entry's distance from.
MID_GREY = (128, 128, 128)

#: Error-diffusion neighbours of the quantising dithers, as ``(row, column, sixteenths)``
#: in the diagonal layout the scan walks. Applied in this order, and each one is read back
#: from the values the earlier ones wrote.
DIFFUSION = ((1, 1, 3), (1, 0, 7), (2, 1, 5), (3, 1, 1))

#: Entries of the pixel-to-palette distance table held at once, which caps its memory.
DISTANCE_BUDGET = 1 << 22


def to_image(rows, size):
    """Build an ``RGB`` image from rows of colour.

    Args:
        rows: ``(width * height, 3)`` uint8 tensor in row-major order.
        size: ``(width, height)`` of the image the rows fill.

    Returns:
        An ``RGB`` PIL image.
    """
    return Image.frombytes("RGB", size, bytes(rows.contiguous().cpu().flatten().tolist()))


def hex_to_rgb(value: str) -> tuple[int, ...]:
    """Parse one ``#rrggbb`` colour.

    Args:
        value: Six hex digits, with or without a leading ``#``. Anything after the sixth
            digit is ignored.

    Returns:
        ``(red, green, blue)``, each 0-255.

    Raises:
        ValueError: The string does not hold three pairs of hex digits, which is what an
            empty line in a palette produces.
    """
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def channel_distance(color1, color2) -> int:
    """Total absolute difference across the three channels of two colours."""
    return sum(abs(c1 - c2) for c1, c2 in zip(color1, color2))


def luminance(rgb):
    """Perceived brightness of a tensor of colours.

    Args:
        rgb: Tensor whose last axis is ``(red, green, blue)``.

    Returns:
        A float tensor of the remaining axes, on the same 0-255 scale as the input.
    """
    return LUMA[0] * rgb[..., 0] + LUMA[1] * rgb[..., 1] + LUMA[2] * rgb[..., 2]


def flatten_colors(image, num_colors, init_mode="random", max_iter=KMEANS_MAX_ITER):
    """Reduce an image to a handful of colours by clustering its pixels.

    Args:
        image: Source image. Must be ``RGB``: the pixel buffer is read as three channels.
        num_colors: Number of clusters, which is the number of colours in the result.
        init_mode: How the cluster centres start, ``"k-means++"`` or ``"random"``.
        max_iter: Refinement passes the clustering is allowed.

    Returns:
        An ``RGB`` image the same size, holding at most ``num_colors`` distinct colours.

    Raises:
        ValueError: ``init_mode`` names no initialiser, or the image has fewer pixels than
            ``num_colors``, which leaves the clustering with fewer samples than clusters.
    """
    centroids, labels, _ = kmeans(
        pixels_of(image),
        num_colors,
        init=init_mode,
        max_iter=max_iter,
        tol=1e-3,
        seed=KMEANS_RANDOM_STATE,
    )
    colors = centroids.clamp(0, 255).to(torch.uint8)
    return to_image(colors[labels], image.size)


def diagonal_layout(plane):
    """Lay an image out with one error-diffusion wavefront per row.

    Args:
        plane: ``(height, width, channels)`` tensor.

    Returns:
        A tensor holding pixel ``(row, column)`` at ``(2 * row + column, row)`` and zero
        elsewhere, with room past the last wavefront for the diffusion to write into.
    """
    height, width, channels = plane.shape
    rows = torch.arange(height, device=plane.device)[:, None].expand(height, width)
    columns = torch.arange(width, device=plane.device)[None, :].expand(height, width)
    skewed = torch.zeros(
        (2 * height + width + 2, height + 1, channels),
        dtype=plane.dtype,
        device=plane.device,
    )
    skewed[(2 * rows + columns).reshape(-1), rows.reshape(-1)] = plane.reshape(-1, channels)
    return skewed


def straight_layout(skewed, height, width):
    """Read an image back out of the diagonal layout.

    Args:
        skewed: Tensor laid out by :func:`diagonal_layout`.
        height: Rows of the image.
        width: Columns of the image.

    Returns:
        A ``(height, width, channels)`` tensor.
    """
    rows = torch.arange(height, device=skewed.device)[:, None].expand(height, width)
    columns = torch.arange(width, device=skewed.device)[None, :].expand(height, width)
    flat = skewed[(2 * rows + columns).reshape(-1), rows.reshape(-1)]
    return flat.reshape(height, width, skewed.shape[2])


def diagonal_span(index, height, width):
    """Image rows one wavefront covers.

    Args:
        index: Wavefront number, which is ``2 * row + column`` for every pixel on it.
        height: Rows of the image.
        width: Columns of the image.

    Returns:
        ``(start, stop)``, a half-open range of image rows, which is also the range of
        columns the wavefront occupies in the diagonal layout. Empty when ``start`` is not
        below ``stop``.
    """
    start = max(0, -((width - 1 - index) // 2))
    stop = min(height - 1, index // 2) + 1
    return start, stop


def diffuse_error(plane, steps):
    """Quantise each channel to ``steps`` values, spreading the error over later pixels.

    Args:
        plane: ``(height, width, channels)`` float tensor scaled to 0.0-1.0.
        steps: Number of values a channel is allowed, at least 2.

    Returns:
        A tensor the same shape and dtype, holding quantised values.
    """
    height, width = plane.shape[0], plane.shape[1]
    scale = steps - 1
    skewed = diagonal_layout(plane)
    for index in range(2 * (height - 1) + width):
        start, stop = diagonal_span(index, height, width)
        if start >= stop:
            continue
        old = skewed[index, start:stop]
        new = torch.round(old * scale) / scale
        error = old - new
        skewed[index, start:stop] = new
        for row, column, sixteenths in DIFFUSION:
            skewed[index + row, start + column:stop + column].add_(error, alpha=sixteenths / 16)
    return straight_layout(skewed, height, width)


def quantise_levels(values, levels, step):
    """Round colour values down onto a ladder of evenly spaced levels.

    Args:
        values: Tensor of 0-255 colour values.
        levels: Number of levels the range is cut into.
        step: Distance between two levels.

    Returns:
        A tensor of the same shape and dtype.
    """
    return (values * levels / 256).trunc() * step


def diffuse_levels(plane, levels, step):
    """Snap each channel onto a ladder of levels, spreading the error over later pixels.

    Args:
        plane: ``(height, width, channels)`` float tensor of 0-255 colour values.
        levels: Number of levels the range is cut into.
        step: Distance between two levels.

    Returns:
        A ``(height, width, channels)`` uint8 tensor.
    """
    height, width = plane.shape[0], plane.shape[1]
    skewed = diagonal_layout(plane)
    picked = torch.zeros(skewed.shape, dtype=torch.uint8, device=plane.device)
    for index in range(2 * (height - 1) + width):
        start, stop = diagonal_span(index, height, width)
        if start >= stop:
            continue
        chosen = quantise_levels(skewed[index, start:stop], levels, step)
        picked[index, start:stop] = chosen.to(torch.uint8)
        for row, column, sixteenths in DIFFUSION:
            target = skewed[index + row, start + column:stop + column]
            neighbour = quantise_levels(target, levels, step)
            spread = neighbour + (neighbour - chosen) * sixteenths / 16
            target.copy_(spread.clamp(0, 255).trunc())
    return straight_layout(picked, height, width)


def floyd_steinberg_dither(img, nc):
    """Dither an image down to ``nc`` levels per channel, diffusing the error forwards.

    Args:
        img: Source image. Must be ``RGB``.
        nc: Number of levels per channel, at least 2.

    Returns:
        An ``RGB`` image the same size.

    Raises:
        ValueError: ``img`` is not in mode ``RGB``.
    """
    width, height = img.size
    plane = pixels_of(img).reshape(height, width, 3).to(torch.float64) / 255
    quantised = diffuse_error(plane, nc) * 255
    return to_image(quantised.to(torch.uint8).reshape(-1, 3), img.size)


def ordered_dither(img, nc):
    """Quantise an image to a power-of-two number of levels and diffuse the error.

    Args:
        img: Source image. Must be ``RGB``.
        nc: Requested number of colours, rounded down to a power of two.

    Returns:
        A new ``RGB`` image holding the quantised result.

    Raises:
        ValueError: ``img`` is not in mode ``RGB``.
    """
    width, height = img.size
    # Rounded down and capped, so 16 colours and 31 colours both give 16 levels.
    levels = min(2 ** int(math.log2(nc)), 16)
    plane = pixels_of(img).reshape(height, width, 3).to(torch.float64)
    dithered = diffuse_levels(plane, levels, 256 // levels)
    return to_image(dithered.reshape(-1, 3), img.size)


def dither_image(image, mode, nc):
    """Dither an image by the named method.

    Args:
        image: Source image. Must be ``RGB``.
        mode: ``"FloydSteinberg"`` or ``"Ordered"``.
        nc: Number of levels the method quantises to.

    Returns:
        The dithered image, or the input unchanged when ``mode`` names no method.

    Raises:
        ValueError: ``image`` is not in mode ``RGB``.
    """
    if mode == "FloydSteinberg":
        return floyd_steinberg_dither(image, nc)
    if mode == "Ordered":
        return ordered_dither(image, nc)
    logger.error("invalid dithering mode `%s` selected.", mode)
    return image


def first_minimum(distances):
    """Column index of the smallest value in each row, ties going to the earliest column.

    Args:
        distances: ``(n, m)`` tensor.

    Returns:
        An ``(n,)`` int64 tensor of column indices.
    """
    columns = distances.shape[1]
    positions = torch.arange(columns, device=distances.device)
    lowest = distances.amin(dim=1, keepdim=True)
    return torch.where(distances == lowest, positions, columns).amin(dim=1)


def nearest_palette_index(pixels, entries, mode):
    """Index of the palette entry closest to each pixel.

    Args:
        pixels: ``(n, 3)`` integer tensor of colours.
        entries: ``(m, 3)`` integer tensor of palette colours, in search order.
        mode: ``"Tonal"`` compares brightness only, ``"BrightnessAndTonal"`` adds the
            brightness difference to the channel difference, and anything else compares
            channels only.

    Returns:
        An ``(n,)`` int64 tensor of indices into ``entries``. Ties go to the earliest entry.
    """
    if mode == "Tonal":
        tones = luminance(pixels.double())[:, None] - luminance(entries.double())[None, :]
        return first_minimum(tones.abs_())
    distances = (pixels[:, None, :] - entries[None, :, :]).abs().sum(dim=-1)
    if mode == "BrightnessAndTonal":
        tones = luminance(pixels.double())[:, None] - luminance(entries.double())[None, :]
        distances = tones.abs_() + distances
    return first_minimum(distances)


def map_to_palette(image, colors, palette_mode="Linear", reverse_palette=False):
    """Repaint an image using only the colours of a given palette.

    Args:
        image: Source image. Must be ``RGB``.
        colors: Palette as ``#rrggbb`` strings, one per entry.
        palette_mode: ``"Linear"`` indexes the palette by the pixel's own red value, which
            ignores colour entirely and produces a banded remap. ``"Brightness"`` picks the
            entry with the smallest total channel difference, ``"Tonal"`` the entry closest
            in brightness, and ``"BrightnessAndTonal"`` weighs both.
        reverse_palette: Reverse the palette before ordering it.

    Returns:
        An ``RGB`` image the same size, holding only palette colours.

    Raises:
        ValueError: ``palette_mode`` names no ordering, ``colors`` is empty, or an entry of
            ``colors`` is not a hex colour.
    """
    color_palette = [hex_to_rgb(color) for color in colors]
    if not color_palette:
        raise ValueError(
            "a palette needs at least one '#rrggbb' colour, and this one holds none. "
            "Connect Image Color Palette, or leave color_palettes unconnected."
        )

    if reverse_palette:
        color_palette = color_palette[::-1]

    if palette_mode == "Linear":
        order = list(range(len(color_palette)))
    elif palette_mode == "Brightness":
        order = sorted(range(len(color_palette)), key=lambda i: sum(color_palette[i]) / 3)
    elif palette_mode == "Tonal":
        order = sorted(
            range(len(color_palette)),
            key=lambda i: channel_distance(color_palette[i], MID_GREY),
        )
    elif palette_mode == "BrightnessAndTonal":
        order = sorted(
            range(len(color_palette)),
            key=lambda i: (
                sum(color_palette[i]) / 3,
                channel_distance(color_palette[i], MID_GREY),
            ),
        )
    else:
        raise ValueError(f"Unsupported mapping mode: {palette_mode}")

    device = compute_device()
    ordered = torch.tensor([color_palette[i] for i in order], dtype=torch.int32, device=device)
    pixels = pixels_of(image).to(device=device, dtype=torch.int32)

    if palette_mode == "Linear":
        chosen = pixels[:, 0] % len(color_palette)
    else:
        # A block of pixels at a time, so the distance table stays inside its budget.
        chosen = torch.empty(pixels.shape[0], dtype=torch.int64, device=device)
        chunk = max(1, DISTANCE_BUDGET // len(color_palette))
        for start in range(0, pixels.shape[0], chunk):
            stop = start + chunk
            chosen[start:stop] = nearest_palette_index(pixels[start:stop], ordered, palette_mode)

    return to_image(ordered[chosen].to(torch.uint8), image.size)


def perceptual_map(image, colors, palette_mode, reverse_palette, dither, smooth, blend, normalize):
    """Repaint an image through :mod:`modules.image.palette_map`.

    Args:
        image: Source image. Must be ``RGB``.
        colors: Palette as colour strings, one per entry.
        palette_mode: ``Perceptual`` or ``Luminance Ramp``.
        reverse_palette: Read the palette in the opposite direction.
        dither: ``none``, ``FloydSteinberg`` or ``Bayer``, spread against the palette
            itself rather than before it.
        smooth: Interpolate along the ramp, for ``Luminance Ramp``.
        blend: How much of the result replaces the original, 0.0 to 1.0.
        normalize: Fit the ramp to the image's own range rather than reading lightness
            absolutely, for ``Luminance Ramp``.

    Returns:
        An ``RGB`` image the same size.

    Raises:
        ValueError: Nothing in ``colors`` was a colour, or ``image`` is not in mode ``RGB``.
    """
    width, height = image.size
    pixels = pixels_of(image).reshape(height, width, 3).to(torch.float32) / 255.0
    mapped = palette_map.apply_palette(
        pixels.numpy(),
        palette_map.parse_palette(colors),
        mode=palette_mode,
        dither=dither,
        smooth=smooth,
        reverse=reverse_palette,
        blend=blend,
        normalize=normalize,
    )
    rows = (torch.from_numpy(mapped) * 255.0).round().to(torch.uint8)
    return to_image(rows.reshape(-1, 3), image.size)


def pixel_art_batch(
    batch,
    min_size,
    num_colors=16,
    init_mode="random",
    max_iter=KMEANS_MAX_ITER,
    palette=None,
    palette_mode="Linear",
    reverse_palette=False,
    dither=False,
    dither_mode="FloydSteinberg",
    palette_dither="none",
    palette_smooth=True,
    palette_blend=1.0,
    palette_normalize=False,
):
    """Turn a batch of images into pixel art.

    Args:
        batch: Batch of image tensors.
        min_size: Longer side of the working image in pixels. An image already smaller than
            this is left alone.
        num_colors: Number of colours to reduce to.
        init_mode: Clustering start, or ``"none"`` to skip the colour reduction.
        max_iter: Refinement passes the clustering is allowed.
        palette: One palette per image, each a list of ``#rrggbb`` strings, or None to keep
            the clustered colours. Fewer palettes than images raises.
        palette_mode: How a pixel is matched to a palette entry. The two names in
            :data:`modules.image.palette_map.MODES` route to the perceptual mapper; the
            four original names keep the channel-difference matching they always had.
        reverse_palette: Reverse each palette before matching.
        dither: Dither after the colour reduction and before any palette mapping.
        dither_mode: Which dither to run.
        palette_dither: Dither spread against the palette itself, for the perceptual modes.
        palette_smooth: Interpolate along the ramp, for ``Luminance Ramp``.
        palette_blend: How much of the palette result replaces the colours under it.
        palette_normalize: Fit the ramp to each image's own range, for ``Luminance Ramp``.

    Returns:
        A batch tensor of the results, each at its original size.

    Raises:
        ValueError: The colour reduction runs on an image with fewer pixels than
            ``num_colors``, or the batch holds no image.
        IndexError: ``palette`` holds fewer palettes than the batch holds images.
        MemoryError: Neither free memory nor a scratch drive can hold the result.
    """
    from ....modules.image import scratch

    pixel_art_images = []
    original_sizes = []
    for plane in batch:
        image = tensor2pil(plane).convert("RGB")
        width, height = image.size
        original_sizes.append((width, height))
        if max(width, height) > min_size:
            if width > height:
                new_width = min_size
                new_height = int(height * (min_size / width))
            else:
                new_height = min_size
                new_width = int(width * (min_size / height))
            pixel_art_images.append(image.resize((new_width, int(new_height)), Image.NEAREST))
        else:
            pixel_art_images.append(image)

    if init_mode != "none":
        pixel_art_images = [
            flatten_colors(image, num_colors, init_mode, max_iter)
            for image in pixel_art_images
        ]
    if dither:
        pixel_art_images = [
            dither_image(image, dither_mode, num_colors) for image in pixel_art_images
        ]
    if palette and palette_mode in palette_map.MODES:
        pixel_art_images = [
            perceptual_map(
                image,
                palette[i],
                palette_mode,
                reverse_palette,
                palette_dither,
                palette_smooth,
                palette_blend,
                palette_normalize,
            )
            for i, image in enumerate(pixel_art_images)
        ]
    elif palette:
        pixel_art_images = [
            map_to_palette(image, palette[i], palette_mode, reverse_palette)
            for i, image in enumerate(pixel_art_images)
        ]
    pixels = None
    for index, (image, size) in enumerate(zip(pixel_art_images, original_sizes)):
        frame = pil2tensor(image.resize(size, Image.NEAREST))
        if pixels is None:
            pixels = scratch.allocate(
                (len(pixel_art_images),) + tuple(frame.shape[1:]),
                node="Image Pixelate",
                advice="Passing fewer or smaller frames also fits it.",
            )
        pixels[index] = frame[0]

    if pixels is None:
        raise ValueError(
            "Image Pixelate was handed an empty image batch. Wire in at least one image."
        )
    return pixels


class ImagePixelate(io.ComfyNode):
    """Turn images into pixel art: chunky pixels and a reduced palette."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Pixelate",
            display_name="Image Pixelate",
            search_aliases=["Image Pixelate", "pixel art", "posterize", "quantize", "8-bit"],
            category="WAS Suite/Image/Process",
            description=(
                (
                    (
                        "Turn an image into pixel art with large blocky pixels and a small "
                        "number of colours, optionally dithered or remapped onto a supplied "
                        "palette. color_palette_mode is ignored with no palette connected, and "
                        "`Perceptual` is the only mode palette_dither works with. `Luminance "
                        "Ramp` throws the original colour away and places each pixel along the "
                        "palette by brightness; set init_mode to `none` with it, since "
                        "reducing the colours first destroys the shading it reads. The four "
                        "older modes compare raw channel numbers: `Brightness` takes the "
                        "closest colour overall, `Tonal` matches brightness only so hues swap "
                        "freely, `BrightnessAndTonal` weighs both, and `Linear` indexes the "
                        "palette by the red value for a hard banded remap. palette_dither "
                        "diffuses against the palette itself, unlike dither, which runs first "
                        "and is then re-quantised; `Bayer` is the one for a sequence."
                    )
                )
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The images to pixelate. Each one is handled on its own.",
                ),
                io.Float.Input(
                    "pixelation_size",
                    default=164,
                    min=16,
                    max=480,
                    step=1,
                    tooltip=(
                        "Width in pixels the image is reduced to before being blown back up, "
                        "which sets how big the blocks look. 16 gives very coarse blocks, 164 "
                        "a recognisable picture, and 480 a subtle effect. An image already "
                        "smaller than this is not touched."
                    ),
                ),
                io.Float.Input(
                    "num_colors",
                    default=16,
                    min=2,
                    max=256,
                    step=1,
                    tooltip=(
                        "How many colours to keep. 2 gives two-tone, 16 is a classic 8-bit "
                        "look, and 256 keeps most of the original shading."
                    ),
                ),
                io.Combo.Input(
                    "init_mode",
                    options=["k-means++", "random", "none"],
                    tooltip=(
                        "How the colour reduction starts. `k-means++` spreads the starting "
                        "colours apart and usually finds the better palette; `random` is "
                        "quicker and can miss a colour that covers little of the image; `none` "
                        "skips the colour reduction altogether and only shrinks the image."
                    ),
                ),
                io.Float.Input(
                    "max_iterations",
                    default=100,
                    min=1,
                    max=256,
                    step=1,
                    tooltip=(
                        "Refinement passes the colour reduction is allowed. 100 settles "
                        "almost any picture; 10 stops early and is quicker but coarser; 256 "
                        "is the most it will spend. Read only when init_mode is not `none`."
                    ),
                ),
                io.Boolean.Input(
                    "dither",
                    default=False,
                    tooltip=(
                        "`on` scatters the rounding error into neighbouring pixels, which "
                        "trades flat bands of colour for a fine speckle and makes a small "
                        "palette look richer; `off` leaves the flat areas flat."
                    ),
                ),
                io.Combo.Input(
                    "dither_mode",
                    options=["FloydSteinberg", "Ordered"],
                    tooltip=(
                        "Which dither to use when dither is on. `FloydSteinberg` gives a "
                        "fine organic stipple; `Ordered` snaps to a power-of-two number of "
                        "levels first, giving a coarser, more regular texture."
                    ),
                ),
                LIST.Input(
                    "color_palettes",
                    optional=True,
                    extra_dict={"forceInput": True},
                    tooltip=(
                        "One palette of '#rrggbb' lines per image, as Image Color Palette "
                        "emits. Connect it to repaint the result in those colours instead of "
                        "the ones found in the image. Leave it unconnected to keep the image's "
                        "own colours."
                    ),
                ),
                io.Combo.Input(
                    "color_palette_mode",
                    options=[
                        "Brightness",
                        "BrightnessAndTonal",
                        "Linear",
                        "Tonal",
                        *palette_map.MODES,
                    ],
                    optional=True,
                    tooltip=(
                        "How a pixel is matched to a palette colour. `Perceptual` picks the "
                        "closest colour as the eye sees it, and is the one to reach for."
                    ),
                ),
                io.Boolean.Input(
                    "reverse_palette",
                    default=False,
                    optional=True,
                    tooltip=(
                        "On flips the palette end to end before matching, which inverts a "
                        "dark-to-light ramp. `Linear` and `Luminance Ramp` change visibly; "
                        "the modes that match on colour rather than on position do not."
                    ),
                ),
                io.Combo.Input(
                    "palette_dither",
                    options=list(palette_map.DITHERS),
                    optional=True,
                    tooltip=(
                        "How the error left by palette matching is spread, in `Perceptual` "
                        "mode. `FloydSteinberg` gives an organic stipple; `Bayer` a fixed 8x8 "
                        "pattern that stays still."
                    ),
                ),
                io.Boolean.Input(
                    "palette_smooth",
                    default=True,
                    optional=True,
                    tooltip=(
                        "Whether `Luminance Ramp` blends between neighbouring palette "
                        "colours. On, the palette reads as a continuous gradient and shading "
                        "survives the mapping. Off, every pixel snaps to one palette colour, "
                        "banding the picture into exactly that many tones, which is usually "
                        "what pixel art wants."
                    ),
                ),
                io.Float.Input(
                    "palette_blend",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip=(
                        "How much of the palette result replaces the colours under it, in "
                        "the two perceptual modes. 1.0 is the palette alone; lower values "
                        "let the reduced original show through, which softens a colourised "
                        "plate back towards its own hues."
                    ),
                ),
                io.Boolean.Input(
                    "palette_normalize",
                    default=False,
                    optional=True,
                    tooltip=(
                        "Whether `Luminance Ramp` stretches the palette across each image's "
                        "darkest and lightest values. Off by default, which reads brightness "
                        "absolutely and keeps a sequence steady."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="The pixelated images, back at their original size.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        images,
        pixelation_size=164,
        num_colors=16,
        init_mode="random",
        max_iterations=100,
        dither=False,
        dither_mode="FloydSteinberg",
        color_palettes=None,
        color_palette_mode="Linear",
        reverse_palette=False,
        palette_dither="none",
        palette_smooth=True,
        palette_blend=1.0,
        palette_normalize=False,
    ) -> io.NodeOutput:
        folded = dynamic.fold(images)
        images = folded.images
        color_palettes_list = []
        if color_palettes:
            for palette in color_palettes:
                color_palettes_list.append([color.strip() for color in palette.splitlines()])

        pixels = pixel_art_batch(
            images,
            int(pixelation_size),
            int(num_colors),
            init_mode,
            int(max_iterations),
            color_palettes_list or None,
            color_palette_mode,
            reverse_palette,
            dither,
            dither_mode,
            palette_dither,
            palette_smooth,
            palette_blend,
            palette_normalize,
        )
        if folded.scale != 1.0:
            pixels.mul_(folded.scale)
        return io.NodeOutput(pixels)

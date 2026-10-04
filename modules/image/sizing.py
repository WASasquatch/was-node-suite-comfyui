"""Bringing images of different sizes to one size, so a batch can hold them all.

:func:`fit` resamples and places the result on a canvas of exactly the target size. Every
image here is a PIL image.
"""

from __future__ import annotations

import io

from PIL import Image, ImageOps, UnidentifiedImageError

from . import draw

__all__ = [
    "ALIGNMENTS",
    "ALIGNMENT_NAMES",
    "BOMB",
    "CHANNELS",
    "CROP_OR_PAD",
    "DEFAULT_ALIGNMENT",
    "DEFAULT_FILTER",
    "FILL_AND_CROP",
    "FILTERS",
    "FILTER_NAMES",
    "FIT_AND_PAD",
    "FOLD_BUDGET",
    "ImageTooLarge",
    "MAX_BATCH_PIXELS",
    "MAX_RESAMPLE_PIXELS",
    "MAX_SOURCE_PIXELS",
    "MODES",
    "NotAnImage",
    "STRETCH",
    "SizingError",
    "as_channels",
    "batch_limit",
    "cover_box",
    "fit",
    "fit_frames",
    "open_bytes",
    "scaled_size",
]

#: The four ways an image of one size reaches a target size. ``stretch`` distorts,
#: ``fit and pad`` shows the whole image inside bars, ``fill and crop`` fills the frame and
#: loses the overhang, and ``crop or pad`` resamples nothing at all.
STRETCH = "stretch"
FIT_AND_PAD = "fit and pad"
FILL_AND_CROP = "fill and crop"
CROP_OR_PAD = "crop or pad"

#: The modes in the order a combo offers them, the two that keep the aspect ratio first.
MODES: tuple[str, ...] = (FIT_AND_PAD, FILL_AND_CROP, STRETCH, CROP_OR_PAD)

#: The resampling filters offered, in the order a combo lists them, under the names the
#: pack's other resize nodes already use so one vocabulary covers them all.
FILTER_NAMES: tuple[str, ...] = ("lanczos", "nearest", "bilinear", "bicubic")

#: What each of :data:`FILTER_NAMES` resamples with. ``nearest`` copies the closest pixel and
#: keeps hard edges, ``bilinear`` and ``bicubic`` are progressively smoother, and ``lanczos``
#: is the sharpest and the slowest.
FILTERS: dict[str, int] = {
    "lanczos": Image.Resampling.LANCZOS,
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
}

#: The filter used when none is named. It is the sharpest of the four when shrinking, which
#: is what most of an archive of photographs needs to reach a batch size.
DEFAULT_FILTER = "lanczos"

#: Where a cropped or padded image sits inside the target, as ``(x, y)`` fractions of the
#: space left over. The pack's one anchor vocabulary, shared with the text and rectangle
#: drawing.
ALIGNMENTS: dict[str, tuple[float, float]] = draw.ANCHORS

#: The alignments in the order a combo lists them, written out rather than read off
#: :data:`ALIGNMENTS`: a combo's order is the positional meaning of every value a workflow
#: saved against it holds, so it cannot follow another module's dictionary order.
ALIGNMENT_NAMES: tuple[str, ...] = (
    "top left", "top center", "top right",
    "middle left", "middle center", "middle right",
    "bottom left", "bottom center", "bottom right",
)

#: The alignment used when none is named, and the one a combo starts on.
DEFAULT_ALIGNMENT = "middle center"

#: The channel counts a batch can carry: three for colour, four for colour and transparency.
CHANNELS: tuple[str, ...] = ("RGB", "RGBA")

#: How many pixels one source image may hold before it is refused unread. Its header is read
#: first and the size compared against this, so a file crafted to decode to tens of gigabytes
#: costs a header rather than the memory it asks for. 64 megapixels is an 8192 by 8192 image.
MAX_SOURCE_PIXELS = 64 * 1024 * 1024

#: How many pixels one batch may hold in total, counting every image in it. A float32 batch
#: costs four bytes a channel a pixel, so this is around half a gigabyte of tensor at four
#: channels. :func:`batch_limit` turns it into a number of images at a given size.
MAX_BATCH_PIXELS = 32 * 1024 * 1024

#: How large a ``fill and crop`` resample may grow the image before :func:`fit` resamples the
#: covering rectangle alone rather than the whole of it. Covering 512 by 512 from an 8192 by 4
#: strip scales by 128, which is a 537 megapixel intermediate for a 0.26 megapixel result. The
#: bound is the count one decoded source may hold, so no resample works on more pixels than an
#: image is allowed to arrive with.
MAX_RESAMPLE_PIXELS = MAX_SOURCE_PIXELS

#: Elements of source and result one run of frames holds while :func:`fit_frames` folds it.
FOLD_BUDGET = 1 << 26

#: What Pillow raises and warns with when a header claims more pixels than it will decode.
#: Neither derives from ``OSError``, and the warning becomes an exception under a warning
#: filter that turns warnings into errors, so both are caught and reported as one thing.
BOMB = (Image.DecompressionBombError, Image.DecompressionBombWarning)


class SizingError(ValueError):
    """An image could not be read or brought to a size."""


class NotAnImage(SizingError):
    """The bytes offered are not an image any bundled decoder reads."""


class ImageTooLarge(SizingError):
    """The image declares more pixels than :data:`MAX_SOURCE_PIXELS`."""


def batch_limit(width: int, height: int) -> int:
    """How many images of one size a batch may hold.

    Args:
        width: Target width in pixels.
        height: Target height in pixels.

    Returns:
        :data:`MAX_BATCH_PIXELS` divided by the area of one image, and never less than 1, so
        a single image larger than the whole budget is still loaded rather than refused.
    """
    area = max(1, int(width)) * max(1, int(height))
    return max(1, MAX_BATCH_PIXELS // area)


def open_bytes(data: bytes, max_pixels: int = MAX_SOURCE_PIXELS) -> Image.Image:
    """One image out of the bytes a file or an archive entry holds.

    Args:
        data: The file's bytes.
        max_pixels: How many pixels the image may hold. The header is read first and its
            size compared against this before any pixel is decoded.

    Returns:
        The image in mode ``RGBA``, with its EXIF orientation applied. Greyscale widens to
        equal channels, a palette resolves to its colours, and a source with no alpha gains
        an opaque one.

    Raises:
        NotAnImage: The bytes are not an image, or are too damaged to decode.
        ImageTooLarge: The image declares more than ``max_pixels`` pixels, or more than
            Pillow's own decompression bomb limit, which it applies while opening.
    """
    try:
        opened = Image.open(io.BytesIO(data))
    except BOMB as error:
        raise ImageTooLarge(_too_many(error)) from error
    except (UnidentifiedImageError, OSError, ValueError, EOFError) as error:
        raise NotAnImage(_undecodable(data, error)) from error

    width, height = opened.size
    pixels = max(0, int(width)) * max(0, int(height))
    if pixels > max_pixels:
        raise ImageTooLarge(
            f"is {width} by {height}, which is {_megapixels(pixels)} against the "
            f"{_megapixels(max_pixels)} one image may hold, so nothing was decoded. The size "
            f"came out of the file's header, which is how a file built to decode to far more "
            f"than it occupies is refused before it costs any memory"
        )
    try:
        # exif_transpose returns a new image, so the orientation is applied rather than
        # carried as a tag no downstream node reads.
        oriented = ImageOps.exif_transpose(opened)
        return oriented.convert("RGBA")
    except BOMB as error:
        raise ImageTooLarge(_too_many(error)) from error
    except (OSError, ValueError, EOFError, SyntaxError) as error:
        raise NotAnImage(
            f"its header reads as {opened.format or 'an image'} and its pixels could not be "
            f"decoded ({error}), so the file is truncated or damaged"
        ) from error


def scaled_size(
    source: tuple[int, int], target: tuple[int, int], mode: str = FIT_AND_PAD
) -> tuple[int, int]:
    """The size an image is resampled to before it is cropped or padded to the target.

    Args:
        source: ``(width, height)`` of the image.
        target: ``(width, height)`` asked for.
        mode: One of :data:`MODES`. An unknown name is read as :data:`FIT_AND_PAD`.

    Returns:
        ``(width, height)`` to resample to. It is the target itself under ``stretch``, the
        source itself under ``crop or pad``, and otherwise the source scaled by the smaller
        or the larger of the two ratios, rounded and then held to the target: ``fit and
        pad`` never overhangs and ``fill and crop`` never falls short.
    """
    width, height = max(1, int(source[0])), max(1, int(source[1]))
    wide, high = max(1, int(target[0])), max(1, int(target[1]))
    if mode == STRETCH:
        return (wide, high)
    if mode == CROP_OR_PAD:
        return (width, height)
    ratios = (wide / width, high / height)
    scale = max(ratios) if mode == FILL_AND_CROP else min(ratios)
    scaled = (int(round(width * scale)), int(round(height * scale)))
    if mode == FILL_AND_CROP:
        # Rounding down by a pixel would leave a one-pixel gap the pad colour shows through,
        # which is the one thing this mode promises not to do.
        return (max(wide, scaled[0]), max(high, scaled[1]))
    return (max(1, min(wide, scaled[0])), max(1, min(high, scaled[1])))


def cover_box(
    source: tuple[int, int], target: tuple[int, int], align: str = DEFAULT_ALIGNMENT
) -> tuple[float, float, float, float]:
    """The part of a source that fills a target without distorting it.

    Args:
        source: ``(width, height)`` of the image.
        target: ``(width, height)`` asked for.
        align: A key of :data:`ALIGNMENTS`, deciding which part of the source is kept.

    Returns:
        ``(left, upper, right, lower)`` in source pixels, carrying the target's aspect ratio
        and clamped to the image. Resampling that rectangle to the target scales by the same
        ratio and starts at the same source position as resampling the whole image to
        :func:`scaled_size` and cropping the overhang at
        :func:`modules.image.draw.anchor_origin`. The edges are fractions of a pixel, so it
        is a resampling box rather than a crop.
    """
    width, height = max(1, int(source[0])), max(1, int(source[1]))
    wide, high = max(1, int(target[0])), max(1, int(target[1]))
    scaled = scaled_size((width, height), (wide, high), FILL_AND_CROP)
    origin = draw.anchor_origin(align, (wide, high), scaled)
    left = max(0.0, -origin[0] * width / scaled[0])
    upper = max(0.0, -origin[1] * height / scaled[1])
    return (
        left,
        upper,
        min(float(width), left + wide * width / scaled[0]),
        min(float(height), upper + high * height / scaled[1]),
    )


def fit(
    image: Image.Image,
    width: int,
    height: int,
    mode: str = FIT_AND_PAD,
    resample: str = DEFAULT_FILTER,
    align: str = DEFAULT_ALIGNMENT,
    pad: tuple[int, int, int, int] = draw.TRANSPARENT,
) -> Image.Image:
    """One image at exactly ``width`` by ``height``.

    Args:
        image: The source, in mode ``RGBA``.
        width: Target width in pixels.
        height: Target height in pixels.
        mode: One of :data:`MODES`.
        resample: A key of :data:`FILTERS`. An unknown name is read as
            :data:`DEFAULT_FILTER`.
        align: A key of :data:`ALIGNMENTS`, deciding which part of an image survives a crop
            and which side carries the wider bar of a pad.
        pad: ``(red, green, blue, alpha)`` filling whatever the image does not cover.

    Returns:
        A new image of exactly the target size in mode ``RGBA``, unless the source is
        already that size and needs no resampling, in which case the source is returned.
    """
    target = (max(1, int(width)), max(1, int(height)))
    scaled = scaled_size(image.size, target, mode)
    chosen = FILTERS.get(resample, FILTERS[DEFAULT_FILTER])
    if mode == FILL_AND_CROP and scaled[0] * scaled[1] > MAX_RESAMPLE_PIXELS:
        # Covering a target from a source of a very different shape scales by the larger of
        # the two ratios, and the crop then throws almost all of it away: 8192 by 4 covering
        # 512 by 512 resamples 537 megapixels to keep 0.26 of them. Past this bound the
        # rectangle that survives is taken in source space and resampled on its own.
        return _covered(image, target, chosen, align)
    resized = image if scaled == image.size else image.resize(scaled, chosen)
    if scaled == target:
        return resized
    canvas = Image.new("RGBA", target, tuple(pad))
    # A negative origin is what a crop is: paste clips whatever falls outside the canvas, so
    # one placement covers cropping the overhang and padding the shortfall.
    canvas.paste(resized, draw.anchor_origin(align, target, scaled))
    return canvas


def _covered(
    image: Image.Image, target: tuple[int, int], resample: int, align: str
) -> Image.Image:
    """The covering rectangle of a source, brought to the target on its own.

    Args:
        image: The source.
        target: ``(width, height)`` asked for, already at least 1 by 1.
        resample: A value of :data:`FILTERS`.
        align: A key of :data:`ALIGNMENTS`.

    Returns:
        A new image of exactly the target size. A rectangle already that size is cropped
        rather than resampled, since resampling ``RGBA`` blends by transparency and would
        take the colour out from under a fully transparent pixel for no gain.
    """
    box = cover_box(image.size, target, align)
    if (box[2] - box[0], box[3] - box[1]) == target:
        left, upper = int(box[0]), int(box[1])
        return image.crop((left, upper, left + target[0], upper + target[1]))
    return image.resize(target, resample, box=box)


def as_channels(image: Image.Image, channels: str = "RGB") -> Image.Image:
    """One image in the channel count a batch carries.

    Args:
        image: Any PIL image.
        channels: ``"RGB"`` or ``"RGBA"``. Anything else is read as ``"RGB"``.

    Returns:
        The image in that mode. ``RGB`` drops the alpha channel and keeps the colour that
        was under it, rather than compositing it onto a background, so nothing decides a
        background colour on the caller's behalf. Two things qualify that: a pixel is only
        carrying its own colour where :func:`fit` did not resample it, since resampling
        ``RGBA`` blends by transparency and a fully transparent pixel comes back black; and
        a partly transparent one keeps its colour, since the blend is divided out again.
    """
    wanted = "RGBA" if channels == "RGBA" else "RGB"
    return image if image.mode == wanted else image.convert(wanted)


def _megapixels(count: int) -> str:
    """A pixel count as a rounded number of megapixels, for a message."""
    return f"{count / (1024 * 1024):.1f} megapixels"


def _too_many(error: Exception) -> str:
    """What to say about a header Pillow itself refused for the size it claims.

    Args:
        error: The bomb error or warning, whose text carries the pixel count.

    Returns:
        A phrase completing "the entry <name> ", quoting the count so the file can be told
        apart from the limit that refused it.
    """
    return (
        f"declares more pixels than are decoded here ({error}), so nothing was decoded. The "
        f"size came out of the file's header, which is how a file built to decode to far more "
        f"than it occupies is refused before it costs any memory"
    )


def _undecodable(data: bytes, error: Exception) -> str:
    """What to say about bytes that are not an image.

    Args:
        data: The bytes that were offered.
        error: What the decoder raised.

    Returns:
        A phrase completing "the entry <name> ", naming what the bytes look like instead so
        a text file or a document under an image extension is told apart from a damaged
        picture.
    """
    if not data:
        return "holds no bytes at all"
    opening = data.lstrip()[:1]
    if data[:2] == b"PK":
        looks = "another zip file, such as a document"
    elif opening in (b"<", b"{"):
        looks = "markup or JSON"
    elif _is_text(data[:512]):
        looks = "plain text"
    else:
        looks = "some other kind of file"
    return (
        f"is not an image any of the bundled decoders reads ({error}); its first bytes look "
        f"like {looks}, so it carries an image extension without holding an image"
    )


def _is_text(prefix: bytes) -> bool:
    """Whether a prefix decodes as UTF-8 text holding no control bytes."""
    try:
        text = prefix.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return all(character >= " " or character in "\t\r\n" for character in text)


def _frame_plans(source, target, mode, resampling, align):
    """Axis plans bringing frames of one size to a target in one of :data:`MODES`.

    Args:
        source: ``(width, height)`` of the frames.
        target: ``(width, height)`` asked for.
        mode: One of :data:`MODES`.
        resampling: A key of :data:`FILTERS`.
        align: A key of :data:`ALIGNMENTS`.

    Returns:
        ``(across, down)`` plans for :func:`modules.image.resample.apply`.
    """
    from . import resample

    (width, height), (wide, high) = source, target
    scaled = scaled_size(source, target, mode)
    if mode == FILL_AND_CROP and scaled[0] * scaled[1] > MAX_RESAMPLE_PIXELS:
        left, upper, right, lower = cover_box(source, target, align)
        if (right - left, lower - upper) == (wide, high):
            left, upper = float(int(left)), float(int(upper))
            right, lower = left + wide, upper + high
        return (
            resample.axis(width, wide, wide, 0, resampling, (left, right)),
            resample.axis(height, high, high, 0, resampling, (upper, lower)),
        )
    across, down = draw.anchor_origin(align, target, scaled)
    return (
        resample.axis(width, scaled[0], wide, across, resampling),
        resample.axis(height, scaled[1], high, down, resampling),
    )


def fit_frames(frames, target, mode=FIT_AND_PAD, resample_name=DEFAULT_FILTER,
               align=DEFAULT_ALIGNMENT, pad=draw.TRANSPARENT, supersample: int = 1,
               scale: float = 1.0):
    """A batch of float frames at exactly the target size, as :func:`fit` places one image.

    Args:
        frames: ``(batch, height, width, channels)`` float tensor, any precision.
        target: ``(width, height)`` asked for.
        mode: One of :data:`MODES`.
        resample_name: A key of :data:`FILTERS`.
        align: A key of :data:`ALIGNMENTS`.
        pad: ``(red, green, blue, alpha)`` in 0 to 255 filling what the frames do not cover.
        supersample: Resample through this many times the target, folded into one pass. Not
            applied under :data:`CROP_OR_PAD`.
        scale: From :func:`modules.image.dynamic.factor`. Each run of frames is folded by it
            before the resample and unfolded after; 1.0 resamples the frames as they are.

    Returns:
        The batch on the frames' own device, resampled on the GPU where there is one. A CPU
        batch comes from :func:`modules.image.scratch.allocate`.

    Raises:
        MemoryError: Neither free memory nor any scratch drive has room for the result.
    """
    import comfy.model_management
    import torch

    from . import dynamic, resample, scratch

    name = resample_name if resample_name in FILTERS else DEFAULT_FILTER
    target = (max(1, int(target[0])), max(1, int(target[1])))
    source = (int(frames.shape[2]), int(frames.shape[1]))
    if supersample > 1 and mode != CROP_OR_PAD:
        large = (target[0] * supersample, target[1] * supersample)
        across, down = _frame_plans(source, large, mode, name, align)
        across = resample.supersampled(across, target[0], name)
        down = resample.supersampled(down, target[1], name)
    else:
        across, down = _frame_plans(source, target, mode, name, align)
    options = {
        "pad": [value / 255.0 for value in pad],
        "device": comfy.model_management.get_torch_device(),
        "premultiply": name != "nearest",
    }
    if scale == 1.0:
        return resample.apply(frames, across, down, **options)

    batch, channels = int(frames.shape[0]), int(frames.shape[3])
    shape = (batch, target[1], target[0], channels)
    if frames.device.type == "cpu":
        result = scratch.allocate(shape, frames.dtype, advice=resample.ADVICE)
    else:
        result = torch.empty(shape, dtype=frames.dtype, device=frames.device)
    per_frame = max(1, (source[0] * source[1] + target[0] * target[1]) * channels)
    step = max(1, FOLD_BUDGET // per_frame)
    for begin in range(0, batch, step):
        folded = dynamic.fold(frames[begin:begin + step], scale)
        held = result[begin:begin + step]
        held.copy_(resample.apply(folded.images, across, down, **options))
        dynamic.unfold(held, folded, in_place=True)
    return result

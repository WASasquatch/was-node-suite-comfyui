"""Writing PNG files at 8 or 16 bits a sample.

Pixels are ``(height, width, channels)`` arrays, float in ``[0, 1]`` or integer at their
stored depth. One to four channels are greyscale, greyscale with alpha, RGB and RGBA.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

__all__ = ["BIT_DEPTHS", "encode", "write"]

#: What a PNG file opens with.
SIGNATURE = b"\x89PNG\r\n\x1a\n"

#: Sample depths this writes, in bits.
BIT_DEPTHS = (8, 16)

#: Channel count -> the colour type written into IHDR.
COLOUR_TYPES = {1: 0, 2: 4, 3: 2, 4: 6}

#: Metres in an inch, which converts a dpi figure into the pixels per metre pHYs carries.
METRES_PER_INCH = 0.0254

#: Unit code pHYs uses for a metre.
PHYS_METRE = 1

#: Deflate level used with and without ``optimize``.
LEVEL_OPTIMISED = 9
LEVEL_FAST = 6

#: Longest keyword a tEXt or iTXt chunk may carry.
KEYWORD_LIMIT = 79


def _chunk(kind: bytes, payload: bytes) -> bytes:
    """One length-prefixed, CRC-suffixed chunk.

    Args:
        kind: Four-byte chunk name.
        payload: The chunk's contents.

    Returns:
        The chunk as it is written to disk.
    """
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _text_chunk(keyword: str, value: str) -> bytes:
    """A tEXt chunk, or an iTXt chunk where the text is outside latin-1.

    Args:
        keyword: Name the value is stored under, truncated to 79 characters.
        value: The text to store.

    Returns:
        The chunk as it is written to disk.
    """
    name = keyword.encode("latin-1", "replace")[:KEYWORD_LIMIT]
    try:
        return _chunk(b"tEXt", name + b"\0" + value.encode("latin-1"))
    except UnicodeEncodeError:
        return _chunk(b"iTXt", name + b"\0\0\0\0\0" + value.encode("utf-8"))


def _quantise(image, depth: int):
    """The pixel array as unsigned samples of the requested depth, big-endian.

    Args:
        image: ``(height, width)`` or ``(height, width, channels)`` array. A floating point
            array is read as ``[0, 1]`` and scaled; an integer array is taken as written.
        depth: 8 or 16.

    Returns:
        A C-contiguous ``uint8`` or big-endian ``uint16`` array shaped
        ``(height, width, channels)``.

    Raises:
        ValueError: The depth is not one this writes, or the array has no usable shape.
    """
    import numpy as np

    if depth not in BIT_DEPTHS:
        raise ValueError(f"a PNG carries 8 or 16 bits a sample, not {depth}")

    array = np.asarray(image)
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3 or array.shape[2] not in COLOUR_TYPES:
        raise ValueError(
            f"a PNG holds one to four channels shaped (height, width, channels), "
            f"not an array of shape {tuple(array.shape)}"
        )

    peak = (1 << depth) - 1
    target = np.dtype(">u2") if depth == 16 else np.dtype("u1")
    if np.issubdtype(array.dtype, np.floating):
        scaled = np.clip(array.astype(np.float64) * peak, 0, peak)
        return np.ascontiguousarray(np.rint(scaled).astype(target))
    return np.ascontiguousarray(np.clip(array, 0, peak).astype(target))


def _filtered(raw, width_bytes: int, bpp: int) -> bytes:
    """Every scanline prefixed with the filter that packs it smallest.

    Args:
        raw: ``(height, width_bytes)`` uint8 view of the sample data.
        width_bytes: Bytes in one scanline.
        bpp: Bytes one whole pixel occupies, at least 1.

    Returns:
        The filtered scanlines, ready for deflate.
    """
    import numpy as np

    height = raw.shape[0]
    line = np.ascontiguousarray(raw)
    prior = np.zeros_like(line)
    prior[1:] = line[:-1]
    shifted = np.zeros_like(line)
    if bpp < width_bytes:
        shifted[:, bpp:] = line[:, :-bpp]

    left = shifted.astype(np.int16)
    up = prior.astype(np.int16)
    upper_left = np.zeros_like(up)
    if bpp < width_bytes:
        upper_left[:, bpp:] = up[:, :-bpp]

    estimate = left + up - upper_left
    to_left, to_up, to_corner = np.abs(estimate - left), np.abs(estimate - up), np.abs(estimate - upper_left)
    paeth = np.where(
        (to_left <= to_up) & (to_left <= to_corner),
        left,
        np.where(to_up <= to_corner, up, upper_left),
    )

    wide = line.astype(np.int16)
    candidates = np.stack((
        line,
        (line - shifted).astype(np.uint8),
        (line - prior).astype(np.uint8),
        (wide - ((left + up) >> 1)).astype(np.uint8),
        (wide - paeth).astype(np.uint8),
    ))
    # The filter is picked by the smallest sum of the scanline read as signed bytes, the
    # heuristic the format's own specification gives; a tie goes to the lower filter.
    scores = np.abs(candidates.view(np.int8).astype(np.int32)).sum(axis=2)
    choice = scores.argmin(axis=0)
    out = np.empty((height, width_bytes + 1), dtype=np.uint8)
    out[:, 0] = choice
    out[:, 1:] = candidates[choice, np.arange(height)]
    return out.tobytes()


def encode(image, *, depth: int = 8, dpi=None, icc: bytes | None = None,
           text=None, optimize: bool = True) -> bytes:
    """A whole PNG file, as bytes.

    Args:
        image: ``(height, width)`` or ``(height, width, channels)`` array. Floating point is
            read as ``[0, 1]``.
        depth: Bits a sample, 8 or 16.
        dpi: Resolution written into pHYs, as a number or an ``(x, y)`` pair. None writes no
            pHYs chunk.
        icc: ICC profile bytes written into iCCP, or None.
        text: Mapping of keyword to text, written as one chunk each.
        optimize: Spend longer on deflate.

    Returns:
        The encoded file.

    Raises:
        ValueError: The depth or the array shape is not one a PNG carries.
    """
    samples = _quantise(image, depth)
    height, width, channels = samples.shape
    if width < 1 or height < 1:
        raise ValueError("a PNG holds at least one pixel in each direction")

    parts = [
        SIGNATURE,
        _chunk(
            b"IHDR",
            struct.pack(">IIBBBBB", width, height, depth, COLOUR_TYPES[channels], 0, 0, 0),
        ),
    ]

    if icc:
        parts.append(
            _chunk(b"iCCP", b"ICC profile\0\0" + zlib.compress(icc, LEVEL_OPTIMISED))
        )
    if dpi is not None:
        pair = dpi if isinstance(dpi, (tuple, list)) else (dpi, dpi)
        per_metre = tuple(max(1, round(float(value) / METRES_PER_INCH)) for value in pair[:2])
        parts.append(_chunk(b"pHYs", struct.pack(">IIB", per_metre[0], per_metre[1], PHYS_METRE)))
    for keyword, value in (text or {}).items():
        parts.append(_text_chunk(str(keyword), str(value)))

    import numpy as np

    width_bytes = width * channels * (depth // 8)
    raw = samples.reshape(height, -1).view(np.uint8).reshape(height, width_bytes)
    level = LEVEL_OPTIMISED if optimize else LEVEL_FAST
    body = _filtered(raw, width_bytes, max(1, channels * (depth // 8)))
    parts.append(_chunk(b"IDAT", zlib.compress(body, level)))
    parts.append(_chunk(b"IEND", b""))
    return b"".join(parts)


def write(path, image, *, depth: int = 8, dpi=None, icc: bytes | None = None,
          text=None, optimize: bool = True) -> None:
    """Encode a PNG and write it to ``path``.

    Args:
        path: Where the file is written.
        image: ``(height, width)`` or ``(height, width, channels)`` array in ``[0, 1]``.
        depth: Bits a sample, 8 or 16.
        dpi: Resolution written into pHYs, or None.
        icc: ICC profile bytes, or None.
        text: Mapping of keyword to text.
        optimize: Spend longer on deflate.

    Raises:
        ValueError: The depth or the array shape is not one a PNG carries.
        OSError: The file could not be written.
    """
    Path(path).write_bytes(
        encode(image, depth=depth, dpi=dpi, icc=icc, text=text, optimize=optimize)
    )

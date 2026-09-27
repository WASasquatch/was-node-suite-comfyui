"""Animated WebP written from frames encoded in parallel.

Each frame is a complete still WebP, placed whole at the origin with blending off, inside
one ``VP8X`` container carrying an ``ANIM`` chunk and an optional ``EXIF`` chunk.
"""

from __future__ import annotations

import io
import os
import struct
from concurrent.futures import ThreadPoolExecutor

#: The most frames encoded at once.
MAX_WORKERS = 8

#: ``VP8X`` feature bits.
FLAG_ANIMATION = 0x02
FLAG_EXIF = 0x08
FLAG_ALPHA = 0x10

#: ``ANMF`` flags for a whole frame drawn without blending and left in place.
FRAME_NO_BLEND = 0x02

#: Largest value a 24 bit field holds.
MAX_24 = (1 << 24) - 1


def _chunk(kind: bytes, payload: bytes) -> bytes:
    """One RIFF chunk, padded to an even length."""
    pad = b"\0" if len(payload) % 2 else b""
    return kind + struct.pack("<I", len(payload)) + payload + pad


def _u24(value: int) -> bytes:
    """A little-endian 24 bit field."""
    return struct.pack("<I", max(0, min(MAX_24, int(value))))[:3]


def _frame_chunks(data: bytes) -> tuple[bytes, bool]:
    """The image chunks inside one still WebP.

    Args:
        data: A complete still WebP file.

    Returns:
        ``(chunks, has_alpha)``: the ``ALPH``, ``VP8 `` and ``VP8L`` chunks in file order,
        and whether the frame carries transparency.

    Raises:
        ValueError: The bytes are not a WebP file.
    """
    if data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise ValueError("the encoder did not return a WebP file")
    position, kept, alpha = 12, [], False
    while position + 8 <= len(data):
        kind = data[position:position + 4]
        size = struct.unpack("<I", data[position + 4:position + 8])[0]
        payload = data[position + 8:position + 8 + size]
        if kind in (b"ALPH", b"VP8 ", b"VP8L"):
            kept.append(_chunk(kind, payload))
            if kind == b"ALPH":
                alpha = True
            elif kind == b"VP8L" and len(payload) >= 5:
                # Bit 28 of the header after the signature byte marks alpha in use.
                alpha = alpha or bool((struct.unpack("<I", payload[1:5])[0] >> 28) & 1)
        position += 8 + size + (size % 2)
    return b"".join(kept), alpha


def encode(frames, duration_ms: int, lossless: bool = True, quality: int = 80,
           method: int = 4, exif: bytes = b"", workers: int | None = None) -> bytes:
    """Encode PIL frames as one looping animated WebP.

    Args:
        frames: PIL images, all the same size, ``RGB`` or ``RGBA``.
        duration_ms: How long each frame shows, in milliseconds.
        lossless: Whether frames are stored losslessly.
        quality: 0 to 100. Lossy picture quality, or lossless effort.
        method: 0 (fastest) to 6 (smallest).
        exif: EXIF bytes to carry, empty for none.
        workers: Frames encoded at once, a value chosen from the CPU count when None.

    Returns:
        The file's bytes.
    """
    width, height = frames[0].size

    def still(frame):
        buffer = io.BytesIO()
        frame.save(buffer, "WEBP", lossless=lossless, quality=quality, method=method)
        return _frame_chunks(buffer.getvalue())

    count = workers or max(1, min(len(frames), os.cpu_count() or 1, MAX_WORKERS))
    if count > 1:
        with ThreadPoolExecutor(max_workers=count) as pool:
            encoded = list(pool.map(still, frames))
    else:
        encoded = [still(frame) for frame in frames]

    if exif.startswith(b"Exif\0\0"):
        exif = exif[6:]
    flags = FLAG_ANIMATION | (FLAG_EXIF if exif else 0)
    if any(alpha for _, alpha in encoded):
        flags |= FLAG_ALPHA

    parts = [
        _chunk(b"VP8X", bytes([flags, 0, 0, 0]) + _u24(width - 1) + _u24(height - 1)),
        # Background colour, then a loop count of 0 for forever.
        _chunk(b"ANIM", struct.pack("<IH", 0, 0)),
    ]
    header = _u24(0) + _u24(0) + _u24(width - 1) + _u24(height - 1) + _u24(duration_ms)
    for chunks, _ in encoded:
        parts.append(_chunk(b"ANMF", header + bytes([FRAME_NO_BLEND]) + chunks))
    if exif:
        parts.append(_chunk(b"EXIF", exif))
    body = b"WEBP" + b"".join(parts)
    return b"RIFF" + struct.pack("<I", len(body)) + body

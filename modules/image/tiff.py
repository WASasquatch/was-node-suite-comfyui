"""Laying out and reading a TIFF file.

Fields are packed in the byte order the file marker names. A picture is one or more strips
of whole rows, with the samples of a pixel side by side.
"""

from __future__ import annotations

__all__ = [
    "ASCII",
    "BYTE",
    "DEFLATE",
    "DOUBLE",
    "FLOAT",
    "IMAGE_SOURCE_DATA",
    "LONG",
    "NONE",
    "RATIONAL",
    "SHORT",
    "SLONG",
    "SRATIONAL",
    "UNDEFINED",
    "WIDTHS",
    "ZIP",
    "build",
    "directory",
    "numbers",
    "plane",
]

import struct
import zlib

#: TIFF field types, by the code written into an entry.
BYTE = 1
ASCII = 2
SHORT = 3
LONG = 4
RATIONAL = 5
UNDEFINED = 7
SLONG = 9
SRATIONAL = 10
FLOAT = 11
DOUBLE = 12

#: Bytes one value of each field type takes up.
WIDTHS = {
    BYTE: 1, ASCII: 1, SHORT: 2, LONG: 4, RATIONAL: 8,
    UNDEFINED: 1, SLONG: 4, SRATIONAL: 8, FLOAT: 4, DOUBLE: 8,
}

#: Denominator every rational is written over.
SCALE = 1000000

#: Compression codes a strip may carry.
NONE = 1
DEFLATE = 8
ZIP = 32946

#: Tag holding the Photoshop layer block of a layered TIFF.
IMAGE_SOURCE_DATA = 37724

#: Tags the layout fills in once it knows where the strips landed.
STRIP_OFFSETS = 273
STRIP_BYTE_COUNTS = 279

#: Tags the reader needs to lay a picture back out.
WIDTH = 256
HEIGHT = 257
BITS_PER_SAMPLE = 258
COMPRESSION = 259
SAMPLES_PER_PIXEL = 277
ROWS_PER_STRIP = 278
PLANAR = 284
PREDICTOR = 317
SAMPLE_FORMAT = 339

#: Sample formats a reader may meet.
UNSIGNED = 1
SIGNED = 2
FLOATING = 3


def pack(kind: int, values, order: str = "<") -> bytes:
    """One field's values as the bytes a TIFF entry stores them in.

    Args:
        kind: One of the field type codes.
        values: A string for ASCII, a bytes object for UNDEFINED, a sequence otherwise.
        order: ``"<"`` for little endian, ``">"`` for big endian.

    Returns:
        The packed values, without any padding.
    """
    if kind == ASCII:
        return values.encode("ascii", "replace") + b"\0"
    if kind in (BYTE, UNDEFINED):
        return bytes(values)
    if kind == SHORT:
        return b"".join(struct.pack(f"{order}H", int(v)) for v in values)
    if kind == LONG:
        return b"".join(struct.pack(f"{order}I", int(v)) for v in values)
    if kind == SLONG:
        return b"".join(struct.pack(f"{order}i", int(v)) for v in values)
    if kind == RATIONAL:
        return b"".join(
            struct.pack(f"{order}II", int(round(v * SCALE)), SCALE) for v in values
        )
    if kind == SRATIONAL:
        return b"".join(
            struct.pack(f"{order}ii", int(round(v * SCALE)), SCALE) for v in values
        )
    if kind == FLOAT:
        return b"".join(struct.pack(f"{order}f", float(v)) for v in values)
    return b"".join(struct.pack(f"{order}d", float(v)) for v in values)


def count(kind: int, values) -> int:
    """How many values a field holds, counting a string's terminator."""
    return len(values) + 1 if kind == ASCII else len(values)


def build(fields, strips, order: str = "<") -> bytes:
    """A whole TIFF file as bytes.

    Args:
        fields: ``(tag, kind, values)`` for every entry but the strip offsets and the
            strip byte counts, which are written from ``strips``.
        strips: The picture, as one blob per run of rows, in row order.
        order: ``"<"`` for a little endian file, ``">"`` for a big endian one.

    Returns:
        The file, header first and the strips last.
    """
    bodies = list(strips)
    listed = list(fields) + [
        (STRIP_OFFSETS, LONG, [0] * len(bodies)),
        (STRIP_BYTE_COUNTS, LONG, [len(body) for body in bodies]),
    ]
    listed.sort(key=lambda entry: entry[0])

    # An entry holds its values inline under four bytes and an offset over them.
    header = 8
    table = 2 + 12 * len(listed) + 4
    overflow_at = header + table
    sizes, at = {}, overflow_at
    for tag, kind, values in listed:
        length = len(pack(kind, values, order))
        if length > 4:
            sizes[tag] = at
            at += length + length % 2

    offsets = []
    for body in bodies:
        offsets.append(at)
        at += len(body)

    entries, overflow = bytearray(), bytearray()
    for tag, kind, values in listed:
        held = offsets if tag == STRIP_OFFSETS else values
        blob = pack(kind, held, order)
        if len(blob) > 4:
            payload = struct.pack(f"{order}I", sizes[tag])
            overflow += blob + (b"\0" if len(blob) % 2 else b"")
        else:
            payload = blob.ljust(4, b"\0")
        entries += struct.pack(f"{order}HHI", tag, kind, count(kind, held)) + payload

    return b"".join(
        [
            struct.pack(f"{order}2sHI", b"II" if order == "<" else b"MM", 42, header),
            struct.pack(f"{order}H", len(listed)),
            bytes(entries),
            struct.pack(f"{order}I", 0),
            bytes(overflow),
            *bodies,
        ]
    )


def directory(data: bytes) -> tuple[str, dict]:
    """The first image file directory of a TIFF.

    Args:
        data: The whole file.

    Returns:
        ``(order, fields)``. The order is ``"<"`` or ``">"``, and each field is keyed by
        its tag and holds ``(kind, count, raw bytes)``.

    Raises:
        ValueError: The bytes carry no TIFF header.
    """
    if len(data) < 8 or data[:2] not in (b"II", b"MM"):
        raise ValueError("this is not a TIFF: the file does not begin with II or MM")
    order = "<" if data[:2] == b"II" else ">"
    version, first = struct.unpack_from(f"{order}HI", data, 2)
    if version != 42:
        raise ValueError(f"this is not a TIFF: its version marker is {version}, not 42")

    found = {}
    total = struct.unpack_from(f"{order}H", data, first)[0]
    for index in range(total):
        at = first + 2 + index * 12
        tag, kind, size = struct.unpack_from(f"{order}HHI", data, at)
        length = WIDTHS.get(kind, 1) * size
        start = struct.unpack_from(f"{order}I", data, at + 8)[0] if length > 4 else at + 8
        found[tag] = (kind, size, data[start : start + length])
    return order, found


def numbers(order: str, entry, fallback=()) -> tuple[int, ...]:
    """One field's values as whole numbers.

    Args:
        order: The file's byte order.
        entry: A ``(kind, count, raw bytes)`` field, or None.
        fallback: What to answer where the field is absent or holds no numbers.

    Returns:
        The values, or ``fallback``.
    """
    if entry is None:
        return tuple(fallback)
    kind, size, blob = entry
    if kind == SHORT:
        return struct.unpack_from(f"{order}{size}H", blob, 0)
    if kind == LONG:
        return struct.unpack_from(f"{order}{size}I", blob, 0)
    if kind in (BYTE, UNDEFINED):
        return tuple(blob[:size])
    return tuple(fallback)


def _undone(body: bytes, width: int, samples: int, bits: int, order: str) -> bytes:
    """A strip with the horizontal predictor taken back out of it."""
    if bits == 8:
        rows = bytearray(body)
        stride = width * samples
        for row in range(len(rows) // stride):
            base = row * stride
            for column in range(samples, stride):
                rows[base + column] = (rows[base + column] + rows[base + column - samples]) & 0xFF
        return bytes(rows)
    if bits == 16:
        values = list(struct.unpack(f"{order}{len(body) // 2}H", body))
        stride = width * samples
        for row in range(len(values) // stride):
            base = row * stride
            for column in range(samples, stride):
                held = values[base + column] + values[base + column - samples]
                values[base + column] = held & 0xFFFF
        return struct.pack(f"{order}{len(values)}H", *values)
    raise ValueError(f"a {bits} bit TIFF with a horizontal predictor is not read here")


def plane(data: bytes) -> tuple[int, int, int, int, int, bytes]:
    """The first picture of a TIFF, as the samples it stores.

    Args:
        data: The whole file.

    Returns:
        ``(width, height, samples, bits, format, body)``. The body holds every row one
        after another, with a pixel's samples side by side, in the file's byte order.

    Raises:
        ValueError: The file is not a TIFF, or it is packed in a way this does not read.
    """
    order, fields = directory(data)
    width = numbers(order, fields.get(WIDTH), (0,))[0]
    height = numbers(order, fields.get(HEIGHT), (0,))[0]
    samples = numbers(order, fields.get(SAMPLES_PER_PIXEL), (1,))[0]
    bits = numbers(order, fields.get(BITS_PER_SAMPLE), (8,))[0]
    packing = numbers(order, fields.get(COMPRESSION), (NONE,))[0]
    layout = numbers(order, fields.get(PLANAR), (1,))[0]
    guess = numbers(order, fields.get(SAMPLE_FORMAT), (UNSIGNED,))[0]
    predictor = numbers(order, fields.get(PREDICTOR), (1,))[0]

    if not width or not height:
        raise ValueError("this TIFF names no picture size")
    if layout != 1:
        raise ValueError("a TIFF storing each colour in its own plane is not read here")
    if packing not in (NONE, DEFLATE, ZIP):
        raise ValueError(
            f"this TIFF is packed with method {packing}, and only an unpacked or a "
            f"deflated one is read here"
        )

    offsets = numbers(order, fields.get(STRIP_OFFSETS))
    lengths = numbers(order, fields.get(STRIP_BYTE_COUNTS))
    rows = numbers(order, fields.get(ROWS_PER_STRIP), (height,))[0] or height

    body = bytearray()
    for index, (start, length) in enumerate(zip(offsets, lengths)):
        strip = data[start : start + length]
        if packing in (DEFLATE, ZIP):
            strip = zlib.decompress(strip)
        if predictor == 2:
            tall = min(rows, height - index * rows)
            strip = _undone(strip, width, samples, bits, order)[: tall * width * samples * (bits // 8)]
        body += strip

    wanted = width * height * samples * (bits // 8)
    if len(body) < wanted:
        raise ValueError(
            f"this TIFF names a {width}x{height} picture of {samples} sample(s) at {bits} "
            f"bits, which needs {wanted} bytes, and its strips hold {len(body)}"
        )
    return width, height, samples, bits, guess, bytes(body[:wanted])

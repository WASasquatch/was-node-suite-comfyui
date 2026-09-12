"""Reading and writing a layered document, as a PSD or as a layered TIFF.

Both carry the same layer block, a TIFF in its image source data tag. A layer is a
:class:`Plate`, shaped ``(height, width, 3)`` beside its coverage.
"""

from __future__ import annotations

__all__ = [
    "BLEND_KEYS",
    "DEPTHS",
    "EXTENSIONS",
    "FORMATS",
    "MAX_SIDE",
    "PACKINGS",
    "Plate",
    "bits_of",
    "read",
    "write",
]

import struct
import zlib
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch

#: Bit depth a document is stored at, in menu order.
DEPTHS = ("8 bit", "16 bit", "32 bit float")

#: Samples per channel behind each depth.
_BITS = {DEPTHS[0]: 8, DEPTHS[1]: 16, DEPTHS[2]: 32}

#: How channel data is packed, in menu order.
PACKINGS = ("rle", "zip", "none")

#: The two container formats, in menu order.
FORMATS = ("psd", "tiff")

#: The extension each format is written with.
EXTENSIONS = {"psd": "psd", "tiff": "tif"}

#: Longest side a PSD may name.
MAX_SIDE = 30000

#: The compositor's blend mode names against the four letter keys a document stores.
BLEND_KEYS = {
    "normal": b"norm",
    "multiply": b"mul ",
    "screen": b"scrn",
    "overlay": b"over",
    "darken": b"dark",
    "lighten": b"lite",
    "color-dodge": b"div ",
    "color-burn": b"idiv",
    "hard-light": b"hLit",
    "soft-light": b"sLit",
    "difference": b"diff",
    "exclusion": b"smud",
    "linear-dodge": b"lddg",
    "linear-burn": b"lbrn",
    "vivid-light": b"vLit",
    "pin-light": b"pLit",
    "linear-light": b"lLit",
    "hard-mix": b"hMix",
    "subtract": b"fsub",
    "divide": b"fdiv",
    "hue": b"hue ",
    "saturation": b"sat ",
    "color": b"colr",
    "luminosity": b"lum ",
}

#: The same table keyed the other way.
_BLEND_NAMES = {key: name for name, key in BLEND_KEYS.items()}

#: Colour mode of an RGB document.
_RGB = 3

#: Layer flags: the modern marker, and the hidden marker.
_FLAG_MODERN = 8
_FLAG_HIDDEN = 2

#: The channel identifier a layer's coverage carries.
_ALPHA = -1

#: What the image source data tag of a layered TIFF opens with.
_TIFF_PREAMBLE = b"Adobe Photoshop Document Data Block\0"

#: The tagged block key holding the layers, by bit depth.
_LAYER_KEYS = {8: b"Layr", 16: b"Lr16", 32: b"Lr32"}

#: Compression codes a channel may carry.
_RAW = 0
_RLE = 1
_ZIP = 2
_ZIP_PREDICTED = 3

#: Longest a packed row may be.
_ROW_LIMIT = 0xFFFF

#: Rows a written TIFF puts in one strip.
_STRIP_ROWS = 64

#: What the resolution resource records, in pixels per inch.
_RESOLUTION = 72.0


class Plate(NamedTuple):
    """One layer of a document, at the size and place it is drawn.

    Attributes:
        name: What the layer is called.
        x: Left edge on the canvas.
        y: Top edge on the canvas.
        image: ``(height, width, 3)`` picture codes.
        alpha: ``(height, width)`` coverage, 1 where the layer paints.
        opacity: How far the layer is faded, 0.0 to 1.0.
        blend_mode: One of the keys of :data:`BLEND_KEYS`.
        visible: Whether an editor draws it.
    """

    name: str
    x: int
    y: int
    image: "torch.Tensor"
    alpha: "torch.Tensor"
    opacity: float = 1.0
    blend_mode: str = "normal"
    visible: bool = True


def bits_of(depth: str) -> int:
    """Samples per channel one depth name asks for.

    Args:
        depth: One of :data:`DEPTHS`.

    Returns:
        8, 16 or 32.

    Raises:
        ValueError: The name is not one of :data:`DEPTHS`.
    """
    if depth not in _BITS:
        raise ValueError(
            f"'{depth}' is not a depth this writes. Pick one of: {', '.join(DEPTHS)}"
        )
    return _BITS[depth]


# ---------------------------------------------------------------------- packing


def _packbits(data: bytes) -> bytes:
    """One row packed the way a PSD scanline is packed."""
    out = bytearray()
    at, total = 0, len(data)
    while at < total:
        run = 1
        while at + run < total and data[at + run] == data[at] and run < 128:
            run += 1
        if run >= 3:
            out += bytes((257 - run, data[at]))
            at += run
            continue
        start = at
        at += 1
        while at < total and at - start < 128:
            if at + 2 < total and data[at] == data[at + 1] == data[at + 2]:
                break
            at += 1
        out.append(at - start - 1)
        out += data[start:at]
    return bytes(out)


def _unpackbits(data: bytes, wanted: int) -> bytes:
    """One packed row back as the samples it held."""
    out = bytearray()
    at = 0
    while at < len(data) and len(out) < wanted:
        header = data[at]
        at += 1
        if header < 128:
            out += data[at : at + header + 1]
            at += header + 1
        elif header > 128:
            out += bytes((data[at],)) * (257 - header)
            at += 1
    return bytes(out)


def _samples(values, bits: int) -> "np.ndarray":
    """One float plane as the big endian samples a document stores."""
    body = values.detach().to(device="cpu", dtype=torch.float32).numpy()
    if bits == 8:
        return np.clip(np.rint(body * 255.0), 0, 255).astype(">u1")
    if bits == 16:
        return np.clip(np.rint(body * 65535.0), 0, 65535).astype(">u2")
    return body.astype(">f4")


def _floats(raw: bytes, bits: int, height: int, width: int) -> "torch.Tensor":
    """Stored samples back as a ``(height, width)`` float plane on a 0 to 1 scale."""
    if bits == 8:
        held = np.frombuffer(raw, dtype=">u1").astype(np.float32) / 255.0
    elif bits == 16:
        held = np.frombuffer(raw, dtype=">u2").astype(np.float32) / 65535.0
    else:
        held = np.frombuffer(raw, dtype=">f4").astype(np.float32)
    return torch.from_numpy(held.reshape(height, width).copy())


def _rle_fits(width: int, bits: int) -> bool:
    """Whether a row of this width can always be packed into a two byte count."""
    row = width * (bits // 8)
    return row + -(-row // 128) <= _ROW_LIMIT


def _channel(values, bits: int, packing: str) -> bytes:
    """One channel of one layer, its compression word first."""
    body = _samples(values, bits)
    height, width = int(body.shape[0]), int(body.shape[1])
    if packing == "rle" and _rle_fits(width, bits):
        rows = [_packbits(body[row].tobytes()) for row in range(height)]
        counts = np.array([len(row) for row in rows], dtype=">u2").tobytes()
        return struct.pack(">H", _RLE) + counts + b"".join(rows)
    if packing != "none":
        return struct.pack(">H", _ZIP) + zlib.compress(body.tobytes(), 6)
    return struct.pack(">H", _RAW) + body.tobytes()


def _merged(planes, bits: int, packing: str) -> bytes:
    """The flattened picture's section: one compression word, then every channel."""
    width = int(planes[0].shape[1])
    if packing != "none" and _rle_fits(width, bits):
        counts, rows = [], []
        for values in planes:
            body = _samples(values, bits)
            for row in range(int(body.shape[0])):
                packed = _packbits(body[row].tobytes())
                counts.append(len(packed))
                rows.append(packed)
        return (
            struct.pack(">H", _RLE)
            + np.array(counts, dtype=">u2").tobytes()
            + b"".join(rows)
        )
    return struct.pack(">H", _RAW) + b"".join(
        _samples(values, bits).tobytes() for values in planes
    )


# ---------------------------------------------------------------------- layer records


def _extra(name: str) -> bytes:
    """One layer record's trailing block: no mask, no ranges, and the name twice over."""
    legacy = name.encode("mac_roman", "replace")[:255]
    pascal = bytes((len(legacy),)) + legacy
    pascal += b"\0" * (-len(pascal) % 4)

    text = name.encode("utf-16-be")
    unicode_name = struct.pack(">I", len(text) // 2) + text
    unicode_name += b"\0" * (-len(unicode_name) % 4)
    block = b"8BIM" + b"luni" + struct.pack(">I", len(unicode_name)) + unicode_name

    return struct.pack(">II", 0, 0) + pascal + block


def _record(plate: Plate, bits: int, packing: str) -> tuple[bytes, bytes]:
    """One layer's record and the channel data that belongs to it."""
    height, width = int(plate.image.shape[0]), int(plate.image.shape[1])
    top, left = int(plate.y), int(plate.x)
    blobs = [
        (_ALPHA, _channel(plate.alpha, bits, packing)),
        (0, _channel(plate.image[..., 0], bits, packing)),
        (1, _channel(plate.image[..., 1], bits, packing)),
        (2, _channel(plate.image[..., 2], bits, packing)),
    ]

    head = struct.pack(">iiii", top, left, top + height, left + width)
    head += struct.pack(">H", len(blobs))
    for ident, blob in blobs:
        head += struct.pack(">hI", ident, len(blob))
    head += b"8BIM" + BLEND_KEYS.get(plate.blend_mode, b"norm")
    head += struct.pack(
        ">BBBB",
        max(0, min(255, int(round(float(plate.opacity) * 255)))),
        0,
        _FLAG_MODERN if plate.visible else _FLAG_MODERN | _FLAG_HIDDEN,
        0,
    )
    extra = _extra(plate.name)
    head += struct.pack(">I", len(extra)) + extra
    return head, b"".join(blob for _ident, blob in blobs)


def _layer_info(plates, bits: int, packing: str, transparent: bool) -> bytes:
    """The layer count, every record and every channel, as one block."""
    records, bodies = [], []
    for plate in plates:
        head, body = _record(plate, bits, packing)
        records.append(head)
        bodies.append(body)
    total = -len(plates) if transparent else len(plates)
    info = struct.pack(">h", total) + b"".join(records) + b"".join(bodies)
    return info + b"\0" * (len(info) % 2)


def _tagged(key: bytes, body: bytes) -> bytes:
    """One additional layer information block, padded to four bytes."""
    padded = body + b"\0" * (-len(body) % 4)
    return b"8BIM" + key + struct.pack(">I", len(padded)) + padded


def _layer_and_mask(info: bytes, bits: int) -> bytes:
    """The whole layer and mask section of a PSD."""
    if bits == 8:
        section = struct.pack(">I", len(info)) + info + struct.pack(">I", 0)
    else:
        section = struct.pack(">II", 0, 0) + _tagged(_LAYER_KEYS[bits], info)
    return struct.pack(">I", len(section)) + section


def _resources() -> bytes:
    """The image resources section, carrying the resolution the document is drawn at."""
    fixed = int(round(_RESOLUTION * 65536))
    body = struct.pack(">IHHIHH", fixed, 1, 1, fixed, 1, 1)
    block = b"8BIM" + struct.pack(">H", 1005) + b"\0\0" + struct.pack(">I", len(body)) + body
    return struct.pack(">I", len(block)) + block


# ---------------------------------------------------------------------- writing


def _prepared(canvas, composite, coverage):
    """Every part a document is assembled from, checked against the canvas it names."""
    width, height = int(canvas[0]), int(canvas[1])
    if width < 1 or height < 1:
        raise ValueError(f"a document cannot be {width}x{height}; both sides must be at least 1")
    if max(width, height) > MAX_SIDE:
        raise ValueError(
            f"a {width}x{height} document is larger than the {MAX_SIDE} pixel limit of this "
            f"format. Reduce the canvas, or write the flattened picture with Image Save"
        )
    if composite.shape[0] != height or composite.shape[1] != width:
        raise ValueError(
            f"the composite is {int(composite.shape[1])}x{int(composite.shape[0])} and the "
            f"canvas is {width}x{height}. They have to match"
        )

    planes = [composite[..., 0], composite[..., 1], composite[..., 2]]
    transparent = coverage is not None and float(coverage.min()) < 1.0
    if transparent:
        planes.append(coverage)
    return width, height, planes, transparent


def _document(plates, canvas, composite, coverage, bits: int, packing: str) -> bytes:
    """A whole PSD file as bytes."""
    width, height, planes, transparent = _prepared(canvas, composite, coverage)
    header = struct.pack(
        ">4sH6xHIIHH", b"8BPS", 1, len(planes), height, width, bits, _RGB
    )
    return b"".join(
        [
            header,
            struct.pack(">I", 0),
            _resources(),
            _layer_and_mask(_layer_info(plates, bits, packing, transparent), bits),
            _merged(planes, bits, packing),
        ]
    )


def _layered_tiff(plates, canvas, composite, coverage, bits: int, packing: str) -> bytes:
    """A whole TIFF file as bytes, carrying the layers beside the flattened picture."""
    from . import tiff

    width, height, planes, transparent = _prepared(canvas, composite, coverage)
    info = _layer_info(plates, bits, packing, transparent)
    block = _TIFF_PREAMBLE + _tagged(_LAYER_KEYS[bits], info)

    built = [_samples(values, bits) for values in planes]
    # Stacking settles on the machine's byte order.
    body = np.stack(built, axis=-1).astype(built[0].dtype).tobytes()
    row = width * len(planes) * (bits // 8)
    strips = []
    for top in range(0, height, _STRIP_ROWS):
        tall = min(_STRIP_ROWS, height - top)
        chunk = body[top * row : (top + tall) * row]
        strips.append(zlib.compress(chunk, 6) if packing != "none" else chunk)

    sample_format = tiff.FLOATING if bits == 32 else tiff.UNSIGNED
    fields = [
        (tiff.WIDTH, tiff.LONG, [width]),
        (tiff.HEIGHT, tiff.LONG, [height]),
        (tiff.BITS_PER_SAMPLE, tiff.SHORT, [bits] * len(planes)),
        (tiff.COMPRESSION, tiff.SHORT, [tiff.NONE if packing == "none" else tiff.DEFLATE]),
        (262, tiff.SHORT, [2]),
        (274, tiff.SHORT, [1]),
        (tiff.SAMPLES_PER_PIXEL, tiff.SHORT, [len(planes)]),
        (tiff.ROWS_PER_STRIP, tiff.LONG, [_STRIP_ROWS]),
        (282, tiff.RATIONAL, [_RESOLUTION]),
        (283, tiff.RATIONAL, [_RESOLUTION]),
        (tiff.PLANAR, tiff.SHORT, [1]),
        (296, tiff.SHORT, [2]),
        (305, tiff.ASCII, "WAS Node Suite"),
        (tiff.SAMPLE_FORMAT, tiff.SHORT, [sample_format] * len(planes)),
        (tiff.IMAGE_SOURCE_DATA, tiff.UNDEFINED, block),
    ]
    if len(planes) == 4:
        # Unassociated alpha.
        fields.append((338, tiff.SHORT, [2]))
    return tiff.build(fields, strips, ">")


def write(
    path,
    plates,
    canvas,
    composite,
    coverage=None,
    depth: str = DEPTHS[0],
    packing: str = PACKINGS[0],
    kind: str = FORMATS[0],
) -> Path:
    """Write a layered document.

    Args:
        path: Where to write. The parent directory must exist.
        plates: The layers, lowest in the stack first.
        canvas: ``(width, height)`` of the document.
        composite: ``(height, width, 3)`` flattened picture codes at the canvas size.
        coverage: ``(height, width)`` coverage of the flattened picture, or None for an
            opaque document.
        depth: One of :data:`DEPTHS`.
        packing: One of :data:`PACKINGS`.
        kind: One of :data:`FORMATS`.

    Returns:
        The path written.

    Raises:
        ValueError: The depth, the packing or the format names nothing known, the canvas
            is empty or larger than :data:`MAX_SIDE`, or the composite is a different size
            from the canvas.
        OSError: The file could not be written.
    """
    bits = bits_of(depth)
    if packing not in PACKINGS:
        raise ValueError(
            f"'{packing}' is not a packing this writes. Pick one of: {', '.join(PACKINGS)}"
        )
    if kind not in FORMATS:
        raise ValueError(
            f"'{kind}' is not a format this writes. Pick one of: {', '.join(FORMATS)}"
        )

    builder = _document if kind == "psd" else _layered_tiff
    data = builder(list(plates), canvas, composite, coverage, bits, packing)
    out = Path(path)
    out.write_bytes(data)
    return out


# ---------------------------------------------------------------------- reading


def _named(extra: bytes) -> str:
    """One layer record's name, preferring the unicode copy of it."""
    at = 0
    for _ in range(2):
        if at + 4 > len(extra):
            return ""
        length = struct.unpack_from(">I", extra, at)[0]
        at += 4 + length
    if at >= len(extra):
        return ""
    start = at
    legacy = extra[at + 1 : at + 1 + extra[at]].decode("mac_roman", "replace")
    at += 1 + extra[at]
    at += -(at - start) % 4

    while at + 12 <= len(extra):
        key = extra[at + 4 : at + 8]
        length = struct.unpack_from(">I", extra, at + 8)[0]
        body = extra[at + 12 : at + 12 + length]
        if key == b"luni" and len(body) >= 4:
            size = struct.unpack_from(">I", body, 0)[0]
            return body[4 : 4 + size * 2].decode("utf-16-be", "replace")
        at += 12 + length + (-length % 2)
    return legacy


def _unpacked(blob: bytes, bits: int, height: int, width: int) -> "torch.Tensor":
    """One stored channel back as a ``(height, width)`` float plane."""
    if height < 1 or width < 1:
        return torch.zeros((max(height, 0), max(width, 0)), dtype=torch.float32)
    packing = struct.unpack_from(">H", blob, 0)[0]
    body = blob[2:]
    stride = width * (bits // 8)

    if packing == _RLE:
        counts = struct.unpack_from(f">{height}H", body, 0)
        at = 2 * height
        rows = []
        for length in counts:
            rows.append(_unpackbits(body[at : at + length], stride))
            at += length
        raw = b"".join(row.ljust(stride, b"\0")[:stride] for row in rows)
    elif packing in (_ZIP, _ZIP_PREDICTED):
        raw = zlib.decompress(body)
        if packing == _ZIP_PREDICTED:
            raw = _unpredicted(raw, bits, height, width)
    else:
        raw = body
    return _floats(raw.ljust(height * stride, b"\0")[: height * stride], bits, height, width)


def _unpredicted(raw: bytes, bits: int, height: int, width: int) -> bytes:
    """A zipped channel with the difference the packer took out put back in."""
    if bits == 8:
        held = np.frombuffer(raw, dtype=np.uint8).reshape(height, width)
        return np.cumsum(held, axis=1, dtype=np.uint64).astype(">u1").tobytes()
    if bits == 16:
        held = np.frombuffer(raw, dtype=">u2").reshape(height, width)
        return np.cumsum(held, axis=1, dtype=np.uint64).astype(">u2").tobytes()
    raise ValueError(
        "this document stores its 32 bit channels with a predictor, which is not read "
        "here. Re-save it from the editor without prediction, or at 16 bit"
    )


def _plates(info: bytes, bits: int) -> list[Plate]:
    """Every raster layer a layer information block holds, lowest in the stack first."""
    total = abs(struct.unpack_from(">h", info, 0)[0])
    at = 2
    heads = []
    for _ in range(total):
        top, left, bottom, right = struct.unpack_from(">iiii", info, at)
        at += 16
        channels = struct.unpack_from(">H", info, at)[0]
        at += 2
        listed = []
        for _ in range(channels):
            ident, length = struct.unpack_from(">hI", info, at)
            at += 6
            listed.append((ident, length))
        at += 4
        key = info[at : at + 4]
        at += 4
        opacity, _clipping, flags, _filler = struct.unpack_from(">BBBB", info, at)
        at += 4
        extra_length = struct.unpack_from(">I", info, at)[0]
        at += 4
        extra = info[at : at + extra_length]
        at += extra_length
        heads.append(
            {
                "box": (left, top, right, bottom),
                "channels": listed,
                "blend_mode": _BLEND_NAMES.get(key, "normal"),
                "opacity": opacity / 255.0,
                "visible": not flags & _FLAG_HIDDEN,
                "name": _named(extra),
            }
        )

    found = []
    for head in heads:
        left, top, right, bottom = head["box"]
        width, height = max(0, right - left), max(0, bottom - top)
        planes = {}
        for ident, length in head["channels"]:
            blob = info[at : at + length]
            at += length
            if width and height and ident >= _ALPHA:
                planes[ident] = _unpacked(blob, bits, height, width)
        if not width or not height or 0 not in planes:
            continue

        colour = torch.stack([planes.get(index, planes[0]) for index in (0, 1, 2)], dim=-1)
        if bits != 32:
            colour = colour.clamp(0.0, 1.0)
        alpha = planes.get(_ALPHA)
        if alpha is None:
            alpha = torch.ones((height, width), dtype=torch.float32)
        found.append(
            Plate(
                name=head["name"],
                x=int(left),
                y=int(top),
                image=colour,
                alpha=alpha.clamp(0.0, 1.0),
                opacity=head["opacity"],
                blend_mode=head["blend_mode"],
                visible=head["visible"],
            )
        )
    return found


def _blocks(data: bytes, at: int, stop: int) -> dict:
    """Every ``8BIM`` tagged block between two offsets, keyed by its four letter key."""
    found = {}
    while at + 12 <= stop:
        if data[at : at + 4] not in (b"8BIM", b"8B64"):
            break
        key = data[at + 4 : at + 8]
        length = struct.unpack_from(">I", data, at + 8)[0]
        found[key] = data[at + 12 : at + 12 + length]
        at += 12 + length + (-length % 4)
    return found


def _read_psd(data: bytes) -> tuple[tuple[int, int], list[Plate], "torch.Tensor | None"]:
    """A PSD's canvas, its layers and its flattened picture."""
    _sign, version, channels, height, width, bits, mode = struct.unpack_from(
        ">4sH6xHIIHH", data, 0
    )
    if version != 1:
        raise ValueError(
            "this is a large document (PSB), and only a PSD is read here. Re-save it as a "
            "PSD from the editor"
        )
    if mode != _RGB:
        raise ValueError(
            f"this document is in colour mode {mode}, and only an RGB one is read here. "
            f"Convert it to RGB in the editor and save it again"
        )
    if bits not in _LAYER_KEYS:
        raise ValueError(f"this document is {bits} bits per channel, which is not read here")

    at = 26
    for _ in range(2):
        at += 4 + struct.unpack_from(">I", data, at)[0]

    section = struct.unpack_from(">I", data, at)[0]
    stop = at + 4 + section
    at += 4
    info_length = struct.unpack_from(">I", data, at)[0]
    at += 4

    if info_length:
        plates = _plates(data[at : at + info_length], bits)
        at += info_length + (info_length % 2)
    else:
        plates = []
    if not plates:
        mask_length = struct.unpack_from(">I", data, at)[0] if at + 4 <= stop else 0
        at += 4 + mask_length
        tagged = _blocks(data, at, stop)
        block = tagged.get(_LAYER_KEYS[bits]) or tagged.get(b"Layr")
        if block:
            plates = _plates(block, bits)

    merged = _merged_picture(data, stop, channels, height, width, bits)
    return (width, height), plates, merged


def _merged_picture(data, at, channels, height, width, bits):
    """A document's flattened picture as ``(height, width, 3)``, or None where unreadable."""
    try:
        packing = struct.unpack_from(">H", data, at)[0]
        body = data[at + 2 :]
        stride = width * (bits // 8)
        planes = []
        if packing == _RLE:
            counts = struct.unpack_from(f">{channels * height}H", body, 0)
            cursor = 2 * channels * height
            for index in range(min(3, channels)):
                rows = []
                for row in range(height):
                    length = counts[index * height + row]
                    rows.append(_unpackbits(body[cursor : cursor + length], stride))
                    cursor += length
                planes.append(
                    _floats(b"".join(rows).ljust(height * stride, b"\0")[: height * stride],
                            bits, height, width)
                )
                if index == 2:
                    break
        elif packing == _RAW:
            for index in range(min(3, channels)):
                start = index * height * stride
                planes.append(_floats(body[start : start + height * stride], bits, height, width))
        else:
            return None
        if len(planes) < 3:
            planes = [planes[0]] * 3 if planes else None
        if not planes:
            return None
        found = torch.stack(planes, dim=-1)
        return found if bits == 32 else found.clamp(0.0, 1.0)
    except (struct.error, ValueError, IndexError):
        return None


def _read_tiff(data: bytes) -> tuple[tuple[int, int], list[Plate], "torch.Tensor | None"]:
    """A layered TIFF's canvas, its layers and its flattened picture."""
    from . import tiff

    _order, fields = tiff.directory(data)
    block = fields.get(tiff.IMAGE_SOURCE_DATA)
    plates = []
    if block is not None:
        body = block[2]
        if body.startswith(_TIFF_PREAMBLE):
            tagged = _blocks(body, len(_TIFF_PREAMBLE), len(body))
            for depth, key in _LAYER_KEYS.items():
                if key in tagged:
                    plates = _plates(tagged[key], depth)
                    break

    width, height, samples, stored, guess, pixels = tiff.plane(data)
    merged = _flat(pixels, width, height, samples, stored, guess)
    return (width, height), plates, merged


def _flat(pixels, width, height, samples, bits, guess):
    """A TIFF's own picture as ``(height, width, 3)`` on a 0 to 1 scale."""
    from . import tiff

    floating = bits == 32 and guess == tiff.FLOATING
    if floating:
        held = np.frombuffer(pixels, dtype=">f4").astype(np.float32)
    elif bits == 16:
        held = np.frombuffer(pixels, dtype=">u2").astype(np.float32) / 65535.0
    elif bits == 8:
        held = np.frombuffer(pixels, dtype=">u1").astype(np.float32) / 255.0
    else:
        return None
    frame = torch.from_numpy(held.reshape(height, width, samples).copy())
    if samples == 1:
        frame = frame.repeat(1, 1, 3)
    frame = frame[..., :3]
    return frame if floating else frame.clamp(0.0, 1.0)


def read(path) -> tuple[tuple[int, int], list[Plate], "torch.Tensor | None"]:
    """Read a layered document.

    Args:
        path: The file to read, a PSD or a TIFF.

    Returns:
        ``(canvas, plates, composite)``. The canvas is ``(width, height)``, the plates are
        the layers lowest in the stack first, and the composite is the flattened picture
        the file stored, or None where it holds none this reads.

    Raises:
        ValueError: The file is neither a PSD nor a TIFF, or it is stored in a way this
            does not read.
        OSError: The file could not be read.
    """
    data = Path(path).read_bytes()
    if data[:4] == b"8BPS":
        return _read_psd(data)
    if data[:2] in (b"II", b"MM"):
        return _read_tiff(data)
    raise ValueError(
        f"{Path(path).name} is neither a Photoshop document nor a TIFF: it begins with "
        f"{data[:4]!r} rather than 8BPS, II or MM"
    )

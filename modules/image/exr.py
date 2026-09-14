"""Linear light read out of and written into OpenEXR files.

Values above one are kept. Channels are stored in the order the format sorts them, alpha
then blue then green then red, at 16 or 32 bits per channel.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path
from typing import NamedTuple

__all__ = [
    "COMPRESSIONS", "DEPTHS", "MAX_CHANNELS", "MAX_PIXELS", "MAX_SIDE", "PACKINGS",
    "Reading", "read", "write",
]

#: What an OpenEXR file opens with, and the version it declares.
MAGIC = 20000630
VERSION = 2

#: Channel sample types, by the code written into the channel list.
UINT, HALF, FLOAT = 0, 1, 2

#: Widget option -> the sample type written.
DEPTHS = {"16 bit half": HALF, "32 bit float": FLOAT}

#: What each sample type is called in a report.
DEPTH_NAMES = {UINT: "32 bit unsigned", HALF: "16 bit half", FLOAT: "32 bit float"}

#: Bytes one sample of each type occupies.
WIDTHS = {UINT: 4, HALF: 2, FLOAT: 4}

#: Compression codes this module names in its own right.
NO_COMPRESSION, RLE, ZIPS, ZIP = 0, 1, 2, 3

#: Every compression the format defines: its name, and the scanlines one block holds.
#: Longest side a data window may name.
MAX_SIDE = 30000

#: Most pixels a data window may cover, whatever the length of its sides.
MAX_PIXELS = 1 << 28

#: Most channels a channel list may name.
MAX_CHANNELS = 1024

#: Bytes of picture one byte of file may describe.
MAX_RATIO = 2048

COMPRESSIONS = {
    NO_COMPRESSION: ("none", 1),
    RLE: ("rle", 1),
    ZIPS: ("zips", 1),
    ZIP: ("zip", 16),
    4: ("piz", 32),
    5: ("pxr24", 16),
    6: ("b44", 32),
    7: ("b44a", 32),
    8: ("dwaa", 32),
    9: ("dwab", 256),
}

#: Widget option -> the compression written.
PACKINGS = {"none": NO_COMPRESSION, "zip": ZIP}

#: The compressions :func:`read` unpacks.
READABLE = frozenset({NO_COMPRESSION, RLE, ZIPS, ZIP})

#: Scanline order, top to bottom and bottom to top.
INCREASING_Y, DECREASING_Y = 0, 1

#: Flags carried in the top three bytes of the version field.
TILED, DEEP, MULTIPART = 0x200, 0x800, 0x1000

#: Colour channels in image order, then the names an alpha and a lone luminance carry.
COLOUR = ("R", "G", "B")
ALPHA = "A"
LUMINANCE = "Y"

#: Where each channel name sits in an RGBA image.
PLANES = {"R": 0, "G": 1, "B": 2, "A": 3}

#: Attributes a header must carry to be read.
REQUIRED = ("channels", "compression", "dataWindow")

#: Bytes one channel list entry occupies after its name.
ENTRY = 16


class Reading(NamedTuple):
    """One image read out of an OpenEXR file.

    Attributes:
        pixels: ``(height, width, 3)`` float32 linear light, unbounded above.
        alpha: ``(height, width)`` float32 coverage, 1 where opaque, or None.
        channels: Every channel name the file holds, in the order it stores them.
        compression: A name from :data:`COMPRESSIONS`.
        depth: A name from :data:`DEPTH_NAMES`, taken from the first colour channel.
    """

    pixels: "torch.Tensor"
    alpha: "torch.Tensor | None"
    channels: tuple[str, ...]
    compression: str
    depth: str


def _string(text: str) -> bytes:
    """One null terminated name."""
    return text.encode("ascii", "replace") + bytes(1)


def _attribute(name: str, kind: str, payload: bytes) -> bytes:
    """One header attribute, named and typed."""
    return _string(name) + _string(kind) + struct.pack("<i", len(payload)) + payload


def _channels(sample: int, names) -> bytes:
    """The channel list, one entry per channel and a terminator."""
    out = bytearray()
    for name in names:
        out += _string(name)
        out += struct.pack("<iBxxxii", sample, 0, 1, 1)
    return bytes(out) + bytes(1)


def _header(width: int, height: int, sample: int, names, packing: int) -> bytes:
    """Every attribute a reader requires, in one block."""
    box = struct.pack("<iiii", 0, 0, width - 1, height - 1)
    return b"".join((
        _attribute("channels", "chlist", _channels(sample, names)),
        _attribute("compression", "compression", bytes((packing,))),
        _attribute("dataWindow", "box2i", box),
        _attribute("displayWindow", "box2i", box),
        _attribute("lineOrder", "lineOrder", bytes((INCREASING_Y,))),
        _attribute("pixelAspectRatio", "float", struct.pack("<f", 1.0)),
        _attribute("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0)),
        _attribute("screenWindowWidth", "float", struct.pack("<f", 1.0)),
    )) + bytes(1)


def _predict(raw: bytes) -> bytes:
    """Split a block into its even and odd bytes, then store each byte as its delta."""
    import torch

    values = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
    half = (values.numel() + 1) // 2
    split = torch.empty_like(values)
    split[:half] = values[0::2]
    split[half:] = values[1::2]
    wide = split.to(torch.int16)
    delta = torch.empty_like(split)
    delta[0] = split[0]
    delta[1:] = ((wide[1:] - wide[:-1] + 128) & 0xFF).to(torch.uint8)
    return delta.numpy().tobytes()


def _unpredict(data: bytes) -> bytes:
    """Undo the delta and the byte split a ZIP or RLE block is stored under."""
    import torch

    values = torch.frombuffer(bytearray(data), dtype=torch.uint8).to(torch.int64)
    values[1:] -= 128
    running = (torch.cumsum(values, dim=0) & 0xFF).to(torch.uint8)
    half = (running.numel() + 1) // 2
    out = torch.empty_like(running)
    out[0::2] = running[:half]
    out[1::2] = running[half:]
    return out.numpy().tobytes()


def _unrle(data: bytes, size: int) -> bytes:
    """Expand one run length encoded block, up to ``size`` bytes."""
    out = bytearray()
    at, end = 0, len(data)
    while at < end and len(out) < size:
        count = data[at]
        at += 1
        if count > 127:
            span = 256 - count
            out += data[at:at + span]
            at += span
        else:
            out += data[at:at + 1] * (count + 1)
            at += 1
    return bytes(out)


def _compress(packing: int, raw: bytes) -> bytes:
    """One block's bytes as they are stored, packed only where packing is smaller."""
    if packing == NO_COMPRESSION:
        return raw
    packed = zlib.compress(_predict(raw))
    return packed if len(packed) < len(raw) else raw


def _decompress(packing: int, payload: bytes, size: int) -> bytes:
    """One block's bytes as they were before packing, up to ``size`` bytes."""
    if packing == RLE:
        return _unpredict(_unrle(payload, size))
    return _unpredict(zlib.decompressobj().decompress(payload, size))


def _attributes(raw: bytes) -> tuple[dict, int]:
    """Every header attribute as ``name -> (type, payload)``, and where the table starts."""
    at = 8
    found = {}
    while at < len(raw) and raw[at] != 0:
        end = raw.index(bytes(1), at)
        name = raw[at:end].decode("ascii", "replace")
        at = end + 1
        end = raw.index(bytes(1), at)
        kind = raw[at:end].decode("ascii", "replace")
        at = end + 1
        size = struct.unpack_from("<i", raw, at)[0]
        at += 4
        if size < 0 or at + size > len(raw):
            raise ValueError(f"the {name!r} attribute declares {size} bytes")
        found[name] = (kind, raw[at:at + size])
        at += size
    return found, at + 1


def _channel_list(payload: bytes) -> list[tuple[str, int, int, int]]:
    """The channel list, as ``(name, sample type, x sampling, y sampling)`` per channel."""
    at = 0
    out = []
    while at < len(payload) and payload[at] != 0:
        end = payload.index(bytes(1), at)
        name = payload[at:end].decode("ascii", "replace")
        at = end + 1
        kind, _linear, across, down = struct.unpack_from("<iBxxxii", payload, at)
        at += ENTRY
        out.append((name, int(kind), int(across), int(down)))
    return out


def _dtype(kind: int):
    """The torch type one sample type is read as."""
    import torch

    return {HALF: torch.float16, FLOAT: torch.float32, UINT: torch.int32}[kind]


def _starts(raw: bytes, table: int, count: int) -> list[int]:
    """Where each block begins, from the offset table or by walking the blocks."""
    first = table + 8 * count
    offsets = [int(one) for one in struct.unpack_from(f"<{count}Q", raw, table)]
    if all(first <= one < len(raw) for one in offsets):
        return offsets
    # An offset table left unwritten is walked instead, since a block names its own size.
    walked, cursor = [], first
    for _ in range(count):
        if cursor + 8 > len(raw):
            break
        walked.append(cursor)
        cursor += 8 + struct.unpack_from("<i", raw, cursor + 4)[0]
    return walked


def write(path, image, depth: str = "16 bit half", packing: str = "none") -> Path:
    """Write one linear image as an OpenEXR file.

    Args:
        path: Where to write. The parent directory must exist.
        image: ``(height, width, 3)`` or ``(height, width, 4)`` tensor holding linear
            light, unbounded above. A fourth channel is written as alpha.
        depth: A key of :data:`DEPTHS`.
        packing: A key of :data:`PACKINGS`.

    Returns:
        The path written.

    Raises:
        ValueError: ``depth`` or ``packing`` names nothing known, or the image is not three
            channel.
    """
    import torch

    sample = DEPTHS.get(depth)
    if sample is None:
        raise ValueError(f"EXR depth must be one of {', '.join(DEPTHS)}, not {depth!r}")
    code = PACKINGS.get(packing)
    if code is None:
        raise ValueError(f"EXR compression must be one of {', '.join(PACKINGS)}, not {packing!r}")
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(
            f"an EXR is written from a (height, width, 3) image and this one is "
            f"{tuple(image.shape)}"
        )

    height, width = int(image.shape[0]), int(image.shape[1])
    names = ("A", "B", "G", "R") if int(image.shape[2]) >= 4 else ("B", "G", "R")
    source = image.to(torch.float16 if sample == HALF else torch.float32).cpu()
    ordered = [source[..., PLANES[name]].contiguous() for name in names]

    header = _header(width, height, sample, names, code)
    per_block = COMPRESSIONS[code][1]
    blocks = []
    for first in range(0, height, per_block):
        rows = min(per_block, height - first)
        raw = b"".join(
            plane[row].numpy().tobytes()
            for row in range(first, first + rows)
            for plane in ordered
        )
        blocks.append((first, _compress(code, raw)))

    table = 8 + len(header) + 8 * len(blocks)
    out = Path(path)
    with out.open("wb") as handle:
        handle.write(struct.pack("<ii", MAGIC, VERSION))
        handle.write(header)
        at = table
        for _row, payload in blocks:
            handle.write(struct.pack("<Q", at))
            at += 8 + len(payload)
        for row, payload in blocks:
            handle.write(struct.pack("<ii", row, len(payload)))
            handle.write(payload)
    return out


def read(path) -> Reading:
    """Read one OpenEXR file as linear light.

    Args:
        path: The file to read.

    Returns:
        A :class:`Reading`.

    Raises:
        ValueError: The file is not a scanline OpenEXR, is packed with a compression this
            reader does not unpack, holds subsampled channels, holds no colour channel, or
            is truncated.
    """
    import torch

    body = Path(path).read_bytes()
    named = Path(path).name
    if len(body) < 8 or struct.unpack_from("<i", body, 0)[0] != MAGIC:
        raise ValueError(
            f"{named} does not open as an OpenEXR file. Check it is an .exr and not a file "
            f"of another format under that name"
        )
    version = struct.unpack_from("<i", body, 4)[0]
    for flag, what in ((TILED, "tiled"), (DEEP, "deep"), (MULTIPART, "multi-part")):
        if version & flag:
            raise ValueError(
                f"{named} is a {what} OpenEXR and a scanline image is read here. Re-export "
                f"it as a flat scanline EXR"
            )

    try:
        found, table = _attributes(body)
    except (struct.error, ValueError, IndexError) as error:
        raise ValueError(
            f"{named} has a header that does not read as an OpenEXR one, so the file is "
            f"truncated or damaged. Write it again from its source"
        ) from error
    missing = [name for name in REQUIRED if name not in found]
    if missing:
        raise ValueError(
            f"{named} carries no {', '.join(missing)} in its header, so it is not a "
            f"readable OpenEXR"
        )

    try:
        left, top, right, bottom = struct.unpack_from("<iiii", found["dataWindow"][1], 0)
        code = found["compression"][1][0]
        channels = _channel_list(found["channels"][1])
    except (struct.error, IndexError, ValueError) as error:
        raise ValueError(
            f"{named} has a header this reader could not follow, so the file is truncated "
            f"or damaged. Write it again from its source"
        ) from error
    width, height = right - left + 1, bottom - top + 1
    if width < 1 or height < 1:
        raise ValueError(f"{named} declares a {width} by {height} data window, which holds no pixels")
    if max(width, height) > MAX_SIDE:
        raise ValueError(
            f"{named} declares a {width} by {height} data window, longer than the "
            f"{MAX_SIDE} pixel limit of this reader, so the file was not read"
        )
    if width * height > MAX_PIXELS:
        raise ValueError(
            f"{named} declares a {width} by {height} data window of {width * height} "
            f"pixels, more than the {MAX_PIXELS} this reader unpacks, so the file was "
            f"not read"
        )
    if len(channels) > MAX_CHANNELS:
        raise ValueError(
            f"{named} names {len(channels)} channels, more than the {MAX_CHANNELS} this "
            f"reader unpacks, so the file was not read"
        )

    packing, per_block = COMPRESSIONS.get(code, (f"code {code}", 1))
    if code not in READABLE:
        unpacked = ", ".join(COMPRESSIONS[one][0] for one in sorted(READABLE))
        raise ValueError(
            f"{named} is compressed with {packing}, and {unpacked} are unpacked here. "
            f"Re-export it with ZIP compression or with none"
        )

    unknown = [name for name, kind, _x, _y in channels if kind not in WIDTHS]
    if unknown:
        raise ValueError(
            f"{named} stores {', '.join(unknown)} as a sample type this reader does not "
            f"know. Re-export it as half or float"
        )
    thinned = [name for name, _k, across, down in channels if across != 1 or down != 1]
    if thinned:
        raise ValueError(
            f"{named} stores {', '.join(thinned)} subsampled, and full resolution channels "
            f"are read here. Re-export it as RGB or RGBA"
        )

    held = tuple(name for name, _k, _x, _y in channels)
    if all(one in held for one in COLOUR):
        wanted = list(COLOUR)
    elif LUMINANCE in held:
        wanted = [LUMINANCE]
    else:
        raise ValueError(
            f"{named} holds {', '.join(held) or 'no channels'}, and an image is read from R, "
            f"G and B or from Y. Re-export it as RGB or RGBA"
        )
    if ALPHA in held:
        wanted.append(ALPHA)

    per_row = sum(width * WIDTHS[kind] for _n, kind, _x, _y in channels)
    count = -(-height // per_block)
    damaged = (
        f"{named} could not be unpacked, so the file is truncated or damaged. Write it "
        f"again from its source"
    )
    carried = len(body) - (table + 8 * count)
    if carried < (height * per_row) // MAX_RATIO:
        raise ValueError(
            f"{named} describes {height * per_row} bytes of pixels and carries "
            f"{max(0, carried)} bytes to unpack them from, so its header does not "
            f"describe the file. Write it again from its source"
        )
    try:
        starts = _starts(body, table, count)
    except struct.error as error:
        raise ValueError(damaged) from error
    if len(starts) != count:
        raise ValueError(damaged)
    gathered = {name: torch.zeros(height, width, dtype=torch.float32) for name in wanted}

    for start in starts:
        try:
            row, size = struct.unpack_from("<ii", body, start)
            rows = min(per_block, top + height - row)
            if rows < 1:
                continue
            expected = rows * per_row
            payload = body[start + 8:start + 8 + size]
            raw = payload if size >= expected else _decompress(code, payload, expected)
            if len(raw) < expected:
                raise ValueError(
                    f"{named} unpacked scanline {row} to {len(raw)} bytes instead of "
                    f"{expected}, so the file is truncated or damaged. Write it again from "
                    f"its source"
                )
            block = torch.frombuffer(
                bytearray(raw[:expected]), dtype=torch.uint8
            ).view(rows, per_row)
        except (struct.error, zlib.error, IndexError) as error:
            raise ValueError(damaged) from error
        column = 0
        for name, kind, _x, _y in channels:
            span = width * WIDTHS[kind]
            if name in gathered:
                strip = block[:, column:column + span].contiguous().view(_dtype(kind))
                if kind == UINT:
                    strip = strip.to(torch.int64) & 0xFFFFFFFF
                gathered[name][row - top:row - top + rows] = strip.float()
            column += span

    if LUMINANCE in gathered:
        pixels = gathered[LUMINANCE].unsqueeze(-1).repeat(1, 1, 3)
    else:
        pixels = torch.stack([gathered[name] for name in COLOUR], dim=-1)
    kinds = {name: kind for name, kind, _x, _y in channels}
    return Reading(
        pixels=pixels,
        alpha=gathered.get(ALPHA),
        channels=held,
        compression=packing,
        depth=DEPTH_NAMES.get(kinds[wanted[0]], "mixed"),
    )

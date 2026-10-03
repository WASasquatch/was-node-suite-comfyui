"""Reading 16-bit PNG files at full precision.

Pixels decode through PyAV as float32 ``(height, width, channels)`` in ``[0, 1]``, with no
transfer curve applied. The file's ``gAMA``, ``sRGB``, ``cICP`` and ``iCCP`` chunks are
read to say what the numbers mean.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass

from .. import log

logger = log.get_logger("image.deep_png")

__all__ = ["Header", "Picture", "header", "is_deep", "read", "tensors"]

#: What every PNG file opens with.
SIGNATURE = b"\x89PNG\r\n\x1a\n"

#: Colour type to (channel count, carries alpha).
COLOUR_TYPES = {0: (1, False), 2: (3, False), 3: (3, False), 4: (2, True), 6: (4, True)}

#: ``gAMA`` value of a linear file, as the chunk stores it.
LINEAR_GAMMA = 100000

#: ``cICP`` transfer characteristics that are linear.
LINEAR_CICP = frozenset({8})

#: EXIF orientation tag.
ORIENTATION = 0x0112

#: Longest chunk the format allows.
MAX_CHUNK = (1 << 31) - 1

#: Leading bytes of a chunk the header reads, the length of an ``IHDR``.
HEAD_BYTES = 13


@dataclass
class Header:
    """What a PNG's leading chunks say about it.

    Attributes:
        width: Pixels across.
        height: Pixels down.
        depth: Bits a sample.
        colour_type: The IHDR colour type.
        alpha: Whether the pixels carry transparency.
        transfer: ``linear``, ``sRGB``, ``gamma 2.20`` and the like, or empty when the file
            does not say.
        profile: Whether an ICC profile is embedded.
    """

    width: int
    height: int
    depth: int
    colour_type: int
    alpha: bool
    transfer: str = ""
    profile: bool = False

    @property
    def linear(self) -> bool:
        """Whether the file marks its numbers as linear light."""
        return self.transfer == "linear"


@dataclass
class Picture:
    """One decoded 16-bit PNG.

    Attributes:
        pixels: ``(height, width, channels)`` float32 in ``[0, 1]``, one to four channels.
        header: What the file said about itself.
    """

    pixels: object
    header: Header


def header(path) -> Header | None:
    """Read the chunks ahead of the pixel data.

    Args:
        path: A file path.

    Returns:
        A :class:`Header`, or None when the file is not a PNG. A chunk longer than
        :data:`MAX_CHUNK` or than the rest of the file ends the walk with what came before it.
    """
    try:
        with open(path, "rb") as handle:
            if handle.read(8) != SIGNATURE:
                return None
            end = os.fstat(handle.fileno()).st_size
            found = None
            while True:
                head = handle.read(8)
                if len(head) < 8:
                    return found
                size, kind = struct.unpack(">I4s", head)
                if kind == b"IDAT" or kind == b"IEND":
                    return found
                if size > MAX_CHUNK or size > end - handle.tell():
                    return found
                body = handle.read(min(size, HEAD_BYTES))
                handle.seek(size - len(body) + 4, 1)
                if kind == b"IHDR" and len(body) >= 10:
                    width, height, depth, colour = struct.unpack(">IIBB", body[:10])
                    found = Header(width, height, depth, colour, COLOUR_TYPES.get(colour, (3, False))[1])
                elif found is None:
                    return None
                elif kind == b"cICP" and len(body) >= 2:
                    found.transfer = "linear" if body[1] in LINEAR_CICP else (found.transfer or f"cicp {body[1]}")
                elif kind == b"sRGB" and found.transfer != "linear":
                    found.transfer = "sRGB"
                elif kind == b"gAMA" and len(body) >= 4 and not found.transfer:
                    stored = struct.unpack(">I", body[:4])[0]
                    found.transfer = "linear" if stored == LINEAR_GAMMA else (
                        f"gamma {100000 / stored:.2f}" if stored else ""
                    )
                elif kind == b"iCCP":
                    found.profile = True
    except OSError:
        return None


def is_deep(path) -> bool:
    """Whether a file is a PNG storing 16 bits a sample.

    Args:
        path: A file path.

    Returns:
        True for a 16-bit PNG.
    """
    found = header(path)
    return bool(found and found.depth == 16)


def _orientation(path) -> int:
    """The EXIF orientation a file carries, 1 when it carries none."""
    try:
        from PIL import Image

        with Image.open(path) as opened:
            return int(opened.getexif().get(ORIENTATION, 1) or 1)
    except Exception:
        return 1


def _oriented(pixels, orientation: int):
    """Pixels turned the way an EXIF orientation says, as Pillow's ``exif_transpose`` does."""
    import numpy as np

    if orientation == 2:
        return pixels[:, ::-1]
    if orientation == 3:
        return pixels[::-1, ::-1]
    if orientation == 4:
        return pixels[::-1]
    if orientation == 5:
        return np.transpose(pixels, (1, 0, 2))
    if orientation == 6:
        return np.rot90(pixels, -1)
    if orientation == 7:
        return np.rot90(np.transpose(pixels, (1, 0, 2)), 2)
    if orientation == 8:
        return np.rot90(pixels, 1)
    return pixels


def read(path) -> Picture:
    """Decode a 16-bit PNG at full precision.

    Args:
        path: A file path, already contained by the caller.

    Returns:
        A :class:`Picture` holding one channel for grey, two for grey with alpha, three for
        colour and four for colour with alpha, turned by any EXIF orientation.

    Raises:
        ValueError: The file is not a PNG, or could not be decoded.
    """
    import av
    import numpy as np

    found = header(path)
    if found is None:
        raise ValueError(f"`{path}` is not a PNG file")
    try:
        with av.open(str(path)) as container:
            frame = next(container.decode(video=0))
            rgba = frame.to_ndarray(format="rgba64le").astype(np.float32) / 65535.0
    except Exception as error:
        raise ValueError(f"`{path}` could not be decoded as a 16-bit PNG ({error})") from error
    channels = COLOUR_TYPES.get(found.colour_type, (3, False))[0]
    if channels == 1:
        pixels = rgba[..., :1]
    elif channels == 2:
        pixels = rgba[..., [0, 3]]
    elif channels == 3:
        pixels = rgba[..., :3]
    else:
        pixels = rgba
    pixels = np.ascontiguousarray(_oriented(pixels, _orientation(path)))
    return Picture(pixels=pixels, header=found)


def tensors(picture: Picture, rgba: bool = False):
    """The image and mask a loader answers with.

    Args:
        picture: What :func:`read` returned.
        rgba: Keep transparency as a fourth channel.

    Returns:
        ``(image, mask)``: a ``(1, height, width, 3 or 4)`` float32 tensor and a
        ``(height, width)`` mask, 1 where the picture is transparent. A picture with no alpha
        answers a 64 by 64 empty mask, as ComfyUI's own loader does.
    """
    import numpy as np
    import torch

    pixels = picture.pixels
    colour = np.repeat(pixels[..., :1], 3, axis=2) if pixels.shape[2] in (1, 2) else pixels[..., :3]
    alpha = pixels[..., -1] if picture.header.alpha else None
    image = np.concatenate([colour, alpha[..., None]], axis=2) if rgba and alpha is not None else colour
    if alpha is not None:
        mask = torch.from_numpy(np.ascontiguousarray(1.0 - alpha))
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32)
    return torch.from_numpy(np.ascontiguousarray(image))[None], mask

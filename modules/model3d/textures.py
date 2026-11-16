"""Texture images inside 3D files: decoding, encoding, and reading the files a model names.

Images decode to `scene.Texture` tensors in ``[0, 1]``. A size limit is checked against the
header before any pixel is decoded.
"""

from __future__ import annotations

__all__ = ["MAX_SIDECAR_BYTES", "MAX_TEXTURE_PIXELS", "decode", "encode", "mime_of", "read_sidecar"]

import io
import os
from pathlib import Path

import numpy as np
import torch

from ..image.sizing import ImageTooLarge, NotAnImage, open_bytes
from ..log import get_logger
from ..util import sandbox
from .scene import Texture

logger = get_logger("model3d.textures")

#: Most pixels one texture may hold: an 8K square.
MAX_TEXTURE_PIXELS = 8192 * 8192

#: Largest file a model may name beside itself.
MAX_SIDECAR_BYTES = 256 * 1024 * 1024

#: Leading bytes of each encoding, against its media type.
_SIGNATURES = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"RIFF", "image/webp"),
    (b"BM", "image/bmp"),
)


def mime_of(data: bytes, fallback: str = "image/png") -> str:
    """The media type an image's leading bytes declare.

    Args:
        data: The encoded image.
        fallback: Returned when no signature matches.

    Returns:
        A media type such as ``"image/jpeg"``.
    """
    for signature, mime in _SIGNATURES:
        if data.startswith(signature):
            return mime
    return fallback


def decode(data: bytes, name: str = "", keep_alpha: bool = True) -> Texture | None:
    """An encoded image as a texture.

    Args:
        data: PNG, JPEG, WebP, BMP or any other format Pillow reads.
        name: Name carried onto the texture.
        keep_alpha: Keep a fourth channel when the image has meaningful alpha.

    Returns:
        The texture, or None when the bytes are not a readable image or are too large. The
        reason is logged.
    """
    try:
        image = open_bytes(data, max_pixels=MAX_TEXTURE_PIXELS)
    except (NotAnImage, ImageTooLarge) as error:
        logger.warning("texture %s was skipped: it %s", name or "(unnamed)", error)
        return None
    array = np.asarray(image, dtype=np.uint8)
    if not keep_alpha or bool((array[..., 3] == 255).all()):
        array = array[..., :3]
    tensor = torch.from_numpy(array.astype(np.float32) / 255.0)
    return Texture(image=tensor, name=name, mime=mime_of(data))


def encode(texture: Texture, mime: str | None = None, quality: int = 92) -> tuple[bytes, str]:
    """A texture encoded for writing into a file.

    Args:
        texture: The texture to encode.
        mime: ``"image/png"`` or ``"image/jpeg"``; None uses the texture's own preference.
            JPEG is replaced by PNG when the texture carries alpha.
        quality: JPEG quality.

    Returns:
        ``(bytes, mime)`` actually written.
    """
    from PIL import Image

    array = (texture.image.detach().float().clamp(0, 1).cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    channels = array.shape[2] if array.ndim == 3 else 1
    chosen = mime or texture.mime or "image/png"
    if chosen not in ("image/png", "image/jpeg") or (chosen == "image/jpeg" and channels == 4):
        chosen = "image/png"
    mode = {1: "L", 3: "RGB", 4: "RGBA"}.get(channels, "RGB")
    if array.ndim == 3 and channels == 1:
        array = array[..., 0]
    image = Image.fromarray(array, mode=mode)
    buffer = io.BytesIO()
    if chosen == "image/jpeg":
        image.save(buffer, format="JPEG", quality=int(quality))
    else:
        image.save(buffer, format="PNG", compress_level=6)
    return buffer.getvalue(), chosen


def read_sidecar(base_dir: str | os.PathLike | None, name: str) -> bytes | None:
    """The bytes of a file a model names beside itself, such as a texture or a buffer.

    Args:
        base_dir: Folder holding the model, or None when the model came from memory.
        name: The reference as the model writes it, already URI-decoded.

    Returns:
        The file's bytes, or None when there is no folder, the name leaves it, the file is
        missing, is not a file, cannot be read, or is larger than `MAX_SIDECAR_BYTES`. The
        reason is logged.
    """
    if base_dir is None or not str(name).strip():
        return None
    try:
        path = sandbox.resolve_read_file(base_dir, name)
    except sandbox.PathNotAllowed as error:
        logger.warning("a file the model names was skipped: %s", error)
        return None
    shown = Path(name).name or str(name).strip()
    try:
        size = Path(path).stat().st_size
    except OSError:
        logger.warning("a file the model names, %s, is missing", shown)
        return None
    if not Path(path).is_file():
        logger.warning("a file the model names, %s, is not a file, so it was skipped", shown)
        return None
    if size > MAX_SIDECAR_BYTES:
        logger.warning("%s is %d MB, more than a model's files may hold, so it was skipped",
                       shown, size // (1024 * 1024))
        return None
    try:
        with open(path, "rb") as handle:
            data = handle.read(min(size, MAX_SIDECAR_BYTES) + 1)
    except OSError as error:
        logger.warning("a file the model names, %s, could not be read (%s)", shown,
                       error.strerror)
        return None
    return data if len(data) <= MAX_SIDECAR_BYTES else None

"""Reading and writing whole 3D model files by format: GLB, glTF, VRM, FBX and OBJ.

Formats are named by id: ``"glb"``, ``"gltf"``, ``"vrm"``, ``"fbx"``, ``"obj"``. Every model read
carries its format in ``source_format`` and its file name in ``meta["source_name"]``.
"""

from __future__ import annotations

__all__ = [
    "FORMATS", "MAX_MODEL_BYTES", "MAX_STEM_CHARS", "SUFFIXES", "check_stem", "encode", "load_bytes",
    "load_file3d", "load_path", "save", "sniff",
]

import importlib.util
import json
import os
import struct
from pathlib import Path, PurePosixPath

from ..log import get_logger
from ..util import sandbox
from .scene import Model3D

logger = get_logger("model3d.files")

#: File extensions a model is read from.
SUFFIXES = (".glb", ".gltf", ".vrm", ".fbx", ".obj")

#: Format id against what it is.
FORMATS = {
    "glb": "Binary glTF, one file with its textures inside",
    "gltf": "glTF text, with its buffer and textures beside it",
    "vrm": "VRM 1.0 avatar, a binary glTF with humanoid bones, expressions and spring bones",
    "fbx": "Autodesk FBX, binary",
    "obj": "Wavefront OBJ with its MTL material library and textures beside it",
}

#: Largest model file read.
MAX_MODEL_BYTES = 512 * 1024 * 1024

#: Largest JSON chunk read while telling a glTF from a VRM.
MAX_SNIFF_JSON_BYTES = 64 * 1024 * 1024

#: Bytes of a text file looked at while telling OBJ from other text.
SNIFF_TEXT_BYTES = 64 * 1024

#: First words of the statements an OBJ file is made of.
_OBJ_WORDS = frozenset({
    b"v", b"vt", b"vn", b"vp", b"f", b"l", b"p", b"o", b"g", b"s", b"usemtl", b"mtllib",
    b"cstype", b"deg", b"curv", b"curv2", b"surf", b"parm", b"end", b"mg", b"bevel",
    b"c_interp", b"d_interp", b"lod", b"shadow_obj", b"trace_obj",
})

#: Longest file stem written, in characters.
MAX_STEM_CHARS = 180

#: Extension names that mark a glTF file as a VRM avatar.
_VRM_EXTENSIONS = ("VRMC_vrm", "VRM")

#: Characters no file name may hold on Windows.
_FORBIDDEN_NAME_CHARS = frozenset('<>:"/\\|?*')


def _has_vrm_module() -> bool:
    """Whether the VRM reader and writer are installed beside this module."""
    return importlib.util.find_spec(f"{__package__}.vrm") is not None


def _is_vrm(document: dict) -> bool:
    """Whether a glTF JSON document is a VRM avatar."""
    if _has_vrm_module():
        from . import vrm
        try:
            return bool(vrm.is_vrm(document))
        except (AttributeError, TypeError, ValueError):
            return False
    used = document.get("extensionsUsed")
    extensions = document.get("extensions")
    used = used if isinstance(used, list) else []
    extensions = extensions if isinstance(extensions, dict) else {}
    return any(name in used or name in extensions for name in _VRM_EXTENSIONS)


def check_stem(stem: str) -> str:
    """A file stem confirmed to be one plain file name.

    Args:
        stem: File name without extension.

    Returns:
        The stem, unchanged.

    Raises:
        ValueError: The stem is empty, too long, holds a folder, a drive, a control character or
            a character Windows refuses, or names a Windows device.
    """
    text = str(stem) if stem is not None else ""
    if not text.strip():
        raise ValueError("the file name is empty; type a name for the model file")
    if len(text) > MAX_STEM_CHARS:
        raise ValueError(f"the file name is {len(text)} characters long, more than {MAX_STEM_CHARS}; "
                         "type a shorter name")
    bad = sorted({c for c in text if c in _FORBIDDEN_NAME_CHARS or ord(c) < 32 or ord(c) == 127})
    if bad:
        shown = " ".join(repr(c) for c in bad[:6])
        raise ValueError(f"the file name {text[:60]!r} holds {shown}, which a file name cannot "
                         "hold; type a plain name with no folder in it")
    reason = sandbox.leaves_folder(text)
    if reason is not None or text.strip(" .") == "":
        raise ValueError(f"the file name {text[:60]!r} {reason or 'is only dots'}; type a plain "
                         "name with no folder in it")
    return text


def _glb_json(data: bytes) -> dict | None:
    """The JSON chunk of a GLB file, or None when it cannot be read."""
    if len(data) < 20:
        return None
    length, kind = struct.unpack_from("<II", data, 12)
    if kind != 0x4E4F534A or length > MAX_SNIFF_JSON_BYTES or 20 + length > len(data):
        return None
    try:
        document = json.loads(data[20:20 + length].decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def _looks_like_obj(head: bytes) -> bool:
    """Whether the start of a text file reads as OBJ statements."""
    statements = known = vertices = 0
    for line in head.splitlines()[:400]:
        words = line.split()
        if not words or words[0].startswith(b"#"):
            continue
        statements += 1
        if words[0] in _OBJ_WORDS:
            known += 1
            vertices += words[0] in (b"v", b"f")
    return vertices > 0 and known >= max(1, int(statements * 0.9))


def sniff(data: bytes, suffix: str = "") -> str:
    """The format of a model file, from its bytes, then from its extension.

    Args:
        data: The file's bytes, or at least its start.
        suffix: The file's extension, with or without the dot, used when the bytes decide nothing.

    Returns:
        A key of `FORMATS`, or ``""`` when neither the bytes nor the extension name one.
    """
    if data[:4] == b"glTF":
        document = _glb_json(data)
        return "vrm" if document is not None and _is_vrm(document) else "glb"
    if data[:18] == b"Kaydara FBX Binary":
        return "fbx"
    head = data[:SNIFF_TEXT_BYTES]
    if head.startswith(b"\xef\xbb\xbf"):
        head = head[3:]
    stripped = head.lstrip()
    if stripped[:1] == b"{":
        if len(data) <= MAX_SNIFF_JSON_BYTES:
            try:
                document = json.loads(data.decode("utf-8-sig"))
            except (UnicodeDecodeError, ValueError):
                document = None
            if isinstance(document, dict) and "asset" in document:
                return "vrm" if _is_vrm(document) else "gltf"
        elif b'"asset"' in head:
            return "gltf"
    if stripped[:5] == b"; FBX" or b"FBXHeaderExtension:" in head:
        return "fbx"
    if b"\x00" not in head and _looks_like_obj(head):
        return "obj"
    chosen = suffix.lower().lstrip(".")
    return chosen if chosen in FORMATS else ""


def _reader(fmt: str):
    """The ``read(data, base_dir, name)`` function of a format.

    Raises:
        ValueError: The format is not one of `FORMATS`.
    """
    if fmt in ("glb", "gltf") or (fmt == "vrm" and not _has_vrm_module()):
        from . import gltf as module
    elif fmt == "vrm":
        from . import vrm as module
    elif fmt == "fbx":
        from . import fbx as module
    elif fmt == "obj":
        from . import obj as module
    else:
        raise ValueError(f"'{fmt}' is not a 3D model format this pack reads; use one of "
                         f"{', '.join(sorted(FORMATS))}")
    return module.read


def _check_size(size: int, name: str) -> None:
    """Refuse a model file past `MAX_MODEL_BYTES`.

    Raises:
        ValueError: The file is too large.
    """
    if size > MAX_MODEL_BYTES:
        raise ValueError(f"{name or 'the model'} is {size // (1024 * 1024)} MB, more than the "
                         f"{MAX_MODEL_BYTES // (1024 * 1024)} MB a model file may be; decimate it "
                         "or split it into several files")


def _leaf(name: str) -> str:
    """The file name of a path, with no folder."""
    return PurePosixPath(str(name).replace("\\", "/")).name


def load_bytes(data: bytes, fmt: str, name: str = "", base_dir: str | None = None) -> Model3D:
    """A model read from bytes in memory.

    Args:
        data: The file's bytes.
        fmt: A key of `FORMATS`, an extension, or ``""`` to tell it from the bytes. A VRM, FBX or
            glTF signature in the bytes wins over the name given.
        name: File name, carried into ``meta["source_name"]``; any folder is dropped.
        base_dir: Folder holding the file, where the files it names beside itself are read, or
            None.

    Returns:
        The model.

    Raises:
        ValueError: The bytes are too large, empty, not a model, or damaged.
    """
    leaf = _leaf(name) if name else ""
    _check_size(len(data), leaf)
    if not data:
        raise ValueError(f"{leaf or 'the model file'} is empty; pick or wire a model file with content")
    given = str(fmt or "").lower().lstrip(".")
    sniffed = sniff(data, given or Path(leaf).suffix)
    chosen = sniffed or given
    if not chosen:
        raise ValueError(f"{leaf or 'the file'} is not a 3D model this pack reads; use a "
                         f"{', '.join(sorted(FORMATS))} file")
    model = _reader(chosen)(data, base_dir, leaf)
    model.source_format = chosen
    if leaf:
        model.meta["source_name"] = leaf
    else:
        model.meta.pop("source_name", None)
    return model


def load_path(path: str | os.PathLike, fmt: str = "") -> Model3D:
    """A model read from a file inside the folders the pack may read.

    Args:
        path: The file path.
        fmt: A key of `FORMATS`, or ``""`` to tell it from the file.

    Returns:
        The model, with the files it names read from its folder.

    Raises:
        sandbox.PathNotAllowed: The file lies outside every folder the pack may read.
        ValueError: The file is missing, too large, not a model, or damaged.
    """
    path = sandbox.resolve_read(path)
    try:
        size = path.stat().st_size
    except OSError as error:
        raise ValueError(f"{path.name} could not be opened ({error.strerror}); pick a model that "
                         "exists") from None
    if not path.is_file():
        raise ValueError(f"{path.name} is not a file; pick a model file")
    _check_size(size, path.name)
    try:
        with open(path, "rb") as handle:
            data = handle.read(MAX_MODEL_BYTES + 1)
    except OSError as error:
        raise ValueError(f"{path.name} could not be read ({error.strerror}); check the file is "
                         "not open in another program") from None
    return load_bytes(data, fmt or path.suffix, name=path.name, base_dir=str(path.parent))


def load_file3d(file3d) -> Model3D:
    """A model read from core's File3D, the value a 3D file socket carries.

    Args:
        file3d: A ``File3D``: a file on disk, or bytes in memory.

    Returns:
        The model. A file on disk has the files it names read from its folder.

    Raises:
        sandbox.PathNotAllowed: The file lies outside the folders the pack may read.
        ValueError: The file is too large, cannot be read, or is not a model.
    """
    fmt = str(getattr(file3d, "format", "") or "").lower().lstrip(".")
    if getattr(file3d, "is_disk_backed", False):
        return load_path(str(file3d.get_source()), fmt)
    source = file3d.get_source()
    if isinstance(source, (bytes, bytearray, memoryview)):
        data = bytes(source[:MAX_MODEL_BYTES + 1])
    elif hasattr(source, "read"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            data = source.read(MAX_MODEL_BYTES + 1)
        except (OSError, ValueError) as error:
            reason = getattr(error, "strerror", None) or type(error).__name__
            raise ValueError(f"the wired 3D file could not be read ({reason}); run the node that "
                             "made it again") from None
    else:
        raise ValueError("the wired 3D file holds neither a file nor bytes; run the node that made "
                         "it again")
    _check_size(len(data), "the wired model")
    return load_bytes(bytes(data), fmt, name=f"model.{fmt}" if fmt in FORMATS else "")


def _fbx_beside(model: Model3D, stem: str) -> dict[str, bytes]:
    """An FBX naming its textures, and the texture files, each named after ``stem``."""
    from . import fbx

    renamed: dict[int, tuple] = {}
    for part in model.parts:
        for texture in part.material.textures().values():
            if id(texture) not in renamed:
                renamed[id(texture)] = (texture, texture.name)
    try:
        for number, (texture, _) in enumerate(renamed.values()):
            texture.name = f"{stem}_texture_{number}"
        produced = {f"{stem}.fbx": fbx.write(model, embed_textures=False)}
        produced.update(fbx.texture_files(model))
    finally:
        for texture, name in renamed.values():
            texture.name = name
    return produced


def encode(model: Model3D, fmt: str, stem: str, embed_textures: bool | None = None) -> dict[str, bytes]:
    """A model as the files one format writes.

    Args:
        model: The model.
        fmt: A key of `FORMATS`.
        stem: File name without extension.
        embed_textures: True puts textures inside a glTF or FBX file, False writes them beside it,
            None takes each format's own way (beside a glTF, inside an FBX). GLB and VRM always
            hold their textures; OBJ never does.

    Returns:
        File name to bytes, the model file first.

    Raises:
        ValueError: The format is unknown or the stem is not a plain file name.
    """
    fmt = str(fmt).lower().lstrip(".")
    if fmt not in FORMATS:
        raise ValueError(f"'{fmt}' is not a 3D model format this pack writes; use one of "
                         f"{', '.join(sorted(FORMATS))}")
    stem = check_stem(stem)
    if fmt == "glb":
        from . import gltf
        return {f"{stem}.glb": gltf.write(model, binary=True)}
    if fmt == "gltf":
        from . import gltf
        if embed_textures:
            return {f"{stem}.gltf": gltf.write(model, binary=False)}
        return gltf.build(model, embed_images=False).to_gltf(stem)
    if fmt == "vrm":
        if not _has_vrm_module():
            raise ValueError("this install has no VRM writer; save the model as glb instead")
        from . import vrm
        return {f"{stem}.vrm": vrm.write(model)}
    if fmt == "fbx":
        if embed_textures is False:
            return _fbx_beside(model, stem)
        from . import fbx
        return {f"{stem}.fbx": fbx.write(model)}
    from . import obj
    return obj.write(model, stem)


def save(model: Model3D, fmt: str, directory: str | os.PathLike, stem: str,
         overwrite: bool = False, embed_textures: bool | None = None,
         input_folder: bool = False) -> list[Path]:
    """Write a model into a folder the pack may write to.

    Args:
        model: The model.
        fmt: A key of `FORMATS`.
        directory: The destination folder, already resolved.
        stem: File name without extension, shared by every file written.
        overwrite: Replace files already there; otherwise an existing file is refused.
        embed_textures: Where textures go, as `encode` takes it.
        input_folder: Also allow ComfyUI's input folder and the read folders, as picture saves do.

    Returns:
        Every path written, the model file first.

    Raises:
        ValueError: The format is unknown, the stem is not a plain file name, or a file could not
            be written.
        sandbox.PathNotAllowed: The folder is outside every folder the pack may write to.
        FileExistsError: A file of the same name is there and ``overwrite`` is off.
    """
    produced = encode(model, fmt, stem, embed_textures=embed_textures)
    if input_folder:
        targets = [(sandbox.resolve_save_file(directory, name), data) for name, data in produced.items()]
    else:
        targets = [(sandbox.resolve_write_file(directory, name), data) for name, data in produced.items()]
    if not overwrite:
        for target, _ in targets:
            if target.exists():
                raise FileExistsError(f"{target.name} is already there; choose another name or "
                                      "turn overwrite on")
    written: list[Path] = []
    for target, data in targets:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as handle:
                handle.write(data)
        except OSError as error:
            raise ValueError(f"{target.name} could not be written ({error.strerror}); check the "
                             "disk has room and the folder is not read-only") from None
        written.append(target)
    logger.info("wrote %s as %s: %d file(s)", f"{stem}.{fmt}", fmt, len(written))
    return written

"""FBX files: a reader for binary and ASCII FBX 7 and a writer for binary FBX 7.4, onto `scene.Model3D`.

The reader converts axes and units to metres, Y up, facing +Z. The writer stores centimetres
with the same axes.
"""

from __future__ import annotations

__all__ = ["FbxError", "read", "texture_files", "write"]

import base64
import math
import os
import re
import struct
import zlib
from collections import defaultdict
from pathlib import PurePosixPath
from typing import Any

import numpy as np
import torch

from ..log import get_logger
from ..util import sandbox
from . import textures as texture_io
from .scene import (
    Animation, Channel, Material, Model3D, MorphTarget, Part, Polygons, Skeleton, Texture,
)

logger = get_logger("model3d.fbx")

#: Most bytes an FBX file may hold.
MAX_FILE_BYTES = 2 * 1024 * 1024 * 1024

#: Most bytes one decoded array may hold.
MAX_ARRAY_BYTES = 1024 * 1024 * 1024

#: Most bytes every decoded array of a file may hold together.
MAX_DECODED_BYTES = 3 * 1024 * 1024 * 1024

#: Decoded bytes allowed per byte of the file, above a 256 MB floor.
DECODED_PER_FILE_BYTE = 200

#: Most bytes one zlib stream may inflate to per compressed byte.
MAX_INFLATE_RATIO = 1100

#: Most records a file may hold.
MAX_NODES = 8_000_000

#: Deepest record nesting read.
MAX_DEPTH = 64

#: Most control points or polygon corners one mesh may hold.
MAX_MESH_ELEMENTS = 100_000_000

#: Most frames times joints one animation is resampled to; a longer clip is resampled more sparsely.
MAX_ANIMATION_SAMPLES = 2_500_000

#: Most frames one animation is resampled to.
MAX_ANIMATION_FRAMES = 36_000

#: Deepest model hierarchy walked.
MAX_HIERARCHY_DEPTH = 4096

#: Corner count above which a polygon is fanned instead of ear clipped.
MAX_EAR_CLIP_CORNERS = 512

#: Most squared corner counts ear clipped in one mesh; concave polygons past it are fanned.
MAX_EAR_CLIP_WORK = 4_000_000

#: FBX time units per second.
TICKS_PER_SECOND = 46186158000

#: Binary FBX version the writer produces.
WRITE_VERSION = 7400

_BINARY_MAGIC = b"Kaydara FBX Binary  \x00"
_HEADER = _BINARY_MAGIC + b"\x1a\x00"
_FILE_ID = b"\x28\xb3\x2a\xeb\xb6\x24\xcc\xc2\xbf\xc8\xb0\x2a\xa9\x2b\xfc\xf1"
_FOOTER_ID = b"\xfa\xbc\xab\x09\xd0\xc8\xd4\x66\xb1\x76\xfb\x83\x1c\xf7\x26\x7e"
_FOOTER_MAGIC = b"\xf8\x5a\x8c\x6a\xde\xf5\xd9\x7e\xec\xe9\x0c\xe3\x75\x8f\x29\x0b"
_CREATION_TIME = "1970-01-01 10:00:00:000"

#: Bytes charged against a file's budget for each record and each single value read.
_RECORD_COST = 192
_VALUE_COST = 48

#: Element size and dtype of each binary array property type.
_ARRAY_TYPES = {
    "f": (4, "<f4"), "d": (8, "<f8"), "l": (8, "<i8"), "i": (4, "<i4"), "b": (1, "u1"),
    "c": (1, "u1"),
}

#: Frames per second of each GlobalSettings TimeMode.
_TIME_MODES = {
    1: 120.0, 2: 100.0, 3: 60.0, 4: 50.0, 5: 48.0, 6: 30.0, 7: 30.0, 8: 29.97, 9: 29.97,
    10: 25.0, 11: 24.0, 12: 1000.0, 13: 23.976, 15: 96.0, 16: 72.0, 17: 59.94, 18: 119.88,
}

#: Texture slots a material property name feeds, by property name.
_TEXTURE_SLOTS = {
    "DiffuseColor": "base_color", "Diffuse": "base_color", "Maya|TEX_color_map": "base_color",
    "Maya|baseColor": "base_color", "3dsMax|Parameters|base_color_map": "base_color",
    "base_color_map": "base_color", "BaseColor": "base_color",
    "NormalMap": "normal", "Maya|TEX_normal_map": "normal",
    "3dsMax|Parameters|bump_map": "normal", "Bump": "bump",
    "EmissiveColor": "emissive", "Emissive": "emissive", "Maya|TEX_emissive_map": "emissive",
    "3dsMax|Parameters|emit_color_map": "emissive",
    "ReflectionFactor": "metallic", "Maya|TEX_metallic_map": "metallic",
    "3dsMax|Parameters|metalness_map": "metallic",
    "ShininessExponent": "roughness", "Maya|TEX_roughness_map": "roughness",
    "3dsMax|Parameters|roughness_map": "roughness",
    "Maya|TEX_ao_map": "occlusion", "3dsMax|Parameters|ao_map": "occlusion",
    "TransparentColor": "opacity", "TransparencyFactor": "opacity",
    "3dsMax|main|base_color_map": "base_color", "3dsMax|main|metalness_map": "metallic",
    "3dsMax|main|roughness_map": "roughness", "3dsMax|main|ao_map": "occlusion",
    "3dsMax|main|norm_map": "normal", "3dsMax|main|emit_color_map": "emissive",
    "3dsMax|main|opacity_map": "opacity", "Maya|metalness": "metallic",
    "Maya|specularRoughness": "roughness", "Maya|emissionColor": "emissive",
    "Maya|normalCamera": "bump",
}

#: Property names holding a PBR base colour, metalness, roughness and emission, by quantity.
_PBR_NAMES = {
    "base": ("Maya|base_color", "Maya|baseColor", "3dsMax|main|basecolor",
             "3dsMax|Parameters|base_color", "BaseColor"),
    "metallic": ("Maya|metallic", "Maya|metalness", "3dsMax|main|metalness",
                 "3dsMax|Parameters|metalness", "Metallic"),
    "roughness": ("Maya|roughness", "Maya|specularRoughness", "3dsMax|main|roughness",
                  "3dsMax|Parameters|roughness", "Roughness"),
    "emissive": ("Maya|emissive", "Maya|emissionColor", "3dsMax|main|emit_color",
                 "3dsMax|Parameters|emit_color"),
    "emission": ("Maya|emissive_intensity", "Maya|emission", "3dsMax|Parameters|emission"),
}

#: Property names that, set to 1, mark a roughness value and map as glossiness.
_GLOSSINESS_FLAGS = ("3dsMax|main|useGlossiness", "3dsMax|Parameters|roughness_inv")


class FbxError(ValueError):
    """An FBX file that cannot be read or a model that cannot be written."""


# ---------------------------------------------------------------------------------------------
# Records


class _Node:
    """One FBX record: a name, its values and its nested records."""

    __slots__ = ("name", "props", "children")

    def __init__(self, name: str, props: list | None = None, children: list | None = None):
        self.name = name
        self.props = props if props is not None else []
        self.children = children if children is not None else []

    def find(self, name: str) -> _Node | None:
        """The first nested record called ``name``, or None."""
        for child in self.children:
            if child.name == name:
                return child
        return None

    def find_all(self, name: str) -> list[_Node]:
        """Every nested record called ``name``."""
        return [child for child in self.children if child.name == name]

    def value(self, name: str, default: Any = None) -> Any:
        """The first value of the nested record called ``name``, or ``default``."""
        child = self.find(name)
        if child is None or not child.props:
            return default
        return child.props[0]

    def array(self, name: str) -> np.ndarray | None:
        """The values of the nested record called ``name`` as one array, or None."""
        child = self.find(name)
        if child is None or not child.props:
            return None
        first = child.props[0]
        if isinstance(first, np.ndarray):
            return first
        try:
            return np.asarray(child.props, dtype=np.float64)
        except (TypeError, ValueError):
            return None


class _Budget:
    """Running totals that bound what one file may allocate."""

    def __init__(self, file_size: int) -> None:
        self.nodes = 0
        self.decoded = 0
        self.ceiling = min(MAX_DECODED_BYTES, max(256 * 1024 * 1024, DECODED_PER_FILE_BYTE * file_size))

    def node(self) -> None:
        """Count one record."""
        self.nodes += 1
        if self.nodes > MAX_NODES:
            raise FbxError(f"The FBX file holds more than {MAX_NODES} records, more than a model "
                           "file can hold. It is damaged or not a model.")
        self._charge(_RECORD_COST)

    def values(self, count: int) -> None:
        """Count ``count`` single values held as Python objects."""
        self._charge(count * _VALUE_COST)

    def take(self, size: int) -> None:
        """Count ``size`` decoded bytes of one array."""
        if size > MAX_ARRAY_BYTES:
            raise FbxError(f"An array in the FBX file declares {size // (1024 * 1024)} MB, more "
                           f"than the {MAX_ARRAY_BYTES // (1024 * 1024)} MB one array may hold.")
        self._charge(size)

    def _charge(self, size: int) -> None:
        """Add ``size`` bytes to the running total."""
        self.decoded += size
        if self.decoded > self.ceiling:
            raise FbxError(f"The FBX file's data decodes to more than {self.ceiling // (1024 * 1024)} "
                           "MB, more than a file of its size can hold. It is damaged, or the model "
                           "must be decimated or split before loading.")


def _inflate(raw: bytes, size: int) -> bytes:
    """Inflate a zlib stream that must hold exactly ``size`` bytes.

    Args:
        raw: The compressed bytes.
        size: Bytes the stream declares.

    Returns:
        The inflated bytes.

    Raises:
        FbxError: The stream is corrupt or holds a different size.
    """
    inflater = zlib.decompressobj()
    try:
        out = inflater.decompress(raw, max(size, 1))
    except zlib.error as error:
        raise FbxError("A compressed array in the FBX file is corrupt. Re-export the file.") from error
    if len(out) != size or inflater.unconsumed_tail:
        raise FbxError("A compressed array in the FBX file does not hold the size it declares. "
                       "Re-export the file.")
    return out


def _read_property(data: bytes, pos: int, end: int, budget: _Budget) -> tuple[Any, int]:
    """One binary property value.

    Args:
        data: The file.
        pos: Offset of the type code.
        end: Offset the property list may not pass.
        budget: Running allocation totals.

    Returns:
        ``(value, offset after it)``.

    Raises:
        FbxError: The value runs past its record or has an unknown type.
    """
    if pos >= end:
        raise FbxError("A record in the FBX file is truncated. Re-export the file.")
    code = chr(data[pos])
    pos += 1
    scalar = {"Y": ("<h", 2), "C": ("<?", 1), "I": ("<i", 4), "F": ("<f", 4), "D": ("<d", 8),
              "L": ("<q", 8)}.get(code)
    if scalar is not None:
        fmt, size = scalar
        if pos + size > end:
            raise FbxError("A record in the FBX file is truncated. Re-export the file.")
        return struct.unpack_from(fmt, data, pos)[0], pos + size
    if code in _ARRAY_TYPES:
        if pos + 12 > end:
            raise FbxError("A record in the FBX file is truncated. Re-export the file.")
        length, encoding, stored = struct.unpack_from("<III", data, pos)
        pos += 12
        item, dtype = _ARRAY_TYPES[code]
        size = length * item
        if pos + stored > end:
            raise FbxError("An array in the FBX file runs past its record. Re-export the file.")
        if encoding not in (0, 1):
            raise FbxError(f"An array in the FBX file uses unknown encoding {encoding}.")
        if encoding == 0 and stored != size:
            raise FbxError("An array in the FBX file does not hold the size it declares.")
        if encoding == 1 and size > stored * MAX_INFLATE_RATIO + 64:
            raise FbxError("A compressed array in the FBX file declares more data than its stream "
                           "can hold. The file is damaged.")
        budget.take(size)
        raw = data[pos:pos + stored]
        if encoding == 1:
            raw = _inflate(raw, size)
        return np.frombuffer(raw, dtype=dtype), pos + stored
    if code in ("S", "R"):
        if pos + 4 > end:
            raise FbxError("A record in the FBX file is truncated. Re-export the file.")
        length = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        if pos + length > end:
            raise FbxError("A string in the FBX file runs past its record. Re-export the file.")
        raw = bytes(data[pos:pos + length])
        if code == "S":
            return raw.decode("utf-8", "replace"), pos + length
        return raw, pos + length
    raise FbxError(f"The FBX file holds a value of unknown type {code!r}. It is damaged.")


def _read_record(data: bytes, pos: int, wide: bool, depth: int, budget: _Budget,
                 limit: int) -> tuple[_Node | None, int]:
    """One binary record and everything nested in it.

    Args:
        data: The file.
        pos: Offset of the record.
        wide: Whether offsets are 64-bit, as in FBX 7.5 and later.
        depth: Nesting depth of the record.
        budget: Running allocation totals.
        limit: Offset the record may not pass: the end of its parent, or of the file.

    Returns:
        ``(record, offset after it)``; the record is None for the null record ending a list.

    Raises:
        FbxError: The record is truncated, nested too deeply or inconsistent.
    """
    head = 25 if wide else 13
    if pos + head > limit:
        raise FbxError("A record in the FBX file is truncated. It is damaged or was not copied "
                       "whole.")
    if wide:
        end, count, length = struct.unpack_from("<QQQ", data, pos)
    else:
        end, count, length = struct.unpack_from("<III", data, pos)
    name_length = data[pos + head - 1]
    if end == 0:
        return None, pos + head
    if depth > MAX_DEPTH:
        raise FbxError(f"The FBX file nests records more than {MAX_DEPTH} deep. It is damaged.")
    if end < pos + head + name_length or end > limit:
        raise FbxError("A record in the FBX file points outside its parent or the file. It is "
                       "damaged or truncated.")
    budget.node()
    cursor = pos + head
    name = bytes(data[cursor:cursor + name_length]).decode("ascii", "replace")
    cursor += name_length
    props_end = cursor + length
    if props_end > end:
        raise FbxError("A record's values in the FBX file run past the record. It is damaged.")
    if count > length:
        raise FbxError("A record in the FBX file declares more values than it holds.")
    budget.values(count)
    props = []
    for _ in range(count):
        value, cursor = _read_property(data, cursor, props_end, budget)
        props.append(value)
    if cursor != props_end:
        raise FbxError("A record's values in the FBX file do not fill the length it declares.")
    children = []
    while cursor < end:
        child, cursor = _read_record(data, cursor, wide, depth + 1, budget, end)
        if child is None:
            break
        children.append(child)
    return _Node(name, props, children), end


def _parse_binary(data: bytes) -> tuple[int, list[_Node]]:
    """Every top-level record of a binary FBX file.

    Args:
        data: The file.

    Returns:
        ``(version, records)``.

    Raises:
        FbxError: The file is truncated, damaged or older than FBX 7.
    """
    if len(data) < 27:
        raise FbxError("The FBX file is truncated.")
    version = struct.unpack_from("<I", data, 23)[0]
    if version < 7000:
        raise FbxError(f"The file is FBX {version / 1000:.1f}. Only FBX 7 (2011 and later) is "
                       "read: re-export it as FBX 2014 or newer.")
    wide = version >= 7500
    head = 25 if wide else 13
    budget = _Budget(len(data))
    roots = []
    pos = 27
    terminated = False
    while pos + head <= len(data):
        node, pos = _read_record(data, pos, wide, 0, budget, len(data))
        if node is None:
            terminated = True
            break
        roots.append(node)
    if not terminated:
        names = {node.name for node in roots}
        if "Objects" not in names or "Connections" not in names:
            raise FbxError("The FBX file is truncated: it ends before its Objects and Connections "
                           "sections are complete. Copy or export it again.")
        logger.warning("the FBX file has no closing record or footer, so it may be truncated")
    return version, roots


#: Tokens of an ASCII FBX file: comment, string, key, array count, number, brace, comma, word.
_ASCII_TOKEN = re.compile(
    r'\s*(?:(;[^\n]*)|"([^"]*)"|([A-Za-z_][\w|\-]*)\s*:|\*(\d+)'
    r'|([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)|(\{)|(\})|(,)|([^\s,{}";]+))'
)


class _AsciiParser:
    """Reads the records of an ASCII FBX file."""

    def __init__(self, text: str, budget: _Budget):
        self.text = text
        self.pos = 0
        self.budget = budget
        self.peeked = None

    def _next(self):
        """The next token as ``(kind, value)``; ``kind`` is None at the end."""
        if self.peeked is not None:
            token, self.peeked = self.peeked, None
            return token
        text = self.text
        while True:
            match = _ASCII_TOKEN.match(text, self.pos)
            if match is None or match.end() == self.pos:
                if text[self.pos:].strip():
                    raise FbxError("The ASCII FBX file holds text that is not FBX near "
                                   f"character {self.pos}.")
                return None, None
            self.pos = match.end()
            if match.group(1) is not None:
                continue
            for kind in (2, 3, 4, 5, 6, 7, 8, 9):
                value = match.group(kind)
                if value is not None:
                    return kind, value

    def _peek(self):
        """The next token without consuming it."""
        if self.peeked is None:
            self.peeked = self._next()
        return self.peeked

    def _array(self, declared: int) -> np.ndarray:
        """The ``{ a: ... }`` body of an array value."""
        kind, _ = self._next()
        if kind != 6:
            raise FbxError("An array in the ASCII FBX file has no body.")
        close = self.text.find("}", self.pos)
        if close < 0:
            raise FbxError("An array in the ASCII FBX file is not closed.")
        body = self.text[self.pos:close]
        self.pos = close + 1
        colon = body.find(":")
        body = body[colon + 1:] if colon >= 0 else body
        if declared > len(body) + 1:
            raise FbxError("An array in the ASCII FBX file declares more values than it holds.")
        if not body.strip():
            return np.zeros(0, dtype=np.float64)
        integral = not any(mark in body for mark in ".eEnN")
        dtype = np.int64 if integral else np.float64
        self.budget.take(max(declared, body.count(",") + 1) * 8)
        stripped = body.strip().rstrip(",")
        items = [item.strip() for item in stripped.split(",")]
        try:
            return np.array(items, dtype=dtype)
        except (ValueError, OverflowError):
            pass
        parsed = np.full(len(items), np.nan)
        for position, item in enumerate(items):
            try:
                parsed[position] = float(item)
            except ValueError:
                continue
        if np.isnan(parsed).all():
            raise FbxError("An array in the ASCII FBX file holds values that are not numbers.")
        logger.warning("an array in the ASCII FBX file holds values that are not numbers; they "
                       "were read as NaN")
        return parsed

    def nodes(self, depth: int, closed: bool) -> list[_Node]:
        """Records until a closing brace or the end of the file.

        Args:
            depth: Nesting depth.
            closed: Whether a closing brace ends the list.

        Returns:
            The records read.
        """
        if depth > MAX_DEPTH:
            raise FbxError(f"The ASCII FBX file nests records more than {MAX_DEPTH} deep.")
        result = []
        while True:
            kind, value = self._next()
            if kind is None:
                if closed:
                    raise FbxError("The ASCII FBX file ends inside a record. It is truncated.")
                return result
            if kind == 7:
                if not closed:
                    raise FbxError("The ASCII FBX file closes a record it never opened.")
                return result
            if kind != 3:
                raise FbxError(f"The ASCII FBX file holds an unexpected value near character {self.pos}.")
            self.budget.node()
            node = _Node(value)
            while True:
                kind, token = self._peek()
                if kind == 8:
                    self._next()
                    continue
                if kind == 4:
                    self._next()
                    node.props.append(self._array(int(token) if len(token) <= 18 else 10 ** 18))
                    break
                if kind in (2, 5, 9):
                    self._next()
                    self.budget.values(1)
                    if kind == 5:
                        integral = not any(mark in token for mark in ".eE")
                        try:
                            token = int(token) if integral else float(token)
                        except ValueError as error:
                            raise FbxError("The ASCII FBX file holds a number too long to read near "
                                           f"character {self.pos}.") from error
                    node.props.append(token)
                    continue
                break
            kind, _ = self._peek()
            if kind == 6:
                self._next()
                node.children = self.nodes(depth + 1, True)
            result.append(node)


def _parse_ascii(data: bytes) -> tuple[int, list[_Node]]:
    """Every top-level record of an ASCII FBX file.

    Args:
        data: The file.

    Returns:
        ``(version, records)``.
    """
    text = data.decode("utf-8", "replace")
    roots = _AsciiParser(text, _Budget(len(data))).nodes(0, False)
    version = 0
    for node in roots:
        if node.name == "FBXHeaderExtension":
            declared = node.value("FBXVersion", 0)
            version = int(declared) if isinstance(declared, int) else 0
    if not version:
        match = re.match(r"\s*;\s*FBX\s+(\d+)\.(\d+)\.(\d+)", text)
        if match:
            version = int(match.group(1)) * 1000 + int(match.group(2)) * 100 + int(match.group(3)) * 10
    return version, roots


def _parse(data: bytes) -> tuple[int, list[_Node], bool]:
    """The records of a binary or ASCII FBX file.

    Args:
        data: The file.

    Returns:
        ``(version, records, binary)``.

    Raises:
        FbxError: The bytes are not FBX, are too large, or are FBX older than version 7.
    """
    if len(data) > MAX_FILE_BYTES:
        raise FbxError(f"The FBX file is larger than {MAX_FILE_BYTES // (1024 ** 3)} GB.")
    if bytes(data[:21]) == _BINARY_MAGIC:
        version, roots = _parse_binary(data)
        binary = True
    elif bytes(data[:18]) == _BINARY_MAGIC[:18]:
        if len(data) < 27:
            raise FbxError("The FBX file is truncated.")
        raise FbxError("The binary FBX file's header is damaged: its zero bytes were changed, as a "
                       "text-mode copy or download does. Copy the file again in binary mode.")
    else:
        head = bytes(data[:4096]).decode("utf-8", "replace")
        if "FBXHeaderExtension" not in head and not head.lstrip().startswith("; FBX"):
            raise FbxError("The file is not an FBX file: it has neither the binary FBX signature "
                           "nor an ASCII FBX header.")
        version, roots = _parse_ascii(data)
        binary = False
    if version and version < 7000:
        raise FbxError(f"The file is FBX {version / 1000:.1f}. Only FBX 7 (2011 and later) is "
                       "read: re-export it as FBX 2014 or newer.")
    return version, roots, binary


# ---------------------------------------------------------------------------------------------
# Objects, connections and properties


def _object_name(raw: Any) -> str:
    """An object's own name, without the class FBX stores beside it."""
    text = raw if isinstance(raw, str) else str(raw)
    if "\x00\x01" in text:
        return text.split("\x00\x01", 1)[0]
    if "::" in text:
        return text.split("::", 1)[1]
    return text


def _properties(node: _Node | None) -> dict[str, list]:
    """The ``Properties70`` values of a record, by property name."""
    found: dict[str, list] = {}
    if node is None:
        return found
    block = node.find("Properties70")
    if block is None:
        return found
    for entry in block.children:
        if entry.name == "P" and entry.props and isinstance(entry.props[0], str):
            found[entry.props[0]] = entry.props[4:]
    return found


class _Object:
    """One entry of the Objects section."""

    __slots__ = ("id", "kind", "name", "sub", "node", "props")

    def __init__(self, node: _Node):
        self.node = node
        self.kind = node.name
        self.id = int(node.props[0])
        self.name = _object_name(node.props[1]) if len(node.props) > 1 else ""
        self.sub = str(node.props[2]) if len(node.props) > 2 else ""
        self.props = _properties(node)


class _Document:
    """The objects of an FBX file, their connections and the property templates."""

    def __init__(self, roots: list[_Node]):
        self.top: dict[str, _Node] = {}
        for node in roots:
            self.top.setdefault(node.name, node)
        self.objects: dict[int, _Object] = {}
        section = self.top.get("Objects")
        for node in section.children if section is not None else []:
            if len(node.props) >= 1 and isinstance(node.props[0], (int, np.integer)):
                entry = _Object(node)
                self.objects.setdefault(entry.id, entry)
        self.children: dict[int, list[tuple[int, str]]] = defaultdict(list)
        self.parents: dict[int, list[tuple[int, str]]] = defaultdict(list)
        section = self.top.get("Connections")
        for node in section.children if section is not None else []:
            if node.name != "C" or len(node.props) < 3:
                continue
            try:
                child, parent = int(node.props[1]), int(node.props[2])
            except (TypeError, ValueError):
                continue
            prop = str(node.props[3]) if len(node.props) > 3 else ""
            self.children[parent].append((child, prop))
            self.parents[child].append((parent, prop))
        self.templates: dict[str, dict[str, list]] = {}
        section = self.top.get("Definitions")
        for node in section.children if section is not None else []:
            if node.name == "ObjectType" and node.props:
                template = node.find("PropertyTemplate")
                if template is not None:
                    self.templates.setdefault(str(node.props[0]), _properties(template))
        self.settings = _properties(self.top.get("GlobalSettings"))

    def prop(self, entry: _Object, name: str, default: Any = None) -> list | Any:
        """A property of an object, falling back to its type's template, then ``default``."""
        if name in entry.props:
            return entry.props[name]
        template = self.templates.get(entry.kind)
        if template is not None and name in template:
            return template[name]
        return default

    def number(self, entry: _Object, name: str, default: float) -> float:
        """A finite scalar property as a float, or ``default``."""
        values = self.prop(entry, name, None)
        try:
            value = float(values[0]) if values else default
        except (TypeError, ValueError, IndexError, OverflowError):
            return default
        return value if math.isfinite(value) else default

    def vector(self, entry: _Object, name: str, default: tuple[float, float, float]) -> np.ndarray:
        """A finite three-component property as a float64 array, or ``default``."""
        values = self.prop(entry, name, None)
        try:
            if values is not None and len(values) >= 3:
                found = np.array([float(v) for v in values[:3]], dtype=np.float64)
                if np.all(np.isfinite(found)):
                    return found
        except (TypeError, ValueError, OverflowError):
            pass
        return np.array(default, dtype=np.float64)

    def linked(self, parent: int, kind: str | None = None, sub: str | None = None) -> list[tuple[_Object, str]]:
        """Objects connected into ``parent``, in file order, with the property they feed."""
        found = []
        for child, prop in self.children.get(parent, []):
            entry = self.objects.get(child)
            if entry is None or (kind is not None and entry.kind != kind):
                continue
            if sub is not None and entry.sub != sub:
                continue
            found.append((entry, prop))
        return found

    def owners(self, child: int, kind: str | None = None) -> list[tuple[_Object, str]]:
        """Objects ``child`` is connected into, in file order, with the property it feeds."""
        found = []
        for parent, prop in self.parents.get(child, []):
            entry = self.objects.get(parent)
            if entry is not None and (kind is None or entry.kind == kind):
                found.append((entry, prop))
        return found


# ---------------------------------------------------------------------------------------------
# Transform math, float64 numpy, column vectors


def _inverse(matrix: np.ndarray) -> np.ndarray:
    """Inverses of square matrices, pseudo-inverses where a matrix is singular."""
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix)


def _translation(vector: np.ndarray) -> np.ndarray:
    """A 4x4 translation matrix, or a stack of them for ``(T, 3)`` input."""
    vector = np.asarray(vector, dtype=np.float64)
    out = np.broadcast_to(np.eye(4), vector.shape[:-1] + (4, 4)).copy()
    out[..., :3, 3] = vector
    return out


def _scaling(vector: np.ndarray) -> np.ndarray:
    """A 4x4 scale matrix, or a stack of them for ``(T, 3)`` input."""
    vector = np.asarray(vector, dtype=np.float64)
    out = np.broadcast_to(np.eye(4), vector.shape[:-1] + (4, 4)).copy()
    out[..., 0, 0] = vector[..., 0]
    out[..., 1, 1] = vector[..., 1]
    out[..., 2, 2] = vector[..., 2]
    return out


def _axis_rotation(angle: np.ndarray, axis: int) -> np.ndarray:
    """4x4 rotations by ``angle`` radians about one axis, one per element of ``angle``."""
    cos, sin = np.cos(angle), np.sin(angle)
    out = np.broadcast_to(np.eye(4), np.shape(angle) + (4, 4)).copy()
    first, second = [(1, 2), (2, 0), (0, 1)][axis]
    out[..., first, first] = cos
    out[..., first, second] = -sin
    out[..., second, first] = sin
    out[..., second, second] = cos
    return out


#: Axis application order of each FBX RotationOrder value, first applied first.
_ROTATION_ORDERS = {0: (0, 1, 2), 1: (0, 2, 1), 2: (1, 2, 0), 3: (1, 0, 2), 4: (2, 0, 1),
                    5: (2, 1, 0), 6: (0, 1, 2)}


def _euler(degrees: np.ndarray, order: int = 0) -> np.ndarray:
    """4x4 rotation from FBX Euler angles in degrees, applied in the given RotationOrder.

    Args:
        degrees: ``(3,)`` or ``(T, 3)`` angles about X, Y and Z.
        order: FBX RotationOrder, 0 for XYZ.

    Returns:
        ``(4, 4)`` or ``(T, 4, 4)`` rotation.
    """
    radians = np.radians(np.asarray(degrees, dtype=np.float64))
    out = np.broadcast_to(np.eye(4), radians.shape[:-1] + (4, 4)).copy()
    for axis in _ROTATION_ORDERS.get(int(order), (0, 1, 2)):
        out = _axis_rotation(radians[..., axis], axis) @ out
    return out


def _decompose(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Translation, rotation and scale of affine matrices with no shear.

    Args:
        matrix: ``(..., 4, 4)`` transforms.

    Returns:
        ``(translation (..., 3), rotation (..., 3, 3), scale (..., 3))``.
    """
    batch = matrix.shape[:-2]
    flat = matrix.reshape(-1, 4, 4)
    translation = flat[:, :3, 3].copy()
    linear = flat[:, :3, :3]
    scale = np.linalg.norm(linear, axis=-2)
    scale = np.where(scale < 1e-12, 1.0, scale)
    rotation = linear / scale[:, None, :]
    flip = np.linalg.det(rotation) < 0
    scale[flip, 0] *= -1
    rotation[flip, :, 0] *= -1
    return (translation.reshape(batch + (3,)), rotation.reshape(batch + (3, 3)),
            scale.reshape(batch + (3,)))


def _quaternion(rotation: np.ndarray) -> np.ndarray:
    """Unit quaternions ``xyzw`` from ``(T, 3, 3)`` rotation matrices."""
    m = rotation
    count = m.shape[0]
    q = np.zeros((count, 4), dtype=np.float64)
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
    cases = [trace > 0,
             (m[:, 0, 0] >= m[:, 1, 1]) & (m[:, 0, 0] >= m[:, 2, 2]),
             m[:, 1, 1] >= m[:, 2, 2],
             np.ones(count, dtype=bool)]
    done = np.zeros(count, dtype=bool)
    for index, case in enumerate(cases):
        sel = case & ~done
        done |= sel
        if not sel.any():
            continue
        r = m[sel]
        if index == 0:
            s = np.sqrt(trace[sel] + 1.0) * 2
            q[sel] = np.stack([(r[:, 2, 1] - r[:, 1, 2]) / s, (r[:, 0, 2] - r[:, 2, 0]) / s,
                               (r[:, 1, 0] - r[:, 0, 1]) / s, 0.25 * s], -1)
        elif index == 1:
            s = np.sqrt(np.maximum(1.0 + r[:, 0, 0] - r[:, 1, 1] - r[:, 2, 2], 1e-30)) * 2
            q[sel] = np.stack([0.25 * s, (r[:, 0, 1] + r[:, 1, 0]) / s,
                               (r[:, 0, 2] + r[:, 2, 0]) / s, (r[:, 2, 1] - r[:, 1, 2]) / s], -1)
        elif index == 2:
            s = np.sqrt(np.maximum(1.0 + r[:, 1, 1] - r[:, 0, 0] - r[:, 2, 2], 1e-30)) * 2
            q[sel] = np.stack([(r[:, 0, 1] + r[:, 1, 0]) / s, 0.25 * s,
                               (r[:, 1, 2] + r[:, 2, 1]) / s, (r[:, 0, 2] - r[:, 2, 0]) / s], -1)
        else:
            s = np.sqrt(np.maximum(1.0 + r[:, 2, 2] - r[:, 0, 0] - r[:, 1, 1], 1e-30)) * 2
            q[sel] = np.stack([(r[:, 0, 2] + r[:, 2, 0]) / s, (r[:, 1, 2] + r[:, 2, 1]) / s,
                               0.25 * s, (r[:, 1, 0] - r[:, 0, 1]) / s], -1)
    q /= np.linalg.norm(q, axis=1, keepdims=True).clip(min=1e-12)
    return q


def _euler_xyz(rotation: np.ndarray) -> np.ndarray:
    """Euler angles in degrees, XYZ order, from ``(J, 3, 3)`` rotation matrices."""
    m = rotation
    sin_y = np.clip(-m[:, 2, 0], -1.0, 1.0)
    y = np.arcsin(sin_y)
    regular = np.abs(sin_y) < 0.9999995
    x = np.where(regular, np.arctan2(m[:, 2, 1], m[:, 2, 2]),
                 np.where(sin_y > 0, np.arctan2(m[:, 0, 1], m[:, 1, 1]),
                          np.arctan2(-m[:, 0, 1], m[:, 1, 1])))
    z = np.where(regular, np.arctan2(m[:, 1, 0], m[:, 0, 0]), 0.0)
    return np.degrees(np.stack([x, y, z], -1))


def _fbx_matrix(values: Any) -> np.ndarray | None:
    """A 4x4 matrix from the 16 column-major values FBX stores, or None."""
    if values is None:
        return None
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size != 16 or not np.all(np.isfinite(flat)):
        return None
    return flat.reshape(4, 4).T.copy()


def _matrix_values(matrix: np.ndarray) -> np.ndarray:
    """The 16 column-major float64 values FBX stores for a 4x4 matrix."""
    return np.ascontiguousarray(np.asarray(matrix, dtype=np.float64).T.reshape(-1))


# ---------------------------------------------------------------------------------------------
# Mesh helpers


def _unique_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distinct rows of an integer table, in order of first appearance.

    Args:
        rows: ``(N, K)`` int32 table.

    Returns:
        ``(first, inverse)``: the row index of each distinct row's first appearance, and the
        distinct row each input row equals.
    """
    count = rows.shape[0]
    if count == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    rows = np.ascontiguousarray(rows)
    digest = np.zeros(count, dtype=np.uint64)
    golden = np.uint64(0x9E3779B97F4A7C15)
    with np.errstate(over="ignore"):
        for column in range(rows.shape[1]):
            value = rows[:, column].astype(np.int64).astype(np.uint64)
            digest ^= value + golden + (digest << np.uint64(6)) + (digest >> np.uint64(2))
    _, first, inverse = np.unique(digest, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    if not np.array_equal(rows[first][inverse], rows):
        packed = rows.view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))).reshape(-1)
        _, first, inverse = np.unique(packed, return_index=True, return_inverse=True)
        inverse = inverse.reshape(-1)
    order = np.argsort(first, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    return first[order].astype(np.int64), rank[inverse].astype(np.int64)


def _bits(values: np.ndarray) -> np.ndarray:
    """Float values as the int32 bit patterns of their float32 form, with ``-0`` folded to ``0``."""
    return (np.ascontiguousarray(values, dtype=np.float32) + np.float32(0.0)).view(np.int32)


def _ear_clip(points: np.ndarray, normal: np.ndarray) -> list[tuple[int, int, int]]:
    """Triangles of one simple polygon, possibly concave, by ear clipping.

    Args:
        points: ``(S, 3)`` corners in order.
        normal: ``(3,)`` polygon normal.

    Returns:
        Corner index triples keeping the polygon's winding.
    """
    axis = int(np.argmax(np.abs(normal)))
    keep = [(1, 2), (2, 0), (0, 1)][axis]
    flat = points[:, keep].astype(np.float64)
    if normal[axis] < 0:
        flat[:, 0] = -flat[:, 0]
    extent = float(np.ptp(flat, axis=0).max()) or 1.0
    epsilon = 1e-12 * extent * extent
    remaining = np.arange(len(points))
    triangles = []

    def side(ax, ay, bx, by, px, py):
        return (bx - ax) * (py - ay) - (by - ay) * (px - ax)

    while remaining.size > 3:
        size = remaining.size
        prev, nxt = np.roll(remaining, 1), np.roll(remaining, -1)
        ax, ay = flat[prev, 0], flat[prev, 1]
        bx, by = flat[remaining, 0], flat[remaining, 1]
        cx, cy = flat[nxt, 0], flat[nxt, 1]
        candidates = np.nonzero(side(ax, ay, bx, by, cx, cy) > epsilon)[0]
        if candidates.size == 0:
            break
        ax, ay, bx, by, cx, cy = (v[candidates, None] for v in (ax, ay, bx, by, cx, cy))
        px, py = flat[remaining, 0][None, :], flat[remaining, 1][None, :]
        inside = ((side(ax, ay, bx, by, px, py) >= 0) & (side(bx, by, cx, cy, px, py) >= 0)
                  & (side(cx, cy, ax, ay, px, py) >= 0))
        rows = np.arange(candidates.size)
        inside[rows, (candidates - 1) % size] = False
        inside[rows, candidates] = False
        inside[rows, (candidates + 1) % size] = False
        ears = candidates[~inside.any(axis=1)]
        if ears.size == 0:
            break
        chosen: list[int] = []
        for position in ears.tolist():
            if size - len(chosen) <= 3:
                break
            if chosen and (position - chosen[-1] < 2 or chosen[0] + size - position < 2):
                continue
            chosen.append(position)
        for position in chosen:
            triangles.append((int(prev[position]), int(remaining[position]), int(nxt[position])))
        remaining = np.delete(remaining, chosen)
    for index in range(1, remaining.size - 1):
        triangles.append((int(remaining[0]), int(remaining[index]), int(remaining[index + 1])))
    return triangles


def _triangulate(points: np.ndarray, sizes: np.ndarray,
                 spent: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Triangles of polygons, fanned when convex and ear clipped when concave.

    Args:
        points: ``(C, 3)`` corner positions, polygon after polygon.
        sizes: ``(P,)`` corner count of each polygon, every count at least 3.
        spent: One-item list of ear clipping work already done, shared across calls and updated.

    Returns:
        ``(triangles (M, 3) corner indices, polygon (M,) index of the polygon each came from)``.
    """
    starts = np.cumsum(sizes) - sizes
    pieces, owners = [], []
    spent = spent if spent is not None else [0]
    work = spent[0]
    for size in np.unique(sizes):
        size = int(size)
        chosen = np.nonzero(sizes == size)[0]
        base = starts[chosen]
        if size == 3:
            pieces.append(base[:, None] + np.arange(3))
            owners.append(chosen)
            continue
        steps = np.arange(1, size - 1)
        if size > MAX_EAR_CLIP_CORNERS:
            pieces.append(np.stack([np.broadcast_to(base[:, None], (base.size, size - 2)),
                                    base[:, None] + steps, base[:, None] + steps + 1], -1).reshape(-1, 3))
            owners.append(np.repeat(chosen, size - 2))
            continue
        corners = points[base[:, None] + np.arange(size)]
        following = np.roll(corners, -1, axis=1)
        normal = np.cross(corners, following).sum(axis=1)
        incoming = corners - np.roll(corners, 1, axis=1)
        outgoing = following - corners
        turn = (np.cross(incoming, outgoing) * normal[:, None, :]).sum(-1)
        scale = (np.linalg.norm(incoming, axis=-1) * np.linalg.norm(outgoing, axis=-1)
                 * np.linalg.norm(normal, axis=-1)[:, None])
        convex = (turn >= -1e-9 * scale).all(axis=1) | (np.linalg.norm(normal, axis=-1) < 1e-20)
        if size == 4:
            pivot = np.where(convex, 0, np.argmin(turn / np.maximum(scale, 1e-300), axis=1))
            order = (pivot[:, None] + np.arange(4)) % 4
            pieces.append((base[:, None, None] + order[:, [[0, 1, 2], [0, 2, 3]]]).reshape(-1, 3))
            owners.append(np.repeat(chosen, 2))
            continue
        clipped = np.zeros(chosen.size, dtype=bool)
        for row in np.nonzero(~convex)[0]:
            if work + size * size > MAX_EAR_CLIP_WORK:
                if work < MAX_EAR_CLIP_WORK:
                    logger.warning("the FBX file holds more concave polygons than are ear clipped; "
                                   "the rest were fanned and may fold")
                work = MAX_EAR_CLIP_WORK
                break
            work += size * size
            clipped[row] = True
            local = _ear_clip(corners[row], normal[row])
            pieces.append(np.asarray(local, dtype=np.int64).reshape(-1, 3) + base[row])
            owners.append(np.full(len(local), chosen[row], dtype=np.int64))
        fanned = ~clipped
        fan = np.stack([np.broadcast_to(base[fanned, None], (int(fanned.sum()), size - 2)),
                        base[fanned, None] + steps, base[fanned, None] + steps + 1], -1)
        pieces.append(fan.reshape(-1, 3))
        owners.append(np.repeat(chosen[fanned], size - 2))
    spent[0] = work
    if not pieces:
        return np.zeros((0, 3), dtype=np.int64), np.zeros(0, dtype=np.int64)
    triangles = np.concatenate([np.asarray(p, dtype=np.int64).reshape(-1, 3) for p in pieces])
    polygon = np.concatenate([np.asarray(o, dtype=np.int64) for o in owners])
    order = np.argsort(polygon, kind="stable")
    return triangles[order], polygon[order]


def _srgb_to_linear(values: np.ndarray) -> np.ndarray:
    """sRGB codes to linear values, element-wise."""
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(values: np.ndarray) -> np.ndarray:
    """Linear values to sRGB codes, element-wise."""
    values = np.clip(values, 0.0, None)
    return np.where(values <= 0.0031308, values * 12.92, 1.055 * np.power(values, 1 / 2.4) - 0.055)


# ---------------------------------------------------------------------------------------------
# Reader


class _Layer:
    """One per-corner attribute layer of a geometry."""

    def __init__(self, values: np.ndarray, width: int):
        self.values = values
        self.width = width


class _SceneReader:
    """Builds a `Model3D` from a parsed FBX document."""

    def __init__(self, doc: _Document, version: int, base_dir: str | None, name: str):
        self.doc = doc
        self.version = version
        self.base_dir = base_dir
        self.name = name
        self.stem = PurePosixPath(name.replace("\\", "/")).stem if name else ""
        self.conversion = self._conversion()
        self.conversion_inverse = np.linalg.inv(self.conversion)
        self.models = {key: entry for key, entry in doc.objects.items() if entry.kind == "Model"}
        self.parent_of = self._parents()
        self.global_cache: dict[int, np.ndarray] = {}
        self.texture_cache: dict[int, Texture | None] = {}
        self.material_cache: dict[int, Material] = {}
        self.combined_cache: dict[tuple, Texture | None] = {}
        self.clip_work = [0]

    # Axes and hierarchy

    def _conversion(self) -> np.ndarray:
        """4x4 change of basis from file coordinates to metres, Y up, facing +Z, mirrored for left-handed axes."""
        settings = self.doc.settings

        def setting(name: str, default: float) -> float:
            values = settings.get(name)
            try:
                value = float(values[0]) if values else default
            except (TypeError, ValueError, OverflowError):
                return default
            return value if math.isfinite(value) else default

        def axis(name: str, default: int) -> np.ndarray:
            index = int(setting(name, default))
            sign = 1.0 if setting(name + "Sign", 1) >= 0 else -1.0
            vector = np.zeros(3)
            vector[index if index in (0, 1, 2) else default] = sign
            return vector

        up = axis("UpAxis", 1)
        coord = axis("CoordAxis", 0)
        if abs(float(np.dot(up, coord))) > 0.5:
            up, coord = np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
        front = np.cross(coord, up)
        if "FrontAxis" in settings:
            declared = axis("FrontAxis", 2)
            if float(np.dot(declared, front)) < -0.5:
                logger.info("the FBX file declares left-handed axes, so it was mirrored front to back")
                front = -front
        unit = setting("UnitScaleFactor", 1.0)
        if unit <= 0:
            unit = 1.0
        out = np.eye(4)
        out[:3, :3] = np.stack([coord, up, front]) * (unit / 100.0)
        return out

    def _to_model_space(self, matrix: np.ndarray) -> np.ndarray:
        """A file-space transform expressed in model space."""
        return self.conversion @ matrix @ self.conversion_inverse

    def _parents(self) -> dict[int, int]:
        """The parent Model of each Model, ``0`` for the scene root."""
        parents = {}
        for key in self.models:
            parent = 0
            for owner, prop in self.doc.owners(key, "Model"):
                if not prop:
                    parent = owner.id
                    break
            parents[key] = parent
        return parents

    def _chain(self, key: int) -> list[int]:
        """``key`` and its Model ancestors, root first."""
        chain, seen = [], set()
        while key in self.models and key not in seen:
            if len(chain) >= MAX_HIERARCHY_DEPTH:
                raise FbxError(f"The FBX file's hierarchy is more than {MAX_HIERARCHY_DEPTH} deep.")
            seen.add(key)
            chain.append(key)
            key = self.parent_of.get(key, 0)
        return chain[::-1]

    def _components(self, entry: _Object) -> dict[str, Any]:
        """The static transform properties of a Model."""
        doc = self.doc
        inherit = int(doc.number(entry, "InheritType", 1))
        return {
            "T": doc.vector(entry, "Lcl Translation", (0, 0, 0)),
            "R": doc.vector(entry, "Lcl Rotation", (0, 0, 0)),
            "S": doc.vector(entry, "Lcl Scaling", (1, 1, 1)),
            "pre": doc.vector(entry, "PreRotation", (0, 0, 0)),
            "post": doc.vector(entry, "PostRotation", (0, 0, 0)),
            "roff": doc.vector(entry, "RotationOffset", (0, 0, 0)),
            "rpiv": doc.vector(entry, "RotationPivot", (0, 0, 0)),
            "soff": doc.vector(entry, "ScalingOffset", (0, 0, 0)),
            "spiv": doc.vector(entry, "ScalingPivot", (0, 0, 0)),
            "order": int(doc.number(entry, "RotationOrder", 0)),
            "inherit": inherit if inherit in (0, 1, 2) else 1,
        }

    @staticmethod
    def _local(parts: dict[str, Any], T=None, R=None, S=None) -> np.ndarray:
        """Local transform of a Model with every pivot and offset, optionally animated."""
        T = parts["T"] if T is None else T
        R = parts["R"] if R is None else R
        S = parts["S"] if S is None else S
        rpiv, spiv = parts["rpiv"], parts["spiv"]
        return (_translation(T) @ _translation(parts["roff"]) @ _translation(rpiv)
                @ _euler(parts["pre"]) @ _euler(R, parts["order"])
                @ _euler(parts["post"]).T @ _translation(-rpiv)
                @ _translation(parts["soff"]) @ _translation(spiv) @ _scaling(S)
                @ _translation(-spiv))

    @staticmethod
    def _inherit(parent: np.ndarray, parent_scale: np.ndarray, parts: dict[str, Any],
                 local: np.ndarray, R=None, S=None) -> np.ndarray:
        """Global transform of a Model from its parent's, honouring InheritType."""
        if parts["inherit"] == 1:
            return parent @ local
        R = parts["R"] if R is None else R
        S = parts["S"] if S is None else S
        _, parent_rotation3, _ = _decompose(parent)
        parent_rotation = np.broadcast_to(np.eye(4), parent.shape).copy()
        parent_rotation[..., :3, :3] = parent_rotation3
        parent_linear = parent.copy()
        parent_linear[..., :3, 3] = 0
        parent_shear_scale = _inverse(parent_rotation) @ parent_linear
        local_rotation = _euler(parts["pre"]) @ _euler(R, parts["order"]) @ _euler(parts["post"]).T
        local_scale = _scaling(S)
        if parts["inherit"] == 0:
            global_rs = parent_rotation @ local_rotation @ parent_shear_scale @ local_scale
        else:
            no_local = parent_shear_scale @ _inverse(_scaling(parent_scale))
            global_rs = parent_rotation @ local_rotation @ no_local @ local_scale
        origin = np.zeros(local.shape[:-2] + (4,))
        origin[..., :3] = local[..., :3, 3]
        origin[..., 3] = 1.0
        moved = (parent @ origin[..., None])[..., 0]
        out = global_rs.copy()
        out[..., :3, 3] = moved[..., :3]
        return out

    def global_matrix(self, key: int) -> np.ndarray:
        """World transform of a Model in file space, from its static properties."""
        if key in self.global_cache:
            return self.global_cache[key]
        world = np.eye(4)
        parent_scale = np.ones(3)
        for node in self._chain(key):
            if node in self.global_cache:
                world = self.global_cache[node]
            else:
                parts = self._components(self.models[node])
                local = self._local(parts)
                world = self._inherit(world, parent_scale, parts, local)
                self.global_cache[node] = world
            parent_scale = self._components(self.models[node])["S"]
        return world

    def geometric(self, key: int) -> np.ndarray:
        """The geometric transform a Model applies to its geometry alone."""
        entry = self.models[key]
        doc = self.doc
        return (_translation(doc.vector(entry, "GeometricTranslation", (0, 0, 0)))
                @ _euler(doc.vector(entry, "GeometricRotation", (0, 0, 0)))
                @ _scaling(doc.vector(entry, "GeometricScaling", (1, 1, 1))))

    # Textures and materials

    def _video_bytes(self, video: _Object) -> bytes | None:
        """The image bytes a Video embeds or names beside the file."""
        content = video.node.find("Content")
        if content is not None and content.props:
            if isinstance(content.props[0], (bytes, bytearray)) and content.props[0]:
                return bytes(content.props[0])
            pieces = [p.strip() for p in content.props if isinstance(p, str) and p.strip()]
            if pieces:
                if sum(len(p) for p in pieces) > texture_io.MAX_SIDECAR_BYTES * 4 // 3 + 4 * len(pieces):
                    logger.warning("an embedded texture in the FBX file is larger than a model's "
                                   "files may hold, so it was skipped")
                    return None
                try:
                    if all(len(p) % 4 == 0 for p in pieces):
                        return b"".join(base64.b64decode(p, validate=False) for p in pieces)
                    return base64.b64decode("".join(pieces), validate=False)
                except (ValueError, TypeError):
                    logger.warning("an embedded texture in the FBX file is not valid base64")
        names = []
        for field in ("RelativeFilename", "Filename", "FileName"):
            value = video.node.value(field)
            if isinstance(value, str) and value.strip():
                names.append(value)
        return self._sidecar(names)

    def _sidecar(self, names: list[str]) -> bytes | None:
        """The bytes of the first named file found beside the model."""
        if self.base_dir is None or not names:
            return None
        tried = []
        for raw in names:
            clean = raw.replace("\\", "/").strip()
            bare = clean.rsplit("/", 1)[-1]
            for candidate in (clean, bare, f"textures/{bare}", f"{self.stem}.fbm/{bare}"):
                if candidate and candidate not in tried and not candidate.startswith("/") and ":" not in candidate:
                    tried.append(candidate)
        for candidate in tried:
            try:
                found = os.path.isfile(sandbox.resolve_read_file(self.base_dir, candidate))
            except sandbox.PathNotAllowed:
                found = False
            if found:
                return texture_io.read_sidecar(self.base_dir, candidate)
        return None

    def texture(self, entry: _Object) -> Texture | None:
        """The decoded image of a Texture object, shared between materials."""
        if entry.id in self.texture_cache:
            return self.texture_cache[entry.id]
        data = None
        label = entry.name
        for video, _ in self.doc.linked(entry.id, "Video"):
            data = self._video_bytes(video)
            if data is not None:
                named = video.node.value("RelativeFilename") or video.node.value("Filename")
                label = named if isinstance(named, str) and named.strip() else video.name or label
                break
        if data is None:
            names = [v for v in (entry.node.value("RelativeFilename"), entry.node.value("FileName"))
                     if isinstance(v, str) and v.strip()]
            data = self._sidecar(names)
        result = None
        if data is not None:
            file_label = PurePosixPath(label.replace("\\", "/")).stem or "texture"
            result = texture_io.decode(data, name=file_label)
            if result is not None:
                wraps = {0: "repeat", 1: "clamp"}
                result.wrap_s = wraps.get(int(self.doc.number(entry, "WrapModeU", 0)), "repeat")
                result.wrap_t = wraps.get(int(self.doc.number(entry, "WrapModeV", 0)), "repeat")
        else:
            logger.warning("texture %s of the FBX file was not found, so it was left out",
                           entry.name or "(unnamed)")
        self.texture_cache[entry.id] = result
        return result

    @staticmethod
    def _gray(texture: Texture) -> torch.Tensor:
        """A texture's first channel as ``(H, W)``."""
        return texture.image[..., 0].float()

    def _combined(self, metallic: Texture | None, roughness: Texture | None,
                  occlusion: Texture | None, glossy: bool = False) -> Texture | None:
        """One glTF metallic-roughness texture from separate grey maps."""
        key = (id(metallic), id(roughness), id(occlusion), glossy)
        if key in self.combined_cache:
            return self.combined_cache[key]
        maps = [m for m in (metallic, roughness, occlusion) if m is not None]
        if not maps:
            return None
        height = max(int(m.image.shape[0]) for m in maps)
        width = max(int(m.image.shape[1]) for m in maps)

        def fit(texture: Texture | None) -> torch.Tensor:
            if texture is None:
                return torch.ones(height, width)
            gray = self._gray(texture)
            if gray.shape != (height, width):
                gray = torch.nn.functional.interpolate(gray[None, None], size=(height, width),
                                                       mode="bilinear", align_corners=False)[0, 0]
            return gray.clamp(0, 1)

        rough = fit(roughness)
        if glossy and roughness is not None:
            rough = 1.0 - rough
        image = torch.stack([fit(occlusion), rough, fit(metallic)], dim=-1).contiguous()
        result = Texture(image=image, name=(roughness or metallic or occlusion).name, mime="image/png")
        self.combined_cache[key] = result
        return result

    def material(self, entry: _Object | None) -> Material:
        """The material an FBX Material object describes."""
        if entry is None:
            return self.material_cache.setdefault(0, Material(name="default", metallic=0.0, roughness=0.8))
        if entry.id in self.material_cache:
            return self.material_cache[entry.id]
        doc = self.doc
        lambert = str(entry.node.value("ShadingModel", "") or "").lower() == "lambert"
        explicit = entry.props

        def numbers(values: list | None, count: int) -> np.ndarray | None:
            try:
                if values is None or len(values) < count:
                    return None
                found = np.array([float(v) for v in values[:count]], dtype=np.float64)
            except (TypeError, ValueError, OverflowError):
                return None
            return found if np.all(np.isfinite(found)) else None

        def pick(names: list[str], count: int = 1) -> list | None:
            for name in names:
                if name in explicit and numbers(explicit[name], count) is not None:
                    return explicit[name]
            return None

        def scalar(names: list[str], default: float) -> float:
            found = numbers(pick(names), 1)
            return float(found[0]) if found is not None else default

        base = doc.vector(entry, "DiffuseColor", (0.8, 0.8, 0.8))
        if "DiffuseColor" not in explicit and "Diffuse" in explicit:
            base = doc.vector(entry, "Diffuse", (0.8, 0.8, 0.8))
        pbr_base = numbers(pick(_PBR_NAMES["base"], 3), 3)
        factor = doc.number(entry, "DiffuseFactor", 1.0)
        if pbr_base is not None:
            base = pbr_base
        else:
            base = base * factor
        opacity = 1.0
        if "Opacity" in explicit:
            opacity = scalar(["Opacity"], 1.0)
        elif "TransparencyFactor" in explicit:
            transparent = doc.vector(entry, "TransparentColor", (1, 1, 1))
            opacity = 1.0 - scalar(["TransparencyFactor"], 0.0) * float(transparent.mean())
        opacity = float(np.clip(opacity, 0.0, 1.0))
        emissive = doc.vector(entry, "EmissiveColor", (0, 0, 0)) * doc.number(entry, "EmissiveFactor", 1.0)
        if "EmissiveColor" not in explicit and "Emissive" in explicit:
            emissive = doc.vector(entry, "Emissive", (0, 0, 0))
        pbr_emissive = numbers(pick(_PBR_NAMES["emissive"], 3), 3)
        if pbr_emissive is not None:
            emissive = pbr_emissive * scalar(_PBR_NAMES["emission"], 1.0)
        metallic_value = pick(_PBR_NAMES["metallic"])
        rough_value = pick(_PBR_NAMES["roughness"])
        glossy = any(scalar([flag], 0.0) == 1.0 for flag in _GLOSSINESS_FLAGS)
        if metallic_value is not None:
            metallic = scalar(_PBR_NAMES["metallic"], 0.0)
        elif lambert:
            metallic = 0.0
        else:
            metallic = scalar(["ReflectionFactor"], 0.0)
        if rough_value is not None:
            roughness = scalar(_PBR_NAMES["roughness"], 1.0)
            roughness = 1.0 - roughness if glossy else roughness
        elif lambert:
            roughness = 1.0
        else:
            shininess = doc.number(entry, "ShininessExponent", doc.number(entry, "Shininess", 20.0))
            roughness = 1.0 - math.sqrt(min(max(shininess, 0.0), 100.0)) / 10.0
        material = Material(
            name=entry.name or "material",
            base_color=(float(np.clip(base[0], 0, 1)), float(np.clip(base[1], 0, 1)),
                        float(np.clip(base[2], 0, 1)), opacity),
            metallic=float(np.clip(metallic, 0.0, 1.0)),
            roughness=float(np.clip(roughness, 0.0, 1.0)),
            emissive=tuple(float(v) for v in np.clip(emissive, 0.0, None)),
        )
        slots: dict[str, tuple[Texture, int]] = {}
        for texture_entry, prop in doc.linked(entry.id, "Texture"):
            slot = _TEXTURE_SLOTS.get(prop)
            if slot is None or slot in slots:
                continue
            image = self.texture(texture_entry)
            if image is not None:
                slots[slot] = (image, texture_entry.id)
        if "base_color" in slots:
            material.base_color_texture = slots["base_color"][0]
            material.base_color = (1.0, 1.0, 1.0, opacity)
        if "normal" in slots:
            material.normal_texture = slots["normal"][0]
        elif "bump" in slots and self._looks_like_normals(slots["bump"][0]):
            material.normal_texture = slots["bump"][0]
        if "emissive" in slots:
            material.emissive_texture = slots["emissive"][0]
            if max(material.emissive) <= 0:
                material.emissive = (1.0, 1.0, 1.0)
        if any(slot in slots for slot in ("metallic", "roughness")):
            material.metallic_roughness_texture = self._combined(
                slots.get("metallic", (None, 0))[0], slots.get("roughness", (None, 0))[0], None, glossy)
            if "roughness" in slots and rough_value is None and \
                    "ShininessExponent" not in explicit and "Shininess" not in explicit:
                material.roughness = 1.0
            if "metallic" in slots and metallic_value is None and "ReflectionFactor" not in explicit:
                material.metallic = 1.0
        if "occlusion" in slots:
            material.occlusion_texture = slots["occlusion"][0]
        if "opacity" in slots:
            alpha_image, alpha_id = slots["opacity"]
            base_entry = slots.get("base_color")
            if base_entry is not None and (base_entry[1] == alpha_id or alpha_image is base_entry[0]):
                material.alpha_mode = "BLEND" if base_entry[0].image.shape[-1] == 4 else material.alpha_mode
            elif base_entry is not None:
                material.base_color_texture = self._with_alpha(base_entry[0], alpha_image)
                material.alpha_mode = "BLEND"
        if opacity < 0.999:
            material.alpha_mode = "BLEND"
        self.material_cache[entry.id] = material
        return material

    @staticmethod
    def _looks_like_normals(texture: Texture) -> bool:
        """Whether a bump map holds tangent-space normals rather than heights."""
        mean = texture.image[..., :3].reshape(-1, 3).float().mean(dim=0)
        return bool(mean[2] > 0.6 and abs(float(mean[0]) - 0.5) < 0.15 and abs(float(mean[1]) - 0.5) < 0.15)

    @staticmethod
    def _with_alpha(color: Texture, alpha: Texture) -> Texture:
        """A colour texture carrying another texture's brightness as alpha."""
        height, width = int(color.image.shape[0]), int(color.image.shape[1])
        gray = alpha.image[..., :3].float().mean(dim=-1)
        if gray.shape != (height, width):
            gray = torch.nn.functional.interpolate(gray[None, None], size=(height, width),
                                                   mode="bilinear", align_corners=False)[0, 0]
        image = torch.cat([color.image[..., :3].float(), gray.clamp(0, 1)[..., None]], dim=-1)
        return Texture(image=image.contiguous(), name=color.name, wrap_s=color.wrap_s,
                       wrap_t=color.wrap_t, mime="image/png")

    # Geometry

    @staticmethod
    def _layer(node: _Node | None, data_name: str, index_name: str, width: int,
               corner_cp: np.ndarray, corner_poly: np.ndarray, corners: int) -> _Layer | None:
        """One layer element spread to one row per polygon corner."""
        if node is None:
            return None
        data = node.array(data_name)
        if data is None:
            return None
        data = np.asarray(data, dtype=np.float64).reshape(-1)
        if data.size < width or data.size % width:
            if data.size >= width:
                data = data[: data.size - data.size % width]
            else:
                return None
        table = data.reshape(-1, width)
        if not np.isfinite(table).all():
            table = np.where(np.isfinite(table), table, 0.0)
        mapping = str(node.value("MappingInformationType", "ByPolygonVertex"))
        reference = str(node.value("ReferenceInformationType", "Direct"))
        if mapping in ("ByPolygonVertex",):
            slot = np.arange(corners)
        elif mapping in ("ByVertex", "ByVertice", "ByControlPoint"):
            slot = corner_cp
        elif mapping == "ByPolygon":
            slot = corner_poly
        elif mapping == "AllSame":
            slot = np.zeros(corners, dtype=np.int64)
        else:
            logger.warning("an FBX layer mapped %s was left out: only per corner, per vertex, per "
                           "polygon and uniform layers are read", mapping)
            return None
        if reference in ("IndexToDirect", "Index"):
            index = node.array(index_name)
            if index is None:
                return None
            index = np.asarray(index, dtype=np.int64).reshape(-1)
            if slot.size and (int(slot.max()) >= index.size):
                logger.warning("an FBX %s layer is shorter than its mesh and was left out", data_name)
                return None
            slot = index[slot]
        if slot.size and (int(slot.min()) < 0 or int(slot.max()) >= table.shape[0]):
            logger.warning("an FBX %s layer points past its own values and was left out", data_name)
            return None
        return _Layer(table[slot], width)

    def _skin(self, geometry: _Object, control_points: int, joint_index: dict[int, int]
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None] | None:
        """Per-control-point joints and weights of a skinned geometry, and its bind matrix."""
        entries = []
        bind = None
        for skin, _ in self.doc.linked(geometry.id, "Deformer", "Skin"):
            for cluster, _ in self.doc.linked(skin.id, "Deformer", "Cluster"):
                bones = [m for m, p in self.doc.linked(cluster.id, "Model") if not p]
                if not bones or bones[0].id not in joint_index:
                    continue
                link = _fbx_matrix(cluster.node.array("TransformLink"))
                transform = _fbx_matrix(cluster.node.array("Transform"))
                if bind is None and link is not None and transform is not None:
                    bind = link @ transform
                indexes = cluster.node.array("Indexes")
                weights = cluster.node.array("Weights")
                if indexes is None or weights is None:
                    continue
                indexes = np.asarray(indexes, dtype=np.int64).reshape(-1)
                weights = np.asarray(weights, dtype=np.float64).reshape(-1)
                count = min(indexes.size, weights.size)
                entries.append((joint_index[bones[0].id], indexes[:count], weights[:count]))
        if not entries:
            return None
        cps = np.concatenate([e[1] for e in entries])
        joints = np.concatenate([np.full(e[1].size, e[0], dtype=np.int64) for e in entries])
        weights = np.concatenate([e[2] for e in entries])
        valid = (cps >= 0) & (cps < control_points) & np.isfinite(weights) & (weights > 0)
        cps, joints, weights = cps[valid], joints[valid], weights[valid]
        table_joints = np.zeros((control_points, 4), dtype=np.int64)
        table_weights = np.zeros((control_points, 4), dtype=np.float64)
        if cps.size:
            order = np.lexsort((-weights, cps))
            cps, joints, weights = cps[order], joints[order], weights[order]
            starts = np.r_[0, np.nonzero(np.diff(cps))[0] + 1]
            group = np.repeat(np.arange(starts.size), np.diff(np.r_[starts, cps.size]))
            rank = np.arange(cps.size) - starts[group]
            width = 4 if int(rank.max()) < 4 else 8
            keep = rank < width
            table_joints = np.zeros((control_points, width), dtype=np.int64)
            table_weights = np.zeros((control_points, width), dtype=np.float64)
            table_joints[cps[keep], rank[keep]] = joints[keep]
            table_weights[cps[keep], rank[keep]] = weights[keep]
        total = table_weights.sum(axis=1, keepdims=True)
        weighted = total[:, 0] > 0
        table_weights = np.where(total > 0, table_weights / np.maximum(total, 1e-12), 0.0)
        return table_joints, table_weights, (bind, weighted)

    def _morphs(self, geometry: _Object, control_points: int) -> list[tuple[str, np.ndarray]]:
        """Shape keys of a geometry as per-control-point position offsets."""
        found = []
        for blend, _ in self.doc.linked(geometry.id, "Deformer", "BlendShape"):
            for channel, _ in self.doc.linked(blend.id, "Deformer", "BlendShapeChannel"):
                shapes = self.doc.linked(channel.id, "Geometry", "Shape")
                if not shapes:
                    continue
                shape = shapes[-1][0]
                indexes = shape.node.array("Indexes")
                vertices = shape.node.array("Vertices")
                if indexes is None or vertices is None:
                    continue
                indexes = np.asarray(indexes, dtype=np.int64).reshape(-1)
                vertices = np.asarray(vertices, dtype=np.float64).reshape(-1)
                count = min(indexes.size, vertices.size // 3)
                indexes, vertices = indexes[:count], vertices[:count * 3].reshape(-1, 3)
                valid = (indexes >= 0) & (indexes < control_points)
                offsets = np.zeros((control_points, 3), dtype=np.float64)
                offsets[indexes[valid]] = vertices[valid]
                name = channel.name or shape.name or f"shape_{len(found)}"
                found.append((name, offsets))
        return found

    def mesh_parts(self, model: _Object, geometry: _Object, joint_index: dict[int, int],
                   rigid_joint: int) -> list[Part]:
        """The parts one mesh Model contributes, one per material it uses."""
        node = geometry.node
        positions = node.array("Vertices")
        polygon_index = node.array("PolygonVertexIndex")
        if positions is None or polygon_index is None:
            return []
        if positions.size // 3 > MAX_MESH_ELEMENTS or polygon_index.size > MAX_MESH_ELEMENTS:
            raise FbxError(f"Mesh {model.name} holds more than {MAX_MESH_ELEMENTS} vertices or "
                           "corners. Decimate it before loading.")
        positions = np.asarray(positions, dtype=np.float64).reshape(-1)
        positions = positions[: positions.size - positions.size % 3].reshape(-1, 3)
        finite = np.isfinite(positions)
        if not finite.all():
            logger.warning("mesh %s of the FBX file holds positions that are not numbers; they "
                           "were set to 0", model.name or "(unnamed)")
            positions = np.where(finite, positions, 0.0)
        control_points = positions.shape[0]
        raw = np.asarray(polygon_index, dtype=np.int64).reshape(-1)
        ends = raw < 0
        if not ends.any():
            return []
        last = int(np.nonzero(ends)[0][-1])
        raw, ends = raw[: last + 1], ends[: last + 1]
        corner_cp = np.where(ends, -raw - 1, raw)
        if corner_cp.size and (int(corner_cp.min()) < 0 or int(corner_cp.max()) >= control_points):
            raise FbxError(f"Mesh {model.name} in the FBX file names vertices it does not have. "
                           "The file is damaged.")
        end_at = np.nonzero(ends)[0]
        starts = np.r_[0, end_at[:-1] + 1]
        sizes = end_at - starts + 1
        corners = corner_cp.size
        corner_poly = np.repeat(np.arange(sizes.size), sizes)

        def first_layer(kind: str) -> _Node | None:
            layers = node.find_all(kind)
            if not layers:
                return None
            return min(layers, key=lambda n: n.props[0] if n.props and isinstance(n.props[0], int) else 0)

        normals = self._layer(first_layer("LayerElementNormal"), "Normals", "NormalsIndex", 3,
                              corner_cp, corner_poly, corners)
        uvs = self._layer(first_layer("LayerElementUV"), "UV", "UVIndex", 2, corner_cp,
                          corner_poly, corners)
        colors = self._layer(first_layer("LayerElementColor"), "Colors", "ColorIndex", 4,
                             corner_cp, corner_poly, corners)
        material_node = first_layer("LayerElementMaterial")
        polygon_material = np.zeros(sizes.size, dtype=np.int64)
        if material_node is not None:
            values = material_node.array("Materials")
            mapping = str(material_node.value("MappingInformationType", "AllSame"))
            if values is not None and values.size:
                values = np.asarray(values, dtype=np.int64).reshape(-1)
                if mapping == "ByPolygon" and values.size >= sizes.size:
                    polygon_material = values[: sizes.size]
                else:
                    polygon_material[:] = values[0]

        skin = self._skin(geometry, control_points, joint_index)
        if skin is not None:
            joint_table, weight_table, (bind, weighted) = skin
            transform = bind if bind is not None else self.global_matrix(model.id) @ self.geometric(model.id)
            if rigid_joint >= 0:
                joint_table[~weighted, 0] = rigid_joint
            weight_table[~weighted, 0] = 1.0
        else:
            transform = self.global_matrix(model.id) @ self.geometric(model.id)
            joint_table = weight_table = None
            if rigid_joint >= 0:
                joint_table = np.full((control_points, 4), 0, dtype=np.int64)
                joint_table[:, 0] = rigid_joint
                weight_table = np.zeros((control_points, 4), dtype=np.float64)
                weight_table[:, 0] = 1.0
        matrix = self.conversion @ transform
        linear = matrix[:3, :3]
        placed = positions @ linear.T + matrix[:3, 3]
        if np.linalg.det(linear) < 0:
            local = np.arange(corners) - starts[corner_poly]
            reorder = starts[corner_poly] + (sizes[corner_poly] - 1 - local)
            corner_cp = corner_cp[reorder]
            normals = _Layer(normals.values[reorder], 3) if normals is not None else None
            uvs = _Layer(uvs.values[reorder], 2) if uvs is not None else None
            colors = _Layer(colors.values[reorder], 4) if colors is not None else None
        normal_values = None
        if normals is not None:
            normal_matrix = _inverse(linear).T
            normal_values = normals.values @ normal_matrix.T
            normal_values /= np.linalg.norm(normal_values, axis=1, keepdims=True).clip(min=1e-12)
        uv_values = None
        if uvs is not None:
            uv_values = uvs.values.copy()
            uv_values[:, 1] = 1.0 - uv_values[:, 1]
        color_values = None
        if colors is not None:
            color_values = colors.values.copy()
            color_values[:, :3] = _srgb_to_linear(np.clip(color_values[:, :3], 0.0, None))
        morphs = [(name, offsets @ linear.T) for name, offsets in self._morphs(geometry, control_points)]

        slots = [entry for entry, prop in self.doc.linked(model.id, "Material") if not prop]
        parts = []
        used = np.unique(polygon_material)
        for slot in used:
            chosen = polygon_material == slot
            valid = chosen & (sizes >= 3)
            if not valid.any():
                continue
            corner_mask = np.repeat(valid, sizes)
            picked = np.nonzero(corner_mask)[0]
            sub_sizes = sizes[valid]
            columns = [corner_cp[picked].astype(np.int32)[:, None]]
            if normal_values is not None:
                columns.append(_bits(normal_values[picked]))
            if uv_values is not None:
                columns.append(_bits(uv_values[picked]))
            if color_values is not None:
                columns.append(_bits(color_values[picked]))
            first, vertex_of = _unique_rows(np.concatenate(columns, axis=1))
            source = picked[first]
            cps = corner_cp[source]
            vertices = placed[cps]
            triangles, owner = _triangulate(vertices[vertex_of], sub_sizes, self.clip_work)
            faces = vertex_of[triangles]
            polygons = None
            if (sub_sizes != 3).any():
                polygons = Polygons(torch.from_numpy(sub_sizes.astype(np.int64)),
                                    torch.from_numpy(vertex_of.astype(np.int64)),
                                    torch.from_numpy(owner.astype(np.int64)))
            material_entry = slots[int(slot)] if 0 <= int(slot) < len(slots) else None
            material = self.material(material_entry)
            name = model.name or geometry.name or f"mesh_{len(parts)}"
            if len(used) > 1:
                name = f"{name}_{material.name}"
            part = Part(
                name=name,
                vertices=torch.from_numpy(vertices.astype(np.float32)),
                faces=torch.from_numpy(faces.astype(np.int64)),
                material=material,
                polygons=polygons,
            )
            if normal_values is not None:
                part.normals = torch.from_numpy(normal_values[source].astype(np.float32))
            if uv_values is not None:
                part.uvs = torch.from_numpy(uv_values[source].astype(np.float32))
            if color_values is not None:
                part.colors = torch.from_numpy(color_values[source].astype(np.float32))
            if joint_table is not None:
                part.joints = torch.from_numpy(joint_table[cps].astype(np.int64))
                part.weights = torch.from_numpy(weight_table[cps].astype(np.float32))
            for morph_name, offsets in morphs:
                part.morphs[morph_name] = MorphTarget(torch.from_numpy(offsets[cps].astype(np.float32)))
            parts.append(part)
        return parts

    # Skeleton

    def skeleton(self) -> tuple[Skeleton | None, dict[int, int]]:
        """The joints of the file, parents first, and the joint index of each joint Model."""
        doc = self.doc
        links: dict[int, np.ndarray] = {}
        for entry in doc.objects.values():
            if entry.kind != "Deformer" or entry.sub != "Cluster":
                continue
            bones = [m for m, p in doc.linked(entry.id, "Model") if not p]
            link = _fbx_matrix(entry.node.array("TransformLink"))
            if bones and link is not None:
                links.setdefault(bones[0].id, link)
        members = {key for key, entry in self.models.items() if entry.sub in ("LimbNode", "Root")}
        members |= set(links)
        if not members:
            return None, {}
        pose: dict[int, np.ndarray] = {}
        for entry in doc.objects.values():
            if entry.kind == "Pose" and (entry.sub == "BindPose" or entry.node.value("Type") == "BindPose"):
                for pose_node in entry.node.find_all("PoseNode"):
                    target = pose_node.value("Node")
                    matrix = _fbx_matrix(pose_node.array("Matrix"))
                    if isinstance(target, (int, np.integer)) and matrix is not None:
                        pose.setdefault(int(target), matrix)
        parent_joint: dict[int, int] = {}
        for key in members:
            parent = 0
            for ancestor in reversed(self._chain(key)[:-1]):
                if ancestor in members:
                    parent = ancestor
                    break
            parent_joint[key] = parent
        ordered: list[int] = []
        placed: set[int] = set()
        for key in (k for k in doc.objects if k in members):
            pending = []
            while key in members and key not in placed:
                pending.append(key)
                placed.add(key)
                key = parent_joint[key]
            ordered.extend(reversed(pending))
        index = {key: position for position, key in enumerate(ordered)}
        rest = np.zeros((len(ordered), 4, 4))
        for key, position in index.items():
            if key in links:
                matrix = links[key]
            elif key in pose:
                matrix = pose[key]
            else:
                matrix = self.global_matrix(key)
            rest[position] = self._to_model_space(matrix)
        singular = np.abs(np.linalg.det(rest[:, :3, :3])) < 1e-12
        if singular.any():
            logger.warning("%d joints of the FBX file have a zero scale at the bind pose; their "
                           "scale was reset to 1", int(singular.sum()))
            rest[singular, :3, :3] = np.eye(3)
        parents = []
        for position, key in enumerate(ordered):
            parent = index.get(parent_joint[key], -1)
            parents.append(parent if parent < position else -1)
        inverse = _inverse(rest)
        skeleton = Skeleton(
            names=[self.models[key].name for key in ordered],
            parents=torch.tensor(parents, dtype=torch.int64),
            rest=torch.from_numpy(rest.astype(np.float32)),
            inverse_bind=torch.from_numpy(inverse.astype(np.float32)),
        )
        return skeleton, index

    # Animation

    def _frame_rate(self) -> float:
        """Frames per second the file declares, 30 when it declares none."""
        settings = self.doc.settings
        try:
            mode = int(settings.get("TimeMode", [0])[0])
        except (TypeError, ValueError, IndexError, OverflowError):
            mode = 0
        if mode == 14:
            try:
                custom = float(settings.get("CustomFrameRate", [30.0])[0])
                if 1.0 <= custom <= 1000.0:
                    return custom
            except (TypeError, ValueError, IndexError):
                pass
        return _TIME_MODES.get(mode, 30.0)

    @staticmethod
    def _evaluate(curve: _Object, times: np.ndarray) -> np.ndarray | None:
        """Values of an AnimationCurve at ``times`` in ticks."""
        key_times = curve.node.array("KeyTime")
        key_values = curve.node.array("KeyValueFloat")
        if key_times is None or key_values is None:
            return None
        key_times = np.asarray(key_times, dtype=np.int64).reshape(-1)
        key_values = np.asarray(key_values, dtype=np.float64).reshape(-1)
        count = min(key_times.size, key_values.size)
        if count == 0:
            return None
        key_times, key_values = key_times[:count], key_values[:count]
        finite = np.isfinite(key_values)
        if not finite.all():
            key_times, key_values = key_times[finite], key_values[finite]
            count = key_times.size
            if count == 0:
                return None
        order = np.argsort(key_times, kind="stable")
        key_times, key_values = key_times[order], key_values[order]
        if count == 1:
            return np.full(times.shape, key_values[0])
        flags = np.full(count, 0x04, dtype=np.int64)
        attr_flags = curve.node.array("KeyAttrFlags")
        ref_counts = curve.node.array("KeyAttrRefCount")
        if attr_flags is not None and ref_counts is not None:
            attr_flags = np.asarray(attr_flags, dtype=np.int64).reshape(-1)
            ref_counts = np.asarray(ref_counts, dtype=np.int64).reshape(-1)
            usable = min(attr_flags.size, ref_counts.size)
            ref_counts = np.clip(ref_counts[:usable], 0, count)
            if usable and int(ref_counts.sum()) == count:
                flags = np.repeat(attr_flags[:usable], ref_counts)[order]
        seconds = key_times.astype(np.float64)
        sample = times.astype(np.float64)
        right = np.clip(np.searchsorted(seconds, sample, side="right"), 1, count - 1)
        left = right - 1
        span = np.maximum(seconds[right] - seconds[left], 1e-9)
        t = np.clip((sample - seconds[left]) / span, 0.0, 1.0)
        v0, v1 = key_values[left], key_values[right]
        result = v0 + (v1 - v0) * t
        kind = flags[left]
        constant = (kind & 0x02) != 0
        cubic = (kind & 0x08) != 0
        if cubic.any():
            before = np.clip(left - 1, 0, count - 1)
            after = np.clip(right + 1, 0, count - 1)
            span_in = np.maximum(seconds[right] - seconds[before], 1e-9)
            span_out = np.maximum(seconds[after] - seconds[left], 1e-9)
            m0 = (key_values[right] - key_values[before]) / span_in * span
            m1 = (key_values[after] - key_values[left]) / span_out * span
            m0 = np.where(left == 0, v1 - v0, m0)
            m1 = np.where(right == count - 1, v1 - v0, m1)
            t2, t3 = t * t, t * t * t
            hermite = ((2 * t3 - 3 * t2 + 1) * v0 + (t3 - 2 * t2 + t) * m0
                       + (-2 * t3 + 3 * t2) * v1 + (t3 - t2) * m1)
            result = np.where(cubic, hermite, result)
        result = np.where(constant & (sample < seconds[right]), v0, result)
        result = np.where(sample <= seconds[0], key_values[0], result)
        result = np.where(sample >= seconds[-1], key_values[-1], result)
        return result

    def animations(self, skeleton: Skeleton, index: dict[int, int]) -> list[Animation]:
        """Every AnimationStack as a clip of joint channels resampled to linear keys."""
        doc = self.doc
        stacks = [entry for entry in doc.objects.values() if entry.kind == "AnimationStack"]
        if not stacks or not index:
            return []
        rest_local = skeleton.local_rest().double().numpy()
        rest_t, rest_r, rest_s = _decompose(rest_local)
        rest_q = _quaternion(rest_r)
        joint_keys = sorted(index, key=index.get)
        needed: list[int] = []
        seen = set()
        for key in joint_keys:
            for node in self._chain(key):
                if node not in seen:
                    seen.add(node)
                    needed.append(node)
        depth = {key: len(self._chain(key)) for key in needed}
        needed.sort(key=lambda k: depth[k])
        clips = []
        fps = self._frame_rate()
        for stack in stacks:
            layers = [entry for entry, prop in doc.linked(stack.id, "AnimationLayer") if not prop]
            if not layers:
                continue
            curves: dict[int, dict[str, tuple[_Object, dict[str, _Object]]]] = defaultdict(dict)
            low, high = None, None
            for curve_node, _ in doc.linked(layers[0].id, "AnimationCurveNode"):
                targets = [(m, p) for m, p in doc.owners(curve_node.id, "Model") if p]
                if not targets:
                    continue
                model, prop = targets[0]
                channel = {"Lcl Translation": "T", "Lcl Rotation": "R", "Lcl Scaling": "S"}.get(prop)
                if channel is None or model.id not in seen:
                    continue
                parts = {}
                for curve, component in doc.linked(curve_node.id, "AnimationCurve"):
                    parts[component] = curve
                    key_times = curve.node.array("KeyTime")
                    if key_times is not None and key_times.size:
                        key_times = np.asarray(key_times, dtype=np.int64)
                        low = int(key_times.min()) if low is None else min(low, int(key_times.min()))
                        high = int(key_times.max()) if high is None else max(high, int(key_times.max()))
                curves[model.id][channel] = (curve_node, parts)
            if not curves or low is None:
                continue
            start = doc.number(stack, "LocalStart", float("nan"))
            stop = doc.number(stack, "LocalStop", float("nan"))
            if not (math.isfinite(start) and math.isfinite(stop) and stop > start):
                start, stop = float(low), float(high)
            duration = (stop - start) / TICKS_PER_SECOND
            rate = fps
            frames = int(round(duration * rate)) + 1
            if frames > MAX_ANIMATION_FRAMES:
                rate = (MAX_ANIMATION_FRAMES - 1) / max(duration, 1e-9)
                frames = MAX_ANIMATION_FRAMES
            if frames * max(len(needed), 1) > MAX_ANIMATION_SAMPLES:
                frames = max(2, MAX_ANIMATION_SAMPLES // max(len(needed), 1))
                rate = (frames - 1) / max(duration, 1e-9)
                logger.warning("animation %s of the FBX file is long for its %d joints, so it was "
                               "resampled at %.2f frames per second", stack.name or "(unnamed)",
                               len(needed), rate)
            seconds = np.arange(frames, dtype=np.float64) / rate
            ticks = start + seconds * TICKS_PER_SECOND
            world: dict[int, np.ndarray] = {}
            scales: dict[int, np.ndarray] = {}
            for key in needed:
                parts = self._components(self.models[key])
                animated = curves.get(key, {})
                values = {}
                for channel in ("T", "R", "S"):
                    base = np.broadcast_to(parts[channel], (frames, 3)).copy()
                    if channel in animated:
                        curve_node, components = animated[channel]
                        for axis, label in enumerate(("d|X", "d|Y", "d|Z")):
                            default = curve_node.props.get(label)
                            if default:
                                try:
                                    base[:, axis] = float(default[0])
                                except (TypeError, ValueError):
                                    pass
                            if label in components:
                                sampled = self._evaluate(components[label], ticks)
                                if sampled is not None:
                                    base[:, axis] = sampled
                    values[channel] = base
                local = self._local(parts, values["T"], values["R"], values["S"])
                parent = self.parent_of.get(key, 0)
                if parent in world:
                    world[key] = self._inherit(world[parent], scales[parent], parts, local,
                                               values["R"], values["S"])
                else:
                    world[key] = local
                scales[key] = values["S"]
            converted = {}
            for key in joint_keys:
                converted[key] = self.conversion @ world.pop(key) @ self.conversion_inverse
            world.clear()
            scales.clear()
            channels = []
            for key in joint_keys:
                joint = index[key]
                parent = int(skeleton.parents[joint])
                matrix = converted[key]
                if parent >= 0:
                    matrix = _inverse(converted[joint_keys[parent]]) @ matrix
                translation, rotation, scale = _decompose(matrix)
                quaternion = _quaternion(rotation)
                if frames > 1:
                    turns = np.where((quaternion[1:] * quaternion[:-1]).sum(axis=1) < 0, -1.0, 1.0)
                    quaternion[1:] *= np.cumprod(turns)[:, None]
                times = torch.from_numpy(seconds.astype(np.float32))
                extent = max(float(np.abs(rest_t).max()), 1e-3)
                if not np.allclose(translation, rest_t[joint], atol=1e-5 * extent):
                    channels.append(Channel(joint, "translation", times,
                                            torch.from_numpy(translation.astype(np.float32))))
                if not (np.abs(np.abs(quaternion @ rest_q[joint]) - 1.0) < 1e-7).all():
                    channels.append(Channel(joint, "rotation", times,
                                            torch.from_numpy(quaternion.astype(np.float32))))
                if not np.allclose(scale, rest_s[joint], atol=1e-5):
                    channels.append(Channel(joint, "scale", times,
                                            torch.from_numpy(scale.astype(np.float32))))
            if channels:
                clips.append(Animation(name=stack.name or f"animation_{len(clips)}", channels=channels))
        return clips

    # Whole model

    def build(self) -> Model3D:
        """The model the document describes."""
        skeleton, index = self.skeleton()
        parts: list[Part] = []
        for key, model in self.models.items():
            geometries = [g for g, p in self.doc.linked(key, "Geometry") if not p and g.sub in ("Mesh", "")]
            if not geometries:
                continue
            rigid = -1
            if skeleton is not None:
                for ancestor in reversed(self._chain(key)[:-1]):
                    if ancestor in index:
                        rigid = index[ancestor]
                        break
            parts.extend(self.mesh_parts(model, geometries[0], index, rigid))
        if skeleton is not None and not any(part.weights is not None for part in parts) and \
                not any(entry.sub in ("LimbNode", "Root") for entry in self.models.values()):
            skeleton = None
        model = Model3D(parts=parts, skeleton=skeleton, source_format="fbx")
        if skeleton is not None:
            for part in parts:
                if part.joints is not None and skeleton.count:
                    part.joints = part.joints.clamp(0, skeleton.count - 1)
            model.animations = self.animations(skeleton, index)
        header = self.doc.top.get("FBXHeaderExtension")
        creator = self.doc.top.get("Creator")
        meta: dict[str, Any] = {}
        if self.name:
            meta["name"] = PurePosixPath(self.name.replace("\\", "/")).stem
        info = header.find("SceneInfo") if header is not None else None
        application = ""
        if info is not None:
            data = info.find("MetaData")
            if data is not None:
                title = data.value("Title")
                author = data.value("Author")
                if isinstance(title, str) and title.strip():
                    meta["name"] = title.strip()
                if isinstance(author, str) and author.strip():
                    meta["authors"] = [author.strip()]
            values = _properties(info)
            for key in ("LastSaved|ApplicationName", "Original|ApplicationName"):
                if values.get(key):
                    application = str(values[key][0])
                    break
        model.meta = meta
        model.extras["fbx"] = {
            "version": self.version,
            "creator": str(creator.props[0]) if creator is not None and creator.props else "",
            "application": application,
        }
        return model


def read(data: bytes, base_dir: str | None = None, name: str = "") -> Model3D:
    """A model from a binary or ASCII FBX 7 file.

    Args:
        data: The file's bytes.
        base_dir: Folder holding the file, for textures it names beside itself, or None.
        name: File name, used for the model name and a ``<stem>.fbm`` texture folder.

    Returns:
        The model in metres, Y up, facing +Z. Quads and n-gons are kept in ``Part.polygons``.
        Animations are resampled to linear keys at the file's frame rate.

    Raises:
        FbxError: The bytes are not FBX 7, are damaged, or declare more than can be held.
    """
    version, roots, _ = _parse(bytes(data) if not isinstance(data, bytes) else data)
    doc = _Document(roots)
    if "Objects" not in doc.top:
        raise FbxError("The FBX file has no Objects section, so it holds no model.")
    model = _SceneReader(doc, version, base_dir, name).build()
    if not model.parts:
        logger.warning("the FBX file %s holds no polygon meshes connected to its scene",
                       PurePosixPath(name.replace("\\", "/")).name if name else "")
    logger.debug("read FBX %d: %d parts, %d vertices, %d joints, %d animations", version,
                 len(model.parts), model.vertex_count,
                 model.skeleton.count if model.skeleton is not None else 0, len(model.animations))
    return model


# ---------------------------------------------------------------------------------------------
# Writer


class _Int64(int):
    """An integer written as a 64-bit FBX value."""


def _encode_value(value: Any, out: bytearray) -> None:
    """Append one binary property value.

    Args:
        value: str, bytes, bool, int, float, `_Int64` or a numpy array.
        out: The buffer written to.
    """
    if isinstance(value, np.ndarray):
        code = {np.dtype(np.float64): b"d", np.dtype(np.float32): b"f", np.dtype(np.int32): b"i",
                np.dtype(np.int64): b"l", np.dtype(np.bool_): b"b"}.get(value.dtype)
        if code is None:
            raise FbxError(f"An array of {value.dtype} cannot be written to FBX.")
        raw = np.ascontiguousarray(value).astype(value.dtype.newbyteorder("<"), copy=False).tobytes()
        if len(raw) > 128:
            packed = zlib.compress(raw, 6)
            out += code + struct.pack("<III", value.size, 1, len(packed)) + packed
        else:
            out += code + struct.pack("<III", value.size, 0, len(raw)) + raw
    elif isinstance(value, bool):
        out += b"C" + struct.pack("<?", value)
    elif isinstance(value, _Int64):
        out += b"L" + struct.pack("<q", int(value))
    elif isinstance(value, int):
        if -2 ** 31 <= value < 2 ** 31:
            out += b"I" + struct.pack("<i", value)
        else:
            out += b"L" + struct.pack("<q", value)
    elif isinstance(value, float):
        out += b"D" + struct.pack("<d", value)
    elif isinstance(value, str):
        raw = value.encode("utf-8")
        out += b"S" + struct.pack("<I", len(raw)) + raw
    elif isinstance(value, (bytes, bytearray)):
        out += b"R" + struct.pack("<I", len(value)) + bytes(value)
    else:
        raise FbxError(f"A {type(value).__name__} value cannot be written to FBX.")


def _encode_node(node: _Node, out: bytearray) -> None:
    """Append one binary FBX 7.4 record and everything nested in it."""
    start = len(out)
    out += b"\x00" * 12
    name = node.name.encode("ascii")
    out += bytes([len(name)]) + name
    props_start = len(out)
    for value in node.props:
        _encode_value(value, out)
    props_length = len(out) - props_start
    for child in node.children:
        _encode_node(child, out)
    if node.children or not node.props:
        out += b"\x00" * 13
    if len(out) >= 2 ** 32:
        raise FbxError("The model is too large for an FBX 7.4 file, which holds at most 4 GB.")
    struct.pack_into("<III", out, start, len(out), len(node.props), props_length)


def _n(name: str, *props: Any, children: list[_Node] | None = None) -> _Node:
    """A record to write."""
    return _Node(name, list(props), children or [])


def _p(name: str, kind: str, label: str, flags: str, *values: Any) -> _Node:
    """A ``Properties70`` entry to write."""
    return _Node("P", [name, kind, label, flags, *values])


def _props70(*entries: _Node) -> _Node:
    """A ``Properties70`` block holding ``entries``."""
    return _Node("Properties70", [], list(entries))


def _class_name(name: str, kind: str) -> str:
    """The ``name\\x00\\x01Class`` string binary FBX stores for an object."""
    return f"{name.replace(chr(0), '').replace(chr(1), '')}\x00\x01{kind}"


def _vector(values: Any) -> tuple[float, float, float]:
    """Three floats."""
    return tuple(float(v) for v in list(values)[:3])


def _safe_file_name(text: str) -> str:
    """A file name made only of letters, digits, dots, dashes and underscores."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    return cleaned[:80] or "texture"


class _TexturePlan:
    """The texture files a model writes, by texture slot."""

    def __init__(self, model: Model3D):
        self.entries: list[dict[str, Any]] = []
        self.by_material: dict[int, list[dict[str, Any]]] = {}
        used_names: set[str] = set()
        cache: dict[tuple, dict[str, Any]] = {}
        for material in self._materials(model):
            listed = []
            for prop, texture, channel in self._slots(material):
                key = (id(texture), channel)
                entry = cache.get(key)
                if entry is None:
                    label = PurePosixPath(texture.name.replace("\\", "/")).stem if texture.name else \
                        f"{material.name}_{prop}"
                    stem = _safe_file_name(label if channel is None else f"{label}_{channel}")
                    base, number = stem, 1
                    while stem.lower() in used_names:
                        number += 1
                        stem = f"{base}_{number}"
                    used_names.add(stem.lower())
                    entry = {"texture": texture, "channel": channel, "stem": stem, "data": None,
                             "file": ""}
                    cache[key] = entry
                    self.entries.append(entry)
                listed.append((prop, entry))
            self.by_material[id(material)] = listed

    @staticmethod
    def _materials(model: Model3D) -> list[Material]:
        """Distinct materials of a model, in part order."""
        seen, found = set(), []
        for part in model.parts:
            if id(part.material) not in seen:
                seen.add(id(part.material))
                found.append(part.material)
        return found

    @staticmethod
    def _slots(material: Material) -> list[tuple[str, Texture, str | None]]:
        """``(FBX property, texture, channel)`` for each texture a material writes."""
        found = []
        if material.base_color_texture is not None:
            found.append(("DiffuseColor", material.base_color_texture, None))
            if material.alpha_mode != "OPAQUE" and material.base_color_texture.image.shape[-1] == 4:
                found.append(("TransparentColor", material.base_color_texture, None))
        if material.normal_texture is not None:
            found.append(("NormalMap", material.normal_texture, None))
        if material.emissive_texture is not None:
            found.append(("EmissiveColor", material.emissive_texture, None))
        if material.metallic_roughness_texture is not None:
            found.append(("ReflectionFactor", material.metallic_roughness_texture, "metallic"))
            found.append(("ShininessExponent", material.metallic_roughness_texture, "roughness"))
        return found

    def encode(self) -> None:
        """Encode every planned texture."""
        for entry in self.entries:
            if entry["data"] is not None:
                continue
            texture = entry["texture"]
            if entry["channel"] is not None:
                column = 2 if entry["channel"] == "metallic" else 1
                image = texture.image[..., column:column + 1] if texture.image.shape[-1] > column \
                    else texture.image[..., :1]
                texture = Texture(image=image.contiguous(), name=texture.name, mime="image/png")
            data, mime = texture_io.encode(texture)
            entry["data"] = data
            entry["file"] = entry["stem"] + (".jpg" if mime == "image/jpeg" else ".png")


def texture_files(model: Model3D) -> dict[str, bytes]:
    """The texture files an FBX written by `write` names, for saving beside it.

    Args:
        model: The model to be written.

    Returns:
        File name against encoded PNG or JPEG bytes.
    """
    plan = _TexturePlan(model)
    plan.encode()
    return {entry["file"]: entry["data"] for entry in plan.entries}


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area-weighted unit vertex normals."""
    normals = np.zeros_like(vertices)
    if faces.size:
        a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
        face_normals = np.cross(b - a, c - a)
        for column in range(3):
            np.add.at(normals, faces[:, column], face_normals)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    return np.where(length > 1e-20, normals / np.maximum(length, 1e-20), np.array([0.0, 1.0, 0.0]))


class _Writer:
    """Builds the records of a binary FBX 7.4 file from a `Model3D`."""

    def __init__(self, model: Model3D, embed: bool):
        self.model = model
        self.embed = embed
        self.next_id = 1_000_000
        self.objects: list[_Node] = []
        self.connections: list[_Node] = []
        self.counts: dict[str, int] = defaultdict(int)

    def _id(self) -> _Int64:
        """A fresh object id."""
        self.next_id += 1
        return _Int64(self.next_id)

    def _connect(self, child: int, parent: int, prop: str = "") -> None:
        """Record one connection."""
        if prop:
            self.connections.append(_n("C", "OP", _Int64(child), _Int64(parent), prop))
        else:
            self.connections.append(_n("C", "OO", _Int64(child), _Int64(parent)))

    def _add(self, kind: str, node: _Node) -> None:
        """Record one object."""
        self.objects.append(node)
        self.counts[kind] += 1

    def _model_node(self, ident: int, name: str, sub: str, translation, rotation, scale) -> _Node:
        """A Model object."""
        return _n("Model", _Int64(ident), _class_name(name, "Model"), sub, children=[
            _n("Version", 232),
            _props70(
                _p("RotationOrder", "enum", "", "", 0),
                _p("InheritType", "enum", "", "", 1),
                _p("DefaultAttributeIndex", "int", "Integer", "", 0),
                _p("Lcl Translation", "Lcl Translation", "", "A", *_vector(translation)),
                _p("Lcl Rotation", "Lcl Rotation", "", "A", *_vector(rotation)),
                _p("Lcl Scaling", "Lcl Scaling", "", "A", *_vector(scale)),
            ),
            _n("Shading", True),
            _n("Culling", "CullingOff"),
        ])

    def _materials(self, plan: _TexturePlan) -> dict[int, int]:
        """Write every material and texture; return the object id of each material."""
        material_ids: dict[int, int] = {}
        texture_ids: dict[int, int] = {}
        for material in _TexturePlan._materials(self.model):
            ident = self._id()
            material_ids[id(material)] = ident
            base = material.base_color
            roughness = float(np.clip(material.roughness, 0.0, 1.0))
            shininess = ((1.0 - roughness) * 10.0) ** 2
            self._add("Material", _n("Material", ident, _class_name(material.name, "Material"), "", children=[
                _n("Version", 102),
                _n("ShadingModel", "phong"),
                _n("MultiLayer", 0),
                _props70(
                    _p("ShadingModel", "KString", "", "", "Phong"),
                    _p("EmissiveColor", "Color", "", "A", *_vector(material.emissive)),
                    _p("EmissiveFactor", "Number", "", "A", 1.0),
                    _p("AmbientColor", "Color", "", "A", 0.0, 0.0, 0.0),
                    _p("DiffuseColor", "Color", "", "A", float(base[0]), float(base[1]), float(base[2])),
                    _p("DiffuseFactor", "Number", "", "A", 1.0),
                    _p("TransparentColor", "Color", "", "A", 1.0, 1.0, 1.0),
                    _p("TransparencyFactor", "Number", "", "A", float(1.0 - base[3])),
                    _p("Opacity", "Number", "", "A", float(base[3])),
                    _p("SpecularColor", "Color", "", "A", 0.2, 0.2, 0.2),
                    _p("SpecularFactor", "Number", "", "A", 1.0),
                    _p("ShininessExponent", "Number", "", "A", shininess),
                    _p("Shininess", "Number", "", "A", shininess),
                    _p("ReflectionColor", "Color", "", "A", 1.0, 1.0, 1.0),
                    _p("ReflectionFactor", "Number", "", "A", float(np.clip(material.metallic, 0.0, 1.0))),
                ),
            ]))
            for prop, entry in plan.by_material.get(id(material), []):
                key = id(entry)
                if key not in texture_ids:
                    texture_ids[key] = self._texture(entry)
                self._connect(texture_ids[key], ident, prop)
        return material_ids

    def _texture(self, entry: dict[str, Any]) -> int:
        """Write one Texture and its Video; return the Texture id."""
        texture_id, video_id = self._id(), self._id()
        name = entry["stem"]
        file_name = entry["file"]
        wraps = {"clamp": 1}
        source = entry["texture"]
        self._add("Video", _n("Video", video_id, _class_name(name, "Video"), "Clip", children=[
            _n("Type", "Clip"),
            _props70(_p("Path", "KString", "XRefUrl", "", file_name)),
            _n("UseMipMap", 0),
            _n("Filename", file_name),
            _n("RelativeFilename", file_name),
            _n("Content", entry["data"] if self.embed else b""),
        ]))
        self._add("Texture", _n("Texture", texture_id, _class_name(name, "Texture"), "", children=[
            _n("Type", "TextureVideoClip"),
            _n("Version", 202),
            _n("TextureName", _class_name(name, "Texture")),
            _props70(
                _p("UseMaterial", "bool", "", "", 1),
                _p("WrapModeU", "enum", "", "", wraps.get(source.wrap_s, 0)),
                _p("WrapModeV", "enum", "", "", wraps.get(source.wrap_t, 0)),
            ),
            _n("Media", _class_name(name, "Video")),
            _n("FileName", file_name),
            _n("RelativeFilename", file_name),
            _n("ModelUVTranslation", 0.0, 0.0),
            _n("ModelUVScaling", 1.0, 1.0),
            _n("Texture_Alpha_Source", "None"),
            _n("Cropping", 0, 0, 0, 0),
        ]))
        self._connect(video_id, texture_id)
        return texture_id

    def _skeleton(self) -> tuple[list[int], np.ndarray]:
        """Write the joints; return their Model ids and centimetre bind matrices."""
        skeleton = self.model.skeleton
        if skeleton is None or skeleton.count == 0:
            return [], np.zeros((0, 4, 4))
        bind = skeleton.inverse_bind.double().numpy()
        try:
            rest = np.linalg.inv(bind)
        except np.linalg.LinAlgError:
            rest = skeleton.rest.double().numpy().copy()
        rest[:, :3, 3] *= 100.0
        local = skeleton.local_rest().double().numpy().copy()
        local[:, :3, 3] *= 100.0
        translation, rotation, scale = _decompose(local)
        angles = _euler_xyz(rotation)
        ids = []
        for joint in range(skeleton.count):
            model_id, attribute_id = self._id(), self._id()
            ids.append(model_id)
            name = skeleton.names[joint] or f"joint_{joint}"
            self._add("NodeAttribute", _n("NodeAttribute", attribute_id,
                                          _class_name(name, "NodeAttribute"), "LimbNode", children=[
                _props70(_p("Size", "double", "Number", "", 100.0)),
                _n("TypeFlags", "Skeleton"),
            ]))
            self._add("Model", self._model_node(model_id, name, "LimbNode", translation[joint],
                                                angles[joint], scale[joint]))
            self._connect(attribute_id, model_id)
        for joint in range(skeleton.count):
            parent = int(skeleton.parents[joint])
            self._connect(ids[joint], ids[parent] if 0 <= parent < len(ids) and parent != joint else 0)
        return ids, rest

    def _part(self, part: Part, index: int, material_id: int, joint_ids: list[int],
              rest: np.ndarray, mesh_ids: list[int]) -> None:
        """Write one part as a mesh Model, its Geometry, skin and shape keys."""
        vertices = part.vertices.double().numpy()
        count = vertices.shape[0]
        faces = part.faces.long().numpy()
        if count == 0 or faces.size == 0:
            return
        if faces.ndim != 2 or faces.shape[1] != 3 or int(faces.max()) >= count or int(faces.min()) < 0:
            raise FbxError(f"Part {part.name} has faces naming vertices it does not hold.")
        for attribute, tensor in part.vertex_attributes().items():
            if tensor.shape[0] != count:
                raise FbxError(f"Part {part.name} holds {tensor.shape[0]} {attribute} for {count} "
                               "vertices. Rebuild the part so every vertex has one of each.")
        for morph_name, morph in part.morphs.items():
            if morph.positions.shape[0] != count:
                raise FbxError(f"Shape key {morph_name} of part {part.name} does not hold one offset "
                               "per vertex. Rebuild the shape key.")
        polygons = part.polygons
        sizes, corners = np.full(faces.shape[0], 3), faces.reshape(-1)
        if polygons is not None:
            polygon_sizes = polygons.sizes.long().numpy().reshape(-1)
            polygon_corners = polygons.indices.long().numpy().reshape(-1)
            if (polygon_sizes.size and int(polygon_sizes.min()) >= 3
                    and int(polygon_sizes.sum()) == polygon_corners.size
                    and int(polygon_corners.min()) >= 0 and int(polygon_corners.max()) < count
                    and int(polygons.triangle_polygon.numel()) == faces.shape[0]):
                sizes, corners = polygon_sizes, polygon_corners
        skinned = part.joints is not None and part.weights is not None and joint_ids
        columns = [_bits(vertices)]
        joints = weights = None
        if skinned:
            joints = part.joints.long().numpy()
            weights = part.weights.double().numpy()
            joints = np.clip(joints, 0, len(joint_ids) - 1)
            columns += [joints.astype(np.int32), _bits(weights)]
        morph_items = list(part.morphs.items())
        for _, morph in morph_items:
            columns.append(_bits(morph.positions.double().numpy()))
        first, point_of = _unique_rows(np.concatenate(columns, axis=1))
        points = vertices[first] * 100.0
        polygon_index = point_of[corners].astype(np.int64)
        ends = np.cumsum(sizes) - 1
        polygon_index[ends] = -polygon_index[ends] - 1
        layers = []
        layer_refs = []
        normals = part.normals.double().numpy() if part.normals is not None else _vertex_normals(vertices, faces)
        layers.append(_n("LayerElementNormal", 0, children=[
            _n("Version", 102), _n("Name", ""),
            _n("MappingInformationType", "ByPolygonVertex"),
            _n("ReferenceInformationType", "Direct"),
            _n("Normals", np.ascontiguousarray(normals[corners], dtype=np.float64).reshape(-1)),
        ]))
        layer_refs.append("LayerElementNormal")
        if part.uvs is not None:
            uvs = part.uvs.double().numpy().copy()
            uvs[:, 1] = 1.0 - uvs[:, 1]
            layers.append(_n("LayerElementUV", 0, children=[
                _n("Version", 101), _n("Name", "UVMap"),
                _n("MappingInformationType", "ByPolygonVertex"),
                _n("ReferenceInformationType", "IndexToDirect"),
                _n("UV", np.ascontiguousarray(uvs, dtype=np.float64).reshape(-1)),
                _n("UVIndex", corners.astype(np.int32)),
            ]))
            layer_refs.append("LayerElementUV")
        if part.colors is not None:
            colors = part.colors.double().numpy().copy()
            if colors.shape[1] == 3:
                colors = np.concatenate([colors, np.ones((colors.shape[0], 1))], axis=1)
            colors[:, :3] = _linear_to_srgb(colors[:, :3])
            layers.append(_n("LayerElementColor", 0, children=[
                _n("Version", 101), _n("Name", "Color"),
                _n("MappingInformationType", "ByPolygonVertex"),
                _n("ReferenceInformationType", "IndexToDirect"),
                _n("Colors", np.ascontiguousarray(colors[:, :4], dtype=np.float64).reshape(-1)),
                _n("ColorIndex", corners.astype(np.int32)),
            ]))
            layer_refs.append("LayerElementColor")
        layers.append(_n("LayerElementMaterial", 0, children=[
            _n("Version", 101), _n("Name", ""),
            _n("MappingInformationType", "AllSame"),
            _n("ReferenceInformationType", "IndexToDirect"),
            _n("Materials", np.zeros(1, dtype=np.int32)),
        ]))
        layer_refs.append("LayerElementMaterial")
        layer = _n("Layer", 0, children=[_n("Version", 100)] + [
            _n("LayerElement", children=[_n("Type", ref), _n("TypedIndex", 0)]) for ref in layer_refs
        ])
        name = part.name or f"part_{index}"
        model_id, geometry_id = self._id(), self._id()
        mesh_ids.append(model_id)
        self._add("Geometry", _n("Geometry", geometry_id, _class_name(name, "Geometry"), "Mesh", children=[
            _props70(),
            _n("GeometryVersion", 124),
            _n("Vertices", np.ascontiguousarray(points, dtype=np.float64).reshape(-1)),
            _n("PolygonVertexIndex", polygon_index.astype(np.int32)),
            *layers,
            layer,
        ]))
        self._add("Model", self._model_node(model_id, name, "Mesh", (0, 0, 0), (0, 0, 0), (1, 1, 1)))
        self._connect(model_id, 0)
        self._connect(geometry_id, model_id)
        self._connect(material_id, model_id)
        if skinned:
            skin_id = self._id()
            self._add("Deformer", _n("Deformer", skin_id, _class_name(name, "Deformer"), "Skin", children=[
                _n("Version", 101), _n("Link_DeformAcuracy", 50.0),
            ]))
            self._connect(skin_id, geometry_id)
            point_joints, point_weights = joints[first], weights[first]
            for joint in np.unique(point_joints[point_weights > 0]):
                rows, slots = np.nonzero((point_joints == joint) & (point_weights > 0))
                link = rest[int(joint)]
                cluster_id = self._id()
                self._add("Deformer", _n("Deformer", cluster_id,
                                         _class_name(f"{name}_{int(joint)}", "SubDeformer"), "Cluster", children=[
                    _n("Version", 100),
                    _n("UserData", "", ""),
                    _n("Indexes", rows.astype(np.int32)),
                    _n("Weights", point_weights[rows, slots].astype(np.float64)),
                    _n("Transform", _matrix_values(_inverse(link))),
                    _n("TransformLink", _matrix_values(link)),
                ]))
                self._connect(cluster_id, skin_id)
                self._connect(joint_ids[int(joint)], cluster_id)
        if morph_items:
            blend_id = self._id()
            self._add("Deformer", _n("Deformer", blend_id, _class_name(name, "Deformer"), "BlendShape", children=[
                _n("Version", 100),
            ]))
            self._connect(blend_id, geometry_id)
            for morph_name, morph in morph_items:
                offsets = morph.positions.double().numpy()[first] * 100.0
                moved = np.nonzero(np.abs(offsets).max(axis=1) > 0)[0]
                if moved.size == 0:
                    moved = np.zeros(1, dtype=np.int64)
                channel_id, shape_id = self._id(), self._id()
                self._add("Deformer", _n("Deformer", channel_id, _class_name(morph_name, "SubDeformer"),
                                         "BlendShapeChannel", children=[
                    _n("Version", 100),
                    _n("DeformPercent", 0.0),
                    _n("FullWeights", np.full(1, 100.0, dtype=np.float64)),
                ]))
                self._add("Geometry", _n("Geometry", shape_id, _class_name(morph_name, "Geometry"), "Shape", children=[
                    _n("Version", 100),
                    _n("Indexes", moved.astype(np.int32)),
                    _n("Vertices", np.ascontiguousarray(offsets[moved], dtype=np.float64).reshape(-1)),
                ]))
                self._connect(channel_id, blend_id)
                self._connect(shape_id, channel_id)

    def _bind_pose(self, mesh_ids: list[int], joint_ids: list[int], rest: np.ndarray) -> None:
        """Write the bind pose of every skinned mesh and joint."""
        pose_id = self._id()
        entries = [_n("PoseNode", children=[_n("Node", _Int64(ident)), _n("Matrix", _matrix_values(np.eye(4)))])
                   for ident in mesh_ids]
        entries += [_n("PoseNode", children=[_n("Node", _Int64(ident)), _n("Matrix", _matrix_values(rest[i]))])
                    for i, ident in enumerate(joint_ids)]
        self._add("Pose", _n("Pose", pose_id, _class_name("BindPose", "Pose"), "BindPose", children=[
            _n("Type", "BindPose"), _n("Version", 100), _n("NbPoseNodes", len(entries)), *entries,
        ]))

    def _definitions(self) -> _Node:
        """The Definitions section with counts and property templates."""
        templates = {
            "Model": ("FbxNode", [
                _p("RotationOrder", "enum", "", "", 0),
                _p("InheritType", "enum", "", "", 0),
                _p("DefaultAttributeIndex", "int", "Integer", "", -1),
                _p("Lcl Translation", "Lcl Translation", "", "A", 0.0, 0.0, 0.0),
                _p("Lcl Rotation", "Lcl Rotation", "", "A", 0.0, 0.0, 0.0),
                _p("Lcl Scaling", "Lcl Scaling", "", "A", 1.0, 1.0, 1.0),
                _p("Visibility", "Visibility", "", "A", 1.0),
            ]),
            "Geometry": ("FbxMesh", [
                _p("Color", "ColorRGB", "Color", "", 0.8, 0.8, 0.8),
                _p("Primary Visibility", "bool", "", "", 1),
                _p("Casts Shadows", "bool", "", "", 1),
                _p("Receive Shadows", "bool", "", "", 1),
            ]),
            "Material": ("FbxSurfacePhong", [
                _p("ShadingModel", "KString", "", "", "Phong"),
                _p("EmissiveColor", "Color", "", "A", 0.0, 0.0, 0.0),
                _p("EmissiveFactor", "Number", "", "A", 1.0),
                _p("AmbientColor", "Color", "", "A", 0.2, 0.2, 0.2),
                _p("DiffuseColor", "Color", "", "A", 0.8, 0.8, 0.8),
                _p("DiffuseFactor", "Number", "", "A", 1.0),
                _p("TransparencyFactor", "Number", "", "A", 0.0),
                _p("SpecularColor", "Color", "", "A", 0.2, 0.2, 0.2),
                _p("ShininessExponent", "Number", "", "A", 20.0),
                _p("ReflectionFactor", "Number", "", "A", 1.0),
            ]),
            "Texture": ("FbxFileTexture", [
                _p("UseMaterial", "bool", "", "", 0),
                _p("WrapModeU", "enum", "", "", 0),
                _p("WrapModeV", "enum", "", "", 0),
            ]),
            "Video": ("FbxVideo", [
                _p("Path", "KString", "XRefUrl", "", ""),
            ]),
            "NodeAttribute": ("FbxSkeleton", [
                _p("Color", "ColorRGB", "Color", "", 0.8, 0.8, 0.8),
                _p("Size", "double", "Number", "", 100.0),
                _p("LimbLength", "double", "Number", "H", 1.0),
            ]),
        }
        types = [_n("Version", 100), _n("Count", 1 + sum(self.counts.values())),
                 _n("ObjectType", "GlobalSettings", children=[_n("Count", 1)])]
        for kind in ("Model", "Geometry", "Material", "Texture", "Video", "NodeAttribute", "Deformer", "Pose"):
            if not self.counts.get(kind):
                continue
            children = [_n("Count", self.counts[kind])]
            if kind in templates:
                template, entries = templates[kind]
                children.append(_n("PropertyTemplate", template, children=[_props70(*entries)]))
            types.append(_n("ObjectType", kind, children=children))
        return _n("Definitions", children=types)

    def build(self) -> list[_Node]:
        """Every top-level record of the file."""
        plan = _TexturePlan(self.model)
        plan.encode()
        material_ids = self._materials(plan)
        joint_ids, rest = self._skeleton()
        mesh_ids: list[int] = []
        skinned_ids: list[int] = []
        for index, part in enumerate(self.model.parts):
            before = len(mesh_ids)
            self._part(part, index, material_ids[id(part.material)], joint_ids, rest, mesh_ids)
            if len(mesh_ids) > before and part.weights is not None and joint_ids:
                skinned_ids.append(mesh_ids[-1])
        if skinned_ids:
            self._bind_pose(skinned_ids, joint_ids, rest)
        meta = self.model.meta or {}
        title = str(meta.get("name") or meta.get("title") or "")
        authors = meta.get("authors") or []
        author = ", ".join(str(a) for a in authors) if isinstance(authors, (list, tuple)) else str(authors)
        header = _n("FBXHeaderExtension", children=[
            _n("FBXHeaderVersion", 1003),
            _n("FBXVersion", WRITE_VERSION),
            _n("EncryptionType", 0),
            _n("CreationTimeStamp", children=[
                _n("Version", 1000), _n("Year", 1970), _n("Month", 1), _n("Day", 1), _n("Hour", 10),
                _n("Minute", 0), _n("Second", 0), _n("Millisecond", 0),
            ]),
            _n("Creator", "WAS Node Suite"),
            _n("SceneInfo", _class_name("GlobalInfo", "SceneInfo"), "UserData", children=[
                _n("Type", "UserData"),
                _n("Version", 100),
                _n("MetaData", children=[
                    _n("Version", 100), _n("Title", title), _n("Subject", ""), _n("Author", author),
                    _n("Keywords", ""), _n("Revision", ""), _n("Comment", ""),
                ]),
                _props70(
                    _p("Original|ApplicationVendor", "KString", "", "", "WAS"),
                    _p("Original|ApplicationName", "KString", "", "", "WAS Node Suite"),
                    _p("LastSaved|ApplicationVendor", "KString", "", "", "WAS"),
                    _p("LastSaved|ApplicationName", "KString", "", "", "WAS Node Suite"),
                ),
            ]),
        ])
        settings = _n("GlobalSettings", children=[
            _n("Version", 1000),
            _props70(
                _p("UpAxis", "int", "Integer", "", 1),
                _p("UpAxisSign", "int", "Integer", "", 1),
                _p("FrontAxis", "int", "Integer", "", 2),
                _p("FrontAxisSign", "int", "Integer", "", 1),
                _p("CoordAxis", "int", "Integer", "", 0),
                _p("CoordAxisSign", "int", "Integer", "", 1),
                _p("OriginalUpAxis", "int", "Integer", "", 1),
                _p("OriginalUpAxisSign", "int", "Integer", "", 1),
                _p("UnitScaleFactor", "double", "Number", "", 1.0),
                _p("OriginalUnitScaleFactor", "double", "Number", "", 1.0),
                _p("AmbientColor", "ColorRGB", "Color", "", 0.0, 0.0, 0.0),
                _p("DefaultCamera", "KString", "", "", "Producer Perspective"),
                _p("TimeMode", "enum", "", "", 6),
                _p("TimeSpanStart", "KTime", "Time", "", _Int64(0)),
                _p("TimeSpanStop", "KTime", "Time", "", _Int64(TICKS_PER_SECOND)),
                _p("CustomFrameRate", "double", "Number", "", -1.0),
            ),
        ])
        document_id = self._id()
        documents = _n("Documents", children=[
            _n("Count", 1),
            _n("Document", document_id, "", "Scene", children=[
                _props70(
                    _p("SourceObject", "object", "", ""),
                    _p("ActiveAnimStackName", "KString", "", "", ""),
                ),
                _n("RootNode", _Int64(0)),
            ]),
        ])
        return [
            header,
            _n("FileId", _FILE_ID),
            _n("CreationTime", _CREATION_TIME),
            _n("Creator", "WAS Node Suite"),
            settings,
            documents,
            _n("References"),
            self._definitions(),
            _n("Objects", children=self.objects),
            _n("Connections", children=self.connections),
            _n("Takes", children=[_n("Current", "")]),
        ]


def write(model: Model3D, embed_textures: bool = True) -> bytes:
    """A binary FBX 7.4 file of a model's meshes, materials, skeleton, skin and shape keys.

    Positions are centimetres, Y up, facing +Z.

    Args:
        model: The model to write.
        embed_textures: Embed texture images in the file; when False the file names them and
            `texture_files` returns the files to save beside it.

    Returns:
        The file's bytes.

    Raises:
        FbxError: A part names vertices it does not hold, holds an attribute or shape key with
            the wrong row count, or the file would pass 4 GB.
    """
    out = bytearray(_HEADER + struct.pack("<I", WRITE_VERSION))
    for node in _Writer(model, embed_textures).build():
        _encode_node(node, out)
    out += b"\x00" * 13
    out += _FOOTER_ID
    out += b"\x00" * 4
    pad = (16 - len(out) % 16) % 16 or 16
    out += b"\x00" * pad
    out += struct.pack("<I", WRITE_VERSION)
    out += b"\x00" * 120
    out += _FOOTER_MAGIC
    return bytes(out)

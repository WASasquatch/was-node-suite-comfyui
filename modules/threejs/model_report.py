"""What a glTF model holds, read from its JSON and its pictures' headers without drawing it.

Sizes are in the model's own units. Texture sizes come from each embedded picture's header.
"""

from __future__ import annotations

import json
import math
import struct

__all__ = ["DECODERS", "MAX_CLIPS", "UNREAD_EXTENSIONS", "describe"]

#: glTF extension -> the geometry decoder the browser loads to read it.
DECODERS = {
    "EXT_meshopt_compression": "meshopt",
    "KHR_meshopt_compression": "meshopt",
    "KHR_draco_mesh_compression": "draco",
}

#: Required extensions the browser has no reader for, against what they hold.
UNREAD_EXTENSIONS = {
    "KHR_texture_basisu": "KTX2 textures",
}

#: Most animation clips named in a description.
MAX_CLIPS = 24

#: Longest generator string kept, in characters.
MAX_GENERATOR = 80

_GLB_MAGIC = b"glTF"
_JSON_CHUNK = 0x4E4F534A
_BIN_CHUNK = 0x004E4942

#: Component type -> the largest value a normalized accessor of that type stores.
_NORMALIZED_MAX = {5120: 127.0, 5121: 255.0, 5122: 32767.0, 5123: 65535.0}

#: Primitive modes that draw triangles: triangles, strip and fan.
_TRIANGLE_MODES = (4, 5, 6)

#: Deepest node hierarchy walked.
_MAX_DEPTH = 256

_IDENTITY = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def _split_glb(body: bytes) -> tuple[dict, memoryview | None]:
    """The JSON document and the binary chunk of a ``.glb``.

    Args:
        body: The file's bytes.

    Returns:
        ``(document, binary)``, the binary chunk as a view or None where the file has none.

    Raises:
        ValueError: The bytes are not a binary glTF, or carry no JSON chunk.
    """
    if len(body) < 20 or body[:4] != _GLB_MAGIC:
        raise ValueError("not a binary glTF")
    declared = struct.unpack_from("<I", body, 8)[0]
    end_of_file = min(declared, len(body))
    offset = 12
    document = None
    binary = None
    while offset + 8 <= end_of_file:
        length, kind = struct.unpack_from("<II", body, offset)
        start = offset + 8
        end = start + length
        if end > end_of_file:
            break
        if kind == _JSON_CHUNK and document is None:
            document = json.loads(bytes(body[start:end]).decode("utf-8"))
        elif kind == _BIN_CHUNK and binary is None:
            binary = memoryview(body)[start:end]
        offset = end
    if not isinstance(document, dict):
        raise ValueError("a binary glTF with no JSON chunk")
    return document, binary


def _picture_size(head: bytes) -> tuple[int, int] | None:
    """A PNG, JPEG or WebP picture's width and height, read from its header.

    Args:
        head: The picture's first bytes.

    Returns:
        ``(width, height)``, or None for another format or a header cut short.
    """
    if head[:8] == b"\x89PNG\r\n\x1a\n" and len(head) >= 24:
        return struct.unpack(">II", head[16:24])
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP" and len(head) >= 30:
        chunk = head[12:16]
        if chunk == b"VP8X":
            width = int.from_bytes(head[24:27], "little") + 1
            height = int.from_bytes(head[27:30], "little") + 1
            return width, height
        if chunk == b"VP8L" and len(head) >= 25:
            bits = int.from_bytes(head[21:25], "little")
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
        if chunk == b"VP8 ":
            width, height = struct.unpack("<HH", head[26:30])
            return width & 0x3FFF, height & 0x3FFF
        return None
    if head[:2] == b"\xff\xd8":
        index = 2
        while index + 9 <= len(head):
            if head[index] != 0xFF:
                index += 1
                continue
            marker = head[index + 1]
            if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7 or marker == 0xFF:
                index += 1 if marker == 0xFF else 2
                continue
            length = struct.unpack(">H", head[index + 2:index + 4])[0]
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                height, width = struct.unpack(">HH", head[index + 5:index + 9])
                return width, height
            index += 2 + length
    return None


def _image_sizes(document: dict, binary: memoryview | None) -> list[tuple[int, int]]:
    """The sizes of the pictures embedded in the binary chunk, where their headers read.

    Args:
        document: The glTF JSON.
        binary: The binary chunk, or None.

    Returns:
        ``(width, height)`` for each picture whose header was found, in image order.
    """
    if binary is None:
        return []
    views = document.get("bufferViews") or []
    sizes = []
    for image in document.get("images") or []:
        if not isinstance(image, dict):
            continue
        index = image.get("bufferView")
        if not isinstance(index, int) or not 0 <= index < len(views):
            continue
        view = views[index]
        if not isinstance(view, dict) or view.get("buffer", 0) != 0:
            continue
        start = int(view.get("byteOffset", 0) or 0)
        length = int(view.get("byteLength", 0) or 0)
        if start < 0 or length <= 0 or start + length > len(binary):
            continue
        # JPEG frame headers can sit behind embedded metadata, so a generous head is read.
        size = _picture_size(bytes(binary[start:start + min(length, 262144)]))
        if size:
            sizes.append(size)
    return sizes


def _multiply(left: tuple, right: tuple) -> tuple:
    """Two column-major 4x4 matrices multiplied, ``left @ right``."""
    return tuple(
        sum(left[row + 4 * k] * right[k + 4 * column] for k in range(4))
        for column in range(4)
        for row in range(4)
    )


def _local_matrix(node: dict) -> tuple:
    """A glTF node's own transform as a column-major 4x4 matrix."""
    matrix = node.get("matrix")
    if isinstance(matrix, list) and len(matrix) == 16:
        return tuple(float(value) for value in matrix)
    tx, ty, tz = (node.get("translation") or [0.0, 0.0, 0.0])[:3]
    x, y, z, w = (node.get("rotation") or [0.0, 0.0, 0.0, 1.0])[:4]
    sx, sy, sz = (node.get("scale") or [1.0, 1.0, 1.0])[:3]
    return (
        (1 - 2 * (y * y + z * z)) * sx, (2 * (x * y + z * w)) * sx, (2 * (x * z - y * w)) * sx, 0.0,
        (2 * (x * y - z * w)) * sy, (1 - 2 * (x * x + z * z)) * sy, (2 * (y * z + x * w)) * sy, 0.0,
        (2 * (x * z + y * w)) * sz, (2 * (y * z - x * w)) * sz, (1 - 2 * (x * x + y * y)) * sz, 0.0,
        float(tx), float(ty), float(tz), 1.0,
    )


def _triangles(primitive: dict, accessors: list) -> int:
    """How many triangles one mesh primitive draws."""
    mode = primitive.get("mode", 4)
    if mode not in _TRIANGLE_MODES:
        return 0
    index = primitive.get("indices")
    if not isinstance(index, int):
        index = (primitive.get("attributes") or {}).get("POSITION")
    if not isinstance(index, int) or not 0 <= index < len(accessors):
        return 0
    count = int((accessors[index] or {}).get("count", 0) or 0)
    return count // 3 if mode == 4 else max(count - 2, 0)


def _walk(document: dict) -> tuple[int, int, list | None]:
    """Triangles and vertices drawn by the default scene, and its bounds.

    Args:
        document: The glTF JSON.

    Returns:
        ``(triangles, vertices, bounds)``, bounds as ``[min xyz, max xyz]`` in the model's
        units or None where no unskinned primitive declares its extent.
    """
    nodes = document.get("nodes") or []
    meshes = document.get("meshes") or []
    accessors = document.get("accessors") or []
    scenes = document.get("scenes") or []
    chosen = document.get("scene", 0)
    if isinstance(chosen, int) and 0 <= chosen < len(scenes) and isinstance(scenes[chosen], dict):
        roots = scenes[chosen].get("nodes") or []
    else:
        children = {child for node in nodes if isinstance(node, dict) for child in node.get("children") or []}
        roots = [index for index in range(len(nodes)) if index not in children]

    low = [math.inf] * 3
    high = [-math.inf] * 3
    totals = [0, 0]
    stack = [(index, _IDENTITY, 0) for index in roots]
    visited = 0
    while stack:
        index, parent, depth = stack.pop()
        if not isinstance(index, int) or not 0 <= index < len(nodes) or depth > _MAX_DEPTH:
            continue
        visited += 1
        if visited > 4 * len(nodes) + 16:
            break
        node = nodes[index] if isinstance(nodes[index], dict) else {}
        world = _multiply(parent, _local_matrix(node))
        mesh = node.get("mesh")
        if isinstance(mesh, int) and 0 <= mesh < len(meshes) and isinstance(meshes[mesh], dict):
            for primitive in meshes[mesh].get("primitives") or []:
                if not isinstance(primitive, dict):
                    continue
                totals[0] += _triangles(primitive, accessors)
                position = (primitive.get("attributes") or {}).get("POSITION")
                if not isinstance(position, int) or not 0 <= position < len(accessors):
                    continue
                accessor = accessors[position] or {}
                totals[1] += int(accessor.get("count", 0) or 0)
                # A skinned mesh's positions sit in its bind space, which only its joints place.
                if node.get("skin") is not None:
                    continue
                lo, hi = accessor.get("min"), accessor.get("max")
                if not (isinstance(lo, list) and isinstance(hi, list) and len(lo) >= 3 and len(hi) >= 3):
                    continue
                # A quantized position stores its bounds as the integers it holds.
                divisor = _NORMALIZED_MAX.get(accessor.get("componentType")) if accessor.get("normalized") else None
                if divisor:
                    lo = [max(value / divisor, -1.0) for value in lo]
                    hi = [max(value / divisor, -1.0) for value in hi]
                for corner in range(8):
                    point = [hi[axis] if corner >> axis & 1 else lo[axis] for axis in range(3)]
                    for axis in range(3):
                        value = (
                            world[axis] * point[0] + world[axis + 4] * point[1]
                            + world[axis + 8] * point[2] + world[axis + 12]
                        )
                        low[axis] = min(low[axis], value)
                        high[axis] = max(high[axis], value)
        for child in node.get("children") or []:
            stack.append((child, world, depth + 1))
    bounds = [low, high] if all(math.isfinite(value) for value in low + high) else None
    return totals[0], totals[1], bounds


def _clips(document: dict) -> list[tuple[str, float]]:
    """Each animation clip's name and length in seconds."""
    accessors = document.get("accessors") or []
    clips = []
    for number, animation in enumerate(document.get("animations") or []):
        if not isinstance(animation, dict):
            continue
        seconds = 0.0
        for sampler in animation.get("samplers") or []:
            index = sampler.get("input") if isinstance(sampler, dict) else None
            if isinstance(index, int) and 0 <= index < len(accessors):
                top = (accessors[index] or {}).get("max")
                if isinstance(top, list) and top and isinstance(top[0], (int, float)):
                    seconds = max(seconds, float(top[0]))
        name = str(animation.get("name") or f"animation {number}")
        clips.append((name, seconds))
    return clips


def describe(body: bytes, suffix: str) -> dict:
    """What a model file holds.

    Args:
        body: The file's bytes.
        suffix: Its format, with or without the dot.

    Returns:
        ``format`` and ``bytes`` for every model. A glTF adds ``meshes``, ``triangles``,
        ``vertices``, ``materials``, ``textures``, ``texture_sizes``, ``clips``, ``skins``,
        ``joints``, ``extensions``, ``required``, ``decoders``, ``unread``, ``bounds`` and
        ``generator``.

    Raises:
        ValueError: A glTF whose JSON cannot be read.
    """
    suffix = str(suffix or "").lower().lstrip(".")
    found: dict = {"format": suffix, "bytes": len(body)}
    if suffix not in ("glb", "gltf"):
        return found
    if suffix == "glb":
        document, binary = _split_glb(body)
    else:
        document, binary = json.loads(body.decode("utf-8")), None
        if not isinstance(document, dict):
            raise ValueError("a glTF whose JSON is not an object")

    used = sorted(str(name) for name in document.get("extensionsUsed") or [])
    required = sorted(str(name) for name in document.get("extensionsRequired") or [])
    triangles, vertices, bounds = _walk(document)
    skins = [skin for skin in document.get("skins") or [] if isinstance(skin, dict)]
    generator = str((document.get("asset") or {}).get("generator") or "")[:MAX_GENERATOR]
    found.update(
        meshes=len(document.get("meshes") or []),
        triangles=triangles,
        vertices=vertices,
        materials=len(document.get("materials") or []),
        textures=len(document.get("images") or []),
        texture_sizes=_image_sizes(document, binary),
        clips=_clips(document)[:MAX_CLIPS],
        clip_count=len(document.get("animations") or []),
        skins=len(skins),
        joints=sum(len(skin.get("joints") or []) for skin in skins),
        extensions=used,
        required=required,
        decoders=sorted({DECODERS[name] for name in used if name in DECODERS}),
        unread=[name for name in required if name in UNREAD_EXTENSIONS],
        bounds=bounds,
        generator=generator,
    )
    return found

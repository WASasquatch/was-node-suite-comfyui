"""Wavefront OBJ files and their MTL libraries, read into and written from `scene.Model3D`.

Quads and n-gons are kept in `scene.Polygons`. Texture ``v`` is flipped to glTF's convention.
Vertex colours are sRGB in the file, linear in the model.
"""

from __future__ import annotations

__all__ = [
    "MAX_CORNERS", "MAX_LIBRARIES", "MAX_LINES", "MAX_MATERIALS", "MAX_MTL_BYTES", "MAX_PARTS",
    "MAX_TEXTURE_PIXELS", "MAX_VERTICES", "read", "write",
]

import io
import math
import re
from array import array
from pathlib import PurePosixPath

import numpy as np
import torch

from ..log import get_logger
from . import textures
from .scene import Material, Model3D, Part, Polygons, Texture

logger = get_logger("model3d.obj")

#: Most lines one OBJ file may hold.
MAX_LINES = 80_000_000

#: Most positions, texture coordinates or normals one OBJ file may declare, each.
MAX_VERTICES = 40_000_000

#: Most face corners one OBJ file may declare in total.
MAX_CORNERS = 60_000_000

#: Largest MTL library read.
MAX_MTL_BYTES = 16 * 1024 * 1024

#: Most materials one MTL library may declare.
MAX_MATERIALS = 10_000

#: Most distinct material libraries one OBJ file may name.
MAX_LIBRARIES = 64

#: Most parts (object, group and material runs) one OBJ file may hold.
MAX_PARTS = 100_000

#: Most texture pixels decoded for one OBJ file, all maps together.
MAX_TEXTURE_PIXELS = 268_435_456

#: Longest object, group or material name kept, in characters.
MAX_NAME_CHARS = 256

#: Polygons with more corners than this are fanned instead of ear-clipped.
EAR_CLIP_CORNERS = 256

#: Ear-clipping steps spent on one file before the remaining concave polygons are fanned.
EAR_CLIP_STEPS = 1_000_000

#: Corners tested for convexity in one batch; a larger polygon is fanned untested.
_CONVEX_TEST_CORNERS = 1_000_000

#: Arguments each texture option takes, as ``(fewest, most)``.
_TEXTURE_OPTIONS = {
    "-blendu": (1, 1), "-blendv": (1, 1), "-boost": (1, 1), "-cc": (1, 1), "-clamp": (1, 1),
    "-imfchan": (1, 1), "-texres": (1, 1), "-type": (1, 1), "-bm": (1, 1), "-mm": (1, 2),
    "-o": (1, 3), "-s": (1, 3), "-t": (1, 3),
}

#: Characters replaced in a written object or material name.
_UNSAFE_NAME = re.compile(r"[^\w.-]+")

#: Characters replaced in a written texture file name.
_UNSAFE_FILE = re.compile(r"[^A-Za-z0-9_.-]+")

#: Longest name written into a file or a texture file name, in characters.
_WRITTEN_NAME_CHARS = 64

#: Roughness of a material that states none.
DEFAULT_ROUGHNESS = 0.5


def _srgb_to_linear(values: np.ndarray) -> np.ndarray:
    """sRGB codes in ``[0, 1]`` as linear values."""
    values = np.clip(values, 0.0, 1.0)
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(values: np.ndarray) -> np.ndarray:
    """Linear values in ``[0, 1]`` as sRGB codes."""
    values = np.clip(values, 0.0, 1.0)
    return np.where(values <= 0.0031308, values * 12.92, 1.055 * np.power(values, 1.0 / 2.4) - 0.055)


def _decoded(raw: bytes) -> str:
    """A name from the file as text, cut to `MAX_NAME_CHARS`."""
    return raw[:MAX_NAME_CHARS * 4].decode("utf-8", errors="replace").strip()[:MAX_NAME_CHARS]


def _after_key(line: bytes, key: bytes) -> str:
    """The name following a statement keyword, leading spacing ignored."""
    return _decoded(line.lstrip()[len(key):])


def _remainder(line: str, index: int) -> str:
    """The text of ``line`` from its ``index``-th word to the end, spacing kept."""
    for number, match in enumerate(re.finditer(r"\S+", line)):
        if number == index:
            return line[match.start():].strip()
    return ""


class _Damaged(ValueError):
    """A statement of the OBJ file that cannot be read."""


def _index(token: bytes, count: int, kind: str, line_number: int) -> int:
    """A one-based or negative OBJ index as a zero-based one.

    Args:
        token: The index as written.
        count: How many elements of its kind are declared so far.
        kind: ``"vertex"``, ``"texture coordinate"`` or ``"normal"``, for the error.
        line_number: Line the index sits on, for the error.

    Returns:
        The zero-based index. A positive index is range-checked once the whole file is read.

    Raises:
        ValueError: The index is zero, not a number, or reaches before the first element.
    """
    try:
        value = int(token)
    except ValueError:
        raise _Damaged(f"line {line_number} of the OBJ file has a face {kind} index "
                       f"{_decoded(token)!r} that is not a number; the file is damaged") from None
    if 0 < value <= MAX_VERTICES:
        return value - 1
    if value < 0 and count + value >= 0:
        return count + value
    raise _Damaged(f"line {line_number} of the OBJ file has a face {kind} index {value} that "
                   f"names no {kind}; the file is damaged or was cut short")


def _normalised(data: bytes) -> bytes:
    """Text with its byte order mark dropped, lines ending in LF and continuations joined."""
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    if b"\n" not in data and b"\r" in data:
        data = data.replace(b"\r", b"\n")
    if b"\\\n" in data or b"\\\r\n" in data:
        data = data.replace(b"\\\r\n", b" ").replace(b"\\\n", b" ")
    return data


def _lines(data: bytes):
    """The lines of normalised text, one at a time."""
    return iter(io.BytesIO(data))


class _Parsed:
    """Raw arrays gathered from one OBJ file."""

    def __init__(self) -> None:
        self.positions = array("f")
        self.colors = array("f")
        self.colored = bytearray()
        self.has_colors = False
        self.uvs = array("f")
        self.normals = array("f")
        self.corner_v = array("i")
        self.corner_t = array("i")
        self.corner_n = array("i")
        self.face_size = array("i")
        self.face_group = array("i")
        self.groups: dict[tuple[str, str, str], int] = {}
        self.libraries: list[str] = []
        self.skipped_faces = 0


def _parse(data: bytes) -> _Parsed:
    """Every statement of an OBJ file gathered into flat arrays.

    Args:
        data: The OBJ file.

    Returns:
        The gathered arrays.

    Raises:
        ValueError: The file is damaged or larger than the OBJ limits.
    """
    data = _normalised(data)
    line_count = data.count(b"\n") + 1
    if line_count > MAX_LINES:
        raise ValueError(f"the OBJ file has {line_count} lines, more than the {MAX_LINES} one model "
                         "may hold; decimate it in a 3D editor and save it again")
    parsed = _Parsed()
    object_name = group_name = material_name = ""
    group_id = -1
    vertex_count = uv_count = normal_count = corner_total = 0
    for line_number, line in enumerate(_lines(data), start=1):
        words = line.split()
        if not words or words[0].startswith(b"#"):
            continue
        key = words[0]
        try:
            if key == b"v":
                if len(words) < 4:
                    raise ValueError
                parsed.positions.extend((float(words[1]), float(words[2]), float(words[3])))
                if len(words) >= 7:
                    if not parsed.has_colors:
                        parsed.has_colors = True
                        parsed.colors.extend([1.0] * (3 * vertex_count))
                        parsed.colored.extend(bytes(vertex_count))
                    parsed.colors.extend((float(words[4]), float(words[5]), float(words[6])))
                    parsed.colored.append(1)
                elif parsed.has_colors:
                    parsed.colors.extend((1.0, 1.0, 1.0))
                    parsed.colored.append(0)
                vertex_count += 1
                if vertex_count > MAX_VERTICES:
                    raise OverflowError("positions")
            elif key == b"vt":
                if len(words) < 2:
                    raise ValueError
                parsed.uvs.extend((float(words[1]), float(words[2]) if len(words) > 2 else 0.0))
                uv_count += 1
                if uv_count > MAX_VERTICES:
                    raise OverflowError("texture coordinates")
            elif key == b"vn":
                if len(words) < 4:
                    raise ValueError
                parsed.normals.extend((float(words[1]), float(words[2]), float(words[3])))
                normal_count += 1
                if normal_count > MAX_VERTICES:
                    raise OverflowError("normals")
            elif key == b"f":
                corners = words[1:]
                if len(corners) < 3:
                    parsed.skipped_faces += 1
                    continue
                corner_total += len(corners)
                if corner_total > MAX_CORNERS:
                    raise OverflowError("face corners")
                if group_id < 0:
                    group_key = (object_name, group_name, material_name)
                    group_id = parsed.groups.setdefault(group_key, len(parsed.groups))
                    if len(parsed.groups) > MAX_PARTS:
                        raise OverflowError("parts")
                for corner in corners:
                    fields = corner.split(b"/")
                    parsed.corner_v.append(_index(fields[0], vertex_count, "vertex", line_number))
                    parsed.corner_t.append(
                        _index(fields[1], uv_count, "texture coordinate", line_number)
                        if len(fields) > 1 and fields[1] else -1)
                    parsed.corner_n.append(
                        _index(fields[2], normal_count, "normal", line_number)
                        if len(fields) > 2 and fields[2] else -1)
                parsed.face_size.append(len(corners))
                parsed.face_group.append(group_id)
            elif key == b"o":
                object_name, group_name, group_id = _after_key(line, key), "", -1
            elif key == b"g":
                group_name, group_id = _after_key(line, key), -1
            elif key == b"usemtl":
                material_name, group_id = _after_key(line, key), -1
            elif key == b"mtllib":
                library = _after_key(line, key)
                if library and library not in parsed.libraries:
                    if len(parsed.libraries) >= MAX_LIBRARIES:
                        raise OverflowError("material libraries")
                    parsed.libraries.append(library)
        except OverflowError as error:
            limit = {"face corners": MAX_CORNERS, "parts": MAX_PARTS,
                     "material libraries": MAX_LIBRARIES}.get(str(error), MAX_VERTICES)
            raise ValueError(f"the OBJ file declares more than {limit} {error}, more than one "
                             "model may hold; decimate it in a 3D editor and save it again") from None
        except _Damaged:
            raise
        except ValueError:
            raise ValueError(f"line {line_number} of the OBJ file is a '{_decoded(key)}' statement "
                             "with missing or unreadable numbers; the file is damaged") from None
    return parsed


def _texture_statement(line: str, key_index: int = 0) -> tuple[str, dict[str, list[str]]]:
    """The file name and options of an MTL map statement.

    Args:
        line: The whole statement.
        key_index: Word index of the statement keyword.

    Returns:
        ``(file name, options)``, options keyed by flag with their argument words.
    """
    words = line.split()
    index = key_index + 1
    options: dict[str, list[str]] = {}
    while index < len(words) and words[index].lower() in _TEXTURE_OPTIONS:
        flag = words[index].lower()
        fewest, most = _TEXTURE_OPTIONS[flag]
        index += 1
        values: list[str] = []
        while index < len(words) and len(values) < most:
            if len(values) >= fewest:
                try:
                    float(words[index])
                except ValueError:
                    break
            values.append(words[index])
            index += 1
        options[flag] = values
    return _remainder(line, index), options


def _number(words: list[str], index: int, default: float) -> float:
    """Word ``index`` as a finite float, or ``default``."""
    try:
        value = float(words[index])
    except (IndexError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _colour(words: list[str], default: tuple[float, float, float]) -> tuple[float, float, float]:
    """An MTL ``Kd``-style colour; a single value is grey, and an unreadable one is ``default``."""
    first = _number(words, 1, math.nan)
    if math.isnan(first):
        return default
    return first, _number(words, 2, first), _number(words, 3, first)


class _TextureCache:
    """Textures read beside an MTL file, each decoded once, within `MAX_TEXTURE_PIXELS` in total."""

    def __init__(self, base_dir) -> None:
        self.base_dir = base_dir
        self.loaded: dict[str, Texture | None] = {}
        self.pixels = 0

    def get(self, name: str) -> Texture | None:
        """The texture a map statement names, or None when it cannot be read."""
        name = name.replace("\\", "/").strip().strip('"')
        if not name:
            return None
        if name not in self.loaded:
            data = textures.read_sidecar(self.base_dir, name)
            leaf = PurePosixPath(name).name
            if data is None and leaf != name:
                data = textures.read_sidecar(self.base_dir, leaf)
            texture = textures.decode(data, name=leaf) if data is not None else None
            if texture is not None:
                pixels = int(texture.image.shape[0]) * int(texture.image.shape[1])
                if self.pixels + pixels > MAX_TEXTURE_PIXELS:
                    logger.warning("texture %s was skipped: the model's textures already hold %d "
                                   "megapixels, the most one OBJ file may load", leaf,
                                   MAX_TEXTURE_PIXELS // 1_000_000)
                    texture = None
                else:
                    self.pixels += pixels
            self.loaded[name] = texture
        return self.loaded[name]


def _channel(texture: Texture, option: list[str] | None, default: str) -> torch.Tensor:
    """One channel of a texture, chosen by an ``-imfchan`` value.

    Args:
        texture: The texture.
        option: The ``-imfchan`` arguments, or None.
        default: Channel used without the option: ``"r"``, ``"m"`` (alpha) or ``"l"``.

    Returns:
        ``(H, W)`` float32.
    """
    image = texture.image
    chosen = (option[0].lower() if option else default)
    if chosen == "m" and image.shape[2] == 4:
        return image[..., 3]
    if chosen in ("g", "b") and image.shape[2] >= 3:
        return image[..., "rgb".index(chosen)]
    if chosen in ("l", "m", "z"):
        return image[..., :3].mean(dim=2)
    return image[..., 0]


def _resized(plane: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """``(H, W, C)`` or ``(H, W)`` resampled to ``height`` by ``width``."""
    if plane.shape[0] == height and plane.shape[1] == width:
        return plane
    flat = plane.dim() == 2
    image = (plane[..., None] if flat else plane).permute(2, 0, 1)[None]
    out = torch.nn.functional.interpolate(image, size=(height, width), mode="bilinear",
                                          align_corners=False)[0].permute(1, 2, 0)
    return out[..., 0] if flat else out.contiguous()


def _material(name: str, statements: list[str], cache: _TextureCache) -> Material:
    """One MTL material built from its statements.

    Args:
        name: The ``newmtl`` name.
        statements: Every line after ``newmtl`` up to the next one.
        cache: Texture reader for the library's folder.

    Returns:
        The material.
    """
    values: dict[str, list[str]] = {}
    maps: dict[str, tuple[str, dict[str, list[str]]]] = {}
    for line in statements:
        words = line.split()
        key = words[0].lower()
        if key.startswith("map_") or key in ("bump", "norm", "disp", "decal", "refl"):
            maps[key] = _texture_statement(line)
        else:
            values[key] = words
    material = Material(name=name, metallic=0.0, roughness=DEFAULT_ROUGHNESS)
    extras: dict[str, object] = {}
    diffuse = _colour(values["kd"], (0.8, 0.8, 0.8)) if "kd" in values else (0.8, 0.8, 0.8)
    alpha = 1.0
    if "d" in values:
        alpha = _number(values["d"], 1, 1.0)
    elif "tr" in values:
        alpha = 1.0 - _number(values["tr"], 1, 0.0)
    alpha = min(max(alpha, 0.0), 1.0)
    material.base_color = (*diffuse, alpha)
    if "ke" in values:
        material.emissive = tuple(min(max(c, 0.0), 1.0) for c in _colour(values["ke"], (0.0, 0.0, 0.0)))
    if "pr" in values:
        material.roughness = min(max(_number(values["pr"], 1, DEFAULT_ROUGHNESS), 0.0), 1.0)
    elif "ns" in values:
        shininess = min(max(_number(values["ns"], 1, 0.0), 0.0), 1000.0)
        material.roughness = 1.0 - (shininess / 1000.0) ** 0.5
    if "pm" in values:
        material.metallic = min(max(_number(values["pm"], 1, 0.0), 0.0), 1.0)
    for key, label in (("ks", "Ks"), ("ka", "Ka")):
        if key in values:
            extras[label] = list(_colour(values[key], (0.0, 0.0, 0.0)))
    for key, label in (("ns", "Ns"), ("ni", "Ni")):
        if key in values:
            extras[label] = _number(values[key], 1, 0.0)
    if "illum" in values:
        illum = min(max(int(_number(values["illum"], 1, 2)), 0), 10)
        extras["illum"] = illum
        material.unlit = illum == 0
    if alpha < 1.0:
        material.alpha_mode = "BLEND"

    def texture_of(key: str) -> tuple[Texture | None, dict[str, list[str]]]:
        if key not in maps:
            return None, {}
        file_name, options = maps[key]
        texture = cache.get(file_name)
        if texture is not None and (options.get("-clamp") or ["off"])[0].lower() == "on":
            texture = Texture(texture.image, texture.name, "clamp", "clamp", texture.mime)
        if options:
            extras.setdefault("map_options", {})[key] = options
        return texture, options

    base, _ = texture_of("map_kd")
    material.base_color_texture = base
    dissolve, dissolve_options = texture_of("map_d")
    if dissolve is not None:
        default = "m" if dissolve.image.shape[2] == 4 else "l"
        plane = _channel(dissolve, dissolve_options.get("-imfchan"), default)
        if base is None:
            rgb = torch.ones(plane.shape[0], plane.shape[1], 3)
            base_name = dissolve.name
            wrap = (dissolve.wrap_s, dissolve.wrap_t)
        else:
            rgb = base.image[..., :3]
            base_name = base.name
            wrap = (base.wrap_s, base.wrap_t)
            plane = _resized(plane, rgb.shape[0], rgb.shape[1])
        material.base_color_texture = Texture(torch.cat([rgb, plane[..., None]], dim=2).contiguous(),
                                              base_name, wrap[0], wrap[1], "image/png")
        material.alpha_mode = "BLEND"
    normal = None
    for key in ("norm", "map_bump", "bump"):
        normal, options = texture_of(key)
        if normal is not None:
            material.normal_scale = _number(options.get("-bm", []), 0, 1.0)
            if normal.image.shape[2] == 4:
                normal = Texture(normal.image[..., :3].contiguous(), normal.name, normal.wrap_s,
                                 normal.wrap_t, normal.mime)
            break
    material.normal_texture = normal
    emissive, _ = texture_of("map_ke")
    material.emissive_texture = emissive
    if emissive is not None and max(material.emissive) <= 0.0:
        material.emissive = (1.0, 1.0, 1.0)
    _metallic_roughness(material, texture_of)
    material.extras = {"obj": extras} if extras else {}
    return material


def _metallic_roughness(material: Material, texture_of) -> None:
    """Build a material's metallic-roughness and occlusion maps from ORM, RMA, Pr and Pm maps.

    Args:
        material: The material, changed in place.
        texture_of: Returns ``(texture, options)`` for a map keyword.
    """
    orm, _ = texture_of("map_orm")
    if orm is not None:
        material.metallic_roughness_texture = orm
        material.occlusion_texture = orm
        material.metallic = material.roughness = 1.0
        return
    rma, _ = texture_of("map_rma")
    if rma is not None:
        image = rma.image[..., :3]
        packed = torch.stack([image[..., 2], image[..., 0], image[..., 1]], dim=2).contiguous()
        texture = Texture(packed, rma.name, rma.wrap_s, rma.wrap_t, "image/png")
        material.metallic_roughness_texture = texture
        material.occlusion_texture = texture
        material.metallic = material.roughness = 1.0
        return
    rough, rough_options = texture_of("map_pr")
    metal, metal_options = texture_of("map_pm")
    if rough is None and metal is None:
        return
    sizes = [t.image.shape[:2] for t in (rough, metal) if t is not None]
    height, width = max(s[0] for s in sizes), max(s[1] for s in sizes)
    ones = torch.ones(height, width)
    green = ones if rough is None else _resized(_channel(rough, rough_options.get("-imfchan"), "r"), height, width)
    blue = ones if metal is None else _resized(_channel(metal, metal_options.get("-imfchan"), "r"), height, width)
    source = rough if rough is not None else metal
    material.metallic_roughness_texture = Texture(torch.stack([ones, green, blue], dim=2).contiguous(),
                                                  source.name, source.wrap_s, source.wrap_t, "image/png")
    if rough is not None:
        material.roughness = 1.0
    if metal is not None:
        material.metallic = 1.0


def _read_library(base_dir, name: str, cache: _TextureCache, found: dict[str, Material]) -> None:
    """Read one MTL library into ``found``.

    Args:
        base_dir: Folder holding the OBJ file.
        name: The ``mtllib`` reference.
        cache: Texture reader for that folder.
        found: Materials by name, added to in place.
    """
    reference = name.replace("\\", "/")
    shown = PurePosixPath(reference).name or reference
    data = textures.read_sidecar(base_dir, reference)
    if data is None and " " in reference:
        for single in reference.split()[:MAX_LIBRARIES]:
            _read_library(base_dir, single, cache, found)
        return
    if data is None:
        logger.warning("the material library %s could not be read, so its materials are plain grey",
                       shown)
        return
    if len(data) > MAX_MTL_BYTES:
        logger.warning("the material library %s is %d MB, more than %d MB, so it was skipped",
                       shown, len(data) // (1024 * 1024), MAX_MTL_BYTES // (1024 * 1024))
        return
    current: str | None = None
    statements: list[str] = []
    blocks: list[tuple[str, list[str]]] = []
    for raw in _lines(_normalised(data)):
        line = raw.decode("utf-8", errors="replace").strip()
        if not line or line.startswith("#"):
            continue
        if line.split()[0].lower() == "newmtl":
            if current is not None:
                blocks.append((current, statements))
            current, statements = (_remainder(line, 1) or "material")[:MAX_NAME_CHARS], []
            if len(blocks) >= MAX_MATERIALS:
                raise ValueError(f"the material library {shown} declares "
                                 f"more than {MAX_MATERIALS} materials; the file is damaged")
        elif current is not None:
            statements.append(line)
    if current is not None:
        blocks.append((current, statements))
    for material_name, lines in blocks:
        found.setdefault(material_name, _material(material_name, lines, cache))


def _triangulate_quads(points: np.ndarray, quads: np.ndarray) -> np.ndarray:
    """Split quads along the diagonal that keeps both halves facing the quad's way.

    Args:
        points: ``(N, 3)`` positions.
        quads: ``(Q, 4)`` vertex indices.

    Returns:
        ``(Q, 2, 3)`` triangles.
    """
    p = points[quads]
    normal = np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 1])

    def facing(a: int, b: int, c: int) -> np.ndarray:
        return (np.cross(p[:, b] - p[:, a], p[:, c] - p[:, a]) * normal).sum(axis=1) > 0

    first_ok = facing(0, 1, 2) & facing(0, 2, 3)
    second_ok = facing(1, 2, 3) & facing(1, 3, 0)
    first_shorter = (np.linalg.norm(p[:, 2] - p[:, 0], axis=1)
                     <= np.linalg.norm(p[:, 3] - p[:, 1], axis=1))
    use_first = first_ok & (first_shorter | ~second_ok) | (~first_ok & ~second_ok)
    first = np.stack([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]], axis=1)
    second = np.stack([quads[:, [1, 2, 3]], quads[:, [1, 3, 0]]], axis=1)
    return np.where(use_first[:, None, None], first, second)


def _fan(count: int) -> np.ndarray:
    """``(count - 2, 3)`` corner index triples fanning a polygon from its first corner."""
    middle = np.arange(1, count - 1, dtype=np.int64)
    return np.stack([np.zeros_like(middle), middle, middle + 1], axis=1)


def _convex(rings: np.ndarray) -> np.ndarray:
    """Which polygons turn the same way at every corner, so a fan triangulates them.

    Args:
        rings: ``(P, n, 3)`` corner positions.

    Returns:
        ``(P,)`` bool. A polygon with no area counts as convex.
    """
    rolled = np.roll(rings, -1, axis=1)
    normal = np.stack([
        ((rings[..., 1] - rolled[..., 1]) * (rings[..., 2] + rolled[..., 2])).sum(axis=1),
        ((rings[..., 2] - rolled[..., 2]) * (rings[..., 0] + rolled[..., 0])).sum(axis=1),
        ((rings[..., 0] - rolled[..., 0]) * (rings[..., 1] + rolled[..., 1])).sum(axis=1),
    ], axis=1)
    incoming = rings - np.roll(rings, 1, axis=1)
    outgoing = rolled - rings
    turn = (np.cross(incoming, outgoing) * normal[:, None, :]).sum(axis=2)
    scale = (np.linalg.norm(incoming, axis=2) * np.linalg.norm(outgoing, axis=2)
             * np.linalg.norm(normal, axis=1)[:, None])
    return (turn >= -1e-7 * scale).all(axis=1)


class _Budget:
    """Ear-clipping steps left for one file."""

    def __init__(self, steps: int) -> None:
        self.steps = steps
        self.fanned = 0


def _ear_clip(points: np.ndarray, budget: _Budget) -> np.ndarray:
    """Triangles of one simple polygon, by ear clipping in its own plane.

    Args:
        points: ``(n, 3)`` corners in order.
        budget: Steps left for the file, spent in place.

    Returns:
        ``(n - 2, 3)`` corner index triples, counter-clockwise like the polygon. What remains is
        fanned when no ear is found in a full lap or the budget runs out.
    """
    count = len(points)
    if count > EAR_CLIP_CORNERS or budget.steps <= 0:
        budget.fanned += 1
        return _fan(count)
    rolled = np.roll(points, -1, axis=0)
    normal = np.array([
        ((points[:, 1] - rolled[:, 1]) * (points[:, 2] + rolled[:, 2])).sum(),
        ((points[:, 2] - rolled[:, 2]) * (points[:, 0] + rolled[:, 0])).sum(),
        ((points[:, 0] - rolled[:, 0]) * (points[:, 1] + rolled[:, 1])).sum(),
    ])
    axis = int(np.abs(normal).argmax())
    u, v = [(1, 2), (2, 0), (0, 1)][axis]
    if normal[axis] < 0:
        u, v = v, u
    flat = points[:, [u, v]]
    remaining = list(range(count))
    triangles: list[tuple[int, int, int]] = []

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    cursor = misses = 0
    while len(remaining) > 3 and misses < len(remaining) and budget.steps > 0:
        budget.steps -= 1
        size = len(remaining)
        previous, current, following = (remaining[(cursor - 1) % size], remaining[cursor % size],
                                        remaining[(cursor + 1) % size])
        a, b, c = flat[previous], flat[current], flat[following]
        is_ear = cross(a, b, c) > 0
        if is_ear:
            others = [index for index in remaining if index not in (previous, current, following)]
            if others:
                q = flat[others]
                d1 = (b[0] - a[0]) * (q[:, 1] - a[1]) - (b[1] - a[1]) * (q[:, 0] - a[0])
                d2 = (c[0] - b[0]) * (q[:, 1] - b[1]) - (c[1] - b[1]) * (q[:, 0] - b[0])
                d3 = (a[0] - c[0]) * (q[:, 1] - c[1]) - (a[1] - c[1]) * (q[:, 0] - c[0])
                is_ear = not bool(((d1 >= 0) & (d2 >= 0) & (d3 >= 0)).any())
        if is_ear:
            triangles.append((previous, current, following))
            remaining.pop(cursor % size)
            cursor = max(cursor - 1, 0)
            misses = 0
        else:
            cursor += 1
            misses += 1
    if len(remaining) == 3:
        triangles.append(tuple(remaining))
    elif len(remaining) > 3:
        budget.fanned += 1
        triangles.extend((remaining[0], remaining[i], remaining[i + 1]) for i in range(1, len(remaining) - 1))
    return np.asarray(triangles, dtype=np.int64).reshape(-1, 3)


def _triangulate(points: np.ndarray, sizes: np.ndarray, indices: np.ndarray,
                 budget: _Budget) -> tuple[np.ndarray, np.ndarray]:
    """Triangles of a run of polygons, in polygon order.

    Args:
        points: ``(N, 3)`` positions.
        sizes: ``(P,)`` corner count of each polygon.
        indices: ``(sizes.sum(),)`` vertex indices.
        budget: Ear-clipping steps left for the file, spent in place.

    Returns:
        ``(triangles (M, 3), triangle_polygon (M,))``, both int64.
    """
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    pieces: list[np.ndarray] = []
    owners: list[np.ndarray] = []
    for size in np.unique(sizes).tolist():
        which = np.nonzero(sizes == size)[0]
        corners = indices[starts[which, None] + np.arange(size)[None, :]]
        if size == 3:
            pieces.append(corners)
            owners.append(which)
        elif size == 4:
            pieces.append(_triangulate_quads(points, corners).reshape(-1, 3))
            owners.append(np.repeat(which, 2))
        else:
            fan = _fan(size)
            convex = np.ones(len(which), dtype=bool)
            if size <= _CONVEX_TEST_CORNERS:
                step = max(1, _CONVEX_TEST_CORNERS // size)
                for start in range(0, len(which), step):
                    convex[start:start + step] = _convex(points[corners[start:start + step]])
            if convex.any():
                pieces.append(corners[convex][:, fan].reshape(-1, 3))
                owners.append(np.repeat(which[convex], size - 2))
            for polygon, ring in zip(which[~convex].tolist(), corners[~convex]):
                local = _ear_clip(points[ring], budget)
                pieces.append(ring[local])
                owners.append(np.full(len(local), polygon, dtype=np.int64))
    triangles = np.concatenate(pieces).astype(np.int64)
    owner = np.concatenate(owners).astype(np.int64)
    order = np.argsort(owner, kind="stable")
    return triangles[order], owner[order]


def _unique_corners(v: np.ndarray, t: np.ndarray, n: np.ndarray, counts: tuple[int, int, int]):
    """The distinct ``(v, vt, vn)`` triples of a run of corners.

    Args:
        v: ``(C,)`` position indices.
        t: ``(C,)`` texture coordinate indices, ``-1`` where absent.
        n: ``(C,)`` normal indices, ``-1`` where absent.
        counts: How many positions, texture coordinates and normals the file declares.

    Returns:
        ``(triples (K, 3), inverse (C,))``.
    """
    span_t, span_n = counts[1] + 1, counts[2] + 1
    if counts[0] * span_t * span_n < 2 ** 62:
        key = (v.astype(np.int64) * span_t + (t + 1)) * span_n + (n + 1)
        unique, inverse = np.unique(key, return_inverse=True)
        triples = np.stack([unique // (span_t * span_n), (unique // span_n) % span_t - 1,
                            unique % span_n - 1], axis=1)
        return triples, inverse.reshape(-1)
    triples, inverse = np.unique(np.stack([v, t, n], axis=1), axis=0, return_inverse=True)
    return triples, inverse.reshape(-1)


def _unique_names(names: list[str]) -> list[str]:
    """Names made unique with a numbered suffix."""
    seen: dict[str, int] = {}
    out = []
    for name in names:
        candidate = name
        while candidate in seen:
            seen[name] = seen.get(name, 0) + 1
            candidate = f"{name}.{seen[name]:03d}"
        seen.setdefault(candidate, 0)
        out.append(candidate)
    return out


def read(data: bytes, base_dir: str | None = None, name: str = "") -> Model3D:
    """An OBJ file as a model, with its MTL materials and textures.

    Args:
        data: The OBJ file.
        base_dir: Folder holding it, where its material libraries and textures are read; None
            reads geometry only.
        name: File name, used for the model's name and parts without one.

    Returns:
        A model with one part per object, group and material, ``source_format`` ``"obj"``.

    Raises:
        ValueError: The file is damaged, holds no faces or is past the OBJ limits.
    """
    parsed = _parse(data)
    stem = PurePosixPath(name.replace("\\", "/")).stem if name else "model"
    positions = np.frombuffer(parsed.positions, dtype=np.float32).reshape(-1, 3)
    uv_table = np.frombuffer(parsed.uvs, dtype=np.float32).reshape(-1, 2)
    normal_table = np.frombuffer(parsed.normals, dtype=np.float32).reshape(-1, 3)
    corner_v = np.frombuffer(parsed.corner_v, dtype=np.int32).astype(np.int64)
    corner_t = np.frombuffer(parsed.corner_t, dtype=np.int32).astype(np.int64)
    corner_n = np.frombuffer(parsed.corner_n, dtype=np.int32).astype(np.int64)
    face_size = np.frombuffer(parsed.face_size, dtype=np.int32).astype(np.int64)
    face_group = np.frombuffer(parsed.face_group, dtype=np.int32).astype(np.int64)
    if not len(face_size):
        raise ValueError(f"{name or 'the OBJ file'} holds no faces, only {len(positions)} points; "
                         "load a model with a surface")
    for values, table, kind in ((corner_v, positions, "vertex"), (corner_t, uv_table, "texture coordinate"),
                                (corner_n, normal_table, "normal")):
        if len(values) and int(values.max()) >= len(table):
            raise ValueError(f"a face in the OBJ file names {kind} {int(values.max()) + 1}, but only "
                             f"{len(table)} are declared; the file is damaged or was cut short")
    if len(uv_table) and not np.isfinite(uv_table).all():
        uv_table = np.nan_to_num(uv_table, nan=0.0, posinf=0.0, neginf=0.0)
    colors = None
    if parsed.has_colors:
        raw = np.nan_to_num(np.frombuffer(parsed.colors, dtype=np.float32).reshape(-1, 3),
                            nan=1.0, posinf=1.0, neginf=0.0)
        if raw.size and float(raw.max()) > 1.0:
            raw = raw / 255.0
        colors = np.concatenate([_srgb_to_linear(raw), np.ones((len(raw), 1), np.float32)], axis=1)
    colored = np.frombuffer(bytes(parsed.colored), dtype=np.uint8) if parsed.has_colors else None

    materials: dict[str, Material] = {}
    if base_dir is not None and parsed.libraries:
        cache = _TextureCache(base_dir)
        for library in parsed.libraries:
            _read_library(base_dir, library, cache, materials)
    missing: set[str] = set()

    corner_group = np.repeat(face_group, face_size)
    face_order = np.argsort(face_group, kind="stable")
    corner_order = np.argsort(corner_group, kind="stable")
    group_ids = np.arange(len(parsed.groups) + 1)
    face_bounds = np.searchsorted(face_group[face_order], group_ids)
    corner_bounds = np.searchsorted(corner_group[corner_order], group_ids)
    keys = sorted(parsed.groups, key=parsed.groups.get)
    materials_per_object: dict[tuple[str, str], int] = {}
    for object_name, group_name, _material_name in keys:
        materials_per_object[(object_name, group_name)] = materials_per_object.get((object_name, group_name), 0) + 1
    names: list[str] = []
    parts: list[Part] = []
    budget = _Budget(EAR_CLIP_STEPS)
    for group_id, (object_name, group_name, material_name) in enumerate(keys):
        faces_here = face_order[face_bounds[group_id]:face_bounds[group_id + 1]]
        if not len(faces_here):
            continue
        corners_here = corner_order[corner_bounds[group_id]:corner_bounds[group_id + 1]]
        v, t, n = corner_v[corners_here], corner_t[corners_here], corner_n[corners_here]
        sizes = face_size[faces_here]
        triples, inverse = _unique_corners(v, t, n, (len(positions), len(uv_table), len(normal_table)))
        vertices = positions[triples[:, 0]]
        if not np.isfinite(vertices).all():
            raise ValueError(f"a face in the OBJ file uses a vertex whose position is not a finite "
                             f"number (part '{group_name or object_name or stem}'); the file is "
                             "damaged, so export it again from the 3D editor")
        triangles, owner = _triangulate(vertices, sizes, inverse, budget)
        part_uvs = part_normals = part_colors = None
        if (triples[:, 1] >= 0).any() and len(uv_table):
            part_uvs = np.where((triples[:, 1] >= 0)[:, None], uv_table[triples[:, 1].clip(0)], 0.0)
            part_uvs[:, 1] = 1.0 - part_uvs[:, 1]
        if (triples[:, 2] >= 0).all() and len(normal_table):
            part_normals = normal_table[triples[:, 2]]
            length = np.linalg.norm(part_normals, axis=1, keepdims=True)
            part_normals = np.where(length > 1e-12, part_normals / np.maximum(length, 1e-12), [0.0, 1.0, 0.0])
        if colors is not None and colored[triples[:, 0]].any():
            part_colors = colors[triples[:, 0]]
        if material_name in materials:
            material = materials[material_name]
        else:
            if material_name and base_dir is not None:
                missing.add(material_name)
            material = materials.setdefault(material_name, Material(
                name=material_name or "default", metallic=0.0, roughness=DEFAULT_ROUGHNESS))
        polygons = None
        if (sizes != 3).any():
            polygons = Polygons(torch.from_numpy(sizes.copy()), torch.from_numpy(inverse.astype(np.int64)),
                                torch.from_numpy(owner))
        base = group_name or object_name or stem
        if materials_per_object[(object_name, group_name)] > 1 and material_name:
            base = f"{base}.{material_name}"
        names.append(base)
        parts.append(Part(
            name=base,
            vertices=torch.from_numpy(np.ascontiguousarray(vertices, dtype=np.float32)),
            faces=torch.from_numpy(np.ascontiguousarray(triangles)),
            material=material,
            polygons=polygons,
            uvs=torch.from_numpy(np.ascontiguousarray(part_uvs, dtype=np.float32)) if part_uvs is not None else None,
            normals=(torch.from_numpy(np.ascontiguousarray(part_normals, dtype=np.float32))
                     if part_normals is not None else None),
            colors=(torch.from_numpy(np.ascontiguousarray(part_colors, dtype=np.float32))
                    if part_colors is not None else None),
        ))
    for part, unique in zip(parts, _unique_names(names)):
        part.name = unique
    if missing:
        logger.warning("the OBJ file uses %d material(s) its libraries do not define, drawn plain grey: %s",
                       len(missing), ", ".join(sorted(missing)[:8]))
    if parsed.skipped_faces:
        logger.warning("%d face(s) with fewer than three corners were skipped", parsed.skipped_faces)
    if budget.fanned:
        logger.warning("%d concave or very large polygon(s) were split as fans, which can fold their "
                       "outline; triangulate them in a 3D editor if the surface looks torn",
                       budget.fanned)
    meta = {"name": stem}
    if name:
        meta["source_name"] = PurePosixPath(name.replace("\\", "/")).name
    return Model3D(parts=parts, meta=meta, source_format="obj")


def _safe_name(name: str, fallback: str) -> str:
    """A name reduced to one word of letters, digits, ``_``, ``.`` and ``-``."""
    cleaned = _UNSAFE_NAME.sub("_", str(name).strip()).strip("._")[:_WRITTEN_NAME_CHARS]
    return cleaned or fallback


def _safe_file_part(name: str, fallback: str) -> str:
    """A name reduced to ASCII characters every file system accepts in a file name."""
    cleaned = _UNSAFE_FILE.sub("_", str(name).strip()).strip("._")[:_WRITTEN_NAME_CHARS]
    return cleaned or fallback


def _rows(values: np.ndarray, prefix: str, pattern: str) -> str:
    """Rows of numbers as OBJ statements, one per line."""
    if not len(values):
        return ""
    line = prefix + " " + " ".join([pattern] * values.shape[1])
    return "\n".join(line % row for row in map(tuple, values.tolist())) + "\n"


def _face_lines(sizes: np.ndarray, corner_text: list[str]) -> str:
    """Face statements from per-corner text and polygon sizes."""
    lines = []
    cursor = 0
    for size in sizes.tolist():
        lines.append("f " + " ".join(corner_text[cursor:cursor + size]))
        cursor += size
    return "\n".join(lines) + "\n" if lines else ""


def _texture_file(texture: Texture, file_name: str, files: dict[str, bytes], mime: str | None = None) -> str:
    """Encode ``texture`` into ``files`` and return the name it was written under."""
    data, written = textures.encode(texture, mime)
    stem = file_name.rsplit(".", 1)[0]
    extension = "jpg" if written == "image/jpeg" else "png"
    final, number = f"{stem}.{extension}", 1
    while final.lower() in (existing.lower() for existing in files):
        final, number = f"{stem}_{number}.{extension}", number + 1
    files[final] = data
    return final


def _map_line(keyword: str, file_name: str, texture: Texture, extra: str = "") -> str:
    """One MTL map statement."""
    clamp = " -clamp on" if texture.wrap_s == "clamp" and texture.wrap_t == "clamp" else ""
    return f"{keyword}{clamp}{extra} {file_name}"


def _mtl_block(material: Material, name: str, stem: str, files: dict[str, bytes],
               written: dict[tuple[int, str], str]) -> list[str]:
    """The MTL statements of one material, encoding its textures into ``files``.

    Args:
        material: The material.
        name: Its unique written name.
        stem: Output file stem.
        files: File name to bytes, added to in place.
        written: Texture already written, by ``(id, channel)``, to its file name.

    Returns:
        The statements.
    """
    obj_extras = material.extras.get("obj") if isinstance(material.extras, dict) else None
    obj_extras = obj_extras if isinstance(obj_extras, dict) else {}
    r, g, b, alpha = (float(x) for x in material.base_color)
    roughness = min(max(float(material.roughness), 0.0), 1.0)
    lines = [f"newmtl {name}", f"Kd {r:.6f} {g:.6f} {b:.6f}"]
    ks = obj_extras.get("Ks")
    if not (isinstance(ks, list) and len(ks) >= 3
            and all(isinstance(x, (int, float)) and math.isfinite(x) for x in ks[:3])):
        ks = [0.5, 0.5, 0.5]
    lines.append("Ks " + " ".join(f"{float(x):.6f}" for x in ks[:3]))
    lines.append(f"Ns {(1.0 - roughness) ** 2 * 1000.0:.6f}")
    lines.append(f"d {alpha if material.alpha_mode == 'BLEND' else 1.0:.6f}")
    e = material.emissive
    if max(e) > 0.0 or material.emissive_texture is not None:
        lines.append(f"Ke {e[0]:.6f} {e[1]:.6f} {e[2]:.6f}")
    lines.append(f"Pr {roughness:.6f}")
    lines.append(f"Pm {min(max(float(material.metallic), 0.0), 1.0):.6f}")
    illum = obj_extras.get("illum", 2)
    illum = 0 if material.unlit else (min(max(illum, 0), 10) if type(illum) is int else 2)
    lines.append(f"illum {illum}")

    def file_for(texture: Texture, channel: str, slot: str, mime: str | None) -> str:
        key = (id(texture), channel)
        if key not in written:
            image = texture.image
            if channel in ("g", "b"):
                image = image[..., "rgb".index(channel):"rgb".index(channel) + 1].contiguous()
            elif channel == "rgb":
                image = image[..., :3].contiguous()
            plane = Texture(image, texture.name, texture.wrap_s, texture.wrap_t, texture.mime)
            file_part = _safe_file_part(name, "material")
            written[key] = _texture_file(plane, f"{stem}_{file_part}_{slot}.png", files, mime)
        return written[key]

    base = material.base_color_texture
    if base is not None:
        base_file = file_for(base, "all", "basecolor", None)
        lines.append(_map_line("map_Kd", base_file, base))
        if base.image.shape[2] == 4 and material.alpha_mode != "OPAQUE":
            lines.append(_map_line("map_d", base_file, base, " -imfchan m"))
    mr = material.metallic_roughness_texture
    if mr is not None:
        lines.append(_map_line("map_Pr", file_for(mr, "g", "roughness", "image/png"), mr))
        lines.append(_map_line("map_Pm", file_for(mr, "b", "metallic", "image/png"), mr))
    normal = material.normal_texture
    if normal is not None:
        scale = f" -bm {float(material.normal_scale):.6f}" if float(material.normal_scale) != 1.0 else ""
        lines.append(_map_line("map_Bump", file_for(normal, "rgb", "normal", "image/png"), normal, scale))
    emissive = material.emissive_texture
    if emissive is not None:
        lines.append(_map_line("map_Ke", file_for(emissive, "rgb", "emissive", None), emissive))
    return lines


def _weld(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distinct rows of ``values`` and the row each input row maps to."""
    unique, inverse = np.unique(values, axis=0, return_inverse=True)
    return unique, inverse.reshape(-1)


def write(model: Model3D, stem: str) -> dict[str, bytes]:
    """A model as an OBJ file, its MTL library and its textures.

    Args:
        model: The model to write.
        stem: File name without extension, shared by every file written.

    Returns:
        File name to bytes: ``<stem>.obj``, ``<stem>.mtl`` and one PNG or JPEG per texture map.

    Raises:
        ValueError: The stem is not a plain file name, or the model has no parts.
    """
    from .files import check_stem

    stem = check_stem(stem)
    if not model.parts:
        raise ValueError("the model has no parts to write; wire a model that holds a mesh")
    files: dict[str, bytes] = {}
    material_names: dict[int, str] = {}
    used: set[str] = set()
    ordered: list[Material] = []
    for part in model.parts:
        if id(part.material) in material_names:
            continue
        name = _safe_name(part.material.name, "material")
        candidate, number = name, 1
        while candidate in used:
            candidate, number = f"{name}_{number}", number + 1
        used.add(candidate)
        material_names[id(part.material)] = candidate
        ordered.append(part.material)
    mtl_lines = []
    written: dict[tuple[int, str], str] = {}
    for material in ordered:
        mtl_lines.extend(_mtl_block(material, material_names[id(material)], stem, files, written))
        mtl_lines.append("")

    chunks = [f"mtllib {stem}.mtl\n"]
    offsets = [0, 0, 0]
    part_names = _unique_names([_safe_name(part.name, f"part_{index}") for index, part in enumerate(model.parts)])
    for part, part_name in zip(model.parts, part_names):
        vertices = part.vertices.detach().float().cpu().numpy()
        if not len(vertices) or not part.triangle_count:
            continue
        chunks.append(f"o {part_name}\nusemtl {material_names[id(part.material)]}\n")
        if part.colors is not None:
            srgb = _linear_to_srgb(part.colors.detach().float().cpu().numpy()[:, :3])
            welded, v_index = _weld(np.concatenate([vertices, srgb], axis=1))
            chunks.append(_rows(welded, "v", "%.6f"))
        else:
            welded, v_index = _weld(vertices)
            chunks.append(_rows(welded, "v", "%.6f"))
        t_index = n_index = None
        if part.uvs is not None:
            uvs = part.uvs.detach().float().cpu().numpy().copy()
            uvs[:, 1] = 1.0 - uvs[:, 1]
            uv_rows, t_index = _weld(uvs)
            chunks.append(_rows(uv_rows, "vt", "%.6f"))
        if part.normals is not None:
            normal_rows, n_index = _weld(part.normals.detach().float().cpu().numpy())
            chunks.append(_rows(normal_rows, "vn", "%.6f"))
        if part.polygons is not None and part.polygons.count:
            sizes = part.polygons.sizes.cpu().numpy()
            corners = part.polygons.indices.cpu().numpy()
        else:
            corners = part.faces.cpu().numpy().reshape(-1)
            sizes = np.full(part.triangle_count, 3, dtype=np.int64)
        v_text = (v_index[corners] + offsets[0] + 1).astype(str).tolist()
        if t_index is not None and n_index is not None:
            t_text = (t_index[corners] + offsets[1] + 1).astype(str).tolist()
            n_text = (n_index[corners] + offsets[2] + 1).astype(str).tolist()
            corner_text = [f"{a}/{b}/{c}" for a, b, c in zip(v_text, t_text, n_text)]
        elif t_index is not None:
            t_text = (t_index[corners] + offsets[1] + 1).astype(str).tolist()
            corner_text = [f"{a}/{b}" for a, b in zip(v_text, t_text)]
        elif n_index is not None:
            n_text = (n_index[corners] + offsets[2] + 1).astype(str).tolist()
            corner_text = [f"{a}//{c}" for a, c in zip(v_text, n_text)]
        else:
            corner_text = v_text
        chunks.append(_face_lines(sizes, corner_text))
        offsets[0] += len(welded)
        offsets[1] += 0 if t_index is None else len(uv_rows)
        offsets[2] += 0 if n_index is None else len(normal_rows)
    files_out = {f"{stem}.obj": "".join(chunks).encode("utf-8"),
                 f"{stem}.mtl": "\n".join(mtl_lines).encode("utf-8")}
    files_out.update(files)
    return files_out

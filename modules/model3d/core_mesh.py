"""Exchange between `scene.Model3D` and core ComfyUI's ``MESH``.

A ``MESH`` holds one material per batch item; parts with different textures are packed into one
texture atlas on the way out. Skeletons, skins and morphs do not cross.
"""

from __future__ import annotations

__all__ = ["ATLAS_PADDING", "FACTOR_TILE", "MAX_ATLAS_SIDE", "from_core_mesh", "to_core_mesh"]

import math

import torch

from ..log import get_logger
from .scene import Material, Model3D, Part, Texture

logger = get_logger("model3d.core_mesh")

#: Longest side of a packed texture atlas, in pixels.
MAX_ATLAS_SIDE = 8192

#: Pixels of repeated edge around each atlas tile.
ATLAS_PADDING = 4

#: Side of the tile a part without textures paints its colour into, in pixels.
FACTOR_TILE = 8

#: How far texture coordinates may stray outside ``[0, 1]`` and still be packed into an atlas.
UV_TOLERANCE = 1e-3

#: Base colour core gives a mesh with no texture, no vertex colours and no factor.
CORE_BARE_COLOR = (0.22, 0.22, 0.22, 1.0)

#: Flat tangent-space normal, as a colour.
FLAT_NORMAL = (0.5, 0.5, 1.0)


def _srgb_to_linear(values: torch.Tensor) -> torch.Tensor:
    """sRGB codes in ``[0, 1]`` as linear values."""
    values = values.clamp(0.0, 1.0)
    return torch.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(values: torch.Tensor) -> torch.Tensor:
    """Linear values in ``[0, 1]`` as sRGB codes."""
    values = values.clamp(0.0, 1.0)
    return torch.where(values <= 0.0031308, values * 12.92, 1.055 * values.clamp_min(1e-12) ** (1.0 / 2.4) - 0.055)


def _row(value, index: int, count: int | None) -> torch.Tensor | None:
    """One batch item of a per-vertex or per-face value, cut to its real length.

    Args:
        value: None, a list per item, a ``(B, N, K)`` tensor or an ``(N, K)`` tensor.
        index: Batch item.
        count: Real length, or None for the whole row.

    Returns:
        ``(N, K)`` on the CPU, or None.
    """
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if index >= len(value) or value[index] is None:
            return None
        item = value[index]
    elif value.dim() == 3:
        item = value[min(index, value.shape[0] - 1)]
    else:
        item = value
    item = item.detach().cpu()
    return item[:count] if count is not None else item


def _image(value, index: int) -> torch.Tensor | None:
    """One batch item of a ``(B, H, W, C)`` image value as ``(H, W, 3)`` float32 in ``[0, 1]``."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        item = value[min(index, len(value) - 1)]
        if item is None:
            return None
    else:
        item = value[min(index, value.shape[0] - 1)] if value.dim() == 4 else value
    item = item.detach().cpu().float()
    if item.dim() == 2:
        item = item[..., None]
    if item.dim() != 3 or not item.shape[0] or not item.shape[1] or not item.shape[2]:
        return None
    if item.shape[-1] < 3:
        item = item[..., :1].expand(-1, -1, 3)
    return torch.nan_to_num(item[..., :3], nan=0.0).clamp(0.0, 1.0).contiguous()


def _matching(value: torch.Tensor | None, count: int, widths: tuple[int, ...], attribute: str,
              index: int) -> torch.Tensor | None:
    """A per-vertex value kept only when it has one row per vertex and an accepted width.

    Args:
        value: ``(N, K)`` value, or None.
        count: Vertices in the item.
        widths: Accepted ``K``.
        attribute: Attribute name, for the log.
        index: Batch item, for the log.

    Returns:
        The value as float32, or None when it is absent or does not match.
    """
    if value is None:
        return None
    if value.dim() != 2 or len(value) != count or value.shape[1] not in widths:
        logger.warning("mesh item %d: %s shaped %s does not match its %d vertices, so it was left out",
                       index, attribute, tuple(value.shape), count)
        return None
    return torch.nan_to_num(value.float(), nan=0.0)


def _batch_size(mesh) -> int:
    """Number of batch items a ``MESH`` holds."""
    vertices = mesh.vertices
    if isinstance(vertices, (list, tuple)):
        return len(vertices)
    return int(vertices.shape[0]) if vertices.dim() == 3 else 1


def _factor(value, size: int, default: tuple) -> tuple:
    """A factor list from core's material dict as a tuple of ``size`` floats."""
    if value is None:
        return default
    numbers = [float(x) for x in list(value)[:size]]
    return tuple(numbers + list(default[len(numbers):]))


def from_core_mesh(mesh) -> Model3D:
    """A core ``MESH`` as a model, one part per batch item.

    Args:
        mesh: The ``MESH``.

    Returns:
        A model whose parts carry the mesh's attributes and a material built from its maps and
        material settings, as core's GLB writer would draw them.

    Raises:
        ValueError: Every batch item is empty.
    """
    count = _batch_size(mesh)
    settings = mesh.material if isinstance(getattr(mesh, "material", None), dict) else {}
    parts: list[Part] = []
    for index in range(count):
        vertex_count = face_count = None
        if getattr(mesh, "vertex_counts", None) is not None and getattr(mesh, "face_counts", None) is not None:
            vertex_count = max(0, int(mesh.vertex_counts[index]))
            face_count = max(0, int(mesh.face_counts[index]))
        vertices = _row(mesh.vertices, index, vertex_count)
        faces = _row(mesh.faces, index, face_count)
        if vertices is None or faces is None or not len(vertices) or not len(faces):
            continue
        if vertices.dim() != 2 or vertices.shape[1] != 3 or faces.dim() != 2 or faces.shape[1] != 3:
            raise ValueError(f"mesh item {index} has vertices shaped {tuple(vertices.shape)} and faces "
                             f"shaped {tuple(faces.shape)}, not (N, 3) each; the node that made it "
                             "produced a damaged mesh")
        faces = faces.long()
        if int(faces.min()) < 0 or int(faces.max()) >= len(vertices):
            raise ValueError(f"mesh item {index} has a face naming vertex {int(faces.max())} or "
                             f"{int(faces.min())}, but holds {len(vertices)} vertices; the node that "
                             "made it produced a damaged mesh")
        if not bool(torch.isfinite(vertices.float()).all()):
            raise ValueError(f"mesh item {index} has vertex positions that are not finite numbers; the "
                             "node that made it produced a damaged mesh")
        uvs, colors, normals, tangents = (
            _matching(_row(getattr(mesh, attribute, None), index, vertex_count), len(vertices), widths,
                      attribute, index)
            for attribute, widths in (("uvs", (2,)), ("vertex_colors", (3, 4)), ("normals", (3,)),
                                      ("tangents", (3, 4))))
        if colors is not None:
            colors = colors.float()
            if colors.shape[1] == 3:
                colors = torch.cat([colors, torch.ones(len(colors), 1)], dim=1)
        if tangents is not None and tangents.shape[1] == 3:
            tangents = torch.cat([tangents.float(), torch.ones(len(tangents), 1)], dim=1)
        name = "mesh" if count == 1 else f"mesh_{index}"
        maps = {
            slot: _image(getattr(mesh, attribute, None), index)
            for slot, attribute in (("base", "texture"), ("mr", "metallic_roughness"),
                                    ("normal", "normal_map"), ("emissive", "emissive"))
        }
        if uvs is None:
            maps = dict.fromkeys(maps)
        textures = {slot: Texture(image, name=f"{name}_{slot}") if image is not None else None
                    for slot, image in maps.items()}
        material = _material_from_settings(settings, textures, colors is not None, name,
                                           bool(getattr(mesh, "unlit", False)),
                                           bool(getattr(mesh, "occlusion_in_mr", False)))
        parts.append(Part(
            name=name,
            vertices=vertices.float().contiguous(),
            faces=faces.long().contiguous(),
            material=material,
            uvs=uvs.float().contiguous() if uvs is not None else None,
            normals=normals.float().contiguous() if normals is not None else None,
            tangents=tangents.float().contiguous() if tangents is not None and normals is not None else None,
            colors=colors.contiguous() if colors is not None else None,
        ))
    if not parts:
        raise ValueError("the mesh holds no triangles; wire a mesh with geometry in it")
    return Model3D(parts=parts, source_format="mesh")


def _material_from_settings(settings: dict, textures: dict[str, Texture | None], has_colors: bool,
                            name: str, unlit: bool, occlusion_in_mr: bool) -> Material:
    """A material equal to what core's GLB writer makes of a ``MESH`` item.

    Args:
        settings: Core's material dict, possibly empty.
        textures: ``"base"``, ``"mr"``, ``"normal"`` and ``"emissive"`` textures, or None.
        has_colors: Whether the item has vertex colours.
        name: Material name.
        unlit: Core's unlit flag.
        occlusion_in_mr: Whether the metallic-roughness map holds occlusion in red.

    Returns:
        The material.
    """
    base, mr = textures["base"], textures["mr"]
    if base is not None or has_colors:
        default_color = (1.0, 1.0, 1.0, 1.0)
    else:
        default_color = CORE_BARE_COLOR
    metallic = float(settings.get("metallic_factor", -1.0))
    roughness = float(settings.get("roughness_factor", -1.0))
    if metallic < 0.0:
        metallic = 1.0 if mr is not None else 0.0
    if roughness < 0.0:
        roughness = 1.0 if (mr is not None or base is not None or has_colors) else 0.5
    emissive = _factor(settings.get("emissive_factor"), 3, (0.0, 0.0, 0.0))
    if textures["emissive"] is not None and max(emissive) <= 0.0:
        emissive = (1.0, 1.0, 1.0)
    material = Material(
        name=name,
        base_color=_factor(settings.get("base_color_factor"), 4, default_color),
        base_color_texture=base,
        metallic=metallic,
        roughness=roughness,
        metallic_roughness_texture=mr,
        normal_texture=textures["normal"],
        normal_scale=float(settings.get("normal_scale", 1.0)),
        occlusion_texture=mr if occlusion_in_mr and mr is not None else None,
        occlusion_strength=float(settings.get("occlusion_strength", 1.0)),
        emissive=tuple(min(1.0, c) for c in emissive),
        emissive_texture=textures["emissive"],
        double_sided=bool(settings.get("double_sided", True)),
        unlit=unlit,
    )
    strength = settings.get("emissive_strength")
    if strength is not None and float(strength) != 1.0:
        material.extras["emissive_strength"] = float(strength)
    return material


def _vertex_normals(part: Part) -> torch.Tensor:
    """Area-weighted vertex normals of a part."""
    vertices, faces = part.vertices.float(), part.faces.long()
    corners = vertices[faces]
    face_normals = torch.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], dim=1)
    normals = torch.zeros_like(vertices)
    for corner in range(3):
        normals.index_add_(0, faces[:, corner], face_normals)
    length = normals.norm(dim=1, keepdim=True)
    fallback = torch.tensor([0.0, 1.0, 0.0]).expand_as(normals)
    return torch.where(length > 1e-20, normals / length.clamp_min(1e-20), fallback)


def _resized(image: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """``(H, W, C)`` resampled to ``height`` by ``width``."""
    if image.shape[0] == height and image.shape[1] == width:
        return image
    shrinking = height < image.shape[0] or width < image.shape[1]
    planar = image.permute(2, 0, 1)[None].float()
    out = torch.nn.functional.interpolate(planar, size=(height, width), mode="bilinear",
                                          align_corners=False, antialias=shrinking)
    return out[0].permute(1, 2, 0).clamp(0.0, 1.0).contiguous()


def _occlusion_plane(material: Material) -> torch.Tensor | None:
    """The occlusion map's red channel with its strength applied, or None."""
    texture = material.occlusion_texture
    if texture is None:
        return None
    plane = texture.image[..., 0].float()
    strength = float(material.occlusion_strength)
    return (1.0 + strength * (plane - 1.0)).clamp(0.0, 1.0)


def _scaled_normals(image: torch.Tensor, scale: float) -> torch.Tensor:
    """A normal map with its strength baked in."""
    if scale == 1.0:
        return image[..., :3]
    vectors = image[..., :3] * 2.0 - 1.0
    vectors = torch.cat([vectors[..., :2] * scale, vectors[..., 2:3]], dim=-1)
    vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return (vectors * 0.5 + 0.5).clamp(0.0, 1.0)


class _Tile:
    """One region of the atlas: a material, its parts, and where it lands."""

    def __init__(self, material: Material, textured: bool) -> None:
        self.material = material
        self.textured = textured
        self.parts: list[int] = []
        if textured:
            sizes = [texture.image.shape[:2] for texture in material.textures().values()]
            base = material.base_color_texture
            height, width = base.image.shape[:2] if base is not None else max(sizes, key=lambda s: s[0] * s[1])
            self.source = (int(height), int(width))
        else:
            self.source = (FACTOR_TILE, FACTOR_TILE)
        self.size = self.source
        self.origin = (0, 0)

    def inner(self) -> tuple[int, int, int, int]:
        """``(x, y, width, height)`` of the tile's content inside the atlas."""
        return self.origin[0] + ATLAS_PADDING, self.origin[1] + ATLAS_PADDING, self.size[1], self.size[0]


def _shelf_pack(tiles: list[_Tile]) -> tuple[int, int]:
    """Place every tile on shelves, tallest first, at the largest common scale that fits.

    Args:
        tiles: Tiles with their source sizes; their ``size`` and ``origin`` are set in place.

    Returns:
        ``(width, height)`` of the atlas.

    Raises:
        ValueError: The tiles cannot fit within `MAX_ATLAS_SIDE` at any scale.
    """
    scale = 1.0
    for _ in range(40):
        for tile in tiles:
            if tile.textured:
                tile.size = (max(4, int(math.ceil(tile.source[0] * scale))),
                             max(4, int(math.ceil(tile.source[1] * scale))))
            else:
                tile.size = tile.source
        padded = [(tile.size[1] + 2 * ATLAS_PADDING, tile.size[0] + 2 * ATLAS_PADDING) for tile in tiles]
        area = sum(w * h for w, h in padded)
        width = max(max(w for w, _ in padded), int(math.ceil(math.sqrt(area))))
        order = sorted(range(len(tiles)), key=lambda i: (-padded[i][1], -padded[i][0]))
        x = y = shelf = 0
        right = 0
        for index in order:
            w, h = padded[index]
            if x + w > width:
                y, x, shelf = y + shelf, 0, 0
            tiles[index].origin = (x, y)
            x += w
            shelf = max(shelf, h)
            right = max(right, x)
        height = y + shelf
        side = max(right, height)
        if side <= MAX_ATLAS_SIDE:
            return right, height
        scale *= MAX_ATLAS_SIDE / side * 0.99
    raise ValueError(f"the model's {len(tiles)} textures cannot be packed into one "
                     f"{MAX_ATLAS_SIDE} pixel atlas; merge parts that share a material first")


def _paint(atlas: torch.Tensor, tile: _Tile, content: torch.Tensor) -> None:
    """Place a tile's content and its repeated-edge padding into the atlas."""
    x, y, width, height = tile.inner()
    content = _resized(content, height, width) if content.shape[0] > 1 or content.shape[1] > 1 \
        else content.expand(height, width, -1)
    padded = torch.nn.functional.pad(content.permute(2, 0, 1)[None], (ATLAS_PADDING,) * 4,
                                     mode="replicate")[0].permute(1, 2, 0)
    atlas[y - ATLAS_PADDING:y + height + ATLAS_PADDING, x - ATLAS_PADDING:x + width + ATLAS_PADDING] = padded


def _base_content(material: Material, textured: bool) -> torch.Tensor:
    """A tile's base colour as sRGB codes with the material's factor baked in."""
    factor = torch.tensor(material.base_color[:3], dtype=torch.float32)
    texture = material.base_color_texture if textured else None
    if texture is None:
        return _linear_to_srgb(factor).view(1, 1, 3)
    image = texture.image[..., :3].float()
    if bool((factor == 1.0).all()):
        return image
    return _linear_to_srgb(_srgb_to_linear(image) * factor)


def _mr_content(material: Material, textured: bool, height: int, width: int) -> torch.Tensor:
    """A tile's occlusion, roughness and metalness with the material's factors baked in."""
    mr = material.metallic_roughness_texture if textured else None
    occlusion = _occlusion_plane(material) if textured else None
    if mr is None and occlusion is None:
        return torch.tensor([1.0, float(material.roughness), float(material.metallic)]).view(1, 1, 3)
    if mr is not None:
        image = _resized(mr.image[..., :3].float(), height, width)
        rough = image[..., 1] * float(material.roughness)
        metal = image[..., 2] * float(material.metallic)
    else:
        rough = torch.full((height, width), float(material.roughness))
        metal = torch.full((height, width), float(material.metallic))
    if occlusion is not None:
        red = _resized(occlusion[..., None], height, width)[..., 0]
    else:
        red = torch.ones(height, width)
    return torch.stack([red, rough, metal], dim=2)


def _normal_content(material: Material, textured: bool) -> torch.Tensor:
    """A tile's tangent-space normals with the material's strength baked in."""
    texture = material.normal_texture if textured else None
    if texture is None:
        return torch.tensor(FLAT_NORMAL).view(1, 1, 3)
    return _scaled_normals(texture.image.float(), float(material.normal_scale))


def _emissive_content(material: Material, textured: bool) -> torch.Tensor:
    """A tile's emissive colour as sRGB codes with the material's factor baked in."""
    factor = torch.tensor(material.emissive, dtype=torch.float32).clamp(0.0, 1.0)
    texture = material.emissive_texture if textured else None
    if texture is None:
        return _linear_to_srgb(factor).view(1, 1, 3)
    image = texture.image[..., :3].float()
    return image if bool((factor == 1.0).all()) else _linear_to_srgb(_srgb_to_linear(image) * factor)


def _has_maps(part: Part) -> bool:
    """Whether a part has texture coordinates and at least one map to sample."""
    return part.uvs is not None and bool(part.material.textures())


def _uvs_inside(part: Part) -> bool:
    """Whether every texture coordinate of a part lies within the unit square."""
    uvs = part.uvs
    return bool(((uvs >= -UV_TOLERANCE) & (uvs <= 1.0 + UV_TOLERANCE)).all())


def _scalars(material: Material) -> tuple:
    """The settings of a material that a ``MESH`` keeps once, for the whole item."""
    return (tuple(round(float(x), 6) for x in material.base_color), round(float(material.metallic), 6),
            round(float(material.roughness), 6), tuple(round(float(x), 6) for x in material.emissive),
            round(float(material.normal_scale), 6), round(float(material.occlusion_strength), 6))


def _settings(material: Material, emissive_texture: bool) -> dict:
    """Core's material dict for one material."""
    settings = {
        "base_color_factor": [float(x) for x in material.base_color],
        "metallic_factor": float(material.metallic),
        "roughness_factor": float(material.roughness),
        "double_sided": bool(material.double_sided),
    }
    if float(material.normal_scale) != 1.0:
        settings["normal_scale"] = float(material.normal_scale)
    if float(material.occlusion_strength) != 1.0:
        settings["occlusion_strength"] = float(material.occlusion_strength)
    emissive = [float(x) for x in material.emissive]
    if any(c > 0.0 for c in emissive) or emissive_texture:
        settings["emissive_factor"] = emissive if any(c > 0.0 for c in emissive) else [1.0, 1.0, 1.0]
    strength = material.extras.get("emissive_strength") if isinstance(material.extras, dict) else None
    if strength is not None:
        settings["emissive_strength"] = float(strength)
    return settings


def _single_maps(material: Material) -> dict:
    """``MESH`` maps for one material used as it is.

    Returns:
        ``texture``, ``metallic_roughness``, ``normal_map``, ``emissive`` (each ``(1, H, W, 3)`` or
        None), ``occlusion_in_mr`` and the material dict.
    """
    def batched(image: torch.Tensor | None) -> torch.Tensor | None:
        return None if image is None else image[..., :3].float().clamp(0.0, 1.0).contiguous()[None]

    base = material.base_color_texture
    mr = material.metallic_roughness_texture
    occlusion = material.occlusion_texture
    settings = _settings(material, material.emissive_texture is not None)
    mr_image = mr.image[..., :3].float() if mr is not None else None
    occlusion_in_mr = False
    if occlusion is not None:
        occlusion_in_mr = True
        if occlusion is not mr:
            if mr_image is None:
                height, width = occlusion.image.shape[:2]
                mr_image = torch.ones(height, width, 3)
            else:
                height, width = mr_image.shape[:2]
                mr_image = mr_image.clone()
            mr_image[..., 0] = _resized(occlusion.image[..., :1].float(), height, width)[..., 0]
    return {
        "texture": batched(base.image if base is not None else None),
        "metallic_roughness": batched(mr_image),
        "normal_map": batched(material.normal_texture.image if material.normal_texture is not None else None),
        "emissive": batched(material.emissive_texture.image if material.emissive_texture is not None else None),
        "occlusion_in_mr": occlusion_in_mr,
        "material": settings,
    }


def _atlas_maps(parts: list[Part]) -> tuple[dict, list[torch.Tensor]]:
    """Pack every part's maps into one atlas and remap each part's texture coordinates.

    Args:
        parts: The parts, in output order.

    Returns:
        ``(maps, uvs)``: the ``MESH`` maps and material dict as `_single_maps` returns them, and
        each part's ``(N, 2)`` texture coordinates inside the atlas.
    """
    tiles: dict[tuple[int, bool], _Tile] = {}
    owner: list[_Tile] = []
    for index, part in enumerate(parts):
        textured = _has_maps(part)
        key = (id(part.material), textured)
        if key not in tiles:
            tiles[key] = _Tile(part.material, textured)
        tiles[key].parts.append(index)
        owner.append(tiles[key])
    layout = list(tiles.values())
    width, height = _shelf_pack(layout)
    materials = [tile.material for tile in layout]
    textured = [tile for tile in layout if tile.textured]
    need_mr = (any(m.metallic_roughness_texture is not None or m.occlusion_texture is not None
                   for m in (t.material for t in textured))
               or len({(round(m.metallic, 6), round(m.roughness, 6)) for m in materials}) > 1)
    need_normal = any(t.material.normal_texture is not None for t in textured)
    need_emissive = (any(t.material.emissive_texture is not None for t in textured)
                     or any(max(m.emissive) > 0.0 for m in materials))
    occlusion_in_mr = any(t.material.occlusion_texture is not None for t in textured)

    base_atlas = torch.zeros(height, width, 3)
    mr_atlas = torch.ones(height, width, 3) if need_mr else None
    normal_atlas = torch.tensor(FLAT_NORMAL).expand(height, width, 3).clone() if need_normal else None
    emissive_atlas = torch.zeros(height, width, 3) if need_emissive else None
    for tile in layout:
        material, use = tile.material, tile.textured
        _paint(base_atlas, tile, _base_content(material, use))
        if mr_atlas is not None:
            _paint(mr_atlas, tile, _mr_content(material, use, tile.source[0], tile.source[1]))
        if normal_atlas is not None:
            _paint(normal_atlas, tile, _normal_content(material, use))
        if emissive_atlas is not None:
            _paint(emissive_atlas, tile, _emissive_content(material, use))

    remapped: list[torch.Tensor] = []
    for part, tile in zip(parts, owner):
        x, y, tile_width, tile_height = tile.inner()
        if tile.textured:
            uvs = part.uvs.float().clamp(0.0, 1.0)
            u = (x + uvs[:, 0] * tile_width) / width
            v = (y + uvs[:, 1] * tile_height) / height
        else:
            u = torch.full((part.vertex_count,), (x + tile_width / 2.0) / width)
            v = torch.full((part.vertex_count,), (y + tile_height / 2.0) / height)
        remapped.append(torch.stack([u, v], dim=1).float())

    first = parts[0].material
    settings = {
        "base_color_factor": [1.0, 1.0, 1.0, 1.0],
        "metallic_factor": 1.0 if need_mr else float(first.metallic),
        "roughness_factor": 1.0 if need_mr else float(first.roughness),
        "double_sided": any(bool(m.double_sided) for m in materials),
    }
    if need_emissive:
        settings["emissive_factor"] = [1.0, 1.0, 1.0]
    strengths = {m.extras.get("emissive_strength") for m in materials if isinstance(m.extras, dict)}
    if len(strengths) == 1 and None not in strengths:
        settings["emissive_strength"] = float(strengths.pop())
    maps = {
        "texture": base_atlas[None],
        "metallic_roughness": mr_atlas[None] if mr_atlas is not None else None,
        "normal_map": normal_atlas[None] if normal_atlas is not None else None,
        "emissive": emissive_atlas[None] if emissive_atlas is not None else None,
        "occlusion_in_mr": occlusion_in_mr,
        "material": settings,
    }
    logger.info("packed %d material(s) into one %dx%d texture atlas", len(layout), width, height)
    return maps, remapped


def _rgba(colors: torch.Tensor) -> torch.Tensor:
    """``(N, 3)`` or ``(N, 4)`` vertex colours as ``(N, 4)`` float32."""
    colors = colors.float()
    if colors.shape[1] >= 4:
        return colors[:, :4]
    return torch.cat([colors[:, :3], torch.ones(len(colors), 4 - colors.shape[1])], dim=1)


def to_core_mesh(model: Model3D):
    """A model as one core ``MESH`` batch item, its parts merged and their textures atlased.

    Args:
        model: The model.

    Returns:
        A ``MESH`` with float32 ``(1, N, 3)`` vertices, int64 ``(1, M, 3)`` faces and the
        texture coordinates, colours, normals, tangents, maps and material dict that apply.

    Raises:
        ValueError: The model has no triangles, or its textures cannot fit one atlas.
    """
    from comfy_api.latest import Types

    parts = [part for part in model.parts if part.vertex_count and part.triangle_count]
    if not parts:
        raise ValueError("the model has no triangles to pass on as a mesh; wire a model with geometry")
    offsets, total = [], 0
    for part in parts:
        offsets.append(total)
        total += part.vertex_count
    vertices = torch.cat([part.vertices.float() for part in parts])
    faces = torch.cat([part.faces.long() + offset for part, offset in zip(parts, offsets)])
    normals = torch.cat([part.normals.float() if part.normals is not None else _vertex_normals(part)
                         for part in parts])
    tangents = None
    if all(part.tangents is not None for part in parts):
        tangents = torch.cat([part.tangents.float() for part in parts])
    colors = None
    if any(part.colors is not None for part in parts):
        colors = torch.cat([_rgba(part.colors) if part.colors is not None
                            else torch.ones(part.vertex_count, 4) for part in parts])

    textured = [part for part in parts if _has_maps(part)]
    materials = {id(part.material): part.material for part in parts}
    same_scalars = len({_scalars(m) for m in materials.values()}) == 1
    uvs_list: list[torch.Tensor | None]
    if not textured and same_scalars:
        maps = {"texture": None, "metallic_roughness": None, "normal_map": None, "emissive": None,
                "occlusion_in_mr": False, "material": _settings(parts[0].material, False)}
        uvs_list = [part.uvs for part in parts]
    elif (len(textured) == len(parts) and same_scalars
          and len({tuple(id(t) for t in part.material.textures().values()) for part in parts}) == 1):
        maps = _single_maps(parts[0].material)
        uvs_list = [part.uvs for part in parts]
    else:
        outside = [part.name for part in textured if not _uvs_inside(part)]
        if outside:
            logger.warning("the textures of %d part(s) could not be packed into one atlas because "
                           "their texture coordinates repeat outside the image (%s); the first "
                           "part's textures are used for the whole mesh", len(outside),
                           ", ".join(outside[:6]))
            maps = _single_maps(textured[0].material)
            uvs_list = [part.uvs for part in parts]
        else:
            maps, uvs_list = _atlas_maps(parts)
    uvs = None
    if any(entry is not None for entry in uvs_list):
        uvs = torch.cat([entry.float() if entry is not None else torch.zeros(part.vertex_count, 2)
                         for entry, part in zip(uvs_list, parts)])
    if uvs is None:
        maps.update(texture=None, metallic_roughness=None, normal_map=None, emissive=None,
                    occlusion_in_mr=False)
    return Types.MESH(
        vertices.contiguous()[None],
        faces.contiguous()[None],
        uvs=uvs.contiguous()[None] if uvs is not None else None,
        vertex_colors=colors.contiguous()[None] if colors is not None else None,
        texture=maps["texture"],
        metallic_roughness=maps["metallic_roughness"],
        unlit=all(bool(part.material.unlit) for part in parts),
        normals=normals.contiguous()[None],
        tangents=tangents.contiguous()[None] if tangents is not None else None,
        normal_map=maps["normal_map"],
        occlusion_in_mr=bool(maps["occlusion_in_mr"]),
        material=maps["material"],
        emissive=maps["emissive"],
    )

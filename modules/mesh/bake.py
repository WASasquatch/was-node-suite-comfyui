"""Baking a source surface's look into a target's texture layout. Texels come from the target's
texture coordinates."""
# Rays cast both ways along the target's normals find the source, whose materials are sampled there;
# without a source, each part's previous layout and material are read on the same triangle. Normal
# maps are in the target's MikkTSpace frame, +Y up.

from __future__ import annotations

__all__ = [
    "DEFAULT_MAPS",
    "MAPS",
    "PREVIOUS_MATERIAL",
    "PREVIOUS_TANGENTS",
    "PREVIOUS_UVS",
    "BakeSettings",
    "Baked",
    "bake",
    "has_previous",
    "keep_previous",
    "linear_to_srgb",
    "srgb_to_linear",
]

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import torch

from ..log import get_logger
from ..model3d.scene import Material, Model3D, Part, Texture
from . import device as mesh_device
from . import raster, tangents, topology
from .bvh import TriangleBVH

logger = get_logger("mesh.bake")

#: Maps a bake can make, in output order.
MAPS = ("base_color", "normal", "metallic_roughness", "emissive", "occlusion")

#: Maps made when none are named.
DEFAULT_MAPS = ("base_color", "normal", "metallic_roughness", "emissive")

#: Part extras key holding ``(F, 3, 2)`` float32 texture coordinates per corner before an unwrap.
PREVIOUS_UVS = "was_previous_uvs"

#: Part extras key holding ``(F, 3, 4)`` float32 tangents per corner before an unwrap.
PREVIOUS_TANGENTS = "was_previous_tangents"

#: Part extras key holding the `Material` a part was drawn with before an unwrap.
PREVIOUS_MATERIAL = "was_previous_material"

#: Widest baked texture, in texels.
MAX_RESOLUTION = 8192

#: Most samples along one side of a texel.
MAX_SAMPLES = 4

#: Most occlusion rays per texel.
MAX_AO_RAYS = 256

#: Most bytes the per-texel sums and images of one bake may hold.
MAX_ACCUMULATOR_BYTES = 6 << 30

#: Occlusion rays traced per pass.
AO_BATCH_RAYS = 1 << 21

#: Distance rays start off the surface, as a share of the target's diagonal.
SURFACE_OFFSET = 2e-6

#: Grid used to find parts sharing texture space.
OVERLAP_GRID = 512

#: Share of covered texels two parts may share before the bake refuses.
OVERLAP_SHARE = 0.01

#: Channels each map sums per texel.
CHANNELS = {"base_color": 4, "normal": 3, "metallic_roughness": 2, "emissive": 3, "occlusion": 1}


@dataclass
class BakeSettings:
    """How a bake runs.

    Attributes:
        resolution: Width and height of every baked texture, in texels.
        maps: Names from `MAPS` to bake.
        cage: Farthest distance searched for the source along each normal, as a share of the
            target's bounding box diagonal.
        samples: Samples along one side of each texel; ``samples ** 2`` per texel.
        padding: Texels each chart's colours are grown by into the empty space around it.
        ao_rays: Occlusion rays per texel.
        ao_distance: Farthest occluder counted, as a share of the target's diagonal.
        seed: Seed of the occlusion ray pattern.
    """

    resolution: int = 2048
    maps: tuple[str, ...] = DEFAULT_MAPS
    cage: float = 0.02
    samples: int = 1
    padding: int = 8
    ao_rays: int = 16
    ao_distance: float = 0.1
    seed: int = 0


@dataclass
class Baked:
    """What a bake produced.

    Attributes:
        images: ``(H, W, C)`` float32 images in ``[0, 1]`` by map name, colour maps as sRGB codes.
        material: A material carrying the baked maps, for every baked part.
        stats: Named figures: samples, hits, misses, coverage, seconds.
    """

    images: dict[str, torch.Tensor]
    material: Material
    stats: dict = field(default_factory=dict)


def srgb_to_linear(codes: torch.Tensor) -> torch.Tensor:
    """sRGB codes in ``[0, 1]`` as linear light."""
    codes = codes.clamp(0.0, 1.0)
    return torch.where(codes <= 0.04045, codes / 12.92, ((codes + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(values: torch.Tensor) -> torch.Tensor:
    """Linear light in ``[0, 1]`` as sRGB codes."""
    values = values.clamp(0.0, 1.0)
    return torch.where(values <= 0.0031308, values * 12.92, 1.055 * values.clamp_min(1e-12) ** (1 / 2.4) - 0.055)


def has_previous(part: Part) -> bool:
    """Whether a part still carries the layout and material it had before an unwrap.

    Args:
        part: The part to test.

    Returns:
        True when the stored layout has one row per triangle of the part.
    """
    uvs = part.extras.get(PREVIOUS_UVS)
    return (isinstance(uvs, torch.Tensor) and uvs.shape == (part.triangle_count, 3, 2)
            and isinstance(part.extras.get(PREVIOUS_MATERIAL), Material))


def keep_previous(part: Part, before: Part) -> None:
    """Store the texture coordinates, tangents and material of a part's earlier state, per corner.

    Args:
        part: The part to store them on, changed in place.
        before: The same triangles, row for row, as they were; it needs texture coordinates.
    """
    faces = before.faces.long()
    part.extras[PREVIOUS_UVS] = before.uvs[faces].float().clone()
    if before.tangents is not None:
        part.extras[PREVIOUS_TANGENTS] = before.tangents[faces].float().clone()
    else:
        part.extras.pop(PREVIOUS_TANGENTS, None)
    part.extras[PREVIOUS_MATERIAL] = before.material


# ---------------------------------------------------------------------------------------------
# Surfaces


@dataclass
class _Surface:
    """Triangles with what a bake reads off them, on one device."""

    vertices: torch.Tensor
    faces: torch.Tensor
    face_part: torch.Tensor
    corner_uvs: torch.Tensor
    normals: torch.Tensor
    corner_tangents: torch.Tensor | None
    colors: torch.Tensor | None
    materials: list[Material]
    textures: dict = field(default_factory=dict)
    face_normals: torch.Tensor | None = None
    tree: TriangleBVH | None = None


def _normals_of(part: Part) -> torch.Tensor:
    """A part's unit vertex normals, smooth ones where it has none."""
    if part.normals is not None and part.normals.shape[0] == part.vertex_count:
        return torch.nn.functional.normalize(part.normals.float(), dim=1)
    return tangents.smooth_normals(part.vertices, part.faces)


def _texture(surface: _Surface, part: int, slot: str, device: torch.device):
    """A part's texture for one slot on the device, with its wrap modes, or None."""
    key = (part, slot)
    if key not in surface.textures:
        texture = getattr(surface.materials[part], f"{slot}_texture", None)
        if texture is None or texture.image.numel() == 0:
            surface.textures[key] = None
        else:
            surface.textures[key] = (texture.image.to(device, torch.float32), texture.wrap_s, texture.wrap_t)
    return surface.textures[key]


def _mikk_corners(part: Part, normals: torch.Tensor, corner_uvs: torch.Tensor) -> torch.Tensor:
    """MikkTSpace tangents of a part's corners under per-corner texture coordinates."""
    faces = part.faces.long()
    if part.uvs is not None and torch.equal(part.uvs[faces].float(), corner_uvs.float()):
        return tangents.corner_tangents(part.vertices, faces, normals, part.uvs)
    flat = faces.reshape(-1)
    split = torch.arange(flat.shape[0]).reshape(-1, 3)
    return tangents.corner_tangents(part.vertices[flat], split, normals[flat], corner_uvs.reshape(-1, 2))


def _assemble(parts: list[Part], corner_uvs: list[torch.Tensor | None], corner_tangents: list[torch.Tensor | None],
              materials: list[Material], device: torch.device, need_tangents: bool) -> _Surface:
    """Parts joined into one `_Surface`.

    Args:
        parts: The parts.
        corner_uvs: Per part, ``(F, 3, 2)`` texture coordinates per corner, or None.
        corner_tangents: Per part, ``(F, 3, 4)`` tangents per corner, or None to work them out
            where the part has a normal map.
        materials: Per part, the material its corners sample.
        device: Device the surface lives on.
        need_tangents: Build tangents for parts with a normal map.
    """
    vertices, faces, face_part, uv_rows, normal_rows, tangent_rows, color_rows = [], [], [], [], [], [], []
    offset = 0
    any_colors = any(part.colors is not None for part in parts)
    for index, part in enumerate(parts):
        part_faces = part.faces.long()
        normals = _normals_of(part)
        uvs = corner_uvs[index]
        vertices.append(part.vertices.float())
        faces.append(part_faces + offset)
        face_part.append(torch.full((part.triangle_count,), index, dtype=torch.int64))
        normal_rows.append(normals)
        uv_rows.append(uvs.float() if uvs is not None else torch.zeros(part.triangle_count, 3, 2))
        if any_colors:
            colors = part.colors.float() if part.colors is not None else torch.ones(part.vertex_count, 4)
            color_rows.append(colors)
        corner = corner_tangents[index]
        if corner is None and need_tangents and uvs is not None and materials[index].normal_texture is not None:
            corner = _mikk_corners(part, normals, uvs)
        tangent_rows.append(corner.float() if corner is not None else torch.zeros(part.triangle_count, 3, 4))
        offset += part.vertex_count
    joined_vertices = torch.cat(vertices).to(device)
    joined_faces = torch.cat(faces).to(device)
    return _Surface(
        vertices=joined_vertices,
        faces=joined_faces,
        face_part=torch.cat(face_part).to(device),
        corner_uvs=torch.cat(uv_rows).to(device),
        normals=torch.cat(normal_rows).to(device),
        corner_tangents=torch.cat(tangent_rows).to(device) if need_tangents else None,
        colors=torch.cat(color_rows).to(device) if any_colors else None,
        materials=list(materials),
        face_normals=topology.face_normals(joined_vertices, joined_faces).float(),
    )


def _target_surface(parts: list[Part], device: torch.device) -> tuple[_Surface, torch.Tensor]:
    """The target parts as a surface carrying MikkTSpace tangents of their current layout.

    Returns:
        ``(surface, offset)``: the surface and the ``(2,)`` whole-tile shift that brings the
        layout into the unit square.
    """
    corner_uvs, corner_tangents = [], []
    for part in parts:
        normals = _normals_of(part)
        corner_uvs.append(part.uvs[part.faces.long()].float())
        corner_tangents.append(tangents.corner_tangents(part.vertices, part.faces, normals, part.uvs))
    surface = _assemble(parts, corner_uvs, corner_tangents, [part.material for part in parts], device, True)
    flat = surface.corner_uvs.reshape(-1, 2)
    low = torch.floor(flat.min(dim=0).values)
    high = flat.max(dim=0).values
    if bool(((high - low) > 1.0 + 1e-3).any()):
        logger.warning("the target's texture coordinates reach past one texture tile; only the part inside "
                       "the tile from %s is baked", [int(v) for v in low.tolist()])
    return surface, low


def _source_surface(model: Model3D, device: torch.device, need_tangents: bool) -> _Surface:
    """A source model's parts as one surface with a tree for ray casting."""
    parts = [part for part in model.parts if part.triangle_count > 0]
    if not parts:
        raise ValueError("the source model has no triangles; wire the model the textures come from into source")
    corner_uvs, corner_tangents = [], []
    for part in parts:
        if part.uvs is None and part.material.textures():
            logger.warning("source part '%s' has textures but no texture coordinates; only its colour "
                           "factors are baked", part.name)
        corner_uvs.append(part.uvs[part.faces.long()].float() if part.uvs is not None else None)
        corner_tangents.append(part.tangents[part.faces.long()].float() if part.tangents is not None else None)
    surface = _assemble(parts, corner_uvs, corner_tangents, [part.material for part in parts], device, need_tangents)
    surface.tree = TriangleBVH(surface.vertices, surface.faces, device=device)
    return surface


def _previous_surface(parts: list[Part], device: torch.device, need_tangents: bool, need_tree: bool) -> _Surface:
    """The target parts read through the layout and material they had before an unwrap."""
    missing = [part.name for part in parts if not has_previous(part)]
    if missing:
        raise ValueError(
            f"nothing to bake from: part(s) {', '.join(missing[:6])} carry no earlier texture layout. Wire the "
            "model the look comes from, such as the high poly before decimation, into source, or run Mesh UV "
            "Unwrap on a textured model just before this node")
    surface = _assemble(parts, [part.extras[PREVIOUS_UVS] for part in parts],
                        [part.extras.get(PREVIOUS_TANGENTS) for part in parts],
                        [part.extras[PREVIOUS_MATERIAL] for part in parts], device, need_tangents)
    if need_tree:
        surface.tree = TriangleBVH(surface.vertices, surface.faces, device=device)
    return surface


# ---------------------------------------------------------------------------------------------
# Sampling


def _interpolate(values: torch.Tensor, faces: torch.Tensor, face: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Per-vertex values blended across each sample's triangle."""
    return (values[faces[face]] * weights[:, :, None]).sum(dim=1)


def _corners(values: torch.Tensor, face: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Per-corner values blended across each sample's triangle."""
    return (values[face] * weights[:, :, None]).sum(dim=1)


def _frame(surface: _Surface, face: torch.Tensor, weights: torch.Tensor):
    """Unit normal, unit tangent and bitangent at samples, the tangent kept square to the normal."""
    normal = torch.nn.functional.normalize(_interpolate(surface.normals, surface.faces, face, weights), dim=1)
    corner = surface.corner_tangents[face]
    tangent = (corner[:, :, :3] * weights[:, :, None]).sum(dim=1)
    tangent = torch.nn.functional.normalize(tangent - (tangent * normal).sum(1, keepdim=True) * normal, dim=1)
    sign = torch.where(corner[torch.arange(face.shape[0], device=face.device), weights.argmax(dim=1), 3] < 0, -1.0, 1.0)
    bitangent = torch.cross(normal, tangent, dim=1) * sign[:, None]
    return normal, tangent, bitangent


def _sample_maps(surface: _Surface, face: torch.Tensor, weights: torch.Tensor, wanted: set[str],
                 target_frame, device: torch.device) -> dict[str, torch.Tensor]:
    """Linear map values of a surface at samples.

    Args:
        surface: The surface sampled.
        face: ``(K,)`` triangle of each sample on it.
        weights: ``(K, 3)`` corner weights.
        wanted: Map names to sample.
        target_frame: ``(normal, tangent, bitangent)`` of the target at each sample.
        device: Device of the samples.

    Returns:
        ``(K, C)`` values by map name: base colour RGBA, the normal in the target's frame,
        roughness and metalness, emissive RGB, and the occlusion the material itself carries.
    """
    count = int(face.shape[0])
    out = {name: torch.zeros(count, CHANNELS[name], device=device) for name in wanted}
    part_of = surface.face_part[face]
    uv = _corners(surface.corner_uvs, face, weights)
    colors = _interpolate(surface.colors, surface.faces, face, weights) if surface.colors is not None else None
    shading = None
    if "normal" in wanted:
        if surface.corner_tangents is not None:
            shading, source_tangent, source_bitangent = _frame(surface, face, weights)
        else:
            shading = torch.nn.functional.normalize(_interpolate(surface.normals, surface.faces, face, weights), dim=1)
    for part in torch.unique(part_of).tolist():
        rows = torch.nonzero(part_of == part).squeeze(1)
        material = surface.materials[part]
        part_uv = uv[rows]

        def read(slot: str):
            found = _texture(surface, part, slot, device)
            if found is None:
                return None
            image, wrap_s, wrap_t = found
            return raster.sample_texture(image, part_uv, wrap_s, wrap_t)

        if "base_color" in wanted:
            factor = torch.tensor(material.base_color, dtype=torch.float32, device=device)
            value = factor.expand(rows.shape[0], 4).clone()
            texel = read("base_color")
            if texel is not None:
                value[:, :3] *= srgb_to_linear(texel[:, :3])
                if texel.shape[1] > 3:
                    value[:, 3] *= texel[:, 3].clamp(0.0, 1.0)
            if colors is not None:
                value *= colors[rows]
            out["base_color"][rows] = value
        if "metallic_roughness" in wanted:
            value = torch.tensor([material.roughness, material.metallic], dtype=torch.float32,
                                 device=device).expand(rows.shape[0], 2).clone()
            texel = read("metallic_roughness")
            if texel is not None and texel.shape[1] >= 3:
                value[:, 0] *= texel[:, 1]
                value[:, 1] *= texel[:, 2]
            out["metallic_roughness"][rows] = value.clamp(0.0, 1.0)
        if "emissive" in wanted:
            strength = float(material.extras.get("emissive_strength", 1.0) or 1.0)
            value = torch.tensor(material.emissive, dtype=torch.float32, device=device).expand(rows.shape[0], 3).clone()
            value = value * strength
            texel = read("emissive")
            if texel is not None:
                value = value * srgb_to_linear(texel[:, :3])
            out["emissive"][rows] = value.clamp_min(0.0)
        if "occlusion" in wanted:
            value = torch.ones(rows.shape[0], 1, device=device)
            texel = read("occlusion")
            if texel is not None:
                value = 1.0 + float(material.occlusion_strength) * (texel[:, :1] - 1.0)
            out["occlusion"][rows] = value.clamp(0.0, 1.0)
        if "normal" in wanted:
            direction = shading[rows]
            texel = read("normal") if surface.corner_tangents is not None else None
            if texel is not None:
                local = texel[:, :3] * 2.0 - 1.0
                local[:, :2] *= float(material.normal_scale)
                direction = torch.nn.functional.normalize(
                    local[:, 0:1] * source_tangent[rows] + local[:, 1:2] * source_bitangent[rows]
                    + local[:, 2:3] * shading[rows], dim=1)
            normal, tangent, bitangent = (value[rows] for value in target_frame)
            out["normal"][rows] = torch.stack([(direction * tangent).sum(1), (direction * bitangent).sum(1),
                                              (direction * normal).sum(1)], dim=1)
    return out


def _hammersley(count: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """``count`` well spread points in the unit square."""
    index = torch.arange(count, dtype=torch.int64)
    bits = torch.zeros(count, dtype=torch.float64)
    scale = 0.5
    value = index.clone()
    for _ in range(max(1, int(count).bit_length())):
        bits += (value & 1).double() * scale
        value >>= 1
        scale *= 0.5
    first = (index.double() + 0.5) / count
    return first.float().to(device), bits.float().to(device)


def _occlusion(tree: TriangleBVH, points: torch.Tensor, normals: torch.Tensor, flat_normals: torch.Tensor,
               rays: int, reach: float, offset: float, generator: torch.Generator) -> torch.Tensor:
    """Share of a cosine-weighted hemisphere above each point left open within ``reach``."""
    device = points.device
    count = int(points.shape[0])
    open_share = torch.ones(count, device=device)
    if count == 0 or rays <= 0:
        return open_share
    first, second = _hammersley(rays, device)
    radius = first.sqrt()
    lift = (1.0 - first).clamp_min(0.0).sqrt()
    side = torch.cross(normals, torch.where(normals[:, :1].abs() < 0.9, torch.tensor([1.0, 0.0, 0.0], device=device),
                                            torch.tensor([0.0, 1.0, 0.0], device=device)).expand_as(normals), dim=1)
    side = torch.nn.functional.normalize(side, dim=1)
    up = torch.cross(normals, side, dim=1)
    facing = torch.where((flat_normals * normals).sum(1, keepdim=True) < 0, -flat_normals, flat_normals)
    batch = max(1, AO_BATCH_RAYS // rays)
    for start in range(0, count, batch):
        stop = min(start + batch, count)
        turn = torch.rand(stop - start, generator=generator).to(device)
        angle = 2.0 * math.pi * (second[None, :] + turn[:, None])
        local_x = radius[None, :] * torch.cos(angle)
        local_y = radius[None, :] * torch.sin(angle)
        direction = (local_x[:, :, None] * side[start:stop, None, :] + local_y[:, :, None] * up[start:stop, None, :]
                     + lift[None, :, None] * normals[start:stop, None, :])
        plane = facing[start:stop, None, :]
        below = (direction * plane).sum(-1, keepdim=True)
        # Rays dipping under the triangle's own plane are mirrored back above it.
        direction = torch.where(below < 0, direction - 2.0 * below * plane, direction)
        origin = (points[start:stop] + facing[start:stop] * offset)[:, None, :].expand_as(direction)
        hit = tree.any_hit(origin.reshape(-1, 3), direction.reshape(-1, 3), t_min=offset, t_max=reach)
        open_share[start:stop] = 1.0 - hit.reshape(stop - start, rays).float().mean(dim=1)
    return open_share


# ---------------------------------------------------------------------------------------------
# Baking


def _check(settings: BakeSettings) -> list[str]:
    """The maps to bake, after the settings are held to their ranges."""
    wanted = [name for name in MAPS if name in settings.maps]
    if not wanted:
        raise ValueError("no maps are switched on; turn on at least one of " + ", ".join(MAPS))
    if not 16 <= int(settings.resolution) <= MAX_RESOLUTION:
        raise ValueError(f"resolution must be 16 to {MAX_RESOLUTION} texels")
    if not 1 <= int(settings.samples) <= MAX_SAMPLES:
        raise ValueError(f"samples must be 1 to {MAX_SAMPLES}")
    if not 0 <= int(settings.ao_rays) <= MAX_AO_RAYS:
        raise ValueError(f"ao_rays must be 0 to {MAX_AO_RAYS}")
    if not settings.cage > 0:
        raise ValueError("cage must be above 0; 0.02 searches 2% of the model's size each way")
    channels = sum(CHANNELS[name] for name in wanted) + 1
    # Sums, one finished image at a time, the nearest-texel map and the flood that finds it.
    needed = int(settings.resolution) ** 2 * (channels * 4 + 16 + 8 + 28)
    if needed > MAX_ACCUMULATOR_BYTES:
        raise ValueError(f"baking {len(wanted)} maps at {settings.resolution} texels needs about "
                         f"{needed / (1 << 30):.1f} GB; lower the resolution or switch maps off")
    return wanted


def _part_overlap(surface: _Surface, offset: torch.Tensor, part_names: list[str]) -> None:
    """Refuse parts whose triangles claim the same texels."""
    if len(part_names) < 2:
        return
    corners = (surface.corner_uvs - offset.to(surface.corner_uvs.device)) * OVERLAP_GRID
    low = torch.full((OVERLAP_GRID * OVERLAP_GRID,), len(part_names), dtype=torch.int64, device=corners.device)
    high = torch.full_like(low, -1)
    for piece in raster.coverage_chunks(corners, OVERLAP_GRID, OVERLAP_GRID):
        inside = piece.barycentric.amin(dim=1) > 0.05
        label = surface.face_part[piece.face[inside]]
        low.scatter_reduce_(0, piece.pixel[inside], label, reduce="amin")
        high.scatter_reduce_(0, piece.pixel[inside], label, reduce="amax")
    covered = high >= 0
    shared = covered & (low != high)
    if int(covered.sum()) and float(shared.sum()) > OVERLAP_SHARE * float(covered.sum()):
        first = part_names[int(low[shared][0])]
        second = part_names[int(high[shared][0])]
        raise ValueError(f"parts '{first}' and '{second}' use the same area of the texture, so one bake would "
                         "paint both; run Mesh UV Unwrap on these parts together first")


def _hits(source: _Surface, points: torch.Tensor, normals: torch.Tensor, reach: float):
    """Nearest source triangle along each normal, both ways, then by distance for the misses."""
    in_t, in_face, in_weights = source.tree.ray_cast(points, -normals, t_max=reach)
    # The outward ray only needs to beat the inward hit, so it stops where that one landed.
    out_t, out_face, out_weights = source.tree.ray_cast(points, normals, t_max=in_t.clamp_max(reach))
    outward = out_t < in_t
    face = torch.where(outward, out_face, in_face)
    weights = torch.where(outward[:, None], out_weights, in_weights)
    missed = face < 0
    if bool(missed.any()):
        _, near_face, near_weights, _ = source.tree.closest_point(points[missed], max_distance=reach)
        face = face.clone()
        weights = weights.clone()
        face[missed] = near_face
        weights[missed] = near_weights.to(weights.dtype)
    return face, weights.float()


def _finish_images(sums: dict[str, torch.Tensor], count: torch.Tensor, side: int, padding: int,
                   occlusion: torch.Tensor | None) -> tuple[dict[str, torch.Tensor], dict]:
    """Averaged, padded and encoded images from per-texel sums."""
    filled = count > 0
    weight = count.clamp_min(1).float()[:, None]
    nearest = raster.nearest_filled(filled.reshape(side, side), int(padding))
    images: dict[str, torch.Tensor] = {}
    extra: dict = {}
    for name in list(sums):
        mean = sums.pop(name) / weight
        if name == "normal":
            mean = torch.nn.functional.normalize(mean, dim=1)
            mean = torch.where(filled[:, None], mean, torch.tensor([0.0, 0.0, 1.0], device=mean.device))
            image = mean * 0.5 + 0.5
            background = torch.tensor([0.5, 0.5, 1.0], device=mean.device)
        elif name == "base_color":
            colour = linear_to_srgb(mean[:, :3])
            alpha = mean[:, 3:4].clamp(0.0, 1.0)
            translucent = bool(filled.any()) and float(alpha[filled].min()) < 0.999
            image = torch.cat([colour, alpha], dim=1) if translucent else colour
            extra["alpha"] = translucent
            background = image[filled].mean(dim=0) if bool(filled.any()) else torch.ones(image.shape[1], device=image.device)
        elif name == "metallic_roughness":
            ao = occlusion if occlusion is not None else torch.ones(mean.shape[0], device=mean.device)
            image = torch.stack([ao, mean[:, 0], mean[:, 1]], dim=1).clamp(0.0, 1.0)
            background = image[filled].mean(dim=0) if bool(filled.any()) else torch.tensor([1.0, 1.0, 0.0], device=image.device)
        elif name == "emissive":
            peak = float(mean[filled].max()) if bool(filled.any()) else 0.0
            extra["emissive_strength"] = max(peak, 1.0)
            extra["emissive_present"] = peak > 1e-6
            image = linear_to_srgb(mean / max(peak, 1.0))
            background = torch.zeros(3, device=image.device)
        else:
            image = mean.clamp(0.0, 1.0).expand(-1, 3)
            background = torch.ones(3, device=image.device)
        image = torch.where(filled[:, None], image, background[None, :].to(image.dtype))
        image = raster.fill_gaps(image.reshape(side, side, -1), nearest, background)
        images[name] = image.float().cpu().contiguous()
    return images, extra


def _material(base: Material, images: dict[str, torch.Tensor], extra: dict, sources: list[Material],
              name: str) -> Material:
    """A copy of the target's material whose baked slots point at the new textures."""
    material = Material(
        name=name, base_color=base.base_color, base_color_texture=base.base_color_texture,
        metallic=base.metallic, roughness=base.roughness,
        metallic_roughness_texture=base.metallic_roughness_texture, normal_texture=base.normal_texture,
        normal_scale=base.normal_scale, occlusion_texture=base.occlusion_texture,
        occlusion_strength=base.occlusion_strength, emissive=base.emissive, emissive_texture=base.emissive_texture,
        alpha_mode=base.alpha_mode, alpha_cutoff=base.alpha_cutoff, double_sided=base.double_sided,
        unlit=base.unlit,
        extras={key: value for key, value in base.extras.items()
                if key not in ("texture_transform", "mtoon", "vrm0_material", "emissive_strength")},
    )
    if "base_color" in images:
        material.base_color = (1.0, 1.0, 1.0, 1.0)
        material.base_color_texture = Texture(images["base_color"], name=f"{name}_base_color")
        if extra.get("alpha"):
            modes = [source.alpha_mode for source in sources]
            material.alpha_mode = "BLEND" if "BLEND" in modes else "MASK"
            cutoffs = [source.alpha_cutoff for source in sources if source.alpha_mode == "MASK"]
            material.alpha_cutoff = cutoffs[0] if cutoffs else 0.5
        else:
            material.alpha_mode = "OPAQUE"
    if "metallic_roughness" in images:
        material.metallic = 1.0
        material.roughness = 1.0
        material.metallic_roughness_texture = Texture(images["metallic_roughness"], name=f"{name}_metallic_roughness")
    if "normal" in images:
        material.normal_texture = Texture(images["normal"], name=f"{name}_normal")
        material.normal_scale = 1.0
    if "occlusion" in images:
        material.occlusion_texture = Texture(images["occlusion"], name=f"{name}_occlusion")
        material.occlusion_strength = 1.0
    if "emissive" in images:
        if extra.get("emissive_present"):
            material.emissive = (1.0, 1.0, 1.0)
            material.emissive_texture = Texture(images["emissive"], name=f"{name}_emissive")
            if extra.get("emissive_strength", 1.0) > 1.0:
                material.extras["emissive_strength"] = float(extra["emissive_strength"])
        else:
            material.emissive = (0.0, 0.0, 0.0)
            material.emissive_texture = None
    return material


def bake(parts: list[Part], source: Model3D | None, settings: BakeSettings,
         progress: Callable[[float], None] | None = None) -> Baked:
    """Bake a source's look into the texture layout the parts share.

    Args:
        parts: Target parts with texture coordinates in one shared layout.
        source: The model the look comes from, or None to read each part's layout and material
            from before its unwrap.
        settings: The `BakeSettings`.
        progress: Called with the share of the samples done, 0 to 1.

    Returns:
        The `Baked` images and the material carrying them.

    Raises:
        ValueError: A part has no triangles or no texture coordinates, parts share texture
            space, there is nothing to bake from, or a setting is out of range.
    """
    wanted = _check(settings)
    for part in parts:
        if part.triangle_count == 0:
            raise ValueError(f"part '{part.name}' has no triangles to bake onto")
        if part.uvs is None:
            raise ValueError(f"part '{part.name}' has no texture coordinates to bake into; run Mesh UV Unwrap "
                             "on it first")
    if not parts:
        raise ValueError("no parts to bake onto")
    device = mesh_device.compute_device()

    def work(run_device: torch.device) -> Baked:
        return _bake(parts, source, settings, wanted, run_device, progress)

    return mesh_device.with_cpu_fallback(work, device, label="texture bake")


def _bake(parts: list[Part], source: Model3D | None, settings: BakeSettings, wanted: list[str],
          device: torch.device, progress: Callable[[float], None] | None) -> Baked:
    """`bake` on one device."""
    clock = time.perf_counter()
    side = int(settings.resolution)
    steps = int(settings.samples)
    grid = side * steps
    target, offset = _target_surface(parts, device)
    _part_overlap(target, offset, [part.name for part in parts])
    low, high = target.vertices.min(dim=0).values, target.vertices.max(dim=0).values
    diagonal = float((high - low).norm()) or 1.0
    reach = float(settings.cage) * diagonal
    surface_offset = SURFACE_OFFSET * diagonal
    sampled = set(wanted)
    need_tangents = "normal" in sampled
    rays = int(settings.ao_rays) if "occlusion" in sampled else 0
    if source is not None:
        surface = _source_surface(source, device, need_tangents)
    else:
        surface = _previous_surface(parts, device, need_tangents, rays > 0)
    prepared = time.perf_counter()
    rays_per_sample = max(1, math.ceil(rays / (steps * steps))) if rays else 0
    generator = torch.Generator().manual_seed(int(settings.seed))
    texel_count = side * side
    sums = {name: torch.zeros(texel_count, CHANNELS[name], device=device) for name in wanted}
    count = torch.zeros(texel_count, dtype=torch.int32, device=device)
    corners = (target.corner_uvs - offset.to(device)) * grid
    total = hits = 0
    first, second = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    expected = max(1.0, float((first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]).abs().sum()) / 2)
    for piece in raster.coverage_chunks(corners, grid, grid):
        if progress is not None:
            progress(min(1.0, total / expected))
        face, weights = piece.face, piece.barycentric.float()
        row, column = piece.pixel // grid, piece.pixel % grid
        texel = (row // steps) * side + column // steps
        frame = _frame(target, face, weights)
        if source is not None:
            points = _interpolate(target.vertices, target.faces, face, weights)
            source_face, source_weights = _hits(surface, points, frame[0], reach)
            found = source_face >= 0
        else:
            source_face, source_weights = face, weights
            found = torch.ones_like(face, dtype=torch.bool)
        total += int(face.shape[0])
        hits += int(found.sum())
        if not bool(found.any()):
            continue
        rows = torch.nonzero(found).squeeze(1)
        values = _sample_maps(surface, source_face[rows], source_weights[rows], sampled - {"occlusion"},
                              tuple(value[rows] for value in frame), device)
        if "occlusion" in sampled:
            occlusion = _sample_maps(surface, source_face[rows], source_weights[rows], {"occlusion"}, None,
                                     device)["occlusion"][:, 0]
            if rays_per_sample:
                hit_points = _interpolate(surface.vertices, surface.faces, source_face[rows], source_weights[rows])
                hit_normals = torch.nn.functional.normalize(
                    _interpolate(surface.normals, surface.faces, source_face[rows], source_weights[rows]), dim=1)
                occlusion = occlusion * _occlusion(surface.tree, hit_points, hit_normals,
                                                   surface.face_normals[source_face[rows]], rays_per_sample,
                                                   float(settings.ao_distance) * diagonal, surface_offset, generator)
            values["occlusion"] = occlusion[:, None]
        where = texel[rows]
        for name, value in values.items():
            sums[name].index_add_(0, where, value.float())
        count.index_add_(0, where, torch.ones_like(where, dtype=torch.int32))
    sampled_at = time.perf_counter()
    occlusion_mean = None
    if "occlusion" in sums and "metallic_roughness" in sums:
        occlusion_mean = (sums["occlusion"][:, 0] / count.clamp_min(1).float())
    images, extra = _finish_images(sums, count, side, int(settings.padding), occlusion_mean)
    name = parts[0].material.name or "baked"
    material = _material(parts[0].material, images, extra, surface.materials, f"{name}_baked")
    filled = int((count > 0).sum())
    stats = {
        "samples": total,
        "hit_share": hits / total if total else 0.0,
        "missed": total - hits,
        "texels": filled,
        "coverage": filled / texel_count,
        "prepare_seconds": prepared - clock,
        "sample_seconds": sampled_at - prepared,
        "seconds": time.perf_counter() - clock,
        "maps": wanted,
        "from_source": source is not None,
    }
    logger.info("baked %s at %d texels from %s: %d samples, %.1f%% found the surface", ", ".join(wanted), side,
                "the source" if source is not None else "the earlier layout", total,
                100.0 * stats["hit_share"])
    return Baked(images, material, stats)

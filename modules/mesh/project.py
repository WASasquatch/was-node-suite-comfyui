"""Pictures of a model projected back into its texture layout and blended into one of its maps."""
# Each texel's surface point is found from the texture coordinates, then read from every picture
# that sees it. A picture counts for more where it faces the surface straight on, away from its
# borders, the model's outline and its occluding edges; the result is blended over the map by how
# well it is seen. Normal pictures hold camera-space normals as ``n * 0.5 + 0.5``, +X right, +Y up
# and +Z towards the camera; they are turned into world space by their camera, blended, and
# written in the part's MikkTSpace frame, +Y up.

from __future__ import annotations

__all__ = ["ENCODINGS", "MAPS", "MAX_TEXTURE", "SLOTS", "ProjectSettings", "Projected", "project"]

import math
import time
from dataclasses import dataclass, field, replace

import torch

from ..model3d.scene import Material, Model3D, Part, Texture
from . import raster, tangents
from .bake import linear_to_srgb, srgb_to_linear
from .views import Views, depth_buffer, screen, vertex_normals

#: Widest texture written, in texels.
MAX_TEXTURE = 8192

#: Texture size given to a part that has coordinates but no map to write into, when none is asked
#: for and it has no colour map to match.
NEW_TEXTURE = 2048

#: Seen weight at which a texel takes the projected value fully: a single picture facing it at
#: about 60 degrees with the default falloff.
FULL_WEIGHT = 0.25

#: Texels worked on at once.
TEXEL_CHUNK = 1 << 20

#: Depth step between neighbouring pixels, as a share of the model's size, at which one surface
#: counts as passing in front of another.
OCCLUSION_JUMP = 0.01

#: The same step in pixel footprints, the floor for a surface seen steeply at a coarse resolution.
OCCLUSION_PIXELS = 6.0

#: Maps the pictures can be projected into.
MAPS = ("base colour", "roughness", "metallic", "normal", "occlusion")

#: Map -> the material slot that holds it.
SLOTS = {
    "base colour": "base_color",
    "roughness": "metallic_roughness",
    "metallic": "metallic_roughness",
    "normal": "normal",
    "occlusion": "occlusion",
}

#: How greyscale pictures hold their values: raised to 1/2.2 for showing, or as they are.
ENCODINGS = ("gamma 2.2", "linear")

#: Power a ``gamma 2.2`` picture is raised to for its value.
GAMMA = 2.2

#: Smallest share of a written tangent-space normal that points out of the surface.
MIN_FACING = 0.05

#: Channel of the metallic-roughness texture each greyscale map is written into.
CHANNEL = {"roughness": 1, "metallic": 2, "occlusion": 0}


@dataclass
class ProjectSettings:
    """How pictures are projected.

    Attributes:
        texture_size: Width and height of the written maps; 0 keeps each map's own size.
        falloff: Power on how squarely a picture faces a surface; higher takes each texel from its
            most head-on picture.
        edge_fade: Share of a picture's size at its borders over which it fades out.
        outline_fade: Share of a picture's size over which it fades in from the model's outline and
            from both sides of every occluding edge.
        strength: How fully the projected value replaces the old one where it is well seen.
        depth_tolerance: How far behind the nearest surface a point still counts as seen, as a
            share of the model's size.
        padding: Texels each chart's values are grown into the empty space around it.
        map: One of `MAPS`.
        encoding: One of `ENCODINGS`, read for the greyscale maps.
    """

    texture_size: int = 0
    falloff: float = 4.0
    edge_fade: float = 0.05
    outline_fade: float = 0.01
    strength: float = 1.0
    depth_tolerance: float = 0.004
    padding: int = 8
    map: str = "base colour"
    encoding: str = "gamma 2.2"


@dataclass
class Projected:
    """What a projection made.

    Attributes:
        textures: The new maps as stored, ``(H, W, C)`` float32, in the order the parts first used
            them.
        coverage: ``(H, W)`` float32 share each texel of the first map took from the pictures.
        stats: Counts and timings.
    """

    textures: list[torch.Tensor]
    coverage: torch.Tensor | None
    stats: dict = field(default_factory=dict)


def _resize(image: torch.Tensor, size: int) -> torch.Tensor:
    """An ``(H, W, C)`` image resampled to ``size`` square."""
    if image.shape[0] == size and image.shape[1] == size:
        return image.float().clone()
    planes = image.float().permute(2, 0, 1)[None]
    resized = torch.nn.functional.interpolate(planes, size=(size, size), mode="bilinear", align_corners=False,
                                              antialias=size < image.shape[0])
    return resized[0].permute(1, 2, 0).clamp(0.0, 1.0)


def _uv_matrix(transform: dict) -> torch.Tensor:
    """The ``(3, 3)`` matrix of a glTF texture transform: offset, then rotation, then scale."""
    def pair(name: str, default: float) -> list[float]:
        held = transform.get(name, [default, default])
        try:
            values = [float(held[0]), float(held[1])]
        except (TypeError, ValueError, IndexError):
            return [default, default]
        return values if all(math.isfinite(value) for value in values) else [default, default]

    offset, scale = pair("offset", 0.0), pair("scale", 1.0)
    try:
        rotation = float(transform.get("rotation", 0.0))
    except (TypeError, ValueError):
        rotation = 0.0
    rotation = rotation if math.isfinite(rotation) else 0.0
    cos, sin = math.cos(rotation), math.sin(rotation)
    translate = torch.tensor([[1.0, 0.0, offset[0]], [0.0, 1.0, offset[1]], [0.0, 0.0, 1.0]], dtype=torch.float64)
    rotate = torch.tensor([[cos, sin, 0.0], [-sin, cos, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
    return translate @ rotate @ torch.diag(torch.tensor([scale[0], scale[1], 1.0], dtype=torch.float64))


def _transform_of(material: Material, slot: str) -> dict | None:
    """The texture transform a material applies to one slot on top of the part's coordinates."""
    held = (material.extras.get("texture_transform") or {}).get(slot)
    return held if isinstance(held, dict) and held else None


def _held(texture: Texture | None, transform: dict | None, size: int) -> torch.Tensor | None:
    """An existing map resampled into the part's own texture space at ``size`` square, 3 channels or more."""
    if texture is None or texture.image.numel() == 0:
        return None
    if transform is None:
        image = _resize(texture.image, size)
    else:
        centres = (torch.arange(size, dtype=torch.float64) + 0.5) / size
        down, across = torch.meshgrid(centres, centres, indexing="ij")
        flat = torch.stack([across.reshape(-1), down.reshape(-1), torch.ones(size * size, dtype=torch.float64)], 1)
        moved = flat @ _uv_matrix(transform).T
        image = raster.sample_texture(texture.image.float(), moved[:, :2], texture.wrap_s, texture.wrap_t)
        image = image.reshape(size, size, -1).clamp(0.0, 1.0)
    if image.shape[2] < 3:
        image = torch.cat([image[..., :1].expand(size, size, 3), image[..., 1:]], dim=2)
    return image


def _inside_reach(mask: torch.Tensor, steps: int) -> torch.Tensor:
    """``(H, W)`` share from 0 at a covered region's outline to 1 at ``steps`` pixels inside it."""
    if steps <= 0:
        return mask.float()
    outside = (mask <= 0.5).float()[None, None]
    reach = torch.zeros_like(mask, dtype=torch.float32)
    for _ in range(steps):
        outside = torch.nn.functional.max_pool2d(outside, 3, stride=1, padding=1)
        reach += 1.0 - outside[0, 0]
    return (reach / steps).clamp(0.0, 1.0)


def _occlusion_edges(depth: torch.Tensor, camera, size: float) -> torch.Tensor:
    """``(H, W)`` true on both sides of every place one surface passes in front of another.

    Args:
        depth: The camera's depth buffer, infinite where nothing is drawn.
        camera: The camera it was drawn from.
        size: The model's bounding sphere radius.

    Returns:
        A mask of the pixels beside a depth step larger than :data:`OCCLUSION_JUMP` of the
        model's size or :data:`OCCLUSION_PIXELS` pixel footprints, whichever is larger.
    """
    finite = torch.isfinite(depth)
    filled = torch.where(finite, depth, torch.zeros_like(depth)).float()
    footprint = 2.0 * float(camera.extent) / max(int(camera.height), 1)
    per_pixel = filled * footprint if camera.projection != "orthographic" else torch.full_like(filled, footprint)
    jump = torch.clamp(OCCLUSION_PIXELS * per_pixel, min=OCCLUSION_JUMP * 2.0 * float(size))
    edges = torch.zeros_like(finite)
    height, width = filled.shape
    for down, across in ((0, 1), (1, 0)):
        near = (slice(0, height - down), slice(0, width - across))
        far = (slice(down, height), slice(across, width))
        step = finite[near] & finite[far] & ((filled[near] - filled[far]).abs() > jump[near])
        edges[near] |= step
        edges[far] |= step
    return edges


def _sample(image: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Bilinear samples of an ``(H, W, C)`` picture at ``(u, v)`` in ``[0, 1]``."""
    grid = torch.stack([u * 2 - 1, v * 2 - 1], dim=1).float()[None, None]
    planes = image.float().permute(2, 0, 1)[None]
    found = torch.nn.functional.grid_sample(planes, grid, mode="bilinear", padding_mode="border",
                                            align_corners=False)
    return found[0, :, 0].T


def _smooth(edge: torch.Tensor) -> torch.Tensor:
    """Smoothstep of a share already clamped to ``[0, 1]``."""
    return edge * edge * (3 - 2 * edge)


def _picture_values(image: torch.Tensor, camera, settings: ProjectSettings) -> torch.Tensor:
    """What a picture says about the map: linear colour, a greyscale value or a world-space normal.

    Args:
        image: ``(H, W, C)`` picture codes.
        camera: The camera the picture was rendered from.
        settings: How to project.

    Returns:
        ``(H, W, 3)`` for colour and normals, ``(H, W, 1)`` for a greyscale map.
    """
    codes = image.float().clamp(0.0, 1.0)
    if codes.shape[2] < 3:
        codes = codes[..., :1].expand(-1, -1, 3)
    codes = codes[..., :3]
    if settings.map == "base colour":
        return srgb_to_linear(codes)
    if settings.map == "normal":
        local = torch.nn.functional.normalize(codes * 2.0 - 1.0, dim=2)
        return local @ camera.view[:3, :3].float()
    grey = codes.mean(dim=2, keepdim=True)
    return grey ** GAMMA if settings.encoding == "gamma 2.2" else grey


class _Picture:
    """One camera's picture, mask and the depth and outline reach needed to read it."""

    def __init__(self, model: Model3D, views: Views, index: int, image: torch.Tensor, mask: torch.Tensor | None,
                 settings: ProjectSettings):
        self.camera = views.cameras[index]
        self.values = _picture_values(image, self.camera, settings)
        self.mask = mask
        self.depth = depth_buffer(model, self.camera)
        covered = torch.isfinite(self.depth)
        steps = int(math.ceil(settings.outline_fade * max(self.camera.width, self.camera.height)))
        # The fade runs in from the outline and from both sides of every occluding edge.
        solid = covered & ~_occlusion_edges(self.depth, self.camera, views.size)
        self.outline = _inside_reach(solid, steps)
        self.tolerance = settings.depth_tolerance * 2 * views.size

    def read(self, points: torch.Tensor, normals: torch.Tensor, settings: ProjectSettings):
        """``(weight, values)`` this picture gives each point."""
        camera = self.camera
        x, y, _, _, distance = screen(camera, points)
        u, v = x / camera.width, y / camera.height
        weight = torch.zeros(points.shape[0], dtype=torch.float32)
        values = torch.zeros(points.shape[0], self.values.shape[2], dtype=torch.float32)
        inside = (u >= 0) & (u < 1) & (v >= 0) & (v < 1) & (distance > 0)
        if not bool(inside.any()):
            return weight, values
        rows = torch.nonzero(inside).squeeze(1)
        u, v, distance = u[rows], v[rows], distance[rows]
        across, down = u * camera.width - 0.5, v * camera.height - 0.5
        left, top = across.floor(), down.floor()
        share_x, share_y = (across - left).float(), (down - top).float()
        seen = torch.zeros(rows.shape[0], dtype=torch.float32)
        for step_y, weight_y in ((0, 1 - share_y), (1, share_y)):
            for step_x, weight_x in ((0, 1 - share_x), (1, share_x)):
                column = (left + step_x).long().clamp(0, camera.width - 1)
                row = (top + step_y).long().clamp(0, camera.height - 1)
                clear = (distance <= self.depth[row, column] + self.tolerance).float()
                seen += clear * weight_x * weight_y * self.outline[row, column]
        if camera.projection == "orthographic":
            towards = (-camera.forward).float().expand(rows.shape[0], 3)
        else:
            towards = torch.nn.functional.normalize((camera.position[None] - points[rows].double()).float(), dim=1)
        facing = (normals[rows] * towards).sum(dim=1).clamp_min(0.0) ** float(settings.falloff)
        border = torch.minimum(torch.minimum(u, 1 - u), torch.minimum(v, 1 - v)).float()
        edge = _smooth((border / max(settings.edge_fade, 1e-6)).clamp(0.0, 1.0)) if settings.edge_fade > 0 \
            else torch.ones_like(border)
        found = facing * edge * seen
        if self.mask is not None:
            found = found * _sample(self.mask[..., None], u, v)[:, 0].clamp(0.0, 1.0)
        keep = found > 0
        if bool(keep.any()):
            weight[rows[keep]] = found[keep]
            values[rows[keep]] = _sample(self.values, u[keep], v[keep])
        return weight, values


def _factors(material: Material, slot: str) -> tuple:
    """The material factors that scale one slot's texture."""
    if slot == "base_color":
        return tuple(float(value) for value in material.base_color[:3])
    if slot == "metallic_roughness":
        return float(material.roughness), float(material.metallic)
    if slot == "normal":
        return (float(material.normal_scale),)
    return (float(material.occlusion_strength),)


def _groups(model: Model3D, chosen: list[int], slot: str) -> list[tuple[Texture | None, list[int]]]:
    """The chosen parts grouped by the map they draw from and the factors scaling it; a part with
    no map stands alone."""
    groups: list[tuple[Texture | None, list[int]]] = []
    by_key: dict[tuple, int] = {}
    for index in chosen:
        material = model.parts[index].material
        texture = getattr(material, f"{slot}_texture")
        if texture is None:
            groups.append((None, [index]))
            continue
        key = (id(texture), _factors(material, slot), repr(_transform_of(material, slot)))
        if key in by_key:
            groups[by_key[key]][1].append(index)
        else:
            by_key[key] = len(groups)
            groups.append((texture, [index]))
    return groups


def _with_frame(part: Part) -> Part:
    """A part carrying unit normals and tangents, MikkTSpace ones worked out where it has none."""
    if part.tangents is not None and tuple(part.tangents.shape) == (part.vertex_count, 4):
        return part if part.normals is not None else replace(part, normals=vertex_normals(part))
    return tangents.with_tangents(replace(part, normals=vertex_normals(part), tangents=None))


def _base(settings: ProjectSettings, material: Material, held: torch.Tensor | None, size: int) -> torch.Tensor:
    """``(size * size, C)`` values the map holds now, in the space the pictures are blended in."""
    if settings.map == "base colour":
        codes = held[..., :3] if held is not None else torch.ones(size, size, 3)
        factor = torch.tensor(material.base_color[:3], dtype=torch.float32).clamp(0.0, 1.0)
        return (srgb_to_linear(codes.clamp(0.0, 1.0)) * factor).reshape(-1, 3)
    if settings.map == "normal":
        if held is None:
            return torch.tensor([0.0, 0.0, 1.0]).repeat(size * size, 1)
        local = held[..., :3].reshape(-1, 3) * 2.0 - 1.0
        local[:, :2] *= float(material.normal_scale)
        return torch.nn.functional.normalize(local, dim=1)
    if settings.map == "occlusion":
        if held is None:
            return torch.ones(size * size, 1)
        strength = float(material.occlusion_strength)
        return (1.0 + strength * (held[..., 0].reshape(-1, 1) - 1.0)).clamp(0.0, 1.0)
    factor = float(material.roughness if settings.map == "roughness" else material.metallic)
    if held is None:
        return torch.full((size * size, 1), min(max(factor, 0.0), 1.0))
    return (held[..., CHANNEL[settings.map]].reshape(-1, 1) * factor).clamp(0.0, 1.0)


def _compose(settings: ProjectSettings, material: Material, held: torch.Tensor | None, mixed: torch.Tensor,
             size: int, shared: bool) -> torch.Tensor:
    """The map image as stored, from the blended values.

    Args:
        settings: How to project.
        material: The material the map belongs to.
        held: The existing map at ``size``, or None.
        mixed: ``(size * size, C)`` blended values.
        size: Texture side.
        shared: The occlusion map is also the metallic-roughness map.

    Returns:
        ``(size, size, C)`` float32: sRGB codes for colour, linear values otherwise, any alpha kept.
    """
    alpha = held[..., 3:4] if held is not None and held.shape[2] > 3 else None
    if settings.map == "base colour":
        image = linear_to_srgb(mixed).reshape(size, size, 3)
    elif settings.map == "normal":
        image = (mixed * 0.5 + 0.5).clamp(0.0, 1.0).reshape(size, size, 3)
    elif settings.map == "occlusion" and not shared:
        image = mixed.reshape(size, size, 1).expand(size, size, 3).clone()
    else:
        if held is not None:
            image = held[..., :3].clone()
            if settings.map != "occlusion":
                image[..., 1] *= float(material.roughness)
                image[..., 2] *= float(material.metallic)
        else:
            image = torch.ones(size, size, 3)
            image[..., 1] = min(max(float(material.roughness), 0.0), 1.0)
            image[..., 2] = min(max(float(material.metallic), 0.0), 1.0)
        image[..., CHANNEL[settings.map]] = mixed.reshape(size, size)
        image = image.clamp(0.0, 1.0)
    return torch.cat([image, alpha], dim=2) if alpha is not None else image


def _written(material: Material, settings: ProjectSettings, texture: Texture) -> Material:
    """A copy of a material whose projected slot draws the new texture and reads it as written."""
    copy = replace(material, extras=dict(material.extras))
    slot = SLOTS[settings.map]
    dropped = {slot}
    if settings.map == "base colour":
        copy.base_color_texture = texture
        copy.base_color = (1.0, 1.0, 1.0, float(material.base_color[3]))
    elif settings.map == "normal":
        copy.normal_texture = texture
        copy.normal_scale = 1.0
    elif settings.map == "occlusion":
        old = material.occlusion_texture
        copy.occlusion_texture = texture
        copy.occlusion_strength = 1.0
        if old is not None and material.metallic_roughness_texture is old:
            copy.metallic_roughness_texture = texture
            dropped.add("metallic_roughness")
    else:
        old = material.metallic_roughness_texture
        copy.metallic_roughness_texture = texture
        copy.metallic = 1.0
        copy.roughness = 1.0
        if old is not None and material.occlusion_texture is old:
            copy.occlusion_texture = texture
            dropped.add("occlusion")
    transforms = copy.extras.get("texture_transform")
    if isinstance(transforms, dict) and dropped & set(transforms):
        kept = {name: value for name, value in transforms.items() if name not in dropped}
        if kept:
            copy.extras["texture_transform"] = kept
        else:
            copy.extras.pop("texture_transform", None)
    return copy


def project(model: Model3D, chosen: list[int], views: Views, images: torch.Tensor,
            masks: torch.Tensor | None, settings: ProjectSettings, progress=None) -> Projected:
    """Project pictures of a model back into one map of its chosen parts, in place.

    Args:
        model: The model the pictures were rendered from; its chosen parts get new materials, and
            tangents where a normal map is written and they had none.
        chosen: Indices of the parts to project onto; each needs texture coordinates.
        views: The cameras the pictures were rendered from.
        images: ``(V, H, W, C)`` pictures in camera order, any size of the cameras' shape.
        masks: ``(V, H, W)`` share of each picture to use, or None for all of it.
        settings: How to project.
        progress: Called with the share done, or None.

    Returns:
        The `Projected` maps and figures.

    Raises:
        ValueError: The picture or mask count differs from the camera count, a size is out of
            range, or the map or encoding is unknown.
    """
    started = time.perf_counter()
    if settings.map not in MAPS:
        raise ValueError(f"map must be one of {', '.join(MAPS)}, not {settings.map!r}")
    if settings.encoding not in ENCODINGS:
        raise ValueError(f"encoding must be one of {', '.join(ENCODINGS)}, not {settings.encoding!r}")
    if images.shape[0] != len(views):
        raise ValueError(f"{images.shape[0]} picture(s) arrived for {len(views)} camera(s); wire the pictures "
                         "rendered from these views, one per camera, in the same order")
    if masks is not None and masks.shape[0] not in (1, len(views)):
        raise ValueError(f"{masks.shape[0]} mask(s) arrived for {len(views)} camera(s); wire one mask per "
                         "picture, or one for all of them")
    if not 0 <= int(settings.texture_size) <= MAX_TEXTURE:
        raise ValueError(f"texture_size must be between 0 and {MAX_TEXTURE}")
    pictures = []
    for index in range(len(views)):
        mask = None if masks is None else masks[min(index, masks.shape[0] - 1)].float()
        pictures.append(_Picture(model, views, index, images[index], mask, settings))
    normal_map = settings.map == "normal"
    if normal_map:
        for index in chosen:
            model.parts[index] = _with_frame(model.parts[index])
    slot = SLOTS[settings.map]
    groups = _groups(model, chosen, slot)
    total_texels = 0
    seen_texels = 0
    by_view = torch.zeros(len(views), dtype=torch.float64)
    textures, first_coverage = [], None
    for number, (texture, members) in enumerate(groups):
        material = model.parts[members[0]].material
        colour = material.base_color_texture
        size = int(settings.texture_size) or (texture.size[0] if texture is not None
                                              else colour.size[0] if colour is not None else NEW_TEXTURE)
        held = _held(texture, _transform_of(material, slot), size)
        base = _base(settings, material, held, size)
        channels = base.shape[1]
        sums = torch.zeros(size * size, channels, dtype=torch.float32)
        weights = torch.zeros(size * size, dtype=torch.float32)
        covered = torch.zeros(size * size, dtype=torch.bool)
        for index in members:
            part = model.parts[index]
            normals = vertex_normals(part)
            frame = part.tangents.float() if normal_map else None
            corners, owner = raster.texel_triangles(part.uvs.float()[part.faces.long()], size, size)
            for piece in raster.coverage_chunks(corners, size, size, chunk_pixels=TEXEL_CHUNK):
                corner = part.faces.long()[owner[piece.face]]
                share = piece.barycentric.float()
                points = (part.vertices.float()[corner] * share[..., None]).sum(dim=1)
                facing = torch.nn.functional.normalize((normals[corner] * share[..., None]).sum(dim=1), dim=1)
                covered[piece.pixel] = True
                found = torch.zeros(points.shape[0], channels, dtype=torch.float32)
                seen = torch.zeros(points.shape[0], dtype=torch.float32)
                for view, picture in enumerate(pictures):
                    weight, values = picture.read(points, facing, settings)
                    found += values * weight[:, None]
                    seen += weight
                    by_view[view] += float(weight.sum())
                if frame is not None:
                    found = _tangent_space(torch.nn.functional.normalize(found, dim=1), facing, frame[corner],
                                           share) * seen[:, None]
                sums.index_add_(0, piece.pixel, found)
                weights.index_add_(0, piece.pixel, seen)
            if progress is not None:
                progress((number + (members.index(index) + 1) / len(members)) / len(groups))
        blend = ((weights / FULL_WEIGHT).clamp(0.0, 1.0) * float(settings.strength))[:, None]
        if normal_map:
            projected = torch.nn.functional.normalize(sums, dim=1)
            projected[:, 2] = projected[:, 2].clamp_min(MIN_FACING)
            projected = torch.nn.functional.normalize(projected, dim=1)
            mixed = torch.nn.functional.normalize(base * (1 - blend) + projected * blend, dim=1)
        else:
            projected = sums / weights.clamp_min(1e-12)[:, None]
            mixed = base * (1 - blend) + projected * blend
        mixed = torch.where((weights > 0)[:, None], mixed, base)
        packed = material.metallic_roughness_texture
        shared = (settings.map == "occlusion" and held is not None and packed is not None
                  and (packed is texture or packed.image is texture.image))
        image = _compose(settings, material, held, mixed, size, shared)
        if normal_map and held is not None and float(material.normal_scale) == 1.0:
            # Texels no picture sees keep the existing normal map's own codes.
            unseen = (weights <= 0).reshape(size, size, 1)
            image[..., :3] = torch.where(unseen, held[..., :3], image[..., :3])
        nearest = raster.nearest_filled(covered.reshape(size, size), int(settings.padding))
        image = raster.fill_gaps(image, nearest)
        name = (texture.name if texture is not None else "") or ("projected" if slot == "base_color"
                                                                   else f"projected_{slot}")
        written = Texture(image.contiguous(), name=name,
                          wrap_s=texture.wrap_s if texture is not None else "repeat",
                          wrap_t=texture.wrap_t if texture is not None else "repeat", mime="image/png")
        for index in members:
            part = model.parts[index]
            part.material = _written(part.material, settings, written)
        textures.append(image)
        texels = int(covered.sum())
        total_texels += texels
        seen_texels += int((weights > 0).sum())
        if first_coverage is None:
            first_coverage = (weights / FULL_WEIGHT).clamp(0.0, 1.0).reshape(size, size)
    view_share = (by_view / by_view.sum().clamp_min(1e-12)).tolist()
    return Projected(textures, first_coverage, {
        "map": settings.map, "maps": len(textures), "texels": total_texels,
        "seen share": seen_texels / max(total_texels, 1),
        "view shares": {views.cameras[index].name: view_share[index] for index in range(len(views))},
        "seconds": time.perf_counter() - started,
    })


def _tangent_space(world: torch.Tensor, normal: torch.Tensor, corners: torch.Tensor,
                   share: torch.Tensor) -> torch.Tensor:
    """World-space unit normals in the surface's tangent frame, +Y along the bitangent.

    Args:
        world: ``(K, 3)`` unit normals.
        normal: ``(K, 3)`` unit surface normals.
        corners: ``(K, 3, 4)`` tangents and handedness at each point's triangle corners.
        share: ``(K, 3)`` corner weights of each point.

    Returns:
        ``(K, 3)`` tangent-space normals.
    """
    raw = (corners[..., :3] * share[..., None]).sum(dim=1)
    tangent = torch.nn.functional.normalize(raw - (raw * normal).sum(dim=1, keepdim=True) * normal, dim=1)
    sign = torch.where((corners[..., 3] * share).sum(dim=1) < 0, -1.0, 1.0)
    bitangent = torch.linalg.cross(normal, tangent, dim=1) * sign[:, None]
    return torch.stack([(world * tangent).sum(dim=1), (world * bitangent).sum(dim=1),
                        (world * normal).sum(dim=1)], dim=1)

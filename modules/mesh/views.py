"""Cameras set round a model, and renders of it from them: colour, depth, normals and coverage."""
# A camera looks down its own -Z with +Y up. A picture's (u, v) in [0, 1] runs across a row from the
# left and down from the top, the way ComfyUI images and glTF textures are laid out.

from __future__ import annotations

__all__ = [
    "MAX_RESOLUTION",
    "PROJECTIONS",
    "VIEW_SETS",
    "Camera",
    "Rendered",
    "Views",
    "depth_buffer",
    "render",
    "rig",
    "screen",
    "vertex_normals",
]

import math
from dataclasses import dataclass

import torch

from ..model3d.scene import Model3D, Part
from . import raster, topology

#: Camera sets round a model standing on Y and facing +Z.
VIEW_SETS = ("six sides", "around", "around, top and bottom", "front and back", "front")

#: How a camera maps the scene onto its picture.
PROJECTIONS = ("orthographic", "perspective")

#: Widest picture rendered, in pixels.
MAX_RESOLUTION = 4096

#: Most cameras in an ``around`` set.
MAX_AROUND = 36

#: Light shading mixed into a lit render: the share every surface keeps, and the light's direction in
#: camera space, from above and to the left of the camera.
AMBIENT = 0.35
LIGHT = (-0.35, 0.55, 0.75)

#: The picture behind the model in a normal render: a surface facing the camera straight on.
FLAT_NORMAL = (0.5, 0.5, 1.0)


@dataclass(frozen=True)
class Camera:
    """One camera round a model.

    Attributes:
        name: Where the camera looks from, such as ``"front"`` or ``"around 45"``.
        view: ``(4, 4)`` float64 world-to-camera matrix.
        projection: One of `PROJECTIONS`.
        extent: Half the picture's height: in world units for an orthographic camera, the tangent of
            half the field of view for a perspective one.
        width: Picture width in pixels.
        height: Picture height in pixels.
        near: Distance from the camera to the nearest point of the model's bounding sphere.
        far: Distance to its farthest point.
    """

    name: str
    view: torch.Tensor
    projection: str
    extent: float
    width: int
    height: int
    near: float
    far: float

    @property
    def position(self) -> torch.Tensor:
        """``(3,)`` float64 world position of the camera."""
        rotation, translation = self.view[:3, :3], self.view[:3, 3]
        return -(rotation.T @ translation)

    @property
    def forward(self) -> torch.Tensor:
        """``(3,)`` float64 unit world direction the camera looks along."""
        return -self.view[2, :3]


@dataclass(frozen=True)
class Views:
    """The cameras a set of pictures was rendered from, in picture order.

    Attributes:
        cameras: One `Camera` per picture.
        size: The model's bounding sphere radius when the cameras were set.
    """

    cameras: tuple[Camera, ...]
    size: float

    def __len__(self) -> int:
        """Number of cameras."""
        return len(self.cameras)


@dataclass
class Rendered:
    """Renders of a model, one per camera.

    Attributes:
        colour: ``(V, H, W, 3)`` float32 sRGB codes.
        depth: ``(V, H, W)`` float32, 1 at the near side of the model's bounding sphere and 0 at
            the far side and behind the model.
        normals: ``(V, H, W, 3)`` float32 camera-space normals as ``n * 0.5 + 0.5``, +X right, +Y
            up, +Z towards the camera.
        mask: ``(V, H, W)`` float32 share of each pixel the model covers.
    """

    colour: torch.Tensor
    depth: torch.Tensor
    normals: torch.Tensor
    mask: torch.Tensor


def _look_at(eye: torch.Tensor, target: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """The ``(4, 4)`` world-to-camera matrix of a camera at ``eye`` looking at ``target``."""
    forward = torch.nn.functional.normalize(target - eye, dim=0)
    right = torch.nn.functional.normalize(torch.linalg.cross(forward, up), dim=0)
    above = torch.linalg.cross(right, forward)
    view = torch.eye(4, dtype=torch.float64)
    view[0, :3], view[1, :3], view[2, :3] = right, above, -forward
    view[:3, 3] = -(view[:3, :3] @ eye)
    return view


def _directions(view_set: str, around: int, elevation: float) -> list[tuple[str, tuple, tuple]]:
    """``(name, direction from the model to the camera, up)`` for each camera of a set."""
    sides = {
        "front": ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
        "back": ((0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
        "left side": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        "right side": ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        "top": ((0.0, 1.0, 0.0), (0.0, 0.0, -1.0)),
        "bottom": ((0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    }
    if view_set == "six sides":
        names = ("front", "left side", "back", "right side", "top", "bottom")
    elif view_set == "front and back":
        names = ("front", "back")
    elif view_set == "front":
        names = ("front",)
    else:
        names = ()
    chosen = [(name, *sides[name]) for name in names]
    if view_set in ("around", "around, top and bottom"):
        tilt = math.radians(elevation)
        for index in range(around):
            turn = 2 * math.pi * index / around
            direction = (math.sin(turn) * math.cos(tilt), math.sin(tilt), math.cos(turn) * math.cos(tilt))
            chosen.append((f"around {round(math.degrees(turn))}", direction, (0.0, 1.0, 0.0)))
        if view_set == "around, top and bottom":
            chosen += [("top", *sides["top"]), ("bottom", *sides["bottom"])]
    return chosen


def rig(model: Model3D, view_set: str = "six sides", around: int = 8, elevation: float = 15.0,
        projection: str = "orthographic", fov: float = 35.0, resolution: int = 1024,
        margin: float = 0.05) -> Views:
    """Cameras round a model, each framing all of it; an orthographic one fits the model's outline from its side.

    Args:
        model: The model, standing on Y and facing +Z.
        view_set: One of `VIEW_SETS`.
        around: Cameras in an ``around`` set.
        elevation: Degrees an ``around`` set looks down from.
        projection: One of `PROJECTIONS`.
        fov: Vertical field of view of a perspective camera, in degrees.
        resolution: Picture width and height in pixels.
        margin: Room left round the model, as a share of its size.

    Returns:
        The `Views`.

    Raises:
        ValueError: The model has no vertices, or a setting is out of range.
    """
    if view_set not in VIEW_SETS:
        raise ValueError(f"views must be one of {', '.join(VIEW_SETS)}, not {view_set!r}")
    if projection not in PROJECTIONS:
        raise ValueError(f"projection must be orthographic or perspective, not {projection!r}")
    if not 16 <= int(resolution) <= MAX_RESOLUTION:
        raise ValueError(f"resolution must be between 16 and {MAX_RESOLUTION} pixels")
    if not 1 <= int(around) <= MAX_AROUND:
        raise ValueError(f"around must be between 1 and {MAX_AROUND} cameras")
    live = [part for part in model.parts if part.vertex_count]
    if not live:
        raise ValueError("the model has no vertices to render")
    low = torch.stack([part.vertices.double().min(dim=0).values for part in live]).min(dim=0).values
    high = torch.stack([part.vertices.double().max(dim=0).values for part in live]).max(dim=0).values
    centre = (low + high) / 2
    radius = max(float((high - low).norm()) / 2, 1e-6)
    framed = radius * (1.0 + max(float(margin), 0.0))
    if projection == "orthographic":
        distance, extent = 3.0 * radius, framed
    else:
        half = math.radians(min(max(float(fov), 1.0), 150.0)) / 2
        distance, extent = framed / math.sin(half), math.tan(half)
    points = torch.cat([part.vertices.double() for part in live]) if projection == "orthographic" else None
    cameras = []
    for name, direction, up in _directions(view_set, int(around), float(elevation)):
        towards = torch.tensor(direction, dtype=torch.float64)
        upward = torch.tensor(up, dtype=torch.float64)
        target, half = centre, extent
        if points is not None:
            target, half = _framed(points, centre, towards, upward, float(margin))
        eye = target + towards * distance
        cameras.append(Camera(name, _look_at(eye, target, upward), projection,
                              float(half), int(resolution), int(resolution),
                              max(distance - radius, 1e-6), distance + radius))
    return Views(tuple(cameras), radius)


def _framed(points: torch.Tensor, centre: torch.Tensor, towards: torch.Tensor, up: torch.Tensor,
            margin: float) -> tuple[torch.Tensor, float]:
    """Where an orthographic camera looking back along ``towards`` aims, and its half height, to fit the model.

    Args:
        points: ``(N, 3)`` float64 world positions of every vertex.
        centre: ``(3,)`` centre of the model's bounding box.
        towards: ``(3,)`` unit direction from the model to the camera.
        up: ``(3,)`` the camera's up direction.
        margin: Room left round the model's outline, as a share of the framed size.

    Returns:
        ``(target, extent)``: the world point the camera looks at and half the picture's height.
    """
    view = _look_at(centre + towards, centre, up)
    local = (points - centre) @ view[:2, :3].T
    low, high = local.min(dim=0).values, local.max(dim=0).values
    middle = (low + high) / 2
    target = centre + view[0, :3] * middle[0] + view[1, :3] * middle[1]
    half = float((high - low).max()) / 2
    return target, max(half * (1.0 + max(margin, 0.0)), 1e-6)


def screen(camera: Camera, points: torch.Tensor, scale: int = 1) -> tuple[torch.Tensor, ...]:
    """Where world points land on a camera's picture.

    Args:
        camera: The camera.
        points: ``(N, 3)`` world positions.
        scale: Picture size multiplier, for rendering finer than the camera's own size.

    Returns:
        ``(x, y, key, weight, depth)``: pixel coordinates, a closeness that is linear across the
        picture (larger is nearer), the perspective weight of each point, and its distance along
        the view.
    """
    local = points.double() @ camera.view[:3, :3].T + camera.view[:3, 3]
    depth = -local[:, 2]
    width, height = camera.width * scale, camera.height * scale
    aspect = width / height
    if camera.projection == "orthographic":
        across, down = local[:, 0] / (camera.extent * aspect), local[:, 1] / camera.extent
        key, weight = -depth, torch.ones_like(depth)
    else:
        safe = depth.clamp_min(1e-9)
        across = local[:, 0] / (safe * camera.extent * aspect)
        down = local[:, 1] / (safe * camera.extent)
        key = weight = 1.0 / safe
    return (across + 1) * 0.5 * width, (1 - down) * 0.5 * height, key, weight, depth


def vertex_normals(part: Part) -> torch.Tensor:
    """A part's unit vertex normals, its own or worked out from its faces across texture seams."""
    if part.normals is not None:
        return torch.nn.functional.normalize(part.normals.float(), dim=1)
    welded, faces, remap = topology.weld(part.vertices, part.faces, tolerance=0)
    return topology.vertex_normals(welded, faces)[remap].float()


def _corners(camera: Camera, part: Part, scale: int):
    """A part's triangles on a camera's picture, and the per-vertex values rendering needs."""
    x, y, key, weight, depth = screen(camera, part.vertices, scale)
    faces = part.faces.long()
    return torch.stack([x, y], dim=1).float()[faces], faces, key, weight, depth


def depth_buffer(model: Model3D, camera: Camera, scale: int = 1) -> torch.Tensor:
    """``(H, W)`` float64 distance along the view to the nearest surface, infinite where none is."""
    width, height = camera.width * scale, camera.height * scale
    nearest = torch.full((width * height,), -math.inf, dtype=torch.float64)
    for part in model.parts:
        if not part.triangle_count:
            continue
        corners, faces, key, _, _ = _corners(camera, part, scale)
        for piece in raster.coverage_chunks(corners, width, height):
            closeness = (key[faces[piece.face]] * piece.barycentric.double()).sum(dim=1)
            nearest.scatter_reduce_(0, piece.pixel, closeness, reduce="amax")
    seen = torch.isfinite(nearest)
    if camera.projection == "orthographic":
        distance = -nearest
    else:
        distance = 1.0 / nearest.clamp_min(1e-30)
    return torch.where(seen, distance, torch.full_like(distance, math.inf)).reshape(height, width)


def _colour(part: Part, faces: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """sRGB codes of a part's base colour at points on its faces."""
    material = part.material
    factor = torch.tensor(material.base_color[:3], dtype=torch.float32)
    texture = material.base_color_texture
    if texture is not None and part.uvs is not None:
        uvs = (part.uvs.float()[faces] * weights[..., None]).sum(dim=1)
        return raster.sample_texture(texture.image.float(), uvs, texture.wrap_s, texture.wrap_t)[:, :3] * factor
    if part.colors is not None:
        return (part.colors.float()[faces][..., :3] * weights[..., None]).sum(dim=1) * factor
    return factor.expand(faces.shape[0], 3)


def _render_one(model: Model3D, camera: Camera, scale: int, lighting: float, background: float) -> tuple:
    """Colour, depth, normals and coverage from one camera at ``scale`` times its size."""
    width, height = camera.width * scale, camera.height * scale
    nearest = torch.full((width * height,), -math.inf, dtype=torch.float64)
    passes = []
    for part in model.parts:
        if not part.triangle_count:
            continue
        corners, faces, key, weight, depth = _corners(camera, part, scale)
        passes.append((part, corners, faces, key, weight, depth))
        for piece in raster.coverage_chunks(corners, width, height):
            closeness = (key[faces[piece.face]] * piece.barycentric.double()).sum(dim=1)
            nearest.scatter_reduce_(0, piece.pixel, closeness, reduce="amax")
    colour = torch.full((width * height, 3), float(background), dtype=torch.float32)
    normals = torch.tensor(FLAT_NORMAL, dtype=torch.float32).repeat(width * height, 1)
    distance = torch.full((width * height,), math.inf, dtype=torch.float64)
    covered = torch.zeros(width * height, dtype=torch.float32)
    turn = camera.view[:3, :3].float()
    light = torch.nn.functional.normalize(torch.tensor(LIGHT, dtype=torch.float32), dim=0)
    for part, corners, faces, key, weight, depth in passes:
        normal = vertex_normals(part)
        for piece in raster.coverage_chunks(corners, width, height):
            corner = faces[piece.face]
            closeness = (key[corner] * piece.barycentric.double()).sum(dim=1)
            win = closeness >= nearest[piece.pixel] - 1e-9 * nearest[piece.pixel].abs().clamp_min(1.0)
            if not bool(win.any()):
                continue
            corner, pixel = corner[win], piece.pixel[win]
            share = piece.barycentric[win].double() * weight[corner]
            share = (share / share.sum(dim=1, keepdim=True).clamp_min(1e-30)).float()
            local = torch.nn.functional.normalize((normal[corner] * share[..., None]).sum(dim=1), dim=1) @ turn.T
            shade = _colour(part, corner, share)
            if lighting > 0:
                lit = AMBIENT + (1 - AMBIENT) * (local @ light).clamp_min(0.0)
                shade = shade * (1 - lighting + lighting * lit[:, None])
            colour[pixel] = shade.clamp(0.0, 1.0)
            normals[pixel] = local * 0.5 + 0.5
            distance[pixel] = (depth[corner] * share.double()).sum(dim=1)
            covered[pixel] = 1.0
    span = max(camera.far - camera.near, 1e-12)
    near_one = ((camera.far - distance) / span).clamp(0.0, 1.0)
    near_one = torch.where(torch.isfinite(distance), near_one, torch.zeros_like(near_one)).float()
    return (colour.reshape(height, width, 3), near_one.reshape(height, width),
            normals.reshape(height, width, 3), covered.reshape(height, width))


def _shrink(image: torch.Tensor, scale: int) -> torch.Tensor:
    """An ``(H, W, C)`` or ``(H, W)`` image averaged down by ``scale``."""
    if scale == 1:
        return image
    flat = image.dim() == 2
    planes = (image[..., None] if flat else image).permute(2, 0, 1)[None]
    shrunk = torch.nn.functional.avg_pool2d(planes, scale)[0].permute(1, 2, 0)
    return shrunk[..., 0] if flat else shrunk


def render(model: Model3D, views: Views, lighting: float = 0.0, background: float = 1.0,
           antialias: bool = True, progress=None) -> Rendered:
    """Render a model from every camera of ``views``.

    Args:
        model: The model.
        views: The cameras, as `rig` sets them.
        lighting: 0 = the colour map as it is; 1 = lit by a soft light from the camera's upper left.
        background: Grey level behind the model, 0 to 1.
        antialias: Render at twice the size and average down, softening edges.
        progress: Called with the share done after each camera, or None.

    Returns:
        The `Rendered` pictures.
    """
    scale = 2 if antialias else 1
    colours, depths, normals, masks = [], [], [], []
    for index, camera in enumerate(views.cameras):
        colour, depth, normal, mask = _render_one(model, camera, scale, float(lighting), float(background))
        normal = _shrink(normal, scale)
        colours.append(_shrink(colour, scale))
        depths.append(_shrink(depth, scale))
        normals.append(torch.nn.functional.normalize(normal * 2 - 1, dim=2) * 0.5 + 0.5)
        masks.append(_shrink(mask, scale))
        if progress is not None:
            progress((index + 1) / len(views.cameras))
    return Rendered(torch.stack(colours), torch.stack(depths), torch.stack(normals), torch.stack(masks))

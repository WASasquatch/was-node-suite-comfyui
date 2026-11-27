"""Drawing a point cloud from a camera as colour, depth and uncovered holes.

Lifted points are round dots, the nearest winning; fitted splats are blended. Pictures are
float32 ``(H, W, C)``; depth is in scene units, ``inf`` where empty.
"""

from __future__ import annotations

__all__ = [
    "DEPTH_STYLES",
    "MAX_PATH_PIXELS",
    "MAX_RADIUS",
    "Frame",
    "Scene",
    "depth_pictures",
    "grow_holes",
    "render",
]

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from . import cameras
from .cloud import Cloud

#: Widest dot drawn, as a radius in pixels; nearer points leave gaps for a model to fill.
MAX_RADIUS = 24

#: Dot radius as a multiple of a point's spread, enough to close the gaps between neighbours.
COVER = 1.5

#: Most pixels one path's pictures may hold in all, counting every view.
MAX_PATH_PIXELS = 192 * 1024 * 1024

#: Pixel and dot pairs scattered per pass.
CHUNK = 1 << 24

#: Nothing drawn yet, in the depth buffer.
EMPTY = torch.iinfo(torch.int64).max

#: How depth is drawn as a picture: ``inverse`` spends its range near the camera as depth
#: estimators do, ``linear`` spreads it evenly.
DEPTH_STYLES = ("inverse", "linear")


@dataclass
class Frame:
    """One drawn view.

    Attributes:
        colour: ``(H, W, 3)`` colours, ``background`` where empty.
        depth: ``(H, W)`` distance ahead of the camera, ``inf`` where empty.
    """

    colour: torch.Tensor
    depth: torch.Tensor


def _offsets(radius: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """The pixel offsets of a round dot of ``radius`` pixels."""
    span = torch.arange(-radius, radius + 1, device=device)
    dy, dx = torch.meshgrid(span, span, indexing="ij")
    inside = (dx * dx + dy * dy) <= (radius + 0.5) ** 2
    return dy[inside], dx[inside]


def render(cloud: Cloud, camera_info: Mapping, width: int, height: int, point_size: float = 1.0,
           background: float = 0.0, device=None) -> Frame:
    """Draw a cloud from one camera.

    Args:
        cloud: The points.
        camera_info: The camera.
        width: Picture width in pixels.
        height: Picture height in pixels.
        point_size: Multiplier on every dot's size.
        background: Grey drawn where no point lands.
        device: Where the work runs; the cloud's own device when None.

    Returns:
        The drawn frame, on ``device``.
    """
    device = device or cloud.positions.device
    colour = torch.full((height, width, 3), float(background), device=device)
    depth = torch.full((height, width), math.inf, device=device)
    if not cloud.count:
        return Frame(colour, depth)
    eye, rows = cameras.view(camera_info)
    eye, rows = eye.float().to(device), rows.float().to(device)
    positions = cloud.positions.to(device)
    local = (positions - eye) @ rows.T
    z = local[:, 2]
    ahead = z > 1e-6
    index = torch.nonzero(ahead, as_tuple=False)[:, 0]
    if not index.numel():
        return Frame(colour, depth)
    z = z[index]
    focal = cameras.focal(camera_info, width, height)
    column = torch.floor(focal * local[index, 0] / z + width / 2.0)
    row = torch.floor(focal * local[index, 1] / z + height / 2.0)
    spread = cloud.spreads.to(device)[index] * (COVER * float(point_size))
    radius = torch.ceil(focal * spread / z - 0.5).clamp(0, MAX_RADIUS)
    inside = (column + radius >= 0) & (column - radius < width) & (row + radius >= 0) & (row - radius < height)
    index, z, column, row, radius = index[inside], z[inside], column[inside].long(), row[inside].long(), radius[inside].long()
    if not index.numel():
        return Frame(colour, depth)
    order = torch.arange(index.numel(), device=device, dtype=torch.int64)
    key = (z.contiguous().view(torch.int32).to(torch.int64) << 32) | order
    buffer = torch.full((height * width,), EMPTY, dtype=torch.int64, device=device)
    for size in torch.unique(radius).tolist():
        chosen = torch.nonzero(radius == size, as_tuple=False)[:, 0]
        dy, dx = _offsets(int(size), device)
        per = max(1, CHUNK // dy.numel())
        for start in range(0, chosen.numel(), per):
            part = chosen[start:start + per]
            rows_hit = row[part][:, None] + dy[None, :]
            cols_hit = column[part][:, None] + dx[None, :]
            valid = (rows_hit >= 0) & (rows_hit < height) & (cols_hit >= 0) & (cols_hit < width)
            flat = (rows_hit * width + cols_hit)[valid]
            keys = key[part][:, None].expand_as(rows_hit)[valid]
            buffer.scatter_reduce_(0, flat, keys, reduce="amin", include_self=True)
    hit = buffer != EMPTY
    winner = (buffer[hit] & 0xFFFFFFFF)
    colour.view(-1, 3)[hit] = cloud.colours.to(device)[index[winner]]
    depth.view(-1)[hit] = z[winner]
    return Frame(colour, depth)


def grow_holes(holes: torch.Tensor, pixels: int) -> torch.Tensor:
    """Holes ``(N, H, W)`` widened by ``pixels`` in every direction."""
    if pixels <= 0:
        return holes
    size = 2 * int(pixels) + 1
    return torch.nn.functional.max_pool2d(holes[:, None].float(), size, stride=1, padding=int(pixels))[:, 0]


def depth_pictures(depths: Sequence[torch.Tensor], style: str = "inverse") -> tuple[torch.Tensor, float, float]:
    """Depth maps as greys on one shared scale, white nearest and black where empty.

    Args:
        depths: ``(H, W)`` depth maps with ``inf`` where empty.
        style: One of :data:`DEPTH_STYLES`.

    Returns:
        ``((N, H, W) greys in [0, 1], nearest, farthest)``, the two distances being the scale's ends.
    """
    stack = torch.stack(list(depths))
    found = stack[torch.isfinite(stack)]
    if not found.numel():
        return torch.zeros_like(stack), 0.0, 0.0
    sample = found[:: max(1, found.numel() // 4_000_000)].float()
    near = float(torch.quantile(sample, 0.01))
    far = float(torch.quantile(sample, 0.99))
    far = max(far, near * 1.0001 + 1e-9)
    finite = torch.isfinite(stack)
    clamped = torch.where(finite, stack.clamp(near, far), torch.full_like(stack, far))
    if style == "linear":
        grey = (far - clamped) / (far - near)
    else:
        grey = (1.0 / clamped - 1.0 / far) / (1.0 / near - 1.0 / far)
    return torch.where(finite, grey.clamp(0, 1), torch.zeros_like(grey)), near, far


class Scene:
    """A SPLAT ready to draw from many cameras: a lifted cloud as dots, fitted splats as splats.

    Attributes:
        dots: Whether the SPLAT is drawn as round dots.
        count: Points or splats held in the still scene.
        moving: Each frame's moving points of a timed SPLAT; empty for any other.
    """

    def __init__(self, splat, device):
        from . import gaussians
        from .cloud import from_splat, moving_from_splat
        self.dots = gaussians.drawn_as_dots(splat)
        self.device = device
        self.moving = moving_from_splat(splat)
        if self.dots:
            self.cloud = from_splat(splat).to(device)
            self.count = self.cloud.count
        else:
            self.splats = gaussians.from_splat(splat).to(device)
            self.count = self.splats.count

    def frame_for(self, index: int, total: int) -> int | None:
        """The frame whose moving points view ``index`` of ``total`` shows, or None for a still scene."""
        if not self.moving:
            return None
        last = len(self.moving) - 1
        return min(last, max(0, round(index * last / max(1, total - 1))))

    def moving_cover(self, camera_info: Mapping, width: int, height: int, frame: int,
                     point_size: float = 1.0) -> torch.Tensor:
        """Where one frame's moving points land in a view, ``(H, W)`` booleans."""
        drawn = render(self.moving[frame], camera_info, width, height, point_size, 0.0, self.device)
        return torch.isfinite(drawn.depth)

    def draw(self, camera_info: Mapping, width: int, height: int, point_size: float = 1.0,
             background: float = 0.0, frame: int | None = None) -> Frame:
        """One view; a fitted pixel counts as empty when its splats cover less than half of it.

        Args:
            camera_info: The camera.
            width: Picture width in pixels.
            height: Picture height in pixels.
            point_size: Multiplier on every dot's or splat's size.
            background: Grey drawn where nothing lands.
            frame: The frame whose moving points are drawn over the still scene; None = none.
        """
        moving = self.moving[frame] if frame is not None and self.moving else None
        if self.dots:
            cloud = self.cloud if moving is None or not moving.count else self.cloud.joined(moving)
            return render(cloud, camera_info, width, height, point_size, background, self.device)
        from . import gaussians
        splats = self.splats
        if point_size != 1.0:
            splats = gaussians.Gaussians(splats.means, splats.scales * float(point_size), splats.rotations,
                                         splats.opacities, splats.sh)
        with torch.no_grad():
            drawn = gaussians.rasterize(splats, camera_info, width, height, background)
        covered = drawn.alpha >= 0.5
        depth = torch.where(covered, drawn.depth / drawn.alpha.clamp_min(1e-6), torch.full_like(drawn.depth, math.inf))
        colour = drawn.colour.clamp(0, 1)
        if moving is not None and moving.count:
            over = render(moving, camera_info, width, height, point_size, background, self.device)
            front = over.depth < depth
            colour = torch.where(front[..., None], over.colour, colour)
            depth = torch.where(front, over.depth, depth)
        return Frame(colour, depth)

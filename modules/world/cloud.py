"""Coloured point clouds: lifting depth, growing from new views, and core's SPLAT type.

Positions are float32 ``(N, 3)`` in the splat frame, colours in ``[0, 1]``. A timed SPLAT's
items are the still scene, then every frame's moving points.
"""

from __future__ import annotations

__all__ = [
    "MAX_POINTS",
    "MOTIONS",
    "Cloud",
    "Lifted",
    "from_splat",
    "lift",
    "moving_counts",
    "moving_from_splat",
    "pack",
    "scene_depth",
    "shown_cloud",
    "splat_items",
    "to_splat",
]

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch

from . import cameras

#: Most points one cloud may hold.
MAX_POINTS = 12_000_000

#: The zeroth spherical harmonic's constant, which turns a colour into a splat's base term.
SH_C0 = 0.28209479177387814

#: Least share of a frame two views must share for one to be scaled to the other.
MIN_OVERLAP = 0.01

#: How a lift treats what moves between pictures: kept with the rest, found and held per frame, or
#: every picture held whole as its own frame.
MOTIONS = ("still scene", "find what moves", "each frame apart")

#: Most that share grows to for far points, whose depth is least certain.
MOST_MOVED_SHARE = 0.6

#: Share of its depth a point sits in front of what another picture sees there to count as seen through,
#: at the scene's typical depth; it grows in step with the depth beyond that.
MOVED_SHARE = 0.12

#: Share of the way from a picture's least to its middle confidence below which a pixel is dropped.
UNSURE_SHARE = 0.08

#: Most other pictures each picture is compared with to find what moves.
COMPARED = 8

#: Pixels a found moving region is widened by, as a share of the picture's longer side.
MOVING_GROW = 0.008

#: Share of its depth a moving region's edge may sit outside the depths of the region's inside.
MOVING_DEPTH_MARGIN = 0.06

#: The attribute a timed SPLAT carries its moving points per frame under.
FRAME_COUNTS = "was_frame_counts"


@dataclass
class Cloud:
    """A coloured point cloud.

    Attributes:
        positions: ``(N, 3)`` float32 splat-frame points.
        colours: ``(N, 3)`` float32 colours in ``[0, 1]``.
        spreads: ``(N,)`` float32 half pixel spacing at each point, in scene units.
    """

    positions: torch.Tensor
    colours: torch.Tensor
    spreads: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.positions.shape[0])

    @staticmethod
    def empty() -> "Cloud":
        return Cloud(torch.zeros(0, 3), torch.zeros(0, 3), torch.zeros(0))

    def joined(self, other: "Cloud") -> "Cloud":
        """This cloud with another's points added after its own."""
        device = self.positions.device
        return Cloud(torch.cat([self.positions, other.positions.to(device)]),
                     torch.cat([self.colours, other.colours.to(device)]),
                     torch.cat([self.spreads, other.spreads.to(device)]))

    def to(self, device) -> "Cloud":
        return Cloud(self.positions.to(device), self.colours.to(device), self.spreads.to(device))


def _splat_class():
    """Core's SPLAT class, or None when the running ComfyUI has none."""
    try:
        from comfy_api.latest import Types
    except Exception:
        return None
    return getattr(Types, "SPLAT", None)


def pack(parts: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
         frame_counts: Sequence[int] | None = None):
    """Sets of splats as the items of one core SPLAT, padded to the longest.

    Args:
        parts: Per item ``(positions (N, 3), scales (N, 3), rotations (N, 4), opacities (N,), sh (N, K, 3))``.
        frame_counts: Moving points per frame, for a timed SPLAT of a still item and a moving item.

    Raises:
        RuntimeError: The running ComfyUI has no SPLAT type.
    """
    kind = _splat_class()
    if kind is None:
        raise RuntimeError("this ComfyUI has no SPLAT type; update ComfyUI to build point clouds")
    longest = max(int(part[0].shape[0]) for part in parts)
    bands = max(int(part[4].shape[1]) for part in parts)

    def padded(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.float().cpu()
        if tensor.shape[0] == longest:
            return tensor
        return torch.cat([tensor, tensor.new_zeros(longest - tensor.shape[0], *tensor.shape[1:])])

    fields = []
    for column in range(5):
        rows = []
        for part in parts:
            value = part[column]
            if column == 3:
                value = value.reshape(-1, 1)
            if column == 4 and value.shape[1] < bands:
                value = torch.nn.functional.pad(value, (0, 0, 0, bands - value.shape[1]))
            rows.append(padded(value))
        fields.append(torch.stack(rows).contiguous())
    counts = None
    if len(parts) > 1:
        counts = torch.tensor([int(part[0].shape[0]) for part in parts], dtype=torch.long)
    splat = kind(*fields, counts)
    if frame_counts is not None:
        setattr(splat, FRAME_COUNTS, torch.tensor([int(c) for c in frame_counts], dtype=torch.long))
    return splat


def _round(cloud: Cloud) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """A cloud's points as round, opaque splats coloured by their base term."""
    count = cloud.count
    rotations = torch.zeros(count, 4)
    rotations[:, 0] = 1.0
    return (cloud.positions.float().cpu(), cloud.spreads.float().cpu()[:, None].expand(count, 3),
            rotations, torch.ones(count), ((cloud.colours.float().cpu() - 0.5) / SH_C0)[:, None, :])


def _joined(clouds: Sequence[Cloud]) -> Cloud:
    """Clouds as one, in order."""
    if not clouds:
        return Cloud.empty()
    device = clouds[0].positions.device
    return Cloud(torch.cat([c.positions.to(device) for c in clouds]),
                 torch.cat([c.colours.to(device) for c in clouds]),
                 torch.cat([c.spreads.to(device) for c in clouds]))


def to_splat(cloud: Cloud, moving: Sequence[Cloud] | None = None):
    """A cloud as core's SPLAT: round, opaque splats coloured by their base term.

    Args:
        cloud: The points, the still scene of a timed SPLAT.
        moving: Each frame's moving points, making the SPLAT timed.

    Raises:
        RuntimeError: The running ComfyUI has no SPLAT type.
    """
    if not moving:
        return pack([_round(cloud)])
    return pack([_round(cloud), _round(_joined(moving))], [m.count for m in moving])


def moving_counts(splat) -> list[int] | None:
    """Moving points per frame of a timed SPLAT, or None for any other."""
    held = getattr(splat, FRAME_COUNTS, None)
    positions = getattr(splat, "positions", None)
    counts = getattr(splat, "counts", None)
    if held is None or positions is None or counts is None or int(positions.shape[0]) != 2:
        return None
    found = [int(c) for c in held.tolist()]
    return found if found and sum(found) == int(counts[1]) else None


def splat_items(splat) -> list[tuple[int, int]]:
    """``(item, length)`` of each item that is the scene: every item, or a timed SPLAT's still one."""
    positions = splat.positions
    counts = getattr(splat, "counts", None)
    items = [(index, int(counts[index]) if counts is not None else int(positions.shape[1]))
             for index in range(int(positions.shape[0]))]
    if moving_counts(splat) is not None:
        items = items[:1]
    return [(index, length) for index, length in items if length > 0]


def _item(splat, index: int, start: int, end: int) -> Cloud:
    """Splats ``start`` to ``end`` of one SPLAT item as a cloud, the nearly clear ones left out."""
    place = splat.positions[index, start:end].float()
    sh = getattr(splat, "sh", None)
    colour = ((sh[index, start:end, 0, :].float() * SH_C0 + 0.5).clamp(0, 1) if sh is not None
              else torch.full((end - start, 3), 0.75))
    scales = getattr(splat, "scales", None)
    spread = scales[index, start:end].float().mean(-1) if scales is not None else torch.full((end - start,), 1e-3)
    opacity = getattr(splat, "opacities", None)
    if opacity is not None:
        solid = opacity[index, start:end, 0] > 0.05
        place, colour, spread = place[solid], colour[solid], spread[solid]
    return Cloud(place, colour, spread)


def from_splat(splat) -> Cloud:
    """A SPLAT's scene as one cloud: every item of a batch, or the still scene of a timed SPLAT.

    Raises:
        ValueError: ``splat`` carries no positions.
    """
    if getattr(splat, "positions", None) is None:
        raise ValueError("the point_cloud input is not a SPLAT; wire Point Cloud From Depth or a splat node")
    return _joined([_item(splat, index, 0, length) for index, length in splat_items(splat)])


def shown_cloud(splat) -> Cloud:
    """A SPLAT's scene, or the first frame's points of a timed SPLAT whose still scene is empty."""
    cloud = from_splat(splat)
    if cloud.count:
        return cloud
    moving = moving_from_splat(splat)
    return moving[0] if moving else cloud


def moving_from_splat(splat) -> list[Cloud]:
    """Each frame's moving points of a timed SPLAT; empty for any other."""
    counts = moving_counts(splat)
    if counts is None:
        return []
    clouds, start = [], 0
    for count in counts:
        clouds.append(_item(splat, 1, start, start + count))
        start += count
    return clouds


def scene_depth(cloud: Cloud, camera_info: Mapping) -> float:
    """Median distance ahead of a camera of the points in front of it, or 1.0 when none are."""
    if not cloud.count:
        return 1.0
    eye, rows = cameras.view(camera_info)
    sample = cloud.positions[:: max(1, cloud.count // 200_000)].double().cpu()
    ahead = (sample - eye) @ rows[2]
    ahead = ahead[ahead > 0]
    return float(ahead.median()) if ahead.numel() else 1.0


@dataclass
class Lifted:
    """What one lift made.

    Attributes:
        cloud: The points, with any cloud grown from first.
        cameras: The camera each picture was lifted from, as ``camera_info``.
        added: Points each picture added.
        scales: The factor each picture's depth was multiplied by to meet the cloud.
        matched: Whether each picture shared enough of the cloud to be scaled to it.
        moving: Each picture's moving points, kept out of ``cloud``; empty for a still scene.
        unsure: Pixels dropped for low depth confidence, per picture.
        seen_through: Pixels most other pictures see through, dropped, per picture.
    """

    cloud: Cloud
    cameras: list[dict]
    added: list[int] = field(default_factory=list)
    scales: list[float] = field(default_factory=list)
    matched: list[bool] = field(default_factory=list)
    moving: list[Cloud] = field(default_factory=list)
    unsure: list[int] = field(default_factory=list)
    seen_through: list[int] = field(default_factory=list)


def _edges(depth: torch.Tensor, keep: torch.Tensor, cut: float) -> torch.Tensor:
    """``keep`` without pixels whose depth jumps by more than ``cut`` of itself to a neighbour."""
    if cut <= 0:
        return keep
    z = torch.where(keep, depth, torch.zeros_like(depth))
    padded = torch.nn.functional.pad(z[None, None], (1, 1, 1, 1), mode="replicate")[0, 0]
    jump = torch.zeros_like(z)
    for dy, dx in ((0, 1), (2, 1), (1, 0), (1, 2)):
        jump = torch.maximum(jump, (z - padded[dy:dy + z.shape[0], dx:dx + z.shape[1]]).abs())
    return keep & (jump <= cut * z)


#: Most a point's spread may grow past its pixel spacing on a surface seen edge on.
MAX_STRETCH = 3.0


def _spreads(points: torch.Tensor, z: torch.Tensor, keep: torch.Tensor, focal: float, step: int) -> torch.Tensor:
    """Half the gap from each point to its farther kept neighbour, ``(H, W)``.

    Held between half the pixel spacing and :data:`MAX_STRETCH` times it.
    """
    spacing = z / focal * (0.5 * step)
    reach = torch.zeros_like(z)
    for shift in ((0, step), (step, 0)):
        dy, dx = shift
        a = points[: points.shape[0] - dy, : points.shape[1] - dx]
        b = points[dy:, dx:]
        both = keep[: keep.shape[0] - dy, : keep.shape[1] - dx] & keep[dy:, dx:]
        gap = torch.where(both, (a - b).norm(dim=-1) * 0.5, torch.zeros_like(a[..., 0]))
        reach[: reach.shape[0] - dy, : reach.shape[1] - dx] = torch.maximum(
            reach[: reach.shape[0] - dy, : reach.shape[1] - dx], gap)
        reach[dy:, dx:] = torch.maximum(reach[dy:, dx:], gap)
    return torch.minimum(torch.maximum(reach, spacing), spacing * MAX_STRETCH)


def _frame_geometry(geometry: Mapping, index: int, device) -> tuple[torch.Tensor, torch.Tensor, float | None,
                                                                         torch.Tensor | None, dict | None, int]:
    """One picture's depth, keep mask, focal length, camera-frame points and estimated camera.

    Returns:
        ``(depth (H, W), keep (H, W), focal in pixels or None, points (H, W, 3) or None,
        camera_info from the estimator or None, pixels dropped as unsure)``.
    """
    depth = geometry.get("depth")
    points = geometry.get("points")
    if depth is None and points is None:
        raise ValueError("the geometry carries no depth; wire MoGe Inference or DA3 Inference")
    if points is not None and points.dim() == 4:
        frame_points = points[min(index, points.shape[0] - 1)].float().to(device)
        z = frame_points[..., 2]
    else:
        frame_points = None
        z = depth[min(index, depth.shape[0] - 1)].float().to(device)
    height, width = z.shape
    keep = torch.isfinite(z) & (z > 0)
    if frame_points is not None:
        keep &= torch.isfinite(frame_points).all(-1)
    mask = geometry.get("mask")
    if mask is not None and mask.dim() == 3:
        keep &= mask[min(index, mask.shape[0] - 1)].to(device).bool()
    sky = geometry.get("sky")
    if sky is not None and sky.dim() == 3:
        keep &= sky[min(index, sky.shape[0] - 1)].to(device) < 0.5
    confidence = geometry.get("confidence")
    if confidence is not None and confidence.dim() == 3 and bool(keep.any()):
        sure = confidence[min(index, confidence.shape[0] - 1)].float().to(device)
        sample = sure[keep][:: max(1, int(keep.sum()) // 500_000)]
        low, middle = torch.quantile(sample, 0.01), torch.quantile(sample, 0.5)
        sure_enough = sure >= low + UNSURE_SHARE * (middle - low)
        unsure = int((keep & ~sure_enough).sum())
        keep &= sure_enough
    else:
        unsure = 0
    z = torch.where(keep, z, torch.zeros_like(z))
    focal = None
    estimated = None
    intrinsics = geometry.get("intrinsics")
    if intrinsics is not None:
        matrix = intrinsics[0, index] if intrinsics.dim() == 4 else intrinsics[min(index, intrinsics.shape[0] - 1)]
        matrix = matrix.double().cpu()
        focal = _pixel_focal(matrix, height)
    extrinsics = geometry.get("extrinsics")
    if extrinsics is not None and extrinsics.dim() == 4 and index < extrinsics.shape[1]:
        pose = extrinsics[0, index].double().cpu()
        if torch.isfinite(pose).all():
            rotation, shift = pose[:3, :3], pose[:3, 3]
            fov = cameras.fov_for_focal(focal or 0.7 * width, width, height)
            estimated = cameras.from_rows(-(rotation.T @ shift), rotation, fov)
    return z, keep, focal, frame_points, estimated, unsure


def _pixel_focal(matrix: torch.Tensor, height: int) -> float:
    """An intrinsics matrix's focal length in pixels of a depth map ``height`` pixels tall.

    Args:
        matrix: ``(3, 3)`` intrinsics, normalised or in pixels of any picture size.
        height: The depth map's height in pixels.
    """
    fy = float(matrix[1, 1])
    if fy < 8.0:
        return fy * height
    centre = float(matrix[1, 2])
    return fy * (height / (2.0 * centre)) if centre > 1.0 else fy


def _shared_focal(geometry: Mapping, frames: int) -> float | None:
    """The median pixel focal length of a multi-view geometry's pictures, or None for one picture."""
    intrinsics = geometry.get("intrinsics")
    if frames < 2 or geometry.get("extrinsics") is None or intrinsics is None or intrinsics.dim() != 4:
        return None
    source = geometry.get("points") if geometry.get("points") is not None else geometry.get("depth")
    height = int(source.shape[1])
    focals = torch.tensor([_pixel_focal(intrinsics[0, index].double().cpu(), height)
                           for index in range(min(frames, int(intrinsics.shape[1])))], dtype=torch.float64)
    focals = focals[torch.isfinite(focals) & (focals > 8.0)]
    return float(focals.median()) if focals.numel() else None


def _unproject(z: torch.Tensor, focal: float) -> torch.Tensor:
    """Camera-frame points of a depth map seen through a centred pinhole."""
    height, width = z.shape
    rows = (torch.arange(height, device=z.device, dtype=torch.float32) + 0.5 - height / 2.0) / focal
    cols = (torch.arange(width, device=z.device, dtype=torch.float32) + 0.5 - width / 2.0) / focal
    return torch.stack([cols[None, :].expand(height, width) * z, rows[:, None].expand(height, width) * z, z], -1)


def _colours(image: torch.Tensor | None, index: int, height: int, width: int, device) -> torch.Tensor:
    """One picture's colours at the depth map's size, or mid grey without a picture."""
    if image is None:
        return torch.full((height, width, 3), 0.5, device=device)
    frame = image[min(index, image.shape[0] - 1)].float().to(device)[..., :3]
    if frame.shape[0] != height or frame.shape[1] != width:
        frame = torch.nn.functional.interpolate(frame.permute(2, 0, 1)[None], size=(height, width),
                                                mode="bilinear", align_corners=False, antialias=True)[0].permute(1, 2, 0)
    return frame.clamp(0, 1)


def planned_points(geometry: Mapping, stride: int) -> int:
    """How many points lifting every picture of ``geometry`` could make at ``stride``."""
    source = geometry.get("points") if geometry.get("points") is not None else geometry.get("depth")
    if source is None:
        return 0
    frames, height, width = int(source.shape[0]), int(source.shape[1]), int(source.shape[2])
    step = max(1, int(stride))
    return frames * math.ceil(height / step) * math.ceil(width / step)


@dataclass
class _Picture:
    """One picture ready to lift.

    Attributes:
        z: ``(H, W)`` depth, 0 where dropped.
        keep: ``(H, W)`` pixels that may become points.
        focal: Focal length in pixels.
        points: ``(H, W, 3)`` camera-frame points.
        camera_info: The camera the picture was taken from.
        unsure: Pixels dropped for low depth confidence.
    """

    z: torch.Tensor
    keep: torch.Tensor
    focal: float
    points: torch.Tensor
    camera_info: dict
    unsure: int = 0


def _picture(geometry: Mapping, index: int, poses: Sequence[Mapping], shared: float | None, edge_cut: float,
             device) -> _Picture:
    """One picture's depth, kept pixels, points and camera, with depth edges cut."""
    z, keep, focal, points, estimated, unsure = _frame_geometry(geometry, index, device)
    height, width = z.shape
    if poses:
        camera_info = dict(poses[min(index, len(poses) - 1)])
        focal = cameras.focal(camera_info, width, height)
        points = _unproject(z, focal)
    else:
        if shared is not None:
            focal = shared
        if focal is None:
            focal = 0.7 * width
        fov = cameras.fov_for_focal(focal, width, height)
        camera_info = dict(estimated, fov=fov) if estimated is not None else cameras.source(fov)
        if points is None:
            points = _unproject(z, focal)
    return _Picture(z, _edges(z, keep, float(edge_cut)), float(focal), points, camera_info, unsure)


def _widened(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    """A boolean ``(H, W)`` mask grown by ``pixels`` in every direction."""
    if pixels <= 0:
        return mask
    size = 2 * pixels + 1
    return torch.nn.functional.max_pool2d(mask[None, None].float(), size, stride=1, padding=pixels)[0, 0] > 0.5


def _found_moving(seen_through: Sequence[torch.Tensor], pictures: Sequence[_Picture]) -> list[torch.Tensor]:
    """Seen-through pixels as moving regions: specks dropped, the rest widened to take in its edge."""
    masks = []
    for moved, picture in zip(seen_through, pictures):
        height, width = picture.z.shape
        solid = ~_widened(~moved, 1)
        masks.append(_widened(solid, max(2, round(MOVING_GROW * max(height, width)))) & picture.keep)
    return masks


def _seen_through(pictures: Sequence[_Picture], device) -> list[torch.Tensor]:
    """Per picture, the kept pixels whose points most of the other pictures see straight through."""
    views = []
    for picture in pictures:
        eye, rows = cameras.view(picture.camera_info)
        views.append((eye.float().to(device), rows.float().to(device)))
    count = len(pictures)
    steps = min(count, COMPARED + 1)
    masks = []
    for index, picture in enumerate(pictures):
        eye, rows = views[index]
        world = picture.points @ rows + eye
        through = torch.zeros_like(picture.z)
        compared = torch.zeros_like(picture.z)
        picks = {round(step * (count - 1) / max(1, steps - 1)) for step in range(steps)} - {index}
        for other_index in sorted(picks):
            other = pictures[other_index]
            other_eye, other_rows = views[other_index]
            typical = float(other.z[other.keep].median()) if bool(other.keep.any()) else 1.0
            local = (world - other_eye) @ other_rows.T
            z = local[..., 2]
            height, width = other.z.shape
            safe = z.clamp_min(1e-9)
            column = torch.floor(other.focal * local[..., 0] / safe + width / 2.0).long()
            row = torch.floor(other.focal * local[..., 1] / safe + height / 2.0).long()
            inside = picture.keep & (z > 1e-6) & (column >= 0) & (column < width) & (row >= 0) & (row < height)
            flat = row.clamp(0, height - 1) * width + column.clamp(0, width - 1)
            seen = other.z.reshape(-1)[flat]
            usable = inside & other.keep.reshape(-1)[flat]
            compared += usable
            margin = (MOVED_SHARE * (seen / max(typical, 1e-9)).clamp_min(1.0)).clamp_max(MOST_MOVED_SHARE)
            through += usable & (z < seen * (1.0 - margin))
        masks.append(picture.keep & (through >= 2) & (through * 2 > compared))
    return masks


def _trimmed(moved: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """A moving mask without the edge pixels whose depth leaves the range of the mask's inside nearby."""
    height, width = z.shape
    grow = max(2, round(MOVING_GROW * max(height, width)))
    core = ~_widened(~moved, grow + 2) & (z > 0)
    if not bool(core.any()):
        return moved
    size = 8 * grow + 1
    far = torch.nn.functional.max_pool2d(torch.where(core, z, torch.zeros_like(z))[None, None], size, stride=1,
                                         padding=size // 2)[0, 0]
    near = -torch.nn.functional.max_pool2d(torch.where(core, -z, torch.full_like(z, -math.inf))[None, None], size,
                                           stride=1, padding=size // 2)[0, 0]
    inside = (z >= near * (1.0 - MOVING_DEPTH_MARGIN)) & (z <= far * (1.0 + MOVING_DEPTH_MARGIN))
    return moved & ((far <= 0) | inside)


def _given_moving(mask: torch.Tensor, index: int, height: int, width: int, device) -> torch.Tensor:
    """One picture's wired moving mask at the depth map's size, widened to take in its edge."""
    frame = mask[min(index, mask.shape[0] - 1)].float().to(device)
    if frame.shape[0] != height or frame.shape[1] != width:
        frame = torch.nn.functional.interpolate(frame[None, None], size=(height, width), mode="bilinear",
                                                align_corners=False)[0, 0]
    return _widened(frame > 0.5, max(2, round(MOVING_GROW * max(height, width))))


def lift(geometry: Mapping, image: torch.Tensor | None, poses: Sequence[Mapping], grow: Cloud | None,
         stride: int, edge_cut: float, add: str, device, render_depth=None, progress=None,
         motion: str = MOTIONS[0], moving_mask: torch.Tensor | None = None) -> Lifted:
    """Turn each picture's depth into points and gather them into one cloud.

    Args:
        geometry: Core's MoGe or Depth Anything 3 geometry.
        image: Colours for the points; the geometry's own picture when None.
        poses: A camera per picture, the last repeated; empty to use the estimator's cameras.
        grow: A cloud to add to. Each picture's depth is first scaled to meet it, as is every
            picture after the first to the points lifted before it.
        stride: Every ``stride``-th pixel in each direction becomes a point. Holes filled in a
            cloud already holding points take every pixel.
        edge_cut: Pixels whose depth jumps by more than this share to a neighbour are dropped.
        add: ``holes`` = only points where the cloud so far showed nothing; ``everything`` = all.
        device: Where the work runs.
        render_depth: ``(cloud, camera_info, width, height) -> depth`` with ``inf`` where empty;
            needed when ``grow`` has points.
        progress: Called with the share of pictures done.
        motion: One of :data:`MOTIONS`; finding what moves takes two or more pictures.
        moving_mask: ``(N, H, W)`` masks of what moves in each picture, used in place of finding it.

    Returns:
        The lift.

    Raises:
        ValueError: The geometry carries no depth, or the cloud would pass :data:`MAX_POINTS`.
    """
    planned = planned_points(geometry, stride) + (grow.count if grow is not None else 0)
    if add != "holes" and planned > MAX_POINTS:
        raise ValueError(f"the cloud would hold {planned:,} points, more than {MAX_POINTS:,}; "
                         "raise stride or lift fewer pictures")
    colours_from = image if image is not None else geometry.get("image")
    source = geometry.get("points") if geometry.get("points") is not None else geometry.get("depth")
    frames = int(source.shape[0])
    cloud = grow.to(device) if grow is not None else Cloud.empty().to(device)
    result = Lifted(cloud, [])
    shared = _shared_focal(geometry, frames)
    placed = not poses and grow is None and geometry.get("extrinsics") is not None
    apart = frames > 1 and moving_mask is None and motion == MOTIONS[2]
    timed = frames > 1 and (moving_mask is not None or motion in MOTIONS[1:])
    ready: list | None = None
    found = None
    through = None
    if frames > 1 and not apart and (poses or geometry.get("extrinsics") is not None):
        ready = [_picture(geometry, index, poses, shared, edge_cut, device) for index in range(frames)]
        through = _seen_through(ready, device)
        if timed and moving_mask is None:
            found = _found_moving(through, ready)
    stride = max(1, int(stride))
    held = 0
    for index in range(frames):
        picture = ready[index] if ready is not None else _picture(geometry, index, poses, shared, edge_cut, device)
        z, keep, focal, points, camera_info = picture.z, picture.keep, picture.focal, picture.points, picture.camera_info
        height, width = z.shape
        moved = None
        if apart:
            moved = keep
            keep = torch.zeros_like(keep)
        elif timed:
            moved = (found[index] if found is not None
                     else _given_moving(moving_mask, index, height, width, device) & keep)
            keep = keep & ~moved
            moved = _trimmed(moved, z)
        dropped = 0
        if through is not None:
            dropped = int((keep & through[index]).sum())
            keep = keep & ~through[index]
        grid = torch.zeros_like(keep)
        grid[::stride, ::stride] = True
        scale, matched = 1.0, False
        step = stride
        if cloud.count and render_depth is not None:
            before = render_depth(cloud, camera_info, width, height).to(device)
            overlap = keep & torch.isfinite(before)
            if placed:
                matched = True
            elif int(overlap.sum()) >= max(64, MIN_OVERLAP * height * width):
                ratio = before[overlap] / z[overlap].clamp_min(1e-9)
                scale = float(ratio.median())
                matched = True
            if add == "holes":
                keep = keep & ~torch.isfinite(before)
                step = 1
        if step > 1:
            keep = keep & grid
        eye, rows = cameras.view(camera_info)
        eye, rows = eye.float().to(device), rows.float().to(device)
        colours = _colours(colours_from, index, height, width, device)
        world = (points[keep] * scale) @ rows + eye
        spreads = _spreads(points * scale, z * scale, keep, focal, step)[keep]
        added = int(world.shape[0])
        if moved is not None:
            moved = moved & grid
            result.moving.append(Cloud((points[moved] * scale) @ rows + eye, colours[moved],
                                       _spreads(points * scale, z * scale, moved, focal, stride)[moved]))
            held += result.moving[-1].count
        if cloud.count + added + held > MAX_POINTS:
            raise ValueError(f"picture {index + 1} would take the cloud past {MAX_POINTS:,} points; raise "
                             "stride or lift fewer pictures")
        cloud = cloud.joined(Cloud(world, colours[keep], spreads))
        result.cameras.append(camera_info)
        result.added.append(added)
        result.scales.append(scale)
        result.matched.append(matched)
        result.unsure.append(picture.unsure)
        result.seen_through.append(dropped)
        if ready is not None:
            ready[index] = None
        if progress is not None:
            progress((index + 1) / frames)
    result.cloud = cloud
    return result

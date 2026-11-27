"""Gaussian splats: core's SPLAT as tensors, view-dependent colour, and a tiled rasterizer.

Means are in the splat frame, scales are linear standard deviations, rotations are wxyz
quaternions and colour is spherical harmonics. Blending runs front to back with gradients.
"""

from __future__ import annotations

__all__ = [
    "FIT_TILE",
    "TILE",
    "Gaussians",
    "Raster",
    "drawn_as_dots",
    "from_cloud",
    "from_splat",
    "rasterize",
    "sh_colour",
    "to_splat",
]

import bisect
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch

from . import cameras
from .cloud import SH_C0, Cloud, pack, splat_items


#: Side of a square tile, in pixels.
TILE = 8

#: Tile side used while fitting, where splats stay small on screen.
FIT_TILE = 4

#: Gaussian and pixel pairs blended per pass.
PASS_ELEMENTS = 1 << 25

#: Shallowest tile list, as a share of the deepest, blended in the same pass.
PASS_SPREAD = 0.5

#: Most opacity one splat lends a pixel.
MOST_ALPHA = 0.99

#: Least opacity a splat must lend a pixel to count.
LEAST_ALPHA = 1.0 / 255.0

#: Variance, in pixels squared, added to every projected splat so none is thinner than a pixel.
DILATION = 0.3

#: Spherical harmonic constants for bands 1 to 3.
SH_C1 = 0.4886025119029199
SH_C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396)
SH_C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
         -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


@dataclass
class Gaussians:
    """One set of Gaussian splats.

    Attributes:
        means: ``(N, 3)`` centres in the splat frame.
        scales: ``(N, 3)`` standard deviations along each splat's own axes.
        rotations: ``(N, 4)`` wxyz quaternions.
        opacities: ``(N,)`` peak opacity in ``[0, 1]``.
        sh: ``(N, K, 3)`` colour harmonics, ``K = (degree + 1) ** 2``.
    """

    means: torch.Tensor
    scales: torch.Tensor
    rotations: torch.Tensor
    opacities: torch.Tensor
    sh: torch.Tensor

    @property
    def count(self) -> int:
        return int(self.means.shape[0])

    @property
    def degree(self) -> int:
        return int(round(math.sqrt(int(self.sh.shape[1])))) - 1

    def to(self, device) -> "Gaussians":
        return Gaussians(self.means.to(device), self.scales.to(device), self.rotations.to(device),
                         self.opacities.to(device), self.sh.to(device))

    def kept(self, keep: torch.Tensor) -> "Gaussians":
        """Only the splats where ``keep`` is true."""
        return Gaussians(self.means[keep], self.scales[keep], self.rotations[keep], self.opacities[keep],
                         self.sh[keep])


def from_splat(splat) -> Gaussians:
    """A SPLAT's scene as one set of splats: every item of a batch, or a timed SPLAT's still scene.

    Raises:
        ValueError: ``splat`` carries no positions.
    """
    positions = getattr(splat, "positions", None)
    if positions is None:
        raise ValueError("the point_cloud input is not a SPLAT; wire Point Cloud From Depth or a splat node")
    parts = []
    for index, length in splat_items(splat):
        means = positions[index, :length].float()
        scales = splat.scales[index, :length].float() if splat.scales is not None else torch.full((length, 3), 1e-3)
        rotations = (splat.rotations[index, :length].float() if splat.rotations is not None
                     else torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(length, 4))
        opacities = (splat.opacities[index, :length, 0].float() if splat.opacities is not None
                     else torch.ones(length))
        sh = splat.sh[index, :length].float() if splat.sh is not None else torch.zeros(length, 1, 3)
        parts.append(Gaussians(means, scales, rotations, opacities, sh))
    if not parts:
        return Gaussians(torch.zeros(0, 3), torch.zeros(0, 3), torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, 1, 3))
    widest = max(part.sh.shape[1] for part in parts)
    padded = [torch.nn.functional.pad(part.sh, (0, 0, 0, widest - part.sh.shape[1])) for part in parts]
    return Gaussians(torch.cat([p.means for p in parts]), torch.cat([p.scales for p in parts]),
                     torch.cat([p.rotations for p in parts]), torch.cat([p.opacities for p in parts]),
                     torch.cat(padded))


def to_splat(gaussians: Gaussians, moving: Sequence[Cloud] | None = None):
    """Splats as core's SPLAT.

    Args:
        gaussians: The splats, the still scene of a timed SPLAT.
        moving: Each frame's moving points, making the SPLAT timed.

    Raises:
        RuntimeError: The running ComfyUI has no SPLAT type.
    """
    g = gaussians.to("cpu")
    still = (g.means, g.scales, torch.nn.functional.normalize(g.rotations.float(), dim=-1), g.opacities, g.sh)
    if not moving:
        return pack([still])
    every = moving[0]
    for cloud in moving[1:]:
        every = every.joined(cloud)
    m = from_cloud(every.to("cpu"))
    return pack([still, (m.means, m.scales, m.rotations, m.opacities, m.sh)], [cloud.count for cloud in moving])


def from_cloud(cloud: Cloud) -> Gaussians:
    """A point cloud's round dots as opaque, round splats coloured by their base term."""
    count = cloud.count
    rotations = torch.zeros(count, 4, device=cloud.positions.device)
    rotations[:, 0] = 1.0
    return Gaussians(cloud.positions.float(), cloud.spreads.float()[:, None].expand(count, 3).contiguous(), rotations,
                     torch.ones(count, device=cloud.positions.device),
                     ((cloud.colours.float() - 0.5) / SH_C0)[:, None, :])


def drawn_as_dots(splat) -> bool:
    """Whether a SPLAT's scene holds only opaque round splats, as a lifted point cloud does."""
    opacities = getattr(splat, "opacities", None)
    scales = getattr(splat, "scales", None)
    if opacities is None or scales is None or not opacities.numel():
        return True
    for index, length in splat_items(splat):
        if float(opacities[index, :length].min()) < 0.999:
            return False
        sample = scales[index, :length][:: max(1, length // 200_000)].float()
        spread = sample.max(-1).values / sample.min(-1).values.clamp_min(1e-12)
        if not bool((spread < 1.001).all()):
            return False
    return True


def sh_colour(sh: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Colours ``(N, 3)`` of harmonics ``(N, K, 3)`` seen along unit ``directions`` ``(N, 3)``."""
    result = SH_C0 * sh[:, 0]
    bands = int(sh.shape[1])
    if bands >= 4:
        x, y, z = directions[:, 0:1], directions[:, 1:2], directions[:, 2:3]
        result = result - SH_C1 * y * sh[:, 1] + SH_C1 * z * sh[:, 2] - SH_C1 * x * sh[:, 3]
        if bands >= 9:
            xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
            result = (result + SH_C2[0] * xy * sh[:, 4] + SH_C2[1] * yz * sh[:, 5]
                      + SH_C2[2] * (2.0 * zz - xx - yy) * sh[:, 6] + SH_C2[3] * xz * sh[:, 7]
                      + SH_C2[4] * (xx - yy) * sh[:, 8])
            if bands >= 16:
                result = (result + SH_C3[0] * y * (3.0 * xx - yy) * sh[:, 9] + SH_C3[1] * xy * z * sh[:, 10]
                          + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh[:, 11]
                          + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[:, 12]
                          + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh[:, 13]
                          + SH_C3[5] * z * (xx - yy) * sh[:, 14] + SH_C3[6] * x * (xx - 3.0 * yy) * sh[:, 15])
    return (result + 0.5).clamp_min(0.0)


def quaternion_matrices(quaternions: torch.Tensor) -> torch.Tensor:
    """Rotation matrices ``(N, 3, 3)`` of wxyz quaternions ``(N, 4)``, normalised first."""
    w, x, y, z = torch.nn.functional.normalize(quaternions, dim=-1).unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(-1, 3, 3)


@dataclass
class Raster:
    """One rasterized view.

    Attributes:
        colour: ``(H, W, 3)`` blended colour over the background.
        alpha: ``(H, W)`` share of each pixel the splats cover.
        depth: ``(H, W)`` coverage-weighted distance ahead of the camera, 0 where nothing lands.
        centres: ``(M, 2)`` pixel centres of the splats ahead of the camera, or None when none are.
        ahead: ``(M,)`` indices of those splats.
    """

    colour: torch.Tensor
    alpha: torch.Tensor
    depth: torch.Tensor
    centres: torch.Tensor | None = None
    ahead: torch.Tensor | None = None


def _weights(mu, co, op, pixels):
    """Blend weights of tile lists over their pixels, with what their gradient needs.

    Returns:
        ``(weight, alpha, through, falloff, dx, dy)``, each ``(C, K, P)``; ``through`` is the
        light reaching each splat, ``falloff`` the Gaussian before opacity.
    """
    dx = pixels[:, None, :, 0] - mu[:, :, None, 0]
    dy = pixels[:, None, :, 1] - mu[:, :, None, 1]
    power = -0.5 * (co[:, :, None, 0] * dx * dx + co[:, :, None, 2] * dy * dy) - co[:, :, None, 1] * dx * dy
    falloff = torch.exp(power.clamp_max(0.0))
    alpha = (op[:, :, None] * falloff).clamp_max(MOST_ALPHA)
    alpha = torch.where(alpha < LEAST_ALPHA, torch.zeros_like(alpha), alpha)
    lost = torch.log1p(-alpha)
    through = torch.exp(torch.cumsum(lost, dim=1) - lost)
    return alpha * through, alpha, through, falloff, dx, dy


class _Blend(torch.autograd.Function):
    """Front to back blending of per-tile splat lists; the last row pads short lists, gets no gradient."""

    @staticmethod
    def forward(ctx, mean2d, conic, opacity, colour, depth, entries, pixels):
        weight = _weights(mean2d[entries], conic[entries], opacity[entries], pixels)[0]
        ctx.save_for_backward(mean2d, conic, opacity, colour, depth, entries, pixels)
        return (torch.einsum("ckp,ckd->cpd", weight, colour[entries]), weight.sum(1),
                torch.einsum("ckp,ck->cp", weight, depth[entries]))

    @staticmethod
    def backward(ctx, grad_shade, grad_cover, grad_depth):
        mean2d, conic, opacity, colour, depth, entries, pixels = ctx.saved_tensors
        mu, co, op, col, dep = mean2d[entries], conic[entries], opacity[entries], colour[entries], depth[entries]
        weight, alpha, through, falloff, dx, dy = _weights(mu, co, op, pixels)
        grad_shade = grad_shade if grad_shade is not None else torch.zeros(*pixels.shape[:2], 3, device=mu.device)
        grad_cover = grad_cover if grad_cover is not None else torch.zeros(pixels.shape[:2], device=mu.device)
        grad_depth = grad_depth if grad_depth is not None else torch.zeros(pixels.shape[:2], device=mu.device)
        d_col = torch.einsum("ckp,cpd->ckd", weight, grad_shade)
        d_dep = torch.einsum("ckp,cp->ck", weight, grad_depth)
        value = (torch.einsum("ckd,cpd->ckp", col, grad_shade) + dep[:, :, None] * grad_depth[:, None, :]
                 + grad_cover[:, None, :])
        share = weight * value
        running = torch.cumsum(share, dim=1)
        behind = running[:, -1:] - running
        d_alpha = through * value - behind / (1.0 - alpha)
        live = (alpha > 0) & (alpha < MOST_ALPHA)
        d_alpha = torch.where(live, d_alpha, torch.zeros_like(d_alpha))
        d_op = (d_alpha * falloff).sum(-1)
        d_power = d_alpha * op[:, :, None] * falloff
        d_co = torch.stack([(-0.5 * d_power * dx * dx).sum(-1), (-d_power * dx * dy).sum(-1),
                            (-0.5 * d_power * dy * dy).sum(-1)], -1)
        a, b, c = co[:, :, None, 0], co[:, :, None, 1], co[:, :, None, 2]
        d_mu = torch.stack([(d_power * (a * dx + b * dy)).sum(-1), (d_power * (c * dy + b * dx)).sum(-1)], -1)
        real = entries < mean2d.shape[0] - 1
        index = entries[real]

        def spread(source: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(source).index_add_(0, index, grad[real])

        return (spread(mean2d, d_mu), spread(conic, d_co), spread(opacity, d_op), spread(colour, d_col),
                spread(depth, d_dep), None, None)


def _blend(mean2d, conic, opacity, colour, depth, entries, pixels):
    """Blend the splats listed per tile in ``entries`` ``(C, K)`` over pixels ``(C, P, 2)``.

    Returns:
        ``(colour (C, P, 3), coverage (C, P), depth (C, P))``.
    """
    return _Blend.apply(mean2d, conic, opacity, colour, depth, entries, pixels)


def rasterize(gaussians: Gaussians, camera_info: Mapping, width: int, height: int, background=0.0,
              eye: torch.Tensor | None = None, rows: torch.Tensor | None = None, tile: int = TILE,
              near: float = 1e-4, widest: float | None = None) -> Raster:
    """Draw splats from one camera, blending each pixel front to back.

    Args:
        gaussians: The splats; gradients flow to every field.
        camera_info: The camera; supplies the focal length, and the pose unless given.
        width: Picture width in pixels.
        height: Picture height in pixels.
        background: Grey, or an ``(3,)`` colour, shown where coverage is incomplete.
        eye: Camera position in the splat frame, to carry a gradient; read from the camera when None.
        rows: ``(3, 3)`` right, down and forward rows in the splat frame, likewise.
        tile: Tile side in pixels; smaller is faster for small splats, larger for big ones.
        near: Splats nearer the camera than this are left out.
        widest: Splats reaching further than this many pixels from their centre are left out.

    Returns:
        The view.
    """
    device = gaussians.means.device
    if eye is None or rows is None:
        eye_given, rows_given = cameras.view(camera_info)
        eye = eye_given.float().to(device) if eye is None else eye
        rows = rows_given.float().to(device) if rows is None else rows
    focal = float(cameras.focal(camera_info, width, height))
    tiles_x, tiles_y = math.ceil(width / tile), math.ceil(height / tile)
    colour_bg = torch.as_tensor(background, dtype=torch.float32, device=device).expand(3)

    def empty() -> Raster:
        return Raster(colour_bg.expand(height, width, 3).clone(), torch.zeros(height, width, device=device),
                      torch.zeros(height, width, device=device))

    if not gaussians.count:
        return empty()
    local = (gaussians.means - eye) @ rows.T
    z = local[:, 2]
    with torch.no_grad():
        ahead = torch.nonzero(z > near, as_tuple=False)[:, 0]
    if not ahead.numel():
        return empty()
    local, z = local[ahead], z[ahead]
    limit_x = 1.3 * (width / 2.0) / focal
    limit_y = 1.3 * (height / 2.0) / focal
    tx = (local[:, 0] / z).clamp(-limit_x, limit_x) * z
    ty = (local[:, 1] / z).clamp(-limit_y, limit_y) * z
    turn = quaternion_matrices(gaussians.rotations[ahead])
    shape = turn * gaussians.scales[ahead][:, None, :]
    cov_world = (shape[:, :, None, :] * shape[:, None, :, :]).sum(-1).reshape(-1, 9)
    s = (cov_world @ torch.kron(rows, rows).T).reshape(-1, 3, 3)
    j0 = focal / z
    j2 = -focal * tx / (z * z)
    j5 = -focal * ty / (z * z)
    a = j0 * j0 * s[:, 0, 0] + 2.0 * j0 * j2 * s[:, 0, 2] + j2 * j2 * s[:, 2, 2] + DILATION
    b = j0 * j0 * s[:, 0, 1] + j0 * j5 * s[:, 0, 2] + j2 * j0 * s[:, 1, 2] + j2 * j5 * s[:, 2, 2]
    c = j0 * j0 * s[:, 1, 1] + 2.0 * j0 * j5 * s[:, 1, 2] + j5 * j5 * s[:, 2, 2] + DILATION
    det = (a * c - b * b).clamp_min(1e-12)
    conic = torch.stack([c / det, -b / det, a / det], -1)
    mean2d = torch.stack([focal * local[:, 0] / z + width / 2.0, focal * local[:, 1] / z + height / 2.0], -1)
    with torch.no_grad():
        middle = 0.5 * (a + c)
        largest = middle + torch.sqrt((middle * middle - det).clamp_min(0.1))
        radius = torch.ceil(3.0 * torch.sqrt(largest))
        u, v = mean2d[:, 0], mean2d[:, 1]
        x0 = torch.floor((u - radius) / tile).clamp(0, tiles_x - 1)
        x1 = torch.floor((u + radius) / tile).clamp(0, tiles_x - 1)
        y0 = torch.floor((v - radius) / tile).clamp(0, tiles_y - 1)
        y1 = torch.floor((v + radius) / tile).clamp(0, tiles_y - 1)
        seen = (u + radius >= 0) & (u - radius < width) & (v + radius >= 0) & (v - radius < height)
        seen &= torch.isfinite(u) & torch.isfinite(v)
        if widest is not None:
            seen &= radius <= widest
        span_x = torch.where(seen, x1 - x0 + 1, torch.zeros_like(x0)).long()
        span_y = torch.where(seen, y1 - y0 + 1, torch.zeros_like(y0)).long()
        spans = span_x * span_y
        pairs = int(spans.sum())
    if not pairs:
        return empty()
    directions = torch.nn.functional.normalize(gaussians.means[ahead] - eye, dim=-1)
    shade = sh_colour(gaussians.sh[ahead], directions)
    opacity = gaussians.opacities[ahead]
    with torch.no_grad():
        owner = torch.repeat_interleave(torch.arange(spans.numel(), device=device), spans)
        first = torch.cumsum(spans, 0) - spans
        step = torch.arange(pairs, device=device) - first[owner]
        slot = ((y0.long()[owner] + step // span_x[owner]) * tiles_x + x0.long()[owner] + step % span_x[owner])
        order = torch.argsort(z.detach())
        rank_of = torch.empty(z.shape[0], dtype=torch.long, device=device)
        rank_of[order] = torch.arange(z.shape[0], device=device)
        key = slot * z.shape[0] + rank_of[owner]
        key, sort = torch.sort(key)
        owner = owner[sort]
        slot = slot[sort]
        del step, first, key
        tile_count = torch.bincount(slot, minlength=tiles_x * tiles_y)
        tile_start = torch.cumsum(tile_count, 0) - tile_count
        busy = torch.nonzero(tile_count, as_tuple=False)[:, 0]
        busy = busy[torch.argsort(tile_count[busy], descending=True)]
        offsets = torch.arange(tile, device=device, dtype=torch.float32) + 0.5
        cell_y, cell_x = torch.meshgrid(offsets, offsets, indexing="ij")
        cell = torch.stack([cell_x.reshape(-1), cell_y.reshape(-1)], -1)
    pad = z.shape[0]
    mean2d_p = torch.cat([mean2d, mean2d.new_zeros(1, 2)])
    conic_p = torch.cat([conic, conic.new_zeros(1, 3)])
    opacity_p = torch.cat([opacity, opacity.new_zeros(1)])
    shade_p = torch.cat([shade, shade.new_zeros(1, 3)])
    depth_p = torch.cat([z, z.new_zeros(1)])
    counts = tile_count[busy].tolist()
    descending = [-count for count in counts]
    pixels_per_tile = tile * tile
    shades, covers, depths, done = [], [], [], []
    start = 0
    while start < len(counts):
        deepest = counts[start]
        shallower = bisect.bisect_right(descending, -deepest * PASS_SPREAD)
        take = max(1, min(PASS_ELEMENTS // (deepest * pixels_per_tile), shallower - start))
        group = busy[start:start + take]
        with torch.no_grad():
            depth_range = torch.arange(deepest, device=device)
            group_count = tile_count[group]
            entries = tile_start[group][:, None] + depth_range[None, :]
            entries = torch.where(depth_range[None, :] < group_count[:, None],
                                  owner[entries.clamp_max(owner.numel() - 1)], torch.full_like(entries, pad))
            corner = torch.stack([(group % tiles_x) * tile, (group // tiles_x) * tile], -1).float()
            pixels = corner[:, None, :] + cell[None, :, :]
        part = _blend(mean2d_p, conic_p, opacity_p, shade_p, depth_p, entries, pixels)
        shades.append(part[0])
        covers.append(part[1])
        depths.append(part[2])
        done.append(group)
        start += int(group.numel())
    order_tiles = torch.cat(done)
    count_tiles = tiles_x * tiles_y
    shade_tiles = torch.zeros(count_tiles, pixels_per_tile, 3, device=device).index_put((order_tiles,), torch.cat(shades))
    cover_tiles = torch.zeros(count_tiles, pixels_per_tile, device=device).index_put((order_tiles,), torch.cat(covers))
    depth_tiles = torch.zeros(count_tiles, pixels_per_tile, device=device).index_put((order_tiles,), torch.cat(depths))

    def untile(values: torch.Tensor) -> torch.Tensor:
        channels = values.shape[2:] if values.dim() > 2 else ()
        grid = values.reshape(tiles_y, tiles_x, tile, tile, *channels).transpose(1, 2)
        return grid.reshape(tiles_y * tile, tiles_x * tile, *channels)[:height, :width]

    cover = untile(cover_tiles)
    colour = untile(shade_tiles) + (1.0 - cover)[..., None] * colour_bg
    return Raster(colour, cover, untile(depth_tiles), mean2d, ahead)

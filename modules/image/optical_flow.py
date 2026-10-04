"""Dense optical flow between video frames, in torch.

Frames are ``(batch, 1, height, width)`` luminance in ``[0, 255]``. A flow is
``(batch, 2, height, width)`` in pixels, x first, mapping frame ``a`` onto frame ``b``:
``a(p) ~ b(p + flow(p))``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

__all__ = ["consistent", "estimate", "luminance", "resize", "resize_flow", "warp"]

#: Side of the square patches the coarse search matches, in pixels.
PATCH = 8

#: Spacing between neighbouring patches, in pixels.
STRIDE = 4

#: Gauss-Newton steps each patch takes on each pyramid level.
PATCH_STEPS = 16

#: Gaussian sigma of the local mean removed before refining, and the share of it kept.
TEXTURE = (3.0, 0.05)

#: Variational refinement on each level: warps, and iterations per warp.
REFINE = (2, 20)

#: Weight of the brightness term in the refinement against the smoothness term.
LAMBDA = 0.15

#: Gaussian sigma applied to both frames before the pyramid is built.
PRESMOOTH = 0.8

#: Forward-backward test: share of the squared motion allowed as disagreement, and a floor in
#: squared pixels.
AGREEMENT = (0.01, 0.5)


def luminance(images):
    """Rec. 601 luma of a channels-last image batch.

    Args:
        images: ``(batch, height, width, channels)`` in ``[0, 1]``. One channel is taken as is.

    Returns:
        ``(batch, 1, height, width)`` float32 in ``[0, 255]``.
    """
    images = images.to(torch.float32)
    if images.shape[-1] >= 3:
        luma = 0.299 * images[..., 0] + 0.587 * images[..., 1] + 0.114 * images[..., 2]
    else:
        luma = images[..., 0]
    return (luma.clamp(0.0, 1.0) * 255.0).unsqueeze(1)


def _kernel(sigma: float, device):
    """A normalised 1D Gaussian reaching three sigma."""
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def gaussian(x, sigma: float):
    """Separable Gaussian blur with replicated edges.

    Args:
        x: ``(batch, channels, height, width)``.
        sigma: Standard deviation in pixels. 0 or below answers ``x``.

    Returns:
        A tensor shaped like ``x``.
    """
    if sigma <= 0:
        return x
    kernel = _kernel(sigma, x.device)
    radius = (kernel.numel() - 1) // 2
    channels = x.shape[1]
    across = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1).contiguous()
    down = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1).contiguous()
    x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="replicate"), across, groups=channels)
    return F.conv2d(F.pad(x, (0, 0, radius, radius), mode="replicate"), down, groups=channels)


def unit(x, device=None):
    """Frames as float32 in ``[0, 1]``, 8-bit codes divided by 255.

    Args:
        x: A float tensor, or a uint8 one.
        device: Where the answer is placed; the tensor's own device when None.

    Returns:
        The float32 tensor.
    """
    x = x.to(device=device if device is not None else x.device)
    if x.dtype == torch.uint8:
        return x.to(torch.float32) / 255.0
    return x.to(torch.float32)


def resize(x, height: int, width: int):
    """Antialiased bilinear resize.

    Args:
        x: ``(batch, channels, height, width)``.
        height: Target height.
        width: Target width.

    Returns:
        The resized tensor.
    """
    if tuple(x.shape[-2:]) == (height, width):
        return x
    return F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False, antialias=True)


def resize_flow(flow, height: int, width: int):
    """Resize a flow and scale its vectors to the new size.

    Args:
        flow: ``(batch, 2, height, width)`` in pixels.
        height: Target height.
        width: Target width.

    Returns:
        ``(batch, 2, height, width)`` in pixels of the new size.
    """
    old_height, old_width = flow.shape[-2:]
    if (old_height, old_width) == (height, width):
        return flow
    resized = F.interpolate(flow, size=(height, width), mode="bilinear", align_corners=False)
    scale = torch.tensor(
        [width / old_width, height / old_height], dtype=flow.dtype, device=flow.device
    ).view(1, 2, 1, 1)
    return resized * scale


def _grid(flow):
    """``grid_sample`` coordinates of ``p + flow(p)``, corners aligned."""
    _, _, height, width = flow.shape
    xs = torch.arange(width, device=flow.device, dtype=flow.dtype).view(1, 1, width)
    ys = torch.arange(height, device=flow.device, dtype=flow.dtype).view(1, height, 1)
    gx = (xs + flow[:, 0]) * (2.0 / max(width - 1, 1)) - 1.0
    gy = (ys + flow[:, 1]) * (2.0 / max(height - 1, 1)) - 1.0
    return torch.stack([gx, gy], -1)


def warp(x, flow):
    """Sample ``x`` at ``p + flow(p)``, bilinear, edges held.

    Args:
        x: ``(batch, channels, height, width)``.
        flow: ``(batch, 2, height, width)`` in pixels.

    Returns:
        A tensor shaped like ``x``.
    """
    return F.grid_sample(x, _grid(flow), mode="bilinear", padding_mode="border", align_corners=True)


def _outside(flow):
    """Where ``p + flow(p)`` leaves the frame, ``(batch, 1, height, width)`` bool."""
    _, _, height, width = flow.shape
    xs = torch.arange(width, device=flow.device, dtype=flow.dtype).view(1, 1, width)
    ys = torch.arange(height, device=flow.device, dtype=flow.dtype).view(1, height, 1)
    x = xs + flow[:, 0]
    y = ys + flow[:, 1]
    return ((x < 0) | (x > width - 1) | (y < 0) | (y > height - 1)).unsqueeze(1)


def consistent(forward, backward):
    """Where a forward flow and the backward flow it lands on agree.

    Args:
        forward: Flow from frame ``a`` to frame ``b``.
        backward: Flow from frame ``b`` to frame ``a``.

    Returns:
        ``(batch, 1, height, width)`` bool on frame ``a``'s pixels. False where the pixel is
        covered in ``b``, leaves the frame, or the two flows disagree.
    """
    returned = warp(backward, forward)
    gap = forward + returned
    share, floor = AGREEMENT
    allowed = share * ((forward * forward).sum(1, keepdim=True) + (returned * returned).sum(1, keepdim=True)) + floor
    return ((gap * gap).sum(1, keepdim=True) < allowed) & ~_outside(forward)


def _gradients(x):
    """Central differences with replicated edges, ``(gx, gy)``."""
    padded = F.pad(x, (1, 1, 1, 1), mode="replicate")
    gx = 0.5 * (padded[..., 1:-1, 2:] - padded[..., 1:-1, :-2])
    gy = 0.5 * (padded[..., 2:, 1:-1] - padded[..., :-2, 1:-1])
    return gx, gy


def _texture(x):
    """``x`` with most of its local mean removed."""
    sigma, kept = TEXTURE
    return x - (1.0 - kept) * gaussian(x, sigma)


def _forward_difference(u):
    """Forward differences, zero on the far edge, ``(ux, uy)``."""
    ux = F.pad(u[..., :, 1:] - u[..., :, :-1], (0, 1, 0, 0))
    uy = F.pad(u[..., 1:, :] - u[..., :-1, :], (0, 0, 0, 1))
    return ux, uy


def _divergence(px, py):
    """Backward-difference divergence of a dual field."""
    return (px - F.pad(px[..., :, :-1], (1, 0, 0, 0))) + (py - F.pad(py[..., :-1, :], (0, 0, 1, 0)))


def _refine(a, b, u, warps: int, iterations: int, lam: float = LAMBDA, theta: float = 0.3, tau: float = 0.25):
    """TV-L1 refinement of a flow on one pyramid level.

    Args:
        a: First frame on this level.
        b: Second frame on this level.
        u: Starting flow.
        warps: Times ``b`` is warped by the current flow.
        iterations: Primal-dual steps per warp.
        lam: Brightness weight.
        theta: Coupling between the data and smoothness steps.
        tau: Dual step size.

    Returns:
        The refined flow.
    """
    bx, by = _gradients(b)
    stacked = torch.cat([b, bx, by], 1)
    lt = lam * theta
    px = torch.zeros_like(u)
    py = torch.zeros_like(u)
    for _ in range(warps):
        warped = warp(stacked, u)
        bw, gx, gy = warped[:, 0:1], warped[:, 1:2], warped[:, 2:3]
        g = torch.cat([gx, gy], 1)
        grad2 = gx * gx + gy * gy
        rho_c = bw - (g * u).sum(1, keepdim=True) - a
        threshold = lt * grad2
        inverse = 1.0 / (grad2 + 1e-10)
        for _ in range(iterations):
            rho = rho_c + (g * u).sum(1, keepdim=True)
            step = torch.where(rho < -threshold, lt, torch.where(rho > threshold, -lt, -rho * inverse))
            v = u + step * g
            u = v + theta * _divergence(px, py)
            ux, uy = _forward_difference(u)
            norm = 1.0 + (tau / theta) * torch.sqrt(ux * ux + uy * uy)
            px = (px + (tau / theta) * ux) / norm
            py = (py + (tau / theta) * uy) / norm
    return u


def _patch_grid(height: int, width: int, device):
    """Patch layout over a level: counts, padded size and the pixel coordinates of every patch."""
    rows = int(math.ceil((height - PATCH) / STRIDE)) + 1
    columns = int(math.ceil((width - PATCH) / STRIDE)) + 1
    padded_height = (rows - 1) * STRIDE + PATCH
    padded_width = (columns - 1) * STRIDE + PATCH
    oy = torch.arange(rows, device=device, dtype=torch.float32) * STRIDE
    ox = torch.arange(columns, device=device, dtype=torch.float32) * STRIDE
    j = torch.arange(PATCH, device=device, dtype=torch.float32)
    py = (oy.view(rows, 1, 1, 1) + j.view(1, 1, PATCH, 1)).expand(rows, columns, PATCH, PATCH)
    px = (ox.view(1, columns, 1, 1) + j.view(1, 1, 1, PATCH)).expand(rows, columns, PATCH, PATCH)
    py = py.reshape(rows * columns, PATCH * PATCH).T.contiguous()
    px = px.reshape(rows * columns, PATCH * PATCH).T.contiguous()
    return rows, columns, padded_height, padded_width, px, py


def _sample_patches(image, px, py, ux, uy):
    """Every patch of ``image`` displaced by its own offset, ``(batch, patch pixels, patches)``."""
    _, _, height, width = image.shape
    gx = (px.unsqueeze(0) + ux.unsqueeze(1)) * (2.0 / max(width - 1, 1)) - 1.0
    gy = (py.unsqueeze(0) + uy.unsqueeze(1)) * (2.0 / max(height - 1, 1)) - 1.0
    grid = torch.stack([gx, gy], -1)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)[:, 0]


def _search(a, b, flow):
    """Inverse-compositional patch search on one level, densified to a flow.

    Args:
        a: First frame on this level.
        b: Second frame on this level.
        flow: Starting flow at this level's size.

    Returns:
        The flow after every patch has been matched.
    """
    batch, _, height, width = a.shape
    rows, columns, padded_height, padded_width, px, py = _patch_grid(height, width, a.device)
    padded = F.pad(a, (0, padded_width - width, 0, padded_height - height), mode="replicate")
    ax, ay = _gradients(padded)
    template = F.unfold(padded, PATCH, stride=STRIDE)
    tx = F.unfold(ax, PATCH, stride=STRIDE)
    ty = F.unfold(ay, PATCH, stride=STRIDE)
    template = template - template.mean(1, keepdim=True)
    tx = tx - tx.mean(1, keepdim=True)
    ty = ty - ty.mean(1, keepdim=True)
    h11 = (tx * tx).sum(1) + 1e-2
    h12 = (tx * ty).sum(1)
    h22 = (ty * ty).sum(1) + 1e-2
    det = h11 * h22 - h12 * h12
    padded_flow = F.pad(flow, (0, padded_width - width, 0, padded_height - height), mode="replicate")
    start = F.avg_pool2d(padded_flow, PATCH, stride=STRIDE).reshape(batch, 2, -1)
    ux, uy = start[:, 0].clone(), start[:, 1].clone()

    def cost(cx, cy):
        sampled = _sample_patches(b, px, py, cx, cy)
        sampled = sampled - sampled.mean(1, keepdim=True)
        residual = sampled - template
        return (residual * residual).sum(1), residual

    def propagate(ux, uy):
        best, _ = cost(ux, uy)
        gx = ux.view(batch, rows, columns)
        gy = uy.view(batch, rows, columns)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            sx = torch.roll(gx, shifts=(dy, dx), dims=(1, 2)).reshape(batch, -1)
            sy = torch.roll(gy, shifts=(dy, dx), dims=(1, 2)).reshape(batch, -1)
            candidate, _ = cost(sx, sy)
            take = candidate < best
            ux = torch.where(take, sx, ux)
            uy = torch.where(take, sy, uy)
            best = torch.where(take, candidate, best)
        return ux, uy

    half = PATCH_STEPS // 2
    for steps in (half, PATCH_STEPS - half):
        ux, uy = propagate(ux, uy)
        for _ in range(steps):
            _, residual = cost(ux, uy)
            b1 = (tx * residual).sum(1)
            b2 = (ty * residual).sum(1)
            ux = ux - (h22 * b1 - h12 * b2) / det
            uy = uy - (h11 * b2 - h12 * b1) / det
    # A patch that moved more than its own width from where it started goes back.
    strayed = ((ux - start[:, 0]) ** 2 + (uy - start[:, 1]) ** 2) > (PATCH * PATCH)
    ux = torch.where(strayed, start[:, 0], ux)
    uy = torch.where(strayed, start[:, 1], uy)
    _, residual = cost(ux, uy)
    weight = 1.0 / torch.clamp(residual.abs(), min=1.0)
    size = (padded_height, padded_width)
    dense_x = F.fold(weight * ux.unsqueeze(1), size, PATCH, stride=STRIDE)
    dense_y = F.fold(weight * uy.unsqueeze(1), size, PATCH, stride=STRIDE)
    total = F.fold(weight, size, PATCH, stride=STRIDE)
    return (torch.cat([dense_x, dense_y], 1) / total)[..., :height, :width]


def estimate(a, b):
    """Dense flow from ``a`` to ``b``.

    Args:
        a: ``(batch, 1, height, width)`` luminance in ``[0, 255]``.
        b: The frames ``a`` is matched against, shaped like ``a``.

    Returns:
        ``(batch, 2, height, width)`` in pixels, with ``a(p) ~ b(p + flow(p))``.
    """
    height, width = a.shape[-2:]
    levels = max(0, int(math.log(max(height, width) / (4.0 * PATCH)) / math.log(2.0) + 0.5))
    sizes = [(height, width)]
    for _ in range(levels):
        sizes.append((max(1, int(round(sizes[-1][0] * 0.5))), max(1, int(round(sizes[-1][1] * 0.5)))))
    a = gaussian(a.to(torch.float32), PRESMOOTH)
    b = gaussian(b.to(torch.float32), PRESMOOTH)
    flow = torch.zeros(a.shape[0], 2, sizes[-1][0], sizes[-1][1], device=a.device)
    warps, iterations = REFINE
    for level_height, level_width in reversed(sizes):
        level_a = resize(a, level_height, level_width)
        level_b = resize(b, level_height, level_width)
        flow = resize_flow(flow, level_height, level_width)
        if min(level_height, level_width) >= PATCH:
            flow = _search(level_a, level_b, flow)
        flow = _refine(_texture(level_a), _texture(level_b), flow, warps, iterations)
    return flow

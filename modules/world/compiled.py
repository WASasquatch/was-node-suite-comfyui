"""Splat drawing on compiled GPU kernels, for CUDA devices when the ``slangpy`` package is installed.

Projection, colour and blending each run as one kernel with its gradient, on torch tensors in place.
"""

from __future__ import annotations

__all__ = ["TILE", "available", "draw"]

import os
import threading

import torch

from .. import deps, log

logger = log.get_logger("world.compiled")

#: Side of a square tile the kernels draw, in pixels.
TILE = 16

_SHADERS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")
_lock = threading.Lock()
_kernels: dict[int, object] = {}
_failed: set[int] = set()


class _Kernels:
    """The compiled kernels on one CUDA device."""

    def __init__(self, spy, index: int):
        self.spy = spy
        with torch.cuda.device(index):
            self.device = spy.create_torch_device(type=spy.DeviceType.cuda, torch_device=index,
                                                  include_paths=[_SHADERS], enable_hot_reload=False)

        def kernel(module: str, entry: str):
            return self.device.create_compute_kernel(self.device.load_program(module, [entry]))

        self.project_forward = kernel("splat_project", "project_forward")
        self.project_backward = kernel("splat_project", "project_backward")
        self.blend_forward = kernel("splat_blend", "blend_forward")
        self.blend_backward = kernel("splat_blend", "blend_backward")

    def run(self, kernel, threads, **values) -> None:
        """Dispatch ``kernel`` over ``threads`` on torch's current stream; tensors pass as device addresses."""
        stream = self.spy.NativeHandle.from_cuda_stream(torch.cuda.current_stream().cuda_stream)
        given = {name: value.data_ptr() if isinstance(value, torch.Tensor) else value for name, value in values.items()}
        kernel.dispatch(list(threads), cuda_stream=stream, **given)


def _get(device: torch.device) -> _Kernels | None:
    """The kernels for ``device``, compiled on first use, or None where they cannot run."""
    if device.type != "cuda":
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    kernels = _kernels.get(index)
    if kernels is not None or index in _failed:
        return kernels
    spy = deps.optional("slangpy")
    if spy is None:
        _failed.add(index)
        return None
    with _lock:
        if index in _kernels:
            return _kernels[index]
        try:
            _kernels[index] = _Kernels(spy, index)
            logger.info("Gaussian splats draw on compiled kernels (slangpy %s).", getattr(spy, "__version__", "?"))
        except Exception as error:
            _failed.add(index)
            first = str(error).splitlines()[0] if str(error) else type(error).__name__
            logger.warning("slangpy is installed but its kernels could not start, so splats draw in torch: %s", first)
            return None
    return _kernels[index]


def available(device) -> bool:
    """Whether the compiled kernels can draw on ``device``."""
    return _get(torch.device(device)) is not None


def _flat(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().float().contiguous()


class _Project(torch.autograd.Function):
    """Splats to screen centres, conics, depths and colours, with each splat's tile rectangle."""

    @staticmethod
    def forward(ctx, means, scales, quats, sh, eye, rows, opacities, focal, size, near, widest):
        kernels = _get(means.device)
        width, height = size
        count = int(means.shape[0])
        bands = int(sh.shape[1])
        tiles_x, tiles_y = -(-width // TILE), -(-height // TILE)
        inputs = [_flat(t) for t in (means, scales, quats, sh, eye, rows)]
        device = means.device
        mean2d = torch.empty(count, 2, device=device)
        conic = torch.empty(count, 3, device=device)
        depth = torch.empty(count, device=device)
        shade = torch.empty(count, 3, device=device)
        rect = torch.empty(count, 4, dtype=torch.int32, device=device)
        kernels.run(kernels.project_forward, (count, 1, 1), means=inputs[0], scales=inputs[1], quats=inputs[2],
                    opacities=_flat(opacities), sh=inputs[3], eye=inputs[4], rows=inputs[5], out_mean2d=mean2d,
                    out_conic=conic, out_depth=depth, out_shade=shade, out_rect=rect, count=count, focal=float(focal),
                    width=float(width), height=float(height), near=float(near),
                    widest=float(widest) if widest is not None else 0.0, bands=bands, tile=TILE, tiles_x=tiles_x,
                    tiles_y=tiles_y)
        ctx.save_for_backward(*inputs, depth)
        ctx.settings = (float(focal), width, height, float(near), bands)
        ctx.mark_non_differentiable(rect)
        return mean2d, conic, depth, shade, rect

    @staticmethod
    def backward(ctx, grad_mean2d, grad_conic, grad_depth, grad_shade, grad_rect):
        means, scales, quats, sh, eye, rows, depth = ctx.saved_tensors
        focal, width, height, near, bands = ctx.settings
        kernels = _get(means.device)
        count = int(means.shape[0])

        def given(grad, like):
            return grad.float().contiguous() if grad is not None else torch.zeros_like(like)

        d_means, d_scales, d_quats = torch.zeros_like(means), torch.zeros_like(scales), torch.zeros_like(quats)
        d_sh, d_eye, d_rows = torch.zeros_like(sh), torch.zeros_like(eye), torch.zeros_like(rows)
        camera = bool(ctx.needs_input_grad[4] or ctx.needs_input_grad[5])
        kernels.run(kernels.project_backward, (count, 1, 1), means=means, scales=scales, quats=quats, sh=sh, eye=eye,
                    rows=rows, in_depth=depth, grad_mean2d=given(grad_mean2d, torch.empty(count, 2, device=means.device)),
                    grad_conic=given(grad_conic, torch.empty(count, 3, device=means.device)),
                    grad_depth=given(grad_depth, depth),
                    grad_shade=given(grad_shade, torch.empty(count, 3, device=means.device)), d_means=d_means,
                    d_scales=d_scales, d_quats=d_quats, d_sh=d_sh, d_eye=d_eye, d_rows=d_rows, count=count,
                    focal=focal, width=float(width), height=float(height), near=near, bands=bands,
                    camera_grad=1 if camera else 0)
        return (d_means, d_scales, d_quats, d_sh, d_eye if camera else None, d_rows if camera else None,
                None, None, None, None, None)


class _Blend(torch.autograd.Function):
    """Front to back blending of splats listed tile by tile, nearest first, with its gradient."""

    @staticmethod
    def forward(ctx, mean2d, conic, opacity, shade, depth, owner, tile_start, tile_count, width, height):
        kernels = _get(mean2d.device)
        tiles_x, tiles_y = -(-width // TILE), -(-height // TILE)
        inputs = [_flat(t) for t in (mean2d, conic, opacity, shade, depth)]
        lists = [t.int().contiguous() for t in (owner, tile_start, tile_count)]
        device = mean2d.device
        colour = torch.empty(height, width, 3, device=device)
        cover = torch.empty(height, width, device=device)
        depth_image = torch.empty(height, width, device=device)
        through = torch.empty(height, width, device=device)
        last = torch.empty(height, width, dtype=torch.int32, device=device)
        kernels.run(kernels.blend_forward, (tiles_x * TILE, tiles_y * TILE, 1), mean2d=inputs[0], conic=inputs[1],
                    opacity=inputs[2], shade=inputs[3], depth=inputs[4], owner=lists[0], tile_start=lists[1],
                    tile_count=lists[2], out_colour=colour, out_cover=cover, out_depth=depth_image,
                    out_through=through, out_last=last, width=width, height=height, tiles_x=tiles_x)
        ctx.save_for_backward(*inputs, *lists, through, last)
        ctx.size = (width, height)
        return colour, cover, depth_image

    @staticmethod
    def backward(ctx, grad_colour, grad_cover, grad_depth):
        mean2d, conic, opacity, shade, depth, owner, tile_start, tile_count, through, last = ctx.saved_tensors
        width, height = ctx.size
        kernels = _get(mean2d.device)
        tiles_x, tiles_y = -(-width // TILE), -(-height // TILE)

        def given(grad, shape):
            return grad.float().contiguous() if grad is not None else torch.zeros(shape, device=mean2d.device)

        d_mean2d, d_conic, d_opacity = torch.zeros_like(mean2d), torch.zeros_like(conic), torch.zeros_like(opacity)
        d_shade, d_depth = torch.zeros_like(shade), torch.zeros_like(depth)
        kernels.run(kernels.blend_backward, (tiles_x * TILE, tiles_y * TILE, 1), mean2d=mean2d, conic=conic,
                    opacity=opacity, shade=shade, depth=depth, owner=owner, tile_start=tile_start,
                    tile_count=tile_count, in_through=through, in_last=last,
                    grad_colour=given(grad_colour, (height, width, 3)), grad_cover=given(grad_cover, (height, width)),
                    grad_depth=given(grad_depth, (height, width)), d_mean2d=d_mean2d, d_conic=d_conic,
                    d_opacity=d_opacity, d_shade=d_shade, d_depth=d_depth, width=width, height=height,
                    tiles_x=tiles_x)
        return d_mean2d, d_conic, d_opacity, d_shade, d_depth, None, None, None, None, None


def _bin(rect: torch.Tensor, depth: torch.Tensor, tiles_x: int, tiles_y: int):
    """Tile lists of splats nearest first: ``(owner, tile_start, tile_count)``, or None when nothing lands."""
    span_x, span_y = rect[:, 2].long(), rect[:, 3].long()
    spans = span_x * span_y
    seen = torch.nonzero(spans, as_tuple=False)[:, 0]
    order = seen[torch.argsort(depth[seen])]
    spans = spans[order]
    pairs = int(spans.sum())
    if not pairs:
        return None
    # Pairs are emitted nearest splat first, so a stable sort on the tile alone keeps depth order.
    owner = torch.repeat_interleave(order, spans)
    first = torch.cumsum(spans, 0) - spans
    step = torch.arange(pairs, device=rect.device) - torch.repeat_interleave(first, spans)
    width_of = span_x[owner]
    slot = ((rect[owner, 1].long() + step // width_of) * tiles_x + rect[owner, 0].long() + step % width_of).int()
    slot, sort = torch.sort(slot, stable=True)
    owner = owner[sort]
    tile_count = torch.bincount(slot, minlength=tiles_x * tiles_y)
    tile_start = torch.cumsum(tile_count, 0) - tile_count
    return owner, tile_start, tile_count


def draw(means, scales, quats, opacities, sh, eye, rows, focal: float, width: int, height: int, near: float,
         widest: float | None):
    """Draw splats from one camera, front to back, with gradients to every input.

    Args:
        means: ``(N, 3)`` centres.
        scales: ``(N, 3)`` standard deviations.
        quats: ``(N, 4)`` wxyz rotations, normalised here.
        opacities: ``(N,)`` opacities.
        sh: ``(N, K, 3)`` spherical harmonic colours, K 1, 4, 9 or 16.
        eye: ``(3,)`` camera position.
        rows: ``(3, 3)`` the picture's right, down and forward directions.
        focal: Focal length in pixels.
        width: Picture width in pixels.
        height: Picture height in pixels.
        near: Splats nearer than this are left out.
        widest: Splats reaching further than this many pixels from their centre are left out; None = none.

    Returns:
        ``(colour (H, W, 3), coverage (H, W), depth (H, W), centres (N, 2))``, colour not yet over a
        background, or None when no splat lands in the picture.
    """
    mean2d, conic, depth, shade, rect = _Project.apply(means, scales, quats, sh, eye, rows, opacities, focal,
                                                       (int(width), int(height)), near, widest)
    tiles_x, tiles_y = -(-int(width) // TILE), -(-int(height) // TILE)
    with torch.no_grad():
        lists = _bin(rect, depth, tiles_x, tiles_y)
    if lists is None:
        return None
    colour, cover, depth_image = _Blend.apply(mean2d, conic, opacities, shade, depth, *lists, int(width), int(height))
    return colour, cover, depth_image, mean2d

"""Fitting Gaussian splats to pictures taken from known cameras, optionally moving the cameras too.

Each step scores one picture's view by L1 and structural similarity; splats are added where
the match is worst. Pictures are float32 ``(N, H, W, 3)``.
"""

from __future__ import annotations

__all__ = ["Fitted", "fit", "psnr", "shrink"]

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from . import cameras
from .gaussians import FIT_TILE, Gaussians, quaternion_matrices, rasterize

#: Share of the loss given to structural similarity, the rest to L1.
SSIM_SHARE = 0.2

#: Opacity a splat starts at when it arrives fully opaque.
START_OPACITY = 0.1

#: Splats fainter than this are removed.
FAINTEST = 0.005

#: Splats nearer a camera than this share of the scene's depth are left out of its view.
NEAR_SHARE = 0.01

#: Splats reaching further than this share of the picture's longer side are left out of a view.
WIDEST_SHARE = 0.5

#: Steps between removals of faint splats.
PRUNE_EVERY = 250

#: Steps between additions of splats.
GROW_EVERY = 100

#: First step splats may be added after.
GROW_FROM = 200

#: Share of the steps after which no splat is added.
GROW_UNTIL = 0.7

#: Most splats one addition makes, as a share of those held.
GROW_SHARE = 0.05

#: Splats longer than this share of the scene's depth are split in two; shorter ones are copied.
BIG_SHARE = 0.01

#: Factor a split splat's halves shrink by.
SPLIT_SHRINK = 1.6

#: Splats held for every pixel of a picture, past which none are added.
MOST_PER_PIXEL = 4

#: The fields held once per splat.
PER_SPLAT = ("means", "log_scales", "quats", "logit_opacity", "sh_base", "sh_rest")

#: Learning rates per field; positions and camera shifts are multiplied by the scene's depth.
RATES = {
    "means": 1.6e-4, "means_end": 1.6e-6, "log_scales": 5e-3, "quats": 1e-3, "logit_opacity": 5e-2,
    "sh_base": 2.5e-3, "sh_rest": 1.25e-4, "turn": 2e-4, "shift": 2e-4,
}


@dataclass
class Fitted:
    """What a fit made.

    Attributes:
        gaussians: The fitted splats, on the CPU.
        cameras: Each picture's camera, moved when cameras were refined.
        steps: Steps taken.
        seconds: Time the steps took.
        removed: Faint splats removed.
        added: Splats added where the match was worst.
        losses: Mean loss over each tenth of the steps.
        moved: Each camera's distance moved, as a share of the scene's depth.
    """

    gaussians: Gaussians
    cameras: list[dict]
    steps: int = 0
    seconds: float = 0.0
    removed: int = 0
    added: int = 0
    losses: list[float] = field(default_factory=list)
    moved: list[float] = field(default_factory=list)


def shrink(pictures: torch.Tensor, size: int) -> torch.Tensor:
    """Pictures ``(N, H, W, 3)`` scaled so the longer side is at most ``size``."""
    height, width = int(pictures.shape[1]), int(pictures.shape[2])
    scale = min(1.0, float(size) / max(height, width))
    if scale >= 1.0:
        return pictures
    new = (max(16, round(height * scale)), max(16, round(width * scale)))
    return torch.nn.functional.interpolate(pictures.permute(0, 3, 1, 2), size=new, mode="bilinear",
                                           align_corners=False, antialias=True).permute(0, 2, 3, 1).contiguous()


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Peak signal to noise ratio of two pictures in ``[0, 1]``, in decibels."""
    error = float(((a.float() - b.float()) ** 2).mean())
    return 99.0 if error <= 1e-10 else -10.0 * math.log10(error)


def _window(device) -> torch.Tensor:
    """An 11-tap Gaussian window, sigma 1.5, as a ``(3, 1, 11, 11)`` kernel."""
    taps = torch.arange(11, dtype=torch.float32, device=device) - 5.0
    line = torch.exp(-(taps * taps) / (2.0 * 1.5 * 1.5))
    line = line / line.sum()
    return (line[:, None] * line[None, :]).expand(3, 1, 11, 11).contiguous()


def _ssim(a: torch.Tensor, b: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    """Structural similarity ``(H, W)`` of two ``(H, W, 3)`` pictures."""
    x, y = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]

    def blur(t):
        return torch.nn.functional.conv2d(t, window, padding=5, groups=3)

    mx, my = blur(x), blur(y)
    vx, vy, cxy = blur(x * x) - mx * mx, blur(y * y) - my * my, blur(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx * mx + my * my + c1) * (vx + vy + c2)))[0].mean(0)


def _loss(drawn: torch.Tensor, target: torch.Tensor, window: torch.Tensor,
          weight: torch.Tensor | None) -> torch.Tensor:
    """L1 and structural similarity of a view against its picture, each pixel counted by ``weight`` ``(H, W)``."""
    each = (1.0 - SSIM_SHARE) * (drawn - target).abs().mean(-1) + SSIM_SHARE * (1.0 - _ssim(drawn, target, window))
    if weight is None:
        return each.mean()
    return (each * weight).sum() / weight.sum().clamp_min(1.0)


def _turns(vectors: torch.Tensor) -> torch.Tensor:
    """Rotation matrices ``(V, 3, 3)`` of axis-angle vectors ``(V, 3)``."""
    angle2 = (vectors * vectors).sum(-1, keepdim=True) + 1e-12
    angle = torch.sqrt(angle2)
    x, y, z = vectors.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(-1, 3, 3)
    eye = torch.eye(3, device=vectors.device).expand_as(skew)
    a = (torch.sin(angle) / angle)[..., None]
    b = ((1.0 - torch.cos(angle)) / angle2)[..., None]
    return eye + a * skew + b * (skew @ skew)


def _scene_depth(means: torch.Tensor, eye: torch.Tensor, rows: torch.Tensor) -> float:
    """Median depth of the splats ahead of a camera, or 1.0 when none are."""
    sample = means[:: max(1, means.shape[0] // 200_000)]
    ahead = (sample - eye) @ rows[2]
    ahead = ahead[ahead > 0]
    return float(ahead.median()) if ahead.numel() else 1.0


def _swap(optimiser: torch.optim.Optimizer, params: dict, name: str, new: torch.nn.Parameter, moments) -> None:
    """Put ``new`` in place of a parameter, its optimiser moments passed through ``moments``."""
    old = params[name]
    for group in optimiser.param_groups:
        if group["params"][0] is old:
            state = optimiser.state.pop(old, None)
            if state:
                for key in ("exp_avg", "exp_avg_sq"):
                    state[key] = moments(state[key])
                optimiser.state[new] = state
            group["params"][0] = new
    params[name] = new


def _add(optimiser: torch.optim.Optimizer, params: dict, score: torch.Tensor, room: int, depth: float,
         generator: torch.Generator) -> int:
    """Add splats beside the worst matched: long ones split in two, short ones copied.

    Args:
        optimiser: The optimiser holding the splats.
        params: The splats' parameters by name.
        score: ``(N,)`` how hard each splat's centre is pulled across the pictures.
        room: Most splats to add.
        depth: The scene's depth.
        generator: Source of the new splats' offsets.

    Returns:
        Splats added.
    """
    count = int(score.shape[0])
    take = min(int(room), max(1, int(GROW_SHARE * count)), int((score > 0).sum()))
    if take <= 0:
        return 0
    with torch.no_grad():
        pick = torch.topk(score, take).indices
        scales = torch.exp(params["log_scales"][pick])
        big = scales.max(-1).values > BIG_SHARE * depth
        noise = torch.randn(take, 3, generator=generator).to(scales.device)
        offset = (quaternion_matrices(params["quats"][pick]) @ (scales * noise)[..., None])[..., 0]
        opacity = torch.sigmoid(params["logit_opacity"][pick])
        shared = (1.0 - torch.sqrt((1.0 - opacity).clamp_min(1e-8))).clamp(1e-4, 1.0 - 1e-4)
        logit = torch.where(big, params["logit_opacity"][pick], torch.log(shared / (1.0 - shared)))
        log_scales = params["log_scales"][pick] - big[:, None] * math.log(SPLIT_SHRINK)
        new = {
            "means": params["means"][pick] + offset,
            "log_scales": log_scales,
            "quats": params["quats"][pick],
            "logit_opacity": logit,
            "sh_base": params["sh_base"][pick],
            "sh_rest": params["sh_rest"][pick],
        }
        params["means"][pick[big]] -= offset[big]
        params["log_scales"][pick] = log_scales
        params["logit_opacity"][pick] = logit
    for name in PER_SPLAT:
        rows = new[name].detach().clone()
        _swap(optimiser, params, name, torch.nn.Parameter(torch.cat([params[name].detach(), rows])),
              lambda moment, rows=rows: torch.cat([moment, torch.zeros_like(rows)]))
    return take


def _keep(optimiser: torch.optim.Optimizer, params: dict, keep: torch.Tensor) -> None:
    """Drop the splats outside ``keep`` from every per-splat parameter and its optimiser state."""
    for name in PER_SPLAT:
        _swap(optimiser, params, name, torch.nn.Parameter(params[name].detach()[keep].clone()),
              lambda moment: moment[keep].clone())


def fit(start: Gaussians, pictures: torch.Tensor, poses: Sequence[dict], steps: int, degree: int,
        refine_cameras: bool, device, progress: Callable[[float], None] | None = None,
        interrupt: Callable[[], None] | None = None, seed: int = 0, weights: torch.Tensor | None = None,
        grow: float = 0.0, movable: int | None = None) -> Fitted:
    """Fit splats to pictures.

    Args:
        start: The splats to begin from; fully opaque ones start faint.
        pictures: ``(N, H, W, 3)`` pictures, one per camera, at the size to fit at.
        poses: One ``camera_info`` per picture.
        steps: Steps to take, one picture per step.
        degree: Spherical harmonic degree of the fitted colour, 0 to 3.
        refine_cameras: Whether every camera after the first may turn and shift.
        device: Where the work runs.
        progress: Called with the share of steps done.
        interrupt: Called each step; raises to stop.
        seed: Seed for the order pictures are visited in.
        weights: ``(N, H, W)`` how much each pixel of each picture counts, 0 to 1; None = fully.
        grow: Most splats added, as a share of those begun with; none are added past
            :data:`MOST_PER_PIXEL` for every pixel of a picture.
        movable: How many leading cameras may move when refining; the rest are held. None = all.

    Returns:
        The fit.
    """
    with torch.inference_mode(False), torch.enable_grad():
        return _fit(start, pictures, poses, steps, degree, refine_cameras, device, progress, interrupt, seed,
                    weights, grow, movable)


def _fit(start, pictures, poses, steps, degree, refine_cameras, device, progress, interrupt, seed, weights,
         grow, movable=None) -> Fitted:
    """:func:`fit` with gradients on, every input copied out of inference mode."""
    views = int(pictures.shape[0])
    height, width = int(pictures.shape[1]), int(pictures.shape[2])
    targets = pictures.float().to(device).clone()
    counted = weights.float().to(device).clone() if weights is not None else None
    start = Gaussians(*(t.clone() for t in (start.means, start.scales, start.rotations, start.opacities, start.sh)))
    bands = (int(degree) + 1) ** 2
    g = start.to(device)
    sh = g.sh
    if sh.shape[1] < bands:
        sh = torch.nn.functional.pad(sh, (0, 0, 0, bands - sh.shape[1]))
    opacity = torch.where(g.opacities >= 0.999, torch.full_like(g.opacities, START_OPACITY), g.opacities)
    opacity = opacity.clamp(1e-4, 1.0 - 1e-4)
    params = {
        "means": torch.nn.Parameter(g.means.float().clone()),
        "log_scales": torch.nn.Parameter(torch.log(g.scales.float().clamp_min(1e-8))),
        "quats": torch.nn.Parameter(torch.nn.functional.normalize(g.rotations.float(), dim=-1).clone()),
        "logit_opacity": torch.nn.Parameter(torch.log(opacity / (1.0 - opacity))),
        "sh_base": torch.nn.Parameter(sh[:, :1].float().clone()),
        "sh_rest": torch.nn.Parameter(sh[:, 1:bands].float().clone()),
    }
    posed = [cameras.view(pose) for pose in poses]
    eyes = torch.stack([eye.float() for eye, _ in posed]).to(device)
    rows = torch.stack([row.float() for _, row in posed]).to(device)
    depth = _scene_depth(params["means"].detach(), eyes[0], rows[0])
    turn = torch.nn.Parameter(torch.zeros(views, 3, device=device))
    shift = torch.nn.Parameter(torch.zeros(views, 3, device=device))
    free = torch.ones(views, 1, device=device)
    free[0] = 0.0
    if movable is not None:
        free[max(1, int(movable)):] = 0.0
    groups = [
        {"params": [params["means"]], "lr": RATES["means"] * depth, "name": "means"},
        {"params": [params["log_scales"]], "lr": RATES["log_scales"], "name": "log_scales"},
        {"params": [params["quats"]], "lr": RATES["quats"], "name": "quats"},
        {"params": [params["logit_opacity"]], "lr": RATES["logit_opacity"], "name": "logit_opacity"},
        {"params": [params["sh_base"]], "lr": RATES["sh_base"], "name": "sh_base"},
        {"params": [params["sh_rest"]], "lr": RATES["sh_rest"], "name": "sh_rest"},
    ]
    moving = bool(refine_cameras) and views > 1
    if moving:
        groups.append({"params": [turn], "lr": RATES["turn"], "name": "turn"})
        groups.append({"params": [shift], "lr": RATES["shift"] * depth, "name": "shift"})
    optimiser = torch.optim.Adam(groups, eps=1e-15)
    window = _window(device)
    generator = torch.Generator().manual_seed(int(seed))
    order: list[int] = []
    removed = 0
    added = 0
    budget = int(max(0.0, float(grow)) * g.count)
    ceiling = MOST_PER_PIXEL * height * width
    pulled = torch.zeros(g.count, device=device)
    seen = torch.zeros(g.count, device=device)
    losses, bucket = [], []
    tenth = max(1, steps // 10)
    began = time.perf_counter()
    for step in range(int(steps)):
        if interrupt is not None:
            interrupt()
        if not order:
            order = torch.randperm(views, generator=generator).tolist()
        index = order.pop()
        fraction = step / max(1, steps - 1)
        optimiser.param_groups[0]["lr"] = depth * math.exp(
            (1.0 - fraction) * math.log(RATES["means"]) + fraction * math.log(RATES["means_end"]))
        eye, row = eyes[index], rows[index]
        if moving:
            row = _turns(turn[index:index + 1] * free[index])[0] @ row
            eye = eye + shift[index] * free[index]
        current = Gaussians(params["means"], torch.exp(params["log_scales"]), params["quats"],
                            torch.sigmoid(params["logit_opacity"]),
                            torch.cat([params["sh_base"], params["sh_rest"]], 1))
        backdrop = torch.rand(3, generator=generator).to(device)
        view = rasterize(current, poses[index], width, height, backdrop, eye=eye, rows=row, tile=FIT_TILE,
                         near=NEAR_SHARE * depth, widest=WIDEST_SHARE * max(width, height))
        room = min(budget - added, ceiling - params["means"].shape[0])
        growing = (room > 0 and step + 1 <= GROW_UNTIL * steps
                   and view.centres is not None and view.centres.requires_grad)
        if growing:
            view.centres.retain_grad()
        loss = _loss(view.colour, targets[index], window, counted[index] if counted is not None else None)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if growing and view.centres.grad is not None:
            pull = view.centres.grad.norm(dim=-1)
            pulled.index_add_(0, view.ahead, pull)
            seen.index_add_(0, view.ahead, (pull > 0).float())
        bucket.append(float(loss.detach()))
        if len(bucket) >= tenth:
            losses.append(sum(bucket) / len(bucket))
            bucket = []
        if (step + 1) % PRUNE_EVERY == 0 and step + 1 < steps:
            with torch.no_grad():
                keep = torch.sigmoid(params["logit_opacity"]) >= FAINTEST
            if not bool(keep.all()):
                removed += int((~keep).sum())
                _keep(optimiser, params, keep)
                pulled, seen = pulled[keep], seen[keep]
        if growing and step + 1 >= GROW_FROM and (step + 1) % GROW_EVERY == 0:
            added += _add(optimiser, params, pulled / seen.clamp_min(1.0), room, depth, generator)
            pulled = torch.zeros(params["means"].shape[0], device=device)
            seen = torch.zeros_like(pulled)
        if progress is not None:
            progress((step + 1) / steps)
    if bucket:
        losses.append(sum(bucket) / len(bucket))
    seconds = time.perf_counter() - began
    with torch.no_grad():
        opacity = torch.sigmoid(params["logit_opacity"])
        keep = opacity >= FAINTEST
        removed += int((~keep).sum())
        fitted = Gaussians(params["means"].detach(), torch.exp(params["log_scales"].detach()),
                           torch.nn.functional.normalize(params["quats"].detach(), dim=-1), opacity.detach(),
                           torch.cat([params["sh_base"], params["sh_rest"]], 1).detach()).kept(keep).to("cpu")
        moved_cameras, moved = [], []
        for index, pose in enumerate(poses):
            if moving and float(free[index]) > 0.0:
                row = _turns(turn[index:index + 1])[0] @ rows[index]
                eye = eyes[index] + shift[index]
                fov = float(pose.get("fov", cameras.DEFAULT_FOV) or cameras.DEFAULT_FOV)
                moved_pose = dict(pose)
                moved_pose.update(cameras.from_rows(eye.double().cpu(), row.double().cpu(), fov))
                moved_cameras.append(moved_pose)
                moved.append(float(shift[index].norm()) / max(depth, 1e-9))
            else:
                moved_cameras.append(dict(pose))
                moved.append(0.0)
    return Fitted(fitted, moved_cameras, int(steps), seconds, removed, added, losses, moved)

"""What a lens does to a frame: corner falloff, bent straight lines, smeared motion.

Images are ``(batch, height, width, channels)`` floats. Each function answers the shape it
was given; :func:`perspective` answers the size asked for.
"""

from __future__ import annotations

__all__ = [
    "BLURS",
    "EDGES",
    "SHAPES",
    "distorted",
    "perspective",
    "smeared",
    "vignette",
]

import math

import torch
from torch.nn import functional

from . import dynamic

#: How a blur travels across the frame, in menu order.
BLURS = ("linear", "zoom", "spin")

#: What is read past the edge of the frame, in menu order.
EDGES = ("hold the edge", "mirror", "empty")

#: How a vignette's falloff is shaped, in menu order.
SHAPES = ("to the frame", "circular")

#: What each of :data:`EDGES` samples with.
_PADDING = {
    EDGES[0]: "border",
    EDGES[1]: "reflection",
    EDGES[2]: "zeros",
}

#: Below this a division is read as a division by zero.
EPSILON = 1e-6


def _grid(height: int, width: int, device, dtype):
    """Sampling coordinates of every pixel, in the -1 to 1 that ``grid_sample`` reads."""
    ys = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return grid_x, grid_y


def _sample(images: torch.Tensor, grid_x, grid_y, edge: str) -> torch.Tensor:
    """One batch read through a sampling grid."""
    planes = images.permute(0, 3, 1, 2)
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
    if grid.shape[0] != planes.shape[0]:
        grid = grid.expand(planes.shape[0], -1, -1, -1)
    read = functional.grid_sample(
        planes,
        grid,
        mode="bilinear",
        padding_mode=_PADDING.get(edge, "border"),
        align_corners=True,
    )
    return read.permute(0, 2, 3, 1)


def _radius(grid_x, grid_y, width: int, height: int, shape: str):
    """Distance from the centre, 1.0 at the edge the shape reaches."""
    if shape == SHAPES[1] and width and height:
        longest = max(width, height)
        return torch.sqrt(
            (grid_x * width / longest) ** 2 + (grid_y * height / longest) ** 2
        )
    return torch.sqrt(grid_x * grid_x + grid_y * grid_y)


def vignette(
    images: torch.Tensor,
    amount: float = 0.5,
    size: float = 0.75,
    feather: float = 0.5,
    shape: str = SHAPES[0],
    centre_x: float = 0.5,
    centre_y: float = 0.5,
) -> torch.Tensor:
    """Darken or lighten the frame away from a centre.

    Args:
        images: ``(batch, height, width, channels)`` in 0 to 1.
        amount: How far the corners move, -1.0 to 1.0. Positive darkens.
        size: Where the falloff starts, as a fraction of the way out. 1.0 reaches the
            corners.
        feather: How much of the way in the falloff is spread over, 0.0 to 1.0.
        shape: One of :data:`SHAPES`.
        centre_x: Middle of the falloff across the frame, 0.0 to 1.0.
        centre_y: Middle of the falloff down the frame, 0.0 to 1.0.

    Returns:
        A tensor of the shape and dtype it was given.
    """
    if not amount:
        return images
    frames = images.to(dtype=torch.float32)
    height, width = int(frames.shape[1]), int(frames.shape[2])
    grid_x, grid_y = _grid(height, width, frames.device, frames.dtype)
    grid_x = grid_x - (float(centre_x) * 2.0 - 1.0)
    grid_y = grid_y - (float(centre_y) * 2.0 - 1.0)

    reach = _radius(grid_x, grid_y, width, height, shape)
    outer = max(float(size), EPSILON) * math.sqrt(2.0)
    inner = outer * (1.0 - min(max(float(feather), 0.0), 1.0))
    step = ((reach - inner) / max(outer - inner, EPSILON)).clamp(0.0, 1.0)
    # Smoothstep, so the falloff has no visible ring where it begins or ends.
    falloff = step * step * (3.0 - 2.0 * step)

    weight = 1.0 - float(amount) * falloff
    shaded = frames[..., :3] * weight.unsqueeze(0).unsqueeze(-1)
    answer = frames.clone()
    answer[..., :3] = shaded.clamp(0.0, 1.0)
    return answer.to(dtype=images.dtype)


def distorted(
    images: torch.Tensor,
    k1: float = 0.0,
    k2: float = 0.0,
    scale: float = 1.0,
    dispersion: float = 0.0,
    edge: str = EDGES[0],
) -> torch.Tensor:
    """Bend straight lines the way a lens does, or take that bend back out.

    Args:
        images: ``(batch, height, width, channels)``.
        k1: First radial term. Positive pincushions, negative barrels.
        k2: Second radial term, which acts furthest from the centre.
        scale: Zoom applied with the bend, which fills the corners a barrel empties.
        dispersion: How far the red and blue channels are scaled apart, as a fraction.
        edge: One of :data:`EDGES`.

    Returns:
        A tensor of the shape and dtype it was given.
    """
    if not k1 and not k2 and scale == 1.0 and not dispersion:
        return images
    frames = images.to(dtype=torch.float32)
    height, width = int(frames.shape[1]), int(frames.shape[2])
    grid_x, grid_y = _grid(height, width, frames.device, frames.dtype)
    squared = grid_x * grid_x + grid_y * grid_y

    def bent(spread: float):
        factor = 1.0 + float(k1) * squared + float(k2) * squared * squared
        factor = factor / max(float(scale) * (1.0 + spread), EPSILON)
        return grid_x * factor, grid_y * factor

    if not dispersion or int(frames.shape[3]) < 3:
        moved_x, moved_y = bent(0.0)
        return _sample(frames, moved_x, moved_y, edge).to(dtype=images.dtype)

    channels = []
    for index, spread in enumerate((float(dispersion), 0.0, -float(dispersion))):
        moved_x, moved_y = bent(spread)
        plane = frames[..., index : index + 1]
        channels.append(_sample(plane, moved_x, moved_y, edge))
    answer = torch.cat(channels, dim=-1)
    if int(frames.shape[3]) > 3:
        moved_x, moved_y = bent(0.0)
        answer = torch.cat([answer, _sample(frames[..., 3:], moved_x, moved_y, edge)], dim=-1)
    return answer.to(dtype=images.dtype)


def perspective(
    images: torch.Tensor,
    corners,
    width: int = 0,
    height: int = 0,
    edge: str = EDGES[0],
    out: torch.Tensor | None = None,
    peak: float | None = None,
) -> torch.Tensor:
    """Pin the four corners of a frame to four new places.

    Args:
        images: ``(batch, height, width, channels)``.
        corners: Four ``(x, y)`` pairs in pixels, clockwise from the top left, saying where
            the source frame's corners land in the answer.
        width: Answer width in pixels, 0 to keep the source's.
        height: Answer height in pixels, 0 to keep the source's.
        edge: One of :data:`EDGES`.
        out: A ``(batch, height, width, channels)`` tensor each answered frame is written
            into. None allocates one on the device of ``images``.
        peak: Scale each frame is brought inside 0 to 1 by with :func:`dynamic.fold` before
            it is read, and its answer multiplied back up by. None reads frames as they are.

    Returns:
        ``out``, or a new ``(batch, height, width, channels)`` tensor, in the dtype of
        ``images``.

    Raises:
        ValueError: Fewer than four corners, the four are collinear so no mapping exists,
            or ``out`` is not shaped like the answer.
    """
    if len(corners) < 4:
        raise ValueError(f"perspective needs four corners, got {len(corners)}.")
    batch, source_h, source_w, channels = (int(side) for side in images.shape)
    out_w = int(width) or source_w
    out_h = int(height) or source_h
    shape = (batch, out_h, out_w, channels)
    if out is None:
        out = torch.empty(shape, dtype=images.dtype, device=images.device)
    elif tuple(out.shape) != shape:
        raise ValueError(
            f"perspective() was given an out tensor shaped {tuple(out.shape)} for an answer "
            f"shaped {shape}"
        )

    grid = _pinned_grid(corners, source_w, source_h, out_w, out_h, images.device)
    for index in range(batch):
        frame = images[index:index + 1]
        folded = dynamic.fold(frame, peak) if peak is not None else None
        if folded is not None:
            frame = folded.images
        read = functional.grid_sample(
            frame.to(dtype=torch.float32).permute(0, 3, 1, 2),
            grid,
            mode="bilinear",
            padding_mode=_PADDING.get(edge, "border"),
            align_corners=True,
        )
        out[index].copy_(read[0].permute(1, 2, 0))
        if folded is not None and folded.scale != 1.0:
            out[index].mul_(folded.scale)
    return out


def _pinned_grid(corners, source_w: int, source_h: int, out_w: int, out_h: int, device):
    """Where each answer pixel of :func:`perspective` reads the source frame.

    Args:
        corners: Four ``(x, y)`` pairs in pixels, clockwise from the top left.
        source_w: Source width in pixels.
        source_h: Source height in pixels.
        out_w: Answer width in pixels.
        out_h: Answer height in pixels.
        device: Device the grid is built on.

    Returns:
        A ``(1, out_h, out_w, 2)`` float32 grid in the -1 to 1 that ``grid_sample`` reads.

    Raises:
        ValueError: The four corners are collinear, so no mapping exists.
    """
    source = [(0.0, 0.0), (source_w - 1.0, 0.0),
              (source_w - 1.0, source_h - 1.0), (0.0, source_h - 1.0)]
    matrix = _homography([(float(x), float(y)) for x, y in corners[:4]], source,
                         device).to(dtype=torch.float32)

    ys = torch.arange(out_h, device=device, dtype=torch.float32)
    xs = torch.arange(out_w, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    ones = torch.ones_like(grid_x)
    stacked = torch.stack([grid_x, grid_y, ones], dim=-1)
    mapped = stacked @ matrix.transpose(0, 1)
    depth = mapped[..., 2:3]
    depth = torch.where(depth.abs() < EPSILON, torch.full_like(depth, EPSILON), depth)
    flat = mapped[..., :2] / depth

    read_x = flat[..., 0] / max(source_w - 1.0, EPSILON) * 2.0 - 1.0
    read_y = flat[..., 1] / max(source_h - 1.0, EPSILON) * 2.0 - 1.0
    return torch.stack([read_x, read_y], dim=-1).unsqueeze(0)


def _homography(source, target, device):
    """The 3x3 mapping taking four ``source`` points onto four ``target`` points.

    Args:
        source: Four ``(x, y)`` pairs the mapping reads from.
        target: Four ``(x, y)`` pairs the mapping answers.
        device: Device the matrix is built on.

    Returns:
        A ``(3, 3)`` float64 tensor whose last entry is 1.

    Raises:
        ValueError: The eight equations have no solution, which means three of the four
            points lie on one line.
    """
    rows, answers = [], []
    for (sx, sy), (tx, ty) in zip(source, target):
        rows.append([sx, sy, 1.0, 0.0, 0.0, 0.0, -sx * tx, -sy * tx])
        answers.append(tx)
        rows.append([0.0, 0.0, 0.0, sx, sy, 1.0, -sx * ty, -sy * ty])
        answers.append(ty)
    system = torch.tensor(rows, dtype=torch.float64, device=device)
    wanted = torch.tensor(answers, dtype=torch.float64, device=device).unsqueeze(-1)
    try:
        solved = torch.linalg.solve(system, wanted).reshape(-1)
    except RuntimeError as flat:
        raise ValueError(
            "the four corners do not describe a shape: three of them lie on one line, so "
            "there is no perspective that maps the frame onto them"
        ) from flat
    return torch.cat([solved, torch.ones(1, dtype=torch.float64, device=device)]).reshape(3, 3)


def smeared(
    images: torch.Tensor,
    blur: str = BLURS[0],
    length: float = 0.05,
    angle: float = 0.0,
    taps: int = 16,
    centre_x: float = 0.5,
    centre_y: float = 0.5,
    edge: str = EDGES[0],
) -> torch.Tensor:
    """Average a frame along a path, the way a moving camera does.

    Args:
        images: ``(batch, height, width, channels)``.
        blur: One of :data:`BLURS`.
        length: How far the smear travels. A fraction of the frame on ``linear``, a
            fraction of the distance to the centre on ``zoom``, and turns on ``spin``.
        angle: Direction of a ``linear`` smear in degrees, 0 pointing right.
        taps: How many samples are averaged. More is smoother and slower.
        centre_x: Where ``zoom`` and ``spin`` turn about, across the frame, 0.0 to 1.0.
        centre_y: Where ``zoom`` and ``spin`` turn about, down the frame, 0.0 to 1.0.
        edge: One of :data:`EDGES`.

    Returns:
        A tensor of the shape and dtype it was given.
    """
    steps = max(1, int(taps))
    if not length or steps == 1:
        return images
    frames = images.to(dtype=torch.float32)
    height, width = int(frames.shape[1]), int(frames.shape[2])
    grid_x, grid_y = _grid(height, width, frames.device, frames.dtype)
    from_x = grid_x - (float(centre_x) * 2.0 - 1.0)
    from_y = grid_y - (float(centre_y) * 2.0 - 1.0)

    radians = math.radians(float(angle))
    across, down = math.cos(radians), math.sin(radians)
    total = None
    for index in range(steps):
        walk = index / (steps - 1) - 0.5
        if blur == BLURS[1]:
            factor = 1.0 + float(length) * walk * 2.0
            read_x = grid_x + from_x * (factor - 1.0)
            read_y = grid_y + from_y * (factor - 1.0)
        elif blur == BLURS[2]:
            turn = math.radians(float(length) * walk * 360.0)
            cos, sin = math.cos(turn), math.sin(turn)
            read_x = grid_x + (from_x * cos - from_y * sin) - from_x
            read_y = grid_y + (from_x * sin + from_y * cos) - from_y
        else:
            read_x = grid_x + across * float(length) * walk * 2.0
            read_y = grid_y + down * float(length) * walk * 2.0
        read = _sample(frames, read_x, read_y, edge)
        total = read if total is None else total + read
    return (total / steps).to(dtype=images.dtype)

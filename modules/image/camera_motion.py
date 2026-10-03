"""The camera's own motion through a clip, fitted from its measured motion, and steadying it.

Transforms are 3 by 3 matrices on pixel coordinates centred on the frame, in pixels of the
measured motion unless :func:`to_frame` has scaled them.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from . import optical_flow

__all__ = [
    "BORDERS",
    "CHANGE_LEVELS",
    "MODELS",
    "MODES",
    "change_after_camera",
    "corrections",
    "background",
    "fit",
    "moving_speed",
    "path",
    "resample",
    "shake",
    "smooth",
    "to_frame",
    "zoom_for",
]

#: What the camera's motion is fitted as.
MODELS = ("translation", "similarity")

#: How a stabilized camera moves: along its own path smoothed, or not at all.
MODES = ("smooth", "lock")

#: What fills the edge a steadied frame pulls away from.
BORDERS = ("zoom", "edge", "mirror", "black")

#: Spacing, in pixels of the measured motion, of the points a fit reads.
FIT_STRIDE = 4

#: Reweighting rounds of the robust fit.
FIT_ROUNDS = 6

#: Residual, in pixels of the measured motion, at which a point's weight halves.
FIT_SCALE = 1.5

#: Frequencies, as multiples of the chosen one, a shake is woven from; each is weighted by its
#: inverse, as handheld motion falls off.
SHAKE_PARTIALS = (0.33, 0.55, 1.0, 1.7, 2.9, 4.8)

#: Weight the fit gives the frame's centre against its edges, where the background usually is.
CENTRE_WEIGHT = 0.25

#: Luminance change, in levels of 255, left after the camera's motion is taken out at which a
#: pixel starts to count as changed, and the span over which it becomes certain.
CHANGE_LEVELS = (4.0, 8.0)

#: Pixels of the measured motion a change is widened by, to cover a moving surface's flat middle.
CHANGE_REACH = 4

#: Control points across and down the smooth field a background's motion is fitted with.
FIELD_GRID = (10, 6)

#: Weight of the field's curvature against how well it follows the motion.
FIELD_BEND = 0.01

#: Residual, in pixels of the measured motion, at which a point's weight halves in the field fit.
FIELD_SCALE = 1.0

#: How finely :func:`zoom_for` settles on a zoom.
ZOOM_STEPS = 24


def _grid(height: int, width: int, device):
    """Centred coordinates of every :data:`FIT_STRIDE`-th pixel, ``(2, n)``, and their indices."""
    ys = torch.arange(FIT_STRIDE // 2, height, FIT_STRIDE, device=device)
    xs = torch.arange(FIT_STRIDE // 2, width, FIT_STRIDE, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    points = torch.stack([xx.reshape(-1) - (width - 1) / 2.0, yy.reshape(-1) - (height - 1) / 2.0]).float()
    return points, yy.reshape(-1), xx.reshape(-1)


def _weighted_median(values, weights) -> float:
    """The weighted median of a 1D tensor."""
    order = torch.argsort(values)
    running = torch.cumsum(weights[order], 0)
    at = int(torch.searchsorted(running, running[-1] * 0.5).clamp(max=values.numel() - 1))
    return float(values[order][at])


def fit(flow, weight=None, model: str = "similarity"):
    """The camera motion that best explains a flow, ignoring what moves on its own.

    Args:
        flow: ``(1, 2, h, w)`` in pixels.
        weight: Optional ``(1, 1, h, w)`` trust per pixel, 0 to 1.
        model: One of :data:`MODELS`.

    Returns:
        A ``(3, 3)`` float64 matrix taking centred coordinates in this frame to the next.
    """
    _, _, height, width = flow.shape
    points, rows, columns = _grid(height, width, flow.device)
    moved = flow[0][:, rows, columns].float()
    base = torch.ones(points.shape[1], device=flow.device)
    if weight is not None:
        base = weight[0, 0][rows, columns].float().clamp(0.0, 1.0)
    x, y = points[0], points[1]
    radius = torch.sqrt((x / max(width / 2.0, 1.0)) ** 2 + (y / max(height / 2.0, 1.0)) ** 2)
    base = base * (CENTRE_WEIGHT + (1.0 - CENTRE_WEIGHT) * (radius / max(float(radius.max()), 1e-6)) ** 2)
    matrix = torch.eye(3, dtype=torch.float64)
    if float(base.sum()) < 2.0:
        return matrix
    u, v = moved[0], moved[1]
    # The first pass weighs each point by its distance from the median shift.
    start_u, start_v = _weighted_median(u, base), _weighted_median(v, base)
    start = torch.sqrt((u - start_u) ** 2 + (v - start_v) ** 2)
    weights = base / (1.0 + (start / FIT_SCALE) ** 2)
    for _ in range(FIT_ROUNDS):
        total = weights.sum().clamp(min=1e-6)
        if model == "translation":
            tx = float((weights * u).sum() / total)
            ty = float((weights * v).sum() / total)
            a, b = 1.0, 0.0
        else:
            # Least squares for x' = a x - b y + tx, y' = b x + a y + ty.
            xt, yt = x + u, y + v
            rows_a = torch.stack([
                torch.stack([x, -y, torch.ones_like(x), torch.zeros_like(x)], 1),
                torch.stack([y, x, torch.zeros_like(x), torch.ones_like(x)], 1),
            ]).reshape(-1, 4).double()
            targets = torch.cat([xt, yt]).double()
            w2 = torch.cat([weights, weights]).double()
            normal = rows_a.T @ (rows_a * w2[:, None])
            right = rows_a.T @ (targets * w2)
            try:
                a, b, tx, ty = (float(value) for value in torch.linalg.solve(
                    normal + 1e-6 * torch.eye(4, dtype=torch.float64, device=normal.device), right
                ))
            except torch.linalg.LinAlgError:
                return matrix
        predicted_u = a * x - b * y + tx - x
        predicted_v = b * x + a * y + ty - y
        residual = torch.sqrt((u - predicted_u) ** 2 + (v - predicted_v) ** 2)
        weights = base / (1.0 + (residual / FIT_SCALE) ** 2)
    matrix[0, 0], matrix[0, 1], matrix[0, 2] = a, -b, tx
    matrix[1, 0], matrix[1, 1], matrix[1, 2] = b, a, ty
    return matrix


def path(motion, model: str = "similarity", device=None, progress=None, subject=None):
    """The camera's position through the clip, restarting at every cut.

    Args:
        motion: A :class:`~.motion.Motion`.
        model: One of :data:`MODELS`.
        device: Where the fits run.
        progress: Optional callable taking a step count, called once per pair.
        subject: Optional ``(frames, height, width)`` mask, 1 on what the fit leaves out.

    Returns:
        ``(cameras, segments)``: one ``(3, 3)`` matrix per frame taking the coordinates of its
        scene's first frame to its own, and the ``(start, stop)`` frames of each scene.
    """
    device = motion.forward.device if device is None else torch.device(device)
    count = motion.count
    cameras = [torch.eye(3, dtype=torch.float64)]
    segments = []
    start = 0
    for index in range(count - 1):
        found = motion.toward(index, 1, device)
        if found is None:
            segments.append((start, index + 1))
            start = index + 1
            cameras.append(torch.eye(3, dtype=torch.float64))
        else:
            flow, seen = found
            if subject is not None:
                plane = subject[min(index, subject.shape[0] - 1)].to(device=flow.device, dtype=torch.float32)
                plane = F.interpolate(plane.view(1, 1, *plane.shape[-2:]), size=motion.size, mode="bilinear", align_corners=False)
                seen = seen * (1.0 - plane.clamp(0.0, 1.0))
            cameras.append(fit(flow, seen, model) @ cameras[-1])
        if progress is not None:
            progress(1)
    segments.append((start, count))
    return cameras, segments


def _parameters(matrix):
    """``(tx, ty, angle, log scale)`` of a similarity matrix."""
    a, b = float(matrix[0, 0]), float(matrix[1, 0])
    scale = max(math.hypot(a, b), 1e-9)
    return [float(matrix[0, 2]), float(matrix[1, 2]), math.atan2(b, a), math.log(scale)]


def _matrix(tx, ty, angle, log_scale):
    """The similarity matrix of ``(tx, ty, angle, log scale)``."""
    scale = math.exp(log_scale)
    a, b = scale * math.cos(angle), scale * math.sin(angle)
    return torch.tensor([[a, -b, tx], [b, a, ty], [0.0, 0.0, 1.0]], dtype=torch.float64)


def smooth(values, sigma: float):
    """A Gaussian-smoothed copy of a list of floats, its ends point-reflected."""
    if sigma <= 0 or len(values) < 2:
        return list(values)
    radius = max(1, int(math.ceil(3.0 * sigma)))
    kernel = [math.exp(-0.5 * (k / sigma) ** 2) for k in range(-radius, radius + 1)]
    total = sum(kernel)
    count = len(values)
    first, last = values[0], values[-1]

    def at(position):
        if position < 0:
            return 2.0 * first - values[min(-position, count - 1)]
        if position >= count:
            return 2.0 * last - values[max(2 * (count - 1) - position, 0)]
        return values[position]

    return [
        sum(weight * at(index + offset) for offset, weight in zip(range(-radius, radius + 1), kernel)) / total
        for index in range(count)
    ]


def corrections(cameras, segments, mode: str = "smooth", smoothing_frames: float = 24.0):
    """Per frame, the matrix taking a steadied frame's coordinates to the source frame's.

    Args:
        cameras: What :func:`path` answered.
        segments: Its scenes.
        mode: One of :data:`MODES`.
        smoothing_frames: Gaussian sigma, in frames, of the smoothed path.

    Returns:
        One ``(3, 3)`` matrix per frame.
    """
    out = [None] * len(cameras)
    for start, stop in segments:
        params = [_parameters(cameras[index]) for index in range(start, stop)]
        angles = [p[2] for p in params]
        for k in range(1, len(angles)):
            while angles[k] - angles[k - 1] > math.pi:
                angles[k] -= 2 * math.pi
            while angles[k] - angles[k - 1] < -math.pi:
                angles[k] += 2 * math.pi
        for k, angle in enumerate(angles):
            params[k][2] = angle
        if mode == "lock":
            steady = [[0.0, 0.0, 0.0, 0.0] for _ in params]
        else:
            columns = [smooth([p[i] for p in params], smoothing_frames) for i in range(4)]
            steady = [[columns[i][k] for i in range(4)] for k in range(len(params))]
        for k, index in enumerate(range(start, stop)):
            out[index] = cameras[index] @ torch.linalg.inv(_matrix(*steady[k]))
    return out


def to_frame(matrix, measured, frame):
    """A matrix on measured coordinates rescaled to a frame's own pixels.

    Args:
        matrix: ``(3, 3)`` on centred coordinates of the measured size.
        measured: ``(h, w)`` the motion was measured at.
        frame: ``(height, width)`` of the frame.

    Returns:
        The ``(3, 3)`` matrix on the frame's centred coordinates.
    """
    sy, sx = frame[0] / measured[0], frame[1] / measured[1]
    scale = torch.diag(torch.tensor([sx, sy, 1.0], dtype=torch.float64))
    return scale @ matrix @ torch.linalg.inv(scale)


def zoom_for(matrices, frame, limit: float):
    """The least zoom that keeps every steadied frame's corners inside its source.

    Args:
        matrices: Per-frame matrices from :func:`corrections`, on the frame's coordinates.
        frame: ``(height, width)``.
        limit: The most zoom allowed.

    Returns:
        A zoom of at least 1, at most ``limit``.
    """
    height, width = frame
    half_w, half_h = (width - 1) / 2.0, (height - 1) / 2.0
    corners = torch.tensor(
        [[-half_w, -half_h, 1.0], [half_w, -half_h, 1.0], [half_w, half_h, 1.0], [-half_w, half_h, 1.0]],
        dtype=torch.float64,
    ).T

    def inside(zoom):
        scaled = corners.clone()
        scaled[:2] /= zoom
        for matrix in matrices:
            landed = matrix @ scaled
            if bool((landed[0].abs() > half_w + 0.5).any() or (landed[1].abs() > half_h + 0.5).any()):
                return False
        return True

    if inside(1.0):
        return 1.0
    if not inside(limit):
        return float(limit)
    low, high = 1.0, float(limit)
    for _ in range(ZOOM_STEPS):
        middle = 0.5 * (low + high)
        if inside(middle):
            high = middle
        else:
            low = middle
    return high


def resample(frame, matrix, zoom: float = 1.0, border: str = "edge"):
    """A frame drawn through a matrix taking output coordinates to its own.

    Args:
        frame: ``(1, channels, height, width)``.
        matrix: ``(3, 3)`` on the frame's centred coordinates.
        zoom: How far to enlarge about the centre.
        border: One of :data:`BORDERS`; what fills where the source runs out.

    Returns:
        The resampled frame, shaped like ``frame``.
    """
    _, _, height, width = frame.shape
    device = frame.device
    scale = torch.diag(torch.tensor([1.0 / zoom, 1.0 / zoom, 1.0], dtype=torch.float64))
    full = (matrix @ scale).to(device=device, dtype=torch.float32)
    ys = torch.arange(height, device=device, dtype=torch.float32) - (height - 1) / 2.0
    xs = torch.arange(width, device=device, dtype=torch.float32) - (width - 1) / 2.0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    sx = full[0, 0] * xx + full[0, 1] * yy + full[0, 2]
    sy = full[1, 0] * xx + full[1, 1] * yy + full[1, 2]
    grid = torch.stack([sx / max((width - 1) / 2.0, 1e-6), sy / max((height - 1) / 2.0, 1e-6)], -1)
    padding = {"black": "zeros", "mirror": "reflection"}.get(border, "border")
    return F.grid_sample(frame, grid.unsqueeze(0), mode="bilinear", padding_mode=padding, align_corners=True)


def _bend(columns: int, rows: int, device):
    """Second differences along every row and column of a control grid, ``(k, rows * columns)``."""
    lines = []
    for j in range(rows):
        for i in range(1, columns - 1):
            line = torch.zeros(rows * columns, device=device, dtype=torch.float64)
            line[j * columns + i - 1], line[j * columns + i], line[j * columns + i + 1] = 1.0, -2.0, 1.0
            lines.append(line)
    for i in range(columns):
        for j in range(1, rows - 1):
            line = torch.zeros(rows * columns, device=device, dtype=torch.float64)
            line[(j - 1) * columns + i], line[j * columns + i], line[(j + 1) * columns + i] = 1.0, -2.0, 1.0
            lines.append(line)
    return torch.stack(lines) if lines else torch.zeros(0, rows * columns, device=device, dtype=torch.float64)


def _basis(rows_at, columns_at, height: int, width: int, columns: int, rows: int):
    """Bilinear weights of every control point at each pixel listed, ``(n, rows * columns)``."""
    fx = columns_at.double() * (columns - 1) / max(width - 1, 1)
    fy = rows_at.double() * (rows - 1) / max(height - 1, 1)
    i0 = fx.floor().clamp(0, columns - 2).long()
    j0 = fy.floor().clamp(0, rows - 2).long()
    tx, ty = fx - i0, fy - j0
    basis = torch.zeros(fx.numel(), rows * columns, device=fx.device, dtype=torch.float64)
    picks = torch.arange(fx.numel(), device=fx.device)
    basis[picks, j0 * columns + i0] += (1 - tx) * (1 - ty)
    basis[picks, j0 * columns + i0 + 1] += tx * (1 - ty)
    basis[picks, (j0 + 1) * columns + i0] += (1 - tx) * ty
    basis[picks, (j0 + 1) * columns + i0 + 1] += tx * ty
    return basis


def background(flow, weight=None):
    """The background's own motion: a smooth field fitted to a flow, leaving out what moves apart.

    Args:
        flow: ``(1, 2, h, w)`` in pixels.
        weight: Optional ``(1, 1, h, w)`` trust per pixel, 0 to 1.

    Returns:
        ``(1, 2, h, w)``: the motion the scene would have with nothing moving in it, parallax
        across receding surfaces included.
    """
    _, _, height, width = flow.shape
    columns, rows = FIELD_GRID
    points, rows_at, columns_at = _grid(height, width, flow.device)
    u = flow[0, 0][rows_at, columns_at].double()
    v = flow[0, 1][rows_at, columns_at].double()
    base = torch.ones_like(u)
    if weight is not None:
        base = weight[0, 0][rows_at, columns_at].double().clamp(0.0, 1.0)
    camera = _camera_field(fit(flow, weight, "similarity"), height, width, flow.device)
    start = torch.sqrt((u - camera[0, 0][rows_at, columns_at]) ** 2 + (v - camera[0, 1][rows_at, columns_at]) ** 2)
    weights = base / (1.0 + (start / (3.0 * FIELD_SCALE)) ** 2)
    basis = _basis(rows_at, columns_at, height, width, columns, rows)
    bend = _bend(columns, rows, flow.device)
    grid_u = grid_v = None
    for _ in range(FIT_ROUNDS):
        weighted = basis * weights[:, None]
        normal = basis.T @ weighted
        normal = normal + FIELD_BEND * float(weights.sum()) * (bend.T @ bend) + 1e-6 * torch.eye(
            rows * columns, device=flow.device, dtype=torch.float64
        )
        try:
            grid_u = torch.linalg.solve(normal, weighted.T @ u)
            grid_v = torch.linalg.solve(normal, weighted.T @ v)
        except torch.linalg.LinAlgError:
            return camera
        residual = torch.sqrt((u - basis @ grid_u) ** 2 + (v - basis @ grid_v) ** 2)
        weights = base / (1.0 + (residual / FIELD_SCALE) ** 2)
    control = torch.stack([grid_u.view(rows, columns), grid_v.view(rows, columns)]).unsqueeze(0).float()
    return F.interpolate(control, size=(height, width), mode="bilinear", align_corners=True)


def moving_speed(motion, index: int, frame, ignore_camera: bool = True, device=None):
    """How fast each pixel moves on its own, and how fast the background under it moves.

    Args:
        motion: A :class:`~.motion.Motion`.
        index: The frame.
        frame: ``(height, width)`` to answer at.
        ignore_camera: Take the background's own motion away first, the camera's movement and
            its parallax.
        device: Where the work runs.

    Returns:
        ``(own, background)``, each ``(1, 1, height, width)`` in pixels per frame of the
        frame's own size; ``background`` is zero when ``ignore_camera`` is off.
    """
    a, _, hidden = motion.paths(index, device)
    height, width = frame
    if ignore_camera:
        under = background(a, 1.0 - hidden)
        own = optical_flow.resize_flow(a - under, height, width)
        behind = optical_flow.resize_flow(under, height, width)
        return own.norm(dim=1, keepdim=True), behind.norm(dim=1, keepdim=True)
    own = optical_flow.resize_flow(a, height, width).norm(dim=1, keepdim=True)
    return own, torch.zeros_like(own)


def _camera_field(matrix, height: int, width: int, device):
    """The flow a camera matrix alone gives, ``(1, 2, height, width)``."""
    camera = matrix.to(device=device, dtype=torch.float32)
    ys = torch.arange(height, device=device, dtype=torch.float32) - (height - 1) / 2.0
    xs = torch.arange(width, device=device, dtype=torch.float32) - (width - 1) / 2.0
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    cu = camera[0, 0] * xx + camera[0, 1] * yy + camera[0, 2] - xx
    cv = camera[1, 0] * xx + camera[1, 1] * yy + camera[1, 2] - yy
    return torch.stack([cu, cv]).unsqueeze(0)


def change_after_camera(motion, index: int, ignore_camera: bool = True, device=None):
    """How surely each pixel changes once the camera's own motion is taken out.

    Args:
        motion: A :class:`~.motion.Motion`.
        index: The frame.
        ignore_camera: Take the camera's motion out before comparing; off compares the frames
            as they stand.
        device: Where the work runs.

    Returns:
        ``(1, 1, h, w)`` in ``[0, 1]`` at the measured size, or ``None`` where the frame has no
        neighbour across which to compare.
    """
    for step in (1, -1):
        found = motion.toward(index, step, device)
        if found is not None:
            break
    else:
        return None
    flow, seen = found
    height, width = motion.size
    here = motion.luma[index:index + 1].to(device=device, dtype=torch.float32)
    there = motion.luma[index + step:index + step + 1].to(device=device, dtype=torch.float32)
    camera = background(flow, seen) if ignore_camera else torch.zeros_like(flow)
    gap = optical_flow.gaussian((optical_flow.warp(there, camera) - here).abs(), 1.0)
    start, span = CHANGE_LEVELS
    sure = ((gap - start) / span).clamp(0.0, 1.0)
    return F.max_pool2d(sure, 2 * CHANGE_REACH + 1, stride=1, padding=CHANGE_REACH)


def shake(times, short_side: float, amplitude: float, rotation: float, zoom: float,
          frequency: float, seed: int, strength=None):
    """Handheld camera shake at the given moments, as matrices taking output to source coordinates.

    Args:
        times: Moments, in seconds.
        short_side: The frame's shorter side, in pixels.
        amplitude: Sway, as a share of ``short_side``; the root mean square of each axis.
        rotation: Roll, in degrees, root mean square.
        zoom: Breathing, as a share of the frame's size, root mean square.
        frequency: The shake's centre frequency in hertz.
        seed: Which shake; the same seed gives the same one.
        strength: Optional multiplier per moment, as long as ``times``.

    Returns:
        One ``(3, 3)`` float64 matrix per moment, on centred coordinates.
    """
    generator = torch.Generator().manual_seed(int(seed))
    weights = torch.tensor([1.0 / k for k in SHAKE_PARTIALS], dtype=torch.float64)
    weights = weights / torch.sqrt((weights ** 2).sum() / 2.0)

    def axis():
        phases = torch.rand(len(SHAKE_PARTIALS), generator=generator, dtype=torch.float64) * 2 * math.pi
        detune = 1.0 + 0.15 * (torch.rand(len(SHAKE_PARTIALS), generator=generator, dtype=torch.float64) - 0.5)
        rates = torch.tensor(SHAKE_PARTIALS, dtype=torch.float64) * float(frequency) * detune
        moments = torch.as_tensor(list(times), dtype=torch.float64)
        return (weights[None, :] * torch.sin(2 * math.pi * rates[None, :] * moments[:, None] + phases[None, :])).sum(1)

    sway_x, sway_y, roll, breathe = axis(), axis(), axis(), axis()
    scale = torch.ones_like(sway_x) if strength is None else torch.as_tensor(list(strength), dtype=torch.float64)
    out = []
    for k in range(len(sway_x)):
        dx = float(amplitude) * short_side * float(sway_x[k]) * float(scale[k])
        dy = float(amplitude) * short_side * float(sway_y[k]) * float(scale[k])
        angle = math.radians(float(rotation) * float(roll[k]) * float(scale[k]))
        size = 1.0 + float(zoom) * float(breathe[k]) * float(scale[k])
        a, b = size * math.cos(angle), size * math.sin(angle)
        out.append(torch.tensor([[a, -b, dx], [b, a, dy], [0.0, 0.0, 1.0]], dtype=torch.float64))
    return out

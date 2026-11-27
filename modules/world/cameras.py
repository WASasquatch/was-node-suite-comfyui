"""Cameras as core's ``camera_info`` dictionaries: projection, poses, preset moves and resampling.

World space is right-handed, Y-up, cameras looking down -Z. The splat frame is world
``(x, -y, -z)``. A field of view spans the picture's shorter side.
"""

from __future__ import annotations

__all__ = [
    "EASINGS",
    "CONTINUE",
    "FOLLOW",
    "MOVES",
    "WIRED",
    "along",
    "turn_direction",
    "upright",
    "as_list",
    "camera",
    "ease",
    "focal",
    "follow",
    "fov_for_focal",
    "from_rows",
    "move",
    "pose",
    "resample",
    "source",
    "view",
]

import math
from collections.abc import Callable, Mapping, Sequence

import torch

#: Splat frame from world: a half turn about X.
FLIP = (1.0, -1.0, -1.0)

#: Field of view, in degrees, a camera carrying none is drawn with.
DEFAULT_FOV = 35.0


def as_list(camera_info) -> list[dict]:
    """One camera or a list of them, as a list.

    Args:
        camera_info: A ``camera_info`` dictionary, a list of them, or None.

    Returns:
        The cameras in order; empty for None.
    """
    if camera_info is None:
        return []
    if isinstance(camera_info, Mapping):
        return [dict(camera_info)]
    return [dict(item) for item in camera_info if isinstance(item, Mapping)]


def _xyz(value, fallback=(0.0, 0.0, 0.0)) -> list[float]:
    """An ``{x, y, z}`` mapping as three floats."""
    if not isinstance(value, Mapping):
        return list(fallback)
    return [float(value.get(axis, default) or 0.0) for axis, default in zip("xyz", fallback)]


def _normal(vector: torch.Tensor) -> torch.Tensor:
    """A vector scaled to unit length."""
    return vector / vector.norm().clamp_min(1e-12)


def quaternion_to_matrix(x: float, y: float, z: float, w: float) -> torch.Tensor:
    """The rotation matrix of a quaternion, normalised first.

    Returns:
        A float64 ``(3, 3)`` matrix whose columns are the rotated X, Y and Z axes.
    """
    norm = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return torch.tensor([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=torch.float64)


def matrix_to_quaternion(matrix: torch.Tensor) -> tuple[float, float, float, float]:
    """The unit quaternion of a rotation matrix.

    Returns:
        ``(x, y, z, w)``.
    """
    m = matrix.double()
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w, x = 0.25 * s, float(m[2, 1] - m[1, 2]) / s
        y, z = float(m[0, 2] - m[2, 0]) / s, float(m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + float(m[0, 0] - m[1, 1] - m[2, 2])) * 2
        w, x = float(m[2, 1] - m[1, 2]) / s, 0.25 * s
        y, z = float(m[0, 1] + m[1, 0]) / s, float(m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + float(m[1, 1] - m[0, 0] - m[2, 2])) * 2
        w, x = float(m[0, 2] - m[2, 0]) / s, float(m[0, 1] + m[1, 0]) / s
        y, z = 0.25 * s, float(m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + float(m[2, 2] - m[0, 0] - m[1, 1])) * 2
        w, x = float(m[1, 0] - m[0, 1]) / s, float(m[0, 2] + m[2, 0]) / s
        y, z = float(m[1, 2] + m[2, 1]) / s, 0.25 * s
    norm = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    return x / norm, y / norm, z / norm, w / norm


def pose(camera_info: Mapping) -> tuple[torch.Tensor, torch.Tensor]:
    """Where a camera stands and how it is turned, in world space.

    Args:
        camera_info: One camera. A quaternion is read when present, else the camera looks from
            ``position`` at ``target`` with world +Y up.

    Returns:
        ``(position (3,), rotation (3, 3))`` in float64; the rotation's columns are the camera's
        right, up and back axes.
    """
    position = torch.tensor(_xyz(camera_info.get("position")), dtype=torch.float64)
    turn = camera_info.get("quaternion")
    if isinstance(turn, Mapping):
        rotation = quaternion_to_matrix(*(float(turn.get(axis, 1.0 if axis == "w" else 0.0) or 0.0)
                                          for axis in "xyzw"))
        return position, rotation
    target = torch.tensor(_xyz(camera_info.get("target"), (0.0, 0.0, -1.0)), dtype=torch.float64)
    back = position - target
    back = _normal(back) if float(back.norm()) > 1e-12 else torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    if abs(float(back @ up)) > 0.999:
        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    right = _normal(torch.linalg.cross(up, back))
    return position, torch.stack([right, torch.linalg.cross(back, right), back], dim=1)


def camera(position, rotation: torch.Tensor, fov: float, zoom: float = 1.0) -> dict:
    """A ``camera_info`` for a world position and rotation.

    Args:
        position: Three world coordinates.
        rotation: ``(3, 3)`` world rotation, columns right, up and back.
        fov: Field of view across the picture's shorter side, in degrees.
        zoom: Focal length multiplier.

    Returns:
        A perspective ``camera_info`` carrying its quaternion and a target one unit ahead.
    """
    position = torch.as_tensor(position, dtype=torch.float64).reshape(3)
    rotation = torch.as_tensor(rotation, dtype=torch.float64)
    target = position - rotation[:, 2]
    x, y, z, w = matrix_to_quaternion(rotation)
    point = lambda v: {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}
    return {
        "position": point(position),
        "target": point(target),
        "quaternion": {"x": x, "y": y, "z": z, "w": w},
        "useCustomUp": True,
        "fov": float(fov),
        "zoom": float(zoom),
        "cameraType": "perspective",
    }


def source(fov: float) -> dict:
    """The camera a picture lifted with no pose was taken from: at the origin, looking down -Z."""
    return camera((0.0, 0.0, 0.0), torch.eye(3, dtype=torch.float64), fov)


def view(camera_info: Mapping) -> tuple[torch.Tensor, torch.Tensor]:
    """A camera in the splat frame.

    Returns:
        ``(eye (3,), rows (3, 3))`` in float64, the rows being the picture's right, down and
        forward directions, so ``(point - eye) @ rows.T`` is a point in the camera's own frame.
    """
    position, rotation = pose(camera_info)
    flip = torch.tensor(FLIP, dtype=torch.float64)
    eye = position * flip
    rows = torch.stack([rotation[:, 0] * flip, -rotation[:, 1] * flip, -rotation[:, 2] * flip])
    return eye, rows


def from_rows(eye: torch.Tensor, rows: torch.Tensor, fov: float) -> dict:
    """The ``camera_info`` of a camera given in the splat frame, the inverse of :func:`view`."""
    flip = torch.tensor(FLIP, dtype=torch.float64)
    rows = torch.as_tensor(rows, dtype=torch.float64)
    rotation = torch.stack([rows[0] * flip, -rows[1] * flip, -rows[2] * flip], dim=1)
    return camera(torch.as_tensor(eye, dtype=torch.float64) * flip, rotation, fov)


def focal(camera_info: Mapping, width: int, height: int) -> float:
    """The focal length in pixels a camera draws a ``width`` by ``height`` picture with."""
    fov = float(camera_info.get("fov") or 0.0) or DEFAULT_FOV
    zoom = float(camera_info.get("zoom") or 1.0)
    fov = min(max(fov, 1e-3), 179.0)
    return (min(width, height) / 2.0) / math.tan(math.radians(fov) / 2.0) * zoom


def fov_for_focal(focal_pixels: float, width: int, height: int) -> float:
    """The field of view, in degrees across the shorter side, of a focal length in pixels."""
    return math.degrees(2.0 * math.atan((min(width, height) / 2.0) / max(float(focal_pixels), 1e-6)))


def _smooth(t: float) -> float:
    return t * t * (3.0 - 2.0 * t)


#: Each easing as the share of the move made by a share of the time.
EASINGS: dict[str, Callable[[float], float]] = {
    "ease in and out": _smooth,
    "linear": lambda t: t,
    "ease in": lambda t: t * t,
    "ease out": lambda t: 1.0 - (1.0 - t) * (1.0 - t),
}


def ease(name: str, t: float) -> float:
    """``t`` in ``[0, 1]`` put through the named easing; an unknown name eases in and out."""
    return EASINGS.get(name, _smooth)(min(max(float(t), 0.0), 1.0))


def _yaw(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=torch.float64)


def _pitch(angle: float) -> torch.Tensor:
    c, s = math.cos(angle), math.sin(angle)
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=torch.float64)


#: Each preset move as ``(offset along right, up and back in travels, yaw, pitch, orbit)`` per unit
#: of progress, the angles in units of the move's angle.
MOVES: dict[str, tuple[tuple[float, float, float], float, float, float]] = {
    "push in": ((0.0, 0.0, -1.0), 0.0, 0.0, 0.0),
    "pull out": ((0.0, 0.0, 1.0), 0.0, 0.0, 0.0),
    "truck left": ((-1.0, 0.0, 0.0), 0.0, 0.0, 0.0),
    "truck right": ((1.0, 0.0, 0.0), 0.0, 0.0, 0.0),
    "crane up": ((0.0, 1.0, 0.0), 0.0, 0.0, 0.0),
    "crane down": ((0.0, -1.0, 0.0), 0.0, 0.0, 0.0),
    "pan left": ((0.0, 0.0, 0.0), 1.0, 0.0, 0.0),
    "pan right": ((0.0, 0.0, 0.0), -1.0, 0.0, 0.0),
    "tilt up": ((0.0, 0.0, 0.0), 0.0, 1.0, 0.0),
    "tilt down": ((0.0, 0.0, 0.0), 0.0, -1.0, 0.0),
    "orbit left": ((0.0, 0.0, 0.0), 0.0, 0.0, -1.0),
    "orbit right": ((0.0, 0.0, 0.0), 0.0, 0.0, 1.0),
    "push in and pan left": ((0.0, 0.0, -1.0), 1.0, 0.0, 0.0),
    "push in and pan right": ((0.0, 0.0, -1.0), -1.0, 0.0, 0.0),
    "push in and crane up": ((0.0, 0.6, -1.0), 0.0, -0.4, 0.0),
    "static": ((0.0, 0.0, 0.0), 0.0, 0.0, 0.0),
}


def move(start: Mapping, name: str, frames: int, travel: float, angle: float, focus: float,
         easing: str = "ease in and out", up: torch.Tensor | None = None) -> list[dict]:
    """A preset camera move as one camera per frame.

    Args:
        start: The camera the move begins at.
        name: A key of :data:`MOVES`.
        frames: Cameras to make, at least 1.
        travel: Distance moved, in scene units.
        angle: Turn made by a pan, tilt or orbit, in degrees.
        focus: Distance in front of ``start`` an orbit circles round, in scene units.
        easing: A key of :data:`EASINGS`.
        up: Unit world axis an orbit turns about; None = the camera's own up.

    Returns:
        The cameras, the first equal to ``start``.

    Raises:
        ValueError: ``name`` is not a preset move.
    """
    if name not in MOVES:
        raise ValueError(f"unknown camera move {name!r}; pick one of: {', '.join(MOVES)}")
    count = max(1, int(frames))
    return [_moved(start, name, ease(easing, index / (count - 1)) if count > 1 else 0.0, travel, angle, focus, up)
            for index in range(count)]


def _spin(axis: torch.Tensor, angle: float) -> torch.Tensor:
    """The ``(3, 3)`` rotation by ``angle`` radians about a unit ``axis``."""
    x, y, z = (float(v) for v in axis)
    skew = torch.tensor([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=torch.float64)
    return torch.eye(3, dtype=torch.float64) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def upright(given: Sequence[Mapping]) -> torch.Tensor | None:
    """The scene's vertical as the mean up direction of several cameras, or None for fewer than two."""
    if len(given) < 2:
        return None
    ups = torch.stack([pose(item)[1][:, 1] for item in given]).mean(0)
    return ups / ups.norm() if float(ups.norm()) > 1e-6 else None


def _moved(start: Mapping, name: str, progress: float, travel: float, angle: float, focus: float,
           up: torch.Tensor | None = None) -> dict:
    """``start`` after a share ``progress`` of a preset move; an orbit turns about ``up`` when given."""
    (dx, dy, dz), yaw, pitch, orbit = MOVES[name]
    position0, rotation0 = pose(start)
    fov = float(start.get("fov") or 0.0) or DEFAULT_FOV
    zoom = float(start.get("zoom") or 1.0)
    turn = math.radians(float(angle))
    offset = torch.tensor([dx, dy, dz], dtype=torch.float64) * (float(travel) * progress)
    rotation = rotation0 @ _yaw(yaw * turn * progress) @ _pitch(pitch * turn * progress)
    position = position0 + rotation0 @ offset
    if orbit:
        pivot = position0 - rotation0[:, 2] * float(focus)
        if up is not None:
            spin = _spin(up, orbit * turn * progress)
        else:
            spin = rotation0 @ _yaw(orbit * turn * progress) @ rotation0.T
        position = pivot + spin @ (position - pivot)
        rotation = spin @ rotation
    return camera(position, rotation, fov, zoom)


#: How a preset move treats a wired list of cameras: begun at its last camera, or added to each in turn.
WIRED = ("carry on from its end", "ride along it")


def along(given: Sequence[Mapping], name: str, frames: int, travel: float, angle: float, focus: float,
          easing: str = "ease in and out", smooth: int = 0) -> list[dict]:
    """A preset move added to a path of cameras, such as a video's, one camera per frame.

    Args:
        given: Two or more cameras, in order.
        name: A key of :data:`MOVES`.
        frames: Cameras to make.
        travel: Distance moved by the last frame, in scene units.
        angle: Turn made by the last frame, in degrees.
        focus: Distance in front of each camera an orbit circles round, in scene units.
        easing: A key of :data:`EASINGS`.
        smooth: Neighbours each side every wired camera is averaged with first; 0 = none.

    Returns:
        The cameras, the first equal to the first given.

    Raises:
        ValueError: ``name`` is not a preset move, or fewer than two cameras are given.
    """
    if name not in MOVES:
        raise ValueError(f"unknown camera move {name!r}; pick one of: {', '.join(MOVES)}")
    base = follow(given, int(frames), int(smooth))
    count = len(base)
    up = upright(given)
    return [_moved(item, name, ease(easing, index / (count - 1)) if count > 1 else 0.0, travel, angle, focus, up)
            for index, item in enumerate(base)]


#: The move that follows the cameras wired into a path rather than a preset.
FOLLOW = "follow the wired cameras"

#: The move that keeps turning the way the wired cameras were turning.
CONTINUE = "carry on the wired turn"


def turn_direction(given: Sequence[Mapping]) -> str:
    """``orbit left`` or ``orbit right``, whichever way the wired cameras turned from first to last.

    Raises:
        ValueError: Fewer than two cameras are given.
    """
    if len(given) < 2:
        raise ValueError("carrying on a turn needs a list of two or more cameras on camera_info, such as Point "
                         "Cloud From Depth's cameras for a video")
    up = upright(given)
    if up is None:
        return "orbit left"
    poses = [pose(item) for item in given]
    # The point the cameras look toward: the one nearest every view ray.
    system = torch.zeros(3, 3, dtype=torch.float64)
    target = torch.zeros(3, dtype=torch.float64)
    for position, rotation in poses:
        ahead = -rotation[:, 2]
        across = torch.eye(3, dtype=torch.float64) - torch.outer(ahead, ahead)
        system += across
        target += across @ position
    flat = lambda v: v - (v @ up) * up
    if float(torch.linalg.eigvalsh(system)[0]) > 1e-3 * len(poses):
        pivot = torch.linalg.solve(system, target)
        turned = float(torch.linalg.cross(flat(poses[0][0] - pivot), flat(poses[-1][0] - pivot)) @ up)
    else:
        turned = float(torch.linalg.cross(-poses[0][1][:, 2], -poses[-1][1][:, 2]) @ up)
    return "orbit right" if turned > 0 else "orbit left"


def follow(given: Sequence[Mapping], frames: int, smooth: int) -> list[dict]:
    """A steady path through a list of cameras, such as a video's estimated poses.

    Args:
        given: Two or more cameras, in order.
        frames: Cameras to make.
        smooth: Neighbours each side every camera is averaged with first; 0 = none.

    Returns:
        One camera per frame on a spline through the averaged cameras, evenly timed.

    Raises:
        ValueError: Fewer than two cameras are given.
    """
    if len(given) < 2:
        raise ValueError("following needs a list of two or more cameras on camera_info, such as Point Cloud "
                         "From Depth's cameras for a video")
    poses = [pose(item) for item in given]
    turns = [torch.tensor(matrix_to_quaternion(rotation), dtype=torch.float64) for _, rotation in poses]
    for index in range(1, len(turns)):
        if float(turns[index] @ turns[index - 1]) < 0:
            turns[index] = -turns[index]
    reach = max(0, int(smooth))
    keys = []
    for index, item in enumerate(given):
        side = min(reach, index, len(given) - 1 - index)
        low, high = index - side, index + side + 1
        position = torch.stack([poses[i][0] for i in range(low, high)]).mean(0)
        turn = torch.stack(turns[low:high]).mean(0)
        turn = turn / turn.norm().clamp_min(1e-12)
        fovs = sorted(float(given[i].get("fov") or 0.0) or DEFAULT_FOV for i in range(low, high))
        keys.append(camera(position, quaternion_to_matrix(*turn.tolist()), fovs[len(fovs) // 2]))
    return resample(keys, list(range(len(keys))), frames)



def _slerp(a: tuple, b: tuple, t: float) -> tuple[float, float, float, float]:
    """Spherical interpolation between two ``(x, y, z, w)`` quaternions."""
    dot = sum(p * q for p, q in zip(a, b))
    if dot < 0.0:
        b, dot = tuple(-q for q in b), -dot
    if dot > 0.9995:
        mixed = [p + (q - p) * t for p, q in zip(a, b)]
    else:
        theta = math.acos(min(dot, 1.0))
        sa, sb = math.sin((1 - t) * theta) / math.sin(theta), math.sin(t * theta) / math.sin(theta)
        mixed = [p * sa + q * sb for p, q in zip(a, b)]
    norm = math.sqrt(sum(v * v for v in mixed)) or 1.0
    return tuple(v / norm for v in mixed)


def _catmull(p0, p1, p2, p3, t: float) -> torch.Tensor:
    """A uniform Catmull-Rom point between ``p1`` and ``p2``."""
    t2, t3 = t * t, t * t * t
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * t3)


def resample(keys: Sequence[Mapping], times: Sequence[float], frames: int,
             easing: str = "linear") -> list[dict]:
    """Cameras placed at times along a smooth path through key cameras.

    Args:
        keys: Two or more cameras.
        times: The time of each key, rising, in any unit.
        frames: Cameras to make across the first to the last key's time.
        easing: A key of :data:`EASINGS` applied to the time between each pair of keys.

    Returns:
        One camera per frame; positions follow a Catmull-Rom spline and turns a spherical blend.

    Raises:
        ValueError: Fewer than two keys, or times that do not rise.
    """
    if len(keys) < 2 or len(times) != len(keys):
        raise ValueError("a path needs at least two key cameras, each with a time")
    stamps = [float(t) for t in times]
    if any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError("key camera times must rise from one key to the next")
    poses = [pose(key) for key in keys]
    points = [position for position, _ in poses]
    turns = [matrix_to_quaternion(rotation) for _, rotation in poses]
    fovs = [float(key.get("fov") or 0.0) or DEFAULT_FOV for key in keys]
    zooms = [float(key.get("zoom") or 1.0) for key in keys]
    count = max(1, int(frames))
    cameras = []
    for index in range(count):
        when = stamps[0] + (stamps[-1] - stamps[0]) * (index / (count - 1) if count > 1 else 0.0)
        span = max(0, min(len(stamps) - 2, next((i for i in range(len(stamps) - 1) if when <= stamps[i + 1]),
                                                 len(stamps) - 2)))
        local = (when - stamps[span]) / (stamps[span + 1] - stamps[span])
        local = ease(easing, local)
        p0 = points[max(span - 1, 0)]
        p3 = points[min(span + 2, len(points) - 1)]
        position = _catmull(p0, points[span], points[span + 1], p3, local)
        x, y, z, w = _slerp(turns[span], turns[span + 1], local)
        rotation = quaternion_to_matrix(x, y, z, w)
        fov = fovs[span] + (fovs[span + 1] - fovs[span]) * local
        zoom = zooms[span] + (zooms[span + 1] - zooms[span]) * local
        cameras.append(camera(position, rotation, fov, zoom))
    return cameras

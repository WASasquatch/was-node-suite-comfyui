"""The navigator's stored path as cameras, and its cloud packed for the browser.

Path: ``{"v": 1, "kind": "keys", "keys": [[s, x, y, z, qx, qy, qz, qw, fov], ...]}``, world
space. Clouds pack positions, spreads, colours; splats positions, covariances, colours.
"""

from __future__ import annotations

__all__ = [
    "KINDS",
    "MAX_KEYS",
    "MAX_PATH_CHARS",
    "MAX_SHOWN",
    "Path",
    "cameras_for",
    "pack",
    "pack_splats",
    "read_path",
]

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from . import cameras
from .cloud import SH_C0, Cloud
from .gaussians import Gaussians, quaternion_matrices

#: Longest stored path read, in characters.
MAX_PATH_CHARS = 4_000_000

#: Most keys a path may hold.
MAX_KEYS = 40_000

#: Most points sent to the browser.
MAX_SHOWN = 1_500_000

#: What a path was made by: keys placed one at a time, or a flight sampled as it was flown.
KINDS = ("keys", "recording")


@dataclass
class Path:
    """A stored path.

    Attributes:
        kind: One of :data:`KINDS`.
        times: Seconds of each key, rising.
        keys: One ``camera_info`` per key.
    """

    kind: str
    times: list[float]
    keys: list[dict]

    @property
    def seconds(self) -> float:
        return self.times[-1] - self.times[0] if len(self.times) > 1 else 0.0


def read_path(text: str) -> Path | None:
    """The path a navigator stored, or None for an empty one.

    Raises:
        ValueError: The text is too long, not a path, or holds a key that is not finite.
    """
    text = (text or "").strip()
    if not text:
        return None
    if len(text) > MAX_PATH_CHARS:
        raise ValueError(f"the stored path is {len(text):,} characters, past {MAX_PATH_CHARS:,}; clear it in "
                         "the navigator and record a shorter one")
    try:
        data = json.loads(text)
    except ValueError as error:
        raise ValueError(f"the stored path is not readable JSON ({error}); clear it in the navigator") from None
    if not isinstance(data, Mapping) or not isinstance(data.get("keys"), list):
        raise ValueError("the stored path holds no keys; clear it in the navigator and place keys again")
    rows = data["keys"]
    if len(rows) > MAX_KEYS:
        raise ValueError(f"the stored path holds {len(rows):,} keys, past {MAX_KEYS:,}; record a shorter path")
    kind = data.get("kind") if data.get("kind") in KINDS else "keys"
    found: dict[float, dict] = {}
    for row in rows:
        if not isinstance(row, list) or len(row) != 9:
            raise ValueError("a stored key is not [seconds, x, y, z, qx, qy, qz, qw, fov]; clear the path")
        values = [float(v) for v in row]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("a stored key holds a number that is not finite; clear the path")
        seconds, x, y, z, qx, qy, qz, qw, fov = values
        rotation = cameras.quaternion_to_matrix(qx, qy, qz, qw)
        found[round(seconds, 6)] = cameras.camera((x, y, z), rotation, min(max(fov, 1.0), 170.0))
    if not found:
        return None
    times = sorted(found)
    return Path(kind, times, [found[t] for t in times])


def cameras_for(path: Path | None, start: Mapping, frames: int, fps: float, easing: str) -> list[dict]:
    """One camera per frame along a stored path.

    Args:
        path: The stored path, or None.
        start: The camera to hold when the path has fewer than two keys.
        frames: Cameras to make; 0 = the path's own length at ``fps``.
        fps: Frames a second for ``frames`` = 0.
        easing: Easing between keys; a recording always moves linearly between its samples.

    Returns:
        The cameras.
    """
    if frames <= 0:
        seconds = path.seconds if path is not None else 0.0
        frames = max(1, int(round(seconds * fps)) + 1)
    if path is None or len(path.keys) < 2:
        hold = path.keys[0] if path is not None and path.keys else dict(start)
        return [dict(hold) for _ in range(frames)]
    return cameras.resample(path.keys, path.times, frames, "linear" if path.kind == "recording" else easing)


def pack(cloud: Cloud) -> tuple[bytes, int]:
    """A cloud as the bytes the navigator draws, thinned to :data:`MAX_SHOWN` points.

    Returns:
        ``(bytes, points sent)``.
    """
    step = max(1, math.ceil(cloud.count / MAX_SHOWN))
    positions = cloud.positions[::step].float().cpu() * torch.tensor([1.0, -1.0, -1.0])
    spreads = cloud.spreads[::step].float().cpu() * step ** 0.5
    colours = (cloud.colours[::step].float().cpu().clamp(0, 1) * 255.0).round().to(torch.uint8)
    body = b"".join(part.contiguous().numpy().tobytes() for part in (positions, spreads, colours))
    return body, int(positions.shape[0])


#: World frame from the splat frame, per axis.
FLIP = torch.tensor([1.0, -1.0, -1.0])


def pack_splats(splats: Gaussians) -> tuple[bytes, int]:
    """Fitted splats as the bytes the navigator blends, the :data:`MAX_SHOWN` most visible kept.

    Returns:
        ``(bytes, splats sent)``.
    """
    g = splats.to("cpu")
    if g.count > MAX_SHOWN:
        weight = g.opacities * g.scales.float().clamp_min(1e-12).log().mean(-1).exp()
        g = g.kept(torch.topk(weight, MAX_SHOWN).indices)
    positions = g.means.float() * FLIP
    shape = quaternion_matrices(g.rotations.float()) * g.scales.float()[:, None, :]
    cov = shape @ shape.transpose(1, 2)
    cov = cov * (FLIP[:, None] * FLIP[None, :])
    upper = torch.stack([cov[:, 0, 0], cov[:, 0, 1], cov[:, 0, 2], cov[:, 1, 1], cov[:, 1, 2], cov[:, 2, 2]], -1)
    colour = (g.sh[:, 0].float() * SH_C0 + 0.5).clamp(0, 1)
    tint = (torch.cat([colour, g.opacities.float()[:, None].clamp(0, 1)], -1) * 255.0).round().to(torch.uint8)
    body = b"".join(part.contiguous().numpy().tobytes() for part in (positions, upper, tint))
    return body, int(positions.shape[0])

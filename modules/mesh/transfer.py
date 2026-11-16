"""Carrying per-vertex attributes from a surface onto points on or near it.

Each point takes the attributes of the closest point on the source surface, interpolated
across that triangle's corners.
"""

from __future__ import annotations

__all__ = ["Samples", "interpolate", "morph_names", "sample", "transfer"]

from dataclasses import dataclass

import torch

from .arrays import MORPH_PREFIX, TriMesh
from .bvh import TriangleBVH

#: Points interpolated per pass, bounding the ``(points, 3, K)`` corner gather.
CHUNK_POINTS = 65536


@dataclass
class Samples:
    """Where each query point lands on a surface.

    Attributes:
        face: ``(Q,)`` int64 triangle index, ``-1`` where nothing lay within reach.
        barycentric: ``(Q, 3)`` weights of that triangle's corners.
        distance: ``(Q,)`` distance to the surface, ``inf`` on a miss.
        point: ``(Q, 3)`` closest point on the surface, the query itself on a miss.
    """

    face: torch.Tensor
    barycentric: torch.Tensor
    distance: torch.Tensor
    point: torch.Tensor

    @property
    def hit(self) -> torch.Tensor:
        """``(Q,)`` bool, True where a triangle was found."""
        return self.face >= 0


def sample(mesh: TriMesh, points: torch.Tensor, bvh: TriangleBVH | None = None,
           max_distance: float = float("inf")) -> Samples:
    """The closest point on a mesh to each query point.

    Args:
        mesh: The source surface.
        points: ``(Q, 3)`` query positions.
        bvh: A tree already built over ``mesh``, or None to build one.
        max_distance: Points farther than this from the surface miss.

    Returns:
        The landing triangle and weights of every point, on the points' device.
    """
    tree = bvh if bvh is not None else TriangleBVH(mesh.vertices, mesh.faces)
    distance, face, barycentric, closest = tree.closest_point(points, max_distance=max_distance)
    return Samples(face, barycentric, distance, closest)


def _finish(name: str, values: torch.Tensor, corner_values: torch.Tensor,
            barycentric: torch.Tensor) -> torch.Tensor:
    """Interpolated values made valid for their attribute: unit normals, summed skin, signs."""
    if name == "normals":
        return torch.nn.functional.normalize(values, dim=-1)
    if name == "tangents":
        direction = torch.nn.functional.normalize(values[:, :3], dim=-1)
        nearest = barycentric.argmax(dim=1)
        sign = corner_values[torch.arange(values.shape[0], device=values.device), nearest, 3]
        return torch.cat([direction, torch.where(sign < 0, -1.0, 1.0).unsqueeze(1)], dim=1)
    if name == "skin":
        values = values.clamp_min(0)
        total = values.sum(dim=1, keepdim=True)
        return torch.where(total > 0, values / total.clamp_min(1e-12), values)
    if name == "colors":
        return values.clamp(0, 1)
    return values


def interpolate(mesh: TriMesh, samples: Samples, names: list[str] | None = None) -> dict[str, torch.Tensor]:
    """Source attributes at the sampled points, normals unit length and skin rows summing to 1.

    Args:
        mesh: The surface the samples were taken on.
        samples: The result of `sample` on that surface.
        names: Attribute names to carry, or None for every attribute the mesh holds.

    Returns:
        ``(Q, K)`` tensors by attribute name, zero where a point missed.
    """
    device = samples.face.device
    wanted = list(mesh.attributes) if names is None else [name for name in names if name in mesh.attributes]
    hit = samples.hit
    faces = mesh.faces.to(device)
    count = int(samples.face.shape[0])
    result: dict[str, torch.Tensor] = {}
    for name in wanted:
        source = mesh.attributes[name].to(device)
        source = source.unsqueeze(1) if source.dim() == 1 else source
        out = torch.zeros(count, source.shape[1], dtype=source.dtype, device=device)
        for start in range(0, count, CHUNK_POINTS):
            stop = min(start + CHUNK_POINTS, count)
            rows = hit[start:stop]
            if not bool(rows.any()):
                continue
            face = samples.face[start:stop][rows]
            weights = samples.barycentric[start:stop][rows].to(source.dtype)
            corners = source[faces[face]]
            values = (corners * weights.unsqueeze(2)).sum(dim=1)
            chunk = out[start:stop]
            chunk[rows] = _finish(name, values, corners, weights)
        result[name] = out
    return result


def transfer(source: TriMesh, points: torch.Tensor, names: list[str] | None = None,
             bvh: TriangleBVH | None = None,
             max_distance: float = float("inf")) -> tuple[dict[str, torch.Tensor], Samples]:
    """Source attributes at the closest surface point to each query point.

    Args:
        source: The surface carrying the attributes.
        points: ``(Q, 3)`` positions to give attributes to.
        names: Attribute names to carry, or None for all. Shape keys are named
            ``"morph:<name>"``.
        bvh: A tree already built over ``source``, or None to build one.
        max_distance: Points farther than this from the surface get zeros.

    Returns:
        ``(attributes, samples)``.
    """
    samples = sample(source, points, bvh=bvh, max_distance=max_distance)
    return interpolate(source, samples, names), samples


def morph_names(mesh: TriMesh) -> list[str]:
    """Attribute names of the shape keys a mesh carries."""
    return [name for name in mesh.attributes if name.startswith(MORPH_PREFIX)]

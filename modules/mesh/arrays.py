"""The triangle mesh the mesh algorithms take and return, and its exchange with a model part.

Attributes are ``(N, K)`` tensors keyed ``"uvs"``, ``"normals"``, ``"tangents"``,
``"colors"``, ``"skin"`` (dense ``(N, J)``) and ``"morph:<name>"`` (offsets).
"""

from __future__ import annotations

__all__ = ["TriMesh", "dense_skin", "from_part", "sparse_skin", "to_part"]

from dataclasses import dataclass, field, replace

import torch

from ..model3d.scene import MorphTarget, Part, Polygons

#: Prefix of an attribute name holding a shape key's position offsets.
MORPH_PREFIX = "morph:"

#: Skin influences kept per vertex when dense weights are written back to a part.
SKIN_INFLUENCES = 4


@dataclass
class TriMesh:
    """Triangles with per-vertex attributes.

    Attributes:
        vertices: ``(N, 3)`` float32 positions.
        faces: ``(M, 3)`` int64 triangles, counter-clockwise seen from the front.
        attributes: Per-vertex ``(N, K)`` tensors by name.
        polygons: Quads or n-gons the triangles came from, or None.
    """

    vertices: torch.Tensor
    faces: torch.Tensor
    attributes: dict[str, torch.Tensor] = field(default_factory=dict)
    polygons: Polygons | None = None

    @property
    def vertex_count(self) -> int:
        """Number of vertices."""
        return int(self.vertices.shape[0])

    @property
    def face_count(self) -> int:
        """Number of triangles."""
        return int(self.faces.shape[0])

    def to(self, device: torch.device | str) -> TriMesh:
        """The same mesh with every tensor on ``device``."""
        polygons = self.polygons
        if polygons is not None:
            polygons = Polygons(
                polygons.sizes.to(device), polygons.indices.to(device),
                polygons.triangle_polygon.to(device),
            )
        return replace(
            self,
            vertices=self.vertices.to(device),
            faces=self.faces.to(device),
            attributes={name: value.to(device) for name, value in self.attributes.items()},
            polygons=polygons,
        )


def dense_skin(joints: torch.Tensor, weights: torch.Tensor, joint_count: int) -> torch.Tensor:
    """Sparse skin influences spread into one weight column per joint.

    Args:
        joints: ``(N, K)`` int64 joint indices.
        weights: ``(N, K)`` float weights.
        joint_count: Number of joints in the skeleton.

    Returns:
        ``(N, joint_count)`` float32 weights.
    """
    dense = torch.zeros(joints.shape[0], joint_count, dtype=torch.float32, device=weights.device)
    dense.scatter_add_(1, joints.long().clamp(0, max(joint_count - 1, 0)), weights.float())
    return dense


def sparse_skin(dense: torch.Tensor, influences: int = SKIN_INFLUENCES) -> tuple[torch.Tensor, torch.Tensor]:
    """The strongest influences of dense weights, renormalised.

    Args:
        dense: ``(N, J)`` weights.
        influences: Influences kept per vertex.

    Returns:
        ``(joints, weights)``: ``(N, influences)`` int64 and float32, rows summing to 1.
    """
    count = min(influences, dense.shape[1])
    weights, joints = dense.clamp_min(0).topk(count, dim=1)
    total = weights.sum(dim=1, keepdim=True)
    weights = torch.where(total > 0, weights / total.clamp_min(1e-12), torch.zeros_like(weights))
    weights[:, 0] = torch.where(total[:, 0] > 0, weights[:, 0], torch.ones_like(weights[:, 0]))
    if count < influences:
        pad = influences - count
        joints = torch.cat([joints, joints.new_zeros(joints.shape[0], pad)], dim=1)
        weights = torch.cat([weights, weights.new_zeros(weights.shape[0], pad)], dim=1)
    return joints.long(), weights.float()


def from_part(part: Part, joint_count: int = 0) -> TriMesh:
    """A part's geometry and attributes as a `TriMesh`.

    Args:
        part: The part to read.
        joint_count: Skeleton size, used to spread skin weights; 0 leaves skin out.

    Returns:
        The mesh, sharing tensors with the part.
    """
    attributes = {name: value for name, value in part.vertex_attributes().items()
                  if name not in ("joints", "weights")}
    if joint_count and part.joints is not None and part.weights is not None:
        attributes["skin"] = dense_skin(part.joints, part.weights, joint_count)
    for name, morph in part.morphs.items():
        attributes[MORPH_PREFIX + name] = morph.positions
    return TriMesh(part.vertices, part.faces, attributes, part.polygons)


def to_part(part: Part, mesh: TriMesh) -> Part:
    """A copy of ``part`` carrying ``mesh``'s geometry and attributes, and no others.

    Morph normals are dropped.

    Args:
        part: The part supplying name, material, label and extras.
        mesh: The new geometry.

    Returns:
        A new part.
    """
    attributes = mesh.attributes
    joints = weights = None
    if "skin" in attributes:
        joints, weights = sparse_skin(attributes["skin"])
    morphs = {
        name[len(MORPH_PREFIX):]: MorphTarget(value.float().cpu())
        for name, value in attributes.items() if name.startswith(MORPH_PREFIX)
    }

    def held(name: str):
        value = attributes.get(name)
        return value.float().cpu() if value is not None else None

    polygons = mesh.polygons
    if polygons is not None:
        polygons = Polygons(polygons.sizes.cpu(), polygons.indices.cpu(), polygons.triangle_polygon.cpu())
    return replace(
        part,
        vertices=mesh.vertices.float().cpu(),
        faces=mesh.faces.long().cpu(),
        polygons=polygons,
        uvs=held("uvs"),
        normals=held("normals"),
        tangents=held("tangents"),
        colors=held("colors"),
        joints=joints.cpu() if joints is not None else None,
        weights=weights.cpu() if weights is not None else None,
        morphs=morphs,
    )

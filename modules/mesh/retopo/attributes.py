"""Building the retopologised triangle mesh with the source's attributes carried onto it.

Skin, shape keys and vertex colours are interpolated from the closest source point. Normals
are recomputed and split at creases. UVs are dropped or carried per source chart.
"""

from __future__ import annotations

__all__ = ["UV_MODES", "carry", "normal_groups", "source_charts"]

import math
from dataclasses import replace

import torch

from .. import topology, transfer
from ...model3d.scene import Polygons
from ..arrays import MORPH_PREFIX, TriMesh
from ..bvh import TriangleBVH
from .polymesh import PolyMesh

#: What happens to texture coordinates: dropped, or carried from the source chart under each face.
UV_MODES = ("drop", "transfer")


def normal_groups(vertices: torch.Tensor, faces: torch.Tensor, sides: torch.Tensor,
                  crease_angle: float) -> torch.Tensor:
    """Smooth-shading group of each triangle corner, split at creases along polygon sides.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(T, 3)`` triangles.
        sides: ``(S, 2)`` int64 polygon sides; triangle edges not among them are never split.
        crease_angle: Dihedral angle in degrees from which a side is hard; 0 = all smooth.

    Returns:
        ``(T * 3,)`` int64 group label of each corner.
    """
    count = int(vertices.shape[0])
    topo = topology.edge_topology(faces, count)
    if crease_angle <= 0 or topo.edge_count == 0:
        return topology.corner_fans(faces, count, topo)
    angle = topology.dihedral_angles(vertices, faces, topo)
    stride = max(count, 1)
    side_keys = torch.unique(torch.minimum(sides[:, 0], sides[:, 1]) * stride + torch.maximum(sides[:, 0], sides[:, 1]))
    edge_keys = topo.edges[:, 0] * stride + topo.edges[:, 1]
    at = torch.searchsorted(side_keys, edge_keys).clamp_max(max(side_keys.shape[0] - 1, 0))
    is_side = side_keys[at] == edge_keys if side_keys.numel() else torch.zeros_like(edge_keys, dtype=torch.bool)
    hard = is_side & (angle >= math.radians(float(crease_angle)))
    cut = replace(topo, edge_face_count=torch.where(hard, torch.ones_like(topo.edge_face_count), topo.edge_face_count))
    return topology.corner_fans(faces, count, cut)


def source_charts(mesh: TriMesh) -> torch.Tensor:
    """UV chart of each source triangle: pieces joined through vertices sharing position and UV.

    Args:
        mesh: The source mesh with ``"uvs"``.

    Returns:
        ``(F,)`` int64 chart labels from 0.
    """
    rows = torch.cat([mesh.vertices.float(), mesh.attributes["uvs"].float()], 1)
    _, welded = torch.unique(rows, dim=0, return_inverse=True)
    faces = welded[mesh.faces]
    return topology.face_components(faces, int(welded.max()) + 1 if welded.numel() else 0)


def _chart_uvs(source: TriMesh, charts: torch.Tensor, points: torch.Tensor,
               point_chart: torch.Tensor) -> torch.Tensor:
    """UVs at points, each read from the closest place on its own source chart."""
    device = points.device
    chart_count = int(charts.max()) + 1 if charts.numel() else 1
    vertex_chart = torch.zeros(source.vertex_count, dtype=torch.int64, device=device)
    vertex_chart[source.faces.reshape(-1)] = charts.repeat_interleave(3)
    low, high = source.vertices.amin(0), source.vertices.amax(0)
    spacing = 4.0 * float((high - low).norm()) + 1.0
    side = max(1, math.ceil(chart_count ** (1.0 / 3.0) - 1e-9))

    def offset(label: torch.Tensor) -> torch.Tensor:
        cell = torch.stack([label % side, (label // side) % side, label // (side * side)], 1)
        return cell.to(source.vertices.dtype) * spacing

    spread = source.vertices + offset(vertex_chart)
    tree = TriangleBVH(spread, source.faces, device=device)
    _, face, bary, _ = tree.closest_point(points + offset(point_chart))
    uvs = source.attributes["uvs"]
    corners = uvs[source.faces[face.clamp_min(0)]]
    values = (corners * bary[:, :, None].to(uvs.dtype)).sum(1)
    return torch.where((face >= 0)[:, None], values, torch.zeros_like(values))


def carry(source: TriMesh, poly: PolyMesh, bvh: TriangleBVH | None, uvs: str = "drop", crease_angle: float = 0.0,
          keep_polygons: bool = True) -> TriMesh:
    """The retopologised mesh as triangles, with polygons and the source's attributes.

    Args:
        source: The source mesh with its attributes.
        poly: The new polygons.
        bvh: Tree over the source triangles, face for face; None when nothing is carried.
        uvs: ``"drop"`` or ``"transfer"``.
        crease_angle: Dihedral angle in degrees from which normals split; 0 = all smooth.
        keep_polygons: Record the polygons on the result; False leaves plain triangles.

    Returns:
        The new `TriMesh`, vertices split where UV charts or hard edges meet.
    """
    device = poly.vertices.device
    faces, owner = poly.triangulate()
    base = poly.vertices
    count = int(base.shape[0])
    names = [name for name in source.attributes
             if name in ("skin", "colors") or name.startswith(MORPH_PREFIX)]
    carried = {}
    if names:
        carried, _ = transfer.transfer(source, base, names, bvh=bvh)
    sides, _ = poly.edges()
    group = normal_groups(base, faces, sides, crease_angle)
    face_normal = topology.face_normals(base, faces)
    angle = topology.corner_angles(base, faces)
    weighted = (face_normal[:, None, :] * angle[..., None]).reshape(-1, 3)
    group_normal = torch.zeros(faces.shape[0] * 3, 3, dtype=base.dtype, device=device).index_add_(0, group, weighted)
    group_normal = torch.nn.functional.normalize(group_normal, dim=1)
    polygon_chart = torch.zeros(poly.polygon_count, dtype=torch.int64, device=device)
    charts = None
    if uvs == "transfer" and "uvs" in source.attributes and source.face_count:
        charts = source_charts(source)
        _, nearest, _, _ = bvh.closest_point(poly.centroids())
        polygon_chart = torch.where(nearest >= 0, charts[nearest.clamp_min(0)], torch.zeros_like(nearest))
    corner_vertex = faces.reshape(-1)
    corner_chart = polygon_chart[owner.repeat_interleave(3)]
    keys = torch.stack([corner_vertex, corner_chart, group], 1)
    unique_keys, new_index = torch.unique(keys, dim=0, return_inverse=True)
    vertex = unique_keys[:, 0]
    first_corner = torch.full((unique_keys.shape[0],), corner_vertex.shape[0], dtype=torch.int64, device=device)
    first_corner.scatter_reduce_(0, new_index, torch.arange(corner_vertex.shape[0], device=device), reduce="amin")
    attributes = {name: value[vertex] for name, value in carried.items()}
    attributes["normals"] = group_normal[group[first_corner]]
    if charts is not None:
        attributes["uvs"] = _chart_uvs(source, charts, base[vertex], unique_keys[:, 1])
    new_faces = new_index.reshape(-1, 3)
    polygons = None
    if keep_polygons and poly.polygon_count:
        stride = max(count, 1)
        corner_key = owner.repeat_interleave(3) * stride + corner_vertex
        lookup_keys, lookup_order = torch.sort(corner_key)
        polygon_key = poly.corner_polygon() * stride + poly.indices
        at = torch.searchsorted(lookup_keys, polygon_key).clamp_max(lookup_keys.shape[0] - 1)
        polygons = Polygons(poly.sizes.clone(), new_index[lookup_order[at]], owner.clone())
    return TriMesh(base[vertex].float(), new_faces, attributes, polygons)

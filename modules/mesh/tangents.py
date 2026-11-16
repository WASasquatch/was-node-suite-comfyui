"""MikkTSpace tangents, and splitting a part's vertices where per-corner values disagree. A port of
mikktspace.c by Morten S. Mikkelsen (zlib licence) with its default 180 degree threshold, triangles
only."""
# Tangent ``w`` follows glTF: ``bitangent = cross(normal, tangent) * w`` points up the image.

from __future__ import annotations

__all__ = ["corner_tangents", "smooth_normals", "split_by_corner_values", "split_part", "with_tangents"]

from dataclasses import replace

import torch

from ..model3d.scene import MorphTarget, Part, Polygons
from . import topology

#: Smallest magnitude treated as non-zero, as mikktspace's ``FLT_MIN`` test.
TINY = torch.finfo(torch.float32).tiny

#: Rounds of orientation hand-down to triangles with no usable texture mapping.
ORIENT_ROUNDS = 16


def _not_zero(vectors: torch.Tensor) -> torch.Tensor:
    """Whether any component of each vector is non-zero."""
    return (vectors.abs() > TINY).any(dim=-1)


def _unit(vectors: torch.Tensor) -> torch.Tensor:
    """Vectors scaled to unit length where they are non-zero, left as they are otherwise."""
    length = vectors.norm(dim=-1, keepdim=True)
    return torch.where(_not_zero(vectors)[..., None], vectors / length.clamp_min(TINY), vectors)


def _project(vectors: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    """Vectors with their component along unit normals removed, then made unit length."""
    return _unit(vectors - (vectors * normals).sum(-1, keepdim=True) * normals)


def _perpendicular(normals: torch.Tensor) -> torch.Tensor:
    """A unit vector perpendicular to each normal, ``+X`` for a zero normal."""
    axis = torch.zeros_like(normals)
    smallest = normals.abs().argmin(dim=-1, keepdim=True)
    axis.scatter_(-1, smallest, 1.0)
    side = torch.cross(normals, axis, dim=-1)
    side = torch.cross(side, normals, dim=-1)
    fallback = torch.zeros_like(normals)
    fallback[..., 0] = 1.0
    length = side.norm(dim=-1, keepdim=True)
    return torch.where(length > 1e-12, side / length.clamp_min(1e-12), fallback)


def smooth_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Angle-weighted vertex normals, shared by every vertex at one position.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` vertex indices.

    Returns:
        ``(V, 3)`` float32 unit normals, ``+Y`` where no face gives one.
    """
    welded, welded_faces, remap = topology.weld(vertices.float(), faces.long(), tolerance=0.0)
    normals = topology.vertex_normals(welded, welded_faces, weighting="angle")[remap]
    up = torch.zeros_like(normals)
    up[:, 1] = 1.0
    return torch.where(normals.norm(dim=1, keepdim=True) > 0, normals, up).float()


def _pair_half_edges(shared: torch.Tensor, active: torch.Tensor, vertex_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Half-edges matched with the one running the opposite way between the same two vertices.

    Args:
        shared: ``(F, 3)`` welded vertex indices.
        active: ``(F,)`` bool, faces that take part.
        vertex_count: Number of welded vertices.

    Returns:
        ``(first, second)`` int64 half-edge indices ``f * 3 + k``, each pair once. Edges with
        more than one half-edge either way are left unmatched.
    """
    device = shared.device
    start = shared.reshape(-1)
    end = shared[:, [1, 2, 0]].reshape(-1)
    half = torch.arange(start.shape[0], device=device)
    usable = active.repeat_interleave(3) & (start != end)
    half, start, end = half[usable], start[usable], end[usable]
    stride = max(int(vertex_count), 1)
    undirected = torch.minimum(start, end) * stride + torch.maximum(start, end)
    forward = (start < end).long()
    order = torch.argsort(undirected * 2 + forward)
    half, undirected, forward = half[order], undirected[order], forward[order]
    _, counts = torch.unique_consecutive(undirected, return_counts=True)
    group_start = torch.cumsum(counts, 0) - counts
    pairs = group_start[counts == 2]
    opposite = forward[pairs] != forward[pairs + 1]
    pairs = pairs[opposite]
    return half[pairs], half[pairs + 1]


def corner_tangents(vertices: torch.Tensor, faces: torch.Tensor, normals: torch.Tensor,
                    uvs: torch.Tensor) -> torch.Tensor:
    """MikkTSpace tangent of every triangle corner.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` vertex indices.
        normals: ``(V, 3)`` unit vertex normals.
        uvs: ``(V, 2)`` texture coordinates, ``v`` down the image.

    Returns:
        ``(F, 3, 4)`` float32: unit tangent and glTF handedness per corner.
    """
    device = vertices.device
    faces = faces.long()
    face_count = int(faces.shape[0])
    if face_count == 0:
        return torch.zeros(0, 3, 4, device=device)
    position = vertices.float()
    normal = normals.float().to(device)
    texture = uvs.float().to(device)
    _, shared_index = torch.unique(torch.cat([position, normal, texture], dim=1), dim=0, return_inverse=True)
    shared = shared_index[faces]
    shared_count = int(shared_index.max()) + 1 if shared_index.numel() else 0
    corners = position[faces]
    corner_uv = texture[faces]
    corner_normal = normal[faces]

    same = lambda a, b: (corners[:, a] == corners[:, b]).all(dim=1)  # noqa: E731
    degenerate = same(0, 1) | same(0, 2) | same(1, 2)
    d1 = corners[:, 1] - corners[:, 0]
    d2 = corners[:, 2] - corners[:, 0]
    t21 = corner_uv[:, 1] - corner_uv[:, 0]
    t31 = corner_uv[:, 2] - corner_uv[:, 0]
    signed_area = t21[:, 0] * t31[:, 1] - t21[:, 1] * t31[:, 0]
    raw_s = t31[:, 1:2] * d1 - t21[:, 1:2] * d2
    raw_t = -t31[:, 0:1] * d1 + t21[:, 0:1] * d2
    preserving = signed_area > 0
    has_area = signed_area.abs() > TINY
    length_s = raw_s.norm(dim=1)
    length_t = raw_t.norm(dim=1)
    sign = torch.where(preserving, 1.0, -1.0)
    v_s = torch.where((has_area & (length_s > TINY))[:, None], raw_s * (sign / length_s.clamp_min(TINY))[:, None],
                      torch.zeros_like(raw_s))
    v_t = torch.where((has_area & (length_t > TINY))[:, None], raw_t * (sign / length_t.clamp_min(TINY))[:, None],
                      torch.zeros_like(raw_t))
    area_size = signed_area.abs().clamp_min(TINY)
    magnitude_s = torch.where(has_area, length_s / area_size, torch.zeros_like(length_s))
    magnitude_t = torch.where(has_area, length_t / area_size, torch.zeros_like(length_t))
    active = ~degenerate
    good = active & (magnitude_s > TINY) & (magnitude_t > TINY)

    first, second = _pair_half_edges(shared, active, shared_count)
    face_a, face_b = first // 3, second // 3

    # Triangles with no usable mapping take their orientation from a neighbour that has one.
    orient = preserving.clone()
    known = good.clone()
    both_faces = torch.cat([face_a, face_b])
    other_faces = torch.cat([face_b, face_a])
    for _ in range(ORIENT_ROUNDS):
        reach = ~known[both_faces] & known[other_faces]
        if not bool(reach.any()):
            break
        target, source = both_faces[reach], other_faces[reach]
        chosen = torch.full((face_count,), face_count, dtype=torch.int64, device=device)
        chosen.scatter_reduce_(0, target, source, reduce="amin")
        newly = torch.unique(target)
        orient[newly] = orient[chosen[newly]]
        known[newly] = True

    side_a, side_b = first % 3, second % 3
    linked = orient[face_a] == orient[face_b]
    face_a, face_b, side_a, side_b = face_a[linked], face_b[linked], side_a[linked], side_b[linked]
    link_from = torch.cat([face_a * 3 + side_a, face_a * 3 + (side_a + 1) % 3])
    link_to = torch.cat([face_b * 3 + (side_b + 1) % 3, face_b * 3 + side_b])
    group = topology.union_find(face_count * 3, link_from, link_to)

    flat_normal = corner_normal.reshape(-1, 3)
    corner_s = _project(v_s.repeat_interleave(3, dim=0), flat_normal)
    previous = corners[:, [2, 0, 1]].reshape(-1, 3)
    following = corners[:, [1, 2, 0]].reshape(-1, 3)
    here = corners.reshape(-1, 3)
    edge_back = _project(previous - here, flat_normal)
    edge_ahead = _project(following - here, flat_normal)
    cosine = (edge_back * edge_ahead).sum(dim=1).clamp(-1.0, 1.0)
    angle = torch.acos(cosine) * good.repeat_interleave(3).float()

    total_s = torch.zeros(face_count * 3, 3, device=device)
    total_s.index_add_(0, group, corner_s * angle[:, None])
    tangent = _unit(total_s[group])

    handedness = torch.where(orient, -1.0, 1.0).repeat_interleave(3)
    fallback = _perpendicular(flat_normal)
    missing = ~_not_zero(tangent)

    # Corners of degenerate triangles copy a sound corner on the same welded vertex.
    flat_shared = shared.reshape(-1)
    sound = ~degenerate.repeat_interleave(3) & ~missing
    if bool(degenerate.any()) and shared_count:
        donor = torch.full((shared_count,), face_count * 3, dtype=torch.int64, device=device)
        corner_index = torch.arange(face_count * 3, device=device)
        donor.scatter_reduce_(0, flat_shared[sound], corner_index[sound], reduce="amin")
        needy = degenerate.repeat_interleave(3)
        source = donor[flat_shared[needy]]
        found = source < face_count * 3
        needy_index = torch.nonzero(needy).squeeze(1)
        take = needy_index[found]
        tangent[take] = tangent[source[found]]
        handedness[take] = handedness[source[found]]
        missing = ~_not_zero(tangent)
    tangent = torch.where(missing[:, None], fallback, tangent)
    return torch.cat([tangent, handedness[:, None]], dim=1).reshape(face_count, 3, 4)


def split_by_corner_values(faces: torch.Tensor, values: torch.Tensor,
                           vertex_count: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Give each vertex one copy per distinct value its corners carry.

    Args:
        faces: ``(F, 3)`` vertex indices.
        values: ``(F, 3, K)`` per-corner values.
        vertex_count: Number of vertices.

    Returns:
        ``(faces, source, vertex_values)``: faces on the new vertices, ``(V2,)`` int64 original
        vertex of each, and ``(V2, K)`` values. A vertex keeps its index for its first value;
        further copies are appended. Vertices in no face get zero values.
    """
    device = faces.device
    faces = faces.long()
    width = int(values.shape[-1])
    flat_vertex = faces.reshape(-1)
    key = torch.cat([flat_vertex[:, None].double(), values.reshape(-1, width).double()], dim=1)
    if key.shape[0] == 0:
        return faces.clone(), torch.arange(vertex_count, device=device), torch.zeros(vertex_count, width, device=device)
    unique_rows, inverse = torch.unique(key, dim=0, return_inverse=True)
    owner = unique_rows[:, 0].long()
    start = torch.ones(owner.shape[0], dtype=torch.bool, device=device)
    start[1:] = owner[1:] != owner[:-1]
    extra = ~start
    new_index = owner.clone()
    new_index[extra] = int(vertex_count) + torch.arange(int(extra.sum()), device=device)
    source = torch.cat([torch.arange(int(vertex_count), device=device), owner[extra]])
    vertex_values = torch.zeros(source.shape[0], width, dtype=values.dtype, device=device)
    vertex_values[new_index] = unique_rows[:, 1:].to(values.dtype)
    return new_index[inverse].reshape(-1, 3), source, vertex_values


def _remap_polygons(polygons: Polygons, old_faces: torch.Tensor, new_faces: torch.Tensor,
                    source: torch.Tensor, old_count: int) -> Polygons:
    """Polygons pointed at the split vertices their own triangles use."""
    device = new_faces.device
    triangle_polygon = polygons.triangle_polygon.to(device).long()
    corner_polygon = triangle_polygon.repeat_interleave(3)
    stride = max(int(old_count), 1)
    key = corner_polygon * stride + old_faces.reshape(-1).long().to(device)
    keys, first = torch.unique(key, return_inverse=True)
    chosen = torch.full((keys.shape[0],), -1, dtype=torch.int64, device=device)
    chosen.scatter_(0, first, new_faces.reshape(-1).long())
    sizes = polygons.sizes.to(device).long()
    owner = torch.repeat_interleave(torch.arange(sizes.shape[0], device=device), sizes)
    old_index = polygons.indices.to(device).long()
    wanted = owner * stride + old_index
    at = torch.searchsorted(keys, wanted).clamp_max(max(keys.shape[0] - 1, 0))
    found = keys.numel() > 0
    hit = (keys[at] == wanted) if found else torch.zeros_like(wanted, dtype=torch.bool)
    first_copy = torch.zeros(stride, dtype=torch.int64, device=device)
    source = source.long().to(device)
    first_copy.scatter_reduce_(0, source, torch.arange(source.shape[0], device=device), reduce="amin",
                               include_self=False)
    fallback = first_copy[old_index.clamp(0, stride - 1)]
    indices = torch.where(hit, chosen[at] if found else fallback, fallback)
    return Polygons(polygons.sizes.clone(), indices.cpu(), polygons.triangle_polygon.clone())


def split_part(part: Part, faces: torch.Tensor, source: torch.Tensor, **attributes) -> Part:
    """A part rebuilt on split vertices, each copying the attributes of its source vertex.

    Args:
        part: The part to rebuild.
        faces: ``(F, 3)`` faces on the new vertices, row for row with the part's faces.
        source: ``(V2,)`` original vertex of each new vertex.
        **attributes: Per-vertex tensors for the new vertices that replace the copied ones,
            such as ``uvs`` or ``tangents``.

    Returns:
        A new part sharing no tensor with the old one.
    """
    source = source.long().cpu()

    def taken(value):
        return value[source].clone() if value is not None else None

    morphs = {
        name: MorphTarget(morph.positions[source].clone(),
                          morph.normals[source].clone() if morph.normals is not None else None)
        for name, morph in part.morphs.items()
    }
    polygons = None
    if part.polygons is not None:
        polygons = _remap_polygons(part.polygons, part.faces.cpu(), faces.cpu(), source, part.vertex_count)
    fields = {name: taken(getattr(part, name)) for name in Part.VERTEX_ATTRIBUTES}
    fields.update({name: value.float().cpu() if value is not None else None for name, value in attributes.items()})
    return replace(part, vertices=part.vertices[source].clone(), faces=faces.long().cpu().clone(),
                   polygons=polygons, morphs=morphs, extras=dict(part.extras), **fields)


def with_tangents(part: Part) -> Part:
    """A part carrying MikkTSpace tangents, its vertices split where corners disagree.

    Args:
        part: A part with texture coordinates. Missing normals are filled with smooth ones.

    Returns:
        A new part with ``normals`` and ``tangents`` set.

    Raises:
        ValueError: The part has no texture coordinates.
    """
    if part.uvs is None:
        raise ValueError(f"part '{part.name}' has no texture coordinates, so it has no tangents; "
                         "unwrap it with Mesh UV Unwrap first")
    normals = part.normals if part.normals is not None else smooth_normals(part.vertices, part.faces)
    corner = corner_tangents(part.vertices, part.faces, normals, part.uvs)
    faces, source, values = split_by_corner_values(part.faces, corner, part.vertex_count)
    if source.shape[0] == part.vertex_count and torch.equal(faces, part.faces.long()):
        return replace(part, normals=normals.float().clone(), tangents=values.float(), extras=dict(part.extras))
    split = split_part(part, faces, source, tangents=values)
    split.normals = normals[source.long()].float().clone()
    return split

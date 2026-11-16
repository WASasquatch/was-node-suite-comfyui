"""Retopologising a triangle mesh into quads, and the selected parts of a whole model.

Sizes are in model units. Faces come out counter-clockwise seen from the front.
"""

from __future__ import annotations

__all__ = [
    "MAX_TARGET_FACES", "OUTPUTS", "Retopology", "SYMMETRIES", "mesh_figures", "retopologize",
    "retopologize_model",
]

import math
import time
from dataclasses import dataclass, field, replace

import torch

from .. import device as mesh_device
from .. import topology
from ...model3d import selection
from ...model3d.scene import Model3D
from ..arrays import MORPH_PREFIX, TriMesh, from_part, to_part
from ..bvh import TriangleBVH
from . import attributes, extract, fields, hierarchy, polymesh, prepare
from .attributes import UV_MODES

#: Kinds of mesh the retopology can produce.
OUTPUTS = ("quads", "quad dominant", "triangles")

#: Mirror planes the retopology can make the layout symmetric across.
SYMMETRIES = ("none", "x")

#: Most faces one run may ask for.
MAX_TARGET_FACES = 200_000

#: Most vertices the working surface may reach after splitting long edges.
MAX_WORK_VERTICES = 6_000_000

#: Most source triangles one part may hold.
MAX_SOURCE_TRIANGLES = 12_000_000

#: Fewest polygons a separate piece of a part is given before any subdivision.
MIN_COMPONENT_POLYGONS = 6

#: Fewest faces asked of one part.
MIN_PART_FACES = 24

#: Fields are read off the coarsest level whose mean edge is at most this many spacings.
EXTRACT_SPACINGS = 0.25

#: Field normals are averaged over the coarsest level whose mean edge is at most this many spacings.
NORMAL_SPACINGS = 0.5

#: Border vertices this many lattice spacings or nearer to the mirror plane are joined across it.
PLANE_SPACINGS = 0.75

#: Seed of every random choice, so a run repeats exactly.
SEED = 7

#: Times holes are closed and crowded polygons removed, until a pass removes none.
REPAIR_PASSES = 4

#: Share of the target the faces of a first layout may miss by before the spacing is rescaled
#: from that layout's count and the layout made again.
COUNT_TOLERANCE = 0.03

#: Lattice spacings from a source border within which a vertex lies on it.
BORDER_SPACINGS = 0.75

#: Most corners off the mirror plane in a notch of the half mesh's cut closed against the plane.
SEAM_CORNERS = 8

#: Deepest notch of the half mesh's cut closed against the mirror plane, in lattice spacings.
SEAM_SPACINGS = 3.0

#: Lattice spacings from a source border within which the middle of a polygon side lies on it.
SIDE_SPACINGS = 0.5

#: Share of the target face count extracted before subdivision, per output kind.
_EXTRACT_SHARE = {"quads": 0.29, "quad dominant": 1.02, "triangles": 0.55}

#: Most a corner of two triangles joined into a quad may depart from 90 degrees, per output kind.
MERGE_DEVIATION = {"quads": 60.0, "quad dominant": 40.0, "triangles": 40.0}


@dataclass
class Retopology:
    """A retopologised mesh and the figures describing it.

    Attributes:
        mesh: The new mesh with polygons and carried attributes.
        figures: Named numbers: polygons, quads, irregular vertices and the rest.
    """

    mesh: TriMesh
    figures: dict[str, float] = field(default_factory=dict)


@dataclass
class _Layout:
    """Polygons laid over the working surface at one spacing field, before relaxation.

    Attributes:
        poly: The polygons.
        sub_v: ``(V, 3)`` working surface the fields were solved on.
        features: Its feature lines.
        levels: The hierarchy.
        finest: Index of the level read off.
        extracted: The extraction.
        counts: Repairs made, by name.
    """

    poly: polymesh.PolyMesh
    sub_v: torch.Tensor
    features: prepare.Features
    levels: list
    finest: int
    extracted: extract.Extraction
    counts: dict[str, int]


def _face_count(poly: polymesh.PolyMesh, output: str) -> int:
    """Faces a laid-out mesh gives the result: its polygons, or their triangles for ``"triangles"``."""
    if output == "triangles":
        return int((poly.sizes - 2).clamp_min(1).sum())
    return poly.polygon_count


def _segments_bvh(points: torch.Tensor, edges: torch.Tensor, device) -> TriangleBVH | None:
    """A tree over line segments, each held as a triangle with a repeated corner."""
    if edges.shape[0] == 0:
        return None
    count = int(edges.shape[0])
    ends = torch.cat([points[edges[:, 0]], points[edges[:, 1]]])
    index = torch.arange(count, device=points.device)
    faces = torch.stack([index, index + count, index + count], 1)
    return TriangleBVH(ends, faces, device=device)


def _on_plane(mesh: polymesh.PolyMesh, axis: int) -> torch.Tensor:
    """``(V,)`` bool, border vertices within `PLANE_SPACINGS` of the mirror plane."""
    src, dst, _ = mesh.half_edges()
    open_side = polymesh.twins(src, dst, mesh.count) < 0
    border = torch.zeros(mesh.count, dtype=torch.bool, device=src.device)
    border[src[open_side]] = True
    border[dst[open_side]] = True
    return border & (mesh.vertices[:, axis].abs() <= PLANE_SPACINGS * mesh.marks["scale"])


def _seal_seam(mesh: polymesh.PolyMesh, axis: int, opening: torch.Tensor,
               keep_size: int) -> tuple[polymesh.PolyMesh, int]:
    """The half mesh with its open sides along the mirror plane lying on the plane.

    Args:
        mesh: The half mesh, marked ``"scale"``.
        axis: Axis of the plane's normal.
        opening: ``(C,)`` bool per corner's side, sides with an opening of the source beyond them.
        keep_size: Most corners a polygon closing a notch keeps.

    Returns:
        ``(mesh, notches closed)``. A notch, where the border leaves the plane and returns, is
        closed by a polygon whose new side lies on the plane, when it has up to `SEAM_CORNERS`
        corners off the plane, none deeper than `SEAM_SPACINGS`, and no side along an opening.
        A corner on the plane held by one polygon between two corners on the plane is then
        taken out of that polygon.
    """
    sides, loop, rank, closed = polymesh.hole_loops(mesh)
    if sides.numel() == 0:
        return mesh, 0
    src = mesh.indices
    depth = (mesh.vertices[:, axis].abs() / mesh.marks["scale"].clamp_min(1e-30)).tolist()
    order = torch.argsort(loop * (sides.shape[0] + 1) + rank)
    starts = src[sides[order]].tolist()
    beyond = opening[sides[order]].tolist()
    owner = loop[order].tolist()
    shut = closed.tolist()
    added: list[list[int]] = []
    begin = 0
    while begin < len(starts):
        end = begin
        while end < len(starts) and owner[end] == owner[begin]:
            end += 1
        if shut[owner[begin]]:
            ring = starts[begin:end]
            ahead = beyond[begin:end]
            count = len(ring)
            anchors = [k for k in range(count) if depth[ring[k]] <= PLANE_SPACINGS]
            for k, first in enumerate(anchors):
                last = anchors[(k + 1) % len(anchors)]
                gap = (last - first) % count
                inside = [ring[(first + step) % count] for step in range(1, gap)]
                if not 1 <= len(inside) <= SEAM_CORNERS or ring[first] == ring[last]:
                    continue
                if max(depth[vertex] for vertex in inside) > SEAM_SPACINGS:
                    continue
                if any(ahead[(first + step) % count] for step in range(gap)):
                    continue
                added.append([ring[first], ring[last], *reversed(inside)])
        begin = end
    if added:
        sizes = torch.tensor([len(item) for item in added], dtype=mesh.sizes.dtype, device=mesh.sizes.device)
        corners = torch.tensor([vertex for item in added for vertex in item], dtype=src.dtype, device=src.device)
        mesh = replace(mesh, sizes=torch.cat([mesh.sizes, sizes]), indices=torch.cat([src, corners]))
        mesh = polymesh.split_large(mesh, keep_size)
    for _ in range(polymesh.DISSOLVE_PASSES):
        plane = mesh.vertices[:, axis].abs() <= PLANE_SPACINGS * mesh.marks["scale"]
        uses = torch.bincount(mesh.indices, minlength=mesh.count)
        following = mesh.indices[mesh.next_corner()]
        previous = torch.empty_like(mesh.indices)
        previous[mesh.next_corner()] = mesh.indices
        lone = plane[mesh.indices] & (uses[mesh.indices] == 1) & plane[following] & plane[previous]
        if not bool(lone.any()):
            break
        sizes = torch.zeros_like(mesh.sizes).index_add_(0, mesh.corner_polygon(), (~lone).long())
        mesh = replace(mesh, sizes=sizes, indices=mesh.indices[~lone])
        mesh = mesh.select(mesh.sizes >= 3).compacted()
    return mesh, len(added)


def _mirror(mesh: polymesh.PolyMesh, axis: int, near_plane: torch.Tensor) -> polymesh.PolyMesh:
    """The mesh joined to its mirror image across a coordinate plane, welded along the plane.

    Args:
        mesh: The half mesh, on the positive side.
        axis: Axis of the plane's normal.
        near_plane: ``(V,)`` bool, vertices lying on the plane, shared by both halves.

    Returns:
        The whole mesh, with the shared vertices marked ``"centre"``.
    """
    device = mesh.vertices.device
    count = mesh.count
    points = mesh.vertices.clone()
    points[near_plane, axis] = 0.0
    copy = points.clone()
    copy[:, axis] = -copy[:, axis]
    copied = torch.where(near_plane, torch.arange(count, device=device), count + torch.arange(count, device=device))
    offsets = mesh.offsets()
    polygon = mesh.corner_polygon()
    within = torch.arange(mesh.indices.shape[0], device=device) - offsets[polygon]
    reverse = offsets[polygon] + mesh.sizes[polygon] - 1 - within
    mirrored = copied[mesh.indices[reverse]]
    marks = {name: torch.cat([value, value]) for name, value in mesh.marks.items()}
    marks["centre"] = torch.cat([near_plane, near_plane])
    whole = polymesh.PolyMesh(torch.cat([points, copy]), torch.cat([mesh.sizes, mesh.sizes]),
                              torch.cat([mesh.indices, mirrored]), marks)
    return whole.compacted()


def _open_sides(vertices: torch.Tensor, faces: torch.Tensor, min_length: float) -> torch.Tensor:
    """Triangle sides no other triangle shares, on border runs at least ``min_length`` long.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` vertex indices.
        min_length: Shortest run of joined open sides kept, in model units.

    Returns:
        ``(K, 2)`` int64 sides in their triangle's winding.
    """
    src = faces.reshape(-1)
    dst = faces[:, [1, 2, 0]].reshape(-1)
    count = int(vertices.shape[0])
    stride = max(count, 1)
    _, inverse, uses = torch.unique(torch.minimum(src, dst) * stride + torch.maximum(src, dst),
                                    return_inverse=True, return_counts=True)
    alone = uses[inverse] == 1
    src, dst = src[alone], dst[alone]
    if src.numel() == 0:
        return torch.stack([src, dst], 1)
    run = topology.union_find(count, src, dst)[src]
    length = (vertices[dst] - vertices[src]).norm(dim=1).double()
    total = torch.zeros(count, dtype=torch.float64, device=vertices.device).index_add_(0, run, length)
    kept = total[run] >= float(min_length)
    return torch.stack([src[kept], dst[kept]], 1)


def _heading(mesh: polymesh.PolyMesh, points: torch.Tensor, sides: torch.Tensor,
             sides_bvh: TriangleBVH | None) -> torch.Tensor:
    """How each polygon side runs along the open sides of the source.

    Args:
        mesh: The mesh, marked ``"scale"``, polygons wound like the source.
        points: ``(V, 3)`` source positions.
        sides: ``(K, 2)`` int64 open sides of the source, in their triangle's winding.
        sides_bvh: Tree over those sides, segment for segment, or None when there are none.

    Returns:
        ``(C,)`` int64 per corner's side: 1 along a source border in its direction, -1 along
        one against it, 0 away from every border.
    """
    heading = torch.zeros(mesh.indices.shape[0], dtype=torch.int64, device=mesh.indices.device)
    if sides_bvh is None or heading.numel() == 0:
        return heading
    src, dst, _ = mesh.half_edges()
    start, end = mesh.vertices[src], mesh.vertices[dst]
    distance, segment, _, _ = sides_bvh.closest_point((start + end) / 2)
    spacing = (mesh.marks["scale"][src] + mesh.marks["scale"][dst]) / 2
    near = (segment >= 0) & (distance.to(spacing.dtype) <= SIDE_SPACINGS * spacing)
    chosen = sides[segment.clamp_min(0)]
    turn = ((end - start) * (points[chosen[:, 1]] - points[chosen[:, 0]]).to(start.dtype)).sum(1)
    return torch.where(near, torch.sign(turn).long(), heading)


def _drop_caps(mesh: polymesh.PolyMesh, heading: torch.Tensor) -> tuple[polymesh.PolyMesh, int]:
    """The mesh without polygons spanning an opening of the source surface.

    Args:
        mesh: The mesh.
        heading: ``(C,)`` int64 per corner's side, as `_heading` gives.

    Returns:
        ``(mesh, polygons removed)``. A polygon spans an opening when at least `polymesh.OPENING_SIDES` of
        its sides run along a source border against that border's direction, and more of
        them do so than run with it.
    """
    if mesh.polygon_count == 0:
        return mesh, 0
    polygon = mesh.corner_polygon()
    against = torch.zeros(mesh.polygon_count, dtype=torch.int64, device=polygon.device)
    against.index_add_(0, polygon, (heading < 0).long())
    along = torch.zeros_like(against).index_add_(0, polygon, (heading > 0).long())
    cap = (against >= polymesh.OPENING_SIDES) & (against > along)
    if not bool(cap.any()):
        return mesh, 0
    return mesh.select(~cap).compacted(), int(cap.sum())


def _clean_normals(normals: torch.Tensor, vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Vertex normals with any zero normal replaced by the normal of a face at that vertex."""
    zero = normals.norm(dim=1) < 0.5
    if not bool(zero.any()):
        return normals
    face_normal = topology.face_normals(vertices, faces)
    spare = torch.zeros_like(normals)
    spare[faces.reshape(-1)] = face_normal.repeat_interleave(3, dim=0)
    spare[spare.norm(dim=1) < 0.5] = torch.tensor([0.0, 1.0, 0.0], dtype=normals.dtype, device=normals.device)
    return torch.where(zero[:, None], spare, normals)


def mesh_figures(poly: polymesh.PolyMesh) -> dict[str, float]:
    """Layout figures of a polygon mesh.

    Args:
        poly: The mesh.

    Returns:
        Polygons, quads, triangles, quad share, irregular vertices, valence counts, the mean and
        95th percentile corner departure from 90 degrees, border and non-manifold edges, and the
        counts of `polymesh.flaws`.
    """
    sizes = poly.sizes
    edges, side_edge = poly.edges()
    edge_uses = torch.bincount(side_edge, minlength=edges.shape[0])
    border_edge = edge_uses == 1
    on_border = torch.zeros(poly.count, dtype=torch.bool, device=poly.vertices.device)
    on_border[edges[border_edge].reshape(-1)] = True
    degree = torch.bincount(edges.reshape(-1), minlength=poly.count)
    used = torch.bincount(poly.indices, minlength=poly.count) > 0
    interior = used & ~on_border
    irregular = (interior & (degree != 4)) | (used & on_border & (degree != 3))
    figures: dict[str, float] = {
        "polygons": int(poly.polygon_count),
        "quads": int((sizes == 4).sum()),
        "triangles": int((sizes == 3).sum()),
        "other polygons": int((sizes > 4).sum()),
        "quad share": round(float((sizes == 4).float().mean()) if poly.polygon_count else 0.0, 4),
        "vertices": int(used.sum()),
        "irregular vertices": int(irregular.sum()),
        "border edges": int(border_edge.sum()),
        "non-manifold edges": int((edge_uses > 2).sum()),
    }
    for valence in (3, 4, 5, 6):
        figures[f"valence {valence}"] = int((interior & (degree == valence)).sum())
    figures["valence 7+"] = int((interior & (degree >= 7)).sum())
    figures.update(polymesh.flaws(poly))
    quads = torch.nonzero(sizes == 4).squeeze(1)
    if quads.numel():
        corners = poly.indices[poly.offsets()[quads][:, None] + torch.arange(4, device=sizes.device)]
        points = poly.vertices[corners]
        off = (polymesh.corner_angles(points, polymesh.polygon_normals(points)) - math.pi / 2).abs().reshape(-1)
        off = torch.rad2deg(off).float()
        figures["angle error mean"] = round(float(off.mean()), 2)
        figures["angle error p95"] = round(float(torch.quantile(off[:1_000_000], 0.95)), 2)
    return figures


def _check(target_faces: int, output: str, symmetry: str, uvs: str) -> None:
    """Refuse options outside what the retopology accepts."""
    if output not in OUTPUTS:
        raise ValueError(f"output must be one of {', '.join(OUTPUTS)}, not {output!r}")
    if symmetry not in SYMMETRIES:
        raise ValueError(f"symmetry must be one of {', '.join(SYMMETRIES)}, not {symmetry!r}")
    if uvs not in UV_MODES:
        raise ValueError(f"uvs must be one of {', '.join(UV_MODES)}, not {uvs!r}")
    if not 1 <= int(target_faces) <= MAX_TARGET_FACES:
        raise ValueError(f"target_faces must be between 1 and {MAX_TARGET_FACES}, not {target_faces}")


def retopologize(mesh: TriMesh, target_faces: int, output: str = "quads", crease_angle: float = 0.0,
                 keep_borders: bool = True, symmetry: str = "none", adaptivity: float = 0.0,
                 smoothing: int = 3, uvs: str = "drop", device: torch.device | str | None = None) -> Retopology:
    """Rebuild a triangle mesh as quads following its shape, features and borders.

    Args:
        mesh: The source mesh and its attributes.
        target_faces: Faces wanted in the result.
        output: One of `OUTPUTS`.
        crease_angle: Dihedral angle in degrees from which an edge is a crease the layout
            follows; 0 = none.
        keep_borders: Whether open borders are followed exactly.
        symmetry: ``"none"`` or ``"x"``, a layout mirrored across the plane ``x = 0``.
        adaptivity: 0 = even quads; 1 = quads down to a quarter the size where the surface
            curves tightly.
        smoothing: Relaxation passes after extraction.
        uvs: ``"drop"`` or ``"transfer"``.
        device: Device to run on; None means the mesh compute device.

    Returns:
        The `Retopology`.

    Raises:
        ValueError: An option is out of range, the mesh has no triangles or area, symmetry
            finds nothing on the positive side, or the work would grow too large.
    """
    _check(target_faces, output, symmetry, uvs)
    if mesh.face_count == 0:
        raise ValueError("the mesh has no triangles to retopologise")
    if mesh.face_count > MAX_SOURCE_TRIANGLES:
        raise ValueError(f"the mesh has {mesh.face_count} triangles, more than the {MAX_SOURCE_TRIANGLES} "
                         "a retopology accepts; decimate it first")
    device = torch.device(device) if device is not None else mesh_device.compute_device()
    started = time.perf_counter()
    generator = torch.Generator().manual_seed(SEED)
    source = mesh.to(device)
    welded, all_faces, _ = prepare.surface(source.vertices, source.faces)
    faces = prepare.orient(welded, prepare.clean_faces(all_faces))
    if faces.shape[0] == 0:
        raise ValueError("every triangle of the mesh is collapsed to a line or a point")
    even = math.sqrt(max(float(topology.face_areas(welded.double(), faces).sum()), 0.0)
                     / (float(target_faces) * _EXTRACT_SHARE[output]))
    carried = uvs == "transfer" or any(name in ("skin", "colors") or name.startswith(MORPH_PREFIX)
                                       for name in source.attributes)
    source_bvh = TriangleBVH(welded, all_faces, device=device) if carried else None
    welded, faces, small_holes = prepare.close_small_holes(welded, faces, prepare.SMALL_HOLE_SPACINGS * even)
    surface_bvh = TriangleBVH(welded, faces, device=device)
    surface_normals = topology.face_normals(welded, faces)
    axis = 0 if symmetry == "x" else None
    if axis is not None:
        work_v, work_f = prepare.cut_half(welded, faces, axis)
        if work_f.shape[0] == 0:
            raise ValueError("symmetry x found nothing on the +X side of the model; centre the model on "
                             "X = 0 or set symmetry to none")
    else:
        work_v, work_f = prepare.compact(welded, faces)
    wanted = float(target_faces) * _EXTRACT_SHARE[output] * (0.5 if axis is not None else 1.0)
    open_sides = _open_sides(work_v, work_f, prepare.SMALL_HOLE_SPACINGS * even)
    open_bvh = _segments_bvh(work_v, open_sides, device)
    border = open_sides
    if axis is not None:
        border = border[(work_v[border[:, 0], axis] != 0) | (work_v[border[:, 1], axis] != 0)]
    border_bvh = _segments_bvh(work_v, border, device)
    face_piece = topology.face_components(faces, welded.shape[0])

    def source_border(mesh: polymesh.PolyMesh) -> torch.Tensor:
        """``(V,)`` bool, vertices on a border the source has or on the symmetry plane."""
        keep = torch.zeros(mesh.count, dtype=torch.bool, device=device)
        if border_bvh is not None and mesh.count:
            distance, _, _, _ = border_bvh.closest_point(mesh.vertices)
            keep = distance.to(mesh.vertices.dtype) <= BORDER_SPACINGS * mesh.marks["scale"]
        if axis is not None and mesh.count:
            keep |= mesh.vertices[:, axis].abs() <= PLANE_SPACINGS * mesh.marks["scale"]
        return keep

    def source_piece(mesh: polymesh.PolyMesh) -> torch.Tensor:
        """``(V,)`` int64, piece of the source surface closest to each vertex, -1 where none."""
        _, face, _, _ = surface_bvh.closest_point(mesh.vertices)
        return torch.where(face >= 0, face_piece[face.clamp_min(0)], torch.full_like(face, -1))

    def opening(mesh: polymesh.PolyMesh) -> torch.Tensor:
        """``(C,)`` bool, polygon sides running along a source border with an opening beyond them."""
        return _heading(mesh, work_v, open_sides, open_bvh) > 0

    def layout(scales: torch.Tensor) -> _Layout:
        """The polygons laid over the working surface at a spacing field, before relaxation."""
        sub_v, sub_f, sub_s = prepare.subdivide(work_v, work_f, scales, MAX_WORK_VERTICES)
        features = prepare.features(sub_v, sub_f, sub_s, crease_angle, keep_borders, axis)
        normals = _clean_normals(topology.vertex_normals(sub_v, sub_f), sub_v, sub_f)
        areas = topology.vertex_areas(sub_v, sub_f, kind="barycentric")
        constraints = hierarchy.Constraints.empty(sub_v)
        if features.count:
            tangent = features.direction - normals * (features.direction * normals).sum(1, keepdim=True)
            constraints.direction = torch.nn.functional.normalize(tangent, dim=1)
            constraints.direction_weight = features.line.to(sub_v.dtype)
            constraints.line = features.direction.clone()
            constraints.point_weight = (features.line | features.corner).to(sub_v.dtype)
        edges = topology.edge_topology(sub_f, sub_v.shape[0]).edges
        levels = hierarchy.build(sub_v, normals, areas, sub_s, edges, constraints, generator)
        finest = smooth_from = 0
        for index, level in enumerate(levels[1:], start=1):
            if level.first.numel() == 0:
                break
            edge = hierarchy.mean_edge_length(level)
            spacing = float(level.scales.median())
            if edge > NORMAL_SPACINGS * spacing:
                break
            smooth_from = index
            if edge <= EXTRACT_SPACINGS * spacing:
                finest = index
        hierarchy.smooth_normals(levels, finest, max(smooth_from, finest))
        for index in range(finest, len(levels)):
            hierarchy.colour(levels[index], generator)
        fields.solve_orientation(levels, finest, generator)
        fields.solve_position(levels, finest, len(levels) - 1, generator)
        extracted = extract.extract(levels[finest])
        keep_size = 5 if output == "quads" else 4
        poly, caps = _drop_caps(extracted.mesh, _heading(extracted.mesh, work_v, open_sides, open_bvh))
        poly = polymesh.split_large(poly, keep_size)
        poly = polymesh.dissolve_valence_two(poly)
        filled = crowded = strays = 0
        for _ in range(REPAIR_PASSES):
            poly, cut_loose = polymesh.drop_fragments(poly, source_piece(poly))
            poly, unpinched = polymesh.open_pinches(poly)
            poly, closed = polymesh.fill_holes(poly, source_border(poly), opening=opening(poly))
            poly = polymesh.split_large(poly, keep_size)
            poly = polymesh.dissolve_valence_two(poly)
            poly, dropped = polymesh.drop_non_manifold(poly)
            poly, folds = polymesh.drop_folds(poly)
            poly, loose = polymesh.drop_stray_shells(poly, source_border(poly))
            filled, strays = filled + closed, strays + loose + cut_loose
            crowded += dropped + unpinched + folds
            if dropped == 0 and folds == 0 and loose == 0 and cut_loose == 0 and unpinched == 0 and closed == 0:
                break
        if output != "triangles":
            fold = min(60.0, float(crease_angle)) if crease_angle > 0 else 60.0
            poly = polymesh.merge_triangles(poly, max_deviation_deg=MERGE_DEVIATION[output], max_fold_deg=fold)
            poly = polymesh.dissolve_valence_two(poly)
        notches = 0
        if axis is not None:
            poly, notches = _seal_seam(poly, axis, _heading(poly, work_v, border, border_bvh) > 0, keep_size)
            poly = polymesh.dissolve_valence_two(poly)
            poly = replace(poly, marks={**poly.marks, "centre": _on_plane(poly, axis)})
        if output == "quads":
            poly = polymesh.subdivide_quads(poly)
        if axis is not None:
            centre = poly.marks["centre"] & _on_plane(poly, axis)
            points = poly.vertices.clone()
            points[centre, axis] = 0.0
            poly = replace(poly, vertices=points, marks={**poly.marks, "centre": centre,
                                                           "feature": poly.marks["feature"] | centre})
        counts = {"holes filled": filled, "seam notches closed": notches, "openings uncovered": caps,
                  "crowded polygons removed": crowded, "stray pieces removed": strays}
        return _Layout(poly, sub_v, features, levels, finest, extracted, counts)

    def made(laid: _Layout) -> float:
        """Faces a layout gives the result, as a share of the target."""
        return _face_count(laid.poly, output) * (2.0 if axis is not None else 1.0) / float(target_faces)

    scales = prepare.scale_field(work_v, work_f, wanted, generator, adaptivity, MIN_COMPONENT_POLYGONS)
    laid = layout(scales)
    passes = 1
    if abs(made(laid) - 1.0) > COUNT_TOLERANCE:
        try:
            again = layout(scales * math.sqrt(made(laid)))
        except ValueError:
            again = None
        if again is not None:
            passes = 2
            if abs(made(again) - 1.0) < abs(made(laid) - 1.0):
                laid = again
    feature_bvh = _segments_bvh(laid.sub_v, laid.features.edges, device)
    poly = polymesh.relax(laid.poly, surface_bvh, feature_bvh, smoothing, plane_axis=axis,
                          surface_normals=surface_normals)
    if axis is not None:
        poly = _mirror(poly, axis, poly.marks["centre"])
    if output == "triangles":
        poly = polymesh.triangles_only(poly)
    figures = mesh_figures(poly)
    result = attributes.carry(source, poly, source_bvh, uvs=uvs, crease_angle=crease_angle,
                              keep_polygons=output != "triangles")
    figures.update({
        "working vertices": int(laid.sub_v.shape[0]),
        "levels": len(laid.levels),
        "extraction level": laid.finest,
        "orientation error deg": round(fields.orientation_energy(laid.levels[laid.finest]), 3),
        "feature edges": laid.features.count,
        **laid.counts,
        "layout passes": passes,
        "source holes covered": small_holes,
        "seconds": round(time.perf_counter() - started, 3),
    })
    figures.update({f"extract {name}": value for name, value in laid.extracted.figures.items()})
    return Retopology(mesh=result, figures=figures)


def _strip_textures(part):
    """A copy of a part whose material keeps its factors and drops every texture."""
    material = replace(part.material, base_color_texture=None, metallic_roughness_texture=None,
                       normal_texture=None, occlusion_texture=None, emissive_texture=None,
                       extras=dict(part.material.extras))
    material.extras.pop("texture_transform", None)
    material.extras.pop("mtoon", None)
    return replace(part, material=material)


def _symmetric_about_zero(part, axis: int = 0) -> bool:
    """Whether a part straddles the plane through the origin evenly enough to mirror."""
    low, high = part.bounds()
    width = float(high[axis] - low[axis])
    return width > 0 and float(low[axis]) < 0 < float(high[axis]) and abs(float(low[axis] + high[axis])) <= 0.25 * width


#: Figures summed across parts; the rest are recomputed for the whole model.
_SUMMED = (
    "polygons", "quads", "triangles", "other polygons", "vertices", "irregular vertices", "border edges",
    "non-manifold edges", "valence 3", "valence 4", "valence 5", "valence 6", "valence 7+", "working vertices",
    "feature edges", "holes filled", "seconds", "duplicate polygons", "folded polygons", "non-manifold vertices",
    "degenerate polygons",
)


def retopologize_model(model: Model3D, parts: str, target_faces: int, progress=None,
                       **options) -> tuple[Model3D, dict]:
    """Retopologise the chosen parts of a model in place, sharing the face budget by area.

    Args:
        model: The model, changed in place.
        parts: Comma-separated part names or labels, wildcards allowed; empty chooses all.
        target_faces: Faces wanted across the chosen parts.
        progress: Called with the share of parts done after each part, or None.
        **options: Passed to `retopologize`.

    Returns:
        ``(model, summary)``. The summary holds ``"parts"`` (names), ``"figures"`` (across
        the parts), ``"per_part"`` (name, figures), ``"unmirrored"`` and ``"untextured"``
        (names of parts left unmirrored, and of parts whose textures were dropped).

    Raises:
        ValueError: No part is chosen, the chosen parts hold no triangles or no area, an
            option is out of range, or a part fails.
    """
    chosen = selection.selected(model, parts)
    if not chosen:
        names = ", ".join(part.name or "(unnamed)" for part in model.parts[:12])
        raise ValueError(f"no part matches parts '{str(parts)[:80]}'; the model's parts are {names}. Leave "
                         "parts empty to retopologise every part")
    chosen = [index for index in chosen if model.parts[index].triangle_count > 0]
    if not chosen:
        raise ValueError("the chosen parts hold no triangles to retopologise; wire a model with faces or "
                         "choose other parts")
    _check(target_faces, options.get("output", "quads"), options.get("symmetry", "none"),
           options.get("uvs", "drop"))
    for index in chosen:
        if model.parts[index].triangle_count > MAX_SOURCE_TRIANGLES:
            raise ValueError(f"part '{model.parts[index].name}' has {model.parts[index].triangle_count:,} "
                             f"triangles, more than the {MAX_SOURCE_TRIANGLES:,} one retopology takes; "
                             "decimate it first")
    areas = [float(topology.face_areas(model.parts[index].vertices.double(), model.parts[index].faces).sum())
             for index in chosen]
    total_area = sum(areas)
    if not total_area > 0 or not math.isfinite(total_area):
        raise ValueError("the chosen parts have no surface area to retopologise; check that they hold real "
                         "triangles rather than lines or points")
    joint_count = model.skeleton.count if model.skeleton is not None else 0
    per_part = []
    unmirrored, untextured = [], []
    for done, (index, area) in enumerate(zip(chosen, areas)):
        part = model.parts[index]
        name = part.name or f"part {index}"
        share = min(MAX_TARGET_FACES, max(MIN_PART_FACES, round(target_faces * area / total_area)))
        part_options = dict(options)
        if part_options.get("symmetry", "none") != "none" and not _symmetric_about_zero(part):
            part_options["symmetry"] = "none"
            unmirrored.append(name)
        source = from_part(part, joint_count if part.weights is not None else 0)
        result = mesh_device.with_cpu_fallback(
            lambda device, source=source, share=share, part_options=part_options: retopologize(
                source, share, device=device, **part_options),
            label="mesh retopology",
        )
        new_part = to_part(part, result.mesh)
        new_part.tangents = None
        if new_part.uvs is None and part.material.textures():
            new_part = _strip_textures(new_part)
            untextured.append(name)
        model.parts[index] = new_part
        per_part.append((name, result.figures))
        if progress is not None:
            progress((done + 1) / len(chosen))
    totals: dict[str, float] = {name: 0 for name in _SUMMED}
    angle_weight = angle_sum = 0.0
    for _, figures in per_part:
        for name in _SUMMED:
            totals[name] += figures.get(name, 0)
        angle_sum += figures.get("angle error mean", 0.0) * figures.get("quads", 0)
        angle_weight += figures.get("quads", 0)
    totals["seconds"] = round(totals["seconds"], 3)
    totals["quad share"] = round(totals["quads"] / totals["polygons"], 4) if totals["polygons"] else 0.0
    totals["angle error mean"] = round(angle_sum / angle_weight, 2) if angle_weight else 0.0
    summary = {
        "parts": [name for name, _ in per_part], "figures": totals, "per_part": per_part,
        "unmirrored": unmirrored, "untextured": untextured,
    }
    return model, summary

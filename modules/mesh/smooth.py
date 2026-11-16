"""Smoothing of a triangle surface: Laplacian, Taubin, HC, implicit cotangent and bilateral normals.

Positions move on the welded surface. Other per-vertex attributes are kept; normals are recomputed
and tangents turned to stay at right angles to them.
"""

from __future__ import annotations

__all__ = [
    "METHODS",
    "orthogonal_tangents",
    "recompute_normals",
    "smooth",
    "smooth_part",
]

import math
import time
import warnings
from dataclasses import replace

import torch

from ..model3d.scene import Part
from . import device as mesh_device
from . import topology
from .arrays import TriMesh

#: Methods `smooth` offers.
METHODS = ("laplacian", "taubin", "hc", "implicit", "bilateral")

#: Most smoothing iterations.
MAX_ITERATIONS = 500

#: Pass-band frequency of Taubin smoothing.
TAUBIN_PASS_BAND = 0.1

#: Weight of the original positions in HC smoothing.
HC_ALPHA = 0.1

#: Share of the push back towards the previous step in HC smoothing.
HC_BETA = 0.6

#: Vertex updates run after each bilateral normal pass.
BILATERAL_VERTEX_STEPS = 10

#: Angle in degrees past which the normals of two seam sides make a hard edge.
HARD_EDGE_DEGREES = 10.0

#: Most conjugate gradient steps per implicit smoothing iteration.
IMPLICIT_CG_STEPS = 400

#: Mean edge lengths a crease line must reach for keep_features to hold it.
MIN_LINE_EDGES = 4.0

#: Vertex classes.
FREE, LINE, HELD = 0, 1, 2


def orthogonal_tangents(tangents: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    """Tangents turned to lie at right angles to the normals, handedness kept.

    Args:
        tangents: ``(N, 4)`` tangents with handedness in ``w``.
        normals: ``(N, 3)`` unit normals.

    Returns:
        ``(N, 4)`` float32 unit tangents.
    """
    direction = tangents[:, :3].float()
    normals = normals.float()
    direction = direction - (direction * normals).sum(1, keepdim=True) * normals
    length = direction.norm(dim=1, keepdim=True)
    fallback = torch.where(normals[:, :1].abs() < 0.9, torch.tensor([[1.0, 0.0, 0.0]]),
                           torch.tensor([[0.0, 1.0, 0.0]]))
    fallback = fallback - (fallback * normals).sum(1, keepdim=True) * normals
    direction = torch.where(length > 1e-8, direction, fallback)
    direction = torch.nn.functional.normalize(direction, dim=1)
    sign = torch.where(tangents[:, 3:4] < 0, -1.0, 1.0).float()
    return torch.cat([direction, sign], dim=1)


def _hard_edges(faces, welded_of, topo, normals, degrees: float) -> torch.Tensor:
    """Welded edges where the normals on the two sides disagree at an end."""
    hard = torch.zeros(topo.edge_count, dtype=torch.bool, device=faces.device)
    if normals is None:
        return hard
    seams = topology.seam_edges(faces, welded_of, topo) & (topo.edge_face_count == 2)
    chosen = torch.nonzero(seams).squeeze(1)
    if chosen.numel() == 0:
        return hard
    f0, f1 = topo.edge_faces[chosen, 0], topo.edge_faces[chosen, 1]
    k0, k1 = topo.edge_corners[chosen, 0], topo.edge_corners[chosen, 1]
    start, end = faces[f0, k0], faces[f0, (k0 + 1) % 3]
    other_start, other_end = faces[f1, k1], faces[f1, (k1 + 1) % 3]
    same = welded_of[other_start] == welded_of[start]
    match_start = torch.where(same, other_start, other_end)
    match_end = torch.where(same, other_end, other_start)
    unit = torch.nn.functional.normalize(normals.float(), dim=1)
    cosine = math.cos(math.radians(degrees))
    differs = ((unit[start] * unit[match_start]).sum(1) < cosine) | ((unit[end] * unit[match_end]).sum(1) < cosine)
    hard[chosen[differs]] = True
    return hard


def recompute_normals(vertices: torch.Tensor, faces: torch.Tensor, welded_of: torch.Tensor,
                      normals: torch.Tensor | None = None, hard_degrees: float = HARD_EDGE_DEGREES) -> torch.Tensor:
    """Angle-weighted vertex normals of a split mesh, smooth across seams unless the given normals
    part there.

    Args:
        vertices: ``(N, 3)`` split positions.
        faces: ``(F, 3)`` triangles on the split vertices.
        welded_of: ``(N,)`` int64 welded vertex of each split vertex.
        normals: ``(N, 3)`` normals whose disagreement across a seam marks a hard edge, or None.
        hard_degrees: Angle between two sides' normals that makes an edge hard.

    Returns:
        ``(N, 3)`` float32 unit normals.
    """
    faces = faces.long()
    welded_of = welded_of.long()
    welded_faces = welded_of[faces]
    welded_count = int(welded_of.max()) + 1 if welded_of.numel() else 0
    topo = topology.edge_topology(welded_faces, welded_count)
    hard = _hard_edges(faces, welded_of, topo, normals, hard_degrees)
    cut = replace(topo, edge_face_count=torch.where(hard, torch.full_like(topo.edge_face_count, 3),
                                                    topo.edge_face_count))
    fans = topology.corner_fans(welded_faces, welded_count, cut)
    positions = vertices.double()
    weighted = (topology.face_normals(positions, faces)[:, None, :]
                * topology.corner_angles(positions, faces)[..., None]).reshape(-1, 3)
    total = torch.zeros_like(weighted)
    total.index_add_(0, fans, weighted)
    corner = torch.nn.functional.normalize(total[fans], dim=1)
    result = torch.zeros(vertices.shape[0], 3, dtype=torch.float64, device=vertices.device)
    result[faces.reshape(-1)] = corner
    missing = result.norm(dim=1) == 0
    result[missing] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64, device=vertices.device)
    return result.float()


def _classes(positions, faces, topo, keep_borders: bool, keep_features: bool, feature_angle: float):
    """Class of each welded vertex, and the edges a line vertex slides along."""
    from .remesh import _chains

    count = int(positions.shape[0])
    device = positions.device
    angle = float(feature_angle) if keep_features else 180.0
    shortest = MIN_LINE_EDGES * float(topology.edge_lengths(positions, topo.edges).mean()) if topo.edge_count else 0.0
    edge_chain, corner, _ = _chains(positions, faces, topo, angle,
                                    torch.zeros(topo.edge_count, dtype=torch.bool, device=device), shortest)
    line = edge_chain >= 0
    along = torch.bincount(topo.edges[line].reshape(-1), minlength=count)
    held = corner | (along >= 3) | (along == 1)
    held |= topology.non_manifold_vertices(faces, count, topo)
    if keep_borders:
        held[topo.edges[topo.boundary].reshape(-1)] = True
    kind = torch.zeros(count, dtype=torch.int8, device=device)
    kind[(along == 2) & ~held] = LINE
    kind[held] = HELD
    used = torch.zeros(count, dtype=torch.bool, device=device)
    used[faces.reshape(-1)] = True
    kind[~used] = HELD
    return kind, line


def _umbrella_links(topo, kind, line):
    """Directed ``(row, col)`` links each vertex averages over."""
    edges = topo.edges
    rows = torch.cat([edges[:, 0], edges[:, 1]])
    cols = torch.cat([edges[:, 1], edges[:, 0]])
    on_line = torch.cat([line, line])
    keep = (kind[rows] == FREE) | ((kind[rows] == LINE) & on_line)
    return rows[keep], cols[keep]


def _umbrella(values: torch.Tensor, rows: torch.Tensor, cols: torch.Tensor) -> torch.Tensor:
    """Mean of the linked values minus each value, zero for a vertex with no links."""
    total = torch.zeros_like(values)
    total.index_add_(0, rows, values[cols])
    counts = torch.zeros(values.shape[0], dtype=values.dtype, device=values.device)
    counts.index_add_(0, rows, torch.ones(rows.shape[0], dtype=values.dtype, device=values.device))
    has = counts > 0
    mean = total / counts.clamp_min(1)[:, None]
    return torch.where(has[:, None], mean - values, torch.zeros_like(values))


def _implicit(positions, faces, kind, strength: float, iterations: int):
    """Backward Euler steps of cotangent flow with held vertices fixed."""
    free = (kind == FREE).to(positions.dtype)[:, None]
    held = 1.0 - free
    edges = topology.edge_topology(faces, positions.shape[0]).edges
    mean_sq = float(topology.edge_lengths(positions, edges).pow(2).mean()) if edges.numel() else 0.0
    step = float(strength) * mean_sq
    current = positions
    for _ in range(iterations):
        operator = topology.cotangent_laplacian(current, faces, nonnegative=True)
        index = operator.indices()
        on = index[0] == index[1]
        own = torch.zeros(current.shape[0], dtype=current.dtype, device=current.device)
        own.index_add_(0, index[0][on], operator.values()[on])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            laplace = operator.to_sparse_csr()
        mass = topology.vertex_areas(current, faces).clamp_min(1e-30)[:, None]
        diagonal = mass[:, 0] + step * own

        def apply(value, laplace=laplace, mass=mass):
            moving = value * free
            return free * (mass * moving + step * (laplace @ moving)) + held * value

        # Solve for each free vertex's move rather than its new position.
        rhs = -step * free * (laplace @ current)
        inverse = torch.where(free[:, 0] > 0, 1.0 / diagonal, torch.ones_like(diagonal))
        move = topology.solve_cg(apply, rhs, x0=torch.zeros_like(current), tol=1e-6, max_iter=IMPLICIT_CG_STEPS,
                                 preconditioner=inverse)
        current = current + free * move
    return current


def _bilateral(positions, faces, topo, kind, strength: float, iterations: int):
    """Bilateral filtering of face normals, then vertices moved to agree with them."""
    current = positions
    ids, pairs = topo.manifold_face_pairs()
    first = torch.cat([pairs[:, 0], pairs[:, 1], torch.arange(faces.shape[0], device=faces.device)])
    second = torch.cat([pairs[:, 1], pairs[:, 0], torch.arange(faces.shape[0], device=faces.device)])
    sigma_normal = 0.1 + 0.5 * float(strength)
    movable = (kind != HELD)[:, None].to(current.dtype)
    counts = torch.bincount(faces.reshape(-1), minlength=current.shape[0]).clamp_min(1).to(current.dtype)
    for _ in range(iterations):
        corners = current[faces]
        centre = corners.mean(1)
        raw = torch.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], dim=1)
        area = 0.5 * raw.norm(dim=1)
        normal = torch.nn.functional.normalize(raw, dim=1)
        gap = (centre[first] - centre[second]).norm(dim=1)
        sigma_centre = float(gap[: 2 * pairs.shape[0]].mean()) if pairs.shape[0] else 1.0
        weight = (area[second] * torch.exp(-gap.pow(2) / (2.0 * sigma_centre ** 2))
                  * torch.exp(-(normal[first] - normal[second]).norm(dim=1).pow(2) / (2.0 * sigma_normal ** 2)))
        filtered = torch.zeros_like(normal)
        filtered.index_add_(0, first, weight[:, None] * normal[second])
        filtered = torch.nn.functional.normalize(filtered, dim=1)
        for _ in range(BILATERAL_VERTEX_STEPS):
            centre = current[faces].mean(1)
            move = torch.zeros_like(current)
            for k in range(3):
                offset = ((centre - current[faces[:, k]]) * filtered).sum(1, keepdim=True) * filtered
                move.index_add_(0, faces[:, k], offset)
            current = current + movable * move / counts[:, None]
    return current


def _caps(faces, topo, count: int):
    """The border edges of a mesh as their faces walk them, and the loop each belongs to.

    Returns:
        ``(start, end, loop, face, labels)``: the border edges as their faces walk them, the loop of
        each, the face beside each, and the loop label of every vertex.
    """
    border = torch.nonzero(topo.boundary).squeeze(1)
    face = topo.edge_faces[border, 0]
    corner = topo.edge_corners[border, 0]
    start, end = faces[face, corner], faces[face, (corner + 1) % 3]
    labels = topology.union_find(count, start, end)
    return start, end, labels[start], face, labels


def _volumes(positions, faces, piece, topo=None):
    """Signed volume and area of each piece, open borders closed by fans to their loop centres."""
    pieces = int(piece.max()) + 1 if piece.numel() else 0
    corners = positions[faces]
    centre = torch.zeros(pieces, 3, dtype=positions.dtype, device=positions.device)
    centre.index_add_(0, piece, corners.mean(1))
    members = torch.bincount(piece, minlength=pieces).clamp_min(1).to(positions.dtype)
    centre = centre / members[:, None]
    local = corners - centre[piece][:, None, :]
    volume = torch.zeros(pieces, dtype=positions.dtype, device=positions.device)
    volume.index_add_(0, piece, (local[:, 0] * torch.cross(local[:, 1], local[:, 2], dim=1)).sum(1) / 6.0)
    if topo is not None and bool(topo.boundary.any()):
        start, end, loop, face, labels = _caps(faces, topo, int(positions.shape[0]))
        middle = torch.zeros(positions.shape[0], 3, dtype=positions.dtype, device=positions.device)
        middle.index_add_(0, loop, positions[start])
        sizes = torch.bincount(loop, minlength=positions.shape[0]).clamp_min(1).to(positions.dtype)
        middle = middle / sizes[:, None]
        origin = centre[piece[face]]
        a, b, c = positions[end] - origin, positions[start] - origin, middle[loop] - origin
        volume.index_add_(0, piece[face], (a * torch.cross(b, c, dim=1)).sum(1) / 6.0)
    area = torch.zeros(pieces, dtype=positions.dtype, device=positions.device)
    area.index_add_(0, piece, topology.face_areas(positions, faces))
    return volume, area


def _restore_volume(positions, faces, topo, kind, target):
    """Move vertices along their normals until each piece, borders capped, holds its earlier volume."""
    piece = topology.face_components(faces, positions.shape[0])
    vertex_piece = torch.zeros(positions.shape[0], dtype=torch.int64, device=positions.device)
    vertex_piece[faces.reshape(-1)] = piece.repeat_interleave(3)
    movable = kind != HELD
    current = positions
    for _ in range(3):
        volume, area = _volumes(current, faces, piece, topo)
        offset = (target - volume) / area.clamp_min(1e-30)
        normal = topology.vertex_normals(current, faces)
        current = current + torch.where(movable[:, None], offset[vertex_piece][:, None] * normal,
                                        torch.zeros_like(current))
    return current


def _run(positions, faces, method: str, iterations: int, strength: float, keep_borders: bool,
         keep_features: bool, keep_volume: bool, feature_angle: float, device):
    """Smoothing of welded positions on one device."""
    positions = positions.to(device).double()
    faces = faces.to(device).long()
    topo = topology.edge_topology(faces, positions.shape[0])
    kind, line = _classes(positions, faces, topo, keep_borders, keep_features, feature_angle)
    piece = topology.face_components(faces, positions.shape[0])
    volume_before, _ = _volumes(positions, faces, piece, topo)
    rows, cols = _umbrella_links(topo, kind, line)
    current = positions
    strength = float(strength)
    if method == "laplacian":
        for _ in range(iterations):
            current = current + strength * _umbrella(current, rows, cols)
    elif method == "taubin":
        grow = 1.0 / (TAUBIN_PASS_BAND - 1.0 / max(strength, 1e-6))
        for _ in range(iterations):
            current = current + strength * _umbrella(current, rows, cols)
            current = current + grow * _umbrella(current, rows, cols)
    elif method == "hc":
        movable = (kind != HELD)[:, None]
        for _ in range(iterations):
            previous = current
            current = previous + strength * _umbrella(previous, rows, cols)
            push = current - (HC_ALPHA * positions + (1.0 - HC_ALPHA) * previous)
            push = torch.where(movable, push, torch.zeros_like(push))
            neighbour = _umbrella(push, rows, cols) + push
            current = current - (HC_BETA * push + (1.0 - HC_BETA) * neighbour)
    elif method == "implicit":
        held = torch.where(kind == LINE, torch.full_like(kind, HELD), kind)
        current = _implicit(current, faces, held, strength, iterations)
    elif method == "bilateral":
        held = torch.where(kind == LINE, torch.full_like(kind, HELD), kind)
        current = _bilateral(current, faces, topo, held, strength, iterations)
    else:
        raise ValueError(f"the smoothing method must be one of {', '.join(METHODS)}, not {method!r}")
    if keep_volume:
        current = _restore_volume(current, faces, topo, kind, volume_before)
    volume_after, _ = _volumes(current, faces, piece, topo)
    moved = (current - positions).norm(dim=1)
    stats = {
        "held_vertices": int((kind == HELD).sum()),
        "line_vertices": int((kind == LINE).sum()),
        "mean_move": float(moved.mean()) if moved.numel() else 0.0,
        "max_move": float(moved.max()) if moved.numel() else 0.0,
        "volume_before": float(volume_before.sum()),
        "volume_after": float(volume_after.sum()),
    }
    return current.cpu(), stats


def smooth(mesh: TriMesh, method: str = "taubin", iterations: int = 10, strength: float = 0.5,
           keep_borders: bool = True, keep_features: bool = True, keep_volume: bool = False,
           feature_angle: float = 45.0, device: torch.device | str | None = None) -> tuple[TriMesh, dict]:
    """Smooth a surface, keeping its seams closed.

    Args:
        mesh: The split mesh.
        method: One of `METHODS`.
        iterations: Smoothing steps, 0 to `MAX_ITERATIONS`.
        strength: Step size, 0 to 1.
        keep_borders: Hold border vertices still.
        keep_features: Hold creases sharper than ``feature_angle``; the explicit methods slide
            vertices along them, implicit and bilateral hold them still.
        keep_volume: Restore the volume of each closed piece afterwards.
        feature_angle: Dihedral angle in degrees from which an edge is a crease.
        device: Device to run on; None means the mesh compute device.

    Returns:
        ``(mesh, stats)``: the mesh with new positions and normals, tangents turned to match,
        other attributes untouched, and the run's figures.

    Raises:
        ValueError: For an unknown method or a mesh with no triangles.
    """
    if method not in METHODS:
        raise ValueError(f"the smoothing method must be one of {', '.join(METHODS)}, not {method!r}")
    if mesh.face_count == 0:
        raise ValueError("the mesh has no triangles to smooth")
    started = time.perf_counter()
    iterations = max(0, min(int(iterations), MAX_ITERATIONS))
    split = mesh.vertices.double()
    diagonal = float((split.amax(0) - split.amin(0)).norm()) if split.shape[0] else 0.0
    welded, faces, remap = topology.weld(split, mesh.faces.long(), tolerance=1e-7 * diagonal)
    sound = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
    positions, stats = mesh_device.with_cpu_fallback(
        lambda on: _run(welded, faces[sound], method, iterations, float(strength), bool(keep_borders),
                        bool(keep_features), bool(keep_volume), float(feature_angle), on),
        device, label="Mesh Smooth")
    vertices = positions[remap].float()
    attributes = dict(mesh.attributes)
    if "normals" in attributes:
        attributes["normals"] = recompute_normals(vertices, mesh.faces, remap, attributes["normals"])
        if "tangents" in attributes:
            attributes["tangents"] = orthogonal_tangents(attributes["tangents"], attributes["normals"])
    stats["seconds"] = round(time.perf_counter() - started, 3)
    return TriMesh(vertices, mesh.faces.clone(), attributes, mesh.polygons), stats


def smooth_part(part: Part, **options) -> tuple[Part, dict]:
    """Smooth one model part.

    Args:
        part: The part to smooth.
        **options: Keyword arguments of `smooth`.

    Returns:
        ``(part, stats)``: a new part with moved vertices, recomputed normals and turned
        tangents, and the run's figures.

    Raises:
        ValueError: As `smooth` does.
    """
    attributes = {}
    if part.normals is not None:
        attributes["normals"] = part.normals
    if part.tangents is not None and part.normals is not None:
        attributes["tangents"] = part.tangents
    result, stats = smooth(TriMesh(part.vertices, part.faces, attributes), **options)
    new_part = replace(part, vertices=result.vertices, normals=result.attributes.get("normals"),
                       tangents=result.attributes.get("tangents", part.tangents))
    return new_part, stats

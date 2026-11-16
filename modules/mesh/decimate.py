"""Mesh decimation by parallel quadric edge collapse, and grid clustering for extreme cuts.

Several meshes are reduced together on welded positions, so parts that meet stay joined.
Positions are normalised by ``size``, and every error is a fraction of it.
"""

from __future__ import annotations

__all__ = [
    "BORDER_MODES",
    "MAX_CARRIED_VALUES",
    "MAX_TRIANGLES",
    "PLACEMENTS",
    "SEAM_MODES",
    "SYMMETRY",
    "DecimateStats",
    "check_size",
    "cluster",
    "decimate",
]

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

import torch

from ..log import get_logger
from . import topology
from .arrays import MORPH_PREFIX, TriMesh
from .device import chunk_size, compute_device, with_cpu_fallback

logger = get_logger("mesh.decimate")

#: Where a collapsed edge's vertex goes: the quadric optimum, or one of its two ends.
PLACEMENTS = ("optimal", "keep vertices")

#: How open edges are treated: simplified along themselves, or never moved.
BORDER_MODES = ("slide", "lock")

#: How attribute seams are treated: simplified along themselves, never moved, or crossed.
SEAM_MODES = ("slide", "lock", "cross")

#: Mirror symmetry kept through the reduction.
SYMMETRY = ("none", "x")

#: Triangles one call accepts.
MAX_TRIANGLES = 20_000_000

#: Per-vertex values of carried attributes one call accepts, such as skin weights and shape keys.
MAX_CARRIED_VALUES = 400_000_000

#: Collapse rounds before stopping.
MAX_ROUNDS = 4000

#: Smallest cosine allowed between a face's normal before and after a collapse.
FLIP_COSINE = 0.25

#: Weight of the planes holding a border or seam, times the squared edge length.
EDGE_WEIGHT = 4.0

#: Weight of the pull of a merged vertex towards the original vertices it replaces.
SPREAD_WEIGHT = 0.3

#: Weight of UV error, as a distance on the surface.
UV_WEIGHT = 1.0

#: Weight of normal error, in model sizes per unit of normal change.
NORMAL_WEIGHT = 0.02

#: Weight of vertex colour error, in model sizes per unit of colour.
COLOR_WEIGHT = 0.05

#: Attributes the quadrics hold, with their widths.
QUADRIC_ATTRIBUTES = (("uvs", 2), ("normals", 3), ("colors", 4))

#: Largest difference under which two split vertices at one position are the same vertex.
SAME_TOLERANCE = {"uvs": 1e-5, "normals": 0.02, "colors": 1.0 / 510.0, "skin": 1e-3, "tangents": None}

#: Same, for shape key offsets, as a fraction of the size.
SAME_MORPH = 1e-5

#: Split vertices compared with each other per welded position.
SAME_PAIRS = 24

#: Share of the cheapest collapses competing first each round.
CANDIDATE_SHARE = 0.25

#: Fewest collapses competing first each round.
CANDIDATE_FLOOR = 512

#: Share of the round's candidates below which the rest of them compete too.
WIDEN_BELOW = 0.01

#: Most claim passes per stage of a round, each adding collapses clear of those already chosen.
CLAIM_PASSES = 32

#: A stage ends once a pass wins fewer claims than this share of its first pass.
CLAIM_FALLOFF = 0.05

#: Mirror partners match within this fraction of the size.
MIRROR_TOLERANCE = 1e-5

#: Grid refinements tried when clustering towards a triangle count.
CLUSTER_STEPS = 24

#: Largest grid side when clustering.
MAX_GRID = 1 << 20

#: Kinds of welded vertex.
MANIFOLD, BORDER, SEAM, LOCKED = 0, 1, 2, 3

#: Collapse kinds of an edge.
NONE, BOTH, TO_Y, TO_X, LINE = 0, 1, 2, 3, 4


@dataclass
class DecimateStats:
    """What a reduction did.

    Attributes:
        triangles_before: Triangles given.
        triangles_after: Triangles returned.
        vertices_before: Vertices given, counting seam splits.
        vertices_after: Vertices returned.
        error: Largest collapse error, as a fraction of the size.
        rounds: Collapse rounds, or grid refinements when clustering.
        collapses: Edges collapsed.
        locked: Welded vertices that could not move at the start.
        stop: ``"target"``, ``"error"``, ``"locked"``, ``"blocked"``, ``"rounds"`` or ``"grid"``.
        mirrored: Share of vertices with a mirror partner, or -1 without symmetry.
        seconds: Wall time.
        device: Device the work ran on.
    """

    triangles_before: int = 0
    triangles_after: int = 0
    vertices_before: int = 0
    vertices_after: int = 0
    error: float = 0.0
    rounds: int = 0
    collapses: int = 0
    locked: int = 0
    stop: str = "target"
    mirrored: float = -1.0
    seconds: float = 0.0
    device: str = "cpu"


# ---------------------------------------------------------------------------------------------
# Quadric arithmetic. A position quadric is (A00, A01, A02, A11, A12, A22, b0, b1, b2, c, W);
# an attribute quadric adds per channel the summed gradient (3) and offset (1).

#: The identity matrix as six symmetric entries.
_IDENTITY6 = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 1.0], dtype=torch.float64)


def _apply_sym(q6: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """``A p`` for symmetric matrices stored as six entries."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    return torch.stack([
        q6[:, 0] * x + q6[:, 1] * y + q6[:, 2] * z,
        q6[:, 1] * x + q6[:, 3] * y + q6[:, 4] * z,
        q6[:, 2] * x + q6[:, 4] * y + q6[:, 5] * z,
    ], dim=1)


def _quadratic(q6: torch.Tensor, b: torch.Tensor, c: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """``p^T A p + 2 b.p + c`` per row."""
    return (p * (_apply_sym(q6, p) + 2.0 * b)).sum(1) + c


def _outer6(vectors: torch.Tensor) -> torch.Tensor:
    """Six-entry outer products ``v v^T`` summed over the second-to-last dimension."""
    x, y, z = vectors[..., 0], vectors[..., 1], vectors[..., 2]
    parts = torch.stack([x * x, x * y, x * z, y * y, y * z, z * z], dim=-1)
    return parts.sum(-2) if vectors.dim() > 2 else parts


def _attribute_energy(qa: torch.Tensor, p: torch.Tensor, values: torch.Tensor, channels: int) -> torch.Tensor:
    """Attribute quadric error at a position with given weighted attribute values."""
    total = _quadratic(qa[:, 0:6], qa[:, 6:9], qa[:, 9], p)
    if channels == 0:
        return total
    gradients = qa[:, 11:11 + 3 * channels].reshape(-1, channels, 3)
    offsets = qa[:, 11 + 3 * channels:11 + 4 * channels]
    predicted = (gradients * p[:, None, :]).sum(-1) + offsets
    return total - 2.0 * (values * predicted).sum(1) + qa[:, 10] * (values * values).sum(1)


def _attribute_optimum(qa: torch.Tensor, p: torch.Tensor, channels: int) -> torch.Tensor:
    """Weighted attribute values that minimise an attribute quadric at a position."""
    gradients = qa[:, 11:11 + 3 * channels].reshape(-1, channels, 3)
    offsets = qa[:, 11 + 3 * channels:11 + 4 * channels]
    weight = qa[:, 10:11]
    predicted = (gradients * p[:, None, :]).sum(-1) + offsets
    return torch.where(weight > 0, predicted / weight.clamp_min(1e-300), torch.zeros_like(predicted))


def _schur(qp: torch.Tensor, qa: torch.Tensor, channels: int):
    """Position quadric left once the attributes take their best values: ``(A6, b, c)``."""
    a6 = qp[:, 0:6] + qa[:, 0:6]
    b = qp[:, 6:9] + qa[:, 6:9]
    c = qp[:, 9] + qa[:, 9]
    if channels:
        weight = qa[:, 10]
        inverse = torch.where(weight > 0, 1.0 / weight.clamp_min(1e-300), torch.zeros_like(weight))
        gradients = qa[:, 11:11 + 3 * channels].reshape(-1, channels, 3)
        offsets = qa[:, 11 + 3 * channels:11 + 4 * channels]
        a6 = a6 - _outer6(gradients) * inverse[:, None]
        b = b - (gradients * offsets[..., None]).sum(1) * inverse[:, None]
        c = c - (offsets * offsets).sum(1) * inverse
    return a6, b, c


def _solve_sym(a6: torch.Tensor, rhs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve symmetric 3x3 systems by cofactors; returns ``(solution, determinant)``."""
    a, b, c, d, e, f = a6.unbind(1)
    co_a = d * f - e * e
    co_b = c * e - b * f
    co_c = b * e - c * d
    co_d = a * f - c * c
    co_e = b * c - a * e
    co_f = a * d - b * b
    det = a * co_a + b * co_b + c * co_c
    safe = torch.where(det.abs() > 0, det, torch.ones_like(det))
    x, y, z = rhs.unbind(1)
    solution = torch.stack([
        co_a * x + co_b * y + co_c * z,
        co_b * x + co_d * y + co_e * z,
        co_c * x + co_e * y + co_f * z,
    ], dim=1) / safe[:, None]
    return solution, det


def _held_values(state: _State) -> tuple[torch.Tensor, torch.Tensor]:
    """Position quadrics of every wedge held at its own attribute values, per wedge and per vertex.

    Returns:
        ``(wedge (W, 10), vertex (V, 10))`` as ``(A6, b, c)`` rows; unused wedges count nowhere.
    """
    quadrics = state.wedge_quadrics
    channels = state.channels
    rows = quadrics[:, 0:10].clone()
    if channels:
        values = state.values
        gradients = quadrics[:, 11:11 + 3 * channels].reshape(-1, channels, 3)
        offsets = quadrics[:, 11 + 3 * channels:11 + 4 * channels]
        rows[:, 6:9] -= (gradients * values[..., None]).sum(1)
        rows[:, 9] += quadrics[:, 10] * (values * values).sum(1) - 2.0 * (values * offsets).sum(1)
    used = torch.zeros(rows.shape[0], dtype=torch.bool, device=rows.device)
    used[state.faces.reshape(-1)] = True
    vertex = torch.zeros(state.positions.shape[0], 10, dtype=rows.dtype, device=rows.device)
    vertex.index_add_(0, state.wedge_vertex[used], rows[used])
    return rows, vertex


# ---------------------------------------------------------------------------------------------
# Setup


class _State:
    """Working tensors of one reduction."""

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.size = 1.0
        self.centre = torch.zeros(3, dtype=torch.float64)
        self.positions = torch.zeros(0, 3, dtype=torch.float64)
        self.vertex_quadrics = torch.zeros(0, 11, dtype=torch.float64)
        self.spread = torch.zeros(0, 10, dtype=torch.float64)
        self.wedge_quadrics = torch.zeros(0, 11, dtype=torch.float64)
        self.values = torch.zeros(0, 0, dtype=torch.float64)
        self.value_weights = torch.zeros(0, 0, dtype=torch.float64)
        self.carried = torch.zeros(0, 0, dtype=torch.float32)
        self.bound_columns = torch.zeros(0, dtype=torch.int64)
        self.wedge_vertex = torch.zeros(0, dtype=torch.int64)
        self.wedge_part = torch.zeros(0, dtype=torch.int64)
        self.hard_wedge = torch.zeros(0, dtype=torch.int64)
        self.hard_vertex = torch.zeros(0, dtype=torch.int64)
        self.faces = torch.zeros(0, 3, dtype=torch.int64)
        self.face_part = torch.zeros(0, dtype=torch.int64)
        self.locked = torch.zeros(0, dtype=torch.bool)
        self.mirror: torch.Tensor | None = None
        self.keep_part = torch.zeros(0, dtype=torch.bool)
        self.slide_borders = True
        self.slide_seams = True
        self.cross_seams = False
        self.channels = 0
        self.quadric_layout: list[tuple[str, int, int]] = []
        self.carried_layout: list[tuple[str, int, int]] = []
        self.mesh_names: list[set[str]] = []
        self.mesh_count = 0


def _position_bound(name: str) -> bool:
    """Whether an attribute belongs to the position rather than to a corner: skin and shape keys."""
    return name == "skin" or name.startswith(MORPH_PREFIX) or name.startswith("morph_normals:")


def _union_names(meshes: list[TriMesh]) -> dict[str, int]:
    """Every attribute name across the meshes with its width."""
    widths: dict[str, int] = {}
    for mesh in meshes:
        for name, value in mesh.attributes.items():
            width = 1 if value.dim() == 1 else int(value.shape[1])
            if widths.setdefault(name, width) != width:
                raise ValueError(f"the parts disagree on the width of their {name}; decimate them separately")
    return widths


def _stacked(meshes: list[TriMesh], name: str, width: int, device: torch.device,
             dtype: torch.dtype) -> torch.Tensor:
    """One attribute across every mesh, zero where a mesh lacks it."""
    rows = []
    for mesh in meshes:
        value = mesh.attributes.get(name)
        if value is None:
            rows.append(torch.zeros(mesh.vertex_count, width, dtype=dtype, device=device))
        else:
            value = value.to(device=device, dtype=dtype)
            rows.append(value.unsqueeze(1) if value.dim() == 1 else value)
    return torch.cat(rows, 0) if rows else torch.zeros(0, width, dtype=dtype, device=device)


def _same_vertices(remap: torch.Tensor, part: torch.Tensor, attributes: dict[str, torch.Tensor],
                   size: float) -> torch.Tensor:
    """Merge split vertices at one position whose attributes agree within tolerance.

    Returns:
        ``(N,)`` int64 wedge index of each split vertex, wedges numbered from 0.
    """
    count = int(remap.shape[0])
    device = remap.device
    if count == 0:
        return remap.clone()
    stride = int(part.max()) + 1
    order = torch.arange(count, device=device)
    uvs = attributes.get("uvs")
    if uvs is not None:
        for column in reversed(range(uvs.shape[1])):
            order = order[torch.argsort(uvs[order, column], stable=True)]
    order = order[torch.argsort(remap[order] * stride + part[order], stable=True)]
    group = remap[order] * stride + part[order]
    start = torch.ones(count, dtype=torch.bool, device=device)
    start[1:] = group[1:] != group[:-1]
    first = torch.cummax(torch.where(start, torch.arange(count, device=device), torch.zeros_like(order)), 0).values
    _, group_size = torch.unique_consecutive(group, return_counts=True)
    size_of = torch.repeat_interleave(group_size, group_size)
    rank = torch.arange(count, device=device) - first
    partners = (size_of - 1 - rank).clamp(0, SAME_PAIRS)
    total = int(partners.sum())
    if total == 0:
        return torch.arange(count, device=device)
    source = torch.repeat_interleave(torch.arange(count, device=device), partners)
    begin = torch.cumsum(partners, 0) - partners
    step = torch.arange(total, device=device) - torch.repeat_interleave(begin, partners) + 1
    first_vertex, second_vertex = order[source], order[source + step]
    equal = torch.ones(total, dtype=torch.bool, device=device)
    for name, value in attributes.items():
        if name.startswith(MORPH_PREFIX) or name.startswith("morph_normals:"):
            tolerance = SAME_MORPH * size if name.startswith(MORPH_PREFIX) else SAME_TOLERANCE["normals"]
        else:
            tolerance = SAME_TOLERANCE.get(name, 1e-6)
        if tolerance is None:
            continue
        difference = (value[first_vertex] - value[second_vertex]).abs().amax(1)
        equal &= difference <= tolerance
    labels = topology.union_find(count, first_vertex[equal], second_vertex[equal])
    _, wedge = torch.unique(labels, return_inverse=True)
    return wedge


def _finish(name: str, values: torch.Tensor) -> torch.Tensor:
    """Attribute rows made valid: unit normals, signed unit tangents, summed skin, clamped colour."""
    if name == "normals":
        return torch.nn.functional.normalize(values, dim=1)
    if name == "tangents" and values.shape[1] == 4:
        direction = torch.nn.functional.normalize(values[:, :3], dim=1)
        return torch.cat([direction, torch.where(values[:, 3:] < 0, -1.0, 1.0).to(values.dtype)], dim=1)
    if name == "skin":
        values = values.clamp_min(0)
        total = values.sum(1, keepdim=True)
        return torch.where(total > 0, values / total.clamp_min(1e-12), values)
    if name == "colors":
        return values.clamp(0, 1)
    return values


def _setup(meshes: list[TriMesh], size: float, centre: torch.Tensor, keep: list[bool],
           borders: str, seams: str, locked_points: torch.Tensor | None,
           device: torch.device, quadric_attributes: bool = True) -> _State:
    """Weld the meshes together and build their wedges, faces and quadrics."""
    state = _State()
    state.device = device
    state.size = float(size)
    state.centre = centre.to(device=device, dtype=torch.float64)
    state.mesh_count = len(meshes)
    state.mesh_names = [set(mesh.attributes) for mesh in meshes]
    state.slide_borders = borders == "slide"
    state.slide_seams = seams != "lock"
    state.cross_seams = seams == "cross"
    state.keep_part = torch.tensor(keep, dtype=torch.bool, device=device)
    widths = _union_names(meshes)
    quadric_names = [(name, width) for name, width in QUADRIC_ATTRIBUTES
                     if quadric_attributes and widths.get(name) == width]
    carried_names = [(name, width) for name, width in widths.items() if (name, width) not in quadric_names]
    attributes = {name: _stacked(meshes, name, width, device,
                                 torch.float64 if (name, width) in quadric_names else torch.float32)
                  for name, width in widths.items()}

    vertices = torch.cat([mesh.vertices.to(device=device, dtype=torch.float64) for mesh in meshes], 0)
    offsets = [0]
    for mesh in meshes:
        offsets.append(offsets[-1] + mesh.vertex_count)
    faces = torch.cat([mesh.faces.to(device=device, dtype=torch.int64) + offsets[index]
                       for index, mesh in enumerate(meshes)], 0)
    face_part = torch.cat([torch.full((mesh.face_count,), index, dtype=torch.int64, device=device)
                           for index, mesh in enumerate(meshes)], 0)
    vertex_part = torch.cat([torch.full((mesh.vertex_count,), index, dtype=torch.int64, device=device)
                             for index, mesh in enumerate(meshes)], 0)

    welded, _, remap = topology.weld(vertices, faces, tolerance=0)
    wedge_of = _same_vertices(remap, vertex_part, attributes, state.size)
    wedge_count = int(wedge_of.max()) + 1 if wedge_of.numel() else 0
    members = torch.bincount(wedge_of, minlength=wedge_count).clamp_min(1)[:, None]
    leader = torch.full((wedge_count,), vertices.shape[0], dtype=torch.int64, device=device)
    leader.scatter_reduce_(0, wedge_of, torch.arange(vertices.shape[0], device=device), reduce="amin")
    state.wedge_vertex = remap[leader]
    state.wedge_part = vertex_part[leader]
    if state.cross_seams:
        bound = {name: value for name, value in attributes.items() if _position_bound(name)}
        hard = _same_vertices(remap, vertex_part, bound, state.size)
        state.hard_wedge = hard[leader]
        state.hard_vertex = torch.zeros(int(hard.max()) + 1, dtype=torch.int64, device=device)
        state.hard_vertex[hard] = remap

    def averaged(value: torch.Tensor) -> torch.Tensor:
        total = torch.zeros(wedge_count, value.shape[1], dtype=value.dtype, device=device)
        total.index_add_(0, wedge_of, value)
        return total / members.to(value.dtype)

    faces_wedged = wedge_of[faces]
    welded_faces = state.wedge_vertex[faces_wedged]
    good = ((welded_faces[:, 0] != welded_faces[:, 1]) & (welded_faces[:, 1] != welded_faces[:, 2])
            & (welded_faces[:, 2] != welded_faces[:, 0]))
    state.faces = faces_wedged[good]
    state.face_part = face_part[good]
    state.positions = (welded - state.centre) / state.size

    start = 0
    for name, width in quadric_names:
        state.quadric_layout.append((name, start, width))
        start += width
    state.channels = start
    start = 0
    bound_columns = []
    for name, width in carried_names:
        state.carried_layout.append((name, start, width))
        if _position_bound(name):
            bound_columns += list(range(start, start + width))
        start += width
    state.bound_columns = torch.tensor(bound_columns, dtype=torch.int64, device=device)
    raw = (torch.cat([averaged(attributes[name]) for name, _ in quadric_names], 1) if quadric_names
           else torch.zeros(wedge_count, 0, dtype=torch.float64, device=device))
    for name, _ in quadric_names:
        del attributes[name]
    state.carried = (torch.cat([averaged(attributes.pop(name)) for name, _ in carried_names], 1)
                     if carried_names else torch.zeros(wedge_count, 0, dtype=torch.float32, device=device))
    weights = _channel_weights(state, meshes)
    state.value_weights = weights[state.wedge_part]
    state.values = raw * state.value_weights
    _renormalise_normals(state, torch.arange(wedge_count, device=device))

    vertex_count = int(state.positions.shape[0])
    welded_faces = state.wedge_vertex[state.faces]
    topo = topology.edge_topology(welded_faces, vertex_count)
    state.locked = topology.non_manifold_vertices(welded_faces, vertex_count, topo)
    if locked_points is not None and locked_points.numel():
        points = locked_points.to(device=device, dtype=torch.float64).reshape(-1, 3)
        joined = torch.cat([welded, points], 0)
        _, inverse = torch.unique(joined, dim=0, return_inverse=True)
        hit = torch.zeros(int(inverse.max()) + 1, dtype=torch.bool, device=device)
        hit[inverse[welded.shape[0]:]] = True
        state.locked |= hit[inverse[:welded.shape[0]]]
    _build_quadrics(state, topo)
    return state


def _channel_weights(state: _State, meshes: list[TriMesh]) -> torch.Tensor:
    """``(meshes, channels)`` weights turning each mesh's attributes into model-size units."""
    weights = torch.zeros(len(meshes), state.channels, dtype=torch.float64, device=state.device)
    for index, mesh in enumerate(meshes):
        for name, start, width in state.quadric_layout:
            if name not in mesh.attributes:
                continue
            if name == "uvs":
                weights[index, start:start + width] = UV_WEIGHT * _uv_scale(mesh, state.size)
            elif name == "normals":
                weights[index, start:start + width] = NORMAL_WEIGHT
            else:
                weights[index, start:start + width] = COLOR_WEIGHT
    return weights


def _uv_scale(mesh: TriMesh, size: float) -> float:
    """Surface length per unit of UV, in model sizes."""
    if mesh.face_count == 0:
        return 1.0
    vertices = mesh.vertices.double() / size
    uvs = mesh.attributes["uvs"].double()
    surface = topology.face_areas(vertices, mesh.faces).sum()
    flat = torch.cat([uvs, torch.zeros(uvs.shape[0], 1, dtype=uvs.dtype, device=uvs.device)], 1)
    texture = topology.face_areas(flat, mesh.faces).sum()
    if not bool(texture > 0) or not bool(surface > 0):
        return 1.0
    return float(torch.sqrt(surface / texture))


def _renormalise_normals(state: _State, wedges: torch.Tensor) -> None:
    """Keep the normal channel of the given wedges at unit length."""
    for name, start, width in state.quadric_layout:
        if name != "normals" or wedges.numel() == 0:
            continue
        weight = state.value_weights[wedges, start:start + width]
        raw = torch.where(weight > 0, state.values[wedges, start:start + width] / weight.clamp_min(1e-300),
                          torch.zeros_like(weight))
        state.values[wedges, start:start + width] = torch.nn.functional.normalize(raw, dim=1) * weight


def _build_quadrics(state: _State, topo: topology.EdgeTopology) -> None:
    """Fill the vertex and wedge quadrics from the faces, borders and seams."""
    device = state.device
    vertex_count = int(state.positions.shape[0])
    wedge_count = int(state.wedge_vertex.shape[0])
    channels = state.channels
    state.vertex_quadrics = torch.zeros(vertex_count, 11, dtype=torch.float64, device=device)
    state.spread = torch.zeros(vertex_count, 10, dtype=torch.float64, device=device)
    state.wedge_quadrics = torch.zeros(wedge_count, 11 + 4 * channels, dtype=torch.float64, device=device)
    face_total = int(state.faces.shape[0])
    chunk = 1 << 20
    for begin in range(0, face_total, chunk):
        wedges = state.faces[begin:begin + chunk]
        corners = state.wedge_vertex[wedges]
        a, b, c = (state.positions[corners[:, k]] for k in range(3))
        first, second = b - a, c - a
        normal = torch.cross(first, second, dim=1)
        double_area = normal.norm(dim=1)
        area = 0.5 * double_area
        unit = torch.where(double_area[:, None] > 0, normal / double_area.clamp_min(1e-300)[:, None],
                           torch.zeros_like(normal))
        offset = -(unit * a).sum(1)
        plane = torch.cat([_outer6(unit) * area[:, None], unit * (offset * area)[:, None],
                           (offset * offset * area)[:, None], area[:, None]], 1)
        # A weak pull along the face towards each original corner keeps merged vertices evenly spaced.
        spread = SPREAD_WEIGHT * area
        tangent = (_IDENTITY6.to(device) - _outer6(unit)) * spread[:, None]
        for k, corner in enumerate((a, b, c)):
            pull = torch.cat([tangent, -_apply_sym(tangent, corner),
                              (corner * _apply_sym(tangent, corner)).sum(1, keepdim=True)], 1)
            state.spread.index_add_(0, corners[:, k], pull)
            row = plane.clone()
            row[:, 0:10] += pull
            state.vertex_quadrics.index_add_(0, corners[:, k], row)
        rows = torch.zeros(wedges.shape[0], 11 + 4 * channels, dtype=torch.float64, device=device)
        rows[:, 10] = area
        if channels:
            values = state.values[wedges]
            g11 = (first * first).sum(1)
            g12 = (first * second).sum(1)
            g22 = (second * second).sum(1)
            det = g11 * g22 - g12 * g12
            usable = det > 1e-30 * (g11 * g22).clamp_min(1e-300)
            inverse = torch.where(usable, 1.0 / det.clamp_min(1e-300), torch.zeros_like(det))
            delta1 = values[:, 1] - values[:, 0]
            delta2 = values[:, 2] - values[:, 0]
            alpha = (delta1 * g22[:, None] - delta2 * g12[:, None]) * inverse[:, None]
            beta = (delta2 * g11[:, None] - delta1 * g12[:, None]) * inverse[:, None]
            gradient = alpha[..., None] * first[:, None, :] + beta[..., None] * second[:, None, :]
            constant = values[:, 0] - (gradient * a[:, None, :]).sum(-1)
            constant = torch.where(usable[:, None], constant, values.mean(1))
            rows[:, 0:6] = _outer6(gradient) * area[:, None]
            rows[:, 6:9] = (gradient * constant[..., None]).sum(1) * area[:, None]
            rows[:, 9] = (constant * constant).sum(1) * area
            rows[:, 11:11 + 3 * channels] = (gradient * area[:, None, None]).reshape(-1, 3 * channels)
            rows[:, 11 + 3 * channels:] = constant * area[:, None]
        for k in range(3):
            state.wedge_quadrics.index_add_(0, wedges[:, k], rows)
    _add_edge_planes(state, topo)


def _hard_seams(state: _State, topo: topology.EdgeTopology) -> torch.Tensor:
    """Interior edges where skin, shape keys or the part change across the edge."""
    hard = topology.seam_edges(state.hard_wedge[state.faces], state.hard_vertex, topo)
    return hard & (topo.edge_face_count == 2)


def _add_edge_planes(state: _State, topo: topology.EdgeTopology) -> None:
    """Planes through every border and kept seam edge, upright on its face, weighted by length."""
    welded = state.wedge_vertex[state.faces]
    if state.cross_seams:
        seam = _hard_seams(state, topo)
    else:
        seam = topology.seam_edges(state.faces, state.wedge_vertex, topo) & (topo.edge_face_count == 2)
    marked = (topo.boundary | seam)[topo.face_edges]
    face, side = torch.nonzero(marked, as_tuple=True)
    if face.numel() == 0:
        return
    start = welded[face, side]
    end = welded[face, (side + 1) % 3]
    apex = welded[face, (side + 2) % 3]
    p0, p1, p2 = state.positions[start], state.positions[end], state.positions[apex]
    edge = p1 - p0
    face_normal = torch.cross(edge, p2 - p0, dim=1)
    upright = torch.nn.functional.normalize(torch.cross(edge, face_normal, dim=1), dim=1)
    weight = EDGE_WEIGHT * (edge * edge).sum(1)
    offset = -(upright * p0).sum(1)
    plane = torch.cat([_outer6(upright) * weight[:, None], upright * (offset * weight)[:, None],
                       (offset * offset * weight)[:, None], torch.zeros_like(weight)[:, None]], 1)
    state.vertex_quadrics.index_add_(0, start, plane)
    state.vertex_quadrics.index_add_(0, end, plane)


# ---------------------------------------------------------------------------------------------
# Mirror symmetry


def _mirror_map(state: _State) -> torch.Tensor:
    """Welded vertex mirrored across the model's x = 0 plane, or -1 where none lies close."""
    positions = state.positions
    count = int(positions.shape[0])
    plane = -float(state.centre[0]) / state.size
    mirrored = positions.clone()
    mirrored[:, 0] = 2.0 * plane - mirrored[:, 0]
    joined = torch.cat([positions, mirrored], 0)
    joined_faces = torch.zeros(0, 3, dtype=torch.int64, device=positions.device)
    _, _, groups = topology.weld(joined, joined_faces, tolerance=MIRROR_TOLERANCE)
    group_count = int(groups.max()) + 1 if groups.numel() else 0
    originals = torch.bincount(groups[:count], minlength=group_count)
    partner = torch.full((group_count,), -1, dtype=torch.int64, device=positions.device)
    partner[groups[:count]] = torch.arange(count, device=positions.device)
    image = groups[count:]
    unique = originals[image] == 1
    mirror = torch.where(unique, partner[image], torch.full_like(image, -1))
    # A partner must name this vertex back.
    back = torch.where(mirror >= 0, mirror[mirror.clamp_min(0)], torch.full_like(mirror, -2))
    mirror = torch.where(back == torch.arange(count, device=positions.device), mirror, torch.full_like(mirror, -1))
    return mirror


# ---------------------------------------------------------------------------------------------
# One round of collapses


@dataclass
class _Plan:
    """Candidate collapses of one round, one row each."""

    edge: torch.Tensor
    x: torch.Tensor
    y: torch.Tensor
    mode: torch.Tensor
    wx: torch.Tensor
    wy: torch.Tensor
    pair2: torch.Tensor
    keep: torch.Tensor
    t: torch.Tensor
    to_y: torch.Tensor
    position: torch.Tensor
    optimal: torch.Tensor
    cost: torch.Tensor
    error: torch.Tensor
    removed_faces: torch.Tensor

    def take(self, rows: torch.Tensor) -> _Plan:
        """These collapses restricted to some rows."""
        return _Plan(**{name: getattr(self, name)[rows] for name in self.__dataclass_fields__})


def _kinds(state: _State, topo: topology.EdgeTopology) -> tuple[torch.Tensor, torch.Tensor]:
    """Kind of every welded vertex and the seam flag of every edge."""
    vertex_count = int(state.positions.shape[0])
    edges = topo.edges
    interior = topo.edge_face_count == 2
    seam = topology.seam_edges(state.faces, state.wedge_vertex, topo) & interior
    border_edges = torch.bincount(edges[topo.boundary].reshape(-1), minlength=vertex_count)
    crowded = torch.bincount(edges[topo.non_manifold].reshape(-1), minlength=vertex_count)
    seam_count = torch.bincount(edges[seam].reshape(-1), minlength=vertex_count)
    used = torch.zeros(state.wedge_vertex.shape[0], dtype=torch.bool, device=state.device)
    used[state.faces.reshape(-1)] = True
    wedges = torch.bincount(state.wedge_vertex[used], minlength=vertex_count)
    free = (crowded == 0) & ~state.locked
    kind = torch.full((vertex_count,), LOCKED, dtype=torch.int64, device=state.device)
    kind[free & (border_edges == 0) & (seam_count == 0) & (wedges == 1)] = MANIFOLD
    if state.slide_borders:
        kind[free & (border_edges == 2) & (seam_count == 0) & (wedges == 1)] = BORDER
    if state.slide_seams:
        kind[free & (border_edges == 0) & (seam_count == 2) & (wedges == 2)] = SEAM
    if state.cross_seams:
        hard_faces = state.hard_wedge[state.faces]
        hard_seam = _hard_seams(state, topo)
        hard_count = torch.bincount(edges[hard_seam].reshape(-1), minlength=vertex_count)
        used_hard = torch.zeros(state.hard_vertex.shape[0], dtype=torch.bool, device=state.device)
        used_hard[hard_faces.reshape(-1)] = True
        groups = torch.bincount(state.hard_vertex[used_hard], minlength=vertex_count)
        soft = free & (hard_count == 0) & (groups == 1)
        kind[soft & (border_edges == 0)] = MANIFOLD
        if state.slide_borders:
            kind[soft & (border_edges == 2)] = BORDER
    return kind, seam


def _edge_modes(topo: topology.EdgeTopology, kind: torch.Tensor, seam: torch.Tensor) -> torch.Tensor:
    """Collapse kind allowed on every edge."""
    x, y = topo.edges[:, 0], topo.edges[:, 1]
    interior = topo.edge_face_count == 2
    kx, ky = kind[x], kind[y]
    mx, my = kx == MANIFOLD, ky == MANIFOLD
    lx, ly = kx == LOCKED, ky == LOCKED
    mode = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
    mode[interior & mx & my] = BOTH
    mode[interior & mx & ~my] = TO_Y
    mode[interior & my & ~mx] = TO_X
    along_seam = seam & (((kx == SEAM) & ((ky == SEAM) | ly)) | ((ky == SEAM) & lx))
    along_border = topo.boundary & (((kx == BORDER) & ((ky == BORDER) | ly)) | ((ky == BORDER) & lx))
    mode[along_seam | along_border] = LINE
    # The last triangle of a loose piece is never collapsed away.
    lone = topo.boundary[topo.face_edges].all(1)
    mode[topo.face_edges[lone].reshape(-1)] = NONE
    return mode


def _wedge_pairs(state: _State, topo: topology.EdgeTopology, welded_faces: torch.Tensor,
                 edges: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Wedges of each edge's ends in its faces: ``(wx (C, 2), wy (C, 2), pair2 (C,))``."""
    x = topo.edges[edges, 0]
    found = []
    for slot in range(2):
        face = topo.edge_faces[edges, slot]
        present = face >= 0
        face = face.clamp_min(0)
        side = topo.edge_corners[edges, slot].clamp_min(0)
        after = (side + 1) % 3
        first, second = state.faces[face, side], state.faces[face, after]
        at_x = welded_faces[face, side] == x
        found.append((torch.where(at_x, first, second), torch.where(at_x, second, first), present))
    (wx1, wy1, _), (wx2, wy2, present2) = found
    pair2 = present2 & ((wx2 != wx1) | (wy2 != wy1))
    wx2 = torch.where(pair2, wx2, wx1)
    wy2 = torch.where(pair2, wy2, wy1)
    return torch.stack([wx1, wx2], 1), torch.stack([wy1, wy2], 1), pair2


def _plan(state: _State, topo: topology.EdgeTopology, welded_faces: torch.Tensor,
          mode: torch.Tensor, kind: torch.Tensor) -> _Plan:
    """Placement and cost of every allowed collapse, in chunks bounding the working memory."""
    edge = torch.nonzero(mode != NONE).squeeze(1)
    held = _held_values(state)
    width = 11 + 4 * state.channels
    step = chunk_size(width * 8 * 12, state.device, minimum=1 << 16, maximum=1 << 20)
    pieces = [_plan_rows(state, topo, welded_faces, edge[begin:begin + step], mode, kind, held)
              for begin in range(0, int(edge.shape[0]), step)]
    if len(pieces) == 1:
        return pieces[0]
    return _Plan(**{name: torch.cat([getattr(piece, name) for piece in pieces], 0)
                    for name in _Plan.__dataclass_fields__})


def _plan_rows(state: _State, topo: topology.EdgeTopology, welded_faces: torch.Tensor,
               edge: torch.Tensor, mode: torch.Tensor, kind: torch.Tensor,
               held: tuple[torch.Tensor, torch.Tensor]) -> _Plan:
    """Placement and cost of the collapses of the given edges."""
    channels = state.channels
    mode = mode[edge]
    count = int(edge.shape[0])
    device = edge.device
    x, y = topo.edges[edge, 0], topo.edges[edge, 1]
    wx, wy, pair2 = _wedge_pairs(state, topo, welded_faces, edge)
    first_face = topo.edge_faces[edge, 0]
    second_face = topo.edge_faces[edge, 1]
    keep = state.keep_part[state.face_part[first_face]]
    keep = keep | torch.where(second_face >= 0, state.keep_part[state.face_part[second_face.clamp_min(0)]], keep)

    px, py = state.positions[x], state.positions[y]
    qp = state.vertex_quadrics[x] + state.vertex_quadrics[y]
    # Wedges of either end not merged by the collapse keep their values and move with it.
    held_wedge, held_vertex = held
    extra_x = pair2 & (wx[:, 1] != wx[:, 0])
    extra_y = pair2 & (wy[:, 1] != wy[:, 0])
    unpaired = (held_vertex[x] + held_vertex[y] - held_wedge[wx[:, 0]] - held_wedge[wy[:, 0]]
                - torch.where(extra_x[:, None], held_wedge[wx[:, 1]], torch.zeros_like(held_wedge[wx[:, 1]]))
                - torch.where(extra_y[:, None], held_wedge[wy[:, 1]], torch.zeros_like(held_wedge[wy[:, 1]])))
    qp[:, 0:10] += unpaired
    qa1 = state.wedge_quadrics[wx[:, 0]] + state.wedge_quadrics[wy[:, 0]]
    second_rows = torch.nonzero(pair2).squeeze(1)
    qa2 = state.wedge_quadrics[wx[second_rows, 1]] + state.wedge_quadrics[wy[second_rows, 1]]
    second_at = torch.cumsum(pair2.long(), 0) - 1

    def energy(rows: torch.Tensor, t: float) -> torch.Tensor:
        p = px[rows] + t * (py[rows] - px[rows])
        s1 = state.values[wx[rows, 0]] + t * (state.values[wy[rows, 0]] - state.values[wx[rows, 0]])
        total = (_quadratic(qp[rows, 0:6], qp[rows, 6:9], qp[rows, 9], p)
                 + _attribute_energy(qa1[rows], p, s1, channels))
        inner = torch.nonzero(pair2[rows]).squeeze(1)
        if inner.numel():
            outer = rows[inner]
            s2 = state.values[wx[outer, 1]] + t * (state.values[wy[outer, 1]] - state.values[wx[outer, 1]])
            total[inner] += _attribute_energy(qa2[second_at[outer]], p[inner], s2, channels)
        return total

    def along(rows: torch.Tensor, free: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        e0, e1 = energy(rows, 0.0), energy(rows, 1.0)
        t = torch.where(e0 <= e1, torch.zeros_like(e0), torch.ones_like(e0))
        cost = torch.minimum(e0, e1)
        inner = torch.nonzero(free).squeeze(1)
        if inner.numel():
            eh = energy(rows[inner], 0.5)
            curve = 2.0 * e1[inner] - 4.0 * eh + 2.0 * e0[inner]
            slope = 4.0 * eh - e1[inner] - 3.0 * e0[inner]
            best = torch.where(curve > 0, (-slope / (2.0 * curve).clamp_min(1e-300)).clamp(0, 1), t[inner])
            t[inner] = best
            cost[inner] = curve * best * best + slope * best + e0[inner]
        return t, cost

    t = torch.zeros(count, dtype=torch.float64, device=device)
    cost = torch.zeros(count, dtype=torch.float64, device=device)
    optimal = (mode == BOTH) & ~keep
    lx, ly = kind[x] == LOCKED, kind[y] == LOCKED
    ends = torch.nonzero(~optimal).squeeze(1)
    if ends.numel():
        fixed = mode[ends]
        free = (fixed == LINE) & ~keep[ends] & ~lx[ends] & ~ly[ends]
        t_end, cost_end = along(ends, free)
        forced_one = (fixed == TO_Y) | ((fixed == LINE) & ly[ends])
        forced_zero = (fixed == TO_X) | ((fixed == LINE) & lx[ends])
        if bool(forced_one.any()):
            rows = torch.nonzero(forced_one).squeeze(1)
            t_end[rows] = 1.0
            cost_end[rows] = energy(ends[rows], 1.0)
        if bool(forced_zero.any()):
            rows = torch.nonzero(forced_zero).squeeze(1)
            t_end[rows] = 0.0
            cost_end[rows] = energy(ends[rows], 0.0)
        t[ends] = t_end
        cost[ends] = cost_end
    position = px + t[:, None] * (py - px)
    if bool(optimal.any()):
        rows = torch.nonzero(optimal).squeeze(1)
        a6, b, c = _schur(qp[rows], qa1[rows], channels)
        middle = 0.5 * (px[rows] + py[rows])
        length = (py[rows] - px[rows]).norm(dim=1)
        ridge = 1e-4 * (a6[:, 0] + a6[:, 3] + a6[:, 5]).abs() / 3.0 + 1e-300
        regular = a6.clone()
        regular[:, 0] += ridge
        regular[:, 3] += ridge
        regular[:, 5] += ridge
        solution, det = _solve_sym(regular, -b + ridge[:, None] * middle)
        energy_opt = _quadratic(a6, b, c, solution)
        good = ((det > 0) & torch.isfinite(solution).all(1) & torch.isfinite(energy_opt)
                & ((solution - middle).norm(dim=1) <= 2.0 * length))
        chosen = rows[good]
        position[chosen] = solution[good]
        cost[chosen] = energy_opt[good]
        direction = py[chosen] - px[chosen]
        t[chosen] = (((solution[good] - px[chosen]) * direction).sum(1)
                     / (direction * direction).sum(1).clamp_min(1e-300)).clamp(0, 1)
        fallback = rows[~good]
        if fallback.numel():
            t_back, cost_back = along(fallback, torch.ones(fallback.shape[0], dtype=torch.bool, device=device))
            t[fallback] = t_back
            cost[fallback] = cost_back
            position[fallback] = px[fallback] + t_back[:, None] * (py[fallback] - px[fallback])
        optimal = torch.zeros_like(optimal)
        optimal[chosen] = True
    to_y = t >= 0.5
    to_y = torch.where((mode == TO_Y) | ((mode == LINE) & ly), torch.ones_like(to_y), to_y)
    to_y = torch.where((mode == TO_X) | ((mode == LINE) & lx), torch.zeros_like(to_y), to_y)
    if state.cross_seams:
        # Keep the end whose wedges the other end's can all map onto.
        splits_x = pair2 & (wx[:, 0] == wx[:, 1]) & (wy[:, 0] != wy[:, 1])
        splits_y = pair2 & (wy[:, 0] == wy[:, 1]) & (wx[:, 0] != wx[:, 1])
        turn = (mode == BOTH) & ((to_y & splits_x & ~splits_y) | (~to_y & splits_y & ~splits_x))
        to_y = to_y ^ turn
    weight = qp[:, 10]
    spread = state.spread[x] + state.spread[y]
    pulled = _quadratic(spread[:, 0:6], spread[:, 6:9], spread[:, 9], position)
    error = torch.where(weight > 0, (cost - pulled).clamp_min(0) / weight.clamp_min(1e-300),
                        torch.zeros_like(cost))
    cost = torch.where(weight > 0, cost.clamp_min(0) / weight.clamp_min(1e-300), torch.zeros_like(cost))
    removed = topo.edge_face_count[edge]
    return _Plan(edge, x, y, mode, wx, wy, pair2, keep, t, to_y, position, optimal, cost, error, removed)


def _link_ok(topo: topology.EdgeTopology, plan: _Plan, vertex_count: int,
             neighbour_table: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Whether each collapse keeps the mesh manifold: the ends share only the edge's apexes."""
    offsets, neighbours = neighbour_table
    x, y = plan.x, plan.y
    degree = offsets[x + 1] - offsets[x]
    rows = torch.repeat_interleave(torch.arange(x.shape[0], device=x.device), degree)
    begin = torch.cumsum(degree, 0) - degree
    local = torch.arange(rows.shape[0], device=x.device) - begin[rows]
    other = neighbours[offsets[x][rows] + local]
    target = y[rows]
    stride = int(vertex_count)
    keys = torch.minimum(target, other) * stride + torch.maximum(target, other)
    edge_keys = topo.edges[:, 0] * stride + topo.edges[:, 1]
    found = torch.searchsorted(edge_keys, keys)
    hit = (found < edge_keys.shape[0]) & (edge_keys[found.clamp_max(edge_keys.shape[0] - 1)] == keys)
    hit &= other != target
    common = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
    common.index_add_(0, rows, hit.long())
    return common == topo.edge_face_count[plan.edge]


def _ends(plan: _Plan) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(removed, survivor, survivor moves)`` of each collapse."""
    survivor = torch.where(plan.to_y, plan.y, plan.x)
    removed = torch.where(plan.to_y, plan.x, plan.y)
    moves = plan.optimal | torch.where(plan.to_y, plan.t < 1, plan.t > 0)
    return removed, survivor, moves


def _flips(state: _State, welded_faces: torch.Tensor, plan: _Plan,
           face_table: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """Whether each collapse turns a remaining face past the flip limit."""
    removed, survivor, moves = _ends(plan)
    offsets, face_index = face_table
    candidate = torch.arange(removed.shape[0], device=removed.device)
    moving = torch.cat([removed, survivor[moves]])
    owner = torch.cat([candidate, candidate[moves]])
    degree = offsets[moving + 1] - offsets[moving]
    rows = torch.repeat_interleave(torch.arange(moving.shape[0], device=moving.device), degree)
    begin = torch.cumsum(degree, 0) - degree
    local = torch.arange(rows.shape[0], device=moving.device) - begin[rows]
    face = face_index[offsets[moving][rows] + local]
    owner = owner[rows]
    corners = welded_faces[face]
    at_removed = corners == removed[owner][:, None]
    at_survivor = corners == survivor[owner][:, None]
    keep_rows = ~(at_removed.any(1) & at_survivor.any(1))
    corners, owner = corners[keep_rows], owner[keep_rows]
    shifted = (at_removed | at_survivor)[keep_rows]
    old = state.positions[corners]
    new = torch.where(shifted[..., None], plan.position[owner][:, None, :], old)
    first, second = old[:, 1] - old[:, 0], old[:, 2] - old[:, 0]
    before = torch.cross(first, second, dim=1)
    after = torch.cross(new[:, 1] - new[:, 0], new[:, 2] - new[:, 0], dim=1)
    before_sq = (before * before).sum(1)
    after_sq = (after * after).sum(1)
    flat = before_sq <= 1e-12 * (first * first).sum(1) * (second * second).sum(1)
    turned = ((before * after).sum(1) <= FLIP_COSINE * torch.sqrt(before_sq * after_sq)) & ~flat
    bad = torch.zeros(removed.shape[0], dtype=torch.bool, device=removed.device)
    bad[owner[turned]] = True
    return bad


def _wedges_ok(plan: _Plan) -> torch.Tensor:
    """Whether the removed end's wedges map onto the survivor's without splitting or merging two."""
    _, _, moves = _ends(plan)
    gone = torch.where(plan.to_y[:, None], plan.wx, plan.wy)
    kept = torch.where(plan.to_y[:, None], plan.wy, plan.wx)
    split = plan.pair2 & (gone[:, 0] == gone[:, 1]) & (kept[:, 0] != kept[:, 1])
    fold = plan.pair2 & (gone[:, 0] != gone[:, 1]) & (kept[:, 0] == kept[:, 1]) & moves
    return ~(split | fold)


def _priority(cost: torch.Tensor) -> torch.Tensor:
    """Distinct int64 keys ordered by cost, ties broken by row."""
    bits = cost.clamp_min(0).float().view(torch.int32).long()
    return (bits << 31) | torch.arange(cost.shape[0], device=cost.device)


def _claim(welded_faces: torch.Tensor, x: torch.Tensor, y: torch.Tensor, priority: torch.Tensor,
           group: torch.Tensor | None, groups: int, vertex_count: int) -> torch.Tensor:
    """Collapses whose group holds every face around its ends, so no two winners touch one face."""
    # Rows sharing a group, such as a collapse and its mirror image, win together.
    device = priority.device
    big = torch.iinfo(torch.int64).max
    best = torch.full((vertex_count,), big, dtype=torch.int64, device=device)
    best.scatter_reduce_(0, x, priority, reduce="amin")
    best.scatter_reduce_(0, y, priority, reduce="amin")
    face_best = torch.minimum(torch.minimum(best[welded_faces[:, 0]], best[welded_faces[:, 1]]),
                              best[welded_faces[:, 2]])
    held = torch.full((vertex_count,), big, dtype=torch.int64, device=device)
    for corner in range(3):
        held.scatter_reduce_(0, welded_faces[:, corner], face_best, reduce="amin")
    wins = (held[x] == priority) & (held[y] == priority)
    if group is None:
        return wins
    together = torch.ones(groups, dtype=torch.int64, device=device)
    together.scatter_reduce_(0, group, wins.long(), reduce="amin")
    return together[group].bool()


def _apply(state: _State, plan: _Plan, kind: torch.Tensor) -> None:
    """Collapse the chosen edges: move survivors, merge quadrics and wedges, drop dead faces."""
    channels = state.channels
    removed, survivor, moves = _ends(plan)
    gone = torch.where(plan.to_y[:, None], plan.wx, plan.wy)
    kept = torch.where(plan.to_y[:, None], plan.wy, plan.wx)
    pair_rows = [torch.ones_like(plan.pair2), plan.pair2]

    updated = []
    for slot in range(2):
        rows = torch.nonzero(pair_rows[slot] & moves).squeeze(1)
        if rows.numel() == 0:
            continue
        lower, upper = plan.wx[rows, slot], plan.wy[rows, slot]
        t = plan.t[rows]
        target = kept[rows, slot]
        values = state.values[lower] + t[:, None] * (state.values[upper] - state.values[lower])
        optimal = plan.optimal[rows]
        if channels and bool(optimal.any()):
            merged = state.wedge_quadrics[lower[optimal]] + state.wedge_quadrics[upper[optimal]]
            best = _attribute_optimum(merged, plan.position[rows[optimal]], channels)
            present = state.value_weights[target[optimal]] > 0
            values[optimal] = torch.where(present, best, values[optimal])
        carried = state.carried[lower] + t.float()[:, None] * (state.carried[upper] - state.carried[lower])
        updated.append((target, values, carried))
    for target, values, carried in updated:
        state.values[target] = values
        state.carried[target] = carried
        _renormalise_normals(state, target)

    for slot in range(2):
        rows = torch.nonzero(pair_rows[slot]).squeeze(1)
        source = gone[rows, slot]
        target = kept[rows, slot]
        distinct = source != target
        state.wedge_quadrics.index_add_(0, target[distinct], state.wedge_quadrics[source[distinct]])
    state.vertex_quadrics.index_add_(0, survivor, state.vertex_quadrics[removed])
    state.spread.index_add_(0, survivor, state.spread[removed])
    state.positions[survivor] = plan.position
    if state.mirror is not None:
        crossing = state.mirror[removed] == survivor
        state.mirror[survivor[crossing]] = survivor[crossing]

    if state.cross_seams:
        # Wedges carried across a seam take the skin and shape keys of the survivor's side.
        vertex_count = int(state.positions.shape[0])
        source = kept[:, 0]
        owner = torch.full((vertex_count,), -1, dtype=torch.int64, device=state.device)
        owner[removed] = source
        relaxed = moves & ((kind[survivor] == MANIFOLD) | (kind[survivor] == BORDER))
        owner[survivor[relaxed]] = source[relaxed]
        wedge_owner = owner[state.wedge_vertex]
        hit = torch.nonzero(wedge_owner >= 0).squeeze(1)
        origin = wedge_owner[hit]
        state.hard_wedge[hit] = state.hard_wedge[origin]
        columns = state.bound_columns
        if columns.numel() and hit.numel():
            state.carried[hit[:, None], columns[None, :]] = state.carried[origin[:, None], columns[None, :]]

    mapping = torch.arange(state.wedge_vertex.shape[0], device=state.device)
    for slot in range(2):
        rows = torch.nonzero(pair_rows[slot]).squeeze(1)
        mapping[gone[rows, slot]] = kept[rows, slot]
    state.faces = mapping[state.faces]
    vertex_map = torch.arange(state.positions.shape[0], device=state.device)
    vertex_map[removed] = survivor
    state.wedge_vertex = vertex_map[state.wedge_vertex]
    if state.cross_seams:
        state.hard_vertex = vertex_map[state.hard_vertex]
    welded = state.wedge_vertex[state.faces]
    alive = (welded[:, 0] != welded[:, 1]) & (welded[:, 1] != welded[:, 2]) & (welded[:, 2] != welded[:, 0])
    state.faces = state.faces[alive]
    state.face_part = state.face_part[alive]


def _adjacent(topo: topology.EdgeTopology, first: torch.Tensor, second: torch.Tensor,
              vertex_count: int) -> torch.Tensor:
    """Whether each pair of welded vertices is one vertex or joined by an edge."""
    stride = int(vertex_count)
    keys = torch.minimum(first, second) * stride + torch.maximum(first, second)
    edge_keys = topo.edges[:, 0] * stride + topo.edges[:, 1]
    found = torch.searchsorted(edge_keys, keys).clamp_max(max(edge_keys.shape[0] - 1, 0))
    return (first == second) | (edge_keys[found] == keys)


def _mirror_plan(state: _State, topo: topology.EdgeTopology, plan: _Plan,
                 kind: torch.Tensor) -> tuple[_Plan, torch.Tensor]:
    """Join each collapse to its mirror image; returns the rows kept and a group per row."""
    # A collapse whose ends have no mirror runs alone; one on the plane stays on it; one across the
    # plane lands on it; any other runs only with its mirror image, placed as its reflection.
    mirror = state.mirror
    vertex_count = int(state.positions.shape[0])
    plane = -float(state.centre[0]) / state.size
    x, y = plan.x, plan.y
    mx, my = mirror[x], mirror[y]
    count = int(x.shape[0])
    row = torch.arange(count, device=x.device)
    stride = vertex_count
    plan_keys = x * stride + y
    order = torch.argsort(plan_keys)
    sorted_keys = plan_keys[order]
    image_keys = torch.minimum(mx, my).clamp_min(0) * stride + torch.maximum(mx, my).clamp_min(0)
    found = torch.searchsorted(sorted_keys, image_keys).clamp_max(max(count - 1, 0))
    mirrored = (mx >= 0) & (my >= 0)
    image = torch.where(mirrored & (sorted_keys[found] == image_keys), order[found], torch.full_like(row, -1))

    on_plane = (mx == x) & (my == y)
    across = (mx == y) & (my == x) & (x != y)
    lonely = (mx < 0) & (my < 0)
    still = (((plan.mode == TO_Y) & (mx < 0) & (my >= 0)) | ((plan.mode == TO_X) & (my < 0) & (mx >= 0)))
    single = lonely | on_plane | still

    half = (plan.mode == TO_Y) | (plan.mode == TO_X)
    locked_end = (kind[x] == LOCKED) | (kind[y] == LOCKED)
    across &= ~half & ~((plan.mode == LINE) & locked_end)
    rows_across = torch.nonzero(across).squeeze(1)
    if rows_across.numel():
        middle = rows_across[~plan.optimal[rows_across]]
        plan.t[middle] = 0.5
        plan.position[middle] = 0.5 * (state.positions[x[middle]] + state.positions[y[middle]])
        plan.position[rows_across, 0] = plane
    rows_plane = torch.nonzero(on_plane & plan.optimal).squeeze(1)
    plan.position[rows_plane, 0] = plane

    first = torch.nonzero(mirrored & ~on_plane & ~across & (image > row)).squeeze(1)
    second = image[first]
    flipped = mx[first] != x[second]
    expected = plan.mode[first].clone()
    expected[flipped & (plan.mode[first] == TO_Y)] = TO_X
    expected[flipped & (plan.mode[first] == TO_X)] = TO_Y
    apart = plan.mode[second] == expected
    for one in (x[first], y[first]):
        for other in (x[second], y[second]):
            apart &= ~_adjacent(topo, one, other, vertex_count)
    agree = apart & (plan.keep[first] == plan.keep[second])
    first, second, flipped = first[agree], second[agree], flipped[agree]

    t_first = plan.t[first]
    plan.t[second] = torch.where(flipped, 1.0 - t_first, t_first)
    plan.to_y[second] = plan.to_y[first] ^ flipped
    reflected = plan.position[first].clone()
    reflected[:, 0] = 2.0 * plane - reflected[:, 0]
    plan.position[second] = reflected
    plan.optimal[second] = plan.optimal[first]
    for name in ("cost", "error"):
        values = getattr(plan, name)
        worst = torch.maximum(values[first], values[second])
        values[first] = worst
        values[second] = worst

    alone = torch.nonzero(single | across).squeeze(1)
    rows = torch.cat([alone, first, second])
    group = torch.cat([alone, first, first])
    return plan.take(rows), group


def _round(state: _State, target_faces: int, limit_sq: float, symmetric: bool) -> tuple[int, float, str]:
    """Run one round of collapses; returns ``(collapses, largest error, stop reason or "")``."""
    device = state.device
    vertex_count = int(state.positions.shape[0])
    welded_faces = state.wedge_vertex[state.faces]
    topo = topology.edge_topology(welded_faces, vertex_count)
    kind, seam = _kinds(state, topo)
    mode = _edge_modes(topo, kind, seam)
    if not bool((mode != NONE).any()):
        return 0, 0.0, "locked"
    plan = _plan(state, topo, welded_faces, mode, kind)
    group: torch.Tensor | None = None
    if symmetric:
        plan, group = _mirror_plan(state, topo, plan, kind)
        _, group = torch.unique(group, return_inverse=True)
    row_count = int(plan.cost.shape[0])
    if row_count == 0:
        return 0, 0.0, "locked"
    if group is None:
        groups = row_count
        group_cost = plan.cost
        group_error = plan.error
    else:
        groups = int(group.max()) + 1
        group_cost = torch.zeros(groups, dtype=torch.float64, device=device)
        group_cost.scatter_reduce_(0, group, plan.cost, reduce="amax", include_self=False)
        group_error = torch.zeros(groups, dtype=torch.float64, device=device)
        group_error.scatter_reduce_(0, group, plan.error, reduce="amax", include_self=False)
    valid = group_error <= limit_sq
    valid_count = int(valid.sum())
    if valid_count == 0:
        return 0, 0.0, "error"
    priority = _priority(group_cost)
    row_priority = priority if group is None else priority[group]

    needed = int(state.faces.shape[0]) - target_faces if target_faces > 0 else torch.iinfo(torch.int64).max
    taken = torch.zeros(row_count, dtype=torch.bool, device=device)
    dead = torch.zeros(row_count, dtype=torch.bool, device=device)
    blocked = torch.zeros(vertex_count, dtype=torch.bool, device=device)
    neighbours = topology.vertex_neighbor_csr(welded_faces, vertex_count, topo)
    around = topology.vertex_face_csr(welded_faces, vertex_count)
    chosen: list[torch.Tensor] = []
    collapsed = 0
    shares = (CANDIDATE_SHARE, 1.0) if valid_count > CANDIDATE_FLOOR else (1.0,)
    for share in shares:
        if share < 1.0:
            rank = max(CANDIDATE_FLOOR, int(math.ceil(valid_count * share)))
            threshold = float(torch.kthvalue(group_cost[valid], rank).values)
            pool = valid & (group_cost <= threshold)
        else:
            pool = valid
        in_pool = pool if group is None else pool[group]
        first_claims = 0
        for _ in range(CLAIM_PASSES):
            open_rows = in_pool & ~taken & ~dead
            if group is not None:
                whole = torch.ones(groups, dtype=torch.int64, device=device)
                whole.scatter_reduce_(0, group, (open_rows & ~blocked[plan.x] & ~blocked[plan.y]).long(),
                                      reduce="amin")
                open_rows &= whole[group].bool()
            else:
                open_rows &= ~blocked[plan.x] & ~blocked[plan.y]
            rows = torch.nonzero(open_rows).squeeze(1)
            if rows.numel() == 0:
                break
            row_group = None if group is None else group[rows]
            wins = rows[_claim(welded_faces, plan.x[rows], plan.y[rows], row_priority[rows], row_group,
                               groups, vertex_count)]
            if wins.numel() == 0 or wins.numel() < CLAIM_FALLOFF * first_claims:
                break
            first_claims = max(first_claims, int(wins.numel()))
            trial = plan.take(wins)
            good = _link_ok(topo, trial, vertex_count, neighbours) & _wedges_ok(trial)
            good &= ~_flips(state, welded_faces, trial, around)
            if group is not None:
                whole = torch.ones(groups, dtype=torch.int64, device=device)
                whole.scatter_reduce_(0, group[wins], good.long(), reduce="amin")
                good = whole[group[wins]].bool()
            dead[wins[~good]] = True
            wins = wins[good]
            if wins.numel() == 0:
                continue
            removed_faces = plan.removed_faces[wins]
            if int(removed_faces.sum()) > needed:
                wins = wins[torch.argsort(row_priority[wins])]
                removed_faces = plan.removed_faces[wins]
                before = torch.cumsum(removed_faces, 0) - removed_faces
                if group is None:
                    wins = wins[before < needed]
                else:
                    first_before = torch.full((groups,), torch.iinfo(torch.int64).max, dtype=torch.int64,
                                              device=device)
                    first_before.scatter_reduce_(0, group[wins], before, reduce="amin")
                    wins = wins[first_before[group[wins]] < needed]
                if wins.numel() == 0:
                    break
            needed -= int(plan.removed_faces[wins].sum())
            taken[wins] = True
            chosen.append(wins)
            collapsed += int(wins.shape[0])
            touched = torch.zeros(vertex_count, dtype=torch.bool, device=device)
            touched[plan.x[wins]] = True
            touched[plan.y[wins]] = True
            blocked[welded_faces[touched[welded_faces].any(1)].reshape(-1)] = True
            if needed <= 0:
                break
        if needed <= 0 or collapsed >= WIDEN_BELOW * row_count:
            break
    if not chosen:
        return 0, 0.0, "blocked" if valid_count == groups else "error"
    rows = torch.cat(chosen)
    plan = plan.take(rows)
    largest = float(plan.error.max())
    pairs = None
    if group is not None:
        chosen_group = group[rows]
        order = torch.argsort(chosen_group, stable=True)
        same = chosen_group[order][1:] == chosen_group[order][:-1]
        pairs = (order[:-1][same], order[1:][same])
    _apply(state, plan, kind)
    if pairs is not None and pairs[0].numel():
        _, survivor, _ = _ends(plan)
        first, second = survivor[pairs[0]], survivor[pairs[1]]
        state.mirror[first] = second
        state.mirror[second] = first
    return int(rows.shape[0]), largest, ""


def _compact(state: _State) -> None:
    """Drop vertices and wedges no face uses, renumbering the rest."""
    device = state.device
    used_wedges = torch.zeros(state.wedge_vertex.shape[0], dtype=torch.bool, device=device)
    used_wedges[state.faces.reshape(-1)] = True
    wedge_ids = torch.nonzero(used_wedges).squeeze(1)
    used_vertices = torch.zeros(state.positions.shape[0], dtype=torch.bool, device=device)
    used_vertices[state.wedge_vertex[wedge_ids]] = True
    vertex_ids = torch.nonzero(used_vertices).squeeze(1)
    vertex_map = torch.full((state.positions.shape[0],), -1, dtype=torch.int64, device=device)
    vertex_map[vertex_ids] = torch.arange(vertex_ids.shape[0], device=device)
    wedge_map = torch.full((state.wedge_vertex.shape[0],), -1, dtype=torch.int64, device=device)
    wedge_map[wedge_ids] = torch.arange(wedge_ids.shape[0], device=device)
    state.positions = state.positions[vertex_ids]
    state.vertex_quadrics = state.vertex_quadrics[vertex_ids]
    state.spread = state.spread[vertex_ids]
    state.locked = state.locked[vertex_ids]
    if state.mirror is not None:
        partner = state.mirror[vertex_ids]
        state.mirror = torch.where(partner >= 0, vertex_map[partner.clamp_min(0)], partner)
    state.wedge_vertex = vertex_map[state.wedge_vertex[wedge_ids]]
    state.wedge_part = state.wedge_part[wedge_ids]
    state.wedge_quadrics = state.wedge_quadrics[wedge_ids]
    state.values = state.values[wedge_ids]
    state.value_weights = state.value_weights[wedge_ids]
    state.carried = state.carried[wedge_ids]
    if state.cross_seams:
        state.hard_wedge = state.hard_wedge[wedge_ids]
        state.hard_vertex = vertex_map[state.hard_vertex].clamp_min(0)
    state.faces = wedge_map[state.faces]


def _extract(state: _State, faces: torch.Tensor, face_part: torch.Tensor, positions: torch.Tensor,
             wedge_vertex: torch.Tensor, raw: torch.Tensor, carried: torch.Tensor) -> list[TriMesh]:
    """Split the working mesh back into one `TriMesh` per input mesh, on the CPU."""
    results = []
    world = positions * state.size + state.centre
    for index in range(state.mesh_count):
        own = faces[face_part == index]
        used, local = torch.unique(own.reshape(-1), return_inverse=True)
        vertices = world[wedge_vertex[used]].float().cpu()
        attributes: dict[str, torch.Tensor] = {}
        for name, start, width in state.quadric_layout:
            if name in state.mesh_names[index]:
                attributes[name] = _finish(name, raw[used, start:start + width]).float().cpu()
        for name, start, width in state.carried_layout:
            if name in state.mesh_names[index]:
                attributes[name] = _finish(name, carried[used, start:start + width].float()).cpu()
        results.append(TriMesh(vertices, local.reshape(-1, 3).cpu(), attributes, None))
    return results


def _raw_values(state: _State) -> torch.Tensor:
    """Quadric attribute values back in their own units."""
    weights = state.value_weights
    return torch.where(weights > 0, state.values / weights.clamp_min(1e-300), torch.zeros_like(state.values))


def check_size(triangles: int, carried_values: int = 0) -> None:
    """Refuse a reduction larger than one call takes, before anything is built for it.

    Args:
        triangles: Triangles to reduce.
        carried_values: Vertices times the width of their skin weights and shape keys.

    Raises:
        ValueError: Naming the limit passed and how to stay under it.
    """
    if triangles > MAX_TRIANGLES:
        raise ValueError(f"the parts chosen hold {triangles:,} triangles, more than the {MAX_TRIANGLES:,} one "
                         "decimation takes; decimate fewer parts at a time with the parts input")
    if carried_values > MAX_CARRIED_VALUES:
        raise ValueError(f"the parts chosen carry {carried_values:,} skin weight and shape key values, more "
                         f"than the {MAX_CARRIED_VALUES:,} one decimation takes; decimate fewer parts at a "
                         "time, or remove shape keys the model does not need")


def _check_input(meshes: list[TriMesh]) -> int:
    """Total triangles, refusing an empty or oversized input."""
    total = sum(mesh.face_count for mesh in meshes)
    if total == 0:
        raise ValueError("the parts chosen have no triangles to decimate; pick parts that hold faces")
    carried = sum(mesh.vertex_count * sum(1 if value.dim() == 1 else int(value.shape[1])
                                          for name, value in mesh.attributes.items()
                                          if name not in dict(QUADRIC_ATTRIBUTES))
                  for mesh in meshes)
    check_size(total, carried)
    return total


def _frame(meshes: list[TriMesh], size: float | None, centre: torch.Tensor | None) -> tuple[float, torch.Tensor]:
    """Size and centre to normalise by, from the meshes when not given."""
    if size is not None and centre is not None and size > 0:
        return float(size), centre.double().cpu()
    points = torch.cat([mesh.vertices.double().cpu() for mesh in meshes if mesh.vertex_count], 0)
    low, high = points.amin(0), points.amax(0)
    extent = float((high - low).max())
    return (extent if extent > 0 else 1.0), (0.5 * (low + high))


def decimate(meshes: list[TriMesh], target_triangles: int = 0, max_error: float = 0.01,
             placement: str | list[str] = "optimal", borders: str = "slide", seams: str = "slide",
             symmetry: str = "none", locked_points: torch.Tensor | None = None,
             size: float | None = None, centre: torch.Tensor | None = None,
             device: torch.device | str | None = None,
             progress: Callable[[float], None] | None = None) -> tuple[list[TriMesh], DecimateStats]:
    """Reduce meshes together by quadric edge collapse, keeping seams, borders and attributes.

    Args:
        meshes: The meshes. Positions they share are welded, so they stay joined.
        target_triangles: Triangles to stop at, across every mesh; 0 stops on error alone.
        max_error: Largest collapse error allowed, as a fraction of ``size``.
        placement: One of `PLACEMENTS`, or one per mesh.
        borders: One of `BORDER_MODES`.
        seams: One of `SEAM_MODES`, for UV, normal and colour seams and where parts meet.
        symmetry: One of `SYMMETRY`; ``"x"`` collapses mirror images across x = 0 together.
        locked_points: ``(K, 3)`` positions that never move, such as where unchosen parts touch.
        size: Length errors are measured against; None uses the meshes' longest side.
        centre: ``(3,)`` centre the positions are normalised about.
        device: Device to run on; None means `compute_device`.
        progress: Called after each round with the share of the work done, 0 to 1.

    Returns:
        ``(meshes, stats)``: one reduced `TriMesh` per input, on the CPU, without polygons.

    Raises:
        ValueError: For no triangles, too many, or an unknown option.
    """
    _check_options(placement, borders, seams, symmetry, len(meshes))
    started = time.perf_counter()
    total = _check_input(meshes)
    size, centre = _frame(meshes, size, centre)
    placements = [placement] * len(meshes) if isinstance(placement, str) else list(placement)
    keep = [choice == "keep vertices" for choice in placements]
    stats = DecimateStats(triangles_before=total, vertices_before=sum(mesh.vertex_count for mesh in meshes))

    def work(where: torch.device) -> list[TriMesh]:
        state = _setup(meshes, size, centre, keep, borders, seams, locked_points, where)
        stats.device = str(where)
        stats.locked = int(state.locked.sum())
        symmetric = symmetry == "x"
        if symmetric:
            state.mirror = _mirror_map(state)
            stats.mirrored = float((state.mirror >= 0).double().mean()) if state.mirror.numel() else 0.0
        limit_sq = float(max_error) ** 2
        target = max(int(target_triangles), 0)
        start_faces = int(state.faces.shape[0])
        stats.rounds = stats.collapses = 0
        stats.error = 0.0
        stats.stop = "target"
        largest_sq = 0.0
        table = int(state.positions.shape[0])
        while target == 0 or state.faces.shape[0] > target:
            if stats.rounds >= MAX_ROUNDS:
                stats.stop = "rounds"
                break
            done, largest, reason = _round(state, target, limit_sq, symmetric)
            if reason:
                stats.stop = reason
                break
            stats.rounds += 1
            stats.collapses += done
            largest_sq = max(largest_sq, largest)
            if progress is not None:
                if target:
                    progress((start_faces - int(state.faces.shape[0])) / max(start_faces - target, 1))
                else:
                    progress(min(math.sqrt(max(largest_sq, 0.0)) / max(float(max_error), 1e-12), 1.0))
            if state.faces.shape[0] < table:
                _compact(state)
                table = int(state.positions.shape[0])
        stats.error = math.sqrt(max(largest_sq, 0.0))
        _compact(state)
        raw = _raw_values(state)
        return _extract(state, state.faces, state.face_part, state.positions, state.wedge_vertex,
                        raw, state.carried)

    results = with_cpu_fallback(work, device if device is not None else compute_device(), "Mesh Decimate")
    stats.triangles_after = sum(mesh.face_count for mesh in results)
    stats.vertices_after = sum(mesh.vertex_count for mesh in results)
    stats.seconds = time.perf_counter() - started
    return results, stats


def _check_options(placement, borders: str, seams: str, symmetry: str, count: int) -> None:
    """Refuse unknown option values with a message naming the choices."""
    placements = [placement] if isinstance(placement, str) else list(placement)
    if not isinstance(placement, str) and len(placements) != count:
        raise ValueError("give one placement, or one per mesh")
    for choice in placements:
        if choice not in PLACEMENTS:
            raise ValueError(f"placement must be one of {', '.join(PLACEMENTS)}, not {choice!r}")
    if borders not in BORDER_MODES:
        raise ValueError(f"borders must be one of {', '.join(BORDER_MODES)}, not {borders!r}")
    if seams not in SEAM_MODES:
        raise ValueError(f"seams must be one of {', '.join(SEAM_MODES)}, not {seams!r}")
    if symmetry not in SYMMETRY:
        raise ValueError(f"symmetry must be one of {', '.join(SYMMETRY)}, not {symmetry!r}")


# ---------------------------------------------------------------------------------------------
# Grid clustering


def _cells(positions: torch.Tensor, low: torch.Tensor, cell: float) -> torch.Tensor:
    """Cell index of every position on a grid of the given cell size."""
    index = torch.floor((positions - low) / cell).long().clamp_min(0)
    span = int(index.max()) + 1 if index.numel() else 1
    if span < (1 << 20):
        keys = (index[:, 0] << 40) | (index[:, 1] << 20) | index[:, 2]
        _, cells = torch.unique(keys, return_inverse=True)
    else:
        _, cells = torch.unique(index, dim=0, return_inverse=True)
    return cells


def _clustered_faces(welded_faces: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """Indices of the faces that survive clustering: three cells, first of their cell triple."""
    corners = cells[welded_faces]
    alive = (corners[:, 0] != corners[:, 1]) & (corners[:, 1] != corners[:, 2]) & (corners[:, 2] != corners[:, 0])
    rows = torch.nonzero(alive).squeeze(1)
    corners = corners[rows]
    lowest = corners.argmin(1)
    turn = torch.stack([lowest, (lowest + 1) % 3, (lowest + 2) % 3], 1)
    canonical = torch.gather(corners, 1, turn)
    _, inverse = torch.unique(canonical, dim=0, return_inverse=True)
    first = torch.full((int(inverse.max()) + 1 if inverse.numel() else 0,), rows.shape[0],
                       dtype=torch.int64, device=rows.device)
    first.scatter_reduce_(0, inverse, torch.arange(rows.shape[0], device=rows.device), reduce="amin")
    return rows[first]


def cluster(meshes: list[TriMesh], target_triangles: int = 0, max_error: float = 0.01,
            symmetry: str = "none", size: float | None = None, centre: torch.Tensor | None = None,
            device: torch.device | str | None = None,
            progress: Callable[[float], None] | None = None) -> tuple[list[TriMesh], DecimateStats]:
    """Reduce meshes by snapping vertices to a grid, ignoring topology, for extreme cuts.

    Args:
        meshes: The meshes, reduced together.
        target_triangles: Triangles to stay at or under; 0 sizes the grid from ``max_error``.
        max_error: Cell size as a fraction of ``size`` when ``target_triangles`` is 0.
        symmetry: One of `SYMMETRY`; ``"x"`` centres the grid on x = 0.
        size: Length errors are measured against; None uses the meshes' longest side.
        centre: ``(3,)`` centre the positions are normalised about.
        device: Device to run on; None means `compute_device`.
        progress: Called after each grid refinement with the share of the work done, 0 to 1.

    Returns:
        ``(meshes, stats)``.

    Raises:
        ValueError: For no triangles, too many, or an unknown option.
    """
    # Each cell's vertex sits at its quadric optimum inside the cell; attributes are averaged per
    # cell and UV island, so islands keep their own texture coordinates.
    _check_options("optimal", "slide", "slide", symmetry, len(meshes))
    started = time.perf_counter()
    total = _check_input(meshes)
    size, centre = _frame(meshes, size, centre)
    stats = DecimateStats(triangles_before=total, vertices_before=sum(mesh.vertex_count for mesh in meshes))

    def work(where: torch.device) -> list[TriMesh]:
        state = _setup(meshes, size, centre, [False] * len(meshes), "slide", "slide", None, where,
                       quadric_attributes=False)
        stats.device = str(where)
        welded_faces = state.wedge_vertex[state.faces]
        positions = state.positions
        low = positions.amin(0) - 1e-9
        plane = -float(state.centre[0]) / state.size
        target = max(int(target_triangles), 0)

        def grid_low(cell: float) -> torch.Tensor:
            if symmetry != "x":
                return low
            shifted = low.clone()
            shifted[0] = plane - cell * (math.floor((plane - float(low[0])) / cell) + 0.5)
            return shifted

        if target == 0:
            cell = max(float(max_error), 1.0 / MAX_GRID)
            stats.rounds = 1
        else:
            fine, coarse = 1.0 / MAX_GRID, 1.0
            cell = coarse
            for step in range(CLUSTER_STEPS):
                trial = math.sqrt(fine * coarse)
                count = int(_clustered_faces(welded_faces, _cells(positions, grid_low(trial), trial)).shape[0])
                stats.rounds = step + 1
                if count <= target:
                    coarse = trial
                    cell = trial
                else:
                    fine = trial
                if progress is not None:
                    progress((step + 1) / CLUSTER_STEPS)
                if coarse / fine < 1.02:
                    break
        cells = _cells(positions, grid_low(cell), cell)
        keep_faces = _clustered_faces(welded_faces, cells)
        cell_count = int(cells.max()) + 1
        quadric = torch.zeros(cell_count, 11, dtype=torch.float64, device=where)
        quadric.index_add_(0, cells, state.vertex_quadrics)
        members = torch.bincount(cells, minlength=cell_count).clamp_min(1).double()[:, None]
        mean = torch.zeros(cell_count, 3, dtype=torch.float64, device=where).index_add_(0, cells, positions) / members
        lowest = torch.full((cell_count, 3), float("inf"), dtype=torch.float64, device=where)
        lowest.scatter_reduce_(0, cells[:, None].expand(-1, 3), positions, reduce="amin")
        highest = torch.full((cell_count, 3), float("-inf"), dtype=torch.float64, device=where)
        highest.scatter_reduce_(0, cells[:, None].expand(-1, 3), positions, reduce="amax")
        a6 = quadric[:, 0:6].clone()
        ridge = 1e-3 * (a6[:, 0] + a6[:, 3] + a6[:, 5]).abs() / 3.0 + 1e-300
        a6[:, 0] += ridge
        a6[:, 3] += ridge
        a6[:, 5] += ridge
        solution, det = _solve_sym(a6, -quadric[:, 6:9] + ridge[:, None] * mean)
        good = (det > 0) & torch.isfinite(solution).all(1)
        placed = torch.where(good[:, None], torch.maximum(torch.minimum(solution, highest), lowest), mean)

        islands = topology.connected_components(state.faces, int(state.wedge_vertex.shape[0]))
        key = cells[state.wedge_vertex] * (int(islands.max()) + 1) + islands
        _, new_wedge = torch.unique(key, return_inverse=True)
        wedge_count = int(new_wedge.max()) + 1
        weight = torch.bincount(new_wedge, minlength=wedge_count).clamp_min(1)[:, None]
        raw = _raw_values(state)
        raw = torch.zeros(wedge_count, raw.shape[1], dtype=raw.dtype, device=where).index_add_(0, new_wedge, raw)
        raw = raw / weight.to(raw.dtype)
        carried = torch.zeros(wedge_count, state.carried.shape[1], dtype=torch.float32,
                              device=where).index_add_(0, new_wedge, state.carried) / weight.float()
        wedge_vertex = torch.zeros(wedge_count, dtype=torch.int64, device=where)
        wedge_vertex[new_wedge] = cells[state.wedge_vertex]
        faces = new_wedge[state.faces[keep_faces]]
        face_part = state.face_part[keep_faces]
        stats.stop = "grid"
        stats.error = cell * math.sqrt(3.0) / 2.0
        return _extract(state, faces, face_part, placed, wedge_vertex, raw, carried)

    results = with_cpu_fallback(work, device if device is not None else compute_device(), "Mesh Decimate")
    stats.triangles_after = sum(mesh.face_count for mesh in results)
    stats.vertices_after = sum(mesh.vertex_count for mesh in results)
    stats.collapses = stats.vertices_before - stats.vertices_after
    stats.seconds = time.perf_counter() - started
    return results, stats

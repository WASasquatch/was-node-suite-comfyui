"""Orientation and position fields of a quad remesh, solved coarse to fine over a hierarchy.
Orientations are unit tangents with four-fold symmetry. Positions are lattice origins in each
vertex's tangent plane, spaced by the vertex's scale."""
# Colours of a level update in turn.

from __future__ import annotations

__all__ = [
    "lattice_round", "middle_point", "orientation_energy", "random_tangents", "solve_orientation",
    "solve_position",
]

import math

import torch

from .hierarchy import Level

#: Smoothing sweeps per level for the orientation field.
ORIENTATION_SWEEPS = 6

#: Smoothing sweeps per level for the position field.
POSITION_SWEEPS = 6


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise dot product."""
    return (a * b).sum(-1)


def _power4(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Real and imaginary parts of ``(x + iy) ** 4``."""
    xx, yy = x * x, y * y
    return (xx - yy) ** 2 - 4.0 * xx * yy, 4.0 * x * y * (xx - yy)


def random_tangents(normals: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Random unit vectors in each vertex's tangent plane.

    Args:
        normals: ``(V, 3)`` unit normals.
        generator: CPU random generator.

    Returns:
        ``(V, 3)`` unit tangents.
    """
    pick = torch.randn(normals.shape[0], 3, generator=generator).to(normals.device, normals.dtype)
    return project_tangent(pick, normals)


def project_tangent(vectors: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    """Vectors projected into each tangent plane and made unit, any tangent where one vanishes.

    Args:
        vectors: ``(V, 3)`` vectors.
        normals: ``(V, 3)`` unit normals.

    Returns:
        ``(V, 3)`` unit tangents.
    """
    flat = vectors - normals * _dot(vectors, normals)[:, None]
    length = flat.norm(dim=1)
    weak = length < 1e-6
    if bool(weak.any()):
        axis = torch.zeros_like(normals[weak])
        axis[:, 0] = 1.0
        other = torch.zeros_like(axis)
        other[:, 1] = 1.0
        axis = torch.where((normals[weak, 0].abs() > 0.9)[:, None], other, axis)
        flat[weak] = torch.cross(normals[weak], axis, dim=1)
    return torch.nn.functional.normalize(flat, dim=1)


def middle_point(p0: torch.Tensor, n0: torch.Tensor, p1: torch.Tensor, n1: torch.Tensor) -> torch.Tensor:
    """Point nearest two points that lies on both of their tangent planes.

    Args:
        p0: ``(E, 3)`` first points.
        n0: ``(E, 3)`` their unit normals.
        p1: ``(E, 3)`` second points.
        n1: ``(E, 3)`` their unit normals.

    Returns:
        ``(E, 3)`` points.
    """
    n0p0, n0p1 = _dot(n0, p0), _dot(n0, p1)
    n1p0, n1p1 = _dot(n1, p0), _dot(n1, p1)
    n0n1 = _dot(n0, n1)
    denom = 1.0 / (1.0 - n0n1 * n0n1 + 1e-4)
    lambda0 = 2.0 * (n0p1 - n0p0 - n0n1 * (n1p0 - n1p1)) * denom
    lambda1 = 2.0 * (n1p0 - n1p1 - n0n1 * (n0p1 - n0p0)) * denom
    return 0.5 * (p0 + p1) - 0.25 * (n0 * lambda0[:, None] + n1 * lambda1[:, None])


def lattice_round(origin: torch.Tensor, q: torch.Tensor, t: torch.Tensor, scale: torch.Tensor,
                  point: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The lattice point nearest a point, for lattices ``origin + scale * (a q + b t)``.

    Args:
        origin: ``(E, 3)`` lattice origins.
        q: ``(E, 3)`` first lattice axes.
        t: ``(E, 3)`` second lattice axes.
        scale: ``(E,)`` spacings.
        point: ``(E, 3)`` points.

    Returns:
        ``(lattice point, a, b)``, ``a`` and ``b`` the integer steps from ``origin``.
    """
    offset = point - origin
    a = torch.round(_dot(offset, q) / scale)
    b = torch.round(_dot(offset, t) / scale)
    return origin + q * (a * scale)[:, None] + t * (b * scale)[:, None], a, b


def _orientation_sweep(level: Level) -> None:
    """One Gauss-Seidel sweep of the orientation field over every colour of a level."""
    q_field, normals = level.q, level.normals
    given = level.constraints
    constrained = bool((given.direction_weight > 0).any())
    for colour in range(level.colours):
        v0, v1 = level.vertex_offsets[colour], level.vertex_offsets[colour + 1]
        e0, e1 = level.edge_offsets[colour], level.edge_offsets[colour + 1]
        if v1 == v0:
            continue
        verts = level.order[v0:v1]
        qi, ni = q_field[verts], normals[verts]
        ti = torch.cross(ni, qi, dim=1)
        real = torch.zeros(v1 - v0, dtype=qi.dtype, device=qi.device)
        imag = torch.zeros_like(real)
        if e1 > e0:
            local, nb, weight = level.src_local[e0:e1], level.dst[e0:e1], level.edge_weight[e0:e1]
            qe, te = qi[local], ti[local]
            qj = q_field[nb]
            tj = torch.cross(normals[nb], qj, dim=1)
            r1, i1 = _power4(_dot(qj, qe), _dot(qj, te))
            r2, i2 = _power4(_dot(tj, qe), _dot(tj, te))
            real.index_add_(0, local, weight * (r1 + r2))
            imag.index_add_(0, local, weight * (i1 + i2))
        if constrained:
            pull = given.direction_weight[verts]
            target = given.direction[verts]
            cr, ci = _power4(_dot(target, qi), _dot(target, ti))
            size = torch.sqrt(real * real + imag * imag).clamp_min(1e-30)
            csize = torch.sqrt(cr * cr + ci * ci).clamp_min(1e-30)
            active = pull > 0
            real = torch.where(active, (1.0 - pull) * real / size + pull * cr / csize, real)
            imag = torch.where(active, (1.0 - pull) * imag / size + pull * ci / csize, imag)
        moved = (real * real + imag * imag) > 0
        angle = torch.where(moved, torch.atan2(imag, real) / 4.0, torch.zeros_like(real))
        turned = qi * torch.cos(angle)[:, None] + ti * torch.sin(angle)[:, None]
        q_field[verts] = torch.nn.functional.normalize(turned, dim=1)


def orientation_energy(level: Level) -> float:
    """Mean angle in degrees between each edge's orientations, matched four-fold."""
    if level.first.numel() == 0:
        return 0.0
    a, b = level.first, level.second
    qa, na = level.q[a], level.normals[a]
    ta = torch.cross(na, qa, dim=1)
    qb, tb = level.q[b], torch.cross(level.normals[b], level.q[b], dim=1)
    best = torch.stack([_dot(qa, qb).abs(), _dot(qa, tb).abs()], 1).amax(1)
    across = torch.stack([_dot(ta, qb).abs(), _dot(ta, tb).abs()], 1).amax(1)
    return float(torch.rad2deg(torch.acos(torch.minimum(best, across).clamp(-1.0, 1.0))).mean())


def solve_orientation(levels: list[Level], finest: int, generator: torch.Generator,
                      sweeps: int = ORIENTATION_SWEEPS) -> None:
    """Solve the orientation field from the coarsest level down to ``finest``.

    Args:
        levels: The hierarchy, coloured from ``finest`` down.
        finest: Index of the finest level solved.
        generator: CPU random generator for the coarsest level's start.
        sweeps: Sweeps per level.
    """
    last = len(levels) - 1
    levels[last].q = random_tangents(levels[last].normals, generator)
    for index in range(last, finest - 1, -1):
        level = levels[index]
        if index != last:
            level.q = project_tangent(levels[index + 1].q[level.parent], level.normals)
        for _ in range(sweeps):
            _orientation_sweep(level)


def _position_sweep(level: Level) -> None:
    """One Gauss-Seidel sweep of the position field over every colour of a level."""
    o_field, q_field, normals = level.o, level.q, level.normals
    vertices, scales = level.vertices, level.scales
    given = level.constraints
    constrained = bool((given.point_weight > 0).any())
    tau = 2.0 * math.pi
    for colour in range(level.colours):
        v0, v1 = level.vertex_offsets[colour], level.vertex_offsets[colour + 1]
        e0, e1 = level.edge_offsets[colour], level.edge_offsets[colour + 1]
        if v1 == v0:
            continue
        verts = level.order[v0:v1]
        vi, ni, qi, si, oi = vertices[verts], normals[verts], q_field[verts], scales[verts], o_field[verts]
        ti = torch.cross(ni, qi, dim=1)
        shift = torch.zeros_like(oi)
        if e1 > e0:
            local, nb, weight = level.src_local[e0:e1], level.dst[e0:e1], level.edge_weight[e0:e1]
            qe, te, se = qi[local], ti[local], si[local]
            nj, qj = normals[nb], q_field[nb]
            tj = torch.cross(nj, qj, dim=1)
            middle = middle_point(vi[local], ni[local], vertices[nb], nj)
            mine, _, _ = lattice_round(oi[local], qe, te, se, middle)
            theirs, _, _ = lattice_round(o_field[nb], qj, tj, scales[nb], mine)
            gap = theirs - mine
            x = tau * _dot(gap, qe) / se
            y = tau * _dot(gap, te) / se
            count = v1 - v0
            sums = torch.zeros(4, count, dtype=oi.dtype, device=oi.device)
            sums[0].index_add_(0, local, weight * torch.sin(x))
            sums[1].index_add_(0, local, weight * torch.cos(x))
            sums[2].index_add_(0, local, weight * torch.sin(y))
            sums[3].index_add_(0, local, weight * torch.cos(y))
            step_q = torch.atan2(sums[0], sums[1]) / tau * si
            step_t = torch.atan2(sums[2], sums[3]) / tau * si
            shift = qi * step_q[:, None] + ti * step_t[:, None]
        moved = oi + shift
        if constrained:
            pull = given.point_weight[verts]
            line = given.line[verts]
            towards = given.point[verts] - moved
            towards = towards - line * _dot(line, towards)[:, None]
            moved = moved + towards * pull[:, None]
            moved = moved - ni * _dot(ni, moved - vi)[:, None]
        o_field[verts], _, _ = lattice_round(moved, qi, ti, si, vi)


def solve_position(levels: list[Level], finest: int, start: int, generator: torch.Generator,
                   sweeps: int = POSITION_SWEEPS) -> None:
    """Solve the position field from level ``start`` down to ``finest``.

    Args:
        levels: The hierarchy with orientations solved, coloured from ``finest`` to ``start``.
        finest: Index of the finest level solved.
        start: Index of the coarsest level solved.
        generator: CPU random generator for the starting origins.
        sweeps: Sweeps per level.
    """
    level = levels[start]
    t = torch.cross(level.normals, level.q, dim=1)
    jitter = (torch.rand(level.count, 2, generator=generator) - 0.5).to(level.vertices.device, level.vertices.dtype)
    seeded = level.vertices + (level.q * jitter[:, :1] + t * jitter[:, 1:]) * level.scales[:, None]
    level.o, _, _ = lattice_round(seeded, level.q, t, level.scales, level.vertices)
    for index in range(start, finest - 1, -1):
        level = levels[index]
        if index != start:
            coarse = levels[index + 1].o[level.parent]
            coarse = coarse - level.normals * _dot(level.normals, coarse - level.vertices)[:, None]
            t = torch.cross(level.normals, level.q, dim=1)
            level.o, _, _ = lattice_round(coarse, level.q, t, level.scales, level.vertices)
        for _ in range(sweeps):
            _position_sweep(level)

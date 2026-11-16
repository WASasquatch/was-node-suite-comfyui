"""Measuring a model before export: topology, triangle quality, texture coordinates and skin.

Edges, shells and holes are counted on vertices welded by exact position. Angles are in degrees, lengths
in model units.
"""

from __future__ import annotations

__all__ = ["inspect", "report_text", "self_intersections", "uv_coverage"]

import math

import torch

from ..model3d.scene import Model3D
from . import topology
from .clean import position_ids

#: Share of an edge's length at each end where a crossing is not counted, so touching neighbours never count.
EDGE_MARGIN = 1e-3

#: Crossings followed along one edge.
MAX_CROSSINGS = 4

#: Pixel tests one texture coordinate raster may make before its resolution is halved.
MAX_UV_CANDIDATES = 40_000_000

#: Pixel tests made per pass of the raster.
UV_CHUNK = 4_000_000

#: Aspect ratio above which a triangle counts as poor; 1 is equilateral.
POOR_ASPECT = 10.0

#: Upper edges of the minimum angle histogram, in degrees.
ANGLE_BINS = (10, 20, 30, 40, 50, 60)

#: Valences listed one by one; the rest are grouped below and above.
VALENCES = (3, 4, 5, 6, 7, 8)

#: Weight below which a skin influence counts as empty.
EMPTY_WEIGHT = 1e-6


def self_intersections(vertices: torch.Tensor, faces: torch.Tensor, device: torch.device | str | None = None,
                       max_crossings: int = MAX_CROSSINGS) -> torch.Tensor:
    """Faces that cross another face of the same mesh, by tracing every edge through a tree.

    Faces lying flat on each other are not counted.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` vertex indices.
        device: Device for the tree; None means the mesh compute device.
        max_crossings: Crossings followed along one edge.

    Returns:
        ``(F,)`` bool on the faces' device.
    """
    from .bvh import TriangleBVH

    home = faces.device
    count = int(faces.shape[0])
    crossing = torch.zeros(count, dtype=torch.bool, device=home)
    if count < 2:
        return crossing
    ids, positions = position_ids(vertices)
    welded = ids[faces.long()]
    point_count = positions.shape[0]
    edges = topology.edge_topology(welded, point_count)
    tree = TriangleBVH(positions, welded, device=device)
    start = positions[edges.edges[:, 0]]
    direction = positions[edges.edges[:, 1]] - start
    active = torch.nonzero(direction.abs().amax(dim=1) > 0).squeeze(1)
    floor = torch.full((edges.edge_count,), EDGE_MARGIN, dtype=torch.float32, device=home)
    hit_edges, hit_faces = [], []
    for _ in range(max(1, int(max_crossings))):
        if not active.numel():
            break
        t, face, _bary = tree.ray_cast(start[active], direction[active], floor[active], 1.0 - EDGE_MARGIN)
        face = face.to(home)
        found = face >= 0
        if not bool(found.any()):
            break
        active = active[found]
        hit_edges.append(active)
        hit_faces.append(face[found])
        floor[active] = t.to(home)[found] + EDGE_MARGIN
    if not hit_edges:
        return crossing
    crossed_edge = torch.zeros(edges.edge_count, dtype=torch.bool, device=home)
    crossed_edge[torch.cat(hit_edges)] = True
    crossing[torch.cat(hit_faces)] = True
    crossing |= crossed_edge[edges.face_edges].any(dim=1)
    return crossing


def uv_coverage(uvs: torch.Tensor, faces: torch.Tensor, resolution: int = 1024) -> tuple[torch.Tensor, int]:
    """How many triangles cover each pixel of the unit texture square.

    Args:
        uvs: ``(V, 2)`` texture coordinates.
        faces: ``(F, 3)`` vertex indices.
        resolution: Pixels along each side; halved until the work fits.

    Returns:
        ``(counts, resolution)``: ``(R * R,)`` int64 triangles per pixel and the resolution used.
    """
    device = uvs.device
    corners = uvs[faces.long()].double()
    finite = torch.isfinite(corners).all(dim=2).all(dim=1)
    corners = corners[finite]
    area = ((corners[:, 1, 0] - corners[:, 0, 0]) * (corners[:, 2, 1] - corners[:, 0, 1])
            - (corners[:, 2, 0] - corners[:, 0, 0]) * (corners[:, 1, 1] - corners[:, 0, 1]))
    corners = corners[area != 0]
    area = area[area != 0]
    corners = torch.where((area < 0)[:, None, None], corners[:, [0, 2, 1]], corners)
    resolution = max(16, int(resolution))
    while True:
        pixels = corners * resolution
        low = torch.floor(pixels.amin(dim=1) - 0.5).clamp(0, resolution - 1).long()
        high = torch.ceil(pixels.amax(dim=1) - 0.5).clamp(-1, resolution - 1).long()
        inside = (pixels.amax(dim=1) >= 0).all(dim=1) & (pixels.amin(dim=1) <= resolution).all(dim=1)
        width = (high[:, 0] - low[:, 0] + 1).clamp_min(0) * inside
        height = (high[:, 1] - low[:, 1] + 1).clamp_min(0) * inside
        work = width * height
        if int(work.sum()) <= MAX_UV_CANDIDATES or resolution <= 16:
            break
        resolution //= 2
    counts = torch.zeros(resolution * resolution, dtype=torch.int64, device=device)
    cumulative = torch.cumsum(work, 0)
    begin = 0
    total = int(cumulative[-1]) if cumulative.numel() else 0
    while begin < corners.shape[0] and total:
        reached = int(cumulative[begin - 1]) if begin else 0
        end = int(torch.searchsorted(cumulative, torch.tensor(reached + UV_CHUNK, device=device), right=True))
        end = max(end, begin + 1)
        rows = torch.arange(begin, min(end, corners.shape[0]), device=device)
        begin = int(rows[-1]) + 1
        size = work[rows]
        if int(size.sum()) == 0:
            continue
        face = torch.repeat_interleave(rows, size)
        rank = torch.arange(face.shape[0], device=device) - torch.repeat_interleave(torch.cumsum(size, 0) - size, size)
        x = low[face, 0] + rank % width[face]
        y = low[face, 1] + rank // width[face]
        centre = torch.stack([x.double() + 0.5, y.double() + 0.5], dim=1)
        p = corners[face] * resolution
        covered = torch.ones(face.shape[0], dtype=torch.bool, device=device)
        for side in range(3):
            a, b = p[:, side], p[:, (side + 1) % 3]
            dx, dy = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]
            value = dx * (centre[:, 1] - a[:, 1]) - dy * (centre[:, 0] - a[:, 0])
            owned = (dy > 0) | ((dy == 0) & (dx < 0))
            covered &= (value > 0) | ((value == 0) & owned)
        counts += torch.bincount(y[covered] * resolution + x[covered], minlength=resolution * resolution)
    return counts, resolution


def _weighted_quantiles(values: torch.Tensor, weights: torch.Tensor, quantiles: tuple[float, ...]) -> list[float]:
    """Quantiles of values weighted by area."""
    if values.numel() == 0:
        return [0.0 for _ in quantiles]
    order = torch.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = torch.cumsum(weights, 0)
    total = float(cumulative[-1])
    if total <= 0:
        return [float(values[int(q * (values.numel() - 1))]) for q in quantiles]
    found = []
    for q in quantiles:
        at = int(torch.searchsorted(cumulative, torch.tensor(q * total, dtype=cumulative.dtype, device=values.device)))
        found.append(float(values[min(at, values.numel() - 1)]))
    return found


def _topology_figures(vertices: torch.Tensor, faces: torch.Tensor) -> dict:
    """Edge, shell, hole, genus and orientation counts of one part."""
    ids, positions = position_ids(vertices)
    welded = ids[faces]
    point_count = positions.shape[0]
    edges = topology.edge_topology(welded, point_count)
    non_manifold_points = topology.non_manifold_vertices(welded, point_count, edges)
    chains, closed = topology.boundary_chains(welded, point_count, edges)
    pair_ids, pairs = edges.manifold_face_pairs()
    sides = edges.edge_corners[pair_ids]
    reversed_flags = edges.face_edge_reversed
    inconsistent = int((reversed_flags[pairs[:, 0], sides[:, 0]] == reversed_flags[pairs[:, 1], sides[:, 1]]).sum())
    piece = topology.face_components(welded, point_count)
    pieces = int(piece.max()) + 1 if piece.numel() else 0
    edge_piece = piece[edges.edge_faces[:, 0].clamp_min(0)]
    used = torch.unique(welded.reshape(-1))
    point_piece = torch.full((point_count,), -1, dtype=torch.int64, device=faces.device)
    point_piece[welded.reshape(-1)] = piece[:, None].expand(-1, 3).reshape(-1)
    v_count = torch.bincount(point_piece[used], minlength=pieces)
    e_count = torch.bincount(edge_piece, minlength=pieces)
    f_count = torch.bincount(piece, minlength=pieces)
    loops = torch.zeros(pieces, dtype=torch.int64, device=faces.device)
    for chain, is_closed in zip(chains, closed.tolist()):
        if is_closed and chain.numel():
            loops[point_piece[chain[0]]] += 1
    bad_piece = torch.zeros(pieces, dtype=torch.bool, device=faces.device)
    bad_piece[edge_piece[edges.non_manifold]] = True
    bad_piece[point_piece[torch.nonzero(non_manifold_points).squeeze(1)].clamp_min(0)] = True
    open_edges = torch.zeros(pieces, dtype=torch.int64, device=faces.device).index_add_(
        0, edge_piece, edges.boundary.long())
    euler = v_count - e_count + f_count
    genus = (2 - euler - loops)
    good = ~bad_piece & (genus >= 0) & (genus % 2 == 0)
    return {
        "welded_vertices": int(used.numel()),
        "edges": edges.edge_count,
        "boundary_edges": int(edges.boundary.sum()),
        "non_manifold_edges": int(edges.non_manifold.sum()),
        "non_manifold_vertices": int(non_manifold_points.sum()),
        "inconsistent_edges": inconsistent,
        "border_loops": int(closed.sum()),
        "open_border_runs": int((~closed).sum()),
        "shells": pieces,
        "closed_shells": int(((open_edges == 0) & ~bad_piece).sum()),
        "genus": int((genus[good] // 2).sum()),
        "euler": int(euler.sum()),
        "positions": positions,
        "welded": welded,
    }


def _triangle_figures(positions: torch.Tensor, welded: torch.Tensor) -> dict:
    """Degenerate and duplicate counts, aspect ratios, smallest angles and valences of one part."""
    corners = positions[welded].double()
    repeated = (welded[:, 0] == welded[:, 1]) | (welded[:, 1] == welded[:, 2]) | (welded[:, 2] == welded[:, 0])
    side = torch.stack([(corners[:, (k + 1) % 3] - corners[:, k]).norm(dim=1) for k in range(3)], dim=1)
    area = 0.5 * torch.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], dim=1).norm(dim=1)
    longest = side.amax(dim=1)
    flat = repeated | (2.0 * area <= 1e-10 * longest * longest)
    aspect = math.sqrt(3.0) * longest * longest / (4.0 * area.clamp_min(1e-300))
    aspect = aspect[~flat]
    angles = torch.rad2deg(topology.corner_angles(positions.double(), welded)).amin(dim=1)[~flat]
    key = welded[~repeated].sort(dim=1).values
    duplicates = int(key.shape[0] - torch.unique(key, dim=0).shape[0]) if key.numel() else 0
    edges = topology.edge_topology(welded[~repeated], positions.shape[0])
    valence = torch.bincount(edges.edges.reshape(-1), minlength=positions.shape[0])
    valence = valence[valence > 0]
    histogram = {"2 or less": int((valence <= 2).sum())}
    for value in VALENCES:
        histogram[str(value)] = int((valence == value).sum())
    histogram[f"{VALENCES[-1] + 1} or more"] = int((valence > VALENCES[-1]).sum())
    bins, lower = [], 0.0
    for upper in ANGLE_BINS:
        bins.append(int(((angles >= lower) & (angles < upper)).sum()) if upper < ANGLE_BINS[-1]
                    else int((angles >= lower).sum()))
        lower = upper
    return {
        "degenerate_faces": int(flat.sum()),
        "duplicate_faces": duplicates,
        "aspect_mean": float(aspect.mean()) if aspect.numel() else 0.0,
        "aspect_p99": float(torch.quantile(aspect[:1_000_000].float(), 0.99)) if aspect.numel() else 0.0,
        "aspect_max": float(aspect.max()) if aspect.numel() else 0.0,
        "poor_triangles": int((aspect > POOR_ASPECT).sum()),
        "min_angle_mean": float(angles.mean()) if angles.numel() else 0.0,
        "min_angle_min": float(angles.min()) if angles.numel() else 0.0,
        "min_angle_histogram": bins,
        "valence_histogram": histogram,
        "valence_mean": float(valence.double().mean()) if valence.numel() else 0.0,
        "area": float(area.sum()),
        "face_area": area,
    }


def inspect(model: Model3D, parts: list[int] | None = None, intersections: bool = True,
            uv_resolution: int = 1024, device: torch.device | str | None = None) -> dict:
    """Every figure a modeller checks before export, for the chosen parts.

    Args:
        model: The model to measure.
        parts: Part indices to measure; None measures every part.
        intersections: Look for faces crossing other faces of their part.
        uv_resolution: Pixels per side of the texture coordinate raster.
        device: Device for the self-intersection tree; None means the mesh compute device.

    Returns:
        Named figures: counts as ints, measures as floats, histograms as lists and dicts.
    """
    chosen = list(range(len(model.parts))) if parts is None else list(parts)
    figures: dict = {
        "parts": len(chosen), "vertices": 0, "welded_vertices": 0, "triangles": 0, "polygons": 0,
        "quads": 0, "edges": 0, "boundary_edges": 0, "non_manifold_edges": 0,
        "non_manifold_vertices": 0, "inconsistent_edges": 0, "border_loops": 0, "open_border_runs": 0,
        "shells": 0, "closed_shells": 0, "genus": 0, "euler": 0, "degenerate_faces": 0,
        "duplicate_faces": 0, "intersecting_faces": -1 if not intersections else 0, "poor_triangles": 0,
        "flipped_uv_faces": 0, "zero_uv_faces": 0, "parts_without_uvs": 0, "uvs_outside": 0,
        "unweighted_vertices": 0, "skinned_parts": 0, "unskinned_parts": 0, "max_influences": 0,
    }
    aspects, angle_means, angle_mins, aspect_p99 = [], [], [], []
    angle_bins = [0] * len(ANGLE_BINS)
    valence_bins: dict[str, int] = {}
    valence_total = valence_weight = 0.0
    density_values, density_weights = [], []
    uv_area_px = surface_area = 0.0
    coverage_groups: dict[int, list[int]] = {}
    influence_bins = [0] * 9
    for index in chosen:
        part = model.parts[index]
        figures["vertices"] += part.vertex_count
        figures["triangles"] += part.triangle_count
        if part.polygons is not None:
            figures["polygons"] += part.polygons.count
            figures["quads"] += int((part.polygons.sizes == 4).sum())
        else:
            figures["polygons"] += part.triangle_count
        if part.triangle_count == 0:
            continue
        faces = part.faces.long()
        found = _topology_figures(part.vertices.float(), faces)
        positions, welded = found.pop("positions"), found.pop("welded")
        for name, value in found.items():
            figures[name] += value
        triangles = _triangle_figures(positions, welded)
        figures["degenerate_faces"] += triangles["degenerate_faces"]
        figures["duplicate_faces"] += triangles["duplicate_faces"]
        figures["poor_triangles"] += triangles["poor_triangles"]
        aspects.append((triangles["aspect_mean"], triangles["aspect_max"], part.triangle_count))
        aspect_p99.append(triangles["aspect_p99"])
        angle_means.append((triangles["min_angle_mean"], part.triangle_count))
        angle_mins.append(triangles["min_angle_min"])
        angle_bins = [a + b for a, b in zip(angle_bins, triangles["min_angle_histogram"])]
        for name, value in triangles["valence_histogram"].items():
            valence_bins[name] = valence_bins.get(name, 0) + value
        weight = sum(triangles["valence_histogram"].values())
        valence_total += triangles["valence_mean"] * weight
        valence_weight += weight
        if intersections:
            figures["intersecting_faces"] += int(self_intersections(part.vertices, faces, device=device).sum())
        face_area = triangles["face_area"]
        surface_area += triangles["area"]
        if part.uvs is None:
            figures["parts_without_uvs"] += 1
        else:
            uv = part.uvs.double()
            figures["uvs_outside"] += int(((uv < -1e-3) | (uv > 1.0 + 1e-3)).any(dim=1).sum())
            corner = uv[faces]
            signed = 0.5 * ((corner[:, 1, 0] - corner[:, 0, 0]) * (corner[:, 2, 1] - corner[:, 0, 1])
                            - (corner[:, 2, 0] - corner[:, 0, 0]) * (corner[:, 1, 1] - corner[:, 0, 1]))
            zero = signed.abs() <= 1e-14
            figures["zero_uv_faces"] += int(zero.sum())
            majority = float((torch.sign(signed) * face_area).sum())
            flipped = (signed < 0) if majority >= 0 else (signed > 0)
            figures["flipped_uv_faces"] += int((flipped & ~zero).sum())
            texture = part.material.base_color_texture
            if texture is not None:
                width, height = texture.size
                pixels = signed.abs() * width * height
                usable = face_area > 0
                density_values.append(torch.sqrt(pixels[usable] / face_area[usable]))
                density_weights.append(face_area[usable])
                uv_area_px += float(pixels.sum())
                coverage_groups.setdefault(id(texture), []).append(index)
            else:
                coverage_groups.setdefault(-1 - index, []).append(index)
        if part.weights is not None and part.joints is not None:
            figures["skinned_parts"] += 1
            filled = (part.weights > EMPTY_WEIGHT).sum(dim=1)
            figures["max_influences"] = max(figures["max_influences"], int(filled.max()) if filled.numel() else 0)
            figures["unweighted_vertices"] += int((part.weights.sum(dim=1) <= EMPTY_WEIGHT).sum())
            for count in range(9):
                influence_bins[count] += int((filled == count).sum()) if count < 8 else int((filled >= 8).sum())
        elif model.skeleton is not None:
            figures["unskinned_parts"] += 1
    measured = sum(weight for *_rest, weight in aspects)
    figures["aspect_mean"] = sum(mean * weight for mean, _top, weight in aspects) / measured if measured else 0.0
    figures["aspect_max"] = max((top for _mean, top, _weight in aspects), default=0.0)
    figures["aspect_p99"] = max(aspect_p99, default=0.0)
    figures["min_angle_mean"] = sum(mean * weight for mean, weight in angle_means) / measured if measured else 0.0
    figures["min_angle_min"] = min(angle_mins, default=0.0)
    figures["min_angle_histogram"] = angle_bins
    figures["valence_histogram"] = valence_bins
    figures["valence_mean"] = valence_total / valence_weight if valence_weight else 0.0
    figures["quad_fraction"] = figures["quads"] / figures["polygons"] if figures["polygons"] else 0.0
    figures["watertight"] = bool(figures["triangles"] > 0 and figures["boundary_edges"] == 0
                                 and figures["non_manifold_edges"] == 0 and figures["non_manifold_vertices"] == 0)
    covered = overlapped = cells = 0
    resolution_used = 0
    for members in coverage_groups.values():
        uvs = torch.cat([model.parts[i].uvs.float() for i in members])
        offsets, faces = 0, []
        for i in members:
            faces.append(model.parts[i].faces.long() + offsets)
            offsets += model.parts[i].vertex_count
        counts, resolution_used = uv_coverage(uvs, torch.cat(faces), uv_resolution)
        covered += int((counts > 0).sum())
        overlapped += int((counts > 1).sum())
        cells += counts.numel()
    figures["uv_groups"] = len(coverage_groups)
    figures["uv_coverage"] = covered / cells if cells else 0.0
    figures["uv_overlap"] = overlapped / covered if covered else 0.0
    figures["uv_resolution"] = resolution_used
    if density_values:
        values, weights = torch.cat(density_values), torch.cat(density_weights)
        figures["texel_density"] = math.sqrt(uv_area_px / surface_area) if surface_area > 0 else 0.0
        figures["texel_p10"], figures["texel_p90"] = _weighted_quantiles(values, weights, (0.1, 0.9))
    else:
        figures["texel_density"] = figures["texel_p10"] = figures["texel_p90"] = 0.0
    figures["influence_histogram"] = influence_bins
    low = high = torch.zeros(3)
    filled_parts = [model.parts[i] for i in chosen if model.parts[i].vertex_count]
    if filled_parts:
        low = torch.stack([part.vertices.amin(dim=0) for part in filled_parts]).amin(dim=0)
        high = torch.stack([part.vertices.amax(dim=0) for part in filled_parts]).amax(dim=0)
    figures["low"] = [float(v) for v in low]
    figures["high"] = [float(v) for v in high]
    figures["size"] = [float(h - l) for l, h in zip(figures["low"], figures["high"])]
    figures["height"] = figures["size"][1]
    figures["surface_area"] = surface_area
    figures["joints"] = model.skeleton.count if model.skeleton is not None else 0
    figures["morphs"] = sum(len(model.parts[i].morphs) for i in chosen)
    figures["expressions"] = len(model.expressions)
    figures["animations"] = len(model.animations)
    figures["spring_chains"] = len(model.springs.chains) if model.springs is not None else 0
    return figures


def _n(value: int) -> str:
    """An integer with thousands separators."""
    return f"{int(value):,}"


def report_text(figures: dict, names: list[str] | None = None) -> str:
    """The figures of `inspect` as a readable report.

    Args:
        figures: What `inspect` returned.
        names: Names of the parts measured, listed at the top.

    Returns:
        Several lines of text.
    """
    f = figures
    lines = []
    shown = ", ".join(names[:6]) + (f" and {len(names) - 6} more" if names and len(names) > 6 else "") if names else ""
    lines.append(f"Model: {_n(f['parts'])} part{'s' if f['parts'] != 1 else ''}" + (f" ({shown})" if shown else ""))
    lines.append(f"  {_n(f['vertices'])} vertices ({_n(f['welded_vertices'])} after welding seams), "
                 f"{_n(f['triangles'])} triangles, {_n(f['polygons'])} polygons, "
                 f"{f['quad_fraction'] * 100:.1f}% quads")
    sx, sy, sz = f["size"]
    lines.append(f"  Size {sx:.4g} x {sy:.4g} x {sz:.4g}, height {f['height']:.4g}, "
                 f"from ({', '.join(f'{v:.4g}' for v in f['low'])}) to ({', '.join(f'{v:.4g}' for v in f['high'])})")
    lines.append(f"  Surface area {f['surface_area']:.4g}")
    lines.append("Topology:")
    lines.append(f"  Watertight: {'yes' if f['watertight'] else 'no'}")
    lines.append(f"  {_n(f['edges'])} edges: {_n(f['boundary_edges'])} on borders, "
                 f"{_n(f['non_manifold_edges'])} non-manifold, {_n(f['inconsistent_edges'])} with faces wound against each other")
    lines.append(f"  {_n(f['non_manifold_vertices'])} non-manifold vertices")
    lines.append(f"  {_n(f['border_loops'])} border loops, {_n(f['open_border_runs'])} broken border runs")
    lines.append(f"  {_n(f['shells'])} shells, {_n(f['closed_shells'])} closed; genus {_n(f['genus'])}, "
                 f"Euler characteristic {f['euler']}")
    lines.append(f"  {_n(f['degenerate_faces'])} degenerate faces, {_n(f['duplicate_faces'])} duplicate faces")
    if f["intersecting_faces"] >= 0:
        lines.append(f"  {_n(f['intersecting_faces'])} faces crossing another face of their part")
    else:
        lines.append("  Self-intersections not checked")
    lines.append("Triangles:")
    lines.append(f"  Aspect ratio mean {f['aspect_mean']:.2f}, 99th percentile {f['aspect_p99']:.2f}, "
                 f"worst {f['aspect_max']:.3g}; {_n(f['poor_triangles'])} above {POOR_ASPECT:g}")
    lines.append(f"  Smallest angle mean {f['min_angle_mean']:.1f} deg, lowest {f['min_angle_min']:.2f} deg")
    edges = [0, *ANGLE_BINS]
    lines.append("  Smallest angle: " + ", ".join(
        f"{edges[i]}-{edges[i + 1]} deg {_n(c)}" for i, c in enumerate(f["min_angle_histogram"])))
    lines.append("  Valence: " + ", ".join(f"{name}: {_n(count)}" for name, count in f["valence_histogram"].items())
                 + f"; mean {f['valence_mean']:.2f}")
    lines.append("Texture coordinates:")
    if f["parts_without_uvs"] == f["parts"]:
        lines.append("  None")
    else:
        if f["parts_without_uvs"]:
            lines.append(f"  {_n(f['parts_without_uvs'])} parts have none")
        lines.append(f"  Coverage {f['uv_coverage'] * 100:.1f}% of the texture, overlap {f['uv_overlap'] * 100:.1f}% "
                     f"of the covered area ({f['uv_resolution']} px raster)")
        lines.append(f"  {_n(f['flipped_uv_faces'])} flipped faces, {_n(f['zero_uv_faces'])} faces with no texture area, "
                     f"{_n(f['uvs_outside'])} vertices outside 0 to 1")
        if f["texel_density"]:
            lines.append(f"  Texel density {f['texel_density']:.0f} px per unit, 10% to 90% of the surface between "
                         f"{f['texel_p10']:.0f} and {f['texel_p90']:.0f}")
    lines.append("Rig:")
    if f["joints"]:
        lines.append(f"  {_n(f['joints'])} joints, {_n(f['skinned_parts'])} skinned parts, "
                     f"{_n(f['unskinned_parts'])} parts without skin")
        lines.append(f"  {_n(f['unweighted_vertices'])} unweighted vertices; up to {f['max_influences']} influences per vertex")
        lines.append("  Influences per vertex: " + ", ".join(
            f"{count if count < 8 else '8+'}: {_n(value)}" for count, value in enumerate(f["influence_histogram"]) if value))
    else:
        lines.append("  No skeleton")
    lines.append(f"  {_n(f['morphs'])} shape keys, {_n(f['expressions'])} expressions, "
                 f"{_n(f['animations'])} animations, {_n(f['spring_chains'])} spring chains")
    return "\n".join(lines)

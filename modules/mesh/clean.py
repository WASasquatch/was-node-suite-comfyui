"""Mesh repair: welding, removing junk faces and shells, splitting non-manifold parts, orienting, filling holes.

Each step takes and returns a `Surface`. Edges, holes and pieces are found on topology ids shared by
vertices at one exact position.
"""

from __future__ import annotations

__all__ = [
    "COUNT_NAMES", "CleanSettings", "Surface", "append_rows", "blend", "blend_skin", "borders_left",
    "carry_polygons", "clean", "clean_part", "compact", "describe", "fill_holes", "orient", "part_of",
    "position_ids", "recompute_normals", "relax_patch", "remove_degenerate", "remove_faces", "remove_flaps",
    "remove_intersecting_shells", "remove_small_shells", "split_non_manifold", "surface_of", "take",
    "watertight", "weld",
]

import math
from dataclasses import dataclass, field, replace

import torch

from ..model3d.scene import Part, Polygons
from . import arrays, topology

#: Prefix of an attribute holding a shape key's normal offsets.
MORPH_NORMAL = "morph_normal:"

#: Largest texture coordinate difference two welded vertices may have.
UV_TOLERANCE = 1e-5

#: Largest vertex colour difference two welded vertices may have.
COLOR_TOLERANCE = 1.0 / 512.0

#: Largest normal component difference two welded vertices may have.
NORMAL_TOLERANCE = 0.01

#: Largest skin weight difference two welded vertices may have.
WEIGHT_TOLERANCE = 1e-3

#: Rounds of the leader pass that groups coincident vertices by their attributes.
WELD_ROUNDS = 64

#: Twice a face's area at or below this share of its longest edge squared marks it degenerate.
DEGENERATE_RATIO = 1e-10

#: Distance split vertices move apart, as a share of their mean edge length.
SPLIT_NUDGE = 1e-5

#: Least distance split vertices move apart, in float32 steps of their coordinates.
SPLIT_STEPS = 4.0

#: Longest hole the filler accepts, in boundary edges.
MAX_HOLE_EDGES = 1000

#: Vertices hole filling may add across one surface.
MAX_FILL_VERTICES = 2_000_000

#: Density factor of the refinement; a triangle splits while its centre is this much farther than the spacing.
DENSITY = math.sqrt(2.0)

#: Split rounds of the hole refinement.
REFINE_ROUNDS = 16

#: Edge flip rounds after each refinement split.
RELAX_ROUNDS = 24

#: Conjugate gradient iterations of the fairing solve.
FAIR_ITERATIONS = 3000

#: Edge weights of the fairing Laplacian, ``"cotangent"`` or ``"uniform"``.
FAIR_WEIGHTS = "cotangent"

#: Cost of a hole triangle edge the mesh already has, in squared border spacings.
CLASH_PENALTY = 1e6

#: Polygon corners squared triangulated together.
HOLE_BATCH = 4_000_000

#: A patch with less area than this share of its border length squared closes a slit and is skipped.
SLIT_RATIO = 1e-8

#: A patch with more than this share of its piece's area would wrap an open sheet and is skipped.
SHEET_SHARE = 0.5

#: Faces sampled per open piece when voting which way it faces.
VOTE_FACES = 2048

#: Boundary share of a piece's edges below which its signed volume decides its facing.
NEARLY_CLOSED = 0.01


@dataclass
class Surface:
    """A part's triangles while they are repaired, with the original face each one came from.

    Attributes:
        vertices: ``(V, 3)`` float32 positions.
        faces: ``(F, 3)`` int64 triangles, counter-clockwise seen from the front.
        attributes: Per-vertex tensors by name: those of `arrays.TriMesh`, plus ``"joints"``,
            ``"weights"`` and ``"morph_normal:<name>"``.
        source: ``(F,)`` int64 index of the original face each face copies, -1 for a new face.
        flipped: ``(F,)`` bool, True where the face runs opposite to its original.
        copy: ``(F,)`` int64 copy number, 1 for a mirrored copy of an original face.
        topo: ``(V,)`` int64 topology id per vertex, or None until computed.
        positions: ``(T, 3)`` float32 position of each topology id, or None.
    """

    vertices: torch.Tensor
    faces: torch.Tensor
    attributes: dict[str, torch.Tensor] = field(default_factory=dict)
    source: torch.Tensor | None = None
    flipped: torch.Tensor | None = None
    copy: torch.Tensor | None = None
    topo: torch.Tensor | None = None
    positions: torch.Tensor | None = None

    def __post_init__(self):
        count = int(self.faces.shape[0])
        device = self.faces.device
        if self.source is None:
            self.source = torch.arange(count, dtype=torch.int64, device=device)
        if self.flipped is None:
            self.flipped = torch.zeros(count, dtype=torch.bool, device=device)
        if self.copy is None:
            self.copy = torch.zeros(count, dtype=torch.int64, device=device)

    @property
    def vertex_count(self) -> int:
        """Number of vertices."""
        return int(self.vertices.shape[0])

    @property
    def face_count(self) -> int:
        """Number of triangles."""
        return int(self.faces.shape[0])

    @property
    def device(self) -> torch.device:
        """Device the tensors live on."""
        return self.vertices.device

    def topology(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Topology id of each vertex and the position of each id, computed once.

        Returns:
            ``(ids, positions)``: ``(V,)`` int64 and ``(T, 3)`` float32.
        """
        if self.topo is None or self.positions is None:
            self.topo, self.positions = position_ids(self.vertices)
        return self.topo, self.positions

    def welded_faces(self) -> torch.Tensor:
        """``(F, 3)`` faces on topology ids."""
        ids, _ = self.topology()
        return ids[self.faces]

    def to(self, device: torch.device | str) -> Surface:
        """The same surface with every tensor on ``device``."""
        def move(value):
            return value.to(device) if value is not None else None
        return Surface(
            self.vertices.to(device), self.faces.to(device),
            {name: value.to(device) for name, value in self.attributes.items()},
            move(self.source), move(self.flipped), move(self.copy), move(self.topo), move(self.positions),
        )


@dataclass
class CleanSettings:
    """Which repairs `clean` runs and their limits.

    Attributes:
        weld: Merge vertices closer than ``weld_distance`` whose attributes agree.
        weld_distance: Merge distance in model units; 0 merges exact duplicates only.
        degenerate: Remove collapsed, zero-area and duplicate faces.
        unreferenced: Remove vertices no face uses.
        shells: Remove small disconnected pieces.
        shell_min_faces: Pieces, and flaps on non-manifold edges, with fewer faces are removed.
        shell_min_area: Pieces and flaps below this share of the part's area are removed.
        non_manifold: Remove small flaps hanging off non-manifold edges, then split non-manifold edges and vertices.
        orient: Wind faces consistently and outward.
        fill: Fill holes.
        max_hole_edges: Longest hole filled, in boundary edges.
        intersections: ``"off"``, ``"count"`` or ``"remove small shells"``.
        intersecting_max_area: Intersecting pieces below this share of the part's area are removed.
        normals: Recompute normals.
        crease_angle: Dihedral angle in degrees at and above which an edge stays sharp.
        compare_normals: Keep coincident vertices apart when their normals differ.
    """

    weld: bool = True
    weld_distance: float = 1e-5
    degenerate: bool = True
    unreferenced: bool = True
    shells: bool = True
    shell_min_faces: int = 8
    shell_min_area: float = 1e-4
    non_manifold: bool = True
    orient: bool = True
    fill: bool = True
    max_hole_edges: int = 100
    intersections: str = "off"
    intersecting_max_area: float = 0.05
    normals: bool = False
    crease_angle: float = 60.0
    compare_normals: bool = True


# ---------------------------------------------------------------------------------------------
# Conversions


def position_ids(vertices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Group vertices at exactly the same position.

    Args:
        vertices: ``(V, 3)`` positions.

    Returns:
        ``(ids, positions)``: ``(V,)`` int64 group of each vertex and ``(T, 3)`` group positions.
    """
    if vertices.shape[0] == 0:
        return torch.zeros(0, dtype=torch.int64, device=vertices.device), vertices.clone()
    positions, ids = torch.unique(vertices, dim=0, return_inverse=True)
    return ids, positions


def surface_of(part: Part) -> Surface:
    """A part's triangles and every per-vertex attribute as a `Surface`.

    Args:
        part: The part to read.

    Returns:
        The surface on the CPU.

    Raises:
        ValueError: The part's faces index past its vertices.
    """
    faces = part.faces.long()
    count = part.vertex_count
    if faces.numel() and (int(faces.min()) < 0 or int(faces.max()) >= count):
        raise ValueError(f"part '{part.name}' has faces that point past its vertices; the file is damaged. "
                         "Load it again or export it again from the program that made it.")
    mesh = arrays.from_part(part, 0)
    attributes = {name: value for name, value in mesh.attributes.items()}
    if part.joints is not None and part.weights is not None:
        attributes["joints"] = part.joints.long()
        attributes["weights"] = part.weights.float()
    for name, morph in part.morphs.items():
        if morph.normals is not None:
            attributes[MORPH_NORMAL + name] = morph.normals.float()
    return Surface(part.vertices.float(), faces.reshape(-1, 3), attributes)


def part_of(part: Part, surface: Surface) -> Part:
    """A copy of ``part`` carrying the surface's geometry, attributes and the polygons that survived.

    Args:
        part: The part the surface was read from.
        surface: The repaired surface.

    Returns:
        A new part on the CPU.
    """
    plain = {name: value for name, value in surface.attributes.items()
             if name not in ("joints", "weights") and not name.startswith(MORPH_NORMAL)}
    mesh = arrays.TriMesh(surface.vertices, surface.faces, plain, None)
    result = arrays.to_part(part, mesh)
    if "joints" in surface.attributes and "weights" in surface.attributes:
        result.joints = surface.attributes["joints"].long().cpu()
        result.weights = surface.attributes["weights"].float().cpu()
    for name, morph in result.morphs.items():
        normals = surface.attributes.get(MORPH_NORMAL + name)
        if normals is not None:
            morph.normals = normals.float().cpu()
    result.polygons = carry_polygons(part, surface)
    result.extras = dict(part.extras)
    return result


# ---------------------------------------------------------------------------------------------
# Attributes


def take(attributes: dict[str, torch.Tensor], index: torch.Tensor) -> dict[str, torch.Tensor]:
    """Rows of every attribute.

    Args:
        attributes: Per-vertex tensors by name.
        index: ``(N,)`` int64 rows, or a bool mask.

    Returns:
        The selected rows by name.
    """
    return {name: value[index] for name, value in attributes.items()}


def blend_skin(joints: torch.Tensor, weights: torch.Tensor, sources: torch.Tensor,
               coefficients: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Skin influences mixed from several rows, the strongest kept and renormalised.

    Args:
        joints: ``(V, K)`` int64 joint indices.
        weights: ``(V, K)`` weights.
        sources: ``(N, S)`` int64 rows mixed into each new row.
        coefficients: ``(N, S)`` weights of those rows.

    Returns:
        ``(joints, weights)``, each ``(N, K)``, weights summing to 1 where any influence remains.
    """
    count, width = sources.shape
    keep = joints.shape[1]
    joint = joints[sources].reshape(count, width * keep)
    weight = (weights[sources].float() * coefficients.float()[:, :, None]).reshape(count, width * keep)
    weight = torch.where(weight > 0, weight, torch.zeros_like(weight))
    order = joint.argsort(dim=1)
    joint = joint.gather(1, order)
    weight = weight.gather(1, order)
    start = torch.ones_like(joint, dtype=torch.bool)
    start[:, 1:] = joint[:, 1:] != joint[:, :-1]
    segment = start.long().cumsum(1) - 1
    summed = torch.zeros_like(weight).scatter_add_(1, segment, weight)
    named = torch.zeros_like(joint).scatter_(1, segment, joint)
    top, pick = summed.topk(min(keep, summed.shape[1]), dim=1)
    chosen = named.gather(1, pick)
    total = top.sum(dim=1, keepdim=True)
    top = torch.where(total > 0, top / total.clamp_min(1e-12), top)
    chosen = torch.where(top > 0, chosen, torch.zeros_like(chosen))
    return chosen, top


def blend(attributes: dict[str, torch.Tensor], sources: torch.Tensor,
          coefficients: torch.Tensor) -> dict[str, torch.Tensor]:
    """New attribute rows mixed from existing rows, each made valid for its kind.

    Args:
        attributes: Per-vertex tensors by name.
        sources: ``(N, S)`` int64 rows mixed into each new row.
        coefficients: ``(N, S)`` weights of those rows, summing to 1.

    Returns:
        ``(N, K)`` tensors by name: normals unit length, skin renormalised, colours clamped.
    """
    result: dict[str, torch.Tensor] = {}
    weights_of = coefficients.float()
    for name, value in attributes.items():
        if name == "joints":
            continue
        if name == "weights" and "joints" in attributes:
            result["joints"], result["weights"] = blend_skin(attributes["joints"], value, sources, weights_of)
            continue
        corner = value[sources].float()
        mixed = (corner * weights_of[:, :, None]).sum(dim=1)
        if name == "normals":
            length = mixed.norm(dim=1, keepdim=True)
            mixed = torch.where(length > 1e-12, mixed / length.clamp_min(1e-12), corner[:, 0])
        elif name == "tangents":
            direction = torch.nn.functional.normalize(mixed[:, :3], dim=1)
            strongest = weights_of.argmax(dim=1)
            sign = corner[torch.arange(corner.shape[0], device=corner.device), strongest, 3:4]
            mixed = torch.cat([direction, torch.where(sign < 0, -1.0, 1.0)], dim=1)
        elif name == "colors":
            mixed = mixed.clamp(0.0, 1.0)
        result[name] = mixed.to(value.dtype)
    return result


def _skin_key(attributes: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Skin influences sorted by joint with empty slots cleared, for comparing two vertices."""
    joints, weights = attributes.get("joints"), attributes.get("weights")
    if joints is None or weights is None:
        return None
    joints = torch.where(weights > 0, joints.long(), torch.full_like(joints.long(), -1))
    order = joints.argsort(dim=1)
    return joints.gather(1, order).float(), weights.float().gather(1, order)


def append_rows(attributes: dict[str, torch.Tensor], extra: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Attribute rows with more rows added below.

    Args:
        attributes: Per-vertex tensors by name.
        extra: Rows to add, by the same names.

    Returns:
        The joined tensors by name.
    """
    return {name: torch.cat([value, extra[name].to(value.dtype)]) for name, value in attributes.items()}


# ---------------------------------------------------------------------------------------------
# Face and vertex bookkeeping


def remove_faces(surface: Surface, keep: torch.Tensor) -> Surface:
    """The surface without the faces ``keep`` leaves out; vertices stay.

    Args:
        surface: The surface.
        keep: ``(F,)`` bool, True for faces kept.

    Returns:
        A new surface sharing the vertex tensors.
    """
    return replace(surface, faces=surface.faces[keep], source=surface.source[keep],
                   flipped=surface.flipped[keep], copy=surface.copy[keep])


def compact(surface: Surface) -> tuple[Surface, int]:
    """Remove vertices no face uses.

    Args:
        surface: The surface.

    Returns:
        ``(surface, removed)``.
    """
    count = surface.vertex_count
    used = torch.zeros(count, dtype=torch.bool, device=surface.device)
    used[surface.faces.reshape(-1)] = True
    removed = count - int(used.sum())
    if removed == 0:
        return surface, 0
    remap = torch.cumsum(used.long(), 0) - 1
    return replace(
        surface, vertices=surface.vertices[used], faces=remap[surface.faces],
        attributes=take(surface.attributes, used),
        topo=surface.topo[used] if surface.topo is not None else None,
    ), removed


def _remap_vertices(surface: Surface, faces: torch.Tensor, rows: torch.Tensor,
                    vertices: torch.Tensor | None = None) -> Surface:
    """A surface whose vertices are ``rows`` of the old ones, on new faces."""
    return replace(
        surface, vertices=surface.vertices[rows] if vertices is None else vertices,
        faces=faces, attributes=take(surface.attributes, rows), topo=None, positions=None,
    )


# ---------------------------------------------------------------------------------------------
# Weld


def weld(surface: Surface, distance: float, compare_normals: bool = True) -> tuple[Surface, int]:
    """Merge vertices closer than a distance whose attributes agree; differing seams stay split at one position.

    Args:
        surface: The surface.
        distance: Merge distance per axis in model units; 0 merges exact duplicates.
        compare_normals: Keep vertices apart when their normals differ.

    Returns:
        ``(surface, welded)``: the welded surface and how many vertices were merged away or moved onto another.
    """
    count = surface.vertex_count
    if count < 2:
        return surface, 0
    device = surface.device
    welded, _, group = topology.weld(surface.vertices, surface.faces, tolerance=max(0.0, float(distance)))
    attributes = surface.attributes
    features: list[tuple[torch.Tensor, float]] = []
    if "uvs" in attributes:
        features.append((attributes["uvs"].float(), UV_TOLERANCE))
    if "colors" in attributes:
        features.append((attributes["colors"].float(), COLOR_TOLERANCE))
    if compare_normals and "normals" in attributes:
        features.append((attributes["normals"].float(), NORMAL_TOLERANCE))
    skin = _skin_key(attributes)
    if skin is not None:
        features.append((skin[0], 0.5))
        features.append((skin[1], WEIGHT_TOLERANCE))
    reach = max(float(distance), 1e-6)
    for name, value in attributes.items():
        if name.startswith(arrays.MORPH_PREFIX):
            features.append((value.float(), reach))
    leader = torch.full((count,), -1, dtype=torch.int64, device=device)
    index = torch.arange(count, dtype=torch.int64, device=device)
    groups = int(group.max()) + 1
    for _ in range(WELD_ROUNDS):
        open_rows = torch.nonzero(leader < 0).squeeze(1)
        if not open_rows.numel():
            break
        first = torch.full((groups,), count, dtype=torch.int64, device=device)
        first.scatter_reduce_(0, group[open_rows], open_rows, reduce="amin")
        lead = first[group[open_rows]]
        match = torch.ones(open_rows.shape[0], dtype=torch.bool, device=device)
        for value, tolerance in features:
            match &= (value[open_rows] - value[lead]).abs().amax(dim=1) <= tolerance
        leader[open_rows[match]] = lead[match]
    leader = torch.where(leader < 0, index, leader)
    is_leader = leader == index
    moved = (welded[group] != surface.vertices).any(dim=1)
    changed = int((moved | ~is_leader).sum())
    if changed == 0:
        return surface, 0
    rows = torch.nonzero(is_leader).squeeze(1)
    renumber = torch.cumsum(is_leader.long(), 0) - 1
    faces = renumber[leader][surface.faces]
    return _remap_vertices(surface, faces, rows, welded[group[rows]]), changed


# ---------------------------------------------------------------------------------------------
# Degenerate, duplicate and small pieces


def _degenerate_mask(positions: torch.Tensor, welded: torch.Tensor) -> torch.Tensor:
    """Faces collapsed onto a repeated vertex, of zero area, or with a corner that is not finite."""
    repeated = (welded[:, 0] == welded[:, 1]) | (welded[:, 1] == welded[:, 2]) | (welded[:, 2] == welded[:, 0])
    corners = positions[welded].double()
    finite = torch.isfinite(corners).all(dim=2).all(dim=1)
    corners = torch.where(finite[:, None, None], corners, torch.zeros_like(corners))
    ab, ac, bc = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0], corners[:, 2] - corners[:, 1]
    twice_area = torch.cross(ab, ac, dim=1).norm(dim=1)
    longest = torch.stack([(ab * ab).sum(1), (ac * ac).sum(1), (bc * bc).sum(1)], dim=1).amax(dim=1)
    flat = twice_area <= DEGENERATE_RATIO * longest
    return repeated | flat | ~finite


def remove_degenerate(surface: Surface, duplicates: bool = True) -> tuple[Surface, int, int]:
    """Remove collapsed and zero-area faces, then faces repeating another face's corners.

    Args:
        surface: The surface.
        duplicates: Also remove duplicate faces, whatever their winding.

    Returns:
        ``(surface, degenerate, duplicate)`` face counts removed.
    """
    if surface.face_count == 0:
        return surface, 0, 0
    _, positions = surface.topology()
    welded = surface.welded_faces()
    bad = _degenerate_mask(positions, welded)
    keep = ~bad
    duplicate = 0
    if duplicates and bool(keep.any()):
        rows = torch.nonzero(keep).squeeze(1)
        key = welded[rows].sort(dim=1).values
        _, inverse = torch.unique(key, dim=0, return_inverse=True)
        first = torch.full((int(inverse.max()) + 1,), rows.shape[0], dtype=torch.int64, device=surface.device)
        first.scatter_reduce_(0, inverse, torch.arange(rows.shape[0], device=surface.device), reduce="amin")
        repeat = torch.ones(rows.shape[0], dtype=torch.bool, device=surface.device)
        repeat[first] = False
        keep[rows[repeat]] = False
        duplicate = int(repeat.sum())
    degenerate = int(bad.sum())
    if degenerate == 0 and duplicate == 0:
        return surface, 0, 0
    return remove_faces(surface, keep), degenerate, duplicate


def _piece_areas(surface: Surface) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Connected piece of each face on topology ids, with face counts and areas per piece."""
    ids, positions = surface.topology()
    welded = ids[surface.faces]
    piece = topology.face_components(welded, positions.shape[0])
    pieces = int(piece.max()) + 1 if piece.numel() else 0
    counts = torch.bincount(piece, minlength=pieces)
    area = topology.face_areas(positions.double(), welded)
    areas = torch.zeros(pieces, dtype=torch.float64, device=surface.device).index_add_(0, piece, area)
    return piece, counts, areas


def remove_small_shells(surface: Surface, min_faces: int, min_share: float) -> tuple[Surface, int, int]:
    """Remove connected pieces with few faces or a small share of the part's area; the largest always stays.

    Args:
        surface: The surface.
        min_faces: Pieces with fewer faces go; 0 turns the count test off.
        min_share: Pieces below this share of the total area go; 0 turns the area test off.

    Returns:
        ``(surface, pieces, faces)`` removed.
    """
    if surface.face_count == 0:
        return surface, 0, 0
    piece, counts, areas = _piece_areas(surface)
    total = float(areas.sum())
    small = torch.zeros_like(counts, dtype=torch.bool)
    if min_faces > 0:
        small |= counts < int(min_faces)
    if min_share > 0 and total > 0:
        small |= areas < float(min_share) * total
    small[int(areas.argmax())] = False
    if not bool(small.any()):
        return surface, 0, 0
    drop = small[piece]
    return remove_faces(surface, ~drop), int(small.sum()), int(drop.sum())


def remove_intersecting_shells(surface: Surface, crossing: torch.Tensor, max_share: float) -> tuple[Surface, int, int]:
    """Remove pieces holding a self-intersecting face and below a share of the part's area.

    Args:
        surface: The surface.
        crossing: ``(F,)`` bool, faces that cross another face.
        max_share: Pieces below this share of the total area may go.

    Returns:
        ``(surface, pieces, faces)`` removed.
    """
    if surface.face_count == 0 or not bool(crossing.any()):
        return surface, 0, 0
    piece, _counts, areas = _piece_areas(surface)
    total = float(areas.sum())
    involved = torch.zeros_like(areas, dtype=torch.bool)
    involved[piece[crossing]] = True
    small = involved & (areas < float(max_share) * total)
    small[int(areas.argmax())] = False
    if not bool(small.any()):
        return surface, 0, 0
    drop = small[piece]
    return remove_faces(surface, ~drop), int(small.sum()), int(drop.sum())


def remove_flaps(surface: Surface, max_faces: int, max_share: float) -> tuple[Surface, int, int]:
    """Remove small open patches hanging off a non-manifold edge, such as stray fins.

    A patch is faces joined through manifold edges; the largest stays.

    Args:
        surface: The surface.
        max_faces: Patches with fewer faces go; 0 turns the count test off.
        max_share: Patches below this share of the total area go; 0 turns the area test off.

    Returns:
        ``(surface, patches, faces)`` removed.
    """
    if surface.face_count == 0:
        return surface, 0, 0
    device = surface.device
    ids, positions = surface.topology()
    welded = ids[surface.faces]
    edges = topology.edge_topology(welded, positions.shape[0])
    if not bool(edges.non_manifold.any()):
        return surface, 0, 0
    _pair_ids, pairs = edges.manifold_face_pairs()
    root = topology.union_find(surface.face_count, pairs[:, 0], pairs[:, 1])
    _roots, patch = torch.unique(root, return_inverse=True)
    patches = int(patch.max()) + 1
    counts = torch.bincount(patch, minlength=patches)
    area = topology.face_areas(positions.double(), welded)
    areas = torch.zeros(patches, dtype=torch.float64, device=device).index_add_(0, patch, area)
    touches = torch.zeros(patches, dtype=torch.bool, device=device)
    touches[patch[edges.non_manifold[edges.face_edges].any(dim=1)]] = True
    open_patch = torch.zeros(patches, dtype=torch.bool, device=device)
    open_patch[patch[edges.boundary[edges.face_edges].any(dim=1)]] = True
    small = torch.zeros(patches, dtype=torch.bool, device=device)
    if max_faces > 0:
        small |= counts < int(max_faces)
    total = float(areas.sum())
    if max_share > 0 and total > 0:
        small |= areas < float(max_share) * total
    flap = touches & open_patch & small
    flap[int(areas.argmax())] = False
    if not bool(flap.any()):
        return surface, 0, 0
    drop = flap[patch]
    return remove_faces(surface, ~drop), int(flap.sum()), int(drop.sum())


# ---------------------------------------------------------------------------------------------
# Non-manifold splitting


def _pair_corners(welded: torch.Tensor, face_a: torch.Tensor, side_a: torch.Tensor,
                  face_b: torch.Tensor, side_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Corner links between faces paired across an edge, joining the corners on each shared vertex."""
    links_a, links_b = [], []
    for end in range(2):
        vertex = welded[face_a, (side_a + end) % 3]
        corners = []
        for face, side in ((face_a, side_a), (face_b, side_b)):
            at_start = welded[face, side] == vertex
            corners.append(face * 3 + torch.where(at_start, side, (side + 1) % 3))
        links_a.append(corners[0])
        links_b.append(corners[1])
    return torch.cat(links_a), torch.cat(links_b)


def _non_manifold_pairs(welded: torch.Tensor, positions: torch.Tensor,
                        edges: topology.EdgeTopology) -> tuple[torch.Tensor, ...]:
    """Faces paired around each edge with more than two faces, so each pair bounds one solid wedge.

    Returns:
        ``(face_a, side_a, face_b, side_b)`` int64 tensors, one row per pair.
    """
    device = welded.device
    empty = torch.zeros(0, dtype=torch.int64, device=device)
    on_bad = edges.non_manifold[edges.face_edges.reshape(-1)]
    half = torch.nonzero(on_bad).squeeze(1)
    if not half.numel():
        return empty, empty, empty, empty
    face, side = half // 3, half % 3
    edge = edges.face_edges.reshape(-1)[half]
    start = edges.edges[edge, 0]
    end = edges.edges[edge, 1]
    origin = positions[start].double()
    axis = torch.nn.functional.normalize(positions[end].double() - origin, dim=1)
    apex = positions[welded[face, (side + 2) % 3]].double() - origin
    across = apex - (apex * axis).sum(1, keepdim=True) * axis
    order = torch.argsort(edge, stable=True)
    face, side, edge, axis, across = face[order], side[order], edge[order], axis[order], across[order]
    count = int(edge.shape[0])
    head = torch.ones(count, dtype=torch.bool, device=device)
    head[1:] = edge[1:] != edge[:-1]
    first = torch.cummax(torch.where(head, torch.arange(count, device=device), torch.zeros_like(edge)), 0).values
    reference = torch.nn.functional.normalize(across[first], dim=1)
    other = torch.cross(axis, reference, dim=1)
    angle = torch.atan2((across * other).sum(1), (across * reference).sum(1))
    order = torch.argsort(edge.double() * 16.0 + (angle + math.pi) / (2 * math.pi) * 8.0, stable=True)
    face, side, edge, angle, axis, reference, other = (
        face[order], side[order], edge[order], angle[order], axis[order], reference[order], other[order])
    head = torch.ones(count, dtype=torch.bool, device=device)
    head[1:] = edge[1:] != edge[:-1]
    index = torch.arange(count, device=device)
    begin = torch.cummax(torch.where(head, index, torch.zeros_like(index)), 0).values
    rank = index - begin
    size = torch.bincount(edge, minlength=edges.edge_count)[edge]
    normal = topology.face_normals(positions.double(), welded[face])
    turning = torch.cos(angle)[:, None] * other - torch.sin(angle)[:, None] * reference
    sign = (normal * turning).sum(1)
    forward = welded[face, side] == start

    def pairing(shift: int):
        low = (rank - shift) % size
        lead = (low % 2 == 0) & ((low + 1) < size)
        lead = lead & ((rank + 1 < size) | (shift == 1))
        partner = begin + (rank + 1) % size
        ok = lead & (sign < 0) & (sign[partner] > 0) & (forward != forward[partner])
        return lead, partner, ok

    lead_a, partner_a, ok_a = pairing(0)
    lead_b, partner_b, ok_b = pairing(1)
    score_a = torch.zeros(edges.edge_count, dtype=torch.int64, device=device).index_add_(0, edge, ok_a.long())
    score_b = torch.zeros(edges.edge_count, dtype=torch.int64, device=device).index_add_(0, edge, ok_b.long())
    use_b = (score_b > score_a)[edge]
    ok = torch.where(use_b, ok_b, ok_a)
    partner = torch.where(use_b, partner_b, partner_a)
    rows = torch.nonzero(ok).squeeze(1)
    return face[rows], side[rows], face[partner[rows]], side[partner[rows]]


def split_non_manifold(surface: Surface) -> tuple[Surface, int, int]:
    """Give every fan of faces around a vertex its own vertex, the copies a tiny fraction of an edge apart.

    Args:
        surface: The surface.

    Returns:
        ``(surface, edges, vertices)``: non-manifold edges and vertices that were split.
    """
    if surface.face_count == 0:
        return surface, 0, 0
    device = surface.device
    ids, positions = surface.topology()
    welded = ids[surface.faces]
    point_count = positions.shape[0]
    edges = topology.edge_topology(welded, point_count)
    bad_edges = int(edges.non_manifold.sum())
    pair_ids, pairs = edges.manifold_face_pairs()
    sides = edges.edge_corners[pair_ids]
    face_a, side_a, face_b, side_b = _non_manifold_pairs(welded, positions, edges)
    link_a, link_b = _pair_corners(
        welded, torch.cat([pairs[:, 0], face_a]), torch.cat([sides[:, 0], side_a]),
        torch.cat([pairs[:, 1], face_b]), torch.cat([sides[:, 1], side_b]),
    )
    corner_count = surface.face_count * 3
    same_a, same_b = [], []
    for corner in range(3):
        nxt = (corner + 1) % 3
        same = torch.nonzero(welded[:, corner] == welded[:, nxt]).squeeze(1)
        same_a.append(same * 3 + corner)
        same_b.append(same * 3 + nxt)
    fan = topology.union_find(corner_count, torch.cat([link_a, *same_a]), torch.cat([link_b, *same_b]))
    point = welded.reshape(-1)
    fans_at_point = torch.unique(point * corner_count + fan) // corner_count
    split_points = torch.bincount(fans_at_point, minlength=point_count) > 1
    split_count = int(split_points.sum())
    if split_count == 0 and bad_edges == 0:
        return surface, 0, 0
    vertex = surface.faces.reshape(-1)
    keys, inverse = torch.unique(vertex * corner_count + fan, return_inverse=True)
    owner = keys // corner_count
    head = torch.ones(owner.shape[0], dtype=torch.bool, device=device)
    head[1:] = owner[1:] != owner[:-1]
    extra = ~head
    new_index = owner.clone()
    count = surface.vertex_count
    new_index[extra] = count + torch.arange(int(extra.sum()), dtype=torch.int64, device=device)
    rows = torch.cat([torch.arange(count, device=device), owner[extra]])
    faces = new_index[inverse].reshape(-1, 3)
    corners = surface.vertices[surface.faces].double()
    centroid = corners.mean(dim=1)
    spans = (corners - corners.roll(1, dims=1)).norm(dim=2)
    corner_span = 0.5 * (spans + spans.roll(-1, dims=1))
    fan_centre = torch.zeros(corner_count, 3, dtype=torch.float64, device=device)
    fan_centre.index_add_(0, fan, centroid[:, None, :].expand(-1, 3, -1).reshape(-1, 3))
    fan_span = torch.zeros(corner_count, dtype=torch.float64, device=device).index_add_(0, fan, corner_span.reshape(-1))
    members = torch.bincount(fan, minlength=corner_count).clamp_min(1).double()
    fan_centre /= members[:, None]
    fan_span /= members
    vertices = surface.vertices[rows].double()
    chosen = torch.nonzero(split_points[ids[owner]]).squeeze(1)
    if chosen.numel():
        target = new_index[chosen]
        fan_of_key = keys[chosen] % corner_count
        start = vertices[target]
        direction = torch.nn.functional.normalize(fan_centre[fan_of_key] - start, dim=1)
        floor = SPLIT_STEPS * torch.finfo(torch.float32).eps * start.abs().amax(dim=1).clamp_min(1e-30)
        reach = torch.maximum(SPLIT_NUDGE * fan_span[fan_of_key], floor)
        vertices[target] = start + direction * reach[:, None]
    result = _remap_vertices(surface, faces, rows, vertices.float())
    return result, bad_edges, split_count


# ---------------------------------------------------------------------------------------------
# Orientation


def orient(surface: Surface, outward: bool = True, device: torch.device | None = None) -> tuple[Surface, int, int]:
    """Wind every piece consistently and turn it outward, by signed volume when closed and cast rays when open.

    Args:
        surface: The surface.
        outward: Turn pieces to face outward; False only makes winding consistent.
        device: Device for ray casting; None means the surface's.

    Returns:
        ``(surface, faces, pieces)``: faces whose winding changed and pieces turned over.
    """
    if surface.face_count == 0:
        return surface, 0, 0
    ids, positions = surface.topology()
    welded = ids[surface.faces]
    point_count = positions.shape[0]
    oriented = topology.orient_faces(welded, point_count, None, outward="none")
    flip = (oriented[:, 1] != welded[:, 1]) | (oriented[:, 2] != welded[:, 2])
    turned_pieces = 0
    if outward:
        piece = topology.face_components(oriented, point_count)
        pieces = int(piece.max()) + 1
        corners = positions[oriented].double()
        centre_sum = torch.zeros(pieces, 3, dtype=torch.float64, device=surface.device)
        centre_sum.index_add_(0, piece, corners.mean(1))
        members = torch.bincount(piece, minlength=pieces).clamp_min(1).double()
        centre = (centre_sum / members[:, None])[piece]
        a, b, c = corners[:, 0] - centre, corners[:, 1] - centre, corners[:, 2] - centre
        volume = torch.zeros(pieces, dtype=torch.float64, device=surface.device)
        volume.index_add_(0, piece, (a * torch.cross(b, c, dim=1)).sum(1))
        edges = topology.edge_topology(oriented, point_count)
        open_edge = edges.boundary | edges.non_manifold
        edge_piece = piece[edges.edge_faces[:, 0].clamp_min(0)]
        open_count = torch.zeros(pieces, dtype=torch.int64, device=surface.device).index_add_(0, edge_piece, open_edge.long())
        edge_count = torch.bincount(edge_piece, minlength=pieces).clamp_min(1)
        open_share = open_count.double() / edge_count.double()
        turn = volume < 0
        vote = torch.nonzero(open_share >= NEARLY_CLOSED).squeeze(1)
        if vote.numel():
            turn[vote] = _ray_vote(positions, oriented, piece, vote, device or surface.device)
        turned_pieces = int(turn.sum())
        flip = flip ^ turn[piece]
    changed = int(flip.sum())
    if changed == 0:
        return surface, 0, 0
    faces = torch.where(flip[:, None], surface.faces[:, [0, 2, 1]], surface.faces)
    return replace(surface, faces=faces, flipped=surface.flipped ^ flip), changed, turned_pieces


def _ray_vote(positions: torch.Tensor, faces: torch.Tensor, piece: torch.Tensor, pieces: torch.Tensor,
              device: torch.device) -> torch.Tensor:
    """For open pieces, whether more rays escape behind their faces than in front of them."""
    from .bvh import TriangleBVH

    generator = torch.Generator(device="cpu").manual_seed(0)
    chosen = []
    member = torch.zeros(int(piece.max()) + 1, dtype=torch.bool, device=piece.device)
    member[pieces] = True
    candidates = torch.nonzero(member[piece]).squeeze(1)
    order = torch.argsort(torch.rand(candidates.shape[0], generator=generator).to(piece.device))
    candidates = candidates[order]
    rank = torch.zeros_like(candidates)
    group = piece[candidates]
    sorted_group, by_group = torch.sort(group, stable=True)
    head = torch.ones_like(sorted_group, dtype=torch.bool)
    head[1:] = sorted_group[1:] != sorted_group[:-1]
    index = torch.arange(sorted_group.shape[0], device=piece.device)
    rank[by_group] = index - torch.cummax(torch.where(head, index, torch.zeros_like(index)), 0).values
    chosen = candidates[rank < VOTE_FACES]
    tree = TriangleBVH(positions, faces, device=device)
    corners = positions[faces[chosen]].float()
    centre = corners.mean(dim=1)
    normal = topology.face_normals(positions.float(), faces[chosen])
    span = (corners[:, 1] - corners[:, 0]).norm(dim=1).clamp_min(1e-9)
    offset = (span * 1e-3)[:, None] * normal
    front = ~tree.any_hit(centre + offset, normal)
    back = ~tree.any_hit(centre - offset, -normal)
    good = torch.zeros_like(member, dtype=torch.int64).index_add_(0, piece[chosen], (front & ~back).long())
    bad = torch.zeros_like(member, dtype=torch.int64).index_add_(0, piece[chosen], (back & ~front).long())
    return (bad > good)[pieces] & (bad[pieces] >= 3)


# ---------------------------------------------------------------------------------------------
# Hole filling


def _triangulate_holes(points: torch.Tensor, beside: torch.Tensor, forbidden: torch.Tensor) -> torch.Tensor:
    """Minimal-area triangulation of equal-length closed polygons, penalising folds and edges the mesh already has.

    Args:
        points: ``(H, n, 3)`` float64 polygon corners in order.
        beside: ``(H, n, 3)`` unit normal of the face beside edge ``i`` to ``i + 1``.
        forbidden: ``(H, n, n)`` bool, corner pairs already joined by a mesh edge that is not a polygon side.

    Returns:
        ``(H, n - 2, 3)`` int64 corner indices of each polygon's triangles.
    """
    holes, n, _ = points.shape
    device = points.device
    span = (points - points.roll(-1, dims=1)).norm(dim=2).mean(dim=1)
    scale = (span * span)[:, None, None]
    best = torch.full((holes, n, n), math.inf, dtype=torch.float64, device=device)
    choice = torch.zeros((holes, n, n), dtype=torch.int64, device=device)
    step = torch.arange(n - 1, device=device)
    best[:, step, step + 1] = 0.0
    for gap in range(2, n):
        first = torch.arange(n - gap, device=device)
        last = first + gap
        middle = first[:, None] + torch.arange(1, gap, device=device)[None, :]
        a = points[:, first][:, :, None, :]
        b = points[:, middle]
        c = points[:, last][:, :, None, :]
        normal = torch.cross(b - a, c - a, dim=-1)
        twice = normal.norm(dim=-1)
        unit = normal / twice.clamp_min(1e-30)[..., None]
        on_left = middle == first[:, None] + 1
        on_right = last[:, None] == middle + 1
        fold = torch.where(on_left, 1.0 - (unit * beside[:, first][:, :, None, :]).sum(-1), 0.0)
        fold = fold + torch.where(on_right, 1.0 - (unit * beside[:, middle]).sum(-1), 0.0)
        clash = (forbidden[:, first[:, None], middle] & ~on_left) | (forbidden[:, middle, last[:, None]] & ~on_right)
        if gap == n - 1:
            fold = fold + 1.0 - (unit * beside[:, n - 1][:, None, None, :]).sum(-1)
        else:
            clash = clash | forbidden[:, first, last][:, :, None]
        cost = 0.5 * twice + scale * (0.5 * fold + CLASH_PENALTY * clash)
        total = best[:, first[:, None], middle] + best[:, middle, last[:, None]] + cost
        value, pick = total.min(dim=-1)
        best[:, first, last] = value
        choice[:, first, last] = first[None, :] + 1 + pick
    triangles = []
    hole = torch.arange(holes, device=device)
    low = torch.zeros(holes, dtype=torch.int64, device=device)
    high = torch.full((holes,), n - 1, dtype=torch.int64, device=device)
    while hole.numel():
        mid = choice[hole, low, high]
        triangles.append(torch.stack([hole, low, mid, high], dim=1))
        left = mid - low >= 2
        right = high - mid >= 2
        hole, low, high = (torch.cat([hole[left], hole[right]]), torch.cat([low[left], mid[right]]),
                           torch.cat([mid[left], high[right]]))
    found = torch.cat(triangles)
    order = torch.argsort(found[:, 0], stable=True)
    return found[order][:, 1:].reshape(holes, n - 2, 3)


def _angle_at(apex: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Angle at ``apex`` between the directions to ``left`` and ``right``."""
    first, second = left - apex, right - apex
    return torch.atan2(torch.cross(first, second, dim=-1).norm(dim=-1), (first * second).sum(-1))


def relax_patch(points: torch.Tensor, faces: torch.Tensor, fixed: torch.Tensor, blocked: torch.Tensor) -> torch.Tensor:
    """Flip edges whose opposite angles sum past a half turn, keeping fixed edges and never making blocked ones.

    Args:
        points: ``(N, 3)`` float64 positions.
        faces: ``(F, 3)`` int64 triangles.
        fixed: Edge keys ``low * N + high`` never flipped.
        blocked: Edge keys never created.

    Returns:
        ``(F, 3)`` triangles after the flips.
    """
    count = points.shape[0]
    for _ in range(RELAX_ROUNDS):
        edges = topology.edge_topology(faces, count)
        keys = edges.edges[:, 0] * count + edges.edges[:, 1]
        inner = (edges.edge_face_count == 2) & ~torch.isin(keys, fixed)
        rows = torch.nonzero(inner).squeeze(1)
        if not rows.numel():
            break
        f0, f1 = edges.edge_faces[rows, 0], edges.edge_faces[rows, 1]
        k0 = edges.edge_corners[rows, 0]
        u = faces[f0, k0]
        w = faces[f0, (k0 + 1) % 3]
        o0 = edges.edge_opposite[rows, 0]
        o1 = edges.edge_opposite[rows, 1]
        p = points
        excess = _angle_at(p[o0], p[u], p[w]) + _angle_at(p[o1], p[u], p[w]) - math.pi
        new_key = torch.minimum(o0, o1) * count + torch.maximum(o0, o1)
        old_normal = torch.cross(p[w] - p[u], p[o0] - p[u], dim=1)
        first = torch.cross(p[o1] - p[u], p[o0] - p[u], dim=1)
        second = torch.cross(p[w] - p[o1], p[o0] - p[o1], dim=1)
        ok = ((excess > 1e-9) & ~torch.isin(new_key, keys) & ~torch.isin(new_key, blocked) & (o0 != o1)
              & ((first * old_normal).sum(1) > 0) & ((second * old_normal).sum(1) > 0))
        if not bool(ok.any()):
            break
        rows, f0, f1, u, w, o0, o1, excess = (rows[ok], f0[ok], f1[ok], u[ok], w[ok], o0[ok], o1[ok], excess[ok])
        face_total = faces.shape[0]
        top = torch.full((face_total,), -math.inf, dtype=excess.dtype, device=faces.device)
        top.scatter_reduce_(0, f0, excess, reduce="amax")
        top.scatter_reduce_(0, f1, excess, reduce="amax")
        best = (excess == top[f0]) & (excess == top[f1])
        lowest = torch.full((face_total,), rows.numel(), dtype=torch.int64, device=faces.device)
        order = torch.arange(rows.numel(), device=faces.device)
        lowest.scatter_reduce_(0, f0[best], order[best], reduce="amin")
        lowest.scatter_reduce_(0, f1[best], order[best], reduce="amin")
        go = best & (lowest[f0] == order) & (lowest[f1] == order)
        if not bool(go.any()):
            break
        faces = faces.clone()
        faces[f0[go]] = torch.stack([u[go], o1[go], o0[go]], dim=1)
        faces[f1[go]] = torch.stack([o1[go], w[go], o0[go]], dim=1)
    return faces


def _fair(local_points: torch.Tensor, local_faces: torch.Tensor, unknown: torch.Tensor,
          rows: torch.Tensor) -> torch.Tensor:
    """Positions of the unknown vertices minimising the squared normalised Laplacian over ``rows``."""
    count = local_points.shape[0]
    device = local_points.device
    if FAIR_WEIGHTS == "cotangent":
        operator = topology.cotangent_laplacian(local_points, local_faces, clamp=1e3, nonnegative=True)
        index, value = operator.indices(), operator.values()
        off = index[0] != index[1]
        source, target, weight = index[0][off], index[1][off], -value[off]
    else:
        edges = topology.edge_topology(local_faces, count).edges
        source = torch.cat([edges[:, 0], edges[:, 1]])
        target = torch.cat([edges[:, 1], edges[:, 0]])
        weight = torch.ones(source.shape[0], dtype=torch.float64, device=device)
    total = torch.zeros(count, dtype=torch.float64, device=device).index_add_(0, source, weight)
    empty = total[source] <= 1e-12
    degree = torch.bincount(source, minlength=count).clamp_min(1).double()
    share = torch.where(empty, 1.0 / degree[source], weight / total[source].clamp_min(1e-12))
    values = torch.cat([-share, torch.ones(count, dtype=torch.float64, device=device)])
    index = torch.stack([torch.cat([source, torch.arange(count, device=device)]),
                         torch.cat([target, torch.arange(count, device=device)])])
    in_rows = torch.zeros(count, dtype=torch.bool, device=device)
    in_rows[rows] = True
    keep = in_rows[index[0]]
    laplacian = torch.sparse_coo_tensor(index[:, keep], values[keep], (count, count),
                                        check_invariants=False).coalesce()
    transpose = laplacian.t().coalesce()
    unknown_mask = torch.zeros(count, dtype=torch.bool, device=device)
    unknown_mask[unknown] = True
    known = torch.where(unknown_mask[:, None], torch.zeros_like(local_points), local_points)
    rhs = -(transpose @ (laplacian @ known))[unknown]
    diagonal = torch.zeros(count, dtype=torch.float64, device=device)
    lap_index, lap_values = laplacian.indices(), laplacian.values()
    diagonal.index_add_(0, lap_index[1], lap_values * lap_values)
    inverse = 1.0 / diagonal[unknown].clamp_min(1e-12)

    def apply(value: torch.Tensor) -> torch.Tensor:
        full = torch.zeros(count, value.shape[1], dtype=torch.float64, device=device)
        full[unknown] = value
        return (transpose @ (laplacian @ full))[unknown]

    return topology.solve_cg(apply, rhs, x0=local_points[unknown], tol=1e-10, max_iter=FAIR_ITERATIONS,
                             preconditioner=inverse)


@dataclass
class _Loops:
    """Boundary loops a hole patch may close, one entry per loop corner in winding order."""

    loops: list[torch.Tensor]
    flat: torch.Tensor
    offsets: torch.Tensor
    lengths: list[int]
    usable: list[bool]
    split_start: torch.Tensor
    seamless: list[bool]
    beside: torch.Tensor
    piece_area: torch.Tensor
    face: torch.Tensor
    side: torch.Tensor


def _hole_loops(surface: Surface, max_edges: int) -> _Loops | None:
    """The closed boundary loops short enough to fill, with the faces and split vertices along each."""
    device = surface.device
    ids, positions = surface.topology()
    welded = ids[surface.faces]
    point_count = positions.shape[0]
    edges = topology.edge_topology(welded, point_count)
    chains, closed = topology.boundary_chains(welded, point_count, edges)
    loops = [chain for chain, is_closed in zip(chains, closed.tolist())
             if is_closed and 3 <= chain.numel() <= max_edges and torch.unique(chain).numel() == chain.numel()]
    if not loops:
        return None
    half = torch.nonzero(edges.boundary[edges.face_edges.reshape(-1)]).squeeze(1)
    nxt = torch.tensor([1, 2, 0], device=device)
    half_key = welded.reshape(-1)[half] * point_count + welded[:, nxt].reshape(-1)[half]
    key_order = torch.argsort(half_key)
    sorted_key = half_key[key_order]
    lengths = torch.tensor([loop.numel() for loop in loops], device=device)
    flat = torch.cat(loops)
    loop_of = torch.repeat_interleave(torch.arange(len(loops), device=device), lengths)
    offsets = torch.cumsum(lengths, 0) - lengths
    position = torch.arange(flat.numel(), device=device) - offsets[loop_of]
    following = offsets[loop_of] + (position + 1) % lengths[loop_of]
    query = flat * point_count + flat[following]
    at = torch.searchsorted(sorted_key, query).clamp_max(sorted_key.numel() - 1)
    found = sorted_key[at] == query
    usable = torch.ones(len(loops), dtype=torch.bool, device=device)
    usable[loop_of[~found]] = False
    halfedge = half[key_order[at]]
    face, side = halfedge // 3, halfedge % 3
    split_start = surface.faces[face, side]
    split_end = surface.faces[face, (side + 1) % 3]
    seamless = torch.ones(len(loops), dtype=torch.bool, device=device)
    seamless[loop_of[split_end != split_start[following]]] = False
    piece = topology.face_components(welded, point_count)
    piece_area = torch.zeros(int(piece.max()) + 1, dtype=torch.float64, device=device).index_add_(
        0, piece, topology.face_areas(positions.double(), welded))
    return _Loops(loops, flat, offsets, lengths.tolist(), usable.tolist(), split_start, seamless.tolist(),
                  topology.face_normals(positions.double(), welded)[face], piece_area[piece[face[offsets]]],
                  face, side)


def _close_slits(surface: Surface, found: _Loops) -> tuple[Surface, int]:
    """Close three-corner holes of no area, where a vertex sits on the edge across, by splitting that edge's face."""
    device = surface.device
    _, positions = surface.topology()
    items = [item for item, ok in enumerate(found.usable) if ok and found.lengths[item] == 3]
    if not items:
        return surface, 0
    start = found.offsets[torch.tensor(items, device=device)]
    entry = start[:, None] + torch.arange(3, device=device)[None, :]
    points = positions[found.flat[entry]].double()
    side_length = (points.roll(-1, dims=1) - points).norm(dim=2)
    perimeter = side_length.sum(dim=1)
    twice = torch.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0], dim=1).norm(dim=1)
    slit = twice <= 2.0 * SLIT_RATIO * perimeter * perimeter
    longest = side_length.argmax(dim=1)
    rows = torch.arange(len(items), device=device)
    across = entry[rows, longest]
    middle = entry[rows, (longest + 2) % 3]
    face = found.face[across]
    first_use = torch.full((surface.face_count,), len(items), dtype=torch.int64, device=device)
    first_use.scatter_reduce_(0, face[slit], rows[slit], reduce="amin")
    slit &= first_use[face] == rows
    if not bool(slit.any()):
        return surface, 0
    face, side, middle = face[slit], found.side[across[slit]], middle[slit]
    corner_a = surface.faces[face, side]
    corner_b = surface.faces[face, (side + 1) % 3]
    corner_c = surface.faces[face, (side + 2) % 3]
    a, b = positions[found.flat[across[slit]]].double(), surface.vertices[corner_b].double()
    target = positions[found.flat[middle]]
    t = ((target.double() - a).norm(dim=1) / (b - a).norm(dim=1).clamp_min(1e-30)).clamp(0.0, 1.0)
    added = blend(surface.attributes, torch.stack([corner_a, corner_b], 1),
                  torch.stack([1.0 - t, t], 1).float())
    new_vertex = surface.vertex_count + torch.arange(face.numel(), device=device)
    faces = surface.faces.clone()
    faces[face] = torch.stack([corner_a, new_vertex, corner_c], 1)
    extra = torch.stack([new_vertex, corner_b, corner_c], 1)
    count = int(face.numel())
    source = surface.source.clone()
    source[face] = -1
    result = Surface(
        torch.cat([surface.vertices, target]), torch.cat([faces, extra]), append_rows(surface.attributes, added),
        torch.cat([source, torch.full((count,), -1, dtype=torch.int64, device=device)]),
        torch.cat([surface.flipped, torch.zeros(count, dtype=torch.bool, device=device)]),
        torch.cat([surface.copy, torch.zeros(count, dtype=torch.int64, device=device)]),
    )
    return result, count


def _plan_patches(surface: Surface, found: _Loops) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Triangulate every usable loop and keep the patches that neither reuse a mesh edge, close a slit nor wrap a sheet."""
    device = surface.device
    ids, positions = surface.topology()
    point_count = positions.shape[0]
    edges = topology.edge_topology(ids[surface.faces], point_count)
    edge_keys = edges.edges[:, 0] * point_count + edges.edges[:, 1]
    by_length: dict[int, list[int]] = {}
    for item, ok in enumerate(found.usable):
        if ok:
            by_length.setdefault(found.lengths[item], []).append(item)
    accepted: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for n, items in sorted(by_length.items()):
        for begin in range(0, len(items), max(1, HOLE_BATCH // (n * n))):
            chunk = items[begin:begin + max(1, HOLE_BATCH // (n * n))]
            item_t = torch.tensor(chunk, device=device)
            base = found.offsets[item_t][:, None]
            rows = base + (n - 1 - torch.arange(n, device=device))[None, :]
            corner_id = found.flat[rows]
            points = positions[corner_id].double()
            beside = found.beside[(rows - 1 - base) % n + base]
            low = torch.minimum(corner_id[:, :, None], corner_id[:, None, :])
            high = torch.maximum(corner_id[:, :, None], corner_id[:, None, :])
            forbidden = torch.isin(low * point_count + high, edge_keys)
            step = torch.arange(n, device=device)
            forbidden[:, step, (step + 1) % n] = False
            forbidden[:, (step + 1) % n, step] = False
            triangles = _triangulate_holes(points, beside, forbidden)
            hole = torch.arange(len(chunk), device=device)[:, None]
            a, b, c = triangles[..., 0], triangles[..., 1], triangles[..., 2]
            reused = (forbidden[hole, a, b] | forbidden[hole, b, c] | forbidden[hole, a, c]).any(dim=1)
            pa, pb, pc = points[hole, a], points[hole, b], points[hole, c]
            area = 0.5 * torch.cross(pb - pa, pc - pa, dim=-1).norm(dim=-1).sum(dim=1)
            perimeter = (points - points.roll(-1, dims=1)).norm(dim=2).sum(dim=1)
            valid = (~reused & (area > SLIT_RATIO * perimeter * perimeter)
                     & (area < SHEET_SHARE * found.piece_area[item_t])).tolist()
            for slot, item in enumerate(chunk):
                if valid[slot]:
                    accepted[item] = (triangles[slot], forbidden[slot])
    return accepted


def fill_holes(surface: Surface, max_edges: int) -> tuple[Surface, int, int, int]:
    """Fill holes with a minimal-area triangulation refined to the border's spacing and faired, attributes from the border.

    Args:
        surface: The surface, consistently wound.
        max_edges: Longest hole filled, in boundary edges.

    Returns:
        ``(surface, holes, faces, flat)``: holes filled, faces added and holes given one texture coordinate.
    """
    max_edges = max(3, min(int(max_edges), MAX_HOLE_EDGES))
    if surface.face_count == 0:
        return surface, 0, 0, 0
    found = _hole_loops(surface, max_edges)
    if found is None:
        return surface, 0, 0, 0
    surface, slits = _close_slits(surface, found)
    if slits:
        found = _hole_loops(surface, max_edges)
        if found is None:
            return surface, slits, slits, 0
    accepted = _plan_patches(surface, found)
    if not accepted:
        return surface, slits, slits, 0
    device = surface.device
    ids, positions = surface.topology()
    hole_list = sorted(accepted)
    base_of: dict[int, int] = {}
    entries, faces_parts, face_hole, vertex_hole, blocked_parts, following = [], [], [], [], [], []
    count = 0
    for item in hole_list:
        n = found.lengths[item]
        entries.append(int(found.offsets[item]) + (n - 1 - torch.arange(n, device=device)))
        base_of[item] = count
        triangles, forbidden = accepted[item]
        faces_parts.append(triangles + count)
        face_hole.append(torch.full((n - 2,), item, dtype=torch.int64, device=device))
        vertex_hole.append(torch.full((n,), item, dtype=torch.int64, device=device))
        pairs = torch.nonzero(torch.triu(forbidden, 1))
        blocked_parts.append(pairs + count)
        following.append(count + (torch.arange(n, device=device) + 1) % n)
        count += n
    loop_count = count
    entry = torch.cat(entries)
    loop_points = positions[found.flat[entry]].double()
    loop_source = found.split_start[entry]
    faces_local = torch.cat(faces_parts)
    hole_of_face = torch.cat(face_hole)
    hole_of_vertex = torch.cat(vertex_hole)
    loop_next = torch.cat(following)
    loop_index = torch.arange(loop_count, device=device)
    spans = (loop_points - loop_points[loop_next]).norm(dim=1)
    previous = torch.empty_like(loop_index)
    previous[loop_next] = loop_index
    spacing = 0.5 * (spans + spans[previous])
    stride = MAX_FILL_VERTICES + loop_count + 1
    fixed = torch.minimum(loop_index, loop_next) * stride + torch.maximum(loop_index, loop_next)
    blocked_pairs = torch.cat(blocked_parts)
    blocked = blocked_pairs[:, 0] * stride + blocked_pairs[:, 1]
    patch_attributes = take(surface.attributes, loop_source)
    points = loop_points
    for _ in range(REFINE_ROUNDS):
        corners = points[faces_local]
        centre = corners.mean(dim=1)
        centre_spacing = spacing[faces_local].mean(dim=1)
        reach = DENSITY * (corners - centre[:, None, :]).norm(dim=2)
        split = ((reach > centre_spacing[:, None]) & (reach > spacing[faces_local])).all(dim=1)
        rows = torch.nonzero(split).squeeze(1)
        if not rows.numel() or points.shape[0] + rows.numel() > stride - 1:
            break
        new_ids = points.shape[0] + torch.arange(rows.numel(), device=device)
        third = torch.full((rows.numel(), 3), 1.0 / 3.0, dtype=torch.float32, device=device)
        patch_attributes = append_rows(patch_attributes, blend(patch_attributes, faces_local[rows], third))
        points = torch.cat([points, centre[rows]])
        spacing = torch.cat([spacing, centre_spacing[rows]])
        hole_of_vertex = torch.cat([hole_of_vertex, hole_of_face[rows]])
        a, b, c = faces_local[rows, 0], faces_local[rows, 1], faces_local[rows, 2]
        keep = ~split
        faces_local = torch.cat([faces_local[keep], torch.stack([a, b, new_ids], 1),
                                 torch.stack([b, c, new_ids], 1), torch.stack([c, a, new_ids], 1)])
        hole_of_face = torch.cat([hole_of_face[keep], hole_of_face[rows], hole_of_face[rows], hole_of_face[rows]])
        size = points.shape[0]
        faces_local = relax_patch(points, faces_local, _rekey(fixed, stride, size), _rekey(blocked, stride, size))
    total = points.shape[0]
    if total > loop_count:
        points = _fair_patch(surface, points, faces_local, found.flat[entry], loop_count, patch_attributes)
    flat_holes = [item for item in hole_list if not found.seamless[item]]
    # Loop corners of a hole crossing a texture seam become new vertices; seamless loops reuse their border.
    duplicate_loop = torch.zeros(loop_count, dtype=torch.bool, device=device)
    for item in flat_holes:
        duplicate_loop[base_of[item]:base_of[item] + found.lengths[item]] = True
    new_rows = torch.cat([torch.nonzero(duplicate_loop).squeeze(1), torch.arange(loop_count, total, device=device)])
    vertex_of = torch.empty(total, dtype=torch.int64, device=device)
    vertex_of[:loop_count] = loop_source
    vertex_of[new_rows] = surface.vertex_count + torch.arange(new_rows.numel(), device=device)
    added_attributes = take(patch_attributes, new_rows)
    if flat_holes and "uvs" in added_attributes:
        first_uv = torch.zeros(len(found.loops), 2, dtype=added_attributes["uvs"].dtype, device=device)
        for item in flat_holes:
            first_uv[item] = surface.attributes["uvs"][loop_source[base_of[item]]]
        is_flat = torch.zeros(len(found.loops), dtype=torch.bool, device=device)
        is_flat[torch.tensor(flat_holes, device=device)] = True
        hole_rows = hole_of_vertex[new_rows]
        on_flat = is_flat[hole_rows]
        added_attributes["uvs"][on_flat] = first_uv[hole_rows[on_flat]]
    faces_new = vertex_of[faces_local]
    added_faces = int(faces_new.shape[0])
    result = Surface(
        torch.cat([surface.vertices, points[new_rows].float()]),
        torch.cat([surface.faces, faces_new]),
        append_rows(surface.attributes, added_attributes),
        torch.cat([surface.source, torch.full((added_faces,), -1, dtype=torch.int64, device=device)]),
        torch.cat([surface.flipped, torch.zeros(added_faces, dtype=torch.bool, device=device)]),
        torch.cat([surface.copy, torch.zeros(added_faces, dtype=torch.int64, device=device)]),
    )
    return result, len(hole_list) + slits, added_faces + slits, len(flat_holes)


def _fair_patch(surface: Surface, points: torch.Tensor, faces_local: torch.Tensor, loop_ids: torch.Tensor,
                loop_count: int, patch_attributes: dict[str, torch.Tensor]) -> torch.Tensor:
    """Patch positions after fairing against the faces around the holes; new vertex normals follow the faired shape."""
    device = surface.device
    ids, positions = surface.topology()
    welded = ids[surface.faces]
    total = points.shape[0]
    ring_faces = welded[torch.isin(welded, loop_ids).any(dim=1)]
    ring_points = torch.unique(ring_faces)
    local_of = torch.full((positions.shape[0],), -1, dtype=torch.int64, device=device)
    others = ring_points[~torch.isin(ring_points, loop_ids)]
    local_of[others] = total + torch.arange(others.numel(), device=device)
    local_of[loop_ids.flip(0)] = torch.arange(loop_count - 1, -1, -1, device=device)
    local_points = torch.cat([points, positions[others].double()])
    local_faces = torch.cat([faces_local, local_of[ring_faces]])
    unknown = torch.arange(loop_count, total, device=device)
    solved = _fair(local_points, local_faces, unknown, torch.arange(total, device=device))
    faired = points.clone()
    faired[loop_count:] = solved
    if "normals" in patch_attributes:
        shaped = local_points.clone()
        shaped[loop_count:total] = solved
        normals = topology.vertex_normals(shaped, local_faces)[loop_count:total]
        patch_attributes["normals"][loop_count:] = normals.to(patch_attributes["normals"].dtype)
    return faired


def _rekey(keys: torch.Tensor, stride: int, count: int) -> torch.Tensor:
    """Edge keys written for one vertex stride, rewritten for a vertex count."""
    return (keys // stride) * count + keys % stride


# ---------------------------------------------------------------------------------------------
# Normals


def recompute_normals(surface: Surface, crease_angle: float) -> tuple[Surface, int]:
    """Recompute normals, smooth across edges flatter than the crease angle and split where sharper.

    Args:
        surface: The surface.
        crease_angle: Dihedral angle in degrees at and above which an edge stays sharp; 180 smooths every edge.

    Returns:
        ``(surface, added)``: the surface and how many vertices were added along sharp edges.
    """
    if surface.face_count == 0:
        return surface, 0
    device = surface.device
    ids, positions = surface.topology()
    welded = ids[surface.faces]
    point_count = positions.shape[0]
    edges = topology.edge_topology(welded, point_count)
    pair_ids, pairs = edges.manifold_face_pairs()
    sides = edges.edge_corners[pair_ids]
    normals = topology.face_normals(positions.double(), welded)
    cosine = (normals[pairs[:, 0]] * normals[pairs[:, 1]]).sum(1)
    smooth = cosine > math.cos(math.radians(min(max(float(crease_angle), 0.0), 180.0))) - 1e-12
    if float(crease_angle) >= 180.0:
        smooth = torch.ones_like(smooth)
    link_a, link_b = _pair_corners(welded, pairs[smooth, 0], sides[smooth, 0], pairs[smooth, 1], sides[smooth, 1])
    corner_count = surface.face_count * 3
    group = topology.union_find(corner_count, link_a, link_b)
    angles = topology.corner_angles(positions.double(), welded)
    weighted = normals[:, None, :] * angles[..., None]
    total = torch.zeros(corner_count, 3, dtype=torch.float64, device=device)
    total.index_add_(0, group, weighted.reshape(-1, 3))
    vertex = surface.faces.reshape(-1)
    keys, inverse = torch.unique(vertex * corner_count + group, return_inverse=True)
    owner = keys // corner_count
    head = torch.ones(owner.shape[0], dtype=torch.bool, device=device)
    head[1:] = owner[1:] != owner[:-1]
    extra = ~head
    count = surface.vertex_count
    new_index = owner.clone()
    new_index[extra] = count + torch.arange(int(extra.sum()), dtype=torch.int64, device=device)
    rows = torch.cat([torch.arange(count, device=device), owner[extra]])
    faces = new_index[inverse].reshape(-1, 3)
    group_of_key = torch.zeros(keys.shape[0], dtype=torch.int64, device=device)
    group_of_key[inverse] = group
    vertex_normal = torch.nn.functional.normalize(total[group_of_key], dim=1)
    result = _remap_vertices(surface, faces, rows, surface.vertices[rows])
    fresh = torch.zeros(rows.shape[0], 3, dtype=torch.float32, device=device)
    old = surface.attributes.get("normals")
    if old is not None:
        fresh[:] = old[rows].float()
    fresh[new_index] = vertex_normal.float()
    result.attributes["normals"] = fresh
    tangents = result.attributes.get("tangents")
    if tangents is not None:
        direction = tangents[:, :3].float()
        direction = direction - (direction * fresh).sum(1, keepdim=True) * fresh
        result.attributes["tangents"] = torch.cat([torch.nn.functional.normalize(direction, dim=1),
                                                   tangents[:, 3:]], dim=1).to(tangents.dtype)
    result.topo, result.positions = ids[rows], positions
    return result, int(extra.sum())


# ---------------------------------------------------------------------------------------------
# Polygons


def carry_polygons(part: Part, surface: Surface) -> Polygons | None:
    """Quads and n-gons of the original part whose triangles all survived intact, the rest as triangles.

    Args:
        part: The part the surface was read from.
        surface: The repaired surface.

    Returns:
        Polygons on the CPU, or None when the part had none.
    """
    polygons = part.polygons
    if polygons is None or surface.face_count == 0:
        return None
    device = surface.device
    original = part.faces.long().to(device)
    sizes = polygons.sizes.long().to(device)
    indices = polygons.indices.long().to(device)
    owner = polygons.triangle_polygon.long().to(device)
    poly_count = int(sizes.shape[0])
    none = torch.zeros(0, dtype=torch.int64, device=device)
    loose_slot = torch.full((surface.face_count,), -1, dtype=torch.int64, device=device)
    if poly_count == 0 or owner.shape[0] != original.shape[0] or original.numel() == 0:
        return _assemble_polygons(surface, loose_slot, torch.zeros(1, dtype=torch.bool, device=device),
                                  none, none, none)
    copies = int(surface.copy.max()) + 1
    slots = poly_count * copies
    source = surface.source
    rows = torch.nonzero((source >= 0) & (source < original.shape[0])).squeeze(1)
    slot = loose_slot.clone()
    slot[rows] = owner[source[rows]] * copies + surface.copy[rows]
    expected = torch.bincount(owner, minlength=poly_count).repeat_interleave(copies)
    present = torch.bincount(slot[rows], minlength=slots)
    flips = torch.bincount(slot[rows], weights=surface.flipped[rows].double(), minlength=slots)
    whole = (present == expected) & (expected > 0) & ((flips == 0) | (flips == present.double()))
    perm = torch.where(surface.flipped[rows, None], torch.tensor([0, 2, 1], device=device),
                       torch.tensor([0, 1, 2], device=device))
    span = int(original.max()) + 1
    pair = (slot[rows][:, None] * span + original[source[rows][:, None], perm]).reshape(-1)
    current = surface.faces[rows].reshape(-1)
    if pair.numel() == 0:
        return _assemble_polygons(surface, loose_slot, torch.zeros(1, dtype=torch.bool, device=device),
                                  none, none, none)
    known, inverse = torch.unique(pair, return_inverse=True)
    lowest = torch.full((known.shape[0],), 1 << 62, dtype=torch.int64, device=device)
    highest = torch.full_like(lowest, -1)
    lowest.scatter_reduce_(0, inverse, current, reduce="amin")
    highest.scatter_reduce_(0, inverse, current, reduce="amax")
    whole[known[lowest != highest] // span] = False
    starts = torch.cumsum(sizes, 0) - sizes
    for _ in range(2):
        kept = torch.nonzero(whole).squeeze(1)
        size = sizes[kept // copies]
        corner_slot = torch.repeat_interleave(kept, size)
        corner_size = torch.repeat_interleave(size, size)
        rank = torch.arange(corner_slot.shape[0], device=device) - torch.repeat_interleave(
            torch.cumsum(size, 0) - size, size)
        rank = torch.where(flips[corner_slot] > 0, (corner_size - rank) % corner_size, rank)
        query = corner_slot * span + indices[starts[corner_slot // copies] + rank]
        at = torch.searchsorted(known, query).clamp_max(known.shape[0] - 1)
        hit = known[at] == query
        if bool(hit.all()):
            return _assemble_polygons(surface, slot, whole, kept, size, lowest[at])
        whole[corner_slot[~hit]] = False
    return _assemble_polygons(surface, loose_slot, torch.zeros(1, dtype=torch.bool, device=device),
                              none, none, none)


def _assemble_polygons(surface: Surface, slot: torch.Tensor, whole: torch.Tensor, kept: torch.Tensor,
                       size: torch.Tensor, mapped: torch.Tensor) -> Polygons:
    """Kept polygons first, then every other face as a triangle, with each face's polygon."""
    device = surface.device
    face_count = surface.face_count
    in_kept = torch.zeros(face_count, dtype=torch.bool, device=device)
    valid = slot >= 0
    in_kept[valid] = whole[slot[valid]]
    number = torch.full((whole.shape[0],), -1, dtype=torch.int64, device=device)
    number[kept] = torch.arange(kept.shape[0], device=device)
    loose = torch.nonzero(~in_kept).squeeze(1)
    triangle_polygon = torch.empty(face_count, dtype=torch.int64, device=device)
    triangle_polygon[in_kept] = number[slot[in_kept]]
    triangle_polygon[loose] = kept.shape[0] + torch.arange(loose.shape[0], device=device)
    sizes = torch.cat([size, torch.full((loose.shape[0],), 3, dtype=torch.int64, device=device)])
    indices = torch.cat([mapped, surface.faces[loose].reshape(-1)])
    return Polygons(sizes.cpu(), indices.cpu(), triangle_polygon.cpu())


# ---------------------------------------------------------------------------------------------
# The whole pass


def borders_left(surface: Surface, max_edges: int) -> tuple[int, int, int, int]:
    """Count what still keeps a surface from being watertight.

    Args:
        surface: The surface.
        max_edges: Longest hole the filler accepts, in boundary edges.

    Returns:
        ``(openings, too_long, broken, non_manifold)``: border runs left, closed loops longer than
        ``max_edges``, runs that do not close into a loop, and non-manifold edges plus vertices.
    """
    if surface.face_count == 0:
        return 0, 0, 0, 0
    ids, positions = surface.topology()
    welded = ids[surface.faces]
    point_count = positions.shape[0]
    edges = topology.edge_topology(welded, point_count)
    non_manifold = int(edges.non_manifold.sum()) + int(
        topology.non_manifold_vertices(welded, point_count, edges).sum())
    if not bool(edges.boundary.any()):
        return 0, 0, 0, non_manifold
    chains, closed = topology.boundary_chains(welded, point_count, edges)
    lengths = torch.tensor([chain.numel() for chain in chains], dtype=torch.int64)
    closed = closed.cpu()
    limit = max(3, min(int(max_edges), MAX_HOLE_EDGES))
    return len(chains), int((closed & (lengths > limit)).sum()), int((~closed).sum()), non_manifold


#: Names of the counts `clean` returns, in the order they are reported.
COUNT_NAMES = (
    "welded", "degenerate", "duplicate", "flaps", "flap_faces", "small_shells", "small_shell_faces",
    "intersecting_faces",
    "intersecting_shells", "non_manifold_edges", "non_manifold_vertices", "flipped_faces",
    "turned_pieces", "holes_filled", "hole_faces", "flat_uv_holes", "normal_splits", "unreferenced",
    "openings_left", "openings_too_long", "openings_broken", "non_manifold_left",
)


def watertight(counts: dict) -> bool:
    """Whether the counts of `clean` describe a surface with no opening and no non-manifold spot."""
    return counts.get("openings_left", 0) == 0 and counts.get("non_manifold_left", 0) == 0


def describe(counts: dict) -> list[str]:
    """The counts of `clean` as readable lines, one per step that changed something, then the verdict.

    Args:
        counts: Counts as `clean` returns them, or summed over several surfaces.

    Returns:
        Lines of text.
    """
    c = {name: int(counts.get(name, 0)) for name in COUNT_NAMES}
    lines = []
    if c["welded"]:
        lines.append(f"{_tally(c['welded'], 'vertex', 'vertices')} welded")
    if c["degenerate"] or c["duplicate"]:
        lines.append(f"{_tally(c['degenerate'], 'degenerate face')} and {_tally(c['duplicate'], 'duplicate face')} "
                     "removed")
    if c["flaps"]:
        lines.append(f"{_tally(c['flaps'], 'loose flap')} on non-manifold edges removed "
                     f"({_tally(c['flap_faces'], 'face')})")
    if c["non_manifold_edges"] or c["non_manifold_vertices"]:
        lines.append(f"{_tally(c['non_manifold_edges'], 'non-manifold edge')} and "
                     f"{_tally(c['non_manifold_vertices'], 'non-manifold vertex', 'non-manifold vertices')} split")
    if c["small_shells"]:
        lines.append(f"{_tally(c['small_shells'], 'small shell')} removed ({_tally(c['small_shell_faces'], 'face')})")
    if c["intersecting_faces"]:
        lines.append(f"{_tally(c['intersecting_faces'], 'face crosses', 'faces cross')} another face")
    if c["intersecting_shells"]:
        lines.append(f"{_tally(c['intersecting_shells'], 'small intersecting shell')} removed")
    if c["flipped_faces"]:
        lines.append(f"{_tally(c['flipped_faces'], 'face')} rewound, "
                     f"{_tally(c['turned_pieces'], 'piece')} turned to face outward")
    if c["holes_filled"]:
        line = f"{_tally(c['holes_filled'], 'hole')} filled with {_tally(c['hole_faces'], 'face')}"
        if c["flat_uv_holes"]:
            verb = "takes" if c["flat_uv_holes"] == 1 else "take"
            line += (f"; {_tally(c['flat_uv_holes'], 'patch crosses', 'patches cross')} a texture seam and {verb} "
                     "the colour of one border point")
        lines.append(line)
    if c["normal_splits"]:
        lines.append(f"normals recomputed, {_tally(c['normal_splits'], 'vertex', 'vertices')} split along sharp edges")
    if c["unreferenced"]:
        lines.append(f"{_tally(c['unreferenced'], 'unused vertex', 'unused vertices')} removed")
    if not lines:
        lines.append("nothing needed repair")
    if watertight(c):
        lines.append("Watertight: yes")
    else:
        reasons = []
        if c["openings_left"]:
            reasons.append(f"{_tally(c['openings_left'], 'opening')} left")
            if c["openings_too_long"]:
                reasons.append(f"{c['openings_too_long']:,} longer than max_hole_edges")
            if c["openings_broken"]:
                reasons.append(f"{c['openings_broken']:,} with a border that does not close")
            rest = c["openings_left"] - c["openings_too_long"] - c["openings_broken"]
            if rest > 0:
                reasons.append(f"{rest:,} on a piece too small or too open to close, or where a patch would "
                               "cross the surface")
        if c["non_manifold_left"]:
            left = _tally(c["non_manifold_left"], "non-manifold edge or vertex", "non-manifold edges and vertices")
            reasons.append(f"{left} left")
        lines.append("Watertight: no; " + ", ".join(reasons))
    return lines


def _tally(count: int, one: str, many: str | None = None) -> str:
    """A count with its noun, singular or plural."""
    return f"{count:,} {one if count == 1 else (many or one + 's')}"


def clean(surface: Surface, settings: CleanSettings, device: torch.device | None = None) -> tuple[Surface, dict]:
    """Run the repairs ``settings`` turns on, in order, and count what each one changed.

    Args:
        surface: The surface to repair.
        settings: Which repairs run.
        device: Device for ray casting and trees; None means the surface's.

    Returns:
        ``(surface, counts)``: the repaired surface and the counts named in `COUNT_NAMES`.
    """
    from .quality import self_intersections

    counts = dict.fromkeys(COUNT_NAMES, 0)
    if settings.weld:
        surface, counts["welded"] = weld(surface, settings.weld_distance, settings.compare_normals and not settings.normals)
    if settings.degenerate:
        surface, counts["degenerate"], counts["duplicate"] = remove_degenerate(surface)
    if settings.non_manifold:
        surface, counts["flaps"], counts["flap_faces"] = remove_flaps(
            surface, settings.shell_min_faces, settings.shell_min_area)
        surface, counts["non_manifold_edges"], counts["non_manifold_vertices"] = split_non_manifold(surface)
    if settings.shells:
        surface, counts["small_shells"], counts["small_shell_faces"] = remove_small_shells(
            surface, settings.shell_min_faces, settings.shell_min_area)
    if settings.intersections in ("count", "remove small shells") and surface.face_count:
        crossing = self_intersections(surface.vertices, surface.faces, device=device)
        counts["intersecting_faces"] = int(crossing.sum())
        if settings.intersections == "remove small shells":
            surface, counts["intersecting_shells"], _faces = remove_intersecting_shells(
                surface, crossing.to(surface.device), settings.intersecting_max_area)
    if settings.orient:
        surface, counts["flipped_faces"], counts["turned_pieces"] = orient(surface, True, device)
    if settings.fill:
        surface, counts["holes_filled"], counts["hole_faces"], counts["flat_uv_holes"] = fill_holes(
            surface, settings.max_hole_edges)
    if settings.normals:
        surface, counts["normal_splits"] = recompute_normals(surface, settings.crease_angle)
    if settings.unreferenced:
        surface, counts["unreferenced"] = compact(surface)
    (counts["openings_left"], counts["openings_too_long"], counts["openings_broken"],
     counts["non_manifold_left"]) = borders_left(surface, settings.max_hole_edges)
    return surface, counts


def clean_part(part: Part, settings: CleanSettings, device: torch.device | str | None = None) -> tuple[Part, dict]:
    """Repair one part on the mesh compute device, falling back to the CPU when memory runs out.

    Args:
        part: The part to repair; it is not changed.
        settings: Which repairs run.
        device: The first device to try; None means the mesh compute device.

    Returns:
        ``(part, counts)``: a new part and the counts of `clean`.

    Raises:
        ValueError: The part's faces point past its vertices.
    """
    from .device import with_cpu_fallback

    surface = surface_of(part)

    def work(target: torch.device) -> tuple[Surface, dict]:
        repaired, counts = clean(surface.to(target), settings, target)
        return repaired.to("cpu"), counts

    repaired, counts = with_cpu_fallback(work, device, label="mesh clean")
    return part_of(part, repaired), counts

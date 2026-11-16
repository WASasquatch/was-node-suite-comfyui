"""Combining and dividing the parts of a model: merging parts, splitting shells, adding a model.

Part indices held by expression binds, first-person settings and shape key animations follow
every change, as do an added model's joint indices.
"""

from __future__ import annotations

__all__ = [
    "MAX_SHELLS", "add_model", "bake_rest", "merge_by_material", "merge_parts", "merge_skeletons",
    "reindex_parts", "split_shells", "subset_part",
]

import copy

import torch

from ..log import get_logger
from ..mesh import topology
from .scene import Animation, Model3D, MorphTarget, Part, Polygons, Skeleton, Springs

logger = get_logger("model3d.merge")

#: Most parts one part is split into; the smallest shells past it share the last part.
MAX_SHELLS = 4096

#: Largest difference from identity of ``rest @ inverse_bind`` treated as a model at its bind pose.
BIND_TOLERANCE = 1e-4


def _triangle_polygons(part: Part) -> Polygons:
    """A part's triangles as one polygon each."""
    count = part.triangle_count
    return Polygons(
        torch.full((count,), 3, dtype=torch.int64),
        part.faces.reshape(-1).long().clone(),
        torch.arange(count, dtype=torch.int64),
    )


def subset_part(part: Part, face_ids: torch.Tensor, name: str | None = None) -> Part:
    """A new part holding some of a part's triangles and only the vertices they use.

    Args:
        part: The source part; its material is shared, not copied.
        face_ids: ``(K,)`` int64 indices of the triangles kept, in the order they are written.
        name: Name of the new part, or None to keep the source's.

    Returns:
        The new part, with attributes, shape keys and polygons cut to match.
    """
    face_ids = face_ids.long()
    faces = part.faces.long()[face_ids]
    used = torch.zeros(part.vertex_count, dtype=torch.bool)
    if faces.numel():
        used[faces.reshape(-1)] = True
    keep = used.nonzero().flatten()
    remap = torch.full((part.vertex_count,), -1, dtype=torch.int64)
    remap[keep] = torch.arange(keep.numel(), dtype=torch.int64)

    def rows(value: torch.Tensor | None) -> torch.Tensor | None:
        return None if value is None else value[keep].clone()

    polygons = None
    if part.polygons is not None and face_ids.numel():
        source = part.polygons
        chosen = source.triangle_polygon.long()[face_ids]
        kept_polygons = torch.zeros(source.count, dtype=torch.bool)
        kept_polygons[chosen] = True
        polygon_of_corner = torch.repeat_interleave(torch.arange(source.count), source.sizes.long())
        indices = remap[source.indices.long()[kept_polygons[polygon_of_corner]]]
        if bool((indices >= 0).all()):
            renumber = torch.cumsum(kept_polygons.long(), 0) - 1
            polygons = Polygons(source.sizes[kept_polygons].clone(), indices, renumber[chosen])
    return Part(
        name=part.name if name is None else name,
        vertices=part.vertices[keep].clone(),
        faces=remap[faces] if faces.numel() else faces.reshape(0, 3),
        material=part.material,
        label=part.label,
        polygons=polygons,
        uvs=rows(part.uvs),
        normals=rows(part.normals),
        tangents=rows(part.tangents),
        colors=rows(part.colors),
        joints=rows(part.joints),
        weights=rows(part.weights),
        morphs={key: MorphTarget(morph.positions[keep].clone(), rows(morph.normals))
                for key, morph in part.morphs.items()},
        extras=copy.deepcopy(part.extras),
    )


def _filled(parts: list[Part], name: str, width: int, fill: float | None) -> torch.Tensor | None:
    """One attribute across parts, missing rows filled with ``fill``; None when no part holds it."""
    values = [getattr(part, name) for part in parts]
    if all(value is None for value in values):
        return None
    if fill is None and any(value is None for value in values):
        return None
    held = []
    for part, value in zip(parts, values):
        if value is None:
            value = torch.full((part.vertex_count, width), float(fill), dtype=torch.float32)
        elif value.shape[1] < width:
            pad = torch.full((part.vertex_count, width - value.shape[1]), 1.0, dtype=torch.float32)
            value = torch.cat([value.float(), pad], dim=1)
        held.append(value.float()[:, :width])
    return torch.cat(held, dim=0)


def merge_parts(parts: list[Part], name: str | None = None) -> Part:
    """One part holding several parts' triangles, drawn with the first part's material.

    Args:
        parts: The parts to join; all skinned or none skinned.
        name: Name of the joined part, or None for the first part's name.

    Returns:
        The joined part. Missing UVs become 0, missing colours white, missing normals are
        computed, and tangents are kept only when every part has them.

    Raises:
        ValueError: The list is empty, or some parts are skinned and some are not.
    """
    if not parts:
        raise ValueError("there are no parts to merge")
    skinned = [part.weights is not None and part.joints is not None for part in parts]
    if any(skinned) and not all(skinned):
        raise ValueError("a skinned part and a static part cannot be merged into one part; merge "
                         "the skinned parts and the static parts separately")
    first = parts[0]
    if len(parts) == 1:
        return subset_part(first, torch.arange(first.triangle_count), name)
    offsets = [0]
    for part in parts:
        offsets.append(offsets[-1] + part.vertex_count)
    normals = None
    if any(part.normals is not None for part in parts):
        normals = torch.cat([
            part.normals.float() if part.normals is not None
            else topology.vertex_normals(part.vertices.float(), part.faces.long())
            for part in parts
        ], dim=0)
    joints = weights = None
    if all(skinned):
        width = max(int(part.joints.shape[1]) for part in parts)
        joints = torch.cat([
            torch.nn.functional.pad(part.joints.long(), (0, width - int(part.joints.shape[1]))) for part in parts
        ], dim=0)
        weights = torch.cat([
            torch.nn.functional.pad(part.weights.float(), (0, width - int(part.weights.shape[1]))) for part in parts
        ], dim=0)
    morph_names: list[str] = []
    for part in parts:
        morph_names += [key for key in part.morphs if key not in morph_names]
    morphs = {}
    for key in morph_names:
        positions = torch.cat([
            part.morphs[key].positions.float() if key in part.morphs
            else torch.zeros(part.vertex_count, 3) for part in parts
        ], dim=0)
        with_normals = any(key in part.morphs and part.morphs[key].normals is not None for part in parts)
        morph_normals = torch.cat([
            part.morphs[key].normals.float()
            if key in part.morphs and part.morphs[key].normals is not None
            else torch.zeros(part.vertex_count, 3) for part in parts
        ], dim=0) if with_normals else None
        morphs[key] = MorphTarget(positions, morph_normals)
    polygons = None
    if any(part.polygons is not None for part in parts):
        held = [part.polygons if part.polygons is not None else _triangle_polygons(part) for part in parts]
        polygon_offsets = [0]
        for entry in held:
            polygon_offsets.append(polygon_offsets[-1] + entry.count)
        polygons = Polygons(
            torch.cat([entry.sizes.long() for entry in held]),
            torch.cat([entry.indices.long() + offset for entry, offset in zip(held, offsets)]),
            torch.cat([entry.triangle_polygon.long() + offset for entry, offset in zip(held, polygon_offsets)]),
        )
    extras = copy.deepcopy(first.extras)
    if any(isinstance(part.extras.get("morph_weights"), (list, tuple)) for part in parts):
        defaults: dict[str, float] = {}
        for part in parts:
            listed = part.extras.get("morph_weights")
            if isinstance(listed, (list, tuple)):
                for key, value in zip(part.morphs, listed):
                    defaults.setdefault(key, float(value))
        extras["morph_weights"] = [defaults.get(key, 0.0) for key in morph_names]
    labels = {part.label for part in parts}
    return Part(
        name=first.name if name is None else name,
        vertices=torch.cat([part.vertices.float() for part in parts], dim=0),
        faces=torch.cat([part.faces.long() + offset for part, offset in zip(parts, offsets)], dim=0),
        material=first.material,
        label=first.label if len(labels) == 1 else "",
        polygons=polygons,
        uvs=_filled(parts, "uvs", 2, 0.0),
        normals=normals,
        tangents=_filled(parts, "tangents", 4, None),
        colors=_filled(parts, "colors", 4, 1.0),
        joints=joints,
        weights=weights,
        morphs=morphs,
        extras=extras,
    )


def reindex_parts(model: Model3D, parts: list[Part], source: list[list[int]]) -> None:
    """Replace a model's parts and carry every part index held elsewhere in the model.

    Args:
        model: The model, changed in place.
        parts: The new parts.
        source: For each new part, the old part indices it came from. An old part may feed
            several new ones, as a split does.
    """
    new_of_old: dict[int, list[int]] = {}
    for new, olds in enumerate(source):
        for old in olds:
            new_of_old.setdefault(int(old), []).append(new)

    def targets(part) -> list[int]:
        return new_of_old.get(part, []) if isinstance(part, int) and not isinstance(part, bool) else []

    model.parts = list(parts)
    for expression in model.expressions.values():
        strongest: dict[tuple[int, str], float] = {}
        for part, morph, weight in expression.binds:
            for target in targets(part):
                if morph in model.parts[target].morphs:
                    key = (target, morph)
                    strongest[key] = max(strongest.get(key, 0.0), float(weight))
        expression.binds = [(part, morph, weight) for (part, morph), weight in strongest.items()]
    annotations = model.extras.get("vrm_first_person")
    if isinstance(annotations, list):
        kept, seen = [], set()
        for entry in annotations:
            for target in targets(entry.get("part") if isinstance(entry, dict) else None):
                if target not in seen:
                    seen.add(target)
                    kept.append({**entry, "part": target})
        model.extras["vrm_first_person"] = kept
    tracks = model.extras.get("morph_animations")
    if isinstance(tracks, list):
        kept_tracks, claimed = [], set()
        for track in tracks:
            if not isinstance(track, dict):
                continue
            values = track.get("values")
            width = int(values.shape[-1]) if isinstance(values, torch.Tensor) and values.dim() else -1
            moved = []
            for part in track.get("parts", []) or []:
                for target in targets(part):
                    key = (str(track.get("animation", "")), target)
                    if key in claimed or len(model.parts[target].morphs) != width:
                        continue
                    claimed.add(key)
                    moved.append(target)
            if moved:
                kept_tracks.append({**track, "parts": moved})
        model.extras["morph_animations"] = kept_tracks


def merge_by_material(model: Model3D, chosen: list[int]) -> int:
    """Join the chosen parts that share one material and are all skinned or all static.

    Args:
        model: The model, changed in place.
        chosen: Indices of the parts that may be joined.

    Returns:
        How many parts the model lost.
    """
    groups: dict[tuple[int, bool], list[int]] = {}
    chosen_set = set(chosen)
    for index, part in enumerate(model.parts):
        if index in chosen_set:
            skinned = part.weights is not None and part.joints is not None
            groups.setdefault((id(part.material), skinned), []).append(index)
    leader = {members[0]: members for members in groups.values() if len(members) > 1}
    absorbed = {index for members in leader.values() for index in members[1:]}
    if not absorbed:
        return 0
    parts, source = [], []
    for index, part in enumerate(model.parts):
        if index in absorbed:
            continue
        if index in leader:
            members = leader[index]
            parts.append(merge_parts([model.parts[member] for member in members]))
            source.append(members)
        else:
            parts.append(part)
            source.append([index])
    reindex_parts(model, parts, source)
    return len(absorbed)


def split_shells(part: Part, max_shells: int = 256) -> list[Part]:
    """A part divided into its connected pieces, largest first.

    Pieces meet where their triangles share a corner position, so UV seams do not divide them.

    Args:
        part: The part to divide.
        max_shells: Most parts returned; the smallest pieces past it share the last part.

    Returns:
        The pieces, named ``<name>_1``, ``<name>_2`` and so on, ``<name>_rest`` for the shared
        last part; the part itself when it is one piece.
    """
    limit = max(1, min(int(max_shells), MAX_SHELLS))
    if part.triangle_count == 0 or limit == 1:
        return [part]
    _, welded, _ = topology.weld(part.vertices.float(), part.faces.long(), tolerance=0)
    labels = topology.face_components(welded, int(welded.max()) + 1)
    count = int(labels.max()) + 1
    if count == 1:
        return [part]
    sizes = torch.bincount(labels, minlength=count)
    order = torch.argsort(-sizes, stable=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(count)
    shell = rank[labels].clamp(max=limit - 1)
    faces_sorted = torch.argsort(shell, stable=True)
    ends = torch.cumsum(torch.bincount(shell, minlength=min(count, limit)), 0).tolist()
    pieces, start = [], 0
    for number, end in enumerate(ends):
        shared = number == limit - 1 and count > limit
        name = f"{part.name}_rest" if shared else f"{part.name}_{number + 1}"
        pieces.append(subset_part(part, faces_sorted[start:end], name))
        start = end
    return pieces


def bake_rest(model: Model3D) -> int:
    """Bake skinned parts into the pose the skeleton stands in, so ``rest @ inverse_bind`` is identity.

    Args:
        model: The model, changed in place.

    Returns:
        How many parts were moved; 0 when the model already stood at its bind pose.
    """
    skeleton = model.skeleton
    if skeleton is None or not skeleton.count:
        return 0
    skinning = skeleton.rest.double() @ skeleton.inverse_bind.double()
    eye = torch.eye(4, dtype=torch.float64)
    if float((skinning - eye).abs().max()) <= BIND_TOLERANCE:
        return 0
    moved = 0
    for part in model.parts:
        if part.joints is None or part.weights is None or not part.vertex_count:
            continue
        joints = part.joints.long().clamp(0, skeleton.count - 1)
        weights = part.weights.double()
        blended = torch.zeros(part.vertex_count, 4, 4, dtype=torch.float64)
        for column in range(joints.shape[1]):
            blended += skinning[joints[:, column]] * weights[:, column, None, None]
        linear = blended[:, :3, :3]
        part.vertices = (torch.einsum("nij,nj->ni", linear, part.vertices.double()) + blended[:, :3, 3]).float()
        if part.normals is not None:
            part.normals = torch.nn.functional.normalize(
                torch.einsum("nij,nj->ni", linear, part.normals.double()), dim=-1).float()
        if part.tangents is not None:
            turned = torch.nn.functional.normalize(
                torch.einsum("nij,nj->ni", linear, part.tangents[:, :3].double()), dim=-1).float()
            part.tangents = torch.cat([turned, part.tangents[:, 3:].float()], dim=1)
        for morph in part.morphs.values():
            morph.positions = torch.einsum("nij,nj->ni", linear, morph.positions.double()).float()
            if morph.normals is not None:
                morph.normals = torch.einsum("nij,nj->ni", linear, morph.normals.double()).float()
        moved += 1
    skeleton.inverse_bind = torch.linalg.inv(skeleton.rest.double()).float()
    return moved


def merge_skeletons(base: Skeleton, other: Skeleton) -> tuple[Skeleton, list[int]]:
    """One skeleton holding a base skeleton's joints and another's, joined by joint name.

    Args:
        base: Joints kept at their indices, with their rest and inverse bind.
        other: Joints whose name the base lacks are appended under their mapped parent.

    Returns:
        ``(skeleton, remap)``: the joined skeleton and, for each joint of ``other``, its index there.
    """
    names = list(base.names)
    index_of: dict[str, int] = {}
    for index, name in enumerate(names):
        index_of.setdefault(name, index)
    parents = [int(value) for value in base.parents]
    rests = [base.rest.float()]
    binds = [base.inverse_bind.float()]
    remap: list[int] = []
    appended: list[int] = []
    for joint, name in enumerate(other.names):
        if name in index_of:
            remap.append(index_of[name])
            continue
        parent = int(other.parents[joint])
        index = len(names)
        names.append(name)
        index_of[name] = index
        parents.append(remap[parent] if 0 <= parent < len(remap) else -1)
        appended.append(joint)
        remap.append(index)
    if appended:
        rows = torch.tensor(appended, dtype=torch.int64)
        rests.append(other.rest.float()[rows])
        binds.append(other.inverse_bind.float()[rows])
    humanoid = dict(base.humanoid)
    taken = set(humanoid.values())
    for bone, joint in other.humanoid.items():
        if isinstance(joint, int) and 0 <= joint < len(remap) and bone not in humanoid and remap[joint] not in taken:
            humanoid[bone] = remap[joint]
            taken.add(remap[joint])
    merged = Skeleton(
        names=names,
        parents=torch.tensor(parents, dtype=torch.int64),
        rest=torch.cat(rests, dim=0),
        inverse_bind=torch.cat(binds, dim=0),
        humanoid=humanoid,
    )
    return merged, remap


def _remap_joint(value, remap: list[int]):
    """A joint index carried through ``remap``; anything else unchanged."""
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value < len(remap):
        return remap[value]
    return value


def _add_springs(model: Model3D, other: Model3D, remap: list[int]) -> int:
    """Append another model's spring chains and colliders; returns chains added."""
    if other.springs is None or not (other.springs.chains or other.springs.colliders):
        return 0
    springs = model.springs if model.springs is not None else Springs()
    base_chains = len(springs.chains)
    collider_offset = len(springs.colliders)
    group_offset = len(springs.collider_groups)
    for collider in other.springs.colliders:
        added = copy.deepcopy(collider)
        added.joint = _remap_joint(added.joint, remap)
        springs.colliders.append(added)
    for index, group in enumerate(other.springs.collider_groups):
        springs.collider_groups.append([member + collider_offset for member in group])
        names = other.springs.group_names
        springs.group_names.append(names[index] if index < len(names) else f"colliders_{group_offset + index}")
    for chain in other.springs.chains:
        added = copy.deepcopy(chain)
        added.joints = [_remap_joint(joint, remap) for joint in added.joints]
        added.collider_groups = [group + group_offset for group in added.collider_groups]
        springs.chains.append(added)
    model.springs = springs
    base_settings = model.extras.get("vrm_spring_settings")
    other_settings = other.extras.get("vrm_spring_settings")
    if isinstance(base_settings, list) or isinstance(other_settings, list):
        blank = {"center": -1, "joints": None}
        kept = list(base_settings) if isinstance(base_settings, list) and len(base_settings) == base_chains \
            else [dict(blank) for _ in range(base_chains)]
        added_count = len(other.springs.chains)
        if isinstance(other_settings, list) and len(other_settings) == added_count:
            for entry in other_settings:
                entry = dict(entry) if isinstance(entry, dict) else dict(blank)
                entry["center"] = _remap_joint(entry.get("center", -1), remap)
                kept.append(entry)
        else:
            kept += [dict(blank) for _ in range(added_count)]
        model.extras["vrm_spring_settings"] = kept
    return len(other.springs.chains)


def _add_animations(model: Model3D, other: Model3D, remap: list[int]) -> int:
    """Append another model's clips, joining channels into a clip of the same name; returns channels added."""
    by_name = {clip.name: clip for clip in model.animations}
    added = 0
    for clip in other.animations:
        target = by_name.get(clip.name)
        if target is None:
            target = Animation(clip.name)
            model.animations.append(target)
            by_name[clip.name] = target
        present = {(channel.joint, channel.path) for channel in target.channels}
        for channel in clip.channels:
            moved = copy.deepcopy(channel)
            moved.joint = _remap_joint(moved.joint, remap)
            if (moved.joint, moved.path) in present:
                continue
            present.add((moved.joint, moved.path))
            target.channels.append(moved)
            added += 1
    return added


def add_model(model: Model3D, other: Model3D) -> dict[str, int]:
    """Add another model's parts to a model, joining skeletons by joint name.

    Args:
        model: The model added to, changed in place. Without a skeleton it takes the other's.
        other: The model whose parts, clips, expressions and springs are added; changed too.
            Without a skeleton its parts stay static. A model posed away from its bind pose
            is baked into that pose first.

    Returns:
        Counts: ``parts added``, ``joints shared``, ``joints added``, ``channels added``,
        ``chains added``.
    """
    offset = len(model.parts)
    counts = {"parts added": len(other.parts), "joints shared": 0, "joints added": 0,
              "channels added": 0, "chains added": 0}
    remap: list[int] = []
    if other.skeleton is not None and other.skeleton.count:
        bake_rest(other)
        if model.skeleton is not None and model.skeleton.count:
            bake_rest(model)
            before = model.skeleton.count
            model.skeleton, remap = merge_skeletons(model.skeleton, other.skeleton)
            counts["joints added"] = model.skeleton.count - before
            counts["joints shared"] = other.skeleton.count - counts["joints added"]
        else:
            model.skeleton = other.skeleton
            remap = list(range(other.skeleton.count))
            counts["joints added"] = other.skeleton.count
        lookup = torch.tensor(remap, dtype=torch.int64)
        for part in other.parts:
            if part.joints is not None:
                part.joints = lookup[part.joints.long().clamp(0, len(remap) - 1)]
        counts["channels added"] = _add_animations(model, other, remap)
        counts["chains added"] = _add_springs(model, other, remap)
        constraints = other.extras.get("vrm_node_constraints")
        if isinstance(constraints, list) and constraints:
            moved = []
            for entry in copy.deepcopy(constraints):
                if not isinstance(entry, dict):
                    continue
                entry["joint"] = _remap_joint(entry.get("joint"), remap)
                for body in (entry.get("constraint") or {}).values():
                    if isinstance(body, dict):
                        body["source"] = _remap_joint(body.get("source"), remap)
                moved.append(entry)
            model.extras["vrm_node_constraints"] = list(model.extras.get("vrm_node_constraints") or []) + moved
    else:
        for part in other.parts:
            part.joints = part.weights = None
    for part in other.parts:
        model.parts.append(part)
    for name, expression in other.expressions.items():
        binds = [(part + offset, morph, weight) for part, morph, weight in expression.binds]
        if name in model.expressions:
            model.expressions[name].binds += binds
        else:
            expression.binds = binds
            model.expressions[name] = expression
    annotations = other.extras.get("vrm_first_person")
    if isinstance(annotations, list) and annotations:
        shifted = [{**entry, "part": entry["part"] + offset} for entry in annotations
                   if isinstance(entry, dict) and isinstance(entry.get("part"), int)]
        model.extras["vrm_first_person"] = list(model.extras.get("vrm_first_person") or []) + shifted
    tracks = other.extras.get("morph_animations")
    if isinstance(tracks, list) and tracks:
        shifted = [{**track, "parts": [part + offset for part in track.get("parts", []) or []
                                        if isinstance(part, int)]}
                   for track in tracks if isinstance(track, dict)]
        model.extras["morph_animations"] = list(model.extras.get("morph_animations") or []) + shifted
    return counts

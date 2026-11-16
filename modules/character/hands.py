"""Finger bones fitted to a character's hands from cross-sections cut along each hand.

Each section is read as a row of lobes, one per finger, split where it thins between two fingers.
Lengths are model units.
"""

from __future__ import annotations

__all__ = ["FINGERS", "Hand", "finger_bones", "fit_hand"]

import math
from dataclasses import dataclass, field

import torch

from ..mesh import sections

#: Fingers from the thumb outwards, as VRM names them.
FINGERS = ("Thumb", "Index", "Middle", "Ring", "Little")

#: Cross-sections cut from `FIRST_LEVEL` of the hand's length to its fingertips.
HAND_LEVELS = 96
FIRST_LEVEL = 0.1

#: Furthest the fingertips are measured past the end of the hand's line, as a share of its length.
LONGEST_REACH = 1.6

#: A loop enclosing less than this share of the hand's length squared is a detail, not a finger.
SMALLEST_LOOP = 0.0015

#: Samples across a section per hand length when its thickness is profiled.
PROFILE_SAMPLES = 160

#: A section thinner than this share of both neighbouring peaks separates two fingers.
DIP_SHARE = 0.8

#: Lobes of one finger on neighbouring sections overlap by at least this share of the narrower.
TRACK_OVERLAP = 0.3

#: A finger whose lobe widens past this multiple of its usual width has joined the palm.
TRACK_GROWTH = 1.7

#: Nearest the wrist a fingertip may be, as a share of the hand's length.
TIP_SHARE = 0.35

#: Shortest thumb standing apart from the palm, as a share of the hand's length.
SHORTEST_THUMB = 0.08

#: Shortest finger standing apart from its neighbours, as a share of the hand's length.
SHORTEST_FINGER = 0.2

#: Span of the hand's length a finger standing apart may meet the palm in.
KNUCKLE_ZONE = (0.35, 0.78)

#: Span of the hand's length the finger knuckles are kept within.
KNUCKLE_RANGE = (0.39, 0.57)

#: A thumb's tip is at least this share of the hand's length nearer the wrist than the fingertips.
THUMB_LEAD = 0.15

#: Knuckle line of fingers that never part, as a share of the hand's length.
KNUCKLE_SHARE = 0.47

#: Knuckle behind the base of a free finger, as a share of the free finger's length.
KNUCKLE_INSET = 0.3

#: Intermediate and distal joints as shares of the way along a finger from knuckle to tip.
INTERMEDIATE_AT, DISTAL_AT = 0.35, 0.6

#: Span of a finger's own knuckle-to-tip length its bending joints are kept within, from either end.
JOINT_REACH = 0.85

#: Thumb's proximal joint behind its base as a share of its free length; distal joint as a share of
#: the way from proximal joint to tip; metacarpal joint behind the proximal joint as a share of the
#: proximal bone.
THUMB_INSET, THUMB_DISTAL_AT, THUMB_METACARPAL = 0.1, 0.5, 1.3

#: Nearest the wrist a joint may sit, as a share of the hand's length.
WRIST_CLEARANCE = 0.05


def finger_bones(prefix: str) -> list[tuple[str, str]]:
    """VRM finger bones of one hand with their parents, parents first.

    Args:
        prefix: ``"left"`` or ``"right"``.

    Returns:
        ``(bone, parent)`` pairs, thumb first, three per finger.
    """
    rows = []
    for finger in FINGERS:
        parent = f"{prefix}Hand"
        segments = ("Metacarpal", "Proximal", "Distal") if finger == "Thumb" else ("Proximal", "Intermediate", "Distal")
        for segment in segments:
            name = f"{prefix}{finger}{segment}"
            rows.append((name, parent))
            parent = name
    return rows


@dataclass
class Hand:
    """Finger joints fitted to one hand.

    Attributes:
        heads: VRM bone name against its ``(3,)`` joint head.
        tips: Distal bone name against its ``(3,)`` fingertip.
        knuckles: ``(3,)`` middle of the finger knuckles, the end of the hand bone, or None.
        apart: Fingers found apart from their neighbours.
        joined: Fingers placed across a part of the hand where they are joined.
        thumb: Whether a thumb was found.
    """

    heads: dict[str, torch.Tensor] = field(default_factory=dict)
    tips: dict[str, torch.Tensor] = field(default_factory=dict)
    knuckles: torch.Tensor | None = None
    apart: int = 0
    joined: int = 0
    thumb: bool = False

    @property
    def fingers(self) -> int:
        """Fingers fitted, the thumb included."""
        return self.apart + self.joined + int(self.thumb)


@dataclass
class _Lobe:
    """One finger-wide lobe of a cross-section: its level, extent across the hand and centre."""

    s: float
    low: float
    high: float
    centre: torch.Tensor
    thickness: float
    points: torch.Tensor

    @property
    def width(self) -> float:
        """Extent across the hand."""
        return self.high - self.low


@dataclass
class _Track:
    """Lobes followed from one level to the next, fingertip first.

    Attributes:
        lobes: The lobe on each level, fingertip first.
        kinds: Per lobe, ``"own"`` where the digit stands apart, ``"shared"`` where the lobe is the
            digit's share of one it has joined.
    """

    lobes: list[_Lobe]
    kinds: list[str]

    def widths(self) -> list[float]:
        """Width of each lobe the digit holds alone."""
        return [lobe.width for lobe, kind in zip(self.lobes, self.kinds) if kind == "own"]

    def usual_width(self) -> float:
        """Median width of the lobes the digit holds alone."""
        widths = sorted(self.widths())
        return widths[len(widths) // 2]

    @property
    def last_own(self) -> int:
        """Index of the last lobe the digit holds alone."""
        return max(index for index, kind in enumerate(self.kinds) if kind == "own")

    @property
    def free(self) -> float:
        """Length the digit stands apart, from its tip to its last own lobe."""
        return self.lobes[0].s - self.lobes[self.last_own].s

    @property
    def own_count(self) -> int:
        """Lobes the digit holds alone."""
        return sum(kind == "own" for kind in self.kinds)


def _frame(found: sections.Profile, levels: list[int], axis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(across, normal)``: the palm's widest direction and its thickness direction, both unit."""
    gathered = []
    for level in levels:
        loops = [loop for loop in found.loops(level) if loop.closed]
        if loops:
            gathered.append(max(loops, key=lambda loop: abs(loop.area)).points.float())
    basis = found.basis.float()
    across = basis[0]
    if gathered:
        flat = torch.cat(gathered) @ basis.T
        flat = flat - flat.mean(dim=0)
        _, vectors = torch.linalg.eigh(flat.T @ flat)
        across = vectors[:, 1] @ basis
    across = across / across.norm()
    normal = torch.linalg.cross(axis, across)
    up, ahead = torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0])
    reference = up if abs(float(normal @ up)) > 0.3 else ahead
    if float(normal @ reference) < 0:
        normal, across = -normal, -across
    return across, normal / normal.norm()


def _lobes(loop: sections.Loop, s: float, axis: torch.Tensor, across: torch.Tensor, normal: torch.Tensor,
           step: float) -> list[_Lobe]:
    """A section's lobes, split where its thickness dips between two peaks."""
    points = loop.points.float()
    pw, pn = points @ across, points @ normal
    low, high = float(pw.min()), float(pw.max())
    count = max(int((high - low) / step), 4)
    ws = torch.linspace(low, high, count + 2)[1:-1]
    aw, bw = pw, torch.roll(pw, -1)
    an, bn = pn, torch.roll(pn, -1)
    crosses = (ws[:, None] >= torch.minimum(aw, bw)[None]) & (ws[:, None] < torch.maximum(aw, bw)[None])
    span = torch.where((bw - aw).abs() > 1e-12, bw - aw, torch.full_like(bw, 1e-12))
    at = an[None] + (ws[:, None] - aw[None]) / span[None] * (bn - an)[None]
    top = torch.where(crosses, at, torch.full_like(at, -math.inf)).max(dim=1).values
    bottom = torch.where(crosses, at, torch.full_like(at, math.inf)).min(dim=1).values
    valid = torch.isfinite(top) & torch.isfinite(bottom)
    thick = torch.where(valid, top - bottom, torch.zeros_like(top)).clamp_min(0)
    middle = torch.where(valid, (top + bottom) / 2, torch.zeros_like(top))
    smooth = thick.clone()
    smooth[1:-1] = (thick[:-2] + thick[1:-1] + thick[2:]) / 3
    peaks = [i for i in range(count)
             if (i == 0 or smooth[i] >= smooth[i - 1]) and (i == count - 1 or smooth[i] > smooth[i + 1])]
    cuts = []
    for left, right in zip(peaks, peaks[1:]):
        dip = left + int(torch.argmin(smooth[left:right + 1]))
        if float(smooth[dip]) < DIP_SHARE * min(float(smooth[left]), float(smooth[right])):
            cuts.append(dip)
    bounds = [0] + cuts + [count]
    lobes = []
    for first, last in zip(bounds, bounds[1:]):
        share = thick[first:last]
        total = float(share.sum())
        if last <= first or total <= 0:
            continue
        w_centre = float((ws[first:last] * share).sum()) / total
        n_centre = float((middle[first:last] * share).sum()) / total
        lo = low if first == 0 else float(ws[first])
        hi = high if last == count else float(ws[last - 1])
        held = points[(pw >= lo) & (pw <= hi)]
        centre = s * axis + w_centre * across + n_centre * normal
        lobes.append(_Lobe(s, lo, hi, centre, float(share.max()), held))
    return lobes


def _share(lobe: _Lobe, low: float, high: float, axis: torch.Tensor, across: torch.Tensor,
           normal: torch.Tensor) -> _Lobe:
    """The part of a lobe between ``low`` and ``high`` across the hand, as a lobe of its own."""
    low, high = max(low, lobe.low), min(high, lobe.high)
    if high <= low:
        low = high = 0.5 * (low + high)
    pw = lobe.points @ across
    held = lobe.points[(pw >= low) & (pw <= high)]
    if held.shape[0] >= 2:
        pn = held @ normal
        n = 0.5 * (float(pn.max()) + float(pn.min()))
    else:
        n = float(lobe.centre @ normal)
    centre = lobe.s * axis + 0.5 * (low + high) * across + n * normal
    return _Lobe(lobe.s, low, high, centre, lobe.thickness, held)


def _track(levels: list[list[_Lobe]], axis: torch.Tensor, across: torch.Tensor, normal: torch.Tensor) -> list[_Track]:
    """Follow lobes from the fingertips to the wrist, sharing a joined lobe among the digits reaching it."""
    tracks: list[_Track] = []
    active: list[int] = []
    for lobes in reversed(levels):
        claims: dict[int, list[int]] = {index: [] for index in range(len(lobes))}
        for number in active:
            last = tracks[number].lobes[-1]
            best, best_overlap = None, 0.0
            for index, lobe in enumerate(lobes):
                overlap = min(last.high, lobe.high) - max(last.low, lobe.low)
                drift = abs(float((lobe.centre - last.centre) @ normal))
                if overlap > TRACK_OVERLAP * min(max(last.width, 1e-9), lobe.width) and overlap > best_overlap \
                        and drift < 0.75 * max(last.thickness, lobe.thickness):
                    best, best_overlap = index, overlap
            if best is not None:
                claims[best].append(number)
        following = []
        for index, lobe in enumerate(lobes):
            owners = claims[index]
            if not owners:
                tracks.append(_Track([lobe], ["own"]))
                following.append(len(tracks) - 1)
                continue
            if len(owners) == 1:
                track = tracks[owners[0]]
                usual = track.usual_width()
                if lobe.width <= TRACK_GROWTH * usual:
                    track.lobes.append(lobe)
                    track.kinds.append("own")
                else:
                    middle = float(track.lobes[-1].centre @ across)
                    track.lobes.append(_share(lobe, middle - 0.5 * usual, middle + 0.5 * usual, axis, across, normal))
                    track.kinds.append("shared")
                following.append(owners[0])
                continue
            owners.sort(key=lambda number: float(tracks[number].lobes[-1].centre @ across))
            widths = [tracks[number].usual_width() for number in owners]
            scale = lobe.width / max(sum(widths), 1e-12)
            low = lobe.low
            for number, width in zip(owners, widths):
                tracks[number].lobes.append(_share(lobe, low, low + width * scale, axis, across, normal))
                tracks[number].kinds.append("shared")
                low += width * scale
                following.append(number)
        active = following
    return tracks


def _tip(points: torch.Tensor, base: torch.Tensor, last: torch.Tensor, radius: float) -> torch.Tensor:
    """Far end of a digit: the point of its surface within ``radius`` of its line reaching furthest."""
    direction = last - base
    length = float(direction.norm())
    if length <= 1e-9:
        return last.clone()
    direction = direction / length
    offset = points - last
    along = offset @ direction
    radial = (offset - along[:, None] * direction).norm(dim=1)
    near = (radial < radius) & (along > -radius)
    reach = float(along[near].max()) if bool(near.any()) else 0.0
    return last + direction * max(reach, 0.0)


def _along(path: list[torch.Tensor], share: float) -> torch.Tensor:
    """The point a share of the way along a polyline."""
    steps = [float((b - a).norm()) for a, b in zip(path, path[1:])]
    total = sum(steps)
    if total <= 0:
        return path[0].clone()
    goal = share * total
    for (a, b), step in zip(zip(path, path[1:]), steps):
        if goal <= step and step > 0:
            return a + (b - a) * (goal / step)
        goal -= step
    return path[-1].clone()


def _at_level(path: list[torch.Tensor], axis: torch.Tensor, level: float) -> torch.Tensor:
    """The first point of a polyline at ``level`` along ``axis``, or its last point when it stops short."""
    for a, b in zip(path, path[1:]):
        low, high = float(a @ axis), float(b @ axis)
        if low <= level <= high and high > low:
            return a + (b - a) * ((level - low) / (high - low))
    return path[-1].clone()


def _clear(point: torch.Tensor, wrist: torch.Tensor, axis: torch.Tensor, length: float) -> torch.Tensor:
    """A joint moved out along the hand's line to at least the wrist's clearance."""
    along = float((point - wrist) @ axis)
    least = WRIST_CLEARANCE * length
    return point + axis * (least - along) if along < least else point


def _knuckle_in_reach(path: list[torch.Tensor], wrist: torch.Tensor, axis: torch.Tensor, length: float) -> None:
    """Slide a finger's knuckle along its first bone so it lies within `KNUCKLE_RANGE` of the hand."""
    along = float((path[0] - wrist) @ axis) / length
    goal = min(max(along, KNUCKLE_RANGE[0]), KNUCKLE_RANGE[1])
    if goal == along:
        return
    direction = path[1] - path[0]
    step = float(direction @ axis)
    if abs(step) < 1e-9:
        direction, step = axis, 1.0
    path[0] = path[0] + direction * ((goal - along) * length / step)


def _path(centres: list[torch.Tensor], tip: torch.Tensor, inset: float) -> list[torch.Tensor]:
    """Knuckle-to-tip polyline of a digit from its centres, base first, the knuckle set in by ``inset``."""
    direction = tip - centres[0]
    free = float(direction.norm())
    knuckle = centres[0] - direction / max(free, 1e-9) * (inset * free)
    return [knuckle] + centres + [tip]


def _bands(rows: list[tuple[float, torch.Tensor]], low: float, high: float, parts: int, axis: torch.Tensor,
           across: torch.Tensor, normal: torch.Tensor) -> list[list[torch.Tensor]]:
    """Centres of ``parts`` joined fingers lying side by side between ``low`` and ``high`` across the hand.

    Args:
        rows: ``(level, section points)`` from the fingertips inwards.
        low: Near edge of the first band.
        high: Far edge of the last band.
        parts: Bands to cut.
        axis: The hand's line.
        across: Unit direction across the hand.
        normal: Unit direction through the hand's thickness.

    Returns:
        Per band, its centre on each level it reaches, base first.
    """
    band = (high - low) / parts
    out: list[list[torch.Tensor]] = [[] for _ in range(parts)]
    for s, points in rows:
        if points.numel() == 0:
            continue
        pw, pn = points @ across, points @ normal
        for number in range(parts):
            first = low + number * band
            held = (pw >= first) & (pw <= first + band)
            if int(held.sum()) < 2:
                continue
            n = 0.5 * (float(pn[held].max()) + float(pn[held].min()))
            out[number].append(s * axis + (first + 0.5 * band) * across + n * normal)
    return [list(reversed(centres)) for centres in out]


def fit_hand(vertices: torch.Tensor, faces: torch.Tensor, wrist: torch.Tensor, tip: torch.Tensor,
             prefix: str) -> Hand:
    """Fit the VRM finger joints of one hand.

    Args:
        vertices: ``(V, 3)`` positions of the hand's surface, from the wrist to the fingertips.
        faces: ``(F, 3)`` its triangles.
        wrist: ``(3,)`` wrist joint.
        tip: ``(3,)`` end of the hand's line; the line runs on to the furthest fingertip.
        prefix: ``"left"`` or ``"right"``.

    Returns:
        The `Hand`; no heads when no finger could be told from the palm.
    """
    hand = Hand()
    wrist, tip = wrist.float(), tip.float()
    axis = tip - wrist
    length = float(axis.norm())
    if length <= 0 or faces.shape[0] == 0:
        return hand
    axis = axis / length
    reach = float(((vertices[faces.reshape(-1).unique()].float() - wrist) @ axis).max())
    length = min(max(length, reach), LONGEST_REACH * length)
    base = float(wrist @ axis)
    shares = torch.linspace(FIRST_LEVEL, 1.0, HAND_LEVELS)
    found = sections.profile(vertices.float(), faces, axis=axis, levels=base + shares * length)
    palm_levels = [level for level, share in enumerate(shares.tolist()) if 0.2 <= share <= 0.45]
    across, normal = _frame(found, palm_levels, axis)
    plane = torch.stack([across, normal])
    levels: list[list[_Lobe]] = []
    for level in range(found.level_count):
        s = float(found.heights[level])
        kept: list[tuple[sections.Loop, torch.Tensor, torch.Tensor]] = []
        for loop in found.loops(level):
            if not loop.closed or abs(loop.area) <= SMALLEST_LOOP * length * length:
                continue
            flat = loop.points.float() @ plane.T
            low, high = flat.min(dim=0).values, flat.max(dim=0).values
            # A loop inside another's extent, such as a hand inside a glove, is left out.
            if not any(bool((low >= outer_low).all() and (high <= outer_high).all()) for _, outer_low, outer_high in kept):
                kept.append((loop, low, high))
        levels.append([lobe for loop, _, _ in kept
                       for lobe in _lobes(loop, s, axis, across, normal, length / PROFILE_SAMPLES)])
    tracks = _track(levels, axis, across, normal)
    points = vertices[faces.reshape(-1).unique()].float()
    digits = [track for track in tracks if track.lobes[0].s - base >= TIP_SHARE * length]
    thumb = None
    if len(digits) >= 2:
        lowest = min(digits, key=lambda track: track.lobes[0].s)
        tips = sorted(track.lobes[0].s for track in digits if track is not lowest)
        standing = lowest.lobes[0].s - lowest.lobes[lowest.last_own].s
        if tips[len(tips) // 2] - lowest.lobes[0].s > THUMB_LEAD * length and standing >= SHORTEST_THUMB * length:
            thumb = lowest
    fingers = [track for track in digits if track is not thumb]
    if len(fingers) > 4:
        fingers = sorted(fingers, key=lambda track: -track.own_count)[:4]
    apart = {id(track) for track in fingers if track.free >= SHORTEST_FINGER * length
             and KNUCKLE_ZONE[0] <= (track.lobes[track.last_own].s - base) / length <= KNUCKLE_ZONE[1]}
    knuckles_at = sorted(track.lobes[track.last_own].s - KNUCKLE_INSET * track.free
                         for track in fingers if id(track) in apart)
    share = (knuckles_at[len(knuckles_at) // 2] - base) / length if len(knuckles_at) >= 2 else KNUCKLE_SHARE
    knuckle_level = base + min(max(share, KNUCKLE_RANGE[0]), KNUCKLE_RANGE[1]) * length
    paths: list[tuple[float, list[torch.Tensor], bool]] = []
    if len(fingers) == 4 and len(apart) >= 3:
        for track in fingers:
            if id(track) in apart:
                lobes, inset = track.lobes[:track.last_own + 1], KNUCKLE_INSET
            else:
                lobes, inset = [lobe for lobe in track.lobes if lobe.s >= knuckle_level] or track.lobes[:1], 0.0
            centres = [lobe.centre for lobe in reversed(lobes)]
            far = _tip(points, centres[0], centres[-1], 0.75 * track.usual_width())
            paths.append((float(centres[-1] @ across), _path(centres, far, inset), id(track) in apart))
    else:
        in_thumb = {id(lobe) for lobe in thumb.lobes} if thumb is not None else set()
        rows = []
        for row in levels:
            held = [lobe.points for lobe in row if id(lobe) not in in_thumb]
            if held and row[0].s >= knuckle_level:
                rows.append((row[0].s, torch.cat(held)))
        rows.reverse()
        if rows:
            flat = rows[-1][1] @ across
            low, high = float(flat.min()), float(flat.max())
            for centres in _bands(rows, low, high, 4, axis, across, normal):
                if len(centres) >= 2:
                    far = _tip(points, centres[0], centres[-1], 0.125 * (high - low))
                    paths.append((float(centres[0] @ across), _path(centres, far, 0.0), False))
    if thumb is not None:
        lobes = thumb.lobes[:thumb.last_own + 1]
        centres = [lobe.centre for lobe in reversed(lobes)]
        far = _tip(points, centres[0], centres[-1], 0.75 * thumb.usual_width())
        path = _path(centres, far, THUMB_INSET)
        path[0] = _clear(path[0], wrist, axis, length)
        distal = _along(path, THUMB_DISTAL_AT)
        metacarpal = _clear(path[0] - (distal - path[0]) * THUMB_METACARPAL, wrist, axis, length)
        hand.heads.update({f"{prefix}ThumbMetacarpal": metacarpal, f"{prefix}ThumbProximal": path[0],
                           f"{prefix}ThumbDistal": distal})
        hand.tips[f"{prefix}ThumbDistal"] = far
        hand.thumb = True
    if not paths:
        return hand
    if thumb is not None:
        middle = sum(across_at for across_at, _, _ in paths) / len(paths)
        sign = 1.0 if float(thumb.lobes[thumb.last_own].centre @ across) >= middle else -1.0
    else:
        up, ahead = torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0])
        reference = ahead if abs(float(across @ ahead)) >= abs(float(across @ up)) else up
        sign = 1.0 if float(across @ reference) >= 0 else -1.0
    paths.sort(key=lambda item: -sign * item[0])
    for _, path, _ in paths[:4]:
        _knuckle_in_reach(path, wrist, axis, length)
    # The fingers bend on lines shared across the hand, at the shares of their middle length.
    first = sorted(float(path[0] @ axis) for _, path, _ in paths[:4])[min(len(paths), 4) // 2]
    last = sorted(float(path[-1] @ axis) for _, path, _ in paths[:4])[min(len(paths), 4) // 2]
    knuckles = []
    for finger, (_, path, separate) in zip(FINGERS[1:], paths[:4]):
        hand.heads[f"{prefix}{finger}Proximal"] = path[0]
        own = float(path[-1] @ axis) - float(path[0] @ axis)
        for segment, share in (("Intermediate", INTERMEDIATE_AT), ("Distal", DISTAL_AT)):
            level = min(max(first + share * (last - first), float(path[0] @ axis) + (1 - JOINT_REACH) * own),
                        float(path[0] @ axis) + JOINT_REACH * own)
            hand.heads[f"{prefix}{finger}{segment}"] = _at_level(path, axis, level)
        hand.tips[f"{prefix}{finger}Distal"] = path[-1]
        knuckles.append(path[0])
        if separate:
            hand.apart += 1
        else:
            hand.joined += 1
    hand.knuckles = torch.stack(knuckles).mean(dim=0)
    return hand

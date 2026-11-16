"""Eye joints placed in a character's head, and blink shape keys closing its upper lids.

Painted eyes are found by colour in a picture of the face, eyeball parts by shape. Model units, Y up,
facing +Z.
"""

from __future__ import annotations

__all__ = ["BLINK_SHAPES", "Eyes", "Opening", "blink_shapes", "find_eyes"]

import math
from dataclasses import dataclass, field

import torch

from ..mesh import raster, sections, topology
from ..model3d.scene import MorphTarget

#: Shape keys a blink is made of, by side.
BLINK_SHAPES = {"left": "blinkLeft", "right": "blinkRight"}

#: Pixels across the picture of the head.
FRONT_PIXELS = 384

#: Lab distance from the skin's colour, lightness counted at half, past which a pixel is not skin.
NOT_SKIN = 22.0

#: Lightness above the skin, and share of its colourfulness below, that reads as the white of an eye;
#: colourfulness under `SCLERA_CHROMA` always reads as white.
SCLERA_LIGHTER, SCLERA_GREY, SCLERA_CHROMA = 5.0, 0.6, 10.0

#: Lightness below the skin that reads as an iris, pupil or lash line.
IRIS_DARKER = 25.0

#: Least share of an eye's pixels that are white of the eye, and that are iris or lash.
SCLERA_SHARE, IRIS_SHARE = 0.03, 0.05

#: Eye joints from face proportions, as shares of the face's half width: out from the middle and up
#: from the nose tip.
EYE_OUT, EYE_UP = 0.42, 0.55

#: Narrowest face, as a share of the height from the neck joint to the top of the head.
SMALLEST_FACE = 0.12

#: Distance either side of the nose tip the face behind it is measured at, and least depth the tip
#: stands in front of it there, as shares of that height.
NOSE_REACH, NOSE_STANDS = 0.06, 0.02

#: Eyeball turning centre behind the surface, as a share of the distance between the eyes.
EYE_DEPTH = 0.27

#: An eyeball part fits a sphere this closely, as a share of its radius, and its radius is within this
#: span of shares of the distance between the eyes.
EYEBALL_FIT, EYEBALL_SIZE = 0.08, (0.1, 0.45)

#: Lash line thickness: this quantile of the dark runs down from the eye's top, and at most this share
#: of the eye's height in each column.
LASH_QUANTILE, LASH_LIMIT = 0.1, 0.35

#: Columns past each corner of an eye, as a share of its width, over which the closing eases out.
CORNER_EASE = 0.15

#: Pixels below the painted eye's lowest pixel the lower lid's edge is taken at.
EDGE_MARGIN = 3.0

#: How far the closed upper lid laps over the lower lid at the eye's middle, as a share of its height.
LID_OVERLAP = 0.06

#: Spread of the folded eye under the closed lid, as a share of the lid's turn.
FOLD_SPREAD = 0.03

#: A painted eye's surface is read as an eyeball when a sphere fits it with a radius in this span of
#: shares of the eye's height.
EYEBALL_BULGE = (0.4, 2.0)

#: Most skin above the eye that moves down with the lid, as a share of the eye's height; the least.
LID_HEIGHT, MIN_LID = 0.9, 0.25

#: Share of the way from the lid to a brow or other mark above it that moves with the lid.
BROW_GAP = 0.75

#: Gap kept between a closed lid and the eye behind it, as a share of the eye's height.
LID_GAP = 0.1

#: Depth behind the face's front within which a vertex moves with the lids, as a share of the eye's
#: height; it fades out over the same depth again.
LID_DEPTH = 0.6


@dataclass
class _Front:
    """A picture of the head looking along -Z: the nearest surface's depth, colour and part per pixel."""

    z: torch.Tensor
    colour: torch.Tensor
    part: torch.Tensor
    left: float
    top: float
    pixel: float

    def xs(self) -> torch.Tensor:
        """X of each column's centre."""
        return self.left + (torch.arange(self.z.shape[1], dtype=torch.float64) + 0.5) * self.pixel

    def ys(self) -> torch.Tensor:
        """Y of each row's centre."""
        return self.top - (torch.arange(self.z.shape[0], dtype=torch.float64) + 0.5) * self.pixel

    def sample(self, points: torch.Tensor) -> torch.Tensor:
        """Depth of the front surface under each ``(N, 3)`` point, ``-inf`` off the picture."""
        column = ((points[:, 0].double() - self.left) / self.pixel).floor().long()
        row = ((self.top - points[:, 1].double()) / self.pixel).floor().long()
        inside = (column >= 0) & (column < self.z.shape[1]) & (row >= 0) & (row < self.z.shape[0])
        out = torch.full((points.shape[0],), -math.inf, dtype=torch.float64)
        out[inside] = self.z[row[inside], column[inside]]
        return out


@dataclass
class Eyes:
    """Eye joints placed in a head, and what the eyes are made of.

    Attributes:
        left: ``(3,)`` turning centre of the left eye, on the character's left (+X).
        right: ``(3,)`` turning centre of the right eye.
        kind: ``"eyeballs"`` for eyeball parts, ``"painted"`` for eyes painted open on the face,
            ``"proportions"`` for eyes placed from the face's proportions.
        eyeballs: ``"left"`` and ``"right"`` against ``(part index, (N,) bool vertices)`` pairs.
        openings: ``"left"`` and ``"right"`` against the eye's `Opening`; empty when no open eye
            was seen.
        front: The picture of the face the eyes were found in, or None.
        face_parts: Parts seen in that picture.
    """

    left: torch.Tensor
    right: torch.Tensor
    kind: str
    eyeballs: dict[str, list[tuple[int, torch.Tensor]]] = field(default_factory=dict)
    openings: dict[str, Opening] = field(default_factory=dict)
    front: _Front | None = None
    face_parts: list[int] = field(default_factory=list)


def _front(model, parts: list[int], neck: float, top: float, middle: float) -> _Front:
    """Render the head above ``neck`` straight on, `FRONT_PIXELS` across."""
    size = (top - neck) * 1.05
    count = FRONT_PIXELS
    pixel = size / count
    left, upper = middle - 0.5 * size, 0.5 * (top + neck) + 0.5 * size
    nearest = torch.full((count * count,), -math.inf, dtype=torch.float64)
    colour = torch.zeros(count * count, 3)
    owner = torch.full((count * count,), -1, dtype=torch.long)
    for index in parts:
        part = model.parts[index]
        vertices = part.vertices.double()
        faces = part.faces[(vertices[part.faces][:, :, 1] >= neck).all(dim=1)]
        if faces.numel() == 0:
            continue
        x = (vertices[:, 0] - left) / pixel
        y = (upper - vertices[:, 1]) / pixel
        corners = torch.stack([x, y], dim=1).float()[faces]
        texture = part.material.base_color_texture
        for piece in raster.coverage_chunks(corners, count, count):
            corner = faces[piece.face]
            depth = (vertices[:, 2][corner] * piece.barycentric.double()).sum(dim=1)
            nearest.scatter_reduce_(0, piece.pixel, depth, reduce="amax")
            win = depth >= nearest[piece.pixel]
            pixels, corner, weights = piece.pixel[win], corner[win], piece.barycentric[win]
            if texture is not None and part.uvs is not None:
                uvs = (part.uvs.float()[corner] * weights[..., None]).sum(dim=1)
                shade = raster.sample_texture(texture.image.float(), uvs, texture.wrap_s, texture.wrap_t)[:, :3]
            elif part.colors is not None:
                shade = (part.colors.float()[corner][..., :3] * weights[..., None]).sum(dim=1)
            else:
                shade = torch.tensor(part.material.base_color[:3], dtype=torch.float32).expand(pixels.shape[0], 3)
            colour[pixels] = shade
            owner[pixels] = index
    return _Front(nearest.reshape(count, count), colour.reshape(count, count, 3), owner.reshape(count, count),
                  left, upper, pixel)


def _lab(rgb: torch.Tensor) -> torch.Tensor:
    """CIE Lab of sRGB codes in ``[0, 1]``."""
    codes = rgb.double().clamp(0, 1)
    linear = torch.where(codes <= 0.04045, codes / 12.92, ((codes + 0.055) / 1.055) ** 2.4)
    matrix = torch.tensor([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]],
                          dtype=torch.float64)
    xyz = linear @ matrix.T / torch.tensor([0.9505, 1.0, 1.089], dtype=torch.float64)
    f = torch.where(xyz > 0.008856, xyz.clamp_min(0) ** (1 / 3), 7.787 * xyz + 16 / 116)
    return torch.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], dim=-1)


def _nose_tip(front: _Front, central: torch.Tensor, reach: float, stands: float) -> tuple[int, int]:
    """``(row, column)`` of the nose tip: the nearest point standing out of the face either side of it.

    Args:
        front: The picture of the head.
        central: Pixels the tip may be at.
        reach: Pixels either side the face is measured at.
        stands: Depth the tip stands in front of the face there.
    """
    width = front.z.shape[1]
    step = max(int(round(reach)), 1)
    beside = torch.full_like(front.z, -math.inf)
    beside[:, step:width - step] = torch.minimum(front.z[:, :width - 2 * step], front.z[:, 2 * step:])
    allowed = central.clone()
    first = None
    for _ in range(3):
        depth = torch.where(allowed, front.z, torch.full_like(front.z, -math.inf))
        if not bool(torch.isfinite(depth).any()):
            break
        row, column = divmod(int(depth.argmax()), width)
        first = first or (row, column)
        if float(front.z[row, column] - beside[row, column]) >= stands:
            return row, column
        allowed[:row + 1] = False
    return first


def _components(mask: torch.Tensor) -> torch.Tensor:
    """Label of each pixel's 4-connected region of ``mask``, ``-1`` off it."""
    height, width = mask.shape
    big = height * width
    labels = torch.where(mask, torch.arange(big).reshape(height, width), torch.full((height, width), big))
    for _ in range(big):
        padded = torch.nn.functional.pad(labels[None, None].float(), (1, 1, 1, 1), value=float(big))[0, 0].long()
        neighbours = torch.stack([padded[1:-1, :-2], padded[1:-1, 2:], padded[:-2, 1:-1], padded[2:, 1:-1], labels])
        updated = torch.where(mask, neighbours.min(dim=0).values, labels)
        if torch.equal(updated, labels):
            break
        labels = updated
    return torch.where(mask, labels, torch.full_like(labels, -1))


def _not_skin(lab: torch.Tensor, skin: torch.Tensor) -> torch.Tensor:
    """Per pixel, whether its colour is further than `NOT_SKIN` from the skin's, lightness counted at half."""
    difference = lab - skin
    difference[..., 0] *= 0.5
    return difference.norm(dim=-1) > NOT_SKIN


def _painted(front: _Front, lab: torch.Tensor, skin: torch.Tensor, middle: float, nose: float,
             half_face: float) -> dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Eyes painted open, ``"left"`` and ``"right"``, each as ``(eye, marks, dark)`` pixel masks.

    Returns:
        Per side, the white and iris pixels of the eye, every other mark that is not skin, and the
        eye's dark pixels; empty when no pair of open eyes is painted.
    """
    seen = torch.isfinite(front.z)
    ys, xs = front.ys(), front.xs()
    y, x = torch.meshgrid(ys, xs, indexing="ij")
    region = seen & (y > nose + 0.1 * half_face) & (y < nose + 1.3 * half_face) \
        & ((x - middle).abs() > 0.1 * half_face) & ((x - middle).abs() < 1.25 * half_face)
    padded = torch.nn.functional.pad(region[None, None].float(), (1, 1, 1, 1))[0, 0] > 0
    edge = region & ~(padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
    edge &= (x - middle).abs() > 0.2 * half_face
    marks = seen & _not_skin(lab, skin)
    other = region & marks
    grey = max(SCLERA_GREY * float(skin[1:].norm()), SCLERA_CHROMA)
    sclera = other & (lab[..., 0] > skin[0] + SCLERA_LIGHTER) & (lab[..., 1:].norm(dim=-1) < grey)
    dark = other & (lab[..., 0] < skin[0] - IRIS_DARKER)
    labels = _components(other)
    found: dict[str, tuple[float, torch.Tensor, torch.Tensor]] = {}
    for label in labels[labels >= 0].unique().tolist():
        held = labels == label
        core = held & (sclera | dark)
        count = int(core.sum())
        whites, irises = int((held & sclera).sum()), int((held & dark).sum())
        if count < 6 or whites < SCLERA_SHARE * count or irises < IRIS_SHARE * count or bool((held & edge).any()):
            continue
        cx, cy = x[core], y[core]
        wide, tall = float(cx.max() - cx.min()), float(cy.max() - cy.min())
        if not (0.12 * half_face <= wide <= 0.9 * half_face and tall <= 0.6 * half_face):
            continue
        side = "left" if float(cx.mean()) > middle else "right"
        score = float(whites + irises)
        if side not in found or score > found[side][0]:
            found[side] = (score, core, held)
    if len(found) < 2:
        return {}
    left_y, right_y = float(y[found["left"][1]].mean()), float(y[found["right"][1]].mean())
    if abs(left_y - right_y) > 0.2 * half_face:
        return {}
    return {side: (core, marks & ~held, core & dark) for side, (_, core, held) in found.items()}


def _smooth(values: torch.Tensor) -> torch.Tensor:
    """A five-sample running mean, ends held."""
    if values.numel() < 5:
        return values
    kernel = torch.ones(1, 1, 5, dtype=torch.float64) / 5
    padded = torch.nn.functional.pad(values[None, None], (2, 2), mode="replicate")
    return torch.nn.functional.conv1d(padded, kernel)[0, 0]


@dataclass
class Opening:
    """An open eye seen from the front, as heights over its columns.

    Attributes:
        x: ``(K,)`` X of each column, corner to corner.
        top: Top of the eye, its lash line included.
        edge: Edge of the upper lid, under its lash line.
        closed: Where the closed upper lid meets the lower lid: one arc through both corners.
        bottom: Edge of the lower lid.
        roof: Highest skin above the eye that moves with the upper lid.
        ease: Share of the closing done in each column: 1 across the eye, falling to 0 past its
            corners.
    """

    x: torch.Tensor
    top: torch.Tensor
    edge: torch.Tensor
    closed: torch.Tensor
    bottom: torch.Tensor
    roof: torch.Tensor
    ease: torch.Tensor

    @property
    def height(self) -> float:
        """Tallest the eye is open."""
        return float((self.top - self.bottom).max())


def _opening(front: _Front, mask: torch.Tensor, marks: torch.Tensor, dark: torch.Tensor | None = None) -> Opening:
    """An eye's `Opening` from its pixels.

    Args:
        front: The picture of the face.
        mask: The eye's white and iris pixels.
        marks: Pixels that are not skin and not this eye; the lid skin stops short of them.
        dark: The eye's dark pixels; the run of them down from its top is the lash line. None for no
            lash line.
    """
    xs = front.xs()
    # The eye runs between the outermost columns showing its white; a painted lash wing lies beyond.
    whites = mask & ~dark if dark is not None else mask
    columns = torch.nonzero((whites if bool(whites.any()) else mask).any(dim=0)).squeeze(1)
    first, last = int(columns.min()), int(columns.max())
    xs_in = xs[first:last + 1]
    height = mask[:, first:last + 1]
    rows = torch.arange(mask.shape[0], dtype=torch.float64)[:, None].expand_as(height)
    top_row = torch.where(height, rows, torch.full_like(rows, math.inf)).min(dim=0).values
    bottom_row = torch.where(height, rows, torch.full_like(rows, -math.inf)).max(dim=0).values
    filled = torch.isfinite(top_row)
    middle_row = float(((top_row + bottom_row) / 2)[filled].mean())
    top_row = torch.where(filled, top_row, torch.full_like(top_row, middle_row))
    bottom_row = torch.where(filled, bottom_row, torch.full_like(bottom_row, middle_row))
    above = marks[:, first:last + 1] & ~height & (rows < top_row[None])
    roof_row = torch.where(above, rows, torch.full_like(rows, -math.inf)).max(dim=0).values
    tall = float((bottom_row - top_row).max() + 2) * front.pixel
    top = _smooth(front.top - top_row * front.pixel)
    bottom = _smooth(front.top - (bottom_row + 1.0 + EDGE_MARGIN) * front.pixel)
    lash = 0.0
    if dark is not None:
        # The lash line is the run of dark pixels down from the top, measured where it stops short.
        lit = ~dark[:, first:last + 1] & height & (rows > top_row[None])
        run = torch.where(lit, rows, torch.full_like(rows, math.inf)).min(dim=0).values - top_row
        run = run[torch.isfinite(run)]
        if run.numel():
            lash = float(run.quantile(LASH_QUANTILE)) * front.pixel
    edge = top - torch.minimum(torch.full_like(top, lash), LASH_LIMIT * (top - bottom).clamp_min(0))
    ends = torch.tensor([float(xs_in[0]), float(xs_in[-1])], dtype=torch.float64)
    corners = 0.5 * (top + bottom)[[0, -1]]
    line = corners[0] + (xs_in - ends[0]) / max(float(ends[1] - ends[0]), 1e-12) * (corners[1] - corners[0])
    bow = (xs_in - ends[0]) * (ends[1] - xs_in)
    sag = max(float(((line - bottom) * bow).sum() / (bow * bow).sum().clamp_min(1e-30)), 0.0)
    overlap = LID_OVERLAP * bow / max(float(bow.max()), 1e-30) * float((top - bottom).max())
    closed = torch.minimum(line - sag * bow - overlap, edge)
    roof = front.top - (roof_row + 1.0) * front.pixel
    roof = torch.where(torch.isfinite(roof), roof, top + LID_HEIGHT * tall)
    roof = torch.minimum(_smooth(roof), top + LID_HEIGHT * tall)
    roof = torch.maximum(top + BROW_GAP * (roof - top), top + MIN_LID * tall)
    # Columns past each corner carry the corner's lines on while the closing eases out.
    pad = max(2, round(CORNER_EASE * xs_in.numel()))
    step = float(xs_in[1] - xs_in[0]) if xs_in.numel() > 1 else front.pixel
    before = xs_in[0] - step * torch.arange(pad, 0, -1, dtype=torch.float64)
    after = xs_in[-1] + step * torch.arange(1, pad + 1, dtype=torch.float64)

    def padded(values: torch.Tensor) -> torch.Tensor:
        return torch.cat([values[:1].expand(pad), values, values[-1:].expand(pad)])

    fade = (1.4 - 1.4 * torch.arange(1, pad + 1, dtype=torch.float64) / (pad + 1)).clamp(0, 1)
    fade = fade * fade * (3 - 2 * fade)
    ease = torch.cat([fade.flip(0), torch.ones_like(xs_in), fade])
    return Opening(torch.cat([before, xs_in, after]), padded(top), padded(edge), padded(closed), padded(bottom),
                   padded(roof), ease)


def _eyeballs(model, parts: list[int], left: torch.Tensor, right: torch.Tensor,
              spacing: float) -> dict[str, list[tuple[int, torch.Tensor, sections.Sphere]]]:
    """Parts or loose pieces of parts shaped as balls sitting near each eye."""
    found: dict[str, list[tuple[int, torch.Tensor, sections.Sphere]]] = {"left": [], "right": []}
    eyes = torch.stack([left.float(), right.float()])
    for index in parts:
        part = model.parts[index]
        if part.vertex_count == 0 or part.triangle_count == 0:
            continue
        welded, faces, remap = topology.weld(part.vertices.float(), part.faces, tolerance=0)
        piece_of = topology.connected_components(faces, welded.shape[0])[remap]
        count = int(piece_of.max()) + 1
        sizes = torch.bincount(piece_of, minlength=count)
        centres = torch.zeros(count, 3).index_add_(0, piece_of, part.vertices.float()) / sizes.clamp_min(1)[:, None]
        distance = torch.cdist(centres, eyes)
        nearest, sides = distance.min(dim=1)
        for piece in torch.nonzero((nearest <= 0.6 * spacing) & (sizes >= 8)).squeeze(1).tolist():
            held = piece_of == piece
            side = ("left", "right")[int(sides[piece])]
            points = part.vertices[held].float()
            ball = sections.fit_sphere(points)
            if not (EYEBALL_SIZE[0] * spacing <= ball.radius <= EYEBALL_SIZE[1] * spacing):
                continue
            if ball.error > EYEBALL_FIT * ball.radius:
                continue
            found[side].append((index, held, ball))
    return found if found["left"] and found["right"] else {}


def find_eyes(model, middle: float, neck: float, top: float) -> Eyes | None:
    """Place both eyes of a head: eyeball parts first, then eyes painted open, then face proportions.

    Args:
        model: The character, standing on Y and facing +Z.
        middle: X the nose tip is looked for around, such as the head joint's.
        neck: Height of the neck joint; the head is everything above it.
        top: Height of the top of the head.

    Returns:
        The `Eyes`, or None when no face could be seen above the neck.
    """
    parts = [index for index, part in enumerate(model.parts)
             if part.vertex_count and float(part.vertices[:, 1].max()) > neck]
    if not parts or top <= neck:
        return None
    front = _front(model, parts, neck, top, middle)
    seen = torch.isfinite(front.z)
    if int(seen.sum()) < 16:
        return None
    ys, xs = front.ys(), front.xs()
    y, x = torch.meshgrid(ys, xs, indexing="ij")
    span = top - neck
    central = seen & ((x - middle).abs() < 0.12 * span) & (y > neck + 0.12 * span) & (y < neck + 0.75 * span)
    if not bool(central.any()):
        return None
    nose_row, nose_column = _nose_tip(front, central, NOSE_REACH * span / front.pixel, NOSE_STANDS * span)
    nose, middle = float(ys[nose_row]), float(xs[nose_column])
    lab = _lab(front.colour)
    bridge = seen & ((x - middle).abs() < 0.03 * span) & (y > nose + 0.02 * span) & (y < nose + 0.08 * span)
    skin = lab[bridge].median(dim=0).values if int(bridge.sum()) >= 4 else lab[nose_row, nose_column]
    # The face's half width is the run of skin through the nose tip.
    on_skin = (seen & ~_not_skin(lab, skin))[nose_row]
    first = last = nose_column
    while first > 0 and bool(on_skin[first - 1]):
        first -= 1
    while last < on_skin.numel() - 1 and bool(on_skin[last + 1]):
        last += 1
    half_face = 0.5 * float(xs[last] - xs[first])
    if half_face < SMALLEST_FACE * span:
        return None
    painted = _painted(front, lab, skin, middle, nose, half_face)
    eyes_at: dict[str, torch.Tensor] = {}
    openings: dict[str, Opening] = {}
    kind = "proportions"
    if painted:
        kind = "painted"
        for side in ("left", "right"):
            opening = _opening(front, *painted[side])
            openings[side] = opening
            column = int(opening.x.numel() // 2)
            eye_x = 0.5 * float(opening.x[0] + opening.x[-1])
            eye_y = 0.5 * float(opening.top[column] + opening.bottom[column])
            eyes_at[side] = torch.tensor([eye_x, eye_y], dtype=torch.float64)
    else:
        for side, sign in (("left", 1.0), ("right", -1.0)):
            eyes_at[side] = torch.tensor([middle + sign * EYE_OUT * half_face, nose + EYE_UP * half_face],
                                         dtype=torch.float64)
    spacing = float((eyes_at["left"][0] - eyes_at["right"][0]).abs())
    if spacing <= 0:
        return None
    centres = {}
    for side, at in eyes_at.items():
        surface = float(front.sample(torch.tensor([[float(at[0]), float(at[1]), 0.0]]))[0])
        if not math.isfinite(surface):
            return None
        centres[side] = torch.tensor([float(at[0]), float(at[1]), surface - EYE_DEPTH * spacing])
    faces_seen = front.part[seen]
    face_parts = [index for index in parts if int((faces_seen == index).sum()) > 0]
    balls = _eyeballs(model, parts, centres["left"], centres["right"], spacing)
    eyeballs: dict[str, list[tuple[int, torch.Tensor]]] = {}
    if balls:
        kind = "eyeballs"
        marks = seen & _not_skin(lab, skin)
        points = torch.stack([x, y, torch.where(seen, front.z, torch.zeros_like(front.z))], dim=-1)
        visible = {}
        for side, found in balls.items():
            weights = torch.tensor([float(held.sum()) for _, held, _ in found])
            centre = sum(ball.centre.float() * weight for (_, _, ball), weight in zip(found, weights)) / weights.sum()
            centres[side] = centre
            eyeballs[side] = [(index, held) for index, held, _ in found]
            # Pixels showing the eyeball's own surface are the open eye.
            mask = torch.zeros_like(seen)
            for index, _, ball in found:
                gap = ((points - ball.centre.double()).norm(dim=-1) - ball.radius).abs()
                mask |= seen & (front.part == index) & (gap < 0.15 * ball.radius)
            visible[side] = mask
        if not openings and all(bool(mask.any()) for mask in visible.values()):
            others = visible["left"] | visible["right"]
            openings = {side: _opening(front, mask, marks & ~others) for side, mask in visible.items()}
    return Eyes(left=centres["left"], right=centres["right"], kind=kind, eyeballs=eyeballs,
                openings=openings, front=front, face_parts=face_parts)


def blink_shapes(model, eyes: Eyes) -> dict[int, dict[str, MorphTarget]]:
    """Shape keys that bring each upper lid down over its open eye.

    Args:
        model: The character the eyes were found in.
        eyes: Its `Eyes`, with ``openings`` for both eyes.

    Returns:
        Part index against shape key name against its `MorphTarget`, position and normal offsets,
        for every part a lid moves; empty when the eyes are not seen open.
    """
    if eyes.front is None or set(eyes.openings) != {"left", "right"}:
        return {}
    front = eyes.front
    on_balls: dict[int, torch.Tensor] = {}
    for found in eyes.eyeballs.values():
        for index, held in found:
            on_balls[index] = on_balls.get(index, torch.zeros_like(held)) | held
    turns = {side: _lid_turns(front, opening, (eyes.left if side == "left" else eyes.right).double())
             for side, opening in eyes.openings.items()}
    out: dict[int, dict[str, torch.Tensor]] = {}
    for index in eyes.face_parts:
        part = model.parts[index]
        if part.label == "hair":
            continue
        points = part.vertices.double()
        surface = front.sample(points)
        for side, opening in eyes.openings.items():
            offsets = _lid_offsets(points, surface, opening, turns[side])
            if offsets is None:
                continue
            if index in on_balls:
                offsets[on_balls[index]] = 0
            if not bool(offsets.any()) or not bool(torch.isfinite(offsets).all()) \
                    or float(offsets.norm(dim=1).max()) > 3 * opening.height:
                continue
            out.setdefault(index, {})[BLINK_SHAPES[side]] = MorphTarget(
                offsets.float(), _normal_offsets(part.vertices, part.faces, offsets.float()))
    return out


def _normal_offsets(vertices: torch.Tensor, faces: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """``(N, 3)`` change in each vertex's smooth normal when ``offsets`` are added to the vertices."""
    welded, triangles, remap = topology.weld(vertices.float(), faces, tolerance=0)
    moved = torch.zeros_like(welded).index_reduce_(0, remap, offsets.float(), "mean", include_self=False)
    before = topology.vertex_normals(welded, triangles)
    after = topology.vertex_normals(welded + moved, triangles)
    return (after - before)[remap].float()


def _lid_turns(front: _Front, opening: Opening, centre: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per column, an eye's line angles and radius about its eyeball's centre, seen along +X.

    Args:
        front: The picture of the face.
        opening: The eye.
        centre: ``(3,)`` turning centre used when the open eye's surface fits no ball.

    Returns:
        ``"pivot"``, the centre; ``"edge"``, ``"closed"``, ``"bottom"`` and ``"roof"`` angles, 0 straight
        ahead and rising upwards; ``"closed_reach"`` and ``"radius"`` in the same plane; ``"ease"``.
    """
    columns = ((opening.x - front.left) / front.pixel).floor().long().clamp(0, front.z.shape[1] - 1)
    ys = front.ys()
    within = (ys[:, None] <= opening.edge[None]) & (ys[:, None] >= opening.bottom[None])
    depth = front.z[:, columns]
    seen = within & torch.isfinite(depth)
    pivot = centre.double()
    if int(seen.sum()) >= 16:
        cloud = torch.stack([opening.x[None].expand_as(depth)[seen], ys[:, None].expand_as(depth)[seen], depth[seen]], 1)
        ball = sections.fit_sphere(cloud)
        if EYEBALL_BULGE[0] * opening.height <= ball.radius <= EYEBALL_BULGE[1] * opening.height \
                and ball.error < 0.1 * ball.radius:
            pivot = ball.centre.double()

    def angle(line: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        depth = front.sample(torch.stack([opening.x, line, torch.zeros_like(line)], dim=1))
        depth = torch.where(torch.isfinite(depth), depth, torch.full_like(depth, float(pivot[2]) + opening.height))
        return torch.atan2(line - pivot[1], depth - pivot[2]), torch.sqrt((line - pivot[1]) ** 2 + (depth - pivot[2]) ** 2)

    radius = torch.sqrt((ys[:, None] - pivot[1]) ** 2 + (depth - pivot[2]) ** 2)
    radius = torch.where(seen, radius, torch.full_like(radius, -math.inf)).max(dim=0).values
    fallback = float(radius[torch.isfinite(radius)].max()) if bool(torch.isfinite(radius).any()) else opening.height
    closed, closed_reach = angle(opening.closed)
    return {"pivot": pivot, "edge": angle(opening.edge)[0], "closed": closed, "closed_reach": closed_reach,
            "bottom": angle(opening.bottom)[0], "roof": angle(opening.roof)[0],
            "radius": torch.where(torch.isfinite(radius), radius, torch.full_like(radius, fallback)),
            "ease": opening.ease}


def _lid_offsets(points: torch.Tensor, surface: torch.Tensor, opening: Opening,
                 turns: dict[str, torch.Tensor]) -> torch.Tensor | None:
    """Offsets turning an upper lid's edge onto the closed line and folding the eye in under it.

    Args:
        points: ``(N, 3)`` vertices.
        surface: ``(N,)`` depth of the face's front under each vertex.
        opening: The eye.
        turns: Its `_lid_turns`.

    Returns:
        ``(N, 3)`` offsets, or None when the eye moves nothing.
    """
    count = opening.x.numel()
    if count < 2 or opening.height <= 0:
        return None
    centre = turns["pivot"]
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    inside = (x >= float(opening.x[0])) & (x <= float(opening.x[-1])) & (z > centre[2])
    if not bool(inside.any()):
        return None
    place = ((x - opening.x[0]) / (opening.x[-1] - opening.x[0]) * (count - 1)).clamp(0, count - 1)
    low = place.floor().long().clamp(max=count - 2)
    share = place - low

    def at(values: torch.Tensor) -> torch.Tensor:
        return values[low] * (1 - share) + values[low + 1] * share

    edge, closed, roof, bottom, radius, ease, closed_reach = (
        at(turns[key]) for key in ("edge", "closed", "roof", "bottom", "radius", "ease", "closed_reach"))
    gap = LID_GAP * opening.height
    turned = torch.atan2(y - centre[1], z - centre[2])
    reach = torch.sqrt((y - centre[1]) ** 2 + (z - centre[2]) ** 2)
    new_turn, new_reach = turned.clone(), reach.clone()
    lid = inside & (turned >= edge) & (turned < roof) & (roof > edge)
    up = ((turned - edge) / (roof - edge).clamp_min(1e-9)).clamp(0, 1)
    # The lid's edge and lash line turn whole; the skin above them takes up the stretch.
    new_turn = torch.where(lid, turned - (edge - closed) * (1 - up * up * (3 - 2 * up)), new_turn)
    travel = ((turned - new_turn) / (edge - closed).clamp_min(1e-9)).clamp(0, 1)
    new_reach = torch.where(lid, reach + (radius + gap - reach).clamp_min(0) * travel, new_reach)
    floor = torch.minimum(bottom, closed)
    eye = inside & (turned >= floor) & (turned < edge) & (edge > floor)
    fold = (edge - turned) / (edge - floor).clamp_min(1e-9)
    new_turn = torch.where(eye, closed - FOLD_SPREAD * (edge - closed) * fold, new_turn)
    new_reach = torch.where(eye, (closed_reach - gap).clamp_min(0), new_reach)
    new_turn = turned + (new_turn - turned) * ease
    new_reach = reach + (new_reach - reach) * ease
    band = LID_DEPTH * opening.height
    behind = (surface - z).clamp_min(0)
    depth_weight = (1 - (behind - band) / band).clamp(0, 1)
    depth_weight = torch.where(torch.isfinite(surface), depth_weight, torch.zeros_like(depth_weight))
    offsets = torch.zeros(points.shape[0], 3, dtype=torch.float64)
    offsets[:, 1] = centre[1] + new_reach * torch.sin(new_turn) - y
    offsets[:, 2] = centre[2] + new_reach * torch.cos(new_turn) - z
    offsets[~(lid | eye)] = 0
    return offsets * depth_weight[:, None]

"""Placing a separately made head, hair and an extra on a body, as one model with labelled parts."""
# The body is turned to face +Z, sized and stood on the floor. The head is fitted over the body's own
# head, or sized by its neck; its neck is eased onto the body's neck ring and its skin below the ring
# tucked just inside the body's neck, and its hair is kept whole.

from __future__ import annotations

__all__ = [
    "CLEAR_HAIR", "EXTRA_FOLLOWS", "FACINGS", "HEAD_FITS", "REMOVE_HEAD", "SKIN_MATCHES", "assemble", "body_neck",
    "clear_hanging_hair", "hair_colour", "head_hair", "match_skin", "parse_placement",
]

import json
import math

import torch

from ..log import get_logger
from ..mesh import clean, cut, raster, sections, topology
from ..mesh.bvh import TriangleBVH
from ..model3d import merge, orient
from ..model3d.scene import Model3D

logger = get_logger("character.assemble")

#: Which way a static body faces before it is turned to +Z; ``auto`` reads it from the body.
FACINGS = ("auto", "+Z", "-Z", "+X", "-X")

#: Whether the body's own head is cut away at its neck.
REMOVE_HEAD = ("auto", "yes", "no")

#: ``auto`` clears the body's own hair below the neck when its head is cut away; ``off`` keeps it.
CLEAR_HAIR = ("auto", "off")

#: Share of the cut-away head, from the crown down, whose colour is read as the hair's.
CROWN_SHARE = 0.25

#: Lab distance a face's colour may sit from the hair's to start a region, beyond three times the
#: hair's own spread.
HAIR_TOLERANCE = (12.0, 6.0, 8.0)

#: How much wider the distance is for a face the region grows into.
HAIR_GROW_FACTOR = 2.0

#: Rounds of averaging each face's colour with its neighbours' before matching.
HAIR_SMOOTHING = 8

#: Rings of faces past the hair's edge removed with it, so no fringe of hair is left around the fill.
HAIR_FRINGE = 6

#: Rings an opening too long for the hole filler is closed with, stepping in to its centre.
CAP_RINGS = 4

#: How far below the neck hanging hair is followed, as a share of the body's height.
HAIR_REACH = 0.45

#: Rounds of region growing, a bound on the walk from the neck down the hair.
HAIR_STEPS = 20000

#: How a head is placed: fitted to the body's own head, sized by its neck, either by what the body
#: has, or moved only as the body moved.
HEAD_FITS = ("auto", "body's head", "neck width", "keep")

#: What an extra part moves with.
EXTRA_FOLLOWS = ("head", "body")

#: Degrees about Y that turn a body facing each direction to face +Z.
_TURNS = {"+Z": 0.0, "-Z": 180.0, "+X": -90.0, "-X": 90.0}

#: Points sampled on the front of the body's own head to fit the new head to.
REGISTER_POINTS = 4000

#: Rounds of fitting, and the share of closest matches each round keeps.
REGISTER_STEPS = 40
REGISTER_KEEP = 0.7

#: Least forward lean of a point on the body's own head used in the fit.
REGISTER_FRONT = 0.15

#: Sizes, against the first guess, the fit starts from; the closest result is kept.
REGISTER_STARTS = (0.8, 1.0, 1.3, 1.7)

#: Scale against the first guess a fit may reach before it is refused.
REGISTER_SCALE = (0.5, 2.5)

#: Largest mean distance a kept fit may leave, as a share of the body's own head height.
REGISTER_ERROR = 0.05

#: Share past the neck ring a piece above the cut may meet it and still be the body's own head.
HEAD_CUT_GIVE = 1.05

#: Reach from the neck's axis, in body head heights, inside which a loose piece above the cut is removed.
HEAD_LOOSE_REACH = 0.6

#: Height above the body's neck ring over which the head's neck eases onto it, in body neck radii;
#: and the depth below the ring a neck too short to reach it is drawn down to before it is cut.
SEAM_BLEND = 1.2
SEAM_SPARE = 0.15

#: Shares of the head's own neck size between which skin fades out of the easing, so the jaw and cheeks
#: above the ring keep their shape.
SEAM_NEAR, SEAM_FAR = 1.15, 1.6

#: The necks meet at the head's own neck, moved at most this share towards the body's; the body's neck
#: takes up the rest over this depth below the seam, in neck radii.
SEAM_GIVE = 1.05
SEAM_BODY_BLEND = 0.6

#: Depth below the body's neck ring, in neck radii, the seam is looked for, the heights it is tried at,
#: and how much worse a lower seam may fit than the best and still be taken, as a share of the neck.
SEAM_SEARCH, SEAM_STEPS, SEAM_SLACK = 1.5, 7, 0.03

#: Model metadata key holding the height of the body's neck ring as a share of the figure's height.
NECK_RING_KEY = "neck_ring_share"

#: Height above the ring, in neck radii, over which the head's normals turn to meet the body's.
SEAM_SHADE = 0.4

#: Depth below the ring, in neck radii, where the slope of the body's neck is read, and the most it
#: may widen there before it is taken for clothing rather than neck.
SEAM_SLOPE = 0.25
SEAM_FLARE = 1.3

#: Lab distance between a head's crown and its face under which the head reads as bald.
BALD_DISTANCE = 15.0


#: ``head to body`` tones the new head's skin to the body's; ``off`` leaves both.
SKIN_MATCHES = ("head to body", "off")

#: Distance in Lab ``a, b`` from the head's skin hue a texel may sit and still be fully toned, as a
#: share of the skin's chroma plus a floor; the toning fades out by `SKIN_FADE` times this. Lightness
#: is left out, so skin in shadow is toned as fully as skin in light.
SKIN_HUE_SHARE, SKIN_HUE_FLOOR = 0.6, 6.0
SKIN_FADE = 1.6

#: Texels with less than this share of the skin's chroma, such as eye whites and brows, fade out of
#: the toning.
SKIN_CHROMA_SHARE = 0.45

#: Texels the toned and the hair areas of a texture are widened by, over the padding round each island.
SKIN_TEXEL_MARGIN = 3

#: Skins nearer than the first Lab distance are left alike; past the second the body's reading is
#: not taken for skin.
SKIN_MATCH_RANGE = (2.0, 30.0)

#: Neck radii above the seam over which the head's skin is drawn from the body's colour at the seam back to its
#: own, and the depth either side of the seam the two necks' colours are read over.
SEAM_TONE_BAND = 0.6
SEAM_TONE_STRIP = 0.05

#: Sectors round the neck its colours are read and matched in, and the passes each sector's step is averaged
#: with its neighbours'.
SEAM_TONE_SECTORS = 96
SEAM_TONE_SMOOTHING = 1

#: Neck radii in or out from the neck's outline a face may lie and be read as neck; the head's skin is toned out
#: to `SEAM_TONE_REACH` times the outline.
SEAM_TONE_GIVE = 0.25
SEAM_TONE_REACH = 2.0

#: Angle steps in a ring's outline.
PLUG_BINS = 256

#: Depth below an open neck or the top of a headless body where its ring is read, in neck radii.
RING_INSET = 0.02

#: Head height above the neck ring, in body neck radii, past which the body has its own head.
OWN_HEAD_RADII = 1.6

#: Share of a full body's height searched for its neck.
BODY_NECK_BAND = (0.68, 0.95)

#: Share of a bust's height searched for its neck.
BUST_NECK_BAND = (0.08, 0.7)

#: Longest placement text read, and the bounds of its values.
MAX_PLACEMENT_CHARS = 20000
MAX_OFFSET = 100.0
SCALE_RANGE = (0.01, 100.0)

#: Placement keys, one per role.
ROLES = ("body", "head", "hair", "extra")


def _matrix_text(values) -> list[float]:
    """Three finite numbers from a placement entry."""
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError("placement offsets and rotations are lists of three numbers, such as [0, 0.02, 0]")
    numbers = [float(value) for value in values]
    if not all(math.isfinite(value) for value in numbers):
        raise ValueError("placement holds a number that is not finite")
    return numbers


def parse_placement(text: str) -> dict[str, dict]:
    """Per-role adjustments from the placement text.

    Args:
        text: JSON such as ``{"head": {"offset": [0, 0.01, 0], "rotate": [0, 10, 0],
            "scale": 1.05}}``; empty means none.

    Returns:
        Role against ``{"offset", "rotate", "scale"}`` with defaults filled in.

    Raises:
        ValueError: The text is too long, not JSON, names an unknown role or holds bad values.
    """
    text = str(text or "").strip()
    if not text:
        return {}
    if len(text) > MAX_PLACEMENT_CHARS:
        raise ValueError(f"placement is {len(text)} characters, more than {MAX_PLACEMENT_CHARS}; clear it "
                         "to place the parts automatically")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(f"placement is not valid JSON ({error.msg}); clear it to place the parts "
                         "automatically") from None
    if not isinstance(data, dict):
        raise ValueError("placement is a JSON object keyed by body, head, hair or extra")
    result = {}
    for role, entry in data.items():
        if role not in ROLES:
            raise ValueError(f"placement names '{str(role)[:40]}'; use body, head, hair or extra")
        if not isinstance(entry, dict):
            raise ValueError(f"placement for {role} is an object with offset, rotate and scale")
        offset = _matrix_text(entry.get("offset", [0, 0, 0]))
        rotate = _matrix_text(entry.get("rotate", [0, 0, 0]))
        scale = float(entry.get("scale", 1.0))
        if not math.isfinite(scale) or not SCALE_RANGE[0] <= scale <= SCALE_RANGE[1]:
            raise ValueError(f"placement scale for {role} is {scale}; use {SCALE_RANGE[0]} to {SCALE_RANGE[1]}")
        if max(abs(value) for value in offset) > MAX_OFFSET or max(abs(value) for value in rotate) > 360:
            raise ValueError(f"placement for {role} moves more than {MAX_OFFSET} m or turns more than 360 degrees")
        result[role] = {"offset": offset, "rotate": rotate, "scale": scale}
    return result


def _geometry(model: Model3D) -> tuple[torch.Tensor, torch.Tensor]:
    """Every part's triangles as one vertex and face table."""
    vertices, faces, base = [], [], 0
    for part in model.parts:
        if not part.vertex_count or not part.triangle_count:
            continue
        vertices.append(part.vertices.float())
        faces.append(part.faces.long() + base)
        base += part.vertex_count
    if not vertices:
        raise ValueError("a wired model has no triangles")
    return torch.cat(vertices), torch.cat(faces)


def _about(pivot: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """``matrix`` applied about ``pivot`` rather than the origin."""
    return orient.similarity(translation=pivot.double()) @ matrix @ orient.similarity(translation=-pivot.double())


def _adjust(entry: dict | None, pivot: torch.Tensor) -> torch.Tensor:
    """The placement adjustment of one role as a matrix about ``pivot``."""
    if not entry:
        return orient.similarity()
    rx, ry, rz = entry["rotate"]
    turn = orient.similarity(orient.rotation_matrix(rx, ry, rz), scale=entry["scale"])
    return orient.similarity(translation=entry["offset"]) @ _about(pivot, turn)


def _body_turn(body: Model3D, facing: str, vertices: torch.Tensor, faces: torch.Tensor) -> tuple[float, str]:
    """Degrees about Y that face the body to +Z, and how that was decided."""
    if facing in _TURNS:
        return _TURNS[facing], f"set to {facing}"
    if body.meta.get(orient.FACES_FORWARD_KEY):
        return 0.0, "already facing +Z"
    angle = orient.facing_angle(body)
    if angle is not None:
        return angle, "read from the skeleton"
    try:
        found = sections.landmarks(vertices, faces, facing="auto")
    except ValueError:
        return 0.0, "taken as +Z, the body did not read as a standing figure"
    snapped = 90.0 * round(float(found.heading) / 90.0)
    snapped = 180.0 if snapped == -180.0 else snapped
    return snapped, f"read from the feet ({float(found.heading):.0f} degrees), squared to {snapped:.0f}"


def _opening(vertices: torch.Tensor, faces: torch.Tensor, band: tuple[float, float],
             radii: tuple[float, float]) -> tuple[torch.Tensor, torch.Tensor, float] | None:
    """The largest roughly level open boundary in a height band: ``(centre, normal up, radius)``.

    Args:
        vertices: ``(V, 3)`` positions.
        faces: ``(F, 3)`` triangles.
        band: Lowest and highest share of the height the opening's centre may sit at.
        radii: Smallest and largest radius, as shares of the height.
    """
    welded, wfaces, _ = topology.weld(vertices, faces, tolerance=0)
    low, high = welded.min(dim=0).values, welded.max(dim=0).values
    height = float(high[1] - low[1])
    best = None
    for loop in topology.boundary_loops(wfaces, int(welded.shape[0])):
        if loop.numel() < 6:
            continue
        points = welded[loop].double()
        area_vector = 0.5 * torch.linalg.cross(points, points.roll(-1, dims=0)).sum(dim=0)
        area = float(area_vector.norm())
        if area <= 0:
            continue
        normal = area_vector / area
        centre = points.mean(dim=0)
        radius = math.sqrt(area / math.pi)
        share = float(centre[1] - low[1]) / max(height, 1e-12)
        if (not band[0] <= share <= band[1] or abs(float(normal[1])) < 0.6
                or not radii[0] * height <= radius <= radii[1] * height):
            continue
        if best is None or area > best[3]:
            best = (centre, normal * (1.0 if float(normal[1]) > 0 else -1.0), radius, area)
    return None if best is None else best[:3]


#: Sections scanned when looking for a neck ring.
NECK_LEVELS = 96

#: A ring at least this share of the widest above it still belongs to the head.
HEAD_PEAK_SHARE = 0.98

#: How many times the neck's area the shoulders' first ring holds.
SHOULDER_JUMP = 2.0


def _inside(points: torch.Tensor, point: torch.Tensor) -> bool:
    """Whether a 2D point lies inside a closed polygon, by crossings."""
    x, y = float(point[0]), float(point[1])
    a = points.double()
    b = a.roll(-1, dims=0)
    crosses = (a[:, 1] > y) != (b[:, 1] > y)
    at = a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1]).where(crosses, torch.ones_like(a[:, 1]))
    return bool((crosses & (at > x)).sum() % 2)


def _axis_neck(vertices: torch.Tensor, faces: torch.Tensor,
               band: tuple[float, float]) -> tuple[torch.Tensor, torch.Tensor, float, float] | None:
    """The neck: the narrowest ring below the head's widest, before the shoulders double it.

    Returns:
        ``(centre, normal up, radius, area above)``, or None when no shoulders follow a head.
    """
    low, high = vertices.min(dim=0).values, vertices.max(dim=0).values
    height = float(high[1] - low[1])
    bottom, top = float(low[1]) + band[0] * height, float(low[1]) + band[1] * height
    found = sections.profile(vertices, faces, levels=NECK_LEVELS, low=bottom, high=top)
    rings: list = [None] * found.level_count
    previous = None
    for level in reversed(range(found.level_count)):
        loops = [loop for loop in found.loops(level) if loop.closed and abs(loop.area) > 0]
        if not loops:
            continue
        if previous is None:
            chosen = max(loops, key=lambda loop: abs(loop.area))
        else:
            holding = [loop for loop in loops if _inside(loop.points[:, [0, 2]], previous[[0, 2]])]
            pool = holding or loops
            chosen = min(pool, key=lambda loop: float((loop.centroid.double() - previous)[[0, 2]].norm()))
            if not holding and float((chosen.centroid.double() - previous)[[0, 2]].norm()) > 0.1 * height:
                continue
        rings[level] = chosen
        previous = chosen.centroid.double()
    areas = [abs(ring.area) if ring is not None else math.nan for ring in rings]
    peak, best = 0.0, None
    for level in reversed(range(len(areas))):
        area = areas[level]
        if not math.isfinite(area):
            continue
        if best is None and area >= HEAD_PEAK_SHARE * peak:
            peak = max(peak, area)
            continue
        if best is not None and area > SHOULDER_JUMP * areas[best]:
            break
        if best is None or area < areas[best]:
            best = level
    else:
        best = None
    if best is None:
        return None
    ring = rings[best]
    above = peak
    return (ring.centroid.double(), torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
            math.sqrt(abs(ring.area) / math.pi), above)


def _top_ring(vertices: torch.Tensor, faces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float] | None:
    """The ring just under the top of a closed headless body: ``(centre, normal up, radius)``."""
    low, high = vertices.min(dim=0).values, vertices.max(dim=0).values
    height = float(high[1] - low[1])
    middle = (low + high) / 2
    loops = sections.slice_plane(vertices, faces, (float(middle[0]), float(high[1]) - 0.04 * height,
                                                   float(middle[2])), (0.0, 1.0, 0.0))
    closed = [loop for loop in loops if loop.closed and abs(loop.area) > 0]
    if not closed:
        return None
    ring = min(closed, key=lambda loop: float(((loop.centroid - middle)[[0, 2]]).norm()))
    return (ring.centroid.double(), torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64),
            math.sqrt(abs(ring.area) / math.pi))


def body_neck(body: Model3D) -> dict:
    """Where a head goes on a body.

    Returns:
        ``centre`` and ``normal`` (float64 ``(3,)``, normal towards the head), ``radius``,
        ``own_head`` (the body still has a head above the ring), ``head_height`` (that head's
        height above the ring, or None) and ``found`` (how).

    Raises:
        ValueError: The body has no neck that can be found.
    """
    vertices, faces = _geometry(body)
    opening = _opening(vertices, faces, (0.75, 1.0), (0.01, 0.15))
    if opening is not None:
        centre, normal, radius = opening
        return {"centre": centre, "normal": normal, "radius": radius, "own_head": False,
                "head_height": None, "found": "open neck"}
    ring = _axis_neck(vertices, faces, BODY_NECK_BAND)
    if ring is not None:
        centre, normal, radius, above = ring
        top = float(vertices[:, 1].max())
        if float(top - centre[1]) > OWN_HEAD_RADII * radius:
            return {"centre": centre, "normal": normal, "radius": radius, "own_head": True,
                    "head_height": float(top - centre[1]), "found": "neck ring"}
    ring = _top_ring(vertices, faces)
    if ring is None:
        raise ValueError("no neck was found on the body; set head_fit to keep and place the head with placement")
    centre, normal, radius = ring
    return {"centre": centre, "normal": normal, "radius": radius, "own_head": False, "head_height": None,
            "found": "top ring"}


def _lab(rgb: torch.Tensor) -> torch.Tensor:
    """sRGB codes ``(N, 3)`` in ``[0, 1]`` as CIE Lab under D65."""
    rgb = rgb.clamp(0, 1).float()
    linear = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    matrix = torch.tensor([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = linear @ matrix.T / torch.tensor([0.95047, 1.0, 1.08883])
    epsilon, kappa = 216 / 24389, 24389 / 27
    f = torch.where(xyz > epsilon, xyz.clamp_min(1e-12) ** (1 / 3), (kappa * xyz + 16) / 116)
    return torch.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], dim=1)


def _face_colours(part) -> torch.Tensor | None:
    """``(F, 3)`` Lab colour of each face, from its texture at the face's centre, else its colours."""
    if not part.triangle_count:
        return None
    material = part.material
    factor = torch.tensor(material.base_color[:3], dtype=torch.float32)
    texture = material.base_color_texture
    if texture is not None and part.uvs is not None:
        centre = part.uvs[part.faces].float().mean(dim=1)
        colour = raster.sample_texture(texture.image.float(), centre, texture.wrap_s, texture.wrap_t)[:, :3]
        return _lab(colour * factor)
    if part.colors is not None:
        return _lab(part.colors[part.faces][..., :3].float().mean(dim=1) * factor)
    return None


def hair_colour(body: Model3D, neck: dict) -> tuple[torch.Tensor, torch.Tensor] | None:
    """The colour of the body's own hair, read on its crown: ``(median Lab, spread)``, or None."""
    top = max(float(part.vertices[:, 1].max()) for part in body.parts if part.vertex_count)
    from_y = top - CROWN_SHARE * (top - float(neck["centre"][1]))
    samples = []
    for part in body.parts:
        colours = _face_colours(part)
        if colours is None:
            continue
        crown = part.vertices[part.faces].float().mean(dim=1)[:, 1] > from_y
        samples.append(colours[crown])
    samples = torch.cat(samples) if samples else torch.zeros(0, 3)
    if samples.shape[0] < 16:
        return None
    middle = samples.median(dim=0).values
    return middle, (samples - middle).abs().median(dim=0).values


def _smoothed_colours(part) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """A part's face colours averaged over neighbours, and its face pairs: ``(colours, first, second)``."""
    colours = _face_colours(part)
    if colours is None:
        return None
    welded, faces, _ = topology.weld(part.vertices, part.faces, tolerance=0)
    _, pairs = topology.edge_topology(faces, welded.shape[0]).manifold_face_pairs()
    first, second = pairs[:, 0], pairs[:, 1]
    for _ in range(HAIR_SMOOTHING):
        total = colours.clone()
        count = torch.ones(colours.shape[0], 1)
        total.index_add_(0, first, colours[second])
        total.index_add_(0, second, colours[first])
        count.index_add_(0, first, torch.ones(first.shape[0], 1))
        count.index_add_(0, second, torch.ones(second.shape[0], 1))
        colours = total / count
    return colours, first, second


def _grow(region: torch.Tensor, allowed: torch.Tensor, first: torch.Tensor, second: torch.Tensor,
          steps: int) -> torch.Tensor:
    """``region`` grown across face pairs into ``allowed`` faces, at most ``steps`` rings."""
    for _ in range(steps):
        grown = region.clone()
        grown[first[region[second]]] = True
        grown[second[region[first]]] = True
        grown &= allowed
        if bool((grown == region).all()):
            break
        region = grown
    return region


def _front_faces(part, level: float, span: float, middle_x: float, width: float) -> torch.Tensor:
    """Faces of a part on the front of a face: between a third and three fifths up, facing forward."""
    corners = part.vertices[part.faces].float()
    centres = corners.mean(dim=1)
    normals = torch.nn.functional.normalize(
        torch.linalg.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), dim=1)
    return ((centres[:, 1] > level + 0.35 * span) & (centres[:, 1] < level + 0.6 * span)
            & (normals[:, 2] > 0.6) & ((centres[:, 0] - middle_x).abs() < 0.15 * width))


def _face_skin(model: Model3D, level: float) -> torch.Tensor:
    """Lab colours sampled on the front of the face of a head standing on ``level``."""
    live = [part for part in model.parts if part.triangle_count]
    if not live:
        return torch.zeros(0, 3)
    top = max(float(part.vertices[:, 1].max()) for part in live)
    low_x = min(float(part.vertices[:, 0].min()) for part in live)
    high_x = max(float(part.vertices[:, 0].max()) for part in live)
    samples = []
    for part in live:
        colours = _face_colours(part)
        if colours is not None:
            samples.append(colours[_front_faces(part, level, top - level, (low_x + high_x) / 2, high_x - low_x)])
    return torch.cat(samples) if samples else torch.zeros(0, 3)


def _srgb(lab: torch.Tensor) -> torch.Tensor:
    """CIE Lab under D65 ``(N, 3)`` as sRGB codes in ``[0, 1]``."""
    fy = (lab[:, 0] + 16) / 116
    f = torch.stack([fy + lab[:, 1] / 500, fy, fy - lab[:, 2] / 200], dim=1)
    epsilon, kappa = 216 / 24389, 24389 / 27
    xyz = torch.where(f ** 3 > epsilon, f ** 3, (116 * f - 16) / kappa) * torch.tensor([0.95047, 1.0, 1.08883])
    matrix = torch.tensor([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]])
    linear = (xyz @ matrix.T).clamp(0, 1)
    return torch.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear.clamp_min(1e-12) ** (1 / 2.4) - 0.055)


def body_skin(body: Model3D, neck: dict) -> torch.Tensor | None:
    """Median Lab colour of the body's skin: on the face of its own head, else on its neck below the ring."""
    level = float(neck["centre"][1])
    if neck["own_head"]:
        samples = _face_skin(body, level)
    else:
        centre, radius = neck["centre"].double(), float(neck["radius"])
        pieces = []
        for part in body.parts:
            colours = _face_colours(part)
            if colours is None:
                continue
            middle = part.vertices[part.faces].double().mean(dim=1)
            radial = (middle[:, [0, 2]] - centre[[0, 2]]).norm(dim=1)
            pieces.append(colours[(middle[:, 1] < level) & (middle[:, 1] > level - 1.5 * radius)
                                  & (radial < 1.3 * radius)])
        samples = torch.cat(pieces) if pieces else torch.zeros(0, 3)
    return samples.median(dim=0).values if samples.shape[0] >= 16 else None


def _texel_cover(parts: list, masks: list, width: int, height: int) -> torch.Tensor:
    """``(H, W)`` bool: texels under the chosen faces of parts sharing one texture, widened by a margin."""
    cover = torch.zeros(height * width, dtype=torch.bool)
    for part, mask in zip(parts, masks):
        chosen = part.faces.long() if mask is None else part.faces.long()[mask]
        if not chosen.numel():
            continue
        corners, _ = raster.texel_triangles(part.uvs.float()[chosen], width, height)
        for piece in raster.coverage_chunks(corners, width, height):
            cover[piece.pixel] = True
    grown = torch.nn.functional.max_pool2d(cover.reshape(1, 1, height, width).float(),
                                           2 * SKIN_TEXEL_MARGIN + 1, stride=1, padding=SKIN_TEXEL_MARGIN)
    return grown.reshape(height, width) > 0


def match_skin(head: Model3D, skin: torch.Tensor, level: float, hair: list | None = None) -> dict:
    """Tone the head's skin towards ``skin`` in its textures and colours, leaving hair, eyes and brows.

    Args:
        head: The placed head, changed in place; a texture it shares is changed once.
        skin: Lab colour to tone towards.
        level: Height of the neck the head stands on.
        hair: Hair face mask per head part, as `head_hair` gives it, or None.

    Returns:
        ``{"skin toned by"}`` in Lab units, and ``notes`` when the head was left as it was.
    """
    samples = _face_skin(head, level)
    if samples.shape[0] < 16:
        return {"skin toned by": 0.0, "notes": ["the head's face could not be read, so its skin was left"]}
    own = samples.median(dim=0).values
    shift = skin.float() - own
    distance = float(shift.norm())
    if distance < SKIN_MATCH_RANGE[0]:
        return {"skin toned by": round(distance, 1), "notes": []}
    if distance > SKIN_MATCH_RANGE[1]:
        return {"skin toned by": 0.0, "notes": ["the body's skin reads too far from the head's to be skin, so "
                                                "neither was toned"]}
    chroma = float(own[1:].norm())
    hue_reach = SKIN_HUE_SHARE * chroma + SKIN_HUE_FLOOR
    hair = hair if hair is not None else [None] * len(head.parts)

    def toned(rgb: torch.Tensor, factor: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        lab = _lab(rgb * factor)
        reach = (lab[:, 1:] - own[1:]).norm(dim=1) / hue_reach
        weight = ((SKIN_FADE - reach) / (SKIN_FADE - 1.0)).clamp(0.0, 1.0)
        weight = weight * (lab[:, 1:].norm(dim=1) / max(SKIN_CHROMA_SHARE * chroma, 1e-6)).clamp(0.0, 1.0)
        weight = weight * allowed.float()
        moved = lab.clone()
        moved[:, 0] = lab[:, 0] * (1 + shift[0] / max(float(own[0]), 1.0) * weight)
        moved[:, 1:] = lab[:, 1:] + shift[1:] * weight[:, None]
        return (_srgb(moved) / factor.clamp_min(1e-6)).clamp(0, 1)

    sharing: dict[int, list] = {}
    for index, part in enumerate(head.parts):
        if not part.triangle_count:
            continue
        factor = torch.tensor(part.material.base_color[:3], dtype=torch.float32)
        texture = part.material.base_color_texture
        if texture is not None and part.uvs is not None:
            sharing.setdefault(id(texture), [texture, factor, []])[2].append(index)
        elif part.colors is not None:
            allowed = torch.ones(part.vertex_count, dtype=torch.bool)
            if hair[index] is not None:
                allowed[part.faces[hair[index]].reshape(-1)] = False
            colors = part.colors.float().clone()
            colors[:, :3] = toned(colors[:, :3], factor, allowed)
            part.colors = colors
    for texture, factor, indices in sharing.values():
        height, width = int(texture.image.shape[0]), int(texture.image.shape[1])
        parts = [head.parts[index] for index in indices]
        masks = [hair[index] for index in indices]
        skin_cover = _texel_cover(parts, [None if mask is None else ~mask for mask in masks], width, height)
        hair_cover = _texel_cover(parts, masks, width, height) if any(mask is not None for mask in masks) \
            else torch.zeros(height, width, dtype=torch.bool)
        image = texture.image.float().clone()
        image[..., :3] = toned(image[..., :3].reshape(-1, 3), factor,
                               (skin_cover & ~hair_cover).reshape(-1)).reshape(height, width, 3)
        texture.image = image
    return {"skin toned by": round(distance, 1), "notes": []}


def _strip_faces(part, low: float, high: float, centre: torch.Tensor, outline: torch.Tensor,
                 give: float) -> torch.Tensor:
    """``(F,)`` bool: standing faces between two heights, centred within ``give`` of an outline about ``centre``."""
    corners = part.vertices[part.faces].double()
    middle = corners.mean(dim=1)
    offset = middle[:, [0, 2]] - centre.double()
    angle = torch.atan2(offset[:, 1], offset[:, 0])
    normal = torch.linalg.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    standing = normal[:, 1].abs() < 0.9 * normal.norm(dim=1)
    return ((middle[:, 1] > low) & (middle[:, 1] < high) & standing
            & ((offset.norm(dim=1) - _at(outline, angle)).abs() < give))


def _seam_colours(model: Model3D, masks: list, centre: torch.Tensor, skin: torch.Tensor | None) -> torch.Tensor:
    """``(SEAM_TONE_SECTORS, 3)`` median Lab colour of the chosen skin faces per sector round the neck, NaN where few."""
    colours, sectors = [], []
    for part, mask in zip(model.parts, masks):
        if mask is None or not bool(mask.any()):
            continue
        found = _face_colours(part)
        if found is None:
            continue
        mask = mask & (_skin_likeness(found, skin) > 0.5)
        offset = part.vertices[part.faces[mask]].double().mean(dim=1)[:, [0, 2]] - centre.double()
        turn = (torch.atan2(offset[:, 1], offset[:, 0]) + math.pi) / (2 * math.pi)
        sectors.append((turn * SEAM_TONE_SECTORS).long().clamp(0, SEAM_TONE_SECTORS - 1))
        colours.append(found[mask])
    result = torch.full((SEAM_TONE_SECTORS, 3), math.nan)
    if colours:
        colours, sectors = torch.cat(colours), torch.cat(sectors)
        for sector in range(SEAM_TONE_SECTORS):
            picked = colours[sectors == sector]
            if picked.shape[0] >= 3:
                result[sector] = picked.median(dim=0).values
    return result


def _sector_filled(values: torch.Tensor, known: torch.Tensor) -> torch.Tensor:
    """Per-sector rows with the unknown sectors read between the nearest known ones round the circle."""
    count = values.shape[0]
    rows = torch.nonzero(known).squeeze(1)
    filled = values.clone()
    for sector in torch.nonzero(~known).squeeze(1).tolist():
        after = (rows - sector) % count
        before = (sector - rows) % count
        ahead, behind = rows[after.argmin()], rows[before.argmin()]
        reach = int(after.min()) + int(before.min())
        share = int(before.min()) / max(reach, 1)
        filled[sector] = values[behind] * (1 - share) + values[ahead] * share
    return filled


def _sector_at(values: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Per-sector rows read at each angle, between the two nearest sector middles."""
    count = values.shape[0]
    position = (angle + math.pi) / (2 * math.pi) * count - 0.5
    low = position.floor()
    first = low.long() % count
    share = (position - low)[:, None]
    return values[first] * (1 - share) + values[(first + 1) % count] * share


def _skin_likeness(lab: torch.Tensor, skin: torch.Tensor | None) -> torch.Tensor:
    """``(N,)`` share 0 to 1 by which Lab colours read as the skin ``skin`` in hue and chroma; all 1 without one."""
    if skin is None:
        return torch.ones(lab.shape[0], dtype=lab.dtype)
    chroma = float(skin[1:].norm())
    reach = (lab[:, 1:] - skin[1:].to(lab.dtype)).norm(dim=1) / (SKIN_HUE_SHARE * chroma + SKIN_HUE_FLOOR)
    hue = ((SKIN_FADE - reach) / (SKIN_FADE - 1.0)).clamp(0.0, 1.0)
    return hue * (lab[:, 1:].norm(dim=1) / max(SKIN_CHROMA_SHARE * chroma, 1e-6)).clamp(0.0, 1.0)


def tone_seam(head: Model3D, body: Model3D, level: float, ring: dict, radius: float,
              hair: list | None = None) -> dict:
    """Draw the head's neck skin onto the body's colour at the seam, easing back to its own over `SEAM_TONE_BAND`.

    Args:
        head: The joined head, changed in place; a texture it shares is changed once.
        body: The body it is joined to, read only.
        level: Height of the seam.
        ring: The neck at the seam, as `_neck_ring` answers.
        radius: The neck's radius.
        hair: Hair face mask per head part, left as it is, or None.

    Returns:
        ``{"seam toned by"}``, the median Lab step at the seam, and ``notes`` when it was left as it was.
    """
    centre, outline = ring["centre"], ring["outline"]
    give, strip, band = SEAM_TONE_GIVE * radius, SEAM_TONE_STRIP * radius, SEAM_TONE_BAND * radius
    hair = hair if hair is not None else [None] * len(head.parts)
    above = []
    for part, mask in zip(head.parts, hair):
        chosen = _strip_faces(part, level, level + strip, centre, outline, give) if part.triangle_count else None
        above.append(chosen if chosen is None or mask is None else chosen & ~mask)
    below = [_strip_faces(part, level - strip, level, centre, outline, give) if part.triangle_count else None
             for part in body.parts]
    samples = _face_skin(head, level)
    skin = samples.median(dim=0).values if samples.shape[0] >= 16 else None
    head_ring, body_ring = _seam_colours(head, above, centre, skin), _seam_colours(body, below, centre, skin)
    known = torch.isfinite(head_ring).all(dim=1) & torch.isfinite(body_ring).all(dim=1)
    if int(known.sum()) < 3:
        return {"seam toned by": 0.0, "notes": ["the necks' colours could not be read at the seam, so it was "
                                                "left as it was"]}
    step = (body_ring - head_ring).nan_to_num(0.0)
    distance = float(step[known].norm(dim=1).median())
    if distance > SKIN_MATCH_RANGE[1]:
        return {"seam toned by": 0.0, "notes": []}
    step = _sector_filled(step, known)
    for _ in range(SEAM_TONE_SMOOTHING):
        step = (step.roll(1, dims=0) + 2 * step + step.roll(-1, dims=0)) / 4
    step = step * (SKIN_MATCH_RANGE[1] / step.norm(dim=1, keepdim=True).clamp_min(SKIN_MATCH_RANGE[1]))

    def eased(points: torch.Tensor) -> torch.Tensor:
        share = ((points[:, 1].double() - level) / band).clamp(0.0, 1.0)
        return 1.0 - share * share * (3 - 2 * share)

    def turned(points: torch.Tensor) -> torch.Tensor:
        offset = points[:, [0, 2]].double() - centre.double()
        return torch.atan2(offset[:, 1], offset[:, 0])

    sharing: dict[int, list] = {}
    for part, mask in zip(head.parts, hair):
        if not part.triangle_count:
            continue
        middle = part.vertices[part.faces].double().mean(dim=1)
        reach = (middle[:, [0, 2]] - centre.double()).norm(dim=1)
        chosen = ((middle[:, 1] > level - strip) & (middle[:, 1] < level + band)
                  & (reach < SEAM_TONE_REACH * _at(outline, turned(middle))))
        if mask is not None:
            chosen &= ~mask
        if not bool(chosen.any()):
            continue
        factor = torch.tensor(part.material.base_color[:3], dtype=torch.float32)
        texture = part.material.base_color_texture
        if texture is not None and part.uvs is not None:
            sharing.setdefault(id(texture), [texture, factor, []])[2].append((part, chosen))
        elif part.colors is not None:
            rows = torch.unique(part.faces[chosen].reshape(-1)).long()
            points = part.vertices[rows].double()
            colors = part.colors.float().clone()
            lab = _lab(colors[rows, :3] * factor)
            moved = (_sector_at(step, turned(points)) * (eased(points) * _skin_likeness(lab, skin))[:, None]).float()
            colors[rows, :3] = (_srgb(lab + moved) / factor.clamp_min(1e-6)).clamp(0, 1)
            part.colors = colors
    for texture, factor, entries in sharing.values():
        height, width = int(texture.image.shape[0]), int(texture.image.shape[1])
        shift = torch.zeros(height * width, 3, dtype=torch.float64)
        seen = torch.zeros(height * width, dtype=torch.float64)
        for part, chosen in entries:
            faces = part.faces.long()[chosen]
            corners, owner = raster.texel_triangles(part.uvs.float()[faces], width, height)
            points = part.vertices[faces].double()
            for piece in raster.coverage_chunks(corners, width, height):
                at = (piece.barycentric[..., None].double() * points[owner[piece.face]]).sum(dim=1)
                shift.index_add_(0, piece.pixel, _sector_at(step, turned(at)).double() * eased(at)[:, None])
                seen.index_add_(0, piece.pixel, torch.ones(piece.pixel.shape[0], dtype=torch.float64))
        filled = seen > 0
        if not bool(filled.any()):
            continue
        shift[filled] /= seen[filled, None]
        others = torch.zeros(height * width, dtype=torch.bool)
        for part in head.parts:
            if part.material.base_color_texture is not texture or part.uvs is None or not part.triangle_count:
                continue
            corners, _ = raster.texel_triangles(part.uvs.float()[part.faces.long()], width, height)
            for piece in raster.coverage_chunks(corners, width, height):
                others[piece.pixel] = True
        nearest = raster.nearest_filled(filled.reshape(height, width), SKIN_TEXEL_MARGIN).reshape(-1)
        rows = torch.nonzero((nearest >= 0) & (filled | ~others)).squeeze(1)
        image = texture.image.float().clone()
        flat = image[..., :3].reshape(-1, 3)
        lab = _lab(flat[rows] * factor)
        moved = shift[nearest[rows]].float() * _skin_likeness(lab, skin)[:, None].float()
        flat[rows] = (_srgb(lab + moved) / factor.clamp_min(1e-6)).clamp(0, 1)
        image[..., :3] = flat.reshape(height, width, 3)
        texture.image = image
    return {"seam toned by": round(distance, 1), "notes": []}


def head_hair(head: Model3D, level: float) -> list[torch.Tensor | None] | None:
    """Which faces of each head part are hair, grown down from the crown by colour.

    Args:
        head: The placed head.
        level: Height of the neck it sits on.

    Returns:
        A bool face mask per part, None for an untextured part; None when the head reads as bald.
    """
    live = [part for part in head.parts if part.triangle_count]
    if not live:
        return None
    top = max(float(part.vertices[:, 1].max()) for part in live)
    low_x = min(float(part.vertices[:, 0].min()) for part in live)
    high_x = max(float(part.vertices[:, 0].max()) for part in live)
    middle_x, width, span = (low_x + high_x) / 2, high_x - low_x, top - level
    if span <= 0 or width <= 0:
        return None
    found, crown_samples, face_samples = [], [], []
    for part in head.parts:
        smoothed = _smoothed_colours(part) if part.triangle_count else None
        found.append(smoothed)
        if smoothed is None:
            continue
        crown = part.vertices[part.faces].float().mean(dim=1)[:, 1] > top - CROWN_SHARE * span
        crown_samples.append(smoothed[0][crown])
        face_samples.append(smoothed[0][_front_faces(part, level, span, middle_x, width)])
    crown_samples = torch.cat(crown_samples) if crown_samples else torch.zeros(0, 3)
    face_samples = torch.cat(face_samples) if face_samples else torch.zeros(0, 3)
    if crown_samples.shape[0] < 16 or face_samples.shape[0] < 4:
        return None
    hair = crown_samples.median(dim=0).values
    skin = face_samples.median(dim=0).values
    if float((hair - skin).norm()) < BALD_DISTANCE:
        return None
    spread = (crown_samples - hair).abs().median(dim=0).values
    strict = spread * 3 + torch.tensor(HAIR_TOLERANCE)
    loose = spread * 3 + torch.tensor(HAIR_TOLERANCE) * HAIR_GROW_FACTOR
    masks: list[torch.Tensor | None] = []
    for part, smoothed in zip(head.parts, found):
        if smoothed is None:
            masks.append(None)
            continue
        colours, first, second = smoothed
        centres = part.vertices[part.faces].float().mean(dim=1)
        nearer_hair = (colours - hair).norm(dim=1) < (colours - skin).norm(dim=1)
        allowed = (((colours - hair) / loose).norm(dim=1) < 1.0) & nearer_hair
        seeds = (((colours - hair) / strict).norm(dim=1) < 1.0) & (centres[:, 1] > top - CROWN_SHARE * span)
        masks.append(_grow(seeds, allowed, first, second, HAIR_STEPS))
    return masks


def clear_hanging_hair(body: Model3D, neck: dict, colour: tuple[torch.Tensor, torch.Tensor]) -> dict:
    """Remove the body's own hair hanging below the capped neck, and close the openings it leaves.

    Args:
        body: The body after its head was cut away, changed in place.
        neck: The ring the head was cut at, as `body_neck` gives it.
        colour: The hair's colour, as `hair_colour` gives it.

    Returns:
        ``{"hair faces removed", "openings closed"}``.
    """
    middle, spread = colour
    strict = spread * 3 + torch.tensor(HAIR_TOLERANCE)
    loose = spread * 3 + torch.tensor(HAIR_TOLERANCE) * HAIR_GROW_FACTOR
    neck_y = float(neck["centre"][1])
    low = min(float(part.vertices[:, 1].min()) for part in body.parts if part.vertex_count)
    reach = neck_y - HAIR_REACH * (neck_y - low)
    removed = closed = 0
    for index, part in enumerate(body.parts):
        smoothed = _smoothed_colours(part)
        if smoothed is None:
            continue
        colours, first, second = smoothed
        centres = part.vertices[part.faces].float().mean(dim=1)
        welded, faces, _ = topology.weld(part.vertices, part.faces, tolerance=0)
        inside = (centres[:, 1] < neck_y) & (centres[:, 1] > reach)
        allowed = (((colours - middle) / loose).norm(dim=1) < 1.0) & inside
        seeds = (((colours - middle) / strict).norm(dim=1) < 1.0) & inside
        region = seeds & (welded[faces][:, :, 1] > neck_y - 1e-4 * max(neck_y - low, 1e-6)).any(dim=1)
        region = _grow(region, allowed, first, second, HAIR_STEPS)
        if bool(region.any()):
            region = _grow(region, inside, first, second, HAIR_FRINGE)
        if not bool(region.any()):
            continue
        kept = merge.subset_part(part, torch.nonzero(~region).squeeze(1))
        repaired, counts = clean.clean_part(kept, clean.CleanSettings(
            weld=False, degenerate=False, unreferenced=True, shells=False, non_manifold=False, orient=False,
            fill=True, max_hole_edges=clean.MAX_HOLE_EDGES, intersections="off", normals=False,
        ))
        repaired, capped = _cap_openings(repaired, neck_y)
        body.parts[index] = repaired
        removed += int(region.sum())
        closed += int(counts.get("holes_filled", 0)) + capped
    return {"hair faces removed": removed, "openings closed": closed}


def _cap_openings(part, below_y: float):
    """Close every opening under ``below_y`` with rings stepping in to its centre; ``(part, closed)``."""
    # Each cap vertex takes the texture coordinates and colours of the border point it steps in from.
    welded, faces, remap = topology.weld(part.vertices, part.faces, tolerance=0)
    loops = [loop for loop in topology.boundary_loops(faces, welded.shape[0])
             if loop.numel() >= 3 and float(welded[loop][:, 1].mean()) < below_y]
    if not loops:
        return part, 0
    first = torch.full((welded.shape[0],), -1, dtype=torch.long)
    first.scatter_reduce_(0, remap, torch.arange(remap.shape[0]), reduce="amin", include_self=False)
    attributes = {name: getattr(part, name) for name in ("uvs", "normals", "tangents", "colors")
                  if getattr(part, name) is not None}
    vertices, new_faces = [part.vertices.float()], [part.faces.long()]
    extra = {name: [value] for name, value in attributes.items()}
    count = part.vertex_count
    for loop in loops:
        border = first[loop]
        points = part.vertices[border].float()
        centre = points.mean(dim=0)
        rings = [border]
        for step in range(1, CAP_RINGS):
            ring = torch.arange(count, count + border.numel())
            vertices.append(points + (centre - points) * (step / CAP_RINGS))
            for name, value in attributes.items():
                extra[name].append(value[border])
            count += border.numel()
            rings.append(ring)
        middle = torch.tensor([count])
        vertices.append(centre[None])
        for name, value in attributes.items():
            extra[name].append(value[border[:1]])
        count += 1
        for outer, inner in zip(rings, rings[1:]):
            a, b = outer, outer.roll(-1)
            c, d = inner, inner.roll(-1)
            new_faces.append(torch.stack([b, a, c], dim=1))
            new_faces.append(torch.stack([b, c, d], dim=1))
        last = rings[-1]
        new_faces.append(torch.stack([last.roll(-1), last, middle.expand(last.numel())], dim=1))
    capped = part.copy()
    capped.vertices = torch.cat(vertices)
    capped.faces = torch.cat(new_faces)
    for name, values in extra.items():
        setattr(capped, name, torch.cat(values))
    capped.polygons = None
    capped.joints = capped.weights = None
    for morph in capped.morphs.values():
        grow = capped.vertex_count - morph.positions.shape[0]
        morph.positions = torch.cat([morph.positions, torch.zeros(grow, 3)])
        morph.normals = None
    return capped, len(loops)


def _surface_points(vertices: torch.Tensor, faces: torch.Tensor, count: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``count`` points spread over a surface by area, and the normal of the face each lies on."""
    corners = vertices[faces].double()
    a, b, c = corners[:, 0], corners[:, 1], corners[:, 2]
    cross = torch.linalg.cross(b - a, c - a)
    area = cross.norm(dim=1)
    generator = torch.Generator().manual_seed(0)
    pick = torch.multinomial(area.float(), count, replacement=True, generator=generator)
    u = torch.rand(count, generator=generator, dtype=torch.float64)
    v = torch.rand(count, generator=generator, dtype=torch.float64)
    flip = u + v > 1
    u, v = torch.where(flip, 1 - u, u), torch.where(flip, 1 - v, v)
    points = a[pick] + u[:, None] * (b - a)[pick] + v[:, None] * (c - a)[pick]
    return points, cross[pick] / area[pick].clamp_min(1e-30)[:, None]


def _own_head_points(vertices: torch.Tensor, faces: torch.Tensor, neck: dict) -> torch.Tensor | None:
    """Points on the front of the body's own head above its neck ring, or None when too few."""
    centre = neck["centre"].double()
    middle = vertices[faces].double().mean(dim=1)
    radial = (middle[:, [0, 2]] - centre[[0, 2]]).norm(dim=1)
    chosen = (middle[:, 1] > float(centre[1])) & (radial < float(neck["head_height"]))
    if int(chosen.sum()) < 20 or float(vertices[faces[chosen]].double().std()) <= 0:
        return None
    points, normals = _surface_points(vertices, faces[chosen], REGISTER_POINTS)
    front = normals[:, 2] > REGISTER_FRONT
    return points[front] if int(front.sum()) >= 50 else None


def _scale_shift_fit(source: torch.Tensor, target: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """The ``(4, 4)`` matrix ``x -> s * rotation @ x + t`` taking ``source`` closest onto ``target``."""
    turned = source @ rotation.T
    source_mean, target_mean = turned.mean(dim=0), target.mean(dim=0)
    xs, xt = turned - source_mean, target - target_mean
    scale = float((xs * xt).sum() / xs.pow(2).sum().clamp_min(1e-30))
    return orient.similarity(rotation, scale=scale, translation=target_mean - scale * source_mean)


def _register(tree: TriangleBVH, points: torch.Tensor, start: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Refine the size and place of ``start``, a similarity from head to body, until the head meets ``points``.

    Args:
        tree: The head's triangles, unmoved.
        points: ``(N, 3)`` points on the body's own head.
        start: ``(4, 4)`` similarity to start from.

    Returns:
        ``(matrix, error)``: the refined similarity, turned as ``start`` is, and the mean distance kept.
    """
    matrix, error = start.clone(), math.inf
    rotation = start[:3, :3] / float(torch.linalg.det(start[:3, :3])) ** (1.0 / 3.0)
    for _ in range(REGISTER_STEPS):
        inverse = torch.linalg.inv(matrix)
        query = points @ inverse[:3, :3].T + inverse[:3, 3]
        closest = tree.closest_point(query.float())[3].cpu().double()
        residual = (closest @ matrix[:3, :3].T + matrix[:3, 3] - points).norm(dim=1)
        keep = residual <= torch.quantile(residual, REGISTER_KEEP)
        error = float(residual[keep].mean())
        updated = _scale_shift_fit(closest[keep], points[keep], rotation)
        moved = float((updated - matrix).abs().max())
        matrix = updated
        if moved < 1e-7:
            break
    return matrix, error


def _ring_at(vertices: torch.Tensor, faces: torch.Tensor, level: float, centre: torch.Tensor):
    """The smallest closed section at height ``level`` around ``centre`` (X, Z), or None."""
    loops = [loop for loop in sections.slice_plane(vertices, faces, (float(centre[0]), level, float(centre[1])),
                                                   (0.0, 1.0, 0.0)) if loop.closed and abs(loop.area) > 0]
    holding = [loop for loop in loops if _inside(loop.points[:, [0, 2]], centre)]
    return min(holding, key=lambda loop: abs(loop.area)) if holding else None


def _outline(polylines: list, centre: torch.Tensor, towards: torch.Tensor | None = None,
             outer: bool = False) -> torch.Tensor:
    """Distance from ``centre`` out to polylines in the X, Z plane at each of `PLUG_BINS` angles.

    Args:
        polylines: ``(points (K, 2), closed)`` pairs.
        centre: ``(2,)`` point the angles are taken about.
        towards: ``(PLUG_BINS,)`` reach to take the closest crossing to at each angle, or None for the
            farthest.
        outer: Smooth by taking the farthest of neighbouring angles, so the outline never falls inside.

    Returns:
        ``(PLUG_BINS,)`` float64 reach per angle, gaps filled from their neighbours.
    """
    dense = []
    for points, closed in polylines:
        corners = points.double()
        if corners.shape[0] < 2:
            continue
        ends = corners.roll(-1, dims=0) if closed else corners[1:]
        starts = corners if closed else corners[:-1]
        edges = ends - starts
        lengths = edges.norm(dim=1)
        step = max(float(lengths.sum()) / (PLUG_BINS * 8), 1e-12)
        pieces = (lengths / step).ceil().clamp_min(1).long()
        which = torch.repeat_interleave(torch.arange(starts.shape[0]), pieces)
        along = torch.arange(which.shape[0]) - torch.repeat_interleave(torch.cumsum(pieces, 0) - pieces, pieces)
        dense.append(starts[which] + edges[which] * (along.double() / pieces[which].double())[:, None])
    radius = torch.full((PLUG_BINS,), -1.0, dtype=torch.float64)
    if dense:
        offset = torch.cat(dense) - centre.double()
        bins = ((torch.atan2(offset[:, 1], offset[:, 0]) + math.pi) / (2 * math.pi) * PLUG_BINS).long()
        bins = bins.clamp(0, PLUG_BINS - 1)
        reach = offset.norm(dim=1)
        if towards is None:
            radius.scatter_reduce_(0, bins, reach, reduce="amax", include_self=True)
        else:
            miss = (reach - towards.double()[bins]).abs()
            best = torch.full((PLUG_BINS,), math.inf, dtype=torch.float64)
            best.scatter_reduce_(0, bins, miss, reduce="amin", include_self=True)
            closest = miss <= best[bins]
            radius[bins[closest]] = reach[closest]
    for _ in range(PLUG_BINS):
        empty = radius < 0
        if not bool(empty.any()):
            break
        left, right = radius.roll(1), radius.roll(-1)
        known = (left >= 0).double() + (right >= 0).double()
        mean = (left.clamp_min(0) + right.clamp_min(0)) / known.clamp_min(1)
        radius = torch.where(empty & (known > 0), mean, radius)
    for _ in range(2):
        if outer:
            radius = torch.maximum(torch.maximum(radius.roll(1), radius), radius.roll(-1))
        else:
            radius = (radius.roll(1) + radius + radius.roll(-1)) / 3
    return radius


def _bins(angle: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The two outline steps each angle falls between, and the share of the way to the second."""
    position = (angle + math.pi) / (2 * math.pi) * PLUG_BINS - 0.5
    low = position.floor()
    first = low.long() % PLUG_BINS
    return first, (first + 1) % PLUG_BINS, position - low


def _at(outline: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """An outline read at each angle, between its steps."""
    first, second, share = _bins(angle)
    return outline[first] * (1 - share) + outline[second] * share


def _neck_ring(vertices: torch.Tensor, faces: torch.Tensor, level: float, centre: torch.Tensor,
               radius: float) -> dict | None:
    """The body's neck at its ring and a little below, as outlines about the ring's centre.

    Returns:
        ``centre`` (X, Z), ``outline`` at ``level``, ``under`` at `SEAM_SLOPE` radii below and ``slope``,
        that depth; None when no ring holds ``centre``.
    """
    first = _ring_at(vertices, faces, level, centre)
    if first is None:
        return None
    middle = first.centroid.double()[[0, 2]]
    polygon = first.points[:, [0, 2]].double()
    outline = _outline([(polygon, True)], middle, outer=True)
    slope = SEAM_SLOPE * radius
    lower = _ring_at(vertices, faces, level - slope, middle)
    under = _outline([(lower.points[:, [0, 2]], True)], middle) if lower is not None else outline
    under = torch.minimum(torch.maximum(under, outline), outline * SEAM_FLARE)
    return {"centre": middle, "polygon": polygon, "outline": outline, "under": under, "slope": slope}


def _seam_level(vertices: torch.Tensor, faces: torch.Tensor, head: Model3D, hair: list, centre: torch.Tensor,
                ring_level: float, radius: float) -> tuple[float, float]:
    """Where down the body's neck the head's own neck best matches it, before clothing or shoulders.

    Returns:
        ``(height, mismatch)``: the seam height and the median share by which the necks differ there.
    """
    low, high = ring_level - (SEAM_SEARCH + 0.5) * radius, ring_level + 0.5 * radius
    skin_vertices, skin_faces, offset = [], [], 0
    for part, mask in zip(head.parts, hair):
        if not part.triangle_count:
            continue
        chosen = part.faces.long() if mask is None else part.faces.long()[~mask]
        middle = part.vertices[chosen].float().mean(dim=1)[:, 1]
        skin_faces.append(chosen[(middle > low) & (middle < high)] + offset)
        skin_vertices.append(part.vertices.float())
        offset += part.vertex_count
    if not skin_faces:
        return ring_level, 0.0
    skin_vertices, skin_faces = torch.cat(skin_vertices), torch.cat(skin_faces)
    base, tried = None, []
    for step in range(SEAM_STEPS):
        height = ring_level - SEAM_SEARCH * radius * step / (SEAM_STEPS - 1)
        loop = _ring_at(vertices, faces, height, centre)
        if loop is None:
            break
        body = _outline([(loop.points[:, [0, 2]], True)], centre, outer=True)
        if base is None:
            base = body
        elif float((body > base * SEAM_FLARE).double().mean()) > 0.25:
            break
        crossing = [(piece.points[:, [0, 2]], piece.closed) for piece in sections.slice_plane(
            skin_vertices, skin_faces, (float(centre[0]), height, float(centre[1])), (0.0, 1.0, 0.0))
            if float((piece.points[:, [0, 2]].double() - centre).norm(dim=1).min()) < 3 * radius]
        if not crossing:
            break
        tube = _outline(crossing, centre, towards=body)
        tried.append((height, float(((body - tube).abs() / body.clamp_min(1e-12)).median())))
    if not tried:
        return ring_level, 0.0
    best = min(mismatch for _, mismatch in tried)
    return min((entry for entry in tried if entry[1] <= best + SEAM_SLACK), key=lambda entry: entry[0])


def _polygon_reach(polygon: torch.Tensor, centre: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Farthest distance from ``centre`` along each angle to where it leaves a closed polygon in X, Z."""
    starts = polygon.double() - centre.double()
    edges = starts.roll(-1, dims=0) - starts
    reach = torch.empty(angle.shape[0], dtype=torch.float64)
    for begin in range(0, int(angle.shape[0]), 4096):
        turn = angle[begin:begin + 4096].double()
        dx, dz = torch.cos(turn)[:, None], torch.sin(turn)[:, None]
        across = dx * edges[None, :, 1] - dz * edges[None, :, 0]
        safe = torch.where(across.abs() > 1e-15, across, torch.full_like(across, 1e-15))
        along = (starts[None, :, 0] * edges[None, :, 1] - starts[None, :, 1] * edges[None, :, 0]) / safe
        share = (starts[None, :, 0] * dz - starts[None, :, 1] * dx) / safe
        hit = (share >= 0) & (share <= 1) & (along > 0) & (across.abs() > 1e-15)
        reach[begin:begin + 4096] = torch.where(hit, along, torch.zeros_like(along)).max(dim=1).values
    return reach


def _bend_normals(part, before: torch.Tensor, moved: torch.Tensor) -> None:
    """Turn the normals of a part's ``moved`` vertices by the change in its surface since ``before``."""
    if part.normals is None or not bool(moved.any()):
        return
    welded, faces, remap = topology.weld(before, part.faces, tolerance=0)
    old = topology.vertex_normals(welded, faces)[remap]
    welded, faces, remap = topology.weld(part.vertices, part.faces, tolerance=0)
    new = topology.vertex_normals(welded, faces)[remap]
    bent = torch.nn.functional.normalize(part.normals.float() + (new - old).float(), dim=1)
    part.normals = torch.where(moved[:, None], bent, part.normals.float()).to(part.normals.dtype)


def _ease_body(body: Model3D, ring: dict, meet: torch.Tensor, level: float, radius: float) -> float:
    """Bring the body's neck top, its cut edge and cap, onto ``meet`` at the seam, fading out below.

    Returns:
        The median share the body's neck moved at the seam.
    """
    centre, outline = ring["centre"], ring["outline"]
    depth = SEAM_BODY_BLEND * radius
    for part in body.parts:
        if not part.vertex_count:
            continue
        points = part.vertices.double()
        band = (points[:, 1] > level - depth) & (points[:, 1] <= level + 1e-5 * radius)
        if not bool(band.any()):
            continue
        offset = points[band][:, [0, 2]] - centre
        distance = offset.norm(dim=1)
        angle = torch.atan2(offset[:, 1], offset[:, 0])
        neck = _at(outline, angle).clamp_min(1e-12)
        on_neck = distance < neck * SEAM_FLARE
        weight = ((points[band, 1] - (level - depth)) / depth).clamp(0.0, 1.0)
        weight = weight * weight * (3 - 2 * weight) * on_neck.double()
        scale = 1 + (_at(meet, angle) / neck - 1) * weight
        before = part.vertices.clone()
        moved = centre + offset * scale[:, None]
        rows = torch.nonzero(band).squeeze(1)
        points[rows, 0], points[rows, 2] = moved[:, 0], moved[:, 1]
        part.vertices = points.to(part.vertices.dtype)
        _bend_normals(part, before, band)
    return round(float(((meet - outline).abs() / outline.clamp_min(1e-12)).median()), 3)


def _cut_head(body: Model3D, level: float, ring: dict, reach: float) -> dict:
    """Remove the body's own head above ``level``, keeping hoods, collars and shoulders that rise past it.

    Args:
        body: The body, changed in place.
        level: Height of the cut.
        ring: The neck at the cut, as `_neck_ring` answers.
        reach: Distance from the neck's axis inside which a loose piece wholly above the cut is removed.

    Returns:
        Counts: ``split_faces``, ``capped``, ``uncapped`` and ``kept_above``, faces kept above the cut.
    """
    point = torch.tensor([0.0, level, 0.0], dtype=torch.float64)
    up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    centre, polygon = ring["centre"].double(), ring["polygon"].double()
    totals = {"split_faces": 0, "capped": 0, "uncapped": 0, "kept_above": 0}
    parts, source = [], []
    for index, part in enumerate(body.parts):
        if part.triangle_count == 0:
            parts.append(part)
            source.append([index])
            continue
        surface = clean.surface_of(part)
        split, side = cut.split_by_plane(surface, point, up)
        above = torch.nonzero(side > 0).squeeze(1)
        if above.numel() == 0:
            parts.append(part)
            source.append([index])
            continue
        ids, positions = split.topology()
        welded = ids[split.faces[above]]
        pieces = topology.face_components(welded, positions.shape[0])
        count = int(pieces.max()) + 1
        span = float((positions.amax(dim=0) - positions.amin(dim=0)).norm())
        tolerance = 1e-6 * max(span, 1e-12)
        corners = positions.double()[welded]
        on_plane = (corners[..., 1] - level).abs() <= tolerance
        meets = torch.zeros_like(on_plane)
        if bool(on_plane.any()):
            offset = corners[on_plane][:, [0, 2]] - centre
            angle = torch.atan2(offset[:, 1], offset[:, 0])
            meets[on_plane] = offset.norm(dim=1) <= _polygon_reach(polygon, centre, angle) * HEAD_CUT_GIVE + tolerance
        head_piece = torch.zeros(count, dtype=torch.bool)
        head_piece[pieces[meets.any(dim=1)]] = True
        touching = torch.zeros(count, dtype=torch.bool)
        touching[pieces[on_plane.any(dim=1)]] = True
        middle = corners.mean(dim=1)[:, [0, 2]]
        totals_xz = torch.zeros(count, 2, dtype=torch.float64).index_add_(0, pieces, middle)
        sizes = torch.zeros(count, dtype=torch.float64).index_add_(0, pieces, torch.ones_like(pieces, dtype=torch.float64))
        near = ((totals_xz / sizes[:, None]) - centre).norm(dim=1) < reach
        removed = (head_piece | (~touching & near))[pieces]
        keep = torch.ones(split.face_count, dtype=torch.bool)
        keep[above[removed]] = False
        totals["split_faces"] += split.face_count - surface.face_count
        totals["kept_above"] += int((~removed).sum())
        kept, _ = clean.compact(clean.remove_faces(split, keep))
        if kept.face_count == 0:
            continue
        kept, capped, failed = cut.cap_openings(kept, point, up, tolerance)
        totals["capped"] += capped
        totals["uncapped"] += failed
        parts.append(clean.part_of(part, kept))
        source.append([index])
    merge.reindex_parts(body, parts, source)
    return totals


def _seam_normals(body: Model3D, level: float, centre: torch.Tensor, radius: float,
                  outline: torch.Tensor) -> torch.Tensor | None:
    """``(PLUG_BINS, 3)`` the body's neck normal just below the seam at each angle, or None without normals."""
    total = torch.zeros(PLUG_BINS, 3, dtype=torch.float64)
    count = torch.zeros(PLUG_BINS, dtype=torch.float64)
    for part in body.parts:
        if part.normals is None or not part.vertex_count:
            continue
        points = part.vertices.double()
        offset = points[:, [0, 2]] - centre.double()
        angle = torch.atan2(offset[:, 1], offset[:, 0])
        normals = part.normals.double()
        chosen = ((points[:, 1] > level - SEAM_SHADE * radius) & (points[:, 1] < level - 1e-4 * radius)
                  & (offset.norm(dim=1) < _at(outline, angle) * SEAM_FLARE) & (normals[:, 1].abs() < 0.9))
        if not bool(chosen.any()):
            continue
        bins = ((angle[chosen] + math.pi) / (2 * math.pi) * PLUG_BINS).long().clamp(0, PLUG_BINS - 1)
        total.index_add_(0, bins, normals[chosen])
        count.index_add_(0, bins, torch.ones(bins.shape[0], dtype=torch.float64))
    if float(count.sum()) < 16:
        return None
    mean = total / count.clamp_min(1.0)[:, None]
    known = count > 0
    for _ in range(PLUG_BINS):
        if bool(known.all()):
            break
        left, right = known.roll(1), known.roll(-1)
        fill = ~known & (left | right)
        both = (mean.roll(1, dims=0) * left[:, None] + mean.roll(-1, dims=0) * right[:, None])
        mean[fill] = both[fill]
        known = known | fill
    for _ in range(2):
        mean = mean.roll(1, dims=0) + mean + mean.roll(-1, dims=0)
    return torch.nn.functional.normalize(mean, dim=1)


def _plug(head: Model3D, body: Model3D, ring: dict, level: float, radius: float, hair: list) -> dict:
    """Cut the head's skin at the seam and meet the two necks there, keeping the hair whole.

    Args:
        head: The placed head, changed in place.
        body: The body, already cut at ``level``, changed in place.
        ring: The body's neck, as `_neck_ring` reads it.
        level: Height of the body's ring.
        radius: The body's neck radius.
        hair: Hair face mask per head part, None where unknown; replaced to match the joined parts.

    Returns:
        Figures for the fit; ``notes`` holds a line when the head could not be joined.
    """
    blend = SEAM_BLEND * radius
    top = level + blend
    centre = ring["centre"]
    notes: list[str] = []
    if all(mask is None for mask in hair) and any(_face_colours(part) is None for part in head.parts
                                                  if part.triangle_count):
        notes.append("the head has no colours to tell its hair by, so all of it below the neck was cut")
    pieces = []
    for index, part in enumerate(head.parts):
        if not part.triangle_count:
            pieces.append((None, None))
            continue
        mask = hair[index]
        if mask is None or not bool(mask.any()):
            pieces.append((part, None))
        elif bool(mask.all()):
            pieces.append((None, part))
        else:
            pieces.append((merge.subset_part(part, torch.nonzero(~mask).squeeze(1)),
                           merge.subset_part(part, torch.nonzero(mask).squeeze(1))))
    skins = [skin for skin, _ in pieces if skin is not None]
    if not skins:
        return {"notes": notes + ["the head reads as all hair, so it was set on the neck unjoined"]}
    everything = [piece for pair in pieces for piece in pair if piece is not None]
    bottom = min(float(skin.vertices[:, 1].min()) for skin in skins)
    stretched = 0.0
    if bottom > level - 1e-3 * radius:
        if bottom >= level + 0.5 * blend:
            return {"notes": notes + ["the head's neck ends above the body's neck, so it was set on it unjoined; "
                                      "generate the head with a longer neck"]}
        floor = level - SEAM_SPARE * radius
        for piece in everything:
            heights = piece.vertices[:, 1].double()
            lower = heights < top
            drawn = top - (top - heights) * (top - floor) / (top - bottom)
            below_neck = heights < bottom
            drawn[below_neck] = heights[below_neck] - (bottom - floor)
            heights[lower] = drawn[lower]
            piece.vertices[:, 1] = heights.to(piece.vertices.dtype)
        stretched = bottom - floor
    faces_before = sum(skin.triangle_count for skin in skins)
    cut_skin = Model3D(parts=skins)
    cut.cut_model(cut_skin, list(range(len(skins))), torch.tensor([0.0, level, 0.0]),
                  torch.tensor([0.0, 1.0, 0.0]), keep="above", cap=False)
    skins = [skin for skin in cut_skin.parts if skin.triangle_count]
    if not skins:
        return {"notes": notes + ["the head lies wholly below the body's neck, so it was left unjoined"]}
    trimmed = faces_before - sum(skin.triangle_count for skin in skins)
    found = iter(cut_skin.parts)
    pieces = [((next(found) if skin is not None else None), hair_part) for skin, hair_part in pieces]
    probe = level + 0.03 * radius
    crossing = []
    for skin in skins:
        crossing += [(loop.points[:, [0, 2]], loop.closed) for loop in sections.slice_plane(
            skin.vertices, skin.faces, (float(centre[0]), probe, float(centre[1])), (0.0, 1.0, 0.0))
            if float((loop.points[:, [0, 2]].double() - centre).norm(dim=1).min()) < 3 * radius]
    if not crossing:
        return {"notes": notes + ["the head's neck does not pass the body's ring, so it was set on it unjoined"]}
    tube = _outline(crossing, centre, towards=ring["outline"])
    meet = torch.minimum(torch.maximum(ring["outline"], tube / SEAM_GIVE), tube * SEAM_GIVE)
    settle = float((meet - tube).median())
    taken = _ease_body(body, ring, meet, level, radius)
    slope = (ring["under"] - ring["outline"]) / ring["slope"]
    body_normals = _seam_normals(body, level, centre, radius, ring["outline"])
    edge = 1e-5 * radius
    for skin, hair_part in pieces:
        for piece, is_hair in ((skin, False), (hair_part, True)):
            if piece is None or not piece.vertex_count:
                continue
            points = piece.vertices.double()
            before = piece.vertices.clone()
            moving = points[:, 1] < top
            if not bool(moving.any()):
                continue
            heights = points[moving, 1]
            weight = ((top - heights) / blend).clamp(0.0, 1.0)
            weight = weight * weight * (3 - 2 * weight)
            offset = points[moving][:, [0, 2]] - centre
            distance = offset.norm(dim=1).clamp_min(1e-12)
            angle = torch.atan2(offset[:, 1], offset[:, 0])
            if is_hair:
                reach = distance + settle * weight
            else:
                seam = (_polygon_reach(ring["polygon"], centre, angle) * _at(meet, angle)
                        / _at(ring["outline"], angle).clamp_min(1e-12))
                own = _at(tube, angle).clamp_min(1e-12)
                near = ((SEAM_FAR - distance / own) / (SEAM_FAR - SEAM_NEAR)).clamp(0.0, 1.0)
                reach = distance + (_at(meet, angle) - own) * weight * near
                rim = heights <= level + edge
                reach[rim] = (distance + (seam - distance) * near)[rim]
            moved = centre + offset / distance[:, None] * reach.clamp_min(0.05 * radius)[:, None]
            points[moving, 0], points[moving, 2] = moved[:, 0], moved[:, 1]
            piece.vertices = points.to(piece.vertices.dtype)
            _bend_normals(piece, before, moving)
            if piece.normals is not None and not is_hair:
                shade = ((level + SEAM_SHADE * radius - heights) / (SEAM_SHADE * radius)).clamp(0.0, 1.0)
                shade = (shade * shade * (3 - 2 * shade)).float()
                if body_normals is not None:
                    first, second, share = _bins(angle)
                    towards = body_normals[first] * (1 - share[:, None]) + body_normals[second] * share[:, None]
                else:
                    towards = torch.stack([torch.cos(angle), _at(slope, angle), torch.sin(angle)], dim=1)
                towards = torch.nn.functional.normalize(towards.float(), dim=1)
                normals = piece.normals.float().clone()
                normals[moving] = torch.nn.functional.normalize(
                    normals[moving] * (1 - shade[:, None]) + towards * shade[:, None], dim=1)
                piece.normals = normals.to(piece.normals.dtype)
    hair_faces = 0
    for index, (skin, hair_part) in enumerate(pieces):
        kept = [piece for piece in (skin, hair_part) if piece is not None and piece.triangle_count]
        if not kept:
            hair[index] = None
            continue
        joined = kept[0] if len(kept) == 1 else merge.merge_parts(kept, name=head.parts[index].name)
        joined.label = head.parts[index].label
        mask = torch.cat([torch.full((piece.triangle_count,), piece is hair_part, dtype=torch.bool)
                          for piece in kept])
        hair_faces += int(mask.sum())
        head.parts[index] = joined
        hair[index] = mask if bool(mask.any()) else None
    eased = float(((meet - tube).abs() / tube.clamp_min(1e-12)).median())
    return {"head joined": True, "neck eased by": round(eased, 3), "body neck eased by": taken,
            "head hair faces": hair_faces,
            "head faces trimmed": trimmed, "neck stretched": round(stretched / radius, 3), "notes": notes}


def _label(model: Model3D, label: str, overwrite: bool) -> None:
    """Give every part ``label``, keeping labels already set unless ``overwrite``."""
    for part in model.parts:
        if overwrite or not part.label:
            part.label = label


def assemble(body: Model3D, head: Model3D | None = None, hair: Model3D | None = None,
             extra: Model3D | None = None, extra_label: str = "extra", extra_follows: str = "head",
             target_height: float = 1.6, facing: str = "auto", remove_body_head: str = "auto",
             clear_body_hair: str = "auto", head_fit: str = "auto", head_scale: float = 1.0, head_turn: float = 0.0,
             hair_follows_head: bool = True, feet_on_ground: bool = True,
             placement: str = "", skin_match: str = "head to body") -> tuple[Model3D, dict]:
    """Join a body, head, hair and extra into one model with labelled parts.

    Args:
        body: The body; a model already assembled may be wired again to add parts.
        head: A separately made head, or None.
        hair: Hair made with the head, or None.
        extra: One more part, or None.
        extra_label: Label given to the extra's parts.
        extra_follows: ``"head"`` or ``"body"``.
        target_height: Body height in metres; 0 keeps its size.
        facing: One of `FACINGS`.
        remove_body_head: One of `REMOVE_HEAD`.
        clear_body_hair: One of `CLEAR_HAIR`.
        head_fit: One of `HEAD_FITS`.
        head_scale: Multiplier on the fitted head size.
        head_turn: Degrees about the head's up axis, added before fitting.
        hair_follows_head: Hair moves with the head; otherwise with the body.
        feet_on_ground: Stand the body on Y = 0, centred in X and Z.
        placement: Per-role adjustments, as `parse_placement` reads them.
        skin_match: One of `SKIN_MATCHES`.

    Returns:
        ``(model, fit)``: the joined model and the figures of the fit.

    Raises:
        ValueError: A model has no triangles, the head has no neck ring, or a value is invalid.
    """
    adjustments = parse_placement(placement)
    fit: dict = {"notes": []}
    vertices, faces = _geometry(body)
    turn, how = _body_turn(body, facing, vertices, faces)
    low, high = body.bounds()
    centre = ((low + high) / 2).double()
    centre[1] = float(low[1])
    body_matrix = _about(centre, orient.similarity(orient.rotation_matrix(0.0, turn, 0.0)))
    if feet_on_ground:
        body_matrix = orient.similarity(translation=-centre) @ body_matrix
    orient.transform(body, body_matrix)
    if not how.startswith("taken as"):
        body.meta[orient.FACES_FORWARD_KEY] = True
    fit.update(body_turn=round(turn, 2), facing=how)
    _label(body, "body", overwrite=False)

    head_matrix = body_matrix
    neck_pivot = None
    if head is not None:
        neck = body_neck(body)
        fit.update(neck_found=neck["found"], neck_centre=[round(float(v), 4) for v in neck["centre"]],
                   neck_radius=round(neck["radius"], 4))
        removing = remove_body_head == "yes" or (remove_body_head == "auto" and neck["own_head"])
        joining = head_fit != "keep" and (removing or not neck["own_head"])
        ring_level = float(neck["centre"][1]) - (0.0 if neck["own_head"] else RING_INSET * neck["radius"])
        level = ring_level
        body_vertices, body_faces = _geometry(body)
        own_points = _own_head_points(body_vertices, body_faces, neck) if neck["own_head"] else None
        skin = body_skin(body, neck) if skin_match == "head to body" else None
        colour = hair_colour(body, neck) if removing and clear_body_hair == "auto" else None
        orient.transform(head, head_matrix)
        if abs(head_turn) > 1e-6:
            hl, hh = head.bounds()
            orient.transform(head, _about(((hl + hh) / 2).double(),
                                          orient.similarity(orient.rotation_matrix(0.0, head_turn, 0.0))))
            head_matrix = _about(((hl + hh) / 2).double(),
                                 orient.similarity(orient.rotation_matrix(0.0, head_turn, 0.0))) @ head_matrix
        if head_fit != "keep":
            hv, hf = _geometry(head)
            try:
                frame = sections.head_frame(hv, hf, forward=(0.0, 0.0, 1.0))
            except ValueError as error:
                raise ValueError(f"the head has no neck to fit by ({error}); set head_fit to keep and place it "
                                 "with placement") from None
            up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
            ring_found = _axis_neck(hv, hf, BUST_NECK_BAND)
            opening = _opening(hv, hf, (0.0, 0.45), (0.03, 0.45)) if ring_found is None else None
            if ring_found is not None:
                head_centre, head_up, head_radius = ring_found[0], up, ring_found[2]
            elif opening is not None:
                head_centre, head_up, head_radius = opening[0], up, opening[2]
            else:
                head_centre, head_up, head_radius = frame.neck_centre.double(), up, frame.neck_radius
            if head_radius <= 1e-9:
                raise ValueError("the head's neck ring has no size; set head_fit to keep")
            by_body = head_fit == "body's head" or (head_fit == "auto" and own_points is not None)
            if by_body and own_points is None:
                raise ValueError("the body has no head of its own to fit the new head to; set head_fit to "
                                 "neck width")
            if by_body:
                up = torch.nn.functional.normalize(head_up.double(), dim=0)
                reach = float(((hv.double() - head_centre.double()) @ up).max())
                if reach <= 1e-9:
                    raise ValueError("the head has no height above its neck; set head_fit to neck width")
                size = float(neck["head_height"]) / reach
            else:
                size = neck["radius"] / head_radius
            target = neck["centre"].double()

            def placed(times: float) -> torch.Tensor:
                return (orient.similarity(translation=target) @ orient.similarity(scale=size * times)
                        @ orient.similarity(translation=-head_centre.double()))

            fitted = placed(1.0)
            if by_body:
                tree = TriangleBVH(hv.float(), hf)
                refined, error = min((_register(tree, own_points, placed(times)) for times in REGISTER_STARTS),
                                     key=lambda result: result[1])
                ratio = (float(torch.linalg.det(refined[:3, :3])) / float(torch.linalg.det(fitted[:3, :3]))) ** (1 / 3)
                share = error / float(neck["head_height"])
                if REGISTER_SCALE[0] <= ratio <= REGISTER_SCALE[1] and share <= REGISTER_ERROR:
                    fitted = refined
                    fit["head fit error"] = round(share, 4)
                else:
                    fit["notes"].append(f"the head did not settle over the body's own head (x{ratio:.2f}, "
                                        f"{share:.0%} of its height apart), so it was sized by height instead")
            fitted = _about(target, orient.similarity(scale=float(head_scale))) @ fitted
            orient.transform(head, fitted)
            head_matrix = fitted @ head_matrix
            fit.update(head_scale=round(float(torch.linalg.det(fitted[:3, :3])) ** (1.0 / 3.0), 5),
                       head_neck_radius=round(head_radius, 4), head_open_neck=opening is not None)
            fit["head sized by"] = "fitted to the body's own head" if by_body else "the body's neck"
            neck_pivot = target
        else:
            neck_pivot = neck["centre"]
        adjust = _adjust(adjustments.get("head"), neck_pivot)
        orient.transform(head, adjust)
        head_matrix = adjust @ head_matrix
        hair_masks = None
        if joining or (skin_match == "head to body" and skin is not None):
            hair_masks = head_hair(head, ring_level) or [None] * len(head.parts)
        centre = neck["centre"].double()[[0, 2]]
        ring = None
        if joining:
            level, mismatch = _seam_level(body_vertices, body_faces, head, hair_masks, centre, ring_level,
                                          neck["radius"])
            fit.update({"seam below the ring": round((ring_level - level) / neck["radius"], 2),
                        "neck mismatch": round(mismatch, 3)})
            ring = _neck_ring(body_vertices, body_faces, level, centre, neck["radius"])
        if removing or (joining and ring is not None):
            at_cut = ring if ring is not None else _neck_ring(body_vertices, body_faces, level, centre,
                                                              neck["radius"])
            if at_cut is not None:
                reach = HEAD_LOOSE_REACH * float(neck["head_height"] or 6.0 * neck["radius"])
                counts = _cut_head(body, level, at_cut, reach)
                fit["body faces kept above the seam"] = counts["kept_above"]
            else:
                counts = cut.cut_model(body, list(range(len(body.parts))), torch.tensor([0.0, level, 0.0]),
                                       torch.tensor([0.0, 1.0, 0.0]), keep="below", cap=True)
            fit["cut"] = {key: value for key, value in counts.items() if isinstance(value, (int, float))}
            if removing:
                fit["body head removed"] = True
                if colour is not None:
                    seam_neck = dict(neck, centre=torch.tensor([float(neck["centre"][0]), level,
                                                                float(neck["centre"][2])], dtype=torch.float64))
                    fit.update(clear_hanging_hair(body, seam_neck, colour))
                elif clear_body_hair == "auto":
                    fit["notes"].append("the body's hair colour could not be read, so its hair was left")
        if joining and ring is not None:
            joined = _plug(head, body, ring, level, neck["radius"], hair_masks)
            fit["notes"].extend(joined.pop("notes"))
            fit.update(joined)
        elif joining:
            fit["notes"].append("the body's neck could not be traced, so the head was set on it unjoined")
        if skin_match == "head to body":
            if skin is None:
                fit["notes"].append("the body's skin could not be read by the neck, so the head's skin was left")
            else:
                toned = match_skin(head, skin, ring_level, hair_masks)
                fit["notes"].extend(toned.pop("notes"))
                fit.update(toned)
            if joining and ring is not None:
                seamed = tone_seam(head, body, level, ring, neck["radius"], hair_masks)
                fit["notes"].extend(seamed.pop("notes"))
                fit.update(seamed)
        _label(head, "head", overwrite=True)
    elif "head" in adjustments:
        fit["notes"].append("placement for head was ignored: no head is wired")

    body_adjust = _adjust(adjustments.get("body"), torch.zeros(3, dtype=torch.float64))
    orient.transform(body, body_adjust)
    ring_point = None
    if head is not None:
        ring_point = body_adjust @ torch.tensor([float(neck["centre"][0]), ring_level, float(neck["centre"][2]), 1.0],
                                                dtype=torch.float64)

    pieces = []
    if head is not None:
        orient.transform(head, body_adjust)
        pieces.append(head)
    pivot = neck_pivot if neck_pivot is not None else torch.zeros(3, dtype=torch.float64)
    for model, role, label, follows_head in (
        (hair, "hair", "hair", hair_follows_head),
        (extra, "extra", str(extra_label or "extra").strip()[:64] or "extra", extra_follows == "head"),
    ):
        if model is None:
            continue
        matrix = head_matrix if follows_head and head is not None else body_matrix
        orient.transform(model, matrix)
        orient.transform(model, _adjust(adjustments.get(role), pivot))
        orient.transform(model, body_adjust)
        _label(model, label, overwrite=True)
        pieces.append(model)

    for piece in pieces:
        merge.add_model(body, piece)
    if ring_point is not None and body.height() > 1e-9:
        low, _ = body.bounds()
        body.meta[NECK_RING_KEY] = round((float(ring_point[1]) - float(low[1])) / body.height(), 5)
    if target_height > 0 and body.height() > 1e-9:
        fit["scale"] = round(orient.scale_to_height(body, float(target_height)), 5)
    if feet_on_ground:
        orient.place(body, "base")
    fit["parts"] = len(body.parts)
    fit["height_m"] = round(body.height(), 4)
    return body, fit

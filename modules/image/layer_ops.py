"""Editing a ComfyUI ``LAYERS`` document: selection, order, duplication and flattening.

A document is ``{"version", "canvas", "layers"}``. Rotation is in radians and a layer's
mask is ``1`` where its picture is cut away.
"""

from __future__ import annotations

__all__ = [
    "BLEND_MODES",
    "MATCHES",
    "MOVES",
    "VISIBILITIES",
    "FITS",
    "Frame",
    "aligned",
    "box_of",
    "composited",
    "described",
    "drawn",
    "framed",
    "drawn_size",
    "fitted",
    "duplicated",
    "entries",
    "found",
    "longest",
    "matching",
    "merged",
    "moved",
    "placed",
    "rebuilt",
    "report",
    "size_of",
    "trimmed",
]

import math
from typing import NamedTuple

import torch

from .. import log
from ..interface import run_result
from . import blend_modes, dynamic

#: Blend modes a layer may carry, in the order the compositor lists them.
BLEND_MODES = blend_modes.MODES

#: How a name is compared against a layer's own, in menu order.
MATCHES = ("contains", "exact")

#: Which layers a filter keeps by their visibility flag, in menu order.
VISIBILITIES = ("any", "visible only", "hidden only")

#: How a layer travels through the stack, in menu order.
MOVES = (
    "to front",
    "to back",
    "up one",
    "down one",
    "to index",
    "sort by name, a to z",
    "sort by name, z to a",
    "sort by area, smallest first",
    "sort by area, largest first",
)

#: Below this a division is read as a division by zero.
EPSILON = blend_modes.EPSILON

#: How a layer reaches the box it is fitted into, in menu order.
FITS = ("fit inside", "fill and overflow", "stretch")

logger = log.get_logger("image.layer_ops")


def _whole(value, fallback: int = 0) -> int:
    """One number as a whole number, or ``fallback`` where it is not one."""
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return fallback


def _number(value, fallback: float = 0.0) -> float:
    """One number as a float, or ``fallback`` where it is not one."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _frames(entry) -> int:
    """How many pictures one layer carries."""
    picture = entry.get("image")
    if not isinstance(picture, torch.Tensor):
        return 0
    while picture.ndim > 4:
        picture = picture[0]
    return int(picture.shape[0]) if picture.ndim == 4 else 1


def _picture(entry, frame: int = 0) -> "torch.Tensor | None":
    """One of a layer's pictures as ``(height, width, channels)``."""
    picture = entry.get("image")
    if not isinstance(picture, torch.Tensor):
        return None
    while picture.ndim > 4:
        picture = picture[0]
    if picture.ndim == 4:
        if int(picture.shape[0]) == 0:
            return None
        picture = picture[min(max(0, frame), int(picture.shape[0]) - 1)]
    return picture


def _cut(entry, frame: int = 0) -> "torch.Tensor | None":
    """One of a layer's mask planes as ``(height, width)``, or None where it has none."""
    cut = entry.get("mask")
    if not isinstance(cut, torch.Tensor):
        return None
    while cut.ndim > 3:
        cut = cut[0]
    if cut.ndim < 3:
        return cut
    if int(cut.shape[0]) == 1:
        return cut[0]
    return cut[frame] if frame < int(cut.shape[0]) else None


def _drawn_size(entry) -> tuple[int, int]:
    """The width and height a layer is drawn at, before any rotation."""
    picture = _picture(entry)
    natural_w = int(picture.shape[1]) if picture is not None else 1
    natural_h = int(picture.shape[0]) if picture is not None else 1
    width = _whole(entry.get("w")) or natural_w
    height = _whole(entry.get("h")) or natural_h
    return max(1, width), max(1, height)


def entries(document) -> list[dict]:
    """Every layer of a document as its own dictionary, lowest in the stack first.

    Args:
        document: A ``LAYERS`` value, or a bare list of layer dictionaries.

    Returns:
        A shallow copy of each entry carrying a picture, sorted by ``z_index``.
    """
    holder = document.get("layers", []) if isinstance(document, dict) else document
    kept = [
        dict(entry)
        for entry in (holder or [])
        if isinstance(entry, dict) and _frames(entry) > 0
    ]
    return sorted(kept, key=lambda entry: _whole(entry.get("z_index")))


def rebuilt(document, layers) -> dict:
    """A document carrying these layers, numbered from 0 at the back.

    Args:
        document: The document they came out of, read for its canvas.
        layers: The layer dictionaries to carry, lowest in the stack first.

    Returns:
        A new ``LAYERS`` document whose entries each carry a type and a contiguous
        ``z_index``.
    """
    stacked = []
    for position, entry in enumerate(layers):
        item = dict(entry)
        item["z_index"] = position
        item.setdefault("type", "raster")
        stacked.append(item)
    built = {"version": 1, "layers": stacked}
    canvas = document.get("canvas") if isinstance(document, dict) else None
    if isinstance(canvas, (tuple, list)) and len(canvas) == 2:
        width, height = _whole(canvas[0]), _whole(canvas[1])
        if width > 0 and height > 0:
            built["canvas"] = (width, height)
    return built


def size_of(document) -> tuple[int, int]:
    """The canvas a document names, or the box its layers reach once placed.

    Args:
        document: A ``LAYERS`` value.

    Returns:
        ``(width, height)`` in pixels, at least one in each direction.
    """
    named = document.get("canvas") if isinstance(document, dict) else None
    if isinstance(named, (tuple, list)) and len(named) == 2:
        width, height = _whole(named[0]), _whole(named[1])
        if width > 0 and height > 0:
            return width, height

    width = height = 1
    for entry in entries(document):
        drawn_w, drawn_h = _drawn_size(entry)
        left, top, box_w, box_h = _bounds(
            _whole(entry.get("x")),
            _whole(entry.get("y")),
            drawn_w,
            drawn_h,
            _number(entry.get("rotation"), 0.0),
        )
        width = max(width, left + box_w)
        height = max(height, top + box_h)
    return width, height


class Frame(NamedTuple):
    """One picture of a layer, at the size, angle and flip the compositor draws it.

    Attributes:
        image: ``(height, width, 3)`` picture codes.
        coverage: ``(height, width)``, 1 where the layer paints.
        x: Left edge of the drawn picture on the canvas.
        y: Top edge of the drawn picture on the canvas.
        name: What the compositor calls the layer, or an empty string.
        visible: Whether the compositor draws it.
    """

    image: "torch.Tensor"
    coverage: "torch.Tensor"
    x: int
    y: int
    name: str
    visible: bool


def drawn(document) -> list[Frame]:
    """Every picture of every layer, drawn as the compositor draws it.

    Args:
        document: A ``LAYERS`` value, or a bare list of layer dictionaries.

    Returns:
        One :class:`Frame` per picture, lowest in the stack first. A layer carrying a
        batch answers one frame per picture, in batch order.
    """
    found = []
    for entry in entries(document):
        name = str(entry.get("name") or "")
        visible = bool(entry.get("visible", True))
        for index in range(_frames(entry)):
            picture, x, y = _rendered(entry, index)
            found.append(
                Frame(
                    image=picture[..., :3],
                    coverage=picture[..., 3],
                    x=x,
                    y=y,
                    name=name,
                    visible=visible,
                )
            )
    return found


def placed(frame: Frame, width: int, height: int, channels: int):
    """One drawn frame laid onto a transparent canvas at its own placement.

    Args:
        frame: The frame to place.
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        channels: Channels the result carries.

    Returns:
        ``(image, coverage)``. The image is ``(height, width, channels)`` and the
        coverage ``(height, width)``, 1 where the layer paints.
    """
    picture = frame.image
    canvas = torch.zeros((height, width, channels), dtype=picture.dtype, device=picture.device)
    cover = torch.zeros((height, width), dtype=picture.dtype, device=picture.device)

    left, top = max(0, frame.x), max(0, frame.y)
    right = min(width, frame.x + int(picture.shape[1]))
    bottom = min(height, frame.y + int(picture.shape[0]))
    if right <= left or bottom <= top:
        return canvas, cover

    rows = slice(top - frame.y, bottom - frame.y)
    columns = slice(left - frame.x, right - frame.x)
    patch = picture[rows, columns]
    if int(patch.shape[2]) < channels:
        patch = patch.repeat(1, 1, channels // int(patch.shape[2]) + 1)
    canvas[top:bottom, left:right] = patch[:, :, :channels]
    cover[top:bottom, left:right] = frame.coverage[rows, columns]
    return canvas, cover


def longest(document) -> int:
    """How many pictures the layer carrying the most holds.

    Args:
        document: A ``LAYERS`` value, or a bare list of layer dictionaries.

    Returns:
        The largest batch any layer carries, and 0 for a document with no layers.
    """
    return max((_frames(entry) for entry in entries(document)), default=0)


def framed(document, frame: int) -> dict:
    """A document with every layer holding one of its pictures.

    Args:
        document: A ``LAYERS`` value.
        frame: Which picture, counting 0. A layer with fewer holds its last.

    Returns:
        A new ``LAYERS`` document carrying the same canvas, whose layers each hold a
        single picture and a single mask plane.
    """
    picked = []
    for entry in entries(document):
        held = dict(entry)
        total = _frames(entry)
        if total:
            index = min(max(0, frame), total - 1)
            picture = _picture(entry, index)
            held["image"] = picture.unsqueeze(0)
            cut = _cut(entry, index)
            held["mask"] = None if cut is None else cut.unsqueeze(0)
        picked.append(held)
    return rebuilt(document, picked)


def composited(layers, width: int, height: int):
    """Every visible layer flattened onto the whole canvas.

    Args:
        layers: The layer dictionaries, lowest in the stack first.
        width: Canvas width in pixels.
        height: Canvas height in pixels.

    Returns:
        ``(image, coverage)``. The image is ``(height, width, 3)`` picture codes and the
        coverage ``(height, width)``, 1 where a layer painted. A stack carrying light
        above white comes back on the scale it arrived on.
    """
    scale = dynamic.peak(*[entry.get("image") for entry in layers])
    canvas, _box = _flattened(layers, width, height, scale)
    image = _to_srgb(canvas[..., :3].clamp(0.0, 1.0)).clamp(0.0, 1.0)
    return (image * scale if scale != 1.0 else image), canvas[..., 3].clamp(0.0, 1.0)


def report(node: str, summary: str, layers, counts=None, facts=None) -> None:
    """Publish what a stack edit did, for the node's own interface to draw. Never raises.

    Args:
        node: The name of the node reporting.
        summary: One line saying what it did.
        layers: The document it answered, read for its canvas and its layer count.
        counts: Named numbers to draw beside the summary.
        facts: Named strings to draw under them.
    """
    try:
        if not run_result.watching():
            return
        stack = entries(layers)
        width, height = size_of(layers)
        run_result.publish(
            status=run_result.OK,
            summary=summary,
            counts={"layers": len(stack), **(counts or {})},
            facts={"canvas": f"{width}x{height}", **(facts or {})},
        )
    except Exception as error:
        logger.debug("%s published no report (%s)", node, error)


def drawn_size(entry) -> tuple[int, int]:
    """The width and height a layer is drawn at, before any rotation.

    Args:
        entry: A layer dictionary.

    Returns:
        ``(width, height)`` in pixels, at least one in each direction.
    """
    return _drawn_size(entry)


def box_of(entry) -> tuple[int, int, int, int]:
    """The rectangle a layer covers on the canvas once drawn and turned.

    Args:
        entry: A layer dictionary.

    Returns:
        ``(x, y, width, height)`` in pixels.
    """
    width, height = _drawn_size(entry)
    return _bounds(
        _whole(entry.get("x")),
        _whole(entry.get("y")),
        width,
        height,
        _number(entry.get("rotation"), 0.0),
    )


def aligned(entry, into: tuple[int, int, int, int], align: tuple[float, float]) -> tuple[int, int]:
    """Where a layer's ``x`` and ``y`` go for its drawn box to sit at an anchor.

    Args:
        entry: The layer being moved.
        into: ``(x, y, width, height)`` it is aligned inside.
        align: ``(across, down)`` fractions of the space left over, 0.0 to 1.0.

    Returns:
        ``(x, y)`` for the layer, which is its untuned top left rather than its drawn one.
    """
    box_x, box_y, box_w, box_h = box_of(entry)
    left = into[0] + int(round((into[2] - box_w) * float(align[0])))
    top = into[1] + int(round((into[3] - box_h) * float(align[1])))
    # A turned layer's drawn box does not start where x and y do, so the gap is carried.
    return _whole(entry.get("x")) + left - box_x, _whole(entry.get("y")) + top - box_y


def fitted(entry, width: int, height: int, fit: str) -> tuple[int, int]:
    """The drawn size a layer takes to reach a box.

    Args:
        entry: The layer being sized.
        width: Box width in pixels.
        height: Box height in pixels.
        fit: One of :data:`FITS`.

    Returns:
        ``(width, height)`` for the layer's ``w`` and ``h``, at least one each way.
    """
    held_w, held_h = _drawn_size(entry)
    if fit == FITS[2]:
        return max(1, int(width)), max(1, int(height))
    across = width / held_w if held_w else 1.0
    down = height / held_h if held_h else 1.0
    scale = min(across, down) if fit == FITS[0] else max(across, down)
    return max(1, int(round(held_w * scale))), max(1, int(round(held_h * scale)))


def trimmed(entry, threshold: float = 0.0) -> tuple[dict, tuple[int, int, int, int]]:
    """A layer with the empty band round its picture cut away and its placement moved back.

    Args:
        entry: The layer dictionary. Its pictures are read at their own size, before any
            drawn size, rotation or flip.
        threshold: Coverage at or below which a pixel counts as empty, 0.0 to 1.0.

    Returns:
        ``(entry, box)``. The entry is a new dictionary; the box is the ``(left, top,
        right, bottom)`` of the picture that was kept, counting the right and bottom edges
        as one past the last column and row.
    """
    pictures = entry.get("image")
    if not isinstance(pictures, torch.Tensor):
        return dict(entry), (0, 0, 0, 0)
    stacked = pictures if pictures.ndim == 4 else pictures.unsqueeze(0)
    height, width = int(stacked.shape[1]), int(stacked.shape[2])

    if int(stacked.shape[3]) >= 4:
        cover = stacked[..., 3].amax(dim=0)
    else:
        cover = torch.ones((height, width), dtype=stacked.dtype, device=stacked.device)
    veil = entry.get("mask")
    if isinstance(veil, torch.Tensor):
        planes = veil if veil.ndim == 3 else veil.unsqueeze(0)
        cover = cover * (1.0 - _fitted(
            planes.amin(dim=0), height, width, cover.device, cover.dtype
        ).clamp(0.0, 1.0))

    painted = cover > float(threshold)
    rows = torch.nonzero(painted.any(dim=1)).flatten()
    columns = torch.nonzero(painted.any(dim=0)).flatten()
    if rows.numel() == 0 or columns.numel() == 0:
        return dict(entry), (0, 0, width, height)

    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(columns[0]), int(columns[-1]) + 1
    if (left, top, right, bottom) == (0, 0, width, height):
        return dict(entry), (left, top, right, bottom)

    cut = dict(entry)
    cut["image"] = stacked[:, top:bottom, left:right]
    if isinstance(veil, torch.Tensor):
        planes = veil if veil.ndim == 3 else veil.unsqueeze(0)
        if int(planes.shape[1]) == height and int(planes.shape[2]) == width:
            cut["mask"] = planes[:, top:bottom, left:right]

    scale_x = (_drawn_size(entry)[0]) / width
    scale_y = (_drawn_size(entry)[1]) / height
    cut["x"] = _whole(entry.get("x")) + int(round(left * scale_x))
    cut["y"] = _whole(entry.get("y")) + int(round(top * scale_y))
    cut["w"] = max(1, int(round((right - left) * scale_x)))
    cut["h"] = max(1, int(round((bottom - top) * scale_y)))
    return cut, (left, top, right, bottom)


def found(layers, index: int, name: str, where: str = "This node", required: bool = True) -> int:
    """Which layer an index and a name pick out.

    Args:
        layers: The layer dictionaries, lowest in the stack first.
        index: Position counting 0 from the back, or -1 from the front.
        name: What the layer is called. Blank reads ``index`` instead.
        where: The node's name, for the message a miss raises with.
        required: Whether a miss raises rather than answering -1.

    Returns:
        The position in ``layers``, or -1 for a miss that does not raise.

    Raises:
        ValueError: Nothing matched and ``required`` is true.
    """
    wanted = (name or "").strip()
    if wanted:
        folded = wanted.casefold()
        spellings = [str(entry.get("name") or "") for entry in layers]
        for position, spelling in enumerate(spellings):
            if spelling.casefold() == folded:
                return position
        for position, spelling in enumerate(spellings):
            if folded in spelling.casefold():
                return position
        if not required:
            return -1
        shown = ", ".join(f"'{spelling}'" for spelling in spellings[:8] if spelling)
        raise ValueError(
            f"{where} found no layer called '{wanted}' in a stack of {len(layers)}. "
            f"It holds {shown or 'layers with no names'}. Type one of those names into "
            f"layer_name, or clear layer_name and set index instead."
        )

    position = index if index >= 0 else len(layers) + index
    if 0 <= position < len(layers):
        return position
    if not required:
        return -1
    raise ValueError(
        f"{where} was given index {index}, which is outside a stack of {len(layers)} "
        f"layer(s). Use 0 to {max(0, len(layers) - 1)} counting from the back, or -1 for "
        f"the front layer."
    )


def matching(layers, first: int, last: int, name: str, match: str, visibility: str) -> list[bool]:
    """Which layers a range, a name and a visibility pick out.

    Args:
        layers: The layer dictionaries, lowest in the stack first.
        first: Lowest position kept, counting 0 from the back or -1 from the front.
        last: Highest position kept, counted the same way.
        name: Text a layer's name carries. Blank matches every name.
        match: One of :data:`MATCHES`.
        visibility: One of :data:`VISIBILITIES`.

    Returns:
        One flag per layer, in the order given.
    """
    total = len(layers)
    low = first if first >= 0 else total + first
    high = last if last >= 0 else total + last
    if low > high:
        low, high = high, low
    wanted = (name or "").strip().casefold()

    flags = []
    for position, entry in enumerate(layers):
        hit = low <= position <= high
        if hit and wanted:
            spelling = str(entry.get("name") or "").casefold()
            hit = spelling == wanted if match == MATCHES[1] else wanted in spelling
        if hit and visibility != VISIBILITIES[0]:
            shown = bool(entry.get("visible", True))
            hit = shown if visibility == VISIBILITIES[1] else not shown
        flags.append(hit)
    return flags


def moved(layers, position: int, move: str, target: int) -> tuple[list[dict], int]:
    """The stack with one layer moved, or the whole stack sorted.

    Args:
        layers: The layer dictionaries, lowest in the stack first.
        position: Which layer moves. Negative moves nothing.
        move: One of :data:`MOVES`.
        target: Landing position for ``to index``, counting 0 from the back.

    Returns:
        ``(layers, position)``, the stack in its new order and where that layer landed.
        The position is -1 where nothing was tracked.
    """
    total = len(layers)
    order = list(range(total))

    if move.startswith("sort by name"):
        order.sort(
            key=lambda slot: str(layers[slot].get("name") or "").casefold(),
            reverse=move.endswith("z to a"),
        )
    elif move.startswith("sort by area"):
        order.sort(
            key=lambda slot: _drawn_size(layers[slot])[0] * _drawn_size(layers[slot])[1],
            reverse=move.endswith("largest first"),
        )
    elif 0 <= position < total:
        order.pop(position)
        if move == MOVES[0]:
            landing = total - 1
        elif move == MOVES[1]:
            landing = 0
        elif move == MOVES[2]:
            landing = min(total - 1, position + 1)
        elif move == MOVES[3]:
            landing = max(0, position - 1)
        else:
            landing = target if target >= 0 else total + target
        order.insert(max(0, min(total - 1, landing)), position)

    landed = order.index(position) if 0 <= position < total else -1
    return [layers[slot] for slot in order], landed


def duplicated(entry, dx: int, dy: int, name: str) -> dict:
    """A copy of one layer, offset and named.

    Args:
        entry: The layer to copy.
        dx: Pixels the copy moves right. A negative number moves it left.
        dy: Pixels the copy moves down. A negative number moves it up.
        name: What the copy is called. Blank adds ``copy`` to the original's name.

    Returns:
        A new layer dictionary sharing the original's picture and mask tensors.
    """
    copy = dict(entry)
    copy["x"] = _whole(entry.get("x")) + int(dx)
    copy["y"] = _whole(entry.get("y")) + int(dy)
    spelling = (name or "").strip()
    copy["name"] = spelling or f"{str(entry.get('name') or 'layer')} copy"
    return copy


def described(document) -> tuple[dict, list[dict]]:
    """A document and its layers as plain data.

    Args:
        document: A ``LAYERS`` value.

    Returns:
        ``(summary, rows)``. The summary carries the canvas and the counts, and each row
        carries one layer's index, name, placement, size, picture count and compositing.
        Rotation is in degrees.
    """
    layers = entries(document)
    width, height = size_of(document)

    rows = []
    for position, entry in enumerate(layers):
        picture = _picture(entry)
        drawn_w, drawn_h = _drawn_size(entry)
        rows.append(
            {
                "index": position,
                "name": str(entry.get("name") or ""),
                "x": _whole(entry.get("x")),
                "y": _whole(entry.get("y")),
                "width": drawn_w,
                "height": drawn_h,
                "source_width": int(picture.shape[1]),
                "source_height": int(picture.shape[0]),
                "channels": int(picture.shape[2]) if picture.ndim > 2 else 1,
                "frames": _frames(entry),
                "rotation": round(math.degrees(_number(entry.get("rotation"), 0.0)), 4),
                "opacity": _number(entry.get("opacity"), 1.0),
                "blend_mode": str(entry.get("blend_mode") or "normal"),
                "visible": bool(entry.get("visible", True)),
                "flip_h": bool(entry.get("flip_h", False)),
                "flip_v": bool(entry.get("flip_v", False)),
                "has_mask": entry.get("mask") is not None,
                "z_index": _whole(entry.get("z_index")),
            }
        )

    summary = {
        "canvas_width": width,
        "canvas_height": height,
        "layers": len(rows),
        "visible": sum(1 for row in rows if row["visible"]),
        "hidden": sum(1 for row in rows if not row["visible"]),
        "names": [row["name"] for row in rows],
    }
    return summary, rows


def _to_linear(colour):
    """Picture codes as linear light."""
    high = ((colour.clamp(min=0.0) + 0.055) / 1.055) ** 2.4
    return torch.where(colour <= 0.04045, colour / 12.92, high)


def _to_srgb(colour):
    """Linear light as picture codes."""
    high = 1.055 * colour.clamp(min=0.0) ** (1.0 / 2.4) - 0.055
    return torch.where(colour <= 0.0031308, 12.92 * colour, high)


def _divided(numerator, denominator):
    """A division answering zero where the divisor is too near zero to use."""
    usable = denominator.abs() >= EPSILON
    safe = torch.where(usable, denominator, torch.ones_like(denominator))
    return torch.where(usable, numerator / safe, torch.zeros_like(denominator))



def _composite_union(backdrop, layer, mixed, cover):
    """The layer painted over the backdrop, widening what the result covers."""
    in_a = backdrop[..., 3]
    layer_a = layer[..., 3] * cover
    new_a = layer_a + (1.0 - layer_a) * in_a
    ratio = _divided(layer_a, new_a)
    blended = (
        ratio.unsqueeze(-1)
        * (in_a.unsqueeze(-1) * (mixed - layer[..., :3]) + layer[..., :3] - backdrop[..., :3])
        + backdrop[..., :3]
    )
    keep = (layer_a == 0) | (new_a == 0)
    rgb = torch.where(
        keep.unsqueeze(-1),
        backdrop[..., :3],
        torch.where((in_a == 0).unsqueeze(-1), layer[..., :3], blended),
    )
    return torch.cat([rgb, new_a.unsqueeze(-1)], dim=-1)


def _composite_clipped(backdrop, layer, mixed, cover):
    """The layer painted only where the backdrop already covers."""
    in_a = backdrop[..., 3]
    layer_a = layer[..., 3] * cover
    mixed_rgb = mixed * layer_a.unsqueeze(-1) + backdrop[..., :3] * (1.0 - layer_a.unsqueeze(-1))
    keep = (in_a == 0) | (layer_a == 0)
    rgb = torch.where(keep.unsqueeze(-1), backdrop[..., :3], mixed_rgb)
    return torch.cat([rgb, in_a.unsqueeze(-1)], dim=-1)


#: Blend mode -> the space it mixes in, and how it covers the backdrop.
_MODE_RULES = {
    "normal": ("linear", _composite_union),
    "multiply": ("linear", _composite_clipped),
    "screen": ("perceptual", _composite_clipped),
    "overlay": ("perceptual", _composite_clipped),
    "darken": ("linear", _composite_clipped),
    "lighten": ("linear", _composite_clipped),
    "color-dodge": ("perceptual", _composite_clipped),
    "color-burn": ("perceptual", _composite_clipped),
    "hard-light": ("perceptual", _composite_clipped),
    "soft-light": ("perceptual", _composite_clipped),
    "difference": ("perceptual", _composite_clipped),
    "exclusion": ("perceptual", _composite_clipped),
    "linear-dodge": ("linear", _composite_clipped),
    "linear-burn": ("perceptual", _composite_clipped),
    "vivid-light": ("perceptual", _composite_clipped),
    "pin-light": ("perceptual", _composite_clipped),
    "linear-light": ("perceptual", _composite_clipped),
    "hard-mix": ("perceptual", _composite_clipped),
    "subtract": ("linear", _composite_clipped),
    "divide": ("linear", _composite_clipped),
    "grain-extract": ("perceptual", _composite_clipped),
    "grain-merge": ("perceptual", _composite_clipped),
    "hue": ("perceptual", _composite_clipped),
    "saturation": ("perceptual", _composite_clipped),
    "color": ("perceptual", _composite_clipped),
    "luminosity": ("linear", _composite_clipped),
}


def _over(backdrop, layer, mode: str, opacity: float):
    """One layer composited onto what is already there.

    Args:
        backdrop: ``(height, width, 4)`` linear RGBA under the layer.
        layer: ``(height, width, 4)`` linear RGBA to draw.
        mode: One of :data:`BLEND_MODES`.
        opacity: 0.0 to 1.0.

    Returns:
        ``(height, width, 4)`` linear RGBA.
    """
    space, composite = _MODE_RULES.get(mode, _MODE_RULES["normal"])
    in_rgb = backdrop[..., :3] if space == "linear" else _to_srgb(backdrop[..., :3])
    layer_rgb = layer[..., :3] if space == "linear" else _to_srgb(layer[..., :3])

    mixed = blend_modes.blend(in_rgb, layer_rgb, mode)
    if space != "linear":
        mixed = _to_linear(mixed)
    return composite(backdrop, layer, mixed, opacity)


def _fitted(cut, height: int, width: int, device, dtype):
    """One mask plane at a picture's own size."""
    plane = cut
    while plane.ndim > 2:
        plane = plane[0]
    plane = plane.to(device=device, dtype=dtype)
    if tuple(plane.shape[:2]) == (height, width):
        return plane
    stretched = torch.nn.functional.interpolate(
        plane.unsqueeze(0).unsqueeze(0), size=(height, width), mode="bilinear", align_corners=False
    )
    return stretched[0, 0]


def _covered(frame, source):
    """One resampled RGBA picture, coverage inside 0 to 1 and colour on the source's scale.

    Args:
        frame: The resampled ``(height, width, 4)`` picture.
        source: What it was resampled from, read for whether it carried more than a picture.

    Returns:
        A tensor the shape of ``frame``. A source already inside 0 to 1 has the resampler's
        overshoot taken back off its colour.
    """
    if int(frame.shape[-1]) < 4:
        return dynamic.hold(frame, source)
    colour = dynamic.hold(frame[..., :3], source[..., :3])
    return torch.cat([colour, frame[..., 3:].clamp(0.0, 1.0)], dim=-1)


def _resized(frame, width: int, height: int):
    """One picture at another size."""
    shrinking = height < int(frame.shape[0]) or width < int(frame.shape[1])
    stretched = torch.nn.functional.interpolate(
        frame.permute(2, 0, 1).unsqueeze(0),
        size=(height, width),
        mode="bicubic",
        align_corners=False,
        antialias=shrinking,
    )
    return _covered(stretched[0].permute(1, 2, 0), frame)


def _bounds(x: float, y: float, width: float, height: float, turn: float):
    """The box a turned rectangle covers, as ``(x, y, width, height)``."""
    cx, cy = x + width / 2.0, y + height / 2.0
    cos, sin = math.cos(turn), math.sin(turn)
    half_w, half_h = width / 2.0, height / 2.0
    corners = ((-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h))
    xs = [cx + dx * cos - dy * sin for dx, dy in corners]
    ys = [cy + dx * sin + dy * cos for dx, dy in corners]
    left, top = math.floor(min(xs)), math.floor(min(ys))
    return left, top, max(1, math.ceil(max(xs)) - left), max(1, math.ceil(max(ys)) - top)


def _turned(frame, x: int, y: int, turn: float):
    """One picture rotated about its own centre, in a box big enough to hold it.

    Args:
        frame: ``(height, width, 4)`` picture.
        x: Its left edge on the canvas before the turn.
        y: Its top edge on the canvas before the turn.
        turn: Angle in radians, positive clockwise.

    Returns:
        ``(picture, x, y)`` with the picture grown to hold every corner.
    """
    height, width = int(frame.shape[0]), int(frame.shape[1])
    left, top, box_w, box_h = _bounds(x, y, width, height, turn)
    cos, sin = math.cos(turn), math.sin(turn)
    cx, cy = x + width / 2.0, y + height / 2.0

    columns = torch.arange(box_w, device=frame.device, dtype=frame.dtype) + (left + 0.5)
    rows = torch.arange(box_h, device=frame.device, dtype=frame.dtype) + (top + 0.5)
    dx = columns.unsqueeze(0) - cx
    dy = rows.unsqueeze(1) - cy
    source_x = dx * cos + dy * sin + width / 2.0
    source_y = -dx * sin + dy * cos + height / 2.0
    grid = torch.stack(
        [
            (2.0 * source_x / width - 1.0).expand(box_h, box_w),
            (2.0 * source_y / height - 1.0).expand(box_h, box_w),
        ],
        dim=-1,
    ).unsqueeze(0)

    sampled = torch.nn.functional.grid_sample(
        frame.permute(2, 0, 1).unsqueeze(0),
        grid,
        mode="bicubic",
        padding_mode="zeros",
        align_corners=False,
    )
    return _covered(sampled[0].permute(1, 2, 0), frame), left, top


def _rendered(entry, frame: int = 0):
    """One of a layer's pictures at the size, flip and angle it is drawn at.

    Args:
        entry: The layer dictionary.
        frame: Which of its pictures, counting 0.

    Returns:
        ``(picture, x, y)``. The picture is ``(height, width, 4)`` picture codes with
        straight alpha, and ``(x, y)`` is where its top left corner lands on the canvas.
    """
    picture = _picture(entry, frame).to(dtype=torch.float32)
    if picture.ndim == 2:
        picture = picture.unsqueeze(-1)
    colour = picture[..., :3] if int(picture.shape[2]) >= 3 else picture[..., :1].repeat(1, 1, 3)

    if int(picture.shape[2]) >= 4:
        alpha = picture[..., 3]
    else:
        alpha = torch.ones(picture.shape[:2], dtype=picture.dtype, device=picture.device)
    cut = _cut(entry, frame)
    if cut is not None:
        covered = _fitted(
            cut, int(picture.shape[0]), int(picture.shape[1]), picture.device, picture.dtype
        )
        alpha = alpha * (1.0 - covered.clamp(0.0, 1.0))

    frame = torch.cat([colour, alpha.clamp(0.0, 1.0).unsqueeze(-1)], dim=-1)
    if bool(entry.get("flip_h", False)):
        frame = torch.flip(frame, dims=(1,))
    if bool(entry.get("flip_v", False)):
        frame = torch.flip(frame, dims=(0,))

    width, height = _drawn_size(entry)
    if (height, width) != (int(frame.shape[0]), int(frame.shape[1])):
        frame = _resized(frame, width, height)

    x, y = _whole(entry.get("x")), _whole(entry.get("y"))
    turn = _number(entry.get("rotation"), 0.0)
    if turn:
        return _turned(frame, x, y, turn)
    return frame, x, y


def _flattened(layers, width: int, height: int, scale: float = 1.0):
    """Every visible layer composited onto a transparent canvas.

    Args:
        layers: The layer dictionaries, lowest in the stack first.
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        scale: What every layer's colour is divided by before it is composited, so the
            blend modes work on the 0 to 1 they are written for.

    Returns:
        ``(picture, box)``. The picture is ``(height, width, 4)`` linear light with
        straight alpha, and the box is the ``(left, top, right, bottom)`` the layers
        reached. A layer carrying a batch is drawn one picture at a time, lowest first.
    """
    canvas = None
    box = None
    for entry in layers:
        if not bool(entry.get("visible", True)):
            continue
        for index in range(_frames(entry)):
            frame, x, y = _rendered(entry, index)
            if canvas is None:
                canvas = torch.zeros(
                    (height, width, 4), dtype=torch.float32, device=frame.device
                )
            frame = frame.to(device=canvas.device)
            left, top = max(0, x), max(0, y)
            right = min(width, x + int(frame.shape[1]))
            bottom = min(height, y + int(frame.shape[0]))
            if right <= left or bottom <= top:
                continue

            patch = frame[top - y : bottom - y, left - x : right - x]
            colour = patch[..., :3] / scale if scale != 1.0 else patch[..., :3]
            layer = torch.cat([_to_linear(colour.clamp(0.0, 1.0)), patch[..., 3:]], dim=-1)
            opacity = min(max(_number(entry.get("opacity"), 1.0), 0.0), 1.0)
            canvas[top:bottom, left:right] = _over(
                canvas[top:bottom, left:right],
                layer,
                str(entry.get("blend_mode") or "normal"),
                opacity,
            )
            box = (
                (left, top, right, bottom)
                if box is None
                else (
                    min(box[0], left),
                    min(box[1], top),
                    max(box[2], right),
                    max(box[3], bottom),
                )
            )

    if canvas is None:
        canvas = torch.zeros((height, width, 4), dtype=torch.float32)
    return canvas, box or (0, 0, width, height)


def merged(layers, width: int, height: int, name: str):
    """A run of layers flattened into one raster layer.

    Args:
        layers: The layer dictionaries to flatten, lowest in the stack first.
        width: Canvas width in pixels.
        height: Canvas height in pixels.
        name: What the merged layer is called.

    Returns:
        ``(entry, image, mask)``. The entry is a layer dictionary drawn normally at full
        opacity; the image is ``(1, height, width, 3)`` and the mask ``(1, height,
        width)``, both cropped to what the layers covered. A stack carrying light above
        white comes back on the scale it arrived on.
    """
    scale = dynamic.peak(*[entry.get("image") for entry in layers])
    canvas, (left, top, right, bottom) = _flattened(layers, width, height, scale)
    cropped = canvas[top:bottom, left:right]
    image = _to_srgb(cropped[..., :3].clamp(0.0, 1.0)).clamp(0.0, 1.0)
    image = (image * scale if scale != 1.0 else image).unsqueeze(0)
    mask = (1.0 - cropped[..., 3].clamp(0.0, 1.0)).unsqueeze(0)

    entry = {
        "image": image,
        "type": "raster",
        "mask": mask,
        "x": int(left),
        "y": int(top),
        "w": int(right - left),
        "h": int(bottom - top),
        "name": name,
        "opacity": 1.0,
        "blend_mode": "normal",
        "visible": True,
        "flip_h": False,
        "flip_v": False,
        "rotation": 0.0,
        "z_index": 0,
    }
    return entry, image, mask

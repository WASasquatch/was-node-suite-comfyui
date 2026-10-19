"""Overlapping tiles of a latent, each an area of the conditioning blended by a feathered mask.

Spans are latent pixels as ``(start, stop)``. A tile mask is ``[1, *dims]``, ``dims`` the latent's
axes after batch and channels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .. import log

__all__ = [
    "ALIGN",
    "TILES_WRAPPER",
    "Grid",
    "auto_tile",
    "crop_plane",
    "masks",
    "numbered",
    "plan",
    "plot",
    "position",
    "tile_weights",
    "tiled_conditioning",
    "tiled_model",
    "with_text",
]

logger = log.get_logger("sampling.tiles")

#: Key the tiled model's wrapper is registered under.
TILES_WRAPPER = "was_tiles"

#: Tile starts and sizes fall on multiples of this many latent pixels.
ALIGN = 2

#: Smallest blend weight inside a tile.
EDGE_WEIGHT = 1e-3

#: Tile side in pixels planned for a model of the SD1 family.
SD1_TILE = 512

#: Tile side in pixels planned for every other model.
NATIVE_TILE = 1024

#: Share of a tile side neighbouring tiles share when planned automatically.
AUTO_OVERLAP = 0.25

#: Keys of a conditioning entry the tile owns, replaced on every tile copy.
TILE_KEYS = ("area", "mask", "mask_strength", "set_area_to_bounds", "was_tile")

#: Key of a conditioning entry naming the tile it belongs to.
TILE_KEY = "was_tile"


@dataclass(frozen=True)
class Grid:
    """Where the tiles of a latent sit.

    Attributes:
        heights: ``(start, stop)`` latent pixels of each band down the frame.
        widths: ``(start, stop)`` latent pixels of each band across the frame.
        size: ``(height, width)`` of the latent.
        scale: Image pixels per latent pixel.
    """

    heights: tuple
    widths: tuple
    size: tuple
    scale: float

    @property
    def count(self) -> int:
        """Tiles in all."""
        return len(self.heights) * len(self.widths)

    @property
    def tile(self) -> tuple:
        """``(height, width)`` of the largest tile in latent pixels."""
        return (max(b - a for a, b in self.heights), max(b - a for a, b in self.widths))

    @property
    def overlap(self) -> tuple:
        """``(down, across)`` latent pixels neighbouring tiles share, ``0`` along an axis with one band."""
        def shared(spans):
            return max((spans[i][1] - spans[i + 1][0] for i in range(len(spans) - 1)), default=0)

        return (shared(self.heights), shared(self.widths))

    def boxes(self) -> list:
        """``(top, bottom, left, right)`` of every tile, row by row."""
        return [(h0, h1, w0, w1) for h0, h1 in self.heights for w0, w1 in self.widths]

    def fractions(self) -> list:
        """``[top, bottom, left, right]`` of every tile as fractions of the frame, row by row."""
        height, width = self.size
        return [[h0 / height, h1 / height, w0 / width, w1 / width] for h0, h1, w0, w1 in self.boxes()]


def _spans(length: int, tile: int, overlap: int, align: int = ALIGN) -> tuple:
    """``(start, stop)`` ranges of one size covering ``length``, none longer than ``tile`` where it allows.

    Args:
        length: Extent to cover.
        tile: Longest range wanted.
        overlap: Least extent neighbouring ranges share.
        align: Starts fall on multiples of this.

    Returns:
        The ranges, evenly spaced.
    """
    length, align = int(length), max(1, int(align))
    tile = max(align, int(tile))
    if tile >= length:
        return ((0, length),)
    overlap = max(0, min(int(overlap), tile - align))
    pieces = max(2, math.ceil((length - overlap) / max(1, tile - overlap)))
    while True:
        stride = max(align, (length - overlap) // pieces // align * align)
        size = length - (pieces - 1) * stride
        # A stride rounded down to the alignment lengthens the ranges; one more range shortens them.
        if size <= tile or stride <= align:
            break
        pieces += 1
    return tuple((index * stride, index * stride + size) for index in range(pieces))


def auto_tile(model) -> int:
    """Tile side in pixels suited to a model's native resolution.

    Args:
        model: A model patcher.

    Returns:
        :data:`SD1_TILE` for an SD1 family model, else :data:`NATIVE_TILE`.
    """
    latent_format = getattr(getattr(model, "model", None), "latent_format", None)
    return SD1_TILE if type(latent_format).__name__ == "SD15" else NATIVE_TILE


def plan(latent_size: tuple, scale: float, tile: tuple, overlap: int, align: int = ALIGN) -> Grid:
    """The tiling of a latent from tile sizes in image pixels.

    Args:
        latent_size: ``(height, width)`` of the latent.
        scale: Image pixels per latent pixel.
        tile: ``(height, width)`` of the largest tile in image pixels.
        overlap: Image pixels neighbouring tiles share.
        align: Tile starts and sizes fall on multiples of this many latent pixels.

    Returns:
        The grid.
    """
    height, width = (int(v) for v in latent_size)
    scale = max(float(scale), 1e-6)
    align = max(1, int(align))

    def latent(pixels):
        return max(align, int(round(float(pixels) / scale / align)) * align)

    shared = max(0, int(round(float(overlap) / scale)))
    return Grid(
        _spans(height, latent(tile[0]), shared, align),
        _spans(width, latent(tile[1]), shared, align),
        (height, width),
        scale,
    )


def _smooth(weight: torch.Tensor) -> torch.Tensor:
    """Raised-cosine form of a linear ramp in ``[0, 1]``; two ramps summing to 1 still do."""
    return 0.5 - 0.5 * torch.cos(math.pi * weight.clamp(0.0, 1.0))


def _ramp(spans: tuple, index: int) -> torch.Tensor:
    """Blend weights along one axis for one tile: raised-cosine ramps across the overlaps it shares."""
    start, stop = spans[index]
    size = stop - start
    position = torch.arange(size, dtype=torch.float32)
    weight = torch.ones(size, dtype=torch.float32)
    if index > 0:
        left = spans[index - 1][1] - start
        if left > 0:
            weight = torch.minimum(weight, (position + 1) / (left + 1))
    if index < len(spans) - 1:
        right = stop - spans[index + 1][0]
        if right > 0:
            weight = torch.minimum(weight, (size - position) / (right + 1))
    return _smooth(weight).clamp(min=EDGE_WEIGHT)


def masks(grid: Grid, dims: tuple) -> list:
    """One blend mask per tile over the whole latent, zero outside the tile.

    Args:
        grid: The tiling.
        dims: The latent's axes after batch and channels, as ``(height, width)`` or
            ``(frames, height, width)``.

    Returns:
        ``[1, *dims]`` float tensors, row by row.
    """
    height, width = grid.size
    out = []
    for h_index, (h0, h1) in enumerate(grid.heights):
        down = _ramp(grid.heights, h_index)
        for w_index, (w0, w1) in enumerate(grid.widths):
            plane = torch.zeros(height, width, dtype=torch.float32)
            plane[h0:h1, w0:w1] = down[:, None] * _ramp(grid.widths, w_index)[None, :]
            if len(dims) > 2:
                plane = plane.expand(*dims[:-2], height, width).contiguous()
            out.append(plane.reshape(1, *dims))
    return out


def tile_weights(grid: Grid) -> list:
    """One blend weight per tile over the tile alone; overlapping weights sum to one.

    Args:
        grid: The tiling.

    Returns:
        ``[height, width]`` float tensors, row by row.
    """
    out = []
    for h_index in range(len(grid.heights)):
        down = _ramp(grid.heights, h_index)
        for w_index in range(len(grid.widths)):
            out.append(down[:, None] * _ramp(grid.widths, w_index)[None, :])
    return out


def position(grid: Grid, index: int) -> str:
    """Where a tile sits in the frame, in words.

    Args:
        grid: The tiling.
        index: The tile, counted row by row.

    Returns:
        A phrase such as ``top left`` or ``centre``, or ``row 2 of 7, column 5 of 7`` on a grid
        more than three tiles across or down.
    """
    rows, columns = len(grid.heights), len(grid.widths)
    row, column = divmod(int(index), columns)
    if rows > 3 or columns > 3:
        return f"row {row + 1} of {rows}, column {column + 1} of {columns}"

    def place(at, count, first, middle, last):
        if count == 1:
            return ""
        if at == 0:
            return first
        if at == count - 1:
            return last
        return middle

    vertical = place(row, rows, "top", "middle", "bottom")
    horizontal = place(column, columns, "left", "centre", "right")
    if vertical == "middle" and horizontal == "centre":
        return "centre"
    words = " ".join(word for word in (vertical, horizontal) if word)
    return words or "whole frame"


def crop_plane(tensor: torch.Tensor, box: tuple, size: tuple) -> torch.Tensor:
    """The part of a picture-shaped tensor under a tile, cut in proportion to its own size.

    Args:
        tensor: Any tensor whose last two axes are height and width.
        box: ``(top, bottom, left, right)`` of the tile in latent pixels.
        size: ``(height, width)`` of the latent.

    Returns:
        A view of ``tensor``.
    """
    return _crop_plane(tensor, box, size)


def _crop_plane(tensor: torch.Tensor, box: tuple, size: tuple) -> torch.Tensor:
    """The part of a picture-shaped tensor under a tile, cut in proportion to its own size."""
    height, width = int(tensor.shape[-2]), int(tensor.shape[-1])
    h0 = int(math.floor(box[0] * height / size[0]))
    h1 = max(h0 + 1, int(math.ceil(box[1] * height / size[0])))
    w0 = int(math.floor(box[2] * width / size[1]))
    w1 = max(w0 + 1, int(math.ceil(box[3] * width / size[1])))
    return tensor[..., h0:h1, w0:w1]


def _cropped_control(control, box: tuple, size: tuple):
    """A copy of a control chain whose hints cover only one tile.

    Args:
        control: The first control of the chain.
        box: ``(top, bottom, left, right)`` of the tile in latent pixels.
        size: ``(height, width)`` of the latent.

    Returns:
        The first copy, linked to copies of the rest.
    """
    chain = []
    link = control
    while link is not None:
        copy = link.copy()
        hint = getattr(link, "cond_hint_original", None)
        if torch.is_tensor(hint) and hint.ndim >= 4:
            copy.cond_hint_original = _crop_plane(hint, box, size)
        extra = getattr(link, "extra_concat_orig", None)
        if extra:
            copy.extra_concat_orig = [
                _crop_plane(item, box, size)
                if torch.is_tensor(item) and item.ndim >= 4 and item.shape[-1] > 1 and item.shape[-2] > 1
                else item
                for item in extra
            ]
        chain.append(copy)
        link = getattr(link, "previous_controlnet", None)
    for current, following in zip(chain, chain[1:]):
        current.set_previous_controlnet(following)
    return chain[0] if chain else None


def with_text(template: list, encoded: list) -> list:
    """Conditioning carrying a new text, with everything else a template entry held.

    Args:
        template: The CONDITIONING whose first entry lends its extra values, as guidance or
            reference latents.
        encoded: The CONDITIONING the new text was encoded to.

    Returns:
        A new CONDITIONING list.
    """
    extras = dict(template[0][1]) if template else {}
    for key in TILE_KEYS:
        extras.pop(key, None)
    return [[tensor, {**extras, **dict(values)}] for tensor, values in encoded]


def tiled_conditioning(conditioning: list, grid: Grid, tile_masks: list, texts=None,
                       controls=None) -> list:
    """Conditioning repeated once per tile, each copy an area of the latent with its blend mask.

    Args:
        conditioning: A CONDITIONING list.
        grid: The tiling.
        tile_masks: :func:`masks` of the grid.
        texts: One CONDITIONING per tile replacing the text of ``conditioning`` there, ``None``
            to keep it, or ``None`` for every tile.
        controls: A dictionary sharing cropped control chains between the conditionings of one
            run, keyed by control and tile.

    Returns:
        A new CONDITIONING list, tile entries row by row.
    """
    controls = {} if controls is None else controls
    full_h, full_w = (int(round(v * grid.scale)) for v in grid.size)
    out = []
    for index, box in enumerate(grid.boxes()):
        h0, h1, w0, w1 = box
        entries = conditioning
        if texts is not None and texts[index] is not None:
            entries = with_text(conditioning, texts[index])
        for tensor, extras in entries:
            values = dict(extras)
            values.update({
                "area": (h1 - h0, w1 - w0, h0, w0),
                "mask": tile_masks[index],
                "mask_strength": 1.0,
                "set_area_to_bounds": False,
                TILE_KEY: index,
                "crop_h": int(round(h0 * grid.scale)),
                "crop_w": int(round(w0 * grid.scale)),
                "target_height": int(round((h1 - h0) * grid.scale)),
                "target_width": int(round((w1 - w0) * grid.scale)),
            })
            values.setdefault("height", full_h)
            values.setdefault("width", full_w)
            control = values.get("control")
            if control is not None:
                key = (id(control), index)
                if key not in controls:
                    controls[key] = (control, _cropped_control(control, box, grid.size))
                values["control"] = controls[key][1]
            out.append([tensor, values])
    return out


def tiled_model(model, grid: Grid, announce=None):
    """A clone of a model that reserves memory for one tile rather than the whole latent.

    Args:
        model: A model patcher.
        grid: The tiling it samples.
        announce: Called as ``announce("sample", tiles)`` with the tiles each model call runs,
            or None.

    Returns:
        The patched clone.
    """
    import comfy.patcher_extension

    wrappers = comfy.patcher_extension.WrappersMP
    tile_h, tile_w = grid.tile
    owners = {}

    def prepare(executor, patcher, noise_shape, conds, *args, **kwargs):
        shape = list(noise_shape)
        if len(shape) >= 4:
            shape[-2], shape[-1] = min(shape[-2], tile_h), min(shape[-1], tile_w)
        return executor(patcher, shape, conds, *args, **kwargs)

    def sample(executor, *args, **kwargs):
        owners.clear()
        for entries in (getattr(executor.class_obj, "original_conds", None) or {}).values():
            for entry in entries or ():
                if isinstance(entry, dict) and TILE_KEY in entry and "uuid" in entry:
                    owners[entry["uuid"]] = int(entry[TILE_KEY])
        return executor(*args, **kwargs)

    def apply(executor, *args, **kwargs):
        options = args[5] if len(args) > 5 else kwargs.get("transformer_options", {})
        try:
            uuids = (options or {}).get("uuids") or ()
            active = [owners[uuid] for uuid in uuids if uuid in owners]
            if active:
                announce("sample", active)
        except Exception as error:
            logger.debug("The active tiles could not be read: %s", error)
        return executor(*args, **kwargs)

    patched = model.clone()
    hooks = [(wrappers.PREPARE_SAMPLING, prepare)]
    if announce is not None:
        hooks += [(wrappers.OUTER_SAMPLE, sample), (wrappers.APPLY_MODEL, apply)]
    for kind in (wrappers.PREPARE_SAMPLING, wrappers.OUTER_SAMPLE, wrappers.APPLY_MODEL):
        patched.remove_wrappers_with_key(kind, TILES_WRAPPER)
    for kind, hook in hooks:
        patched.add_wrapper_with_key(kind, TILES_WRAPPER, hook)
    return patched


def plot(image: torch.Tensor, grid: Grid, size: int = 960) -> torch.Tensor:
    """The tiles drawn over a picture, each numbered, overlaps shaded.

    Args:
        image: ``[H, W, C]`` or ``[1, H, W, C]`` in ``[0, 1]``.
        grid: The tiling.
        size: Longest side of the answer in pixels.

    Returns:
        An IMAGE tensor ``[1, h, w, 3]``.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    frame = image[0] if image.ndim == 4 else image
    frame = frame[..., :3].float().clamp(0.0, 1.0)
    if frame.shape[-1] == 1:
        frame = frame.expand(*frame.shape[:-1], 3)
    height, width = int(frame.shape[0]), int(frame.shape[1])
    ratio = min(1.0, size / max(height, width))
    out_w, out_h = max(1, int(round(width * ratio))), max(1, int(round(height * ratio)))
    picture = Image.fromarray((frame.cpu().numpy() * 255.0).round().astype(np.uint8))
    picture = picture.resize((out_w, out_h), Image.LANCZOS)
    picture = Image.blend(picture, Image.new("RGB", picture.size, (14, 16, 22)), 0.35)
    shade = Image.new("RGBA", picture.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(shade)
    unit_h, unit_w = out_h / grid.size[0], out_w / grid.size[1]
    band = (214, 170, 72)
    # The bands neighbouring tiles share, where their predictions blend.
    for (_, stop), (start, _) in zip(grid.heights, grid.heights[1:]):
        draw.rectangle([0, start * unit_h, out_w, stop * unit_h], fill=(*band, 70))
    for (_, stop), (start, _) in zip(grid.widths, grid.widths[1:]):
        draw.rectangle([start * unit_w, 0, stop * unit_w, out_h], fill=(*band, 70))
    tile_side = min(out_h / len(grid.heights), out_w / len(grid.widths))
    font = _font(max(12, int(tile_side / 4)))
    for index, (h0, h1, w0, w1) in enumerate(grid.boxes()):
        centre = ((w0 + w1) / 2 * unit_w, (h0 + h1) / 2 * unit_h)
        label = str(index + 1)
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        radius = max(right - left, bottom - top) * 0.8 + 4
        draw.ellipse([centre[0] - radius, centre[1] - radius, centre[0] + radius, centre[1] + radius],
                     fill=(14, 16, 22, 190), outline=(*band, 255), width=2)
        draw.text((centre[0] - (left + right) / 2, centre[1] - (top + bottom) / 2), label,
                  fill=(240, 240, 245, 255), font=font)
    picture = Image.alpha_composite(picture.convert("RGBA"), shade).convert("RGB")
    return torch.from_numpy(np.asarray(picture, dtype=np.float32) / 255.0)[None]


def numbered(image: torch.Tensor, grid: Grid, active: int, size: int = 768) -> torch.Tensor:
    """A picture with its tiles ruled and numbered at full brightness, one tile outlined in yellow.

    Args:
        image: ``[H, W, C]`` or ``[1, H, W, C]`` in ``[0, 1]``.
        grid: The tiling.
        active: The tile outlined, counted row by row from 0.
        size: Longest side of the answer in pixels.

    Returns:
        An IMAGE tensor ``[1, h, w, 3]``.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    frame = image[0] if image.ndim == 4 else image
    frame = frame[..., :3].float().clamp(0.0, 1.0)
    height, width = int(frame.shape[0]), int(frame.shape[1])
    ratio = min(1.0, size / max(height, width))
    out_w, out_h = max(1, int(round(width * ratio))), max(1, int(round(height * ratio)))
    picture = Image.fromarray((frame.cpu().numpy() * 255.0).round().astype(np.uint8))
    picture = picture.resize((out_w, out_h), Image.LANCZOS).convert("RGBA")
    marks = Image.new("RGBA", picture.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(marks)
    unit_h, unit_w = out_h / grid.size[0], out_w / grid.size[1]
    # Seams fall halfway across each overlap, so the ruled cells do not overlap.
    rows = [0.0] + [(a[1] + b[0]) / 2 * unit_h for a, b in zip(grid.heights, grid.heights[1:])] + [out_h]
    cols = [0.0] + [(a[1] + b[0]) / 2 * unit_w for a, b in zip(grid.widths, grid.widths[1:])] + [out_w]
    line = (255, 255, 255, 150)
    for y in rows[1:-1]:
        draw.line([(0, y), (out_w, y)], fill=line, width=1)
    for x in cols[1:-1]:
        draw.line([(x, 0), (x, out_h)], fill=line, width=1)
    cell = min(out_h / len(grid.heights), out_w / len(grid.widths))
    font = _font(max(10, int(cell / 6)))
    columns = len(grid.widths)
    h0, h1, w0, w1 = grid.boxes()[active]
    box = [w0 * unit_w, h0 * unit_h, w1 * unit_w - 1, h1 * unit_h - 1]
    yellow = (255, 214, 40, 255)
    for index in range(grid.count):
        row, column = divmod(index, columns)
        x, y = cols[column] + 3, rows[row] + 3
        # Another tile's number inside the outlined tile would name the wrong tile.
        if index != active and box[0] <= x <= box[2] and box[1] <= y <= box[3]:
            continue
        label = str(index + 1)
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        own = index == active
        draw.rectangle([x, y, x + (right - left) + 6, y + (bottom - top) + 6],
                       fill=yellow if own else (14, 16, 22, 170))
        draw.text((x + 3 - left, y + 3 - top), label, font=font,
                  fill=(14, 16, 22, 255) if own else (255, 255, 255, 255))
    edge = max(2, round(min(out_w, out_h) / 160))
    draw.rectangle(box, outline=yellow, width=edge)
    picture = Image.alpha_composite(picture, marks).convert("RGB")
    return torch.from_numpy(np.asarray(picture, dtype=np.float32) / 255.0)[None]


def _font(size: int):
    """PIL's default font at ``size`` pixels, or its fixed bitmap font where sizing is not offered."""
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()

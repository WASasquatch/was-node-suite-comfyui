"""Masks for local adjustments: linear and radial gradients, colour and luminance ranges.

Masks are ``(batch, height, width)`` floats in 0 to 1. Positions are shares of the picture
from its top left, and pixel centres sit at half steps.
"""

from __future__ import annotations

import math
import re
from typing import NamedTuple

import torch
import torch.nn.functional as F

from ..image import raw
from ..image.develop import matte_for, to_oklab
from ..image.guided import filter_with_guide

__all__ = [
    "COMBINE_MODES",
    "MAX_SAMPLES",
    "PROFILES",
    "RANGE_MODES",
    "Sample",
    "color_range",
    "combine",
    "figures",
    "finish",
    "linear_gradient",
    "luminance_range",
    "parse_samples",
    "radial_gradient",
    "ramp",
    "refine",
    "sample_colors",
    "shown_frames",
    "summary",
    "thumbnails",
]

#: How a new mask joins the mask wired in.
COMBINE_MODES = ("replace", "add", "subtract", "intersect")

#: The shape a fade takes between full and none.
PROFILES = ("smooth", "linear")

#: What a range mask selects by.
RANGE_MODES = ("color", "luminance")

#: Lightness, beyond each end of a luminance band, that a smoothness of 1 fades over.
LUMINANCE_FADE = 0.25

#: Side of the square one clicked colour sample averages, as a share of the short side.
SAMPLE_SIDE = 0.01

#: Fewest pixels across a sample's square.
SAMPLE_LEAST = 3

#: Most colour samples one mask reads.
MAX_SAMPLES = 16

#: Refined values within this of 0 or 1 are set to it.
REFINE_SNAP = 1.0 / 255.0

#: Guided filter epsilon for edge refinement, in squared picture codes.
REFINE_EPSILON = 1e-3

#: Frames a panel is sent, evenly spaced, and the long edge they are sent at.
SHOWN_FRAMES = 16
SHOWN_EDGE = 512

#: Shortest a gradient may be, in pixels between its start and end lines.
LEAST_SPAN = 0.5

_HEX = re.compile(r"^#?([0-9a-fA-F]{6})$")


class Sample(NamedTuple):
    """One colour sample.

    Attributes:
        x: Centre across, as a share of the width; NaN for a typed colour.
        y: Centre down, as a share of the height; NaN for a typed colour.
        width: Width of the sampled area as a share of the width, 0 for a click.
        height: Height of the sampled area as a share of the height, 0 for a click.
        frame: Batch index the sample was taken on.
        code: ``(red, green, blue)`` picture codes of a typed colour, or None.
    """

    x: float
    y: float
    width: float
    height: float
    frame: int
    code: tuple[float, float, float] | None


def ramp(t: torch.Tensor, profile: str) -> torch.Tensor:
    """A rise from 0 to 1 over ``t`` in 0 to 1.

    Args:
        t: Positions; held to 0 to 1.
        profile: ``linear`` answers ``t``; ``smooth`` answers ``3t^2 - 2t^3``.

    Returns:
        A tensor shaped like ``t``.
    """
    t = t.clamp(0.0, 1.0)
    if profile == "linear":
        return t
    return t * t * (3.0 - 2.0 * t)


def _fade(distance: torch.Tensor, full: float, zero: float, profile: str) -> torch.Tensor:
    """1 up to ``full``, 0 from ``zero`` on, and the profile's fall between."""
    if zero - full <= 1e-12:
        return (distance <= zero).to(distance.dtype)
    return 1.0 - ramp((distance - full) / (zero - full), profile)


def _grid(height: int, width: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pixel centre coordinates as float64 columns and rows: ``(height, 1)`` and ``(1, width)``."""
    rows = (torch.arange(height, dtype=torch.float64) + 0.5)[:, None]
    columns = (torch.arange(width, dtype=torch.float64) + 0.5)[None, :]
    return rows, columns


def linear_gradient(height: int, width: int, start, end, profile: str = "smooth", node: str = "") -> torch.Tensor:
    """A linear gradient: 1 on and before the start line, 0 on and past the end line.

    Args:
        height: Picture height in pixels.
        width: Picture width in pixels.
        start: ``(x, y)`` the start line passes through, in shares of the picture.
        end: ``(x, y)`` the end line passes through. Both lines are square to the way between.
        profile: One of :data:`PROFILES`.
        node: The node's name, for messages.

    Returns:
        ``(height, width)`` float32 on the CPU.

    Raises:
        ValueError: The two lines are closer than :data:`LEAST_SPAN` pixels.
    """
    sx, sy = float(start[0]) * width, float(start[1]) * height
    ex, ey = float(end[0]) * width, float(end[1]) * height
    dx, dy = ex - sx, ey - sy
    span = dx * dx + dy * dy
    if span < LEAST_SPAN * LEAST_SPAN:
        raise ValueError(
            f"{node}: the start and end lines are the same line on this {width}x{height} picture, so "
            "there is no direction to fade in. Drag a gradient on the node, or move end_x and end_y "
            "away from start_x and start_y."
        )
    rows, columns = _grid(height, width)
    along = ((columns - sx) * dx + (rows - sy) * dy) / span
    return (1.0 - ramp(along, profile)).float()


def radial_gradient(height: int, width: int, centre, radii, angle: float, feather: float,
                    profile: str = "smooth") -> torch.Tensor:
    """A radial gradient: 1 inside the feather's inner ellipse, 0 on the ellipse and outside it.

    Args:
        height: Picture height in pixels.
        width: Picture width in pixels.
        centre: ``(x, y)`` of the centre, in shares of the picture.
        radii: ``(across, down)`` before turning, as shares of the width and of the height.
        angle: Degrees the ellipse is turned clockwise.
        feather: 0 to 100, the share of the radius the fade spans, inward from the edge.
        profile: One of :data:`PROFILES`.

    Returns:
        ``(height, width)`` float32 on the CPU.
    """
    cx, cy = float(centre[0]) * width, float(centre[1]) * height
    across = max(float(radii[0]) * width, 1e-6)
    down = max(float(radii[1]) * height, 1e-6)
    turn = math.radians(float(angle))
    cos, sin = math.cos(turn), math.sin(turn)
    rows, columns = _grid(height, width)
    dx, dy = columns - cx, rows - cy
    u = (dx * cos + dy * sin) / across
    v = (-dx * sin + dy * cos) / down
    reach = torch.sqrt(u * u + v * v)
    share = min(max(float(feather), 0.0), 100.0) / 100.0
    if share <= 0.0:
        return (reach < 1.0).float()
    return _fade(reach, 1.0 - share, 1.0, profile).float()


def combine(made: torch.Tensor, given: torch.Tensor | None, mode: str) -> torch.Tensor:
    """Join a new mask with the mask wired in.

    Args:
        made: The new mask.
        given: The mask wired in, broadcastable onto ``made``, or None.
        mode: ``replace`` answers ``made``; ``add`` the larger of the two; ``subtract``
            ``given * (1 - made)``; ``intersect`` ``given * made``.

    Returns:
        A tensor shaped like ``made``. With no ``given``, ``made`` itself.

    Raises:
        ValueError: ``mode`` is not one of :data:`COMBINE_MODES`.
    """
    if mode not in COMBINE_MODES:
        raise ValueError(f"combine must be one of {', '.join(COMBINE_MODES)}, not {mode!r}.")
    if given is None or mode == "replace":
        return made
    if mode == "add":
        return torch.maximum(given, made)
    if mode == "subtract":
        return given * (1.0 - made)
    return given * made


def finish(images: torch.Tensor, made: torch.Tensor, mask, mode: str, invert: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Invert a new mask where asked, bring it to every frame and join the mask wired in.

    Args:
        images: The batch the mask is for, ``(batch, height, width, channels)``.
        made: ``(height, width)`` or ``(batch, height, width)``.
        mask: The ``MASK`` wired in, any size or length, or None.
        mode: One of :data:`COMBINE_MODES`.
        invert: Whether ``made`` is turned over before joining.

    Returns:
        ``(result, given)``: the ``(batch, height, width)`` mask on the CPU, and the wired mask
        at the batch's size and length, or None.
    """
    batch = int(images.shape[0])
    made = made.float().cpu()
    if made.ndim == 2:
        made = made[None].expand(batch, -1, -1)
    if invert:
        made = 1.0 - made
    given = matte_for(images, mask) if mask is not None else None
    result = combine(made, given, mode).clamp(0.0, 1.0)
    return result.contiguous(), given


def parse_samples(text: str, node: str) -> list[Sample]:
    """Read colour samples.

    Args:
        text: Entries separated by ``;`` or new lines: ``x,y`` for a click, ``x,y,w,h`` for an
            area, either with ``@frame``, or ``#rrggbb`` for a typed colour. Empty for none.
        node: The node's name, for messages.

    Returns:
        The samples, positions held to 0 to 1.

    Raises:
        ValueError: An entry is none of those shapes, or there are more than :data:`MAX_SAMPLES`.
    """
    example = "such as 0.42,0.18@0; 0.7,0.25,0.1,0.06@12; #6fa0e8"
    entries = [entry.strip() for entry in re.split(r"[;\n]+", str(text or "")) if entry.strip()]
    if len(entries) > MAX_SAMPLES:
        raise ValueError(
            f"{node}: samples holds {len(entries)} entries and at most {MAX_SAMPLES} are read. Remove some, "
            "or drag an area over a run of similar colours to sample them as one."
        )
    samples = []
    for entry in entries:
        typed = _HEX.match(entry)
        if typed:
            digits = typed.group(1)
            code = tuple(int(digits[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
            samples.append(Sample(math.nan, math.nan, 0.0, 0.0, 0, code))
            continue
        place, _, frame = entry.partition("@")
        try:
            numbers = [float(part) for part in place.split(",")]
            index = int(frame.strip()) if frame.strip() else 0
        except ValueError as error:
            raise ValueError(
                f"{node}: the sample {entry!r} is not x,y or x,y,w,h with an optional @frame, {example}. "
                "Click the picture on the node to sample a colour."
            ) from error
        if len(numbers) not in (2, 4) or not all(math.isfinite(value) for value in numbers) or index < 0:
            raise ValueError(f"{node}: the sample {entry!r} needs 2 or 4 numbers and a frame of 0 or more, {example}.")
        x, y = (min(max(value, 0.0), 1.0) for value in numbers[:2])
        width, height = (min(max(value, 0.0), 1.0) for value in (numbers[2:] or [0.0, 0.0]))
        samples.append(Sample(x, y, width, height, index, None))
    return samples


def _area(sample: Sample, height: int, width: int) -> tuple[int, int, int, int]:
    """The ``(top, bottom, left, right)`` pixel rows and columns one sample averages."""
    short = min(height, width)
    side = max(SAMPLE_LEAST, round(SAMPLE_SIDE * short))
    across = max(side, round(sample.width * width)) if sample.width > 0 else side
    down = max(side, round(sample.height * height)) if sample.height > 0 else side
    across, down = min(across, width), min(down, height)
    left = min(max(round(sample.x * width - across / 2.0), 0), width - across)
    top = min(max(round(sample.y * height - down / 2.0), 0), height - down)
    return top, top + down, left, left + across


def sample_colors(images: torch.Tensor, samples) -> tuple[torch.Tensor, list[dict]]:
    """The colour each sample stands for.

    Args:
        images: ``(batch, height, width, channels)`` picture codes.
        samples: From :func:`parse_samples`. A sample past the last frame reads the last.

    Returns:
        ``(colours, described)``: ``(count, 3)`` Oklab float32, and per sample its ``hex``,
        ``L``, ``a``, ``b``, ``frame`` and the pixel ``area`` averaged, empty for a typed colour.
    """
    batch, height, width = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    lights, described = [], []
    for sample in samples:
        if sample.code is not None:
            light = raw.linearise(torch.tensor(sample.code, dtype=torch.float32))
            frame, area = None, []
        else:
            frame = min(sample.frame, batch - 1)
            top, bottom, left, right = _area(sample, height, width)
            patch = images[frame, top:bottom, left:right, :3].float().cpu()
            light = raw.linearise(patch).reshape(-1, 3).mean(dim=0)
            area = [left, top, right - left, bottom - top]
        lights.append(light)
        code = raw.encode(light).clamp(0.0, 1.0)
        lab = to_oklab(light[None])[0]
        described.append({
            "hex": "#" + "".join(f"{round(float(value) * 255):02x}" for value in code),
            "L": round(float(lab[0]), 4),
            "a": round(float(lab[1]), 4),
            "b": round(float(lab[2]), 4),
            "frame": frame,
            "area": area,
        })
    if not lights:
        return torch.zeros((0, 3)), described
    return to_oklab(torch.stack(lights)).float(), described


def color_range(codes: torch.Tensor, colours: torch.Tensor, reach: float, smoothness: float,
                lightness_weight: float, profile: str = "smooth") -> torch.Tensor:
    """How near each pixel's colour is to the nearest sampled colour.

    Args:
        codes: ``(frames, height, width, 3)`` picture codes.
        colours: ``(count, 3)`` Oklab colours.
        reach: Distance at which the selection reaches 0.
        smoothness: 0 to 1, the share of ``reach`` the fade spans.
        lightness_weight: How much a lightness difference counts; 1 is plain Oklab distance.
        profile: One of :data:`PROFILES`.

    Returns:
        ``(frames, height, width)`` on the device ``codes`` is on; zeros with no colours.
    """
    frames, height, width = int(codes.shape[0]), int(codes.shape[1]), int(codes.shape[2])
    if int(colours.shape[0]) == 0:
        return torch.zeros((frames, height, width), dtype=torch.float32, device=codes.device)
    lab = to_oklab(raw.linearise(codes[..., :3].float()))
    weights = torch.tensor([float(lightness_weight), 1.0, 1.0], dtype=lab.dtype, device=lab.device)
    nearest = None
    for colour in colours.to(lab.device, lab.dtype):
        squared = ((lab - colour) ** 2 * weights).sum(dim=-1)
        nearest = squared if nearest is None else torch.minimum(nearest, squared)
    distance = nearest.sqrt()
    share = min(max(float(smoothness), 0.0), 1.0)
    return _fade(distance, float(reach) * (1.0 - share), float(reach), profile)


def luminance_range(codes: torch.Tensor, low: float, high: float, smoothness: float,
                    profile: str = "smooth") -> torch.Tensor:
    """How far inside a band of Oklab lightness each pixel sits.

    Args:
        codes: ``(frames, height, width, 3)`` picture codes.
        low: Darkest lightness fully selected, 0 to 1.
        high: Lightest lightness fully selected. The two may come in either order.
        smoothness: 0 to 1; the fade spans ``smoothness * LUMINANCE_FADE`` beyond each end.
        profile: One of :data:`PROFILES`.

    Returns:
        ``(frames, height, width)`` on the device ``codes`` is on.
    """
    lightness = to_oklab(raw.linearise(codes[..., :3].float()))[..., 0]
    low, high = sorted((float(low), float(high)))
    outside = torch.maximum(low - lightness, lightness - high).clamp(min=0.0)
    fade = min(max(float(smoothness), 0.0), 1.0) * LUMINANCE_FADE
    return _fade(outside, 0.0, fade, profile)


def refine(masks: torch.Tensor, codes: torch.Tensor, amount: float) -> torch.Tensor:
    """Snap masks to the edges of the pictures they were made from.

    Args:
        masks: ``(frames, height, width)``.
        codes: ``(frames, height, width, channels)`` picture codes, the guide.
        amount: Guided filter radius in hundredths of the short side; 0 returns ``masks``.

    Returns:
        ``(frames, height, width)``, values within :data:`REFINE_SNAP` of 0 or 1 set to it.
    """
    if amount <= 0.0:
        return masks
    radius = max(1, round(float(amount) / 100.0 * min(int(masks.shape[1]), int(masks.shape[2]))))
    guide = codes[..., :3].float().clamp(0.0, 1.0)
    held = filter_with_guide(masks[..., None].float(), guide, radius, REFINE_EPSILON)[..., 0]
    held = torch.where(held <= REFINE_SNAP, torch.zeros_like(held), held)
    return torch.where(held >= 1.0 - REFINE_SNAP, torch.ones_like(held), held)


def shown_frames(count: int) -> list[int]:
    """Up to :data:`SHOWN_FRAMES` evenly spaced batch indices, first and last among them."""
    if count <= SHOWN_FRAMES:
        return list(range(count))
    return sorted({round(i * (count - 1) / (SHOWN_FRAMES - 1)) for i in range(SHOWN_FRAMES)})


def thumbnails(pictures: torch.Tensor, indices) -> torch.Tensor:
    """Chosen frames of pictures or masks, brought down to :data:`SHOWN_EDGE` on their long side.

    Args:
        pictures: ``(batch, height, width, channels)`` or ``(batch, height, width)``.
        indices: Batch indices to take, one frame reduced at a time.

    Returns:
        The same layout holding those frames, reduced by area averaging where they are larger.
    """
    height, width = int(pictures.shape[1]), int(pictures.shape[2])
    scale = SHOWN_EDGE / max(height, width)
    size = (max(1, round(height * scale)), max(1, round(width * scale)))
    kept = []
    for index in indices:
        frame = pictures[index:index + 1].float().cpu()
        if scale < 1.0:
            planes = frame[..., None] if pictures.ndim == 3 else frame
            planes = F.interpolate(planes.movedim(-1, 1), size=size, mode="area").movedim(1, -1)
            frame = planes[..., 0] if pictures.ndim == 3 else planes
        kept.append(frame)
    return torch.cat(kept, dim=0)


def summary(result: torch.Tensor, given, mode: str) -> tuple[bool, str, str]:
    """What one run made, in words.

    Args:
        result: The ``(batch, height, width)`` mask answered.
        given: The wired mask, or None.
        mode: One of :data:`COMBINE_MODES`.

    Returns:
        ``(warn, line, joined)``: whether the run deserves a warning, the summary line, and how
        the mask was joined.
    """
    count = int(result.shape[0])
    noun = "frame" if count == 1 else "frames"
    if given is None:
        joined = "alone" if mode == "replace" else f"alone: no mask is wired for `{mode}` to join"
    else:
        joined = {
            "replace": "alone, the wired mask unused",
            "add": "added to the wired mask",
            "subtract": "taken out of the wired mask",
            "intersect": "kept only where the wired mask is",
        }[mode]
    measured = figures(result)
    if measured["empty"]:
        return True, f"the mask came out empty on {count} {noun}", joined
    line = f"covers {measured['coverage']:.2f}% of {count} {noun}, {joined}"
    return given is None and mode != "replace", line, joined


def figures(result: torch.Tensor) -> dict:
    """Coverage of a mask batch as a panel prints it.

    Args:
        result: ``(batch, height, width)``.

    Returns:
        ``coverage``, the mean value as a percentage; ``selected``, the percentage at 0.5 or
        above; ``full``, the percentage at 1; and ``empty``, whether every value is 0.
    """
    count = max(1, result.numel())
    return {
        "coverage": round(float(result.double().mean()) * 100.0, 2),
        "selected": round(float((result >= 0.5).sum()) * 100.0 / count, 2),
        "full": round(float((result >= 1.0).sum()) * 100.0 / count, 2),
        "empty": not bool((result > 0).any()),
    }

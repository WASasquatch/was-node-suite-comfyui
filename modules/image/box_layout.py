"""Layout boxes: reading the layout JSON, refitting it onto a frame, and the node's outputs.

Boxes are in integer pixels from the frame's top left. Prompt elements hold ``[ymin, xmin,
ymax, xmax]`` on a 0 to 1000 grid.
"""

from __future__ import annotations

import json
import math
import re

__all__ = [
    "BBOXES_MODES",
    "KINDS",
    "MAX_BOXES",
    "MAX_COORDINATE",
    "MAX_FRAME",
    "MAX_LAYOUT_CHARS",
    "MAX_MASK_VALUES",
    "MAX_PALETTE",
    "MAX_TEXT_CHARS",
    "PALETTE",
    "PREVIEW_LONG_EDGE",
    "RESIZE_MODES",
    "character_numbers",
    "colour",
    "incoming_boxes",
    "mask_refusal",
    "masks",
    "palette_of",
    "parse_layout",
    "read_box",
    "refit",
    "render_preview",
    "report_box",
    "to_bbox_frame",
    "to_elements",
]

#: Display colours given to boxes in order, repeating after the last.
PALETTE = (
    "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2",
    "#D55E00", "#CC79A7", "#A6D854", "#E78AC3", "#B3B3B3",
)

#: What a box stands for. ``character`` boxes are numbered on the ``character_boxes`` output.
KINDS = ("object", "character", "text")

#: How a layout drawn at one frame size is placed on another.
RESIZE_MODES = ("fit", "fill", "stretch", "keep size")

#: How incoming boxes join the layout's own.
BBOXES_MODES = ("add", "replace", "ignore")

#: Longest layout text read, in characters.
MAX_LAYOUT_CHARS = 262144

#: Most boxes one run outputs, the layout's own and incoming together.
MAX_BOXES = 64

#: Longest caption or text one box carries, in characters.
MAX_TEXT_CHARS = 2000

#: Largest absolute value of any box coordinate, in pixels.
MAX_COORDINATE = 65536

#: Largest side of the frame a layout records, in pixels.
MAX_FRAME = 16384

#: Most values the masks output holds, masks times height times width.
MAX_MASK_VALUES = 1 << 27

#: Most colours one box's palette holds.
MAX_PALETTE = 5

#: Longest edge of the preview image, in pixels.
PREVIEW_LONG_EDGE = 1024

#: ``#rgb``, ``#rrggbb`` or ``#rrggbbaa``.
HEX_COLOUR = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")

#: Preview backdrop when no image is wired.
BACKDROP = (0x20, 0x21, 0x24)

#: Brightness the wired image is drawn at behind the boxes.
BACKDROP_DIM = 0.3

#: Alpha of a box's fill, out of 255.
FILL_ALPHA = 36

#: Caption text and the plate behind it.
CAPTION_INK = (240, 240, 240, 255)
CAPTION_PLATE = (0, 0, 0, 165)

#: Longest preview of a rejected value quoted in an error.
QUOTED_CHARS = 40

#: What a pasted value is accepted as, for the error naming it.
PASTE_SHAPES = (
    "a layout from another Bounding Box Editor, a Create Bounding Boxes box list or a "
    "prompt element list"
)


def colour(value) -> str | None:
    """One hex colour as lowercase ``#rrggbb``, or None when it is not one.

    Args:
        value: ``#rgb``, ``#rrggbb`` or ``#rrggbbaa``; the alpha pair is dropped.

    Returns:
        The colour, or None for anything else.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not HEX_COLOUR.match(text):
        return None
    digits = text[1:].lower()
    if len(digits) == 3:
        digits = "".join(digit * 2 for digit in digits)
    return "#" + digits[:6]


def palette_of(value) -> list[str]:
    """The colours of a palette as :func:`colour` reads them, at most :data:`MAX_PALETTE`.

    Args:
        value: A list of colour strings, a dict whose values are colours, or one colour.

    Returns:
        The valid colours in order; the rest are dropped.
    """
    if isinstance(value, str):
        value = [value]
    elif isinstance(value, dict):
        value = list(value.values())
    elif not isinstance(value, (list, tuple)):
        return []
    found = []
    for entry in value:
        normal = colour(entry)
        if normal is not None:
            found.append(normal)
            if len(found) == MAX_PALETTE:
                break
    return found


def _quoted(value) -> str:
    """A rejected value, shortened for an error message."""
    text = repr(value)
    return text if len(text) <= QUOTED_CHARS else text[:QUOTED_CHARS] + "..."


def _number(value, what: str) -> float:
    """One finite number within :data:`MAX_COORDINATE`.

    Args:
        value: The number, or text holding one.
        what: How the error names the value, such as ``layout box 3 x``.

    Returns:
        The value as a float.

    Raises:
        ValueError: It is missing, not a number, not finite or past the ceiling.
    """
    if value is None:
        raise ValueError(
            f"The {what} is missing. Every box needs x, y, width and height in pixels; draw the "
            f"box again in the panel, or add the value to the pasted text."
        )
    if isinstance(value, bool):
        raise ValueError(f"The {what} is {value!r}, not a number of pixels. Correct it in the pasted text.")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"The {what} is {_quoted(value)}, not a number of pixels. Correct it in the pasted text."
        ) from None
    if not math.isfinite(number):
        raise ValueError(f"The {what} is {value!r}, not a finite number. Correct it in the pasted text.")
    if abs(number) > MAX_COORDINATE:
        raise ValueError(
            f"The {what} is {number:g}, past the {MAX_COORDINATE} pixel limit on any coordinate. "
            f"Move the box inside the frame in the panel, or correct the pasted text."
        )
    return number


def _words(value, what: str) -> str:
    """A caption or text, held to :data:`MAX_TEXT_CHARS`.

    Raises:
        ValueError: It is longer than the ceiling.
    """
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(
            f"The {what} is {len(text)} characters, past the {MAX_TEXT_CHARS} one box carries. "
            f"Shorten it in the panel or in the pasted text."
        )
    return text


def _from_metadata(box: dict) -> dict:
    """A bounding box carrying ``metadata``, or a detector's ``label``, in the layout shape."""
    meta = box.get("metadata") if isinstance(box.get("metadata"), dict) else None
    label = box.get("label") if isinstance(box.get("label"), str) else ""
    if meta is None:
        if label and "caption" not in box:
            return {**box, "caption": label}
        return box
    kind = meta.get("kind")
    if kind not in KINDS:
        kind = "text" if meta.get("type") == "text" else "object"
    desc = meta.get("desc")
    return {
        "x": box.get("x"),
        "y": box.get("y"),
        "width": box.get("width"),
        "height": box.get("height"),
        "kind": kind,
        "caption": desc if isinstance(desc, str) and desc else label,
        "text": meta.get("text", ""),
        "palette": meta.get("palette", []),
        "color": meta.get("color"),
    }


def read_box(raw: dict, index: int, where: str = "layout box") -> dict:
    """One box in the layout shape, every value checked.

    Args:
        raw: The box as pasted or written by the panel, or a bounding box with ``metadata``.
        index: Its position, counting from 0, which picks the default colour.
        where: How errors name the box, followed by its 1-based number.

    Returns:
        A new box dict holding only the layout keys.

    Raises:
        ValueError: A coordinate, caption or text breaks a ceiling, or the box is under 1 pixel.
    """
    raw = _from_metadata(raw)
    name = f"{where} {index + 1}"
    x = round(_number(raw.get("x"), f"{name} x"))
    y = round(_number(raw.get("y"), f"{name} y"))
    width = round(_number(raw.get("width"), f"{name} width"))
    height = round(_number(raw.get("height"), f"{name} height"))
    if width < 1 or height < 1:
        raise ValueError(
            f"The {name} is {width}x{height} pixels after rounding; a box needs at least 1 pixel "
            f"each way. Draw it larger in the panel, or remove it from the pasted text."
        )
    kind = raw.get("kind")
    ratio = raw.get("ratio")
    return {
        "kind": kind if kind in KINDS else "object",
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "caption": _words(raw.get("caption"), f"caption of {name}"),
        "text": _words(raw.get("text"), f"text of {name}"),
        "palette": palette_of(raw.get("palette")),
        "color": colour(raw.get("color")) or PALETTE[index % len(PALETTE)],
        "ratio": ratio if isinstance(ratio, str) else "",
        "locked": raw.get("locked") is True,
        "hidden": raw.get("hidden") is True,
    }


def _is_element(value) -> bool:
    """Whether a dict is a prompt element, holding a four-value ``bbox``."""
    bbox = value.get("bbox") if isinstance(value, dict) else None
    return isinstance(bbox, (list, tuple)) and len(bbox) == 4


def _is_box(value) -> bool:
    """Whether a dict holds ``x``, ``y``, ``width`` and ``height``."""
    return isinstance(value, dict) and all(key in value for key in ("x", "y", "width", "height"))


def _element_box(element: dict, size, what: str) -> dict:
    """A prompt element as a bounding box in pixels of ``size``.

    Raises:
        ValueError: Its ``bbox`` is not four finite numbers.
    """
    width, height = size
    bbox = element.get("bbox")
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        raise ValueError(
            f"The {what} has no valid 'bbox' [ymin, xmin, ymax, xmax] on the 0 to 1000 grid. "
            f"Correct the pasted element list."
        )
    try:
        ymin, xmin, ymax, xmax = (float(value) / 1000.0 for value in bbox)
    except (TypeError, ValueError):
        raise ValueError(
            f"The {what} 'bbox' must hold four numbers on the 0 to 1000 grid. Correct the "
            f"pasted element list."
        ) from None
    if not all(math.isfinite(value) for value in (ymin, xmin, ymax, xmax)):
        raise ValueError(f"The {what} 'bbox' holds a value that is not finite. Correct the pasted element list.")
    kind = "text" if element.get("type") == "text" else "object"
    return {
        "x": round(_number(min(xmin, xmax) * width, f"{what} x")),
        "y": round(_number(min(ymin, ymax) * height, f"{what} y")),
        "width": round(_number(abs(xmax - xmin) * width, f"{what} width")),
        "height": round(_number(abs(ymax - ymin) * height, f"{what} height")),
        "kind": kind,
        "caption": element.get("desc", ""),
        "text": element.get("text", "") if kind == "text" else "",
        "palette": element.get("color_palette", []) or [],
    }


def _frame(value) -> tuple[int, int]:
    """A layout's recorded frame as ``(width, height)``.

    Raises:
        ValueError: It is not two numbers, or a side is outside 1 to :data:`MAX_FRAME`.
    """
    if not (isinstance(value, (list, tuple)) and len(value) == 2):
        raise ValueError(
            f"The layout's frame is {_quoted(value)}, not [width, height]. Draw the layout again "
            f"in the panel, or correct 'frame' in the pasted layout."
        )
    sides = []
    for side in value:
        if isinstance(side, bool) or not isinstance(side, (int, float)) or not math.isfinite(side):
            sides = None
            break
        sides.append(round(side))
    if sides is None or not all(1 <= side <= MAX_FRAME for side in sides):
        raise ValueError(
            f"The layout's frame is {_quoted(value)}, and each side must be 1 to {MAX_FRAME} "
            f"pixels. Draw the layout again in the panel, or correct 'frame' in the pasted layout."
        )
    return sides[0], sides[1]


def _too_many(count: int, what: str, fix: str) -> None:
    """Refuse more than :data:`MAX_BOXES` boxes.

    Raises:
        ValueError: ``count`` is past the ceiling.
    """
    if count > MAX_BOXES:
        raise ValueError(f"{what} holds {count} boxes, past the {MAX_BOXES} the node reads. {fix}")


def parse_layout(text, size) -> tuple[list[dict], tuple[int, int] | None]:
    """Read the layout widget into boxes and the frame they were drawn at.

    Args:
        text: The layout JSON, or a pasted box list, prompt element list or core box value.
        size: ``(width, height)`` of the run's frame, which pasted elements are placed in.

    Returns:
        ``(boxes, frame)``. Hidden boxes are included. ``frame`` is None when the text
        recorded none, so the boxes are already in the run's pixels.

    Raises:
        ValueError: The text is unreadable or breaks a ceiling.
    """
    if text is None:
        return [], None
    if not isinstance(text, str):
        raise ValueError(f"The layout is a {type(text).__name__}, not text. Wire a string, or draw the boxes in the panel.")
    if len(text) > MAX_LAYOUT_CHARS:
        raise ValueError(
            f"The layout text is {len(text)} characters, past the {MAX_LAYOUT_CHARS} the node "
            f"reads. Clear the layout and draw the boxes again in the panel, or paste a shorter one."
        )
    if not text.strip():
        return [], None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"The layout text is not readable ({error.msg} at line {error.lineno}, column "
            f"{error.colno}). Draw the boxes again in the panel, or paste a valid layout: "
            f"{PASTE_SHAPES}."
        ) from None

    frame = None
    if isinstance(data, dict):
        if "boxes" in data:
            if data.get("frame") is not None:
                frame = _frame(data["frame"])
            data = data["boxes"]
            if data is None:
                data = []
            if not isinstance(data, list):
                raise ValueError(
                    f"The layout's 'boxes' is a {type(data).__name__}, not a list. Draw the "
                    f"layout again in the panel, or paste {PASTE_SHAPES}."
                )
        elif _is_element(data) or _is_box(data):
            data = [data]
        else:
            raise ValueError(
                f"The layout text holds an object with no 'boxes', 'bbox' or x, y, width and "
                f"height. Draw the layout again in the panel, or paste {PASTE_SHAPES}."
            )
    if not isinstance(data, list):
        raise ValueError(
            f"The layout text holds a {type(data).__name__}, not a layout. Draw the layout "
            f"again in the panel, or paste {PASTE_SHAPES}."
        )
    if data and isinstance(data[0], list):
        data = data[0]
    entries = [entry for entry in data if isinstance(entry, dict)]
    _too_many(
        len(entries), "The layout",
        "Delete boxes in the panel, or paste a shorter list.",
    )
    if entries and _is_element(entries[0]):
        here = frame or tuple(size)
        entries = [
            _element_box(entry, here, f"pasted element {index + 1}")
            for index, entry in enumerate(entries)
        ]
    return [read_box(entry, index) for index, entry in enumerate(entries)], frame


def refit(boxes, frame, size, mode: str, grid: int) -> list[dict]:
    """Place boxes drawn on ``frame`` onto ``size``, then hold them inside it.

    Args:
        boxes: Boxes in the layout shape, in pixels of ``frame``.
        frame: ``(width, height)`` the boxes were drawn at.
        size: ``(width, height)`` of the run's frame.
        mode: A member of :data:`RESIZE_MODES`.
        grid: Pixels a scaled corner and a scaled size snap to; 1 rounds to the pixel.

    Returns:
        New boxes with integer edges inside ``size``, every other key carried over. Corner and
        size snap separately, so equal sides stay equal. A box left under 1 pixel either way is
        dropped. Equal sizes only clamp.

    Raises:
        ValueError: ``mode`` is not one of :data:`RESIZE_MODES`.
    """
    if mode not in RESIZE_MODES:
        raise ValueError(
            f"on_resize is {_quoted(mode)}; it takes one of {', '.join(RESIZE_MODES)}."
        )
    fw, fh = frame
    W, H = size
    grid = int(grid)
    same = fw == W and fh == H
    sx = sy = 1.0
    ox = oy = 0.0
    if not same:
        if mode == "stretch":
            sx, sy = W / fw, H / fh
            ox = oy = 0.0
        elif mode == "keep size":
            sx = sy = 1.0
            ox = (W - fw) / 2
            oy = (H - fh) / 2
        else:
            s = min(W / fw, H / fh) if mode == "fit" else max(W / fw, H / fh)
            sx = sy = s
            ox = (W - fw * s) / 2
            oy = (H - fh * s) / 2
    placed = []
    for box in boxes:
        x, y, w, h = box["x"], box["y"], box["width"], box["height"]
        if same:
            edges = (x, y, x + w, y + h)
        else:
            if grid > 1:
                left, top = round((x * sx + ox) / grid) * grid, round((y * sy + oy) / grid) * grid
                wide, high = (max(grid, round(w * sx / grid) * grid), max(grid, round(h * sy / grid) * grid))
            else:
                left, top = round(x * sx + ox), round(y * sy + oy)
                wide, high = max(1, round(w * sx)), max(1, round(h * sy))
            edges = (left, top, left + wide, top + high)
        x0 = max(0, min(W, edges[0]))
        y0 = max(0, min(H, edges[1]))
        x1 = max(0, min(W, edges[2]))
        y1 = max(0, min(H, edges[3]))
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        placed.append({**box, "x": int(x0), "y": int(y0), "width": int(x1 - x0), "height": int(y1 - y0)})
    return placed


def incoming_boxes(data, size, start: int) -> list[dict]:
    """Read the ``bboxes`` input into layout boxes in the run's pixels, held inside the frame.

    Args:
        data: A bounding box, a list of them, a detector's per-frame lists (the first frame
            is read), prompt elements, or JSON text holding any of those.
        size: ``(width, height)`` of the run's frame.
        start: How many boxes the layout holds, which offsets the default colours.

    Returns:
        Boxes in the layout shape. A box with no area inside the frame is dropped.

    Raises:
        ValueError: The value is unreadable or breaks a ceiling.
    """
    if data is None:
        return []
    if isinstance(data, str):
        if len(data) > MAX_LAYOUT_CHARS:
            raise ValueError(
                f"The bboxes text is {len(data)} characters, past the {MAX_LAYOUT_CHARS} the "
                f"node reads. Wire fewer boxes, or set bboxes_mode to ignore."
            )
        if not data.strip():
            return []
        try:
            data = json.loads(data)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"The bboxes text is not readable as JSON ({error.msg} at line {error.lineno}, "
                f"column {error.colno}). Wire bounding boxes or prompt elements, or correct the text."
            ) from None
    if isinstance(data, dict):
        if not (_is_element(data) or _is_box(data)):
            raise ValueError(
                "The bboxes input is an object that is neither a bounding box (x, y, width, "
                "height) nor a prompt element (with a 'bbox'). Wire boxes from a detector, "
                "Create Bounding Boxes or another Bounding Box Editor."
            )
        data = [data]
    if not isinstance(data, list):
        raise ValueError(
            f"The bboxes input is a {type(data).__name__}; it takes bounding boxes, prompt "
            f"elements or JSON text holding either."
        )
    if data and isinstance(data[0], list):
        data = data[0]
    entries = [entry for entry in data if isinstance(entry, dict)]
    _too_many(
        len(entries), "The bboxes input",
        "Thin them with Bounding Boxes Filter, or set bboxes_mode to ignore.",
    )
    if not entries:
        return []
    if _is_element(entries[0]):
        entries = [
            _element_box(entry, size, f"incoming element {index + 1}")
            for index, entry in enumerate(entries)
        ]
    elif not _is_box(entries[0]):
        raise ValueError(
            "The bboxes input holds items that are neither bounding boxes (x, y, width, "
            "height) nor prompt elements (with a 'bbox'). Wire boxes from a detector, Create "
            "Bounding Boxes or another Bounding Box Editor."
        )
    found = []
    for index, entry in enumerate(entries):
        raw = dict(_from_metadata(entry))
        name = f"incoming box {index + 1}"
        x = _number(raw.get("x"), f"{name} x")
        y = _number(raw.get("y"), f"{name} y")
        w = _number(raw.get("width"), f"{name} width")
        h = _number(raw.get("height"), f"{name} height")
        if w < 0:
            x, w = x + w, -w
        if h < 0:
            y, h = y + h, -h
        x, y, w, h = round(x), round(y), round(w), round(h)
        if w < 1 or h < 1:
            continue
        found.append({
            "kind": raw.get("kind") if raw.get("kind") in KINDS else "object",
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "caption": _words(raw.get("caption"), f"caption of {name}"),
            "text": _words(raw.get("text"), f"text of {name}"),
            "palette": palette_of(raw.get("palette")),
            "color": colour(raw.get("color")) or PALETTE[(start + index) % len(PALETTE)],
        })
    return refit(found, size, size, RESIZE_MODES[0], 1)


def report_box(box: dict) -> dict:
    """One box with the keys the panel reads for an incoming box."""
    keys = ("kind", "x", "y", "width", "height", "caption", "text", "palette", "color")
    return {key: box[key] for key in keys}


def to_bbox_frame(boxes) -> list:
    """Boxes as one frame of ``BOUNDING_BOX`` values, ``[]`` when there are none.

    Returns:
        ``[[{x, y, width, height, metadata}]]``, metadata holding ``type``, ``text``,
        ``desc``, ``palette``, ``kind`` and ``color``.
    """
    frame = [
        {
            "x": box["x"],
            "y": box["y"],
            "width": box["width"],
            "height": box["height"],
            "metadata": {
                "type": "text" if box["kind"] == "text" else "obj",
                "text": box["text"],
                "desc": box["caption"],
                "palette": list(box["palette"]),
                "kind": box["kind"],
                "color": box["color"],
            },
        }
        for box in boxes
    ]
    return [frame] if frame else []


def _grid(value: float) -> int:
    """A fraction of the frame on the 0 to 1000 grid, clamped."""
    return max(0, min(1000, round(value * 1000)))


def _element_bbox(region: dict) -> list[int]:
    """A fractional region as ``[ymin, xmin, ymax, xmax]`` on the 0 to 1000 grid."""
    x, y = region.get("x", 0.0), region.get("y", 0.0)
    w, h = region.get("w", 0.0), region.get("h", 0.0)
    ymin, xmin, ymax, xmax = _grid(y), _grid(x), _grid(y + h), _grid(x + w)
    if ymin > ymax:
        ymin, ymax = ymax, ymin
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    return [ymin, xmin, ymax, xmax]


def _element_palette(colours) -> list[str]:
    """Palette entries as prompt elements carry them, upper case."""
    if isinstance(colours, dict):
        colours = colours.values()
    return [entry.upper() for entry in colours if isinstance(entry, str) and entry]


def to_elements(boxes, size) -> list[dict]:
    """Boxes as prompt elements, the list MOSS-Image and FLUX 3 layout prompts read.

    Args:
        boxes: Boxes in the layout shape.
        size: ``(width, height)`` of the frame the boxes are in.

    Returns:
        One ``{type, bbox, text, desc, color_palette}`` element per box. ``text`` is on text
        boxes only and ``color_palette`` only where a palette is set.
    """
    width = size[0] or 1
    height = size[1] or 1
    elements = []
    for box in boxes:
        region = {
            "x": box["x"] / width,
            "y": box["y"] / height,
            "w": box["width"] / width,
            "h": box["height"] / height,
        }
        kind = "text" if box["kind"] == "text" else "obj"
        element = {"type": kind, "bbox": _element_bbox(region)}
        if kind == "text":
            element["text"] = box["text"]
        element["desc"] = box["caption"]
        palette = _element_palette(box["palette"])
        if palette:
            element["color_palette"] = palette[:MAX_PALETTE]
        elements.append(element)
    return elements


def character_numbers(boxes) -> str:
    """The 1-based numbers of the ``character`` boxes, as ``"2, 5"``, empty when none."""
    return ", ".join(str(number) for number, box in enumerate(boxes, 1) if box["kind"] == "character")


def mask_refusal(count: int, size) -> str | None:
    """Why the masks output cannot be built, or None when it fits :data:`MAX_MASK_VALUES`.

    Args:
        count: How many boxes are output; none still builds one empty mask.
        size: ``(width, height)`` of the frame.
    """
    width, height = size
    planes = max(1, int(count))
    values = planes * int(width) * int(height)
    if values <= MAX_MASK_VALUES:
        return None
    return (
        f"Bounding Box Editor's masks output would hold {planes} mask(s) of {width}x{height}, "
        f"{values} values, past the {MAX_MASK_VALUES} it builds. Use fewer boxes or a smaller "
        f"frame, or leave masks unconnected."
    )


def masks(boxes, size):
    """One mask per box, 1.0 inside it, shaped ``(count, height, width)``.

    Args:
        boxes: Boxes in the layout shape, inside the frame.
        size: ``(width, height)`` of the frame.

    Returns:
        A float32 tensor; one all-zero mask when there are no boxes.

    Raises:
        ValueError: The masks would pass :data:`MAX_MASK_VALUES`.
    """
    import torch

    refusal = mask_refusal(len(boxes), size)
    if refusal:
        raise ValueError(refusal)
    width, height = size
    out = torch.zeros((max(1, len(boxes)), height, width), dtype=torch.float32)
    for index, box in enumerate(boxes):
        out[index, box["y"]:box["y"] + box["height"], box["x"]:box["x"] + box["width"]] = 1.0
    return out


def _rgb(value: str) -> tuple[int, int, int]:
    """A colour as an RGB triple, mid grey when it is not one."""
    normal = colour(value)
    if normal is None:
        return (140, 140, 140)
    return (int(normal[1:3], 16), int(normal[3:5], 16), int(normal[5:7], 16))


def _font(size: int):
    """Pillow's default font at ``size``, or at its own size where it cannot scale."""
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size)
    except Exception:
        return ImageFont.load_default()


def _wrap(draw, text: str, font, width: float) -> list[str]:
    """Text broken into lines no wider than ``width``, paragraphs kept."""
    lines = []
    for paragraph in text.split("\n"):
        line = ""
        for word in paragraph.split():
            trial = word if not line else line + " " + word
            if line and draw.textlength(trial, font=font) > width:
                lines.append(line)
                line = word
            else:
                line = trial
        lines.append(line)
    return lines


def _backdrop(image, width: int, height: int):
    """The preview's base picture: the image's first frame dimmed, or a flat backdrop."""
    from PIL import Image, ImageEnhance

    if image is not None:
        from ..convert.tensors import image_planes, plane2pil

        planes = image_planes(image)
        if planes:
            picture = plane2pil(planes[0]).convert("RGB").resize((width, height), Image.LANCZOS)
            return ImageEnhance.Brightness(picture).enhance(BACKDROP_DIM)
    return Image.new("RGB", (width, height), BACKDROP)


def render_preview(boxes, size, image=None):
    """Draw the boxes, numbered and captioned, over the frame.

    Args:
        boxes: Boxes in the layout shape, back to front.
        size: ``(width, height)`` of the frame.
        image: An ``IMAGE`` tensor whose first frame is the backdrop, or None.

    Returns:
        A float32 ``(1, height, width, 3)`` tensor, long edge at most :data:`PREVIEW_LONG_EDGE`.
    """
    import numpy as np
    import torch
    from PIL import ImageDraw

    width, height = size
    scale = min(1.0, PREVIEW_LONG_EDGE / (max(width, height) or 1))
    rw, rh = max(1, round(width * scale)), max(1, round(height * scale))
    picture = _backdrop(image, rw, rh)
    draw = ImageDraw.Draw(picture, "RGBA")
    size_px = max(10, round(rh / 64))
    font = _font(size_px)
    tag_font = _font(max(9, size_px - 2))
    line_height = size_px + 2

    for number, box in enumerate(boxes, 1):
        r, g, b = _rgb(box["color"])
        x0 = max(0, min(rw, round(box["x"] * scale)))
        y0 = max(0, min(rh, round(box["y"] * scale)))
        x1 = max(0, min(rw, round((box["x"] + box["width"]) * scale)))
        y1 = max(0, min(rh, round((box["y"] + box["height"]) * scale)))
        if x1 <= x0 or y1 <= y0:
            continue
        draw.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(r, g, b, FILL_ALPHA), outline=(r, g, b, 255), width=2)

        top = y0
        swatches = box["palette"][:MAX_PALETTE]
        if swatches and (x1 - x0) > 2:
            strip = max(5, size_px // 2)
            share = (x1 - x0) / len(swatches)
            for slot, entry in enumerate(swatches):
                left = x0 + round(slot * share)
                right = x0 + round((slot + 1) * share) - 1
                draw.rectangle([left, y0, max(left, right), y0 + strip - 1], fill=_rgb(entry) + (255,))
            top = y0 + strip

        tag = str(number)
        tag_width = draw.textlength(tag, font=tag_font)
        draw.rectangle([x0, top, x0 + tag_width + 6, top + size_px + 2], fill=(r, g, b, 255))
        ink = (0, 0, 0, 255) if (0.299 * r + 0.587 * g + 0.114 * b) > 140 else (255, 255, 255, 255)
        draw.text((x0 + 3, top + 1), tag, fill=ink, font=tag_font)

        body = box["caption"]
        if box["kind"] == "text" and box["text"]:
            body = f'"{box["text"]}"' + (f"\n{body}" if body else "")
        if not body or (x1 - x0) <= 12:
            continue
        lines = []
        text_top = top + size_px + 5
        for line in _wrap(draw, body, font, x1 - x0 - 12):
            if text_top + (len(lines) + 1) * line_height > y1:
                break
            lines.append(line)
        if not lines:
            continue
        widest = max(draw.textlength(line, font=font) for line in lines)
        draw.rectangle(
            [x0 + 2, text_top - 2, min(x1 - 1, x0 + 2 + widest + 8), text_top + len(lines) * line_height],
            fill=CAPTION_PLATE,
        )
        for row, line in enumerate(lines):
            draw.text((x0 + 6, text_top + row * line_height), line, fill=CAPTION_INK, font=font)

    array = np.asarray(picture, dtype=np.float32) / 255.0
    return torch.from_numpy(array.copy()).unsqueeze(0)

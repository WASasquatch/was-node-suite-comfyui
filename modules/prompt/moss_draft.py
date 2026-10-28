"""MOSS-Image caption and layout drafts, compiled into the prompt MOSS-Image reads.

A draft box is ``[x_min, y_min, x_max, y_max]`` in pixels from the top left; a compiled box is
``[y_min, x_min, y_max, x_max]`` from the centre.
"""

from __future__ import annotations

import json
import math
import re
from typing import NamedTuple

from ..image import box_layout

__all__ = [
    "CAPTION",
    "CUT_NOTE",
    "ITEM_KEYS",
    "LABELS",
    "LAYOUT",
    "MAX_DRAFT_CHARS",
    "PLAIN",
    "SUMMARY_KEYS",
    "Compiled",
    "compile_draft",
    "find_draft",
]

#: Longest draft read, in characters.
MAX_DRAFT_CHARS = 262144

#: ``{`` positions a draft is searched from before giving up.
MAX_OPENINGS = 64

#: Longest caption, text or tag kept, in characters.
MAX_FIELD_CHARS = 8000

#: Most tags or palette entries kept on one box or the summary.
MAX_LIST_ITEMS = 32

#: What a box stands for.
LABELS = ("object", "character", "text")

#: Keys a layout's summary carries into the prompt, in the order they are written.
SUMMARY_KEYS = ("meta", "canvas", "caption", "palette")

#: Keys one box carries into the prompt, in the order they are written.
ITEM_KEYS = ("label", "caption", "text", "tags", "palette", "bbox")

#: Values ``summary.canvas`` takes.
CANVASES = ("single", "multi_panel", "collage")

#: What a draft turned out to be.
CAPTION = "caption"
LAYOUT = "layout"
PLAIN = "plain text"

#: The note a draft cut off before it closed carries.
CUT_NOTE = (
    "the draft was cut off before it closed, so only the part inside it that closed was read. "
    "Raise max_tokens on the model node, or turn thinking off"
)

#: A fenced code block, its body captured.
_FENCE = re.compile(r"```[a-zA-Z0-9_-]*[ \t]*\r?\n(.*?)```", re.S)


class Compiled(NamedTuple):
    """A compiled draft.

    Attributes:
        kind: :data:`CAPTION`, :data:`LAYOUT` or :data:`PLAIN`.
        prompt: What MOSS-Image reads: the caption, or the layout as one line of JSON.
        caption: The caption, or the layout's summary caption.
        items: The boxes as compiled, empty for a caption.
        frame: ``(width, height)`` the layout was drafted at, ``None`` for a caption.
        notes: What was dropped or changed on the way, one line each.
        complete: False when the draft was cut off before it closed and only part of it was read.
    """

    kind: str
    prompt: str
    caption: str
    items: tuple
    frame: tuple | None
    notes: tuple
    complete: bool = True


def find_draft(text: str):
    """The JSON object in a model's answer, fenced, bare or inside other words.

    Args:
        text: The answer.

    Returns:
        The first object holding ``caption`` or ``items``, or ``None`` when there is none.
    """
    return _search(text)[0]


def _search(text: str) -> tuple:
    """``(draft, cut)``: the first draft found, and whether it lay inside an object left unclosed."""
    source = str(text or "")[:MAX_DRAFT_CHARS]
    candidates = [match.group(1) for match in _FENCE.finditer(source)] + [source]
    decoder = json.JSONDecoder()
    for candidate in candidates:
        stripped = candidate.strip()
        try:
            value = json.loads(stripped)
        except (ValueError, RecursionError):
            value = None
        if _is_draft(value):
            return value, False
        start = 0
        broke_at = None
        for _attempt in range(MAX_OPENINGS):
            at = stripped.find("{", start)
            if at < 0:
                break
            try:
                value, end = decoder.raw_decode(stripped, at)
            except (ValueError, RecursionError) as error:
                value, end = None, at
                if broke_at is None:
                    broke_at = getattr(error, "pos", None)
            if _is_draft(value):
                return value, broke_at is not None and broke_at >= end
            start = at + 1
    return None, False


def compile_draft(text: str, size, mode: str = "fit") -> Compiled:
    """Compile a draft into the prompt MOSS-Image reads.

    Args:
        text: The draft, or a model's answer holding it. Text with no draft is taken as the caption.
        size: ``(width, height)`` of the image being generated, which a layout is placed onto.
        mode: How a layout drafted at another size is placed, a member of ``box_layout.RESIZE_MODES``.

    Returns:
        The compiled draft.

    Raises:
        ValueError: The text is empty, or the draft breaks the layout format, naming the field.
    """
    source = str(text or "")
    if len(source) > MAX_DRAFT_CHARS:
        raise ValueError(
            f"the draft is {len(source)} characters, past the {MAX_DRAFT_CHARS} read. Shorten it"
        )
    if not source.strip():
        raise ValueError("the draft is empty. Wire in the answer of the model that wrote it")
    source = _after_thinking(source)
    if not source.strip():
        raise ValueError(
            "the draft holds only the model's thinking, with no answer after it. Raise max_tokens "
            "on the model node, or turn thinking off"
        )
    draft, cut = _search(source)
    if draft is None:
        caption = source.strip()
        return Compiled(PLAIN, caption, caption, (), None, ("no JSON draft was found, so the whole text is the caption",))
    lead = (CUT_NOTE,) if cut else ()
    if "items" not in draft:
        caption = _words(draft.get("caption"), "caption")
        notes = lead + tuple(f"`{key}` is not part of a caption draft and was left out" for key in draft if key != "caption")
        return Compiled(CAPTION, caption, caption, (), None, notes, not cut)
    done = _layout(draft, size, mode)
    return done._replace(notes=lead + done.notes, complete=not cut)


def _after_thinking(text: str) -> str:
    """The text after the last ``</think>``, cut at a ``<think>`` left open."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    opened = text.find("<think>")
    return text[:opened] if opened >= 0 else text


def _is_draft(value) -> bool:
    """Whether a decoded value is a caption or layout draft."""
    return isinstance(value, dict) and ("items" in value or isinstance(value.get("caption"), str))


def _layout(draft: dict, size, mode: str) -> Compiled:
    """Compile a layout draft."""
    notes: list[str] = []
    space = draft.get("bbox_coord_space", "abs")
    order = draft.get("bbox_format", "xyxy")
    if space != "abs" or order != "xyxy":
        raise ValueError(
            f"the layout's boxes are `{str(space)[:20]}` `{str(order)[:20]}`; a draft holds them as "
            "`abs` `xyxy`, pixels from the top left. Ask the model for the draft again"
        )
    frame = (_side(draft.get("width"), "width"), _side(draft.get("height"), "height"))
    width, height = int(size[0]), int(size[1])

    summary_in = draft.get("summary")
    if not isinstance(summary_in, dict):
        caption = draft.get("caption")
        if not isinstance(caption, str):
            raise ValueError("the layout has no `summary` with a `caption`. Ask the model for the draft again")
        summary_in = {"caption": caption}
        notes.append("the caption was read from the top of the draft, not from `summary`")
    summary = {}
    for key in SUMMARY_KEYS:
        if key not in summary_in:
            continue
        if key == "caption":
            summary[key] = _words(summary_in[key], "summary.caption")
        elif key == "canvas":
            if summary_in[key] in CANVASES:
                summary[key] = summary_in[key]
            else:
                notes.append(f"summary.canvas `{str(summary_in[key])[:20]}` is not one of {', '.join(CANVASES)} and was left out")
        else:
            listed = _strings(summary_in[key])
            if listed:
                summary[key] = listed
    if "caption" not in summary:
        raise ValueError("the layout's `summary` has no `caption`. Ask the model for the draft again")
    dropped = sorted(set(summary_in) - set(SUMMARY_KEYS))
    if dropped:
        notes.append(f"summary fields left out: {', '.join(dropped)[:120]}")

    raw_items = draft.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("the layout's `items` is not a list. Ask the model for the draft again")
    if len(raw_items) > box_layout.MAX_BOXES:
        raise ValueError(
            f"the layout holds {len(raw_items)} boxes, past the {box_layout.MAX_BOXES} compiled. "
            "Ask for fewer"
        )
    boxes = []
    for index, raw in enumerate(raw_items, 1):
        box = _box(raw, index, frame, notes)
        if box is not None:
            boxes.append(box)
    placed = box_layout.refit(boxes, frame, (width, height), mode, 1)
    if len(placed) < len(boxes):
        notes.append(f"{len(boxes) - len(placed)} box(es) fell under a pixel on the image and were left out")
    if frame != (width, height):
        notes.append(f"drafted at {frame[0]}x{frame[1]}, placed on {width}x{height} by {mode}")

    half_w, half_h = width // 2, height // 2
    items = []
    for box in placed:
        item = {}
        for key in ITEM_KEYS:
            if key == "bbox":
                item[key] = [
                    box["y"] - half_h,
                    box["x"] - half_w,
                    box["y"] + box["height"] - half_h,
                    box["x"] + box["width"] - half_w,
                ]
            elif key in box["fields"]:
                item[key] = box["fields"][key]
        items.append(item)
    if not items:
        notes.append("no box was left, so the summary caption is the prompt")
        return Compiled(LAYOUT, summary["caption"], summary["caption"], (), frame, tuple(notes))
    compiled = {"summary": summary, "items": items, "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
    prompt = json.dumps(compiled, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return Compiled(LAYOUT, prompt, summary["caption"], tuple(items), frame, tuple(notes))


def _box(raw, index: int, frame, notes: list) -> dict | None:
    """One draft item as a layout box carrying its prompt fields, or ``None`` when it has no area."""
    where = f"items[{index - 1}]"
    if not isinstance(raw, dict):
        raise ValueError(f"{where} is not an object. Ask the model for the draft again")
    label = raw.get("label", "object")
    if label not in LABELS:
        notes.append(f"{where}.label `{str(label)[:20]}` is not one of {', '.join(LABELS)}, so it is an object")
        label = "object"
    fields = {"label": label, "caption": _words(raw.get("caption", ""), f"{where}.caption", empty=True)}
    if label == "text" or "text" in raw:
        written = raw.get("text")
        if isinstance(written, str) and written.strip():
            fields["text"] = written[:MAX_FIELD_CHARS]
        elif label == "text":
            raise ValueError(f"{where} is a text box with no `text`. Ask the model for the draft again")
    for key in ("tags", "palette"):
        listed = _strings(raw.get(key))
        if listed:
            fields[key] = listed
    bbox = raw.get("bbox")
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        raise ValueError(f"{where}.bbox is not [x_min, y_min, x_max, y_max]. Ask the model for the draft again")
    edges = []
    for value in bbox:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{where}.bbox holds a value that is not a number. Ask the model for the draft again")
        edges.append(float(value))
    x0, y0, x1, y1 = edges
    fw, fh = frame
    clamped = (max(0.0, min(fw, x0)), max(0.0, min(fh, y0)), max(0.0, min(fw, x1)), max(0.0, min(fh, y1)))
    if clamped != (x0, y0, x1, y1):
        notes.append(f"{where}.bbox reached past the {fw}x{fh} canvas and was held inside it")
    x0, y0, x1, y1 = clamped
    left, top = math.floor(round(x0, 9)), math.floor(round(y0, 9))
    right, bottom = math.ceil(round(x1, 9)), math.ceil(round(y1, 9))
    if right - left < 1 or bottom - top < 1:
        notes.append(f"{where}.bbox has no area and was left out")
        return None
    return {"x": left, "y": top, "width": right - left, "height": bottom - top, "fields": fields}


def _side(value, name: str) -> int:
    """One side of the drafted canvas, in pixels."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"the layout's `{name}` is not a number of pixels. Ask the model for the draft again")
    side = int(round(value))
    if not 1 <= side <= box_layout.MAX_FRAME:
        raise ValueError(f"the layout's `{name}` is {side}; it takes 1 to {box_layout.MAX_FRAME} pixels")
    return side


def _words(value, where: str, empty: bool = False) -> str:
    """A caption, held to :data:`MAX_FIELD_CHARS`."""
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"`{where}` is not text. Ask the model for the draft again")
    return value.strip()[:MAX_FIELD_CHARS]


def _strings(value) -> list:
    """The non-empty strings of a list, at most :data:`MAX_LIST_ITEMS`."""
    if not isinstance(value, list):
        return []
    kept = [item.strip()[:MAX_FIELD_CHARS] for item in value if isinstance(item, str) and item.strip()]
    return kept[:MAX_LIST_ITEMS]

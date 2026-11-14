"""Find the things in a picture with SAM 3, and turn the ones picked for editing into edit jobs.

Boxes are ``(left, top, right, bottom)`` as shares of the picture. Masks are ``(height, width)``
floats in ``[0, 1]``.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from .. import log

__all__ = [
    "ACTIONS",
    "REFERENCES",
    "VOCABULARY",
    "EditJob",
    "EditJobs",
    "Segment",
    "build_jobs",
    "compose",
    "find",
    "label_layers",
    "names_from",
    "read_edits",
]

logger = log.get_logger("image.edit_regions")

#: The things looked for when no list is given.
VOCABULARY = (
    "person", "man", "woman", "child", "face", "hair", "hand", "beard",
    "coat", "jacket", "raincoat", "shirt", "t-shirt", "sweater", "dress", "skirt", "trousers",
    "jeans", "shorts", "suit", "scarf", "hat", "cap", "beret", "helmet", "glasses", "sunglasses",
    "shoe", "boot", "bag", "handbag", "backpack", "shopping bag", "umbrella", "tie", "glove",
    "watch", "necklace", "earring",
    "dog", "cat", "bird", "pigeon", "horse",
    "car", "bus", "truck", "van", "bicycle", "motorcycle", "scooter", "boat", "tram", "wheel",
    "licence plate", "headlight",
    "road", "pavement", "cobblestones", "kerb", "street lamp", "traffic light", "sign",
    "shop sign", "bench", "fence", "railing", "pole", "bin", "bollard",
    "building", "wall", "window", "door", "balcony", "awning", "roof", "column", "arch",
    "stairs", "shutter", "shop window", "chalkboard", "poster", "graffiti", "flag",
    "table", "chair", "sofa", "bed", "desk", "shelf", "cabinet", "lamp", "chandelier", "mirror",
    "curtain", "rug", "cushion", "clock", "painting", "picture frame", "television", "laptop",
    "phone", "book", "newspaper", "menu",
    "cup", "mug", "saucer", "glass", "wine glass", "bottle", "plate", "bowl", "teapot", "vase",
    "candle", "food", "bread", "cake", "fruit", "napkin", "sugar bowl",
    "tree", "bush", "plant", "potted plant", "flower", "flower box", "grass", "leaves", "sky",
    "cloud", "mountain", "water", "sea", "river", "beach", "sand", "snow",
    "shadow", "reflection", "light", "text", "logo",
)

#: Side SAM 3 sees a picture at.
DETECT_SIDE = 1008

#: Names detected in one pass of the detector.
CHUNK = 8

#: Mask overlap at which two finds are one thing under two names.
SAME_THING = 0.7

#: Side finds are compared at, in pixels.
COMPARE_SIDE = 96

#: Least share of the picture a find covers.
LEAST_AREA = 0.0002

#: Most finds kept for one picture; a label layer numbers them from 1 to this.
MAX_SEGMENTS = 255

#: Longest side of the label layers the panel reads.
MAP_SIDE = 1024

#: Share of a find that another find already on a layer may cover.
LAYER_OVERLAP = 0.02

#: What an edit does, and whether its edit is pasted through its box rather than its shape.
ACTIONS = {"edit": False, "recolour": False, "remove": True, "replace": True}

#: Edit model -> the words its prompts name the picture by.
REFERENCES = {"MiniMax H3 Image": "<Picture 1>", "other": "the picture"}

#: Passes of SAM 3's decoder a picked find is refined with.
REFINE_PASSES = 2

#: Share of a find's box added around it when it is refined.
REFINE_MARGIN = 0.1

#: Share of a job's box a shadow must touch to join it.
SHADOW_REACH = 0.25

#: Pictures whose finds are held between runs.
MAX_HELD = 4

#: Bounds on the edits text: characters, jobs, picks and points per job, prompt characters.
MAX_EDITS_CHARS = 256 * 1024
MAX_JOBS = 64
MAX_PICKS = 32
MAX_POINTS = 16
MAX_PROMPT_CHARS = 4000

#: Colour words by hue in degrees, each up to its upper bound.
HUES = ((12, "red"), (35, "orange"), (68, "yellow"), (165, "green"), (195, "teal"), (255, "blue"),
        (285, "purple"), (330, "pink"), (360, "red"))

_held: OrderedDict = OrderedDict()
_lock = threading.Lock()


@dataclass
class Segment:
    """One thing found in a picture.

    Attributes:
        label: The name it was found under with the highest score.
        names: Every name it was found under, best first.
        score: The detector's confidence, 0 to 1.
        box: Its box as shares of the picture.
        coarse: Its mask logits at the detector's size.
        area: Share of the picture it covers.
        colour: A word for its main colour.
    """

    label: str
    names: list[str]
    score: float
    box: tuple[float, float, float, float]
    coarse: torch.Tensor
    area: float
    colour: str = ""


@dataclass
class EditJob:
    """One change to make to a picture.

    Attributes:
        mask: Where it applies, ``(height, width)``.
        label: What it changes, in words.
        prompt: The instruction for the edit model.
        whole: Its edit is pasted through its box rather than its shape.
    """

    mask: torch.Tensor
    label: str
    prompt: str
    whole: bool = False


@dataclass
class EditJobs:
    """The changes to make to one picture.

    Attributes:
        width: The picture's width.
        height: The picture's height.
        reference: The words a prompt names the picture by.
        jobs: One per change.
    """

    width: int
    height: int
    reference: str
    jobs: list[EditJob] = field(default_factory=list)


def names_from(text: str) -> tuple[str, ...]:
    """The names to look for: :data:`VOCABULARY` and those listed in ``text``, one per line or comma."""
    extra = [part.strip().lower() for line in str(text or "").splitlines() for part in line.split(",")]
    seen, names = set(), []
    for name in list(VOCABULARY) + extra:
        name = " ".join(name.replace(":", " ").replace("(", " ").replace(")", " ").split())[:60]
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return tuple(names)


def _sam(model, node: str):
    """SAM 3's network inside a MODEL."""
    inner = getattr(getattr(model, "model", None), "diffusion_model", None)
    if inner is None or not hasattr(inner, "detector") or not hasattr(inner, "forward_segment"):
        raise ValueError(
            f"{node} needs SAM 3 on `model`. Load sam3.1_multiplex_fp16.safetensors with Load Checkpoint "
            "and wire its MODEL and CLIP."
        )
    return inner


def _encoded(clip, names: tuple[str, ...], node: str) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Each name as ``(embedding, mask)`` from SAM 3's text encoder."""
    tokens = clip.tokenize(", ".join(names))
    cond = clip.encode_from_tokens_scheduled(tokens)
    extra = cond[0][1]
    multi = extra.get("sam3_multi_cond")
    if multi is None and len(names) == 1:
        multi = [{"cond": cond[0][0], "attention_mask": extra.get("attention_mask")}]
    if multi is None or len(multi) != len(names):
        raise ValueError(
            f"{node} needs SAM 3's own text encoder on `clip`: wire the CLIP from the Load Checkpoint "
            "that loads SAM 3."
        )
    encoded = []
    for entry in multi:
        embedding = entry["cond"]
        mask = entry["attention_mask"]
        if mask is None:
            mask = torch.ones(embedding.shape[:2], dtype=torch.int64)
        encoded.append((embedding, mask))
    return encoded


def _digest(picture: torch.Tensor) -> str:
    """A key for a picture, from its size and a small copy of it."""
    small = F.interpolate(picture[None, ..., :3].movedim(-1, 1).float(), size=(64, 64), mode="area")
    data = (small.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha1(data + repr(tuple(picture.shape)).encode()).hexdigest()


def _frame(picture: torch.Tensor, device, dtype) -> torch.Tensor:
    """A picture as SAM 3 reads it: ``(1, 3, side, side)``."""
    import comfy.utils

    moved = picture[None, ..., :3].movedim(-1, 1)
    return comfy.utils.common_upscale(moved, DETECT_SIDE, DETECT_SIDE, "bilinear", crop="disabled").to(device=device, dtype=dtype)


def _colour_word(rgb: torch.Tensor) -> str:
    """A word for an ``(r, g, b)`` colour in ``[0, 1]``."""
    red, green, blue = (float(value) for value in rgb)
    high, low = max(red, green, blue), min(red, green, blue)
    saturation = (high - low) / high if high > 0 else 0.0
    if high < 0.16:
        return "black"
    if saturation < 0.14:
        return "white" if high > 0.82 else "grey"
    if high == red:
        hue = (60 * ((green - blue) / (high - low))) % 360
    elif high == green:
        hue = 60 * ((blue - red) / (high - low)) + 120
    else:
        hue = 60 * ((red - green) / (high - low)) + 240
    word = next(name for bound, name in HUES if hue < bound)
    if word == "pink" and (saturation > 0.6 or high < 0.6):
        word = "red"
    if word == "red":
        if saturation < 0.45 and high > 0.7:
            return "pink"
        return "dark red" if high < 0.3 else "red"
    if word in ("orange", "yellow") and high < 0.55 and saturation > 0.3:
        return "brown"
    if word in ("orange", "yellow") and saturation < 0.35:
        return "beige"
    if word == "blue" and high < 0.4:
        return "navy"
    return word


def _merged(candidates: list, picture: torch.Tensor) -> list[Segment]:
    """Finds sorted by score, each thing kept once under every name it was found under."""
    candidates.sort(key=lambda found: -found[1])
    small = F.interpolate(picture[None, ..., :3].movedim(-1, 1).float(), size=(COMPARE_SIDE, COMPARE_SIDE), mode="area")[0]
    kept: list[Segment] = []
    flat: list[torch.Tensor] = []
    for name, score, box, coarse in candidates:
        binary = F.interpolate(coarse[None, None].float(), size=(COMPARE_SIDE, COMPARE_SIDE), mode="bilinear", align_corners=False)[0, 0] > 0
        area = float(binary.float().mean())
        if area < LEAST_AREA:
            continue
        row = binary.flatten().float()
        if flat:
            stack = torch.stack(flat)
            shared = stack @ row
            union = stack.sum(dim=1) + row.sum() - shared
            overlap = shared / union.clamp(min=1.0)
            best = int(torch.argmax(overlap))
            if float(overlap[best]) >= SAME_THING:
                if name not in kept[best].names:
                    kept[best].names.append(name)
                continue
        pixels = small[:, binary]
        colour = _colour_word(pixels.median(dim=1).values) if pixels.shape[1] else ""
        kept.append(Segment(name, [name], float(score), box, coarse, area, colour))
        flat.append(row)
        if len(kept) >= MAX_SEGMENTS:
            break
    return kept


def find(model, clip, picture: torch.Tensor, names: tuple[str, ...], threshold: float, node: str) -> list[Segment]:
    """Every thing in a picture that SAM 3 finds under one of ``names``.

    Args:
        model: SAM 3 as a MODEL.
        clip: SAM 3's text encoder as a CLIP.
        picture: ``(height, width, channels)``.
        names: What to look for.
        threshold: Least confidence a find keeps, 0 to 1.
        node: The node's name, for messages.

    Returns:
        The finds, best first, at most :data:`MAX_SEGMENTS`.
    """
    import comfy.model_management as management

    sam = _sam(model, node)
    key = (_digest(picture), id(model.model), names, round(float(threshold), 4))
    with _lock:
        if key in _held:
            _held.move_to_end(key)
            return _held[key]
    management.load_model_gpu(model)
    device = management.get_torch_device()
    dtype = model.model.get_dtype()
    detector = sam.detector
    backbone = detector.backbone["vision_backbone"]
    resizer = detector.backbone["language_backbone"]["resizer"]
    encoded = _encoded(clip, names, node)
    candidates = []
    with torch.no_grad():
        trunk = backbone.trunk(_frame(picture, device, dtype))
        for start in range(0, len(names), CHUNK):
            management.throw_exception_if_processing_interrupted()
            part = encoded[start:start + CHUNK]
            length = max(embedding.shape[1] for embedding, _ in part)
            embeddings = torch.cat([F.pad(embedding, (0, 0, 0, length - embedding.shape[1])) for embedding, _ in part])
            masks = torch.cat([F.pad(mask, (0, length - mask.shape[1])) for _, mask in part])
            count = embeddings.shape[0]
            out = detector.forward_from_trunk(
                trunk.expand(count, *trunk.shape[1:]),
                resizer(embeddings.to(device=device, dtype=dtype)),
                masks.to(device),
            )
            chance = out["scores"].float().sigmoid()
            for row in range(count):
                for query in torch.nonzero(chance[row] > threshold).flatten().tolist():
                    box = tuple(min(1.0, max(0.0, float(value))) for value in out["boxes"][row, query])
                    if box[2] <= box[0] or box[3] <= box[1]:
                        continue
                    coarse = out["masks"][row, query].detach().to("cpu", torch.float16)
                    candidates.append((names[start + row], float(chance[row, query]), box, coarse))
    found = _merged(candidates, picture)
    with _lock:
        _held[key] = found
        while len(_held) > MAX_HELD:
            _held.popitem(last=False)
    logger.info("%s found %d thing(s) from %d name(s)", node, len(found), len(names))
    return found


def label_layers(segments: list[Segment], width: int, height: int) -> tuple[list[torch.Tensor], tuple[int, int]]:
    """The finds drawn as layers of ids, each find on the first layer it hardly overlaps.

    Args:
        segments: From :func:`find`.
        width: The picture's width.
        height: The picture's height.

    Returns:
        ``(layers, (width, height))``: ``uint8`` planes where a pixel holds a find's place in
        ``segments`` plus one, or 0, and the size they were drawn at.
    """
    scale = min(1.0, MAP_SIDE / max(width, height))
    size = (max(1, round(height * scale)), max(1, round(width * scale)))
    layers: list[torch.Tensor] = []
    for index in sorted(range(len(segments)), key=lambda at: -segments[at].area):
        marked = F.interpolate(segments[index].coarse[None, None].float(), size=size, mode="bilinear", align_corners=False)[0, 0] > 0
        total = int(marked.sum())
        if total == 0:
            continue
        for layer in layers:
            if int(((layer > 0) & marked).sum()) <= LAYER_OVERLAP * total:
                layer[marked] = index + 1
                break
        else:
            layer = torch.zeros(size, dtype=torch.uint8)
            layer[marked] = index + 1
            layers.append(layer)
    return layers, (size[1], size[0])


def _refined(sam, picture: torch.Tensor, segment: Segment, device, dtype) -> torch.Tensor:
    """A find's mask at the picture's size, refined by SAM 3's decoder inside its box."""
    import comfy.utils

    height, width = picture.shape[:2]
    left, top, right, bottom = segment.box
    pad_x, pad_y = (right - left) * REFINE_MARGIN, (bottom - top) * REFINE_MARGIN
    x0, y0 = max(0, int((left - pad_x) * width)), max(0, int((top - pad_y) * height))
    x1, y1 = min(width, int(math.ceil((right + pad_x) * width))), min(height, int(math.ceil((bottom + pad_y) * height)))
    coarse = segment.coarse.float()
    whole = F.interpolate(coarse[None, None], size=(height, width), mode="bilinear", align_corners=False)[0, 0] > 0
    if x1 - x0 < 4 or y1 - y0 < 4:
        return whole.float()
    crop = picture[y0:y1, x0:x1, :3][None].movedim(-1, 1)
    frame = comfy.utils.common_upscale(crop, DETECT_SIDE, DETECT_SIDE, "bilinear", crop="disabled").to(device=device, dtype=dtype)
    mask_height, mask_width = coarse.shape
    mx0, my0 = int(x0 / width * mask_width), int(y0 / height * mask_height)
    mx1, my1 = max(mx0 + 1, int(x1 / width * mask_width)), max(my0 + 1, int(y1 / height * mask_height))
    logits = coarse[my0:my1, mx0:mx1][None, None].to(device=device, dtype=dtype)
    with torch.no_grad():
        for _ in range(REFINE_PASSES):
            logits = sam.forward_segment(frame, mask_inputs=F.interpolate(logits, size=(DETECT_SIDE, DETECT_SIDE), mode="bilinear", align_corners=False))
    piece = F.interpolate(logits.float(), size=(y1 - y0, x1 - x0), mode="bilinear", align_corners=False)[0, 0].cpu() > 0
    mask = whole.clone()
    mask[y0:y1, x0:x1] |= piece
    return mask.float()


def _pointed(sam, picture: torch.Tensor, points: list, device, dtype) -> torch.Tensor:
    """The thing under one or more points, as a mask at the picture's size."""
    height, width = picture.shape[:2]
    frame = _frame(picture, device, dtype)
    coords = torch.tensor([[[x * DETECT_SIDE, y * DETECT_SIDE] for x, y in points]], dtype=dtype, device=device)
    labels = torch.ones((1, len(points)), dtype=torch.int32, device=device)
    with torch.no_grad():
        logits = sam.forward_segment(frame, point_inputs={"point_coords": coords, "point_labels": labels})
        for _ in range(REFINE_PASSES - 1):
            logits = sam.forward_segment(frame, mask_inputs=logits)
    return (F.interpolate(logits.float(), size=(height, width), mode="bilinear", align_corners=False)[0, 0].cpu() > 0).float()


def _share(value) -> float | None:
    """A number held to ``[0, 1]``, or None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return min(1.0, max(0.0, number)) if math.isfinite(number) else None


def read_edits(text: str, node: str) -> list[dict]:
    """The jobs an edits text names, each with its picks, points, action, prompt, paste and shadow.

    Args:
        text: The JSON the panel writes: ``{"jobs": [{"picks": [{"label", "box"}], "points": [[x, y]],
            "action", "prompt", "paste", "shadow"}]}``, positions as shares of the picture.
        node: The node's name, for messages.

    Returns:
        The jobs, each read to its bounds; an empty text gives none.

    Raises:
        ValueError: The text is not that JSON.
    """
    if not str(text or "").strip():
        return []
    if len(text) > MAX_EDITS_CHARS:
        raise ValueError(f"{node}'s edits are over {MAX_EDITS_CHARS // 1024} KB. Remove some edits in its panel.")
    try:
        data = json.loads(text)
    except ValueError as error:
        raise ValueError(f"{node} could not read its edits ({error.msg}). Clear them in its panel and pick again.") from None
    rows = data.get("jobs") if isinstance(data, dict) else None
    if not isinstance(rows, list):
        raise ValueError(f"{node}'s edits hold no job list. Clear them in its panel and pick again.")
    jobs = []
    for row in rows[:MAX_JOBS]:
        if not isinstance(row, dict):
            continue
        picks = []
        for pick in (row.get("picks") or [])[:MAX_PICKS] if isinstance(row.get("picks"), list) else []:
            box = pick.get("box") if isinstance(pick, dict) else None
            if not isinstance(box, list) or len(box) != 4:
                continue
            shares = [_share(value) for value in box]
            if None in shares or shares[2] <= shares[0] or shares[3] <= shares[1]:
                continue
            picks.append({"label": str(pick.get("label") or "")[:60], "box": tuple(shares)})
        points = []
        for point in (row.get("points") or [])[:MAX_POINTS] if isinstance(row.get("points"), list) else []:
            if isinstance(point, list) and len(point) == 2:
                shares = [_share(value) for value in point]
                if None not in shares:
                    points.append(tuple(shares))
        action = str(row.get("action") or "edit")
        action = action if action in ACTIONS else "edit"
        paste = row.get("paste")
        jobs.append({
            "picks": picks,
            "points": points,
            "action": action,
            "prompt": str(row.get("prompt") or "")[:MAX_PROMPT_CHARS],
            "whole": ACTIONS[action] if paste not in ("shape", "box") else paste == "box",
            "shadow": bool(row.get("shadow", False)),
            "label": str(row.get("label") or "")[:120],
        })
    return jobs


def _box_overlap(first, second) -> float:
    """Shared area over joined area of two boxes."""
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    shared = width * height
    joined = (first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1]) - shared
    return shared / joined if joined > 0 else 0.0


def _matched(pick: dict, segments: list[Segment]) -> int | None:
    """The find a pick names: its name with the closest box, else any find with a box close enough."""
    best, best_overlap = None, 0.0
    for index, segment in enumerate(segments):
        overlap = _box_overlap(pick["box"], segment.box)
        named = not pick["label"] or pick["label"] in segment.names
        if (named and overlap >= 0.5 or overlap >= 0.75) and overlap > best_overlap:
            best, best_overlap = index, overlap
    return best


def build_jobs(model, picture: torch.Tensor, segments: list[Segment], edits: list[dict], reference: str, node: str):
    """The jobs an edits list names, each with its mask at the picture's size.

    Args:
        model: SAM 3 as a MODEL.
        picture: ``(height, width, channels)``.
        segments: From :func:`find` on the same picture.
        edits: From :func:`read_edits`.
        reference: The words a prompt names the picture by.
        node: The node's name, for messages.

    Returns:
        ``(jobs, notes)``: an :class:`EditJobs`, and one line for each job passed over.
    """
    import comfy.model_management as management

    sam = _sam(model, node)
    height, width = picture.shape[:2]
    jobs = EditJobs(width, height, reference)
    notes = []
    if not edits:
        return jobs, notes
    management.load_model_gpu(model)
    device = management.get_torch_device()
    dtype = model.model.get_dtype()
    shadows = [index for index, segment in enumerate(segments) if "shadow" in segment.names]
    for number, edit in enumerate(edits, 1):
        management.throw_exception_if_processing_interrupted()
        prompt = edit["prompt"].strip()
        if not prompt:
            notes.append(f"edit {number} has no prompt")
            continue
        chosen = []
        for pick in edit["picks"]:
            found = _matched(pick, segments)
            if found is None:
                notes.append(f"edit {number}: `{pick['label'] or 'a pick'}` was not found again")
            elif found not in chosen:
                chosen.append(found)
        if edit["shadow"] and chosen:
            for index in shadows:
                if index in chosen:
                    continue
                for own in chosen:
                    left, top, right, bottom = segments[own].box
                    reach = SHADOW_REACH * max(right - left, bottom - top)
                    if _box_overlap(segments[index].box, (left - reach, top - reach, right + reach, bottom + reach)) > 0:
                        chosen.append(index)
                        break
        mask = torch.zeros((height, width))
        for index in chosen:
            mask = torch.maximum(mask, _refined(sam, picture, segments[index], device, dtype))
        if edit["points"]:
            mask = torch.maximum(mask, _pointed(sam, picture, edit["points"], device, dtype))
        if float(mask.max()) <= 0:
            notes.append(f"edit {number} marks nothing")
            continue
        label = edit["label"] or ", ".join(segments[index].label for index in chosen) or "area"
        jobs.jobs.append(EditJob(mask, label, prompt, edit["whole"]))
    return jobs, notes


def compose(instructions: list[str], reference: str) -> str:
    """One prompt holding every instruction for a crop, closed by the line that keeps the rest."""
    tail = f"Keep everything else in {reference} exactly the same."
    texts = [text.strip() for text in instructions if text.strip()]
    if not texts:
        return ""
    if len(texts) == 1:
        return f"{texts[0]} {tail}"
    listed = " ".join(f"{number}. {text}" for number, text in enumerate(texts, 1))
    return f"Make these changes to {reference}: {listed} {tail}"

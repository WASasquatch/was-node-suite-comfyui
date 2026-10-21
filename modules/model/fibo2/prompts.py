"""Fibo 2.0's prompt forms, its trained image sizes and its sampling schedule.

A caption is Bria's VGL JSON. An edit reads one JSON object: a vision marker per image, the
result's caption and the instruction.
"""

from __future__ import annotations

import copy
import json
import math

#: The null caption guidance runs against: every key of a caption, every value empty.
NULL_CAPTION = (
    '{"short_description":"","objects":[],"background_setting":"","lighting":{},'
    '"aesthetics":{},"photographic_characteristics":{},"style_medium":"",'
    '"context":"","artistic_style":""}'
)

#: Bria's negative caption for text to image.
GENERATION_NEGATIVE = (
    '{"short_description":"","objects":[{"description":"the main subject","texture":"smooth, featureless, '
    'plastic-like, airbrushed, no fine surface detail","appearance_details":"soft blurred edges, simplified '
    'shapes, no fine detail","skin_tone_and_texture":"waxy airbrushed skin, poreless, plastic smooth"}],'
    '"background_setting":"a plain seamless studio backdrop, perfectly smooth gradient, blurred and featureless, '
    'no visible texture","lighting":{},"aesthetics":{"artifacts":["oversaturated colours","neon colour cast",'
    '"overexposed highlights","lens flare","slight blur","motion blur","haze"]},"photographic_characteristics":'
    '{"depth_of_field":"shallow, heavy bokeh obscuring most of the frame","focus":"soft, slightly out of focus, '
    'lacking fine texture"},"style_medium":"","context":"","artistic_style":""}'
)

#: Where an image stands in a Qwen3-VL prompt.
VISION_MARKER = "<|vision_start|><|image_pad|><|vision_end|>"

#: The chat turn every prompt is read in.
CHAT = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

#: The most images one edit reads.
MOST_IMAGES = 5

#: The longest side Qwen3-VL sees an image to edit at.
VISION_SIDE = 768

#: A generation shows the text images from this noise level down.
TEXT_IMAGE_SIGMA = 0.9

#: The image area sizes are chosen around, and the multiple every side is.
AREA = 1024 * 1024
MULTIPLE = 16

#: Caption keys training removed.
SCORES = ("aesthetic_score", "preference_score")


def _empty(value) -> bool:
    if isinstance(value, float):
        return math.isnan(value)
    return value is None or (isinstance(value, (str, dict, list)) and not value)


def _cleaned(value):
    """The value with every empty value removed, at any depth."""
    if isinstance(value, dict):
        kept = {key: _cleaned(item) for key, item in value.items()}
        return {key: item for key, item in kept.items() if not _empty(item)}
    if isinstance(value, list):
        return [item for item in map(_cleaned, value) if not _empty(item)]
    return value


def split(prompt: str) -> tuple[dict | str | None, str | None]:
    """A prompt's caption, in the form training saw, and its edit instruction.

    Args:
        prompt: A JSON caption, an edit object, or plain text.

    Returns:
        ``(caption, instruction)``. Plain text comes back as the caption with no instruction.
    """
    try:
        caption = json.loads(prompt)
    except json.JSONDecodeError:
        return prompt, None
    if not isinstance(caption, dict):
        return prompt, None
    instruction = caption.pop("edit_instruction", None)
    if "structured_caption" in caption:
        caption = caption.pop("structured_caption")
        if not isinstance(caption, dict):
            return caption or None, instruction
    keys = set(caption)
    caption = {
        key: {inner: item for inner, item in value.items() if inner not in SCORES} if isinstance(value, dict) else value
        for key, value in _cleaned(caption).items()
        if key not in SCORES
    }
    caption = {key: value for key, value in caption.items() if value != {}}
    if keys == {"short_description", "text_render"}:
        caption = {
            "short_description": caption.get("short_description") or "",
            "text_render": caption.get("text_render") or [],
        }
    return caption or None, instruction


def _compact(value) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def texts(caption) -> list[str]:
    """The texts of a caption's ``text_render`` entries that have one, in order."""
    entries = caption.get("text_render") or [] if isinstance(caption, dict) else []
    return [entry["text"] for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get("text"), str) and entry["text"]]


def _marked(caption: dict, marker: str) -> dict:
    """A copy of the caption with each text entry's ``image_tokens`` set to ``marker``."""
    caption = copy.deepcopy(caption)
    for entry in caption.get("text_render") or []:
        if isinstance(entry, dict) and isinstance(entry.get("text"), str) and entry["text"]:
            entry["image_tokens"] = marker
    return caption


def generation(prompt: str, text_images: bool) -> tuple[str, str | None]:
    """The text a generation reads, and the one its first steps read when text images are shown.

    Args:
        prompt: The caption or plain text.
        text_images: Whether the caption's text images are shown.

    Returns:
        ``(text, early)``: ``early`` is ``None`` when no text image is shown.
    """
    caption, _ = split(prompt)
    if isinstance(caption, str):
        return prompt, None
    caption = caption or {}
    if not (text_images and texts(caption)):
        return _compact(caption), None
    return _compact(_marked(caption, VISION_MARKER)), _compact(_marked(caption, ""))


def edit(prompt: str, count: int, text_images: bool) -> str:
    """The JSON an edit of ``count`` images reads.

    Args:
        prompt: The instruction alone, or a JSON caption of the result with ``edit_instruction``.
        count: Images to edit.
        text_images: Whether the caption's text images follow the images to edit.

    Raises:
        ValueError: A JSON prompt has no ``edit_instruction``, or a ``structured_caption`` beside
            other keys.
    """
    try:
        parsed = json.loads(prompt)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        if "edit_instruction" not in parsed:
            raise ValueError(
                'An edit prompt in JSON needs an "edit_instruction" key beside the caption of the result. '
                "Add one, or write the instruction alone as plain text."
            )
        if "structured_caption" in parsed and (
            not isinstance(parsed["structured_caption"], (dict, str)) or len(parsed) > 2
        ):
            raise ValueError(
                'An edit prompt with "structured_caption" holds the caption there and nothing else but '
                '"edit_instruction".'
            )
    caption, instruction = split(prompt)
    if instruction is None:
        caption, instruction = None, prompt
    if text_images and isinstance(caption, dict):
        caption = _marked(caption, VISION_MARKER)
    out = {"image": VISION_MARKER} if count == 1 else {f"image_{k}": VISION_MARKER for k in range(1, count + 1)}
    if caption:
        out["structured_caption"] = caption
    out["edit_instruction"] = instruction
    return _compact(out)


def trained_size(aspect: float, area: int = AREA) -> tuple[int, int]:
    """The ``(width, height)`` trained for an aspect ratio: least crop, then nearest ``area``, then widest.

    Args:
        aspect: Width over height.
        area: The area sizes lie within 10% of.

    Returns:
        A size whose sides are multiples of 16 and at least 64.
    """
    smallest = 4 * MULTIPLE
    best, best_key = None, None
    for width in range(smallest, 2 * area // smallest, MULTIPLE):
        tallest = 2 * area // width
        low = max(smallest, int(0.9 * area / width) // MULTIPLE * MULTIPLE)
        for height in range(low, tallest, MULTIPLE):
            if width * height > 1.1 * area:
                break
            if width * height < 0.9 * area:
                continue
            ratio = width / height
            key = (round(1 - min(ratio / aspect, aspect / ratio), 9), abs(width * height - area), -width)
            if best_key is None or key < best_key:
                best, best_key = (width, height), key
    return best


def edit_size(width: int, height: int, area: int = AREA) -> tuple[int, int]:
    """The ``(width, height)`` an image is edited at.

    Args:
        width: The image's width in pixels.
        height: The image's height in pixels.
        area: Images up to 10% past this keep their own size.

    Returns:
        The image's own size cut to multiples of 16, or :func:`trained_size` for a larger image.
    """
    if width * height <= 1.1 * area:
        smallest = 4 * MULTIPLE
        return max(smallest, width // MULTIPLE * MULTIPLE), max(smallest, height // MULTIPLE * MULTIPLE)
    return trained_size(width / height, area)


def sigmas(steps: int, tokens: int) -> list[float]:
    """The released schedule: ``linspace(1, 1 / steps)`` shifted for the image size, then 0.

    Args:
        steps: Denoising steps.
        tokens: Image tokens, ``(height / 16) * (width / 16)``.
    """
    import numpy as np

    slope = (1.15 - 0.5) / (4096 - 256)
    mu = tokens * slope + (0.5 - slope * 256)
    times = np.linspace(1.0, 1 / steps, steps).astype(np.float32)
    shifted = math.exp(mu) / (math.exp(mu) + (1 / times - 1) ** 1.0)
    return [float(value) for value in shifted.astype(np.float32)] + [0.0]

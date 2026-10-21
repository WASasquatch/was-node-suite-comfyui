"""Fibo 2.0 captions written by a Qwen-VL language model from a short prompt, images or both.

:data:`FIELDS` lists a caption's keys in training's order, which every reply is put back into.
"""

from __future__ import annotations

import json

from PIL import Image

#: What the writer is asked to do.
TASKS = ("auto", "generate", "edit", "inspire", "caption", "refine")

#: Most images an edit description reads.
MOST_IMAGES = 4

#: Pixels a picture is shrunk to before the language model sees it.
VISION_PIXELS = 262144

#: Sampling Bria's writer runs at.
TEMPERATURE = 0.2

#: A caption's keys in training's order, each with what it holds.
FIELDS = (
    ("short_description", "string: what the image shows, at most 200 words."),
    ("objects", "array of the prominent objects, each an object with the keys below."),
    ("background_setting", "string: the environment and background, with anything notable not in objects."),
    ("lighting", "object: conditions (as 'golden hour'), direction (as 'side-lit from left'), shadows."),
    ("aesthetics", "object: composition (as 'rule of thirds', 'close-up'), color_scheme, mood_atmosphere."),
    ("photographic_characteristics",
     "object: depth_of_field, focus, camera_angle (as 'eye-level'), lens_focal_length (as 'portrait lens 85mm')."),
    ("style_medium", "string: 'photograph', 'oil painting', '3D render', 'pencil sketch' and so on."),
    ("text_render", "array: every text in the image, exactly as it reads, one entry per line or label; [] when none."),
    ("context", "string: what kind of image this is, as 'an editorial photograph for a travel magazine'."),
    ("artistic_style", "string: three words at most, 'realistic' for a photograph."),
)

#: The keys of each object in ``objects``, in training's order.
OBJECT_FIELDS = (
    ("description", "string: the object, at most 100 words."),
    ("location", "string: as 'center', 'top-left', 'bottom-right foreground'."),
    ("relationship", "string: how it relates to the other objects."),
    ("relative_size", "string: as 'small', 'medium', 'large within frame'."),
    ("shape_and_color", "string: its shape and main colour."),
    ("texture", "string: as 'smooth', 'metallic', 'furry'."),
    ("appearance_details", "string: anything else visible."),
    ("number_of_objects", "integer: how many, for a group of alike objects."),
    ("pose", "string: for a person, the body position."),
    ("expression", "string: for a person, the facial expression."),
    ("clothing", "string: for a person, the attire."),
    ("action", "string: for a person, what they are doing."),
    ("gender", "string: for a person."),
    ("skin_tone_and_texture", "string: for a person."),
    ("orientation", "string: as 'facing left', 'upright', 'tilted 45 degrees'."),
)

#: The keys of each ``text_render`` entry, in training's order.
TEXT_FIELDS = (
    ("text", "string: the exact text."),
    ("location", "string: where it sits."),
    ("size", "string: as 'medium', 'large within frame'."),
    ("color", "string: its colour."),
    ("font", "string: as 'serif typeface', 'hand-lettered', 'minimalist'."),
    ("appearance_details", "string: anything else about it."),
)

#: Ordering of the nested objects.
NESTED = {
    "lighting": ("conditions", "direction", "shadows"),
    "aesthetics": ("composition", "color_scheme", "mood_atmosphere"),
    "photographic_characteristics": ("depth_of_field", "focus", "camera_angle", "lens_focal_length"),
}

_RULES = (
    "Answer with one JSON object and nothing else: no heading, no code fence, no comment.\n"
    "Describe people's left and right from their own point of view, not the viewer's.\n"
    "Leave out every key that does not apply, such as the person keys of an object that is not a person; "
    "never write n/a or none.\n"
    "Keep the fields consistent with each other and do not repeat one field in another."
)

#: Values a language model writes for a key that does not apply.
PLACEHOLDERS = {"n/a", "na", "none", "null", "not applicable", "unknown", "-", "nothing"}

#: Keys a reply may hold the edit instruction under, lowercased with spaces and hyphens as underscores.
INSTRUCTION_KEYS = ("edit_instruction", "editinstruction", "instruction", "edit")

_GENERATE = (
    "You are an art director turning a short idea into a complete description of one image for an image "
    "generator. Invent the concrete details the idea leaves open: setting, light, composition, colour and "
    "mood. Unless the idea asks for another medium, make it a photograph, style_medium 'photograph' and "
    "artistic_style 'realistic'. When a person is the subject, frame them so the face and upper body are "
    "prominent, a medium shot or close-up with a standard or portrait lens, unless the idea calls for a wide "
    "scene. Every text the image should show goes in text_render, written exactly; [] when there is none."
)

_INSPIRE = (
    "You are an art director using a picture as a visual reference for a new image. Read the picture's "
    "subject, pose, appearance, composition and style, take from it what the instruction asks to keep, and "
    "describe the new scene the instruction asks for. Describe the new image, never the reference itself. "
    "When the reference is a photograph and the instruction names no other medium, keep it a photograph."
)

_CAPTION = (
    "You are an art director describing a picture exactly as it is, for an image generator to reproduce. "
    "Describe only what is visible, and copy every text in it into text_render as it reads."
)

_REFINE = (
    "You are an art director revising a description of an image. Change only the fields the instruction "
    "affects, and every field that follows from them: a new setting changes the background, the light and "
    "the mood, a new text is a new text_render entry. Copy every other field from the description unchanged."
)

_EDIT = (
    "You are an art director describing the image an edit should produce. Keep everything the instruction "
    "does not change: the identity, pose and appearance of every subject, every object's place and size, the "
    "composition, the camera and the style. Change only what the edit needs, and what follows from it.\n"
    "Describe the finished image as it is, never the change: write 'his hands are empty', not 'the book was "
    "removed'. Outside edit_instruction, never write changed, added, removed, replaced, now, instead of or "
    "originally.\n"
    "When the picture is a control map, such as edges, depth, a pose skeleton, a colour layout or a low "
    "resolution tile, read it only for layout and describe the finished image the instruction asks for.\n"
    "Add text to text_render only when the instruction gives it or the picture already shows it.\n"
    "Add the key edit_instruction last: one imperative sentence starting with a verb such as Add, Change, "
    "Make, Move, Remove, Replace or Turn, covering every change, never mentioning a reference or original image."
)

_MASK = (
    "The grey area of the picture is the part to fill. Use the rest as context and describe the finished "
    "image with the grey area holding what the instruction asks for; never describe the grey itself."
)

_MULTI = (
    "The pictures are numbered image 1, image 2 and so on, in the order given, and the instruction may name "
    "them so. Read what each contributes and describe the one composed image the instruction asks for."
)


def field_guide(edit: bool = False) -> str:
    """The keys a caption holds, written for the language model."""
    lines = ["The JSON object holds these keys, in this order:"]
    lines += [f"- {key}: {about}" for key, about in FIELDS]
    lines.append("Each object in objects holds:")
    lines += [f"  - {key}: {about}" for key, about in OBJECT_FIELDS]
    lines.append("Each entry in text_render holds:")
    lines += [f"  - {key}: {about}" for key, about in TEXT_FIELDS]
    if edit:
        lines.append("- edit_instruction: string, the imperative sentence described above.")
    return "\n".join(lines)


def task_for(task: str, prompt: str, images: int, caption: str) -> str:
    """The task ``auto`` stands for: edit with images and a prompt, caption with images alone,
    refine with a caption alone, else generate."""
    if task != "auto":
        return task
    if images:
        return "edit" if prompt.strip() else "caption"
    return "refine" if caption.strip() else "generate"


def instructions(task: str, images: int, masked: bool, style: str = "") -> str:
    """The system prompt for one task."""
    parts = [{"generate": _GENERATE, "inspire": _INSPIRE, "caption": _CAPTION, "refine": _REFINE,
              "edit": _EDIT}[task]]
    if task == "edit" and masked:
        parts.append(_MASK)
    if task == "edit" and images > 1:
        parts.append(_MULTI)
    if style.strip() and task in ("generate", "inspire"):
        parts.append("Apply these style guidelines: the request decides what is in the image, the guidelines how "
                     f"it looks.\n{style.strip()}")
    parts.append(_RULES)
    parts.append(field_guide(edit=task == "edit"))
    return "\n\n".join(parts)


def request(task: str, prompt: str, caption: str) -> str:
    """The user turn for one task."""
    prompt = prompt.strip()
    if task == "generate":
        return f"Write the JSON description for this idea: {prompt}"
    if task == "caption":
        return "Write the JSON description of this picture."
    if task == "refine":
        return f"The description:\n{caption.strip()}\n\nThe instruction: {prompt}\n\nWrite the revised JSON."
    lead = f"The current picture's description:\n{caption.strip()}\n\n" if caption.strip() else ""
    if task == "inspire":
        return f"{lead}The instruction: {prompt}\n\nWrite the JSON description of the new image."
    return f"{lead}The edit: {prompt}\n\nWrite the JSON description of the edited image."


def shrunk(picture: Image.Image) -> Image.Image:
    """A picture no larger than :data:`VISION_PIXELS`, its shape kept."""
    picture = picture.convert("RGB")
    area = picture.width * picture.height
    if area <= VISION_PIXELS:
        return picture
    ratio = (VISION_PIXELS / area) ** 0.5
    return picture.resize((int(picture.width * ratio), int(picture.height * ratio)), Image.LANCZOS)


def greyed(picture: Image.Image, mask: Image.Image) -> Image.Image:
    """The picture with the mask's white area painted mid-grey."""
    grey = Image.new("RGB", picture.size, (128, 128, 128))
    return Image.composite(grey, picture.convert("RGB"), mask.convert("L"))


def _filled(value, key: str = "") -> bool:
    if key != "text" and isinstance(value, str) and value.strip().lower().rstrip(".") in PLACEHOLDERS:
        return False
    return not (value is None or value in ("", [], {}) or (isinstance(value, float) and value != value))


def _drop_empty(value):
    if isinstance(value, dict):
        kept = {key: _drop_empty(item) for key, item in value.items()}
        return {key: item for key, item in kept.items() if _filled(item, key)}
    if isinstance(value, list):
        return [item for item in map(_drop_empty, value) if _filled(item)]
    return value


def _ordered(value: dict, keys) -> dict:
    return {key: value[key] for key in keys if key in value}


def instruction_of(found: dict) -> str:
    """The edit instruction a reply holds under any of :data:`INSTRUCTION_KEYS`, or ``""``."""
    named = {str(key).strip().lower().replace(" ", "_").replace("-", "_"): value for key, value in found.items()}
    for key in INSTRUCTION_KEYS:
        value = named.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def as_instruction(prompt: str) -> str:
    """A prompt as one sentence: whitespace collapsed, first letter capitalised, closed with a full stop."""
    text = " ".join(prompt.split())
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?" else f"{text}."


def tidy(found: dict, edit: bool, instruction: str = "") -> dict:
    """A parsed reply as a caption: known keys only, in training's order, empty values dropped.

    Args:
        found: The parsed reply.
        edit: Whether the caption carries an ``edit_instruction``.
        instruction: The edit as asked, used when the reply holds no instruction.

    Raises:
        ValueError: The reply is not a JSON object, has no short_description, or an edit has no
            instruction from either source.
    """
    if not isinstance(found, dict):
        raise ValueError("the model answered with JSON that is not an object")
    given = instruction_of(found) if edit else ""
    found = _drop_empty(found)
    keys = [key for key, _ in FIELDS] + (["edit_instruction"] if edit else [])
    caption = _ordered(found, keys)
    if isinstance(caption.get("objects"), list):
        caption["objects"] = [_ordered(item, [key for key, _ in OBJECT_FIELDS])
                              for item in caption["objects"] if isinstance(item, dict)]
    if isinstance(caption.get("text_render"), list):
        caption["text_render"] = [_ordered(item, [key for key, _ in TEXT_FIELDS])
                                  for item in caption["text_render"] if isinstance(item, dict)]
    for key, order in NESTED.items():
        if isinstance(caption.get(key), dict):
            caption[key] = _ordered(caption[key], order)
    caption = _drop_empty(caption)
    if not caption.get("short_description"):
        raise ValueError("the model's JSON has no short_description")
    if edit:
        caption["edit_instruction"] = given or as_instruction(instruction)
        if not caption["edit_instruction"]:
            raise ValueError("the model's JSON has no edit_instruction")
    return caption


def compact(caption: dict) -> str:
    """A caption as one line of JSON."""
    return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)


def write(clip, task: str, prompt: str, pictures=(), mask: Image.Image | None = None, caption: str = "",
          style: str = "", max_length: int = 4096, sampling: dict | None = None, fast: bool = True):
    """Write a Fibo 2.0 caption with the language model inside a CLIP.

    Args:
        clip: A CLIP holding a chat language model; one that reads images for any task with pictures.
        task: One of :data:`TASKS`.
        prompt: The idea, the instruction, or empty to caption a picture.
        pictures: RGB pictures, at most :data:`MOST_IMAGES`.
        mask: White where the single picture is repainted, or ``None``.
        caption: An existing caption to refine, or the current picture's caption for an edit.
        style: Style guidelines for generate and inspire.
        max_length: The most tokens the reply may run to.
        sampling: ``do_sample``, ``temperature``, ``top_k``, ``top_p``, ``min_p``,
            ``repetition_penalty`` and ``seed``.
        fast: Whether to decode on ComfyUI's fixed-cache path where the model allows it.

    Returns:
        ``(prompt, task, report)``: one line of JSON, the task run and the decode report.

    Raises:
        ValueError: The task needs a picture, a prompt or a caption it was not given, the model
            cannot see pictures, or the reply is not a caption.
    """
    from ...latent.h3_writer import json_of, sees_pictures
    from .. import text_decode

    pictures = list(pictures)
    task = task_for(task, prompt, len(pictures), caption)
    if task in ("edit", "inspire", "caption") and not pictures:
        raise ValueError(f"The {task} task describes from a picture. Connect image_1.")
    if task in ("generate", "edit", "inspire", "refine") and not prompt.strip():
        raise ValueError(f"The {task} task needs a prompt saying what to make or change.")
    if task == "refine" and not caption.strip():
        raise ValueError("The refine task revises a caption. Connect the caption to revise to caption.")
    if len(pictures) > MOST_IMAGES:
        raise ValueError(f"The writer reads up to {MOST_IMAGES} pictures, and was given {len(pictures)}.")
    if mask is not None and (task != "edit" or len(pictures) != 1):
        raise ValueError("A mask marks the area to fill in one picture to edit. Use task edit with one image.")
    if task not in ("edit", "inspire", "caption"):
        pictures = []
    elif not sees_pictures(clip):
        raise ValueError("This language model cannot read pictures. Load a Qwen-VL model, such as "
                         "qwen3vl_4b_bf16.safetensors, in Load CLIP.")
    if task == "inspire":
        pictures = pictures[:1]
    if mask is not None:
        layer = mask.convert("L").resize(pictures[0].size, Image.NEAREST)
        pictures = [greyed(pictures[0], layer)]

    import numpy as np
    import torch

    images = [torch.from_numpy(np.asarray(shrunk(picture), dtype=np.float32) / 255.0).unsqueeze(0)
              for picture in pictures]
    system = instructions(task, len(pictures), mask is not None, style)
    asked = request(task, prompt, caption)
    tokens = clip.tokenize(asked, images=images, system_prompt=system, min_length=1, thinking=False)
    sampling = dict(sampling or {})
    options = {
        "do_sample": bool(sampling.get("do_sample", True)),
        "temperature": float(sampling.get("temperature", TEMPERATURE)),
        "top_k": int(sampling.get("top_k", 0)),
        "top_p": float(sampling.get("top_p", 1.0)),
        "min_p": float(sampling.get("min_p", 0.0)),
        "repetition_penalty": float(sampling.get("repetition_penalty", 1.0)),
        "presence_penalty": 0.0,
        "seed": sampling.get("seed", 0),
        "mtp": True,
    }
    ids, report = text_decode.generate(clip, tokens, int(max_length), fast=fast, **options)
    reply = clip.decode(ids)
    try:
        found = tidy(json_of(reply), edit=task == "edit", instruction=prompt)
    except ValueError as error:
        raise ValueError(
            f"The language model's reply is not a Fibo 2.0 caption ({error}). Raise max_length if the reply was "
            f"cut short, or change the seed. It began: {reply.strip()[:160]!r}"
        ) from error
    return compact(found), task, report

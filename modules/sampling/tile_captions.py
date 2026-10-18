"""Captions for the tiles of a picture, written by a vision language model.

Answers are cached per captioner, source picture, enlarged size, tile, seed and rules.

Pictures are IMAGE tensors ``[1, H, W, C]`` in ``[0, 1]``.
"""

from __future__ import annotations

import hashlib
import re
import weakref
from collections import OrderedDict

import torch

from .. import log
from ..latent import h3_writer
from . import tiles

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "OVERVIEW_CONTEXT",
    "OVERVIEW_QUESTION",
    "SYSTEM_CONTEXT",
    "TILE_CONTEXT",
    "TILE_QUESTION",
    "TILE_QUESTION_PANELS",
    "caption_tiles",
    "encode",
    "require_captioner",
    "takes_pictures",
    "takes_several_pictures",
]

logger = log.get_logger("sampling.tile_captions")

#: The rules every tile caption is written under.
DEFAULT_SYSTEM_PROMPT = (
    "You write the prompt an image model uses to repaint a crop of a larger picture.\n"
    "- You are shown the whole picture with its tiles ruled and numbered and the crop's tile "
    "outlined in yellow, and the crop itself at full detail. Describe only what lies inside "
    "the crop.\n"
    "- Name each subject and object inside the crop, largest first, including any only partly "
    "in frame. For each, say which part of it shows, where that part sits within the frame and "
    "how much of the frame it fills. For a person or animal only partly in the crop, name "
    "exactly which part of the body shows and whose it is, as the whole picture shows it. Then "
    "describe their surfaces, textures, colours and lighting, and any legible text in double "
    "quotes.\n"
    "- Use the whole picture and the overview only to recognise and name what the crop shows. "
    "Most of what they show lies outside the crop. Never name anything that is not inside it.\n"
    "- Keep the medium and style of the whole picture.\n"
    "- Where the crop holds only empty background, sky, water or a plain surface, name that "
    "surface and its colour in a few words.\n"
    "- Describe what is there directly. Write one plain paragraph of at most 60 words, with no "
    "preamble and no list. Never write the words tile, crop, overview, outlined, numbered, "
    "attached or image, and never say where the crop sits in the whole picture."
)

#: What the model is asked about the whole picture.
OVERVIEW_QUESTION = (
    "Describe this picture for an image model in at most 80 words: the subject, the setting, "
    "the medium and style, the lighting and the palette. One plain paragraph, no preamble."
)

#: What precedes the overview and tile questions where the picture's own prompt is known;
#: ``{context}`` is filled in.
OVERVIEW_CONTEXT = (
    "The picture was made from this prompt, which names what it shows: {context}\n"
    "Call things what the prompt calls them.\n\n"
)

#: What precedes each tile question where the picture's own prompt is known; ``{context}`` is
#: filled in.
TILE_CONTEXT = "The whole picture was made from this prompt, given for names only: {context}\n\n"

#: What is added to the system prompt where the picture's own prompt is known.
SYSTEM_CONTEXT = "\n\nThe original prompt and context of the image was:\n\n{context}"

#: What the model is asked about each tile, shown as two pictures; ``{overview}``,
#: ``{number}`` and ``{place}`` are filled in.
TILE_QUESTION = (
    "Overview of the whole picture, given for names only: {overview}\n\n"
    "The first attached picture is the whole picture with its tiles ruled and numbered; tile "
    "{number} ({place}) is outlined in yellow. The second is tile {number} at full detail.\n\n"
    "Write the prompt for the second picture only. Start with its largest subject. For "
    "everything cut off by an edge of the second picture, say which edge it touches and which "
    "part of it shows. Leave out anything outside tile {number}, even where the overview or "
    "the prompt names it, and copy no sentence from either."
)

#: What the model is asked about each tile, shown as one picture of two panels.
TILE_QUESTION_PANELS = (
    "Overview of the whole picture, given for names only: {overview}\n\n"
    "The attached picture has two panels. On the left is the whole picture with its tiles "
    "ruled and numbered, tile {number} ({place}) outlined in yellow; on the right is tile "
    "{number} at full detail.\n\n"
    "Write the prompt for the right panel only. Start with its largest subject. For "
    "everything cut off by an edge of the right panel, say which edge it touches and which "
    "part of it shows. Leave out anything outside tile {number}, even where the overview or "
    "the prompt names it, and copy no sentence from either."
)

#: Longest side in pixels of the picture the overview is written from.
OVERVIEW_SIDE = 1024

#: Longest side in pixels of the picture a tile caption is written from.
TILE_SIDE = 768

#: Longest side in pixels of the whole picture shown beside each tile.
CONTEXT_SIDE = 768

#: Most tokens an overview may run to.
OVERVIEW_TOKENS = 200

#: Most tokens a tile caption may run to.
TILE_TOKENS = 160

#: How the captioner draws its words.
SAMPLING = h3_writer.Sampling(temperature=0.3, thinking=False, fast=True)

#: Answers kept for each loaded captioner.
MOST_CACHED = 2048

#: Answers already written, by captioner model and then by picture and question.
_CACHE: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def require_captioner(captioner, clip, node: str) -> None:
    """Refuse a captioner that cannot read pictures, or one with no text encoder beside it.

    Args:
        captioner: The CLIP that writes the captions.
        clip: The CLIP that encodes them, or None.
        node: The node's display name, for the message.

    Raises:
        ValueError: The captioner reads no pictures, or ``clip`` is missing.
    """
    if clip is None:
        raise ValueError(
            f"{node} has a captioner but no clip to encode its captions with. Wire the CLIP "
            "output of the loader the model came from into clip, or disconnect captioner to "
            "give every tile the positive prompt as it is"
        )
    if not (h3_writer.sees_pictures(captioner) or takes_pictures(captioner)):
        raise ValueError(
            f"{node} needs a captioner that reads pictures. Load a vision language model with "
            "Load CLIP, as qwen3vl_8b_fp8_scaled.safetensors or qwen_2.5_vl_7b_fp8_scaled.safetensors, "
            "and wire it into captioner"
        )


def _flat(tokens) -> list:
    """Every entry of a tokenizer's answer, in order."""
    batches = tokens.values() if isinstance(tokens, dict) else [tokens]
    return [item for batch in batches for row in batch for item in row]


def _pictures_in(tokens) -> int:
    """How many entries of a tokenizer's answer carry a picture rather than a token id."""
    count = 0
    for item in _flat(tokens):
        head = item[0] if isinstance(item, tuple) and item else item
        if isinstance(head, dict):
            count += 1
    return count


def takes_pictures(clip) -> bool:
    """Whether a CLIP's tokenizer turns a picture into tokens.

    Args:
        clip: A ComfyUI ``CLIP``.

    Returns:
        True where tokenizing with a picture answers different tokens than without one.
    """
    try:
        plain = clip.tokenize("probe", min_length=1)
        seen = clip.tokenize("probe", image=torch.zeros(1, 64, 64, 3), min_length=1)
    except Exception:
        return False
    return len(_flat(seen)) != len(_flat(plain)) or _pictures_in(seen) > 0


def takes_several_pictures(clip) -> bool:
    """Whether a CLIP's tokenizer takes two pictures into one question.

    Args:
        clip: A ComfyUI ``CLIP``.

    Returns:
        True where two pictures handed together both arrive in the tokens.
    """
    try:
        probe = torch.zeros(1, 64, 64, 3)
        return _pictures_in(clip.tokenize("probe", images=[probe, probe], min_length=1)) == 2
    except Exception:
        return False


def _panels(context: torch.Tensor, crop: torch.Tensor) -> torch.Tensor:
    """The context picture and the crop side by side at one height, a dark gap between them."""
    height = int(context.shape[1])
    width = max(1, round(crop.shape[2] * height / max(1, crop.shape[1])))
    resized = torch.nn.functional.interpolate(crop.movedim(-1, 1), size=(height, width), mode="bicubic",
                                              align_corners=False, antialias=True).movedim(1, -1)
    gap = torch.zeros(1, height, max(4, height // 64), 3)
    return torch.cat([context, gap, resized.clamp(0.0, 1.0)], dim=2)


def _fit(picture: torch.Tensor, side: int) -> torch.Tensor:
    """A picture shrunk so its longest side is at most ``side``, three channels, on the CPU."""
    picture = picture[..., :3].float().cpu().clamp(0.0, 1.0)
    height, width = int(picture.shape[1]), int(picture.shape[2])
    ratio = min(1.0, side / max(height, width))
    if ratio >= 1.0:
        return picture
    size = (max(1, round(height * ratio)), max(1, round(width * ratio)))
    shrunk = torch.nn.functional.interpolate(picture.movedim(-1, 1), size=size, mode="bicubic",
                                             align_corners=False, antialias=True)
    return shrunk.movedim(1, -1).clamp(0.0, 1.0)


def _digest(picture: torch.Tensor) -> str:
    """A fingerprint of a picture's 8-bit pixels and shape."""
    codes = (picture * 255.0).round().to(torch.uint8).numpy()
    return hashlib.sha1(repr(codes.shape).encode() + codes.tobytes()).hexdigest()


def _store(captioner) -> OrderedDict:
    """The answers cached for one captioner, created on first use."""
    model = getattr(captioner, "cond_stage_model", None)
    if model is None:
        return OrderedDict()
    store = _CACHE.get(model)
    if store is None:
        store = _CACHE[model] = OrderedDict()
    return store


def _tidy(text: str) -> str:
    """A model's reply as one plain line, without thinking, quotes or a label."""
    reply = h3_writer.clean_reply(text)
    reply = re.sub(r"^\s*(?:prompt|caption|tile prompt)\s*:\s*", "", reply, flags=re.I)
    reply = re.sub(r"^(?:the\s+)?(?:attached\s+|given\s+|provided\s+)?(?:picture|image|crop)\s+"
                   r"(?:is|shows|depicts|contains|features)\s+", "", reply, flags=re.I)
    reply = reply[:1].upper() + reply[1:]
    reply = re.sub(r"\s+", " ", reply).strip()
    if len(reply) > 1 and reply[0] == reply[-1] and reply[0] in "\"'":
        reply = reply[1:-1].strip()
    return reply


def _ask(write, store: OrderedDict, key: tuple, pictures: list, question: str, budget: int,
         ruled: bool) -> str:
    """One answer about pictures, read from the cache where the same key was asked before."""
    if key in store:
        store.move_to_end(key)
        return store[key]
    if len(pictures) > 1:
        reply = write(question, images=pictures, max_length=budget, ruled=ruled)
    else:
        reply = write(question, image=pictures[0], max_length=budget, ruled=ruled)
    answer = _tidy(reply)
    store[key] = answer
    while len(store) > MOST_CACHED:
        store.popitem(last=False)
    return answer


def caption_tiles(captioner, image: torch.Tensor, grid: tiles.Grid, system_prompt: str,
                  progress=None, announce=None, context: str = "", seed: int = 0, source=None,
                  written=None) -> tuple:
    """An overview of a picture, then one caption for each tile of it.

    Args:
        captioner: A CLIP holding a vision language model.
        image: The picture the grid was planned over, ``[1, H, W, C]``.
        grid: The tiling.
        system_prompt: The rules each tile caption is written under.
        progress: A ComfyUI progress bar advanced once per answer, or None.
        announce: Called as ``announce("overview")`` before the overview and
            ``announce("caption", [tile])`` before each tile, or None.
        context: The prompt the picture was made from, given to the overview and every
            caption, empty for none.
        seed: Seeds the words each answer is drawn with.
        source: The picture before enlargement, ``[1, h, w, C]``, whose pixels identify the
            answers in the cache; ``image`` when None.
        written: Called as ``written(tile, caption)`` once each answer is in, ``tile`` being
            ``-1`` for the overview, or None.

    Returns:
        ``(overview, captions)``, ``captions`` holding one string per tile, row by row, empty
        where the model wrote nothing.
    """
    import comfy.model_management

    context = str(context or "").strip()
    system = str(system_prompt or "")
    if context:
        system += SYSTEM_CONTEXT.format(context=context)
    write = h3_writer.make_writer(captioner, int(seed), SAMPLING, system)
    store = _store(captioner)
    ident = (_digest(image if source is None else source), tuple(image.shape), int(seed), system)
    lead = OVERVIEW_CONTEXT.format(context=context) if context else ""
    tile_lead = TILE_CONTEXT.format(context=context) if context else ""
    if announce is not None:
        announce("overview")
    question = lead + OVERVIEW_QUESTION
    overview = _ask(write, store, (ident, "overview", question, OVERVIEW_TOKENS),
                    [_fit(image, OVERVIEW_SIDE)], question, OVERVIEW_TOKENS, ruled=False)
    if written is not None:
        written(-1, overview)
    if progress is not None:
        progress.update(1)
    height, width = int(image.shape[1]), int(image.shape[2])
    several = takes_several_pictures(captioner)
    whole = _fit(image, CONTEXT_SIDE)
    captions = []
    for index, (h0, h1, w0, w1) in enumerate(grid.boxes()):
        comfy.model_management.throw_exception_if_processing_interrupted()
        if announce is not None:
            announce("caption", [index])
        top, bottom = int(round(h0 * grid.scale)), min(height, int(round(h1 * grid.scale)))
        left, right = int(round(w0 * grid.scale)), min(width, int(round(w1 * grid.scale)))
        crop = _fit(image[:, top:max(bottom, top + 1), left:max(right, left + 1)], TILE_SIDE)
        context_picture = tiles.numbered(whole, grid, index, CONTEXT_SIDE)
        template = TILE_QUESTION if several else TILE_QUESTION_PANELS
        pictures = [context_picture, crop] if several else [_panels(context_picture, crop)]
        question = tile_lead + template.format(overview=overview or "not available",
                                               number=index + 1, place=tiles.position(grid, index))
        key = (ident, (h0, h1, w0, w1), float(grid.scale), question, TILE_TOKENS)
        captions.append(_ask(write, store, key, pictures, question, TILE_TOKENS, ruled=True))
        if written is not None:
            written(index, captions[-1])
        if progress is not None:
            progress.update(1)
    return overview, captions


def encode(clip, text: str) -> list:
    """Encode one prompt as CLIP Text Encode does.

    Args:
        clip: The model's text encoder.
        text: The prompt.

    Returns:
        A CONDITIONING list.
    """
    return clip.encode_from_tokens_scheduled(clip.tokenize(text))

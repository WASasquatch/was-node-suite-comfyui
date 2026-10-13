"""Captions for the tiles of a picture, written by a vision language model and cached per picture.

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
    "TILE_QUESTION",
    "caption_tiles",
    "encode",
    "require_captioner",
    "takes_pictures",
]

logger = log.get_logger("sampling.tile_captions")

#: The rules every tile caption is written under.
DEFAULT_SYSTEM_PROMPT = (
    "You write the prompt an image model uses to repaint a crop of a larger picture.\n"
    "- Look only at the attached crop. Name each subject and object visible in it, largest "
    "first, including any only partly in frame, and say which part of it shows. Then describe "
    "their surfaces, textures, colours and lighting, and any legible text in double quotes.\n"
    "- The overview describes the whole picture, and most of what it names lies outside this "
    "crop. Use it only to name and style what the crop shows. Never name anything that is not "
    "visible in the crop.\n"
    "- Keep the medium and style of the whole picture.\n"
    "- Where the crop holds only empty background, sky, water or a plain surface, name that "
    "surface and its colour in a few words.\n"
    "- Describe what is there directly. Write one plain paragraph of at most 60 words, with no "
    "preamble and no list. Never write the words tile, crop, section, corner, overview, "
    "attached or image, and never say where in the picture the crop sits."
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

#: What is added to the system prompt where the picture's own prompt is known.
SYSTEM_CONTEXT = "\n\nThe original prompt and context of the image was:\n\n{context}"

#: What the model is asked about each tile; ``{overview}`` and ``{place}`` are filled in.
TILE_QUESTION = (
    "Overview of the whole picture: {overview}\n\n"
    "The attached picture is a crop of that picture ({place}). Write the prompt for everything "
    "visible in the attached picture."
)

#: Longest side in pixels of the picture the overview is written from.
OVERVIEW_SIDE = 1024

#: Longest side in pixels of the picture a tile caption is written from.
TILE_SIDE = 768

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


def takes_pictures(clip) -> bool:
    """Whether a CLIP's tokenizer turns a picture into tokens.

    Args:
        clip: A ComfyUI ``CLIP``.

    Returns:
        True where tokenizing with a picture answers different tokens than without one.
    """
    def flat(tokens):
        batches = tokens.values() if isinstance(tokens, dict) else [tokens]
        return [item for batch in batches for row in batch for item in row]

    try:
        plain = flat(clip.tokenize("probe", min_length=1))
        seen = flat(clip.tokenize("probe", image=torch.zeros(1, 64, 64, 3), min_length=1))
    except Exception:
        return False
    return len(seen) != len(plain) or any(not isinstance(item, (int, tuple)) for item in seen)


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


def _ask(write, store: OrderedDict, picture: torch.Tensor, question: str, system: str,
         budget: int, ruled: bool) -> str:
    """One answer about a picture, read from the cache where the same question was asked before."""
    key = (_digest(picture), question, system if ruled else "", int(budget))
    if key in store:
        store.move_to_end(key)
        return store[key]
    answer = _tidy(write(question, image=picture, max_length=budget, ruled=ruled))
    store[key] = answer
    while len(store) > MOST_CACHED:
        store.popitem(last=False)
    return answer


def caption_tiles(captioner, image: torch.Tensor, grid: tiles.Grid, system_prompt: str,
                  progress=None, announce=None, context: str = "") -> tuple:
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

    Returns:
        ``(overview, captions)``, ``captions`` holding one string per tile, row by row, empty
        where the model wrote nothing.
    """
    import comfy.model_management

    context = str(context or "").strip()
    system = str(system_prompt or "")
    if context:
        system += SYSTEM_CONTEXT.format(context=context)
    write = h3_writer.make_writer(captioner, 0, SAMPLING, system)
    store = _store(captioner)
    lead = OVERVIEW_CONTEXT.format(context=context) if context else ""
    if announce is not None:
        announce("overview")
    overview = _ask(write, store, _fit(image, OVERVIEW_SIDE), lead + OVERVIEW_QUESTION, system,
                    OVERVIEW_TOKENS, ruled=False)
    if progress is not None:
        progress.update(1)
    height, width = int(image.shape[1]), int(image.shape[2])
    captions = []
    for index, (h0, h1, w0, w1) in enumerate(grid.boxes()):
        comfy.model_management.throw_exception_if_processing_interrupted()
        if announce is not None:
            announce("caption", [index])
        top, bottom = int(round(h0 * grid.scale)), min(height, int(round(h1 * grid.scale)))
        left, right = int(round(w0 * grid.scale)), min(width, int(round(w1 * grid.scale)))
        crop = image[:, top:max(bottom, top + 1), left:max(right, left + 1)]
        question = lead + TILE_QUESTION.format(overview=overview or "not available",
                                               place=tiles.position(grid, index))
        captions.append(_ask(write, store, _fit(crop, TILE_SIDE), question, system, TILE_TOKENS,
                             ruled=True))
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

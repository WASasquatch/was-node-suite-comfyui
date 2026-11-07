"""Edit instructions and revised prompts written, and pictures judged, by a CLIP's language model.

Scores run from 0 to 10: an edit on following, consistency and quality, a generated picture on
following and quality.
"""

from __future__ import annotations

#: Every score, and the least each must reach for a picture to pass.
PASS = {"following": 9, "consistency": 8, "quality": 6}

#: The scores a picture generated from a prompt is given.
GENERATED = ("following", "quality")

#: Pixels a picture is brought down to before the model reads it.
VISION_PIXELS = 1024 * 1024

#: The most pictures one judging pass compares.
MOST_PICTURES = 6

REWRITE_SYSTEM = (
    "You rewrite image editing instructions for an instruction-based image editor. You see the picture to"
    " edit and the user's instruction. Write one explicit instruction in English that the editor can "
    "follow without guessing. Give every change the user asks for its own sentence, in the user's order, "
    "so no change is merged into another. Name the subject as it appears, every part that changes and "
    "how, with colours, materials, lighting and any effects the request implies, such as a glow or an "
    "aura that belongs to a named transformation or style. Then name what stays, starting with "
    "'Preserve', such as the person, face, pose, framing and background, leaving out whatever the change "
    "itself alters. Never add a change the user did not ask for or imply. When a judge's notes on an "
    "earlier attempt are given, write a new instruction that corrects every flaw they name in plain "
    "positive terms, such as a faint translucent glow where the last glow was overexposed, and keeps what"
    " worked. Reply with the instruction alone, at most 90 words, with no preamble, quotes or list."
)

REVISE_SYSTEM = (
    "You revise prompts for a text-to-image model. You get the user's prompt, the prompt the last "
    "attempt was made from, a judge's notes on what its picture got wrong, and that picture when it is "
    "shown. Write one new prompt that asks for everything the user's prompt asks for and corrects every "
    "flaw the notes name. Put what the last picture got wrong first, since the model weighs the start of "
    "a prompt most. Where the picture fell back on a more common arrangement than the one asked for, such "
    "as the reverse of who rides, carries or holds whom, describe the asked arrangement as plain physical "
    "layout: which subject is on top and which below, what touches what, which way each faces and what "
    "each one's limbs are doing. State counts, positions, colours and written text outright, and describe "
    "what belongs in the picture rather than what must not be there. Never contradict the user's prompt, "
    "and never repeat the last attempt's prompt: when it failed, word the fix differently. Keep the user's "
    "subject and style, and keep a tag list a tag list and sentences sentences. Never add a subject the "
    "user did not ask for. Reply with the prompt alone, with no preamble, quotes or notes."
)

#: Further asks when a revision comes back as the last attempt's prompt, and their sampling temperature.
REVISE_RETRIES = 2
REVISE_RETRY_TEMPERATURE = 0.7

_SCALE = (
    "Use the whole scale: 10 means no visible flaw at all, each visible flaw costs at least one point, "
    "and pictures that differ must not share every score. Write each picture's notes before its scores."
)

JUDGE_EDIT_SYSTEM = (
    "You are a strict judge of image edits. The picture labelled Source is the picture before the edit. "
    "Each picture labelled Edit followed by a number is one edit of it, made from the same instruction. "
    "Compare every edit with the Source, then look at each closely and against the other edits, and "
    "score it from 0 to 10 on three things. "
    "following: how fully and exactly it makes the change asked for, implied parts included. "
    "consistency: how well everything the instruction leaves alone is kept, such as identity, face, "
    "expression, pose, composition, background and text. quality: how free it is of artifacts, smears, "
    "banding, distorted anatomy, garbled detail or a pasted-on look. " + _SCALE + " Reply with JSON "
    'only, one entry per edit with its number: {"images": [{"image": 0, "notes": "the concrete flaws '
    'you see", "following": n, "consistency": n, "quality": n}], "best": number of the edit you would keep}'
)

JUDGE_PROMPT_SYSTEM = (
    "You are a strict judge of generated images. Each picture labelled Image followed by a number was "
    "generated from the same prompt. Look at each closely and against the others, then score it from 0 "
    "to 10 on two things. following: how fully and exactly it shows what the "
    "prompt asks for, every subject, attribute, count, position, style and written text included. "
    "quality: how free it is of artifacts, smears, banding, distorted anatomy and garbled detail or "
    "text, and how well it is composed and lit. " + _SCALE + " Reply with JSON only, "
    'one entry per image with its number: {"images": [{"image": 0, "notes": "the concrete flaws you see", '
    '"following": n, "quality": n}], "best": number of the image you would keep}'
)


def _pixels(picture):
    """A PIL picture no larger than :data:`VISION_PIXELS`, as a ``(1, H, W, 3)`` tensor in ``[0, 1]``."""
    import numpy as np
    import torch
    from PIL import Image

    picture = picture.convert("RGB")
    area = picture.width * picture.height
    if area > VISION_PIXELS:
        ratio = (VISION_PIXELS / area) ** 0.5
        picture = picture.resize((max(1, int(picture.width * ratio)), max(1, int(picture.height * ratio))),
                                 Image.LANCZOS)
    return torch.from_numpy(np.asarray(picture, dtype=np.float32) / 255.0).unsqueeze(0)


def _ask(clip, system: str, request: str, pictures, max_length: int, temperature: float, seed: int,
         labels=None):
    """The model's reply to ``request`` about ``pictures``, each named by ``labels`` when given, and the decode report."""
    from ..latent.h3_writer import sees_pictures
    from . import text_decode

    if pictures and not sees_pictures(clip):
        raise ValueError(
            "This language model cannot read pictures. Load qwen3.5_9b_int8_convrot.safetensors with Load "
            "CLIP and type krea2, or another Qwen-VL model."
        )
    tokens = _tokens(clip, system, request, pictures, labels)
    options = {
        "do_sample": temperature > 0,
        "temperature": float(temperature) if temperature > 0 else 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "seed": int(seed),
        "mtp": True,
    }
    ids, report = text_decode.generate(clip, tokens, int(max_length), **options)
    return clip.decode(ids), report


def _tokens(clip, system: str, request: str, pictures, labels):
    """``clip.tokenize`` of one user turn, each picture after its label where the model reads ChatML."""
    from ..llm.clip_llm import CHATML, EMPTY_THINK, VISION_BLOCK, family_of

    pixels = [_pixels(p) for p in pictures]
    family = family_of(clip)
    if not pixels:
        if family.kind != CHATML:
            return clip.tokenize(f"{system}\n\n{request}", min_length=1)
        text = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{request}<|im_end|>\n<|im_start|>assistant\n"
        if family.think in ("open", "hybrid"):
            text += EMPTY_THINK
        return clip.tokenize(text, llama_template="{}", skip_template=True, min_length=1, thinking=False)
    if labels is None or family.kind != CHATML or not family.vision:
        if labels:
            request = "The pictures, in order: " + ", ".join(labels) + ".\n" + request
        return clip.tokenize(request, images=pixels, system_prompt=system, min_length=1, thinking=False)
    slots = "".join(f"{label}: {VISION_BLOCK}\n" for label in labels)
    text = (
        f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{slots}{request}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    if family.think in ("open", "hybrid"):
        text += EMPTY_THINK
    return clip.tokenize(text, llama_template="{}", images=pixels, skip_template=True, min_length=1, thinking=False)


def rewrite(clip, picture, instruction: str, feedback: str = "", previous: str = "", max_length: int = 256,
            temperature: float = 0.0, seed: int = 0):
    """An explicit editing instruction for ``picture`` written from a short one.

    Args:
        clip: A CLIP holding a language model that reads pictures.
        picture: The PIL picture to edit.
        instruction: The user's instruction.
        feedback: A judge's notes on the last attempt, empty for the first.
        previous: The instruction the last attempt was made from, empty for the first.
        max_length: The most tokens the reply may run to.
        temperature: 0 for the likeliest words every time; above 0 to sample.
        seed: Seed for sampling.

    Returns:
        ``(instruction, report)``.

    Raises:
        ValueError: The instruction is empty, the model cannot read pictures, or it wrote nothing.
    """
    from ..latent.h3_writer import clean_reply

    if not instruction.strip():
        raise ValueError("Write an instruction for the language model to make explicit.")
    request = f"The instruction: {instruction.strip()}"
    if (previous or "").strip():
        request += f"\nThe last attempt was made from this instruction: {previous.strip()}"
    if (feedback or "").strip():
        request += f"\nA judge found these flaws in its result: {feedback.strip()}"
    reply, report = _ask(clip, REWRITE_SYSTEM, request, [picture], max_length, temperature, seed)
    text = " ".join(clean_reply(reply).strip().strip('"').split())
    if not text:
        raise ValueError("The language model wrote no instruction. Raise max_length or change the seed.")
    return text, report


def revise(clip, prompt: str, feedback: str, previous: str = "", picture=None, max_length: int = 384,
           temperature: float = 0.0, seed: int = 0):
    """A text-to-image prompt rewritten so the next attempt fixes what a judge found wrong with the last.

    Args:
        clip: A CLIP holding a language model; one that reads pictures also looks at ``picture``.
        prompt: The user's prompt.
        feedback: The judge's notes on the last attempt.
        previous: The prompt the last attempt was made from, empty when it was ``prompt``.
        picture: The last attempt's PIL picture, or ``None``.
        max_length: The most tokens the reply may run to.
        temperature: 0 for the likeliest words every time; above 0 to sample.
        seed: Seed for sampling.

    Returns:
        ``(prompt, report)``.

    Raises:
        ValueError: The prompt or the notes are empty, or the model wrote nothing.
    """
    from ..latent.h3_writer import clean_reply, sees_pictures

    if not prompt.strip():
        raise ValueError("Write the prompt the pictures are meant to show.")
    if not (feedback or "").strip():
        raise ValueError("A revision needs the judge's notes on the last attempt.")
    request = f"The user's prompt: {prompt.strip()}"
    if (previous or "").strip() and previous.strip() != prompt.strip():
        request += f"\nThe last attempt was made from: {previous.strip()}"
    request += f"\nThe judge's notes on its picture: {feedback.strip()}"
    pictures = [picture] if picture is not None and sees_pictures(clip) else []
    labels = ["The last attempt"] if pictures else None
    last = " ".join((previous or prompt).split()).lower()
    text, report = "", None
    for attempt in range(1 + REVISE_RETRIES):
        heat = temperature if attempt == 0 else max(temperature, REVISE_RETRY_TEMPERATURE)
        asked = request if attempt == 0 else request + "\nYour last answer repeated that prompt; word it differently."
        reply, report = _ask(clip, REVISE_SYSTEM, asked, pictures, max_length, heat, seed + attempt, labels)
        text = clean_reply(reply).strip().strip('"').strip()
        if text and " ".join(text.split()).lower() != last:
            break
    if not text:
        raise ValueError("The language model wrote no prompt. Raise max_length or change the seed.")
    return text, report


def judge(clip, candidates, prompt: str, source=None, least: dict | None = None, max_length: int | None = None):
    """Scores for several pictures made from one prompt, compared side by side.

    Args:
        clip: A CLIP holding a language model that reads pictures.
        candidates: The PIL pictures to score, at most :data:`MOST_PICTURES`.
        prompt: The instruction the edits followed, or the prompt the pictures were generated from.
        source: The PIL picture before the edits, ``None`` for generated pictures.
        least: The least each score must reach to pass, :data:`PASS` where left out.
        max_length: The most tokens the reply may run to, enough for every picture's notes when ``None``.

    Returns:
        ``(scored, pick, report)``: per picture, its scores as floats from 0 to 10, ``notes`` and
        ``passed``, in the order given; the picture the model would keep, or ``None``.

    Raises:
        ValueError: There are no pictures or more than :data:`MOST_PICTURES`, the model cannot
            read pictures, or its reply does not score every picture.
    """

    candidates = list(candidates)
    if not candidates or len(candidates) > MOST_PICTURES:
        raise ValueError(f"Judge 1 to {MOST_PICTURES} pictures at a time; {len(candidates)} were given.")
    names = tuple(PASS) if source is not None else GENERATED
    floor = {name: float({**PASS, **(least or {})}[name]) for name in names}
    edits = source is not None
    system = JUDGE_EDIT_SYSTEM if edits else JUDGE_PROMPT_SYSTEM
    word = "Edit" if edits else "Image"
    labels = (["Source"] if edits else []) + [f"{word} {index}" for index in range(len(candidates))]
    request = (
        f"The {'instruction' if edits else 'prompt'}: {prompt.strip()}\n"
        f"Score {', '.join(labels[1:] if edits else labels)}."
    )
    pictures = ([source] if edits else []) + candidates
    length = max_length or 256 + 192 * len(candidates)
    reply, report = _ask(clip, system, request, pictures, length, 0.0, 0, labels)
    try:
        found, pick = _reply_rows(reply)
        rows = {int(row.get("image", row.get("edit"))): row for row in found}
        scored = []
        for index in range(len(candidates)):
            row = rows[index]
            scores = {name: max(0.0, min(10.0, float(_score(row, name)))) for name in names}
            scores["notes"] = " ".join(str(row.get("notes", "")).split())
            scores["passed"] = all(scores[name] >= floor[name] for name in names)
            scored.append(scores)
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise ValueError(
            f"The language model's reply does not score every picture ({error}). It began: "
            f"{reply.strip()[:160]!r}"
        ) from error
    return scored, pick, report


def _reply_rows(reply: str):
    """Every scored row in a judge's reply, wrapped, listed or loose, and the number it would keep or ``None``.

    Raises:
        ValueError: The reply holds no JSON.
    """
    import json
    import re

    from ..latent.h3_writer import clean_reply

    text = re.sub(r"```(?:json)?", "", clean_reply(reply))
    decoder = json.JSONDecoder()
    values, position = [], 0
    while True:
        starts = [i for i in (text.find("{", position), text.find("[", position)) if i >= 0]
        if not starts:
            break
        start = min(starts)
        try:
            value, end = decoder.raw_decode(text, start)
        except ValueError:
            position = start + 1
            continue
        values.append(value)
        position = end
    if not values:
        raise ValueError("no JSON in the reply")
    rows, pick = [], None
    for value in values:
        items = value if isinstance(value, list) else [value]
        for item in items:
            if not isinstance(item, dict):
                continue
            listed = item.get("images", item.get("edits"))
            if isinstance(listed, list):
                rows.extend(row for row in listed if isinstance(row, dict))
            elif "image" in item or "edit" in item:
                rows.append(item)
            if pick is None and item.get("best") is not None:
                try:
                    pick = int(item["best"])
                except (TypeError, ValueError):
                    pick = None
    return rows, pick


def _score(row: dict, name: str):
    """One score from a reply row, under its own name or with an ``instruction_`` or ``prompt_`` prefix.

    Raises:
        KeyError: The row has no such score.
    """
    for key in (name, f"instruction_{name}", f"prompt_{name}"):
        if key in row:
            return row[key]
    raise KeyError(name)


def best(scored: list, pick: int | None = None) -> int:
    """The picture to keep: passing ones first, then the highest total, then the model's pick, then the earliest."""
    def key(item):
        index, scores = item
        return (scores["passed"], sum(scores.get(name, 0.0) for name in PASS), index == pick, -index)

    return max(enumerate(scored), key=key)[0]

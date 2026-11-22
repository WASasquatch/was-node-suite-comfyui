"""Picture rows for MiniMax H3 Image conditioning.

A picture row is the mean vision row of a flat grey picture. Copies join the text span tagged ``0``,
after the prompt or just before it.
"""

from __future__ import annotations

import weakref

import torch

__all__ = ["AFTER", "BEFORE", "PLACEMENTS", "ROWS_PER_PICTURE", "add_rows", "picture_row", "row_count"]

#: Vision rows a 1024 px picture adds to the text span, its start and end marks included.
ROWS_PER_PICTURE = 1026

#: Side of the grey picture the row is read from, in pixels.
BLANK_SIDE = 1024

#: Grey level of that picture, 0 to 1: ``#808080``.
BLANK_GREY = 128 / 255

#: How MiniMax H3 Image labels a picture in the prompt.
PICTURE_LABEL = "Picture %d: "

#: Where the rows go in the text span.
AFTER = "after prompt"
BEFORE = "before prompt"
PLACEMENTS = (AFTER, BEFORE)

_ROWS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def row_count(strength: float) -> int:
    """Rows for ``strength``, where ``1`` is one picture's worth."""
    return max(0, int(round(float(strength) * ROWS_PER_PICTURE)))


def picture_row(clip) -> torch.Tensor:
    """The mean vision row of a grey picture, ``[1, 1, C]`` on the CPU, kept per encoder and patch set.

    Raises:
        ValueError: The text encoder does not tag vision rows, so it is not MiniMax H3 Image's.
    """
    model = clip.cond_stage_model
    key = getattr(clip.patcher, "patches_uuid", None)
    kept = _ROWS.get(model)
    if kept is not None and key in kept:
        return kept[key]
    picture = torch.full((1, BLANK_SIDE, BLANK_SIDE, 3), BLANK_GREY, dtype=torch.float32)
    cond = clip.encode_from_tokens_scheduled(clip.tokenize("", images=[picture], picture_label=PICTURE_LABEL))
    rows, extras = cond[0][0], cond[0][1]
    tags = extras.get("minimax_token_tags")
    vision = None if tags is None else tags.reshape(-1).to(rows.device) == 0
    if vision is None or not bool(vision.any()):
        raise ValueError(
            "H3 Image VisRow Conditioning needs the MiniMax H3 Image text encoder: Load CLIP with type minimax."
        )
    row = rows[:, vision].mean(dim=1, keepdim=True).cpu()
    _ROWS.setdefault(model, {})[key] = row
    return row


def _prompt_start(tags: torch.Tensor) -> int:
    """Index of the first prompt row: just past the last vision row, or ``0`` with no pictures."""
    vision = (tags == 0).nonzero()
    return int(vision[-1]) + 1 if len(vision) else 0


def add_rows(conditioning: list, row: torch.Tensor, count: int, placement: str = AFTER) -> list:
    """Conditioning with ``count`` copies of ``row`` added to each entry's text span.

    Args:
        conditioning: MiniMax H3 Image CONDITIONING.
        row: A ``[1, 1, C]`` picture row.
        count: Rows to add; ``0`` returns the conditioning as it is.
        placement: ``AFTER`` the prompt, or ``BEFORE`` it, after any reference pictures.

    Returns:
        New CONDITIONING; the input is left as it was.
    """
    if count <= 0:
        return conditioning
    out = []
    for rows, extras in conditioning:
        tags = extras.get("minimax_token_tags")
        tags = torch.ones(rows.shape[1], dtype=torch.long) if tags is None else tags.reshape(-1)
        at = rows.shape[1] if placement == AFTER else _prompt_start(tags)
        block = row.to(device=rows.device, dtype=rows.dtype).expand(rows.shape[0], count, rows.shape[2])
        zeros = torch.zeros(count, dtype=tags.dtype, device=tags.device)
        entry = dict(extras)
        entry["minimax_token_tags"] = torch.cat([tags[:at], zeros, tags[at:]])
        out.append([torch.cat([rows[:, :at], block, rows[:, at:]], dim=1), entry])
    return out

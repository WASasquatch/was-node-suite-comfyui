"""Choosing, renaming and labelling a model's parts from the text a node is given.

Patterns are comma-separated wildcards matched against part names and labels, as
`selection` reads them. Pairs are written ``pattern = value``.
"""

from __future__ import annotations

__all__ = ["MAX_PAIRS", "filter_parts", "label_parts", "pairs", "part_names", "rename_parts"]

from .merge import reindex_parts
from .scene import Model3D
from .selection import MAX_SELECTION_CHARS, matches

#: Most ``pattern = value`` pairs read from one input.
MAX_PAIRS = 256

#: Longest part name or label written, in characters.
MAX_NAME_CHARS = 128


def part_names(model: Model3D, limit: int = 12) -> str:
    """The model's part names, comma-separated and cut to ``limit`` entries, for a message."""
    names = [part.name or "(unnamed)" for part in model.parts]
    shown = ", ".join(names[:limit])
    return shown + (f" and {len(names) - limit} more" if len(names) > limit else "")


def pairs(text: str, input_name: str) -> list[tuple[str, str]]:
    """The ``pattern = value`` pairs in a comma-separated text.

    Args:
        text: Such as ``"mesh_0 = body, mesh_1 = head"``.
        input_name: The node input the text came from, named in an error.

    Returns:
        ``(pattern, value)`` pairs in order, at most `MAX_PAIRS`.

    Raises:
        ValueError: A piece has no ``=``, or an empty pattern or value.
    """
    found = []
    for piece in str(text or "")[:MAX_SELECTION_CHARS].split(","):
        if not piece.strip():
            continue
        pattern, equals, value = piece.partition("=")
        pattern, value = pattern.strip(), value.strip()[:MAX_NAME_CHARS]
        if not equals or not pattern or not value:
            raise ValueError(f"{input_name} reads '{piece.strip()[:60]}', which is not a pair; write "
                             "each as pattern = value, separated by commas, such as mesh_0 = body")
        found.append((pattern, value))
    return found[:MAX_PAIRS]


def filter_parts(model: Model3D, keep: str, drop: str) -> int:
    """Keep the parts ``keep`` matches, then remove the ones ``drop`` matches.

    Args:
        model: The model, changed in place.
        keep: Patterns of parts to keep; empty keeps every part.
        drop: Patterns of parts to remove; empty removes none.

    Returns:
        How many parts were removed.

    Raises:
        ValueError: ``keep`` matches no part, or nothing would be left.
    """
    keep, drop = str(keep or "").strip(), str(drop or "").strip()
    if not keep and not drop:
        return 0
    kept = [index for index, part in enumerate(model.parts) if not keep or matches(part, keep)]
    if keep and not kept:
        raise ValueError(f"keep '{keep[:80]}' matches none of the model's parts ({part_names(model)}); "
                         "use a name or label from that list, or leave keep empty to keep every part")
    if drop:
        kept = [index for index in kept if not matches(model.parts[index], drop)]
    if not kept:
        raise ValueError(f"drop '{drop[:80]}' removes every part that was left ({part_names(model)}); "
                         "narrow the pattern so at least one part remains")
    removed = len(model.parts) - len(kept)
    if removed:
        reindex_parts(model, [model.parts[index] for index in kept], [[index] for index in kept])
    return removed


def rename_parts(model: Model3D, text: str) -> tuple[int, list[str]]:
    """Rename parts by ``pattern = new name`` pairs; several matches are numbered.

    Args:
        model: The model, changed in place.
        text: The pairs, such as ``"tripo_node_* = body"``.

    Returns:
        ``(parts renamed, patterns that matched nothing)``.

    Raises:
        ValueError: A pair is malformed.
    """
    renamed, unmatched = 0, []
    for pattern, name in pairs(text, "rename"):
        hits = [part for part in model.parts if matches(part, pattern)]
        if not hits:
            unmatched.append(pattern)
        for number, part in enumerate(hits):
            part.name = name if number == 0 else f"{name}_{number + 1}"
        renamed += len(hits)
    return renamed, unmatched


def label_parts(model: Model3D, text: str) -> tuple[int, list[str]]:
    """Set a role label, such as ``head`` or ``hair``, on parts by ``pattern = label`` pairs.

    Args:
        model: The model, changed in place.
        text: The pairs, such as ``"*hair* = hair, body = body"``.

    Returns:
        ``(parts labelled, patterns that matched nothing)``.

    Raises:
        ValueError: A pair is malformed.
    """
    labelled, unmatched = 0, []
    for pattern, label in pairs(text, "labels"):
        hits = [part for part in model.parts if matches(part, pattern)]
        if not hits:
            unmatched.append(pattern)
        for part in hits:
            part.label = label
        labelled += len(hits)
    return labelled, unmatched

"""Choosing parts of a model by name or label, as the ``parts`` input of the mesh nodes reads.

A selection is comma-separated wildcard patterns matched against each part's name and label,
case-insensitively. An empty selection chooses every part.
"""

from __future__ import annotations

__all__ = ["matches", "selected"]

from fnmatch import fnmatchcase

from .scene import Model3D, Part

#: Longest selection text read.
MAX_SELECTION_CHARS = 2000


def _patterns(text: str) -> list[str]:
    """The non-empty patterns in a selection, lower-cased."""
    return [piece.strip().lower() for piece in str(text or "")[:MAX_SELECTION_CHARS].split(",")
            if piece.strip()]


def matches(part: Part, text: str) -> bool:
    """Whether a part is chosen by a selection.

    Args:
        part: The part to test.
        text: Comma-separated patterns such as ``"head, hair*"``; empty chooses every part.

    Returns:
        True when any pattern matches the part's name or label.
    """
    patterns = _patterns(text)
    if not patterns:
        return True
    names = (part.name.lower(), part.label.lower())
    return any(fnmatchcase(name, pattern) for pattern in patterns for name in names if name)


def selected(model: Model3D, text: str) -> list[int]:
    """Indices of the parts a selection chooses.

    Args:
        model: The model whose parts are tested.
        text: Comma-separated patterns; empty chooses every part.

    Returns:
        Part indices in model order.
    """
    return [index for index, part in enumerate(model.parts) if matches(part, text)]

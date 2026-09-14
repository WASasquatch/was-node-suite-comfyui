"""Numeric bounds the pack shares with core ComfyUI.

:func:`max_resolution` bounds a pixel dimension, a pixel coordinate and an index into a
batch, and :data:`MAX_RESOLUTION` mirrors the constant of that name in ComfyUI's
``nodes.py``.
"""

from __future__ import annotations

import sys

__all__ = ["MAX_RESOLUTION", "max_resolution"]

#: Widest side, coordinate or index core ComfyUI accepts, as ``nodes.py`` declares it.
#: Read only where that module is absent, so the two spellings of the bound cannot
#: disagree on an install that has one.
MAX_RESOLUTION = 16384

#: What :func:`max_resolution` settled on, or ``None`` before its first call.
_resolved: int | None = None


def max_resolution() -> int:
    """Core ComfyUI's bound for a pixel dimension, a coordinate or a batch index.

    Returns:
        ``MAX_RESOLUTION`` as the running ComfyUI declares it, or
        :data:`MAX_RESOLUTION` where ComfyUI is not importable or binds no positive
        integer under that name.
    """
    # A widget's max is enforced when a prompt is queued, so an install that lowers
    # MAX_RESOLUTION refuses a workflow saved against a larger one.
    global _resolved
    if _resolved is None:
        _resolved = _from_core()
    return _resolved


def _from_core() -> int:
    """Read ``MAX_RESOLUTION`` out of ComfyUI's ``nodes`` module.

    Returns:
        The value ComfyUI declares, or :data:`MAX_RESOLUTION`.
    """
    module = sys.modules.get("nodes")
    if module is None:
        try:
            import nodes as module
        except Exception:
            return MAX_RESOLUTION
    value = getattr(module, "MAX_RESOLUTION", None)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return MAX_RESOLUTION

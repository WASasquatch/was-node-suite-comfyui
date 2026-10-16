"""Number inputs held to the range their schema declares, typed, linked or posted alike.

:func:`guard` wraps a node's ``execute`` to refuse an ``INT`` or ``FLOAT`` outside its range.
"""

from __future__ import annotations

import functools
import inspect
import math
import threading

from .. import log

__all__ = ["guard", "out_of_range"]

logger = log.get_logger("compat.ranges")

#: Share of a ``FLOAT`` range's bound a value may pass it by, for rounding upstream.
FLOAT_SLACK = 1e-9

_lock = threading.Lock()
_ranges: dict[type, dict] = {}


def _collected(inputs, found: dict) -> dict:
    """Each number input's ``(min, max, is_float)`` by id, a DynamicCombo's nested under its id."""
    from comfy_api.latest import io

    for item in inputs or ():
        if isinstance(item, (io.Int.Input, io.Float.Input)):
            if item.min is not None or item.max is not None:
                found[item.id] = (item.min, item.max, isinstance(item, io.Float.Input))
        elif isinstance(item, io.DynamicCombo.Input):
            nested: dict = {}
            for option in item.options:
                _collected(option.inputs, nested)
            if nested:
                found[item.id] = nested
    return found


def _declared(node_class) -> dict:
    """The declared number ranges of one node class, read from its schema once."""
    with _lock:
        held = _ranges.get(node_class)
    if held is None:
        held = _collected(node_class.define_schema().inputs, {})
        with _lock:
            _ranges[node_class] = held
    return held


def out_of_range(ranges: dict, values: dict) -> tuple[str, object, object, object] | None:
    """The first value outside its declared range.

    Args:
        ranges: Declared ranges, as :func:`_collected` builds them.
        values: The inputs as ``execute`` receives them.

    Returns:
        ``(input id, value, min, max)``, or ``None`` when every value is in range.
    """
    for name, declared in ranges.items():
        value = values.get(name)
        if isinstance(declared, dict):
            if isinstance(value, dict):
                found = out_of_range(declared, value)
                if found is not None:
                    return found
            continue
        low, high, floating = declared
        for one in value if isinstance(value, list) else (value,):
            if isinstance(one, bool) or not isinstance(one, (int, float)):
                continue
            if floating and isinstance(one, float) and not math.isfinite(one):
                return name, one, low, high
            under = low is not None and one < low - (FLOAT_SLACK * max(1.0, abs(low)) if floating else 0)
            over = high is not None and one > high + (FLOAT_SLACK * max(1.0, abs(high)) if floating else 0)
            if under or over:
                return name, one, low, high
    return None


def _refuse(node_class, values: dict) -> None:
    """Raise when one of a node's number inputs is outside its declared range."""
    try:
        ranges = _declared(node_class)
    except Exception as error:
        logger.debug("the input ranges of %s could not be read (%s)", node_class.__name__, error)
        return
    found = out_of_range(ranges, values)
    if found is None:
        return
    name, value, low, high = found
    raise ValueError(
        f"`{name}` received {value}, outside its range of {low} to {high}. Bring the value "
        f"feeding `{name}` into that range."
    )


def guard(node_class):
    """Refuse number inputs outside their declared range before a node's ``execute`` runs.

    Args:
        node_class: A ``ComfyNode`` subclass.

    Returns:
        The same class, its ``execute`` wrapped.
    """
    method = getattr(node_class, "execute", None)
    original = getattr(method, "__func__", None)
    if original is None or getattr(original, "_was_ranges", False):
        return node_class

    if inspect.iscoroutinefunction(original):
        @functools.wraps(original)
        async def execute(cls, *args, **kwargs):
            _refuse(node_class, kwargs)
            return await original(cls, *args, **kwargs)
    else:
        @functools.wraps(original)
        def execute(cls, *args, **kwargs):
            _refuse(node_class, kwargs)
            return original(cls, *args, **kwargs)

    execute._was_ranges = True
    node_class.execute = classmethod(execute)
    return node_class

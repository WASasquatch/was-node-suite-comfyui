"""Reading a Power LoRA node's slots out of the values it is sent.

Each slot arrives as ``lora_<n>``, ``lora_<n>_enabled`` and ``lora_<n>_weight``, numbered
from 1 and declared by the node, so an API call fills them as readily as the editor.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

__all__ = ["NAME_KEY", "Row", "first_weight", "rows_from_inputs", "to_bool", "weight_list"]

#: ``lora_3_enabled`` -> slot 3. Matched on the whole key, lowercased.
ENABLED_KEY = re.compile(r"lora_(\d+)_enabled")

#: ``lora_3_weight`` -> slot 3.
WEIGHT_KEY = re.compile(r"lora_(\d+)_weight")

#: ``lora_3`` -> slot 3. The file name widget itself.
NAME_KEY = re.compile(r"lora_(\d+)")

#: Strings a checkbox may arrive as, from a widget value that made a round trip through
#: JSON or through a text field.
TRUE_WORDS = ("true", "1", "yes", "y", "on")
FALSE_WORDS = ("false", "0", "no", "n", "off", "")


class Row:
    """One LoRA slot: which file, how strongly, and whether it is switched on.

    Attributes:
        on: Whether the slot's checkbox is ticked. A slot that is off is kept in the list
            so the numbering is unchanged, and dropped by :func:`rows_from_inputs`.
        lora: File name as it appears in the LoRA folder, a full path wired in, or ``None``
            for an empty slot.
        weight: Multiplier applied to this LoRA's contribution, or a list of them, one per
            sampling step.
    """

    __slots__ = ("on", "lora", "weight")

    def __init__(self, on: bool = True, lora: str | None = None, weight: float | list[float] = 1.0):
        self.on = on
        self.lora = lora
        self.weight = weight

    def __repr__(self) -> str:
        return f"Row(on={self.on!r}, lora={self.lora!r}, weight={self.weight!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Row):
            return NotImplemented
        return (self.on, self.lora, self.weight) == (other.on, other.lora, other.weight)


def to_bool(value: Any, default: bool = True) -> bool:
    """Read a checkbox value that may have arrived as a bool, a number or a word.

    Args:
        value: The value as it came off the wire.
        default: Answer for ``None`` and for a word that is neither true nor false.

    Returns:
        The boolean the value stands for.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(int(value))
    if isinstance(value, str):
        word = value.strip().lower()
        if word in TRUE_WORDS:
            return True
        if word in FALSE_WORDS:
            return False
        return default
    return bool(value)


def _widget_rows(values: Mapping[str, Any]) -> list[Row]:
    """Rows from the per-widget keys, ordered by slot number."""
    enabled: dict[int, bool] = {}
    names: dict[int, str | None] = {}
    weights: dict[int, float | list[float]] = {}

    for key, value in values.items():
        if not isinstance(key, str):
            continue
        name = key.lower()
        if not name.startswith("lora_"):
            continue

        match = ENABLED_KEY.fullmatch(name)
        if match:
            enabled[int(match.group(1))] = to_bool(value, default=True)
            continue

        match = WEIGHT_KEY.fullmatch(name)
        if match:
            slot = int(match.group(1))
            if isinstance(value, (list, tuple)):
                weights[slot] = _listed_weight(slot, value)
                continue
            try:
                weights[slot] = float(value)
            except (TypeError, ValueError):
                weights[slot] = 1.0
            continue

        match = NAME_KEY.fullmatch(name)
        if match:
            usable = isinstance(value, str) and value and value != "None"
            names[int(match.group(1))] = value if usable else None

    rows = []
    for slot in sorted(set(enabled) | set(names) | set(weights)):
        rows.append(
            Row(
                on=to_bool(enabled.get(slot, True), default=True),
                lora=names.get(slot, None),
                weight=weights.get(slot, 1.0),
            )
        )
    return rows


def rows_from_inputs(values: Mapping[str, Any]) -> list[Row]:
    """Return the LoRA slots that will take part in a merge.

    Args:
        values: Every extra keyword argument the node was called with. Keys that are not
            a LoRA slot are ignored, so anything else the node declares costs nothing.

    Returns:
        One :class:`Row` per slot that is switched on, names a file and carries a non-zero
        weight, in slot order. A slot that fails any of those three is left out, which is
        how a row is muted without deleting it. A full path is given as its name inside the
        LoRA folder.

    Raises:
        PathNotAllowed: A row that takes part names a path outside the LoRA folders.
    """
    from ..io import picker

    rows = [
        row for row in _widget_rows(values)
        if row.on and row.lora and row.lora != "None" and any(value != 0.0 for value in weight_list(row.weight))
    ]
    for row in rows:
        row.lora = picker.model_name(row.lora, ("loras",))
    return rows


def _listed_weight(slot: int, value) -> float | list[float]:
    """A row's weight from a list: the number itself when the list holds one.

    Args:
        slot: The row's number, for the message.
        value: The list as it arrived.

    Returns:
        A number, or a list of at least two numbers.

    Raises:
        ValueError: The list is empty or holds an entry that is not a number.
    """
    from ..util.numbers import as_number

    numbers = []
    for position, entry in enumerate(value, 1):
        number = as_number(entry)
        if number is None:
            raise ValueError(
                f"Row {slot}'s weight list holds {str(entry)[:40]!r} at entry {position}, which is "
                f"not a number. Every entry has to be a strength, as 0.8."
            )
        numbers.append(number)
    if not numbers:
        raise ValueError(
            f"Row {slot}'s weight received an empty list. Check the node wired into lora_{slot}_weight, "
            f"or disconnect it to type one weight."
        )
    return numbers[0] if len(numbers) == 1 else numbers


def weight_list(weight: float | list[float]) -> list[float]:
    """A row's weight as a list, one entry for a single number.

    Args:
        weight: A row's weight.

    Returns:
        The weights in order.
    """
    return list(weight) if isinstance(weight, list) else [float(weight)]


def first_weight(weight: float | list[float]) -> float:
    """The weight a row starts at.

    Args:
        weight: A row's weight.

    Returns:
        The number, or the first entry of a list.
    """
    return weight_list(weight)[0]

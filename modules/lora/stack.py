"""LoRA stacks: lists of ``(lora_name, model_strength, clip_strength)``, written and read.

Names are relative to the LoRA folder. A stack arrives as a list, or as a Power LoRA node's
MODEL link, its rows read from the prompt.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Mapping

from .rows import NAME_KEY, WEIGHT_KEY, Row, first_weight, rows_from_inputs, to_bool

__all__ = ["Entry", "consumed", "entries", "from_rows", "resolve"]

#: One stacked LoRA: its name in the LoRA folder, its model strength and its clip strength.
Entry = tuple[str, float, float]


def from_rows(rows: Iterable[Row], strength: float = 1.0) -> list[Entry]:
    """A stack from Power LoRA rows, each at its weight times ``strength`` on model and clip.

    Args:
        rows: Switched-on rows, as :func:`~modules.lora.rows.rows_from_inputs` returns them.
            A row carrying a list of weights counts at its first.
        strength: Multiplier over every row's weight.

    Returns:
        One entry per row whose weight stays above zero, in row order.
    """
    stack = []
    for row in rows:
        weight = first_weight(row.weight) * float(strength)
        if weight != 0.0:
            stack.append((row.lora, weight, weight))
    return stack


def _number(value: Any, default: float) -> float:
    """A strength from a prompt value, or ``default`` for a link or anything else unreadable."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _rgthree_rows(inputs: Mapping[str, Any]) -> list[Entry] | None:
    """The switched-on rows of an rgthree Power Lora Loader, or None where the inputs hold none."""
    rows = [value for key, value in inputs.items()
            if str(key).lower().startswith("lora_") and isinstance(value, dict) and "lora" in value]
    if not rows:
        return None
    stack = []
    for row in rows:
        name = row.get("lora")
        if not to_bool(row.get("on", True)) or not isinstance(name, str) or name in ("", "None"):
            continue
        model = _number(row.get("strength"), 1.0)
        clip = _number(row.get("strengthTwo"), model)
        if model != 0.0 or clip != 0.0:
            stack.append((name, model, clip))
    return stack


def _power_lora_rows(inputs: Mapping[str, Any]) -> list[Entry] | None:
    """The switched-on rows of a Power LoRA Loader or Merger, or None where the inputs hold none."""
    if not any(NAME_KEY.fullmatch(str(key).lower()) and isinstance(value, str)
               for key, value in inputs.items()):
        return None
    for key, value in inputs.items():
        match = WEIGHT_KEY.fullmatch(str(key).lower())
        wired = isinstance(value, (list, tuple)) and len(value) == 2 and isinstance(value[0], str)
        if (match or str(key).lower() == "strength") and wired:
            raise ValueError(
                f"The Power LoRA node read here takes {key} from a wire, and a wire's value cannot "
                f"be read from the prompt. Wire that node's lora_stack output into stack_data "
                f"instead."
            )
    return from_rows(rows_from_inputs(inputs), _number(inputs.get("strength", 1.0), 1.0))


def _node_inputs(prompt: Any, node_id: Any) -> Mapping[str, Any]:
    """One node's inputs out of the prompt, or an empty mapping."""
    node = prompt.get(str(node_id)) if isinstance(prompt, Mapping) else None
    inputs = node.get("inputs") if isinstance(node, Mapping) else None
    return inputs if isinstance(inputs, Mapping) else {}


def entries(value: Any, prompt: Any, unique_id: Any, input_name: str = "stack_data") -> list[Entry] | None:
    """The LoRAs arriving on a stack input.

    Args:
        value: What the input received: a stack, the MODEL of a Power LoRA node, or ``None``.
        prompt: The prompt the run was queued with.
        unique_id: The id of the node reading the input.
        input_name: The input's name on that node.

    Returns:
        The stack in its order, an empty list for a linked input holding nothing, or ``None``
        where the input is not linked.

    Raises:
        ValueError: The input is linked to a node with no LoRA rows to read.
    """
    if isinstance(value, (list, tuple)):
        stack = []
        for item in value:
            if not isinstance(item, (list, tuple)) or len(item) < 2 or not isinstance(item[0], str):
                continue
            model = _number(item[1], 1.0)
            stack.append((item[0], model, _number(item[2], model) if len(item) > 2 else model))
        return stack

    link = _node_inputs(prompt, unique_id).get(input_name)
    if not (isinstance(link, (list, tuple)) and len(link) == 2):
        return None
    source = _node_inputs(prompt, link[0])
    for reader in (_power_lora_rows, _rgthree_rows):
        stack = reader(source)
        if stack is not None:
            return stack
    if value is None:
        return []
    node = prompt.get(str(link[0])) if isinstance(prompt, Mapping) else None
    kind = node.get("class_type", "the linked node") if isinstance(node, Mapping) else "the linked node"
    raise ValueError(
        f"{input_name} is linked to {kind}, which has no LoRA rows. Link a lora_stack output, or "
        f"the model output of a Power LoRA Loader."
    )


def consumed(prompt: Any, unique_id: Any, slot: int, ignore: Iterable[str] = ("stack_data",)) -> bool:
    """Whether any node in the prompt reads one output of a node as a value.

    Args:
        prompt: The prompt the run was queued with.
        unique_id: The id of the node whose output is asked about.
        slot: The output's index.
        ignore: Input names whose links read the node's rows rather than the output's value.

    Returns:
        True when a link in the prompt starts at that output and ends on an input not ignored.
    """
    if not isinstance(prompt, Mapping):
        return False
    skipped = set(ignore)
    for node in prompt.values():
        inputs = node.get("inputs") if isinstance(node, Mapping) else None
        for name, value in (inputs or {}).items():
            if (name not in skipped and isinstance(value, (list, tuple)) and len(value) == 2
                    and str(value[0]) == str(unique_id) and value[1] == slot):
                return True
    return False


def resolve(name: str) -> str | None:
    """The full path of a LoRA named relative to the LoRA folder, found by file name if it moved.

    Args:
        name: The name as a stack carries it.

    Returns:
        The path, or ``None`` where no LoRA of that name is on disk.
    """
    import folder_paths

    if not isinstance(name, str) or name in ("", "None"):
        return None
    found = folder_paths.get_full_path("loras", name)
    if found is not None:
        return found
    wanted = os.path.basename(name.replace("\\", "/")).lower()
    for candidate in folder_paths.get_filename_list("loras"):
        if os.path.basename(candidate.replace("\\", "/")).lower() == wanted:
            return folder_paths.get_full_path("loras", candidate)
    return None

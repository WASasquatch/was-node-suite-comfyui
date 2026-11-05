"""Fusing several LoRAs into one state dict.

:func:`fuse_state_dicts` joins plain up and down factors along the rank axis, exactly. A layer
in another form, such as a LoKr, stays with its own LoRA.
"""

from __future__ import annotations

import torch

from ..log import get_logger

__all__ = ["DOWN_SUFFIXES", "UP_SUFFIXES", "fuse_state_dicts"]

logger = get_logger("lora.fuse")

#: Suffixes naming the two factors of a plain LoRA, in the spellings that reach a state dict.
DOWN_SUFFIXES = (".lora_down.weight", ".lora_A.weight", ".lora_down")
UP_SUFFIXES = (".lora_up.weight", ".lora_B.weight", ".lora_up")

#: Parts a layer may hold and still be joined.
PLAIN_PARTS = frozenset({"down", "up", "alpha"})


def _module_of(key: str) -> tuple[str, str] | None:
    """Split a factor key into the module it belongs to and which factor it is.

    Args:
        key: A state dict key.

    Returns:
        ``(module, "down")`` or ``(module, "up")``, and None for a key that is neither.
    """
    for suffix in DOWN_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], "down"
    for suffix in UP_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], "up"
    return None


def _part_of(key: str) -> tuple[str, str]:
    """Split any key into its module and the part it holds.

    Args:
        key: A state dict key.

    Returns:
        ``(module, part)``: ``down`` and ``up`` for plain factors, ``alpha``, or the key's last
        name, as ``lokr_w1``, ``dora_scale`` or ``lora_mid``.
    """
    split = _module_of(key)
    if split is not None:
        return split
    head, _, tail = key.rpartition(".")
    if tail == "weight" and "." in head:
        module, _, inner = head.rpartition(".")
        if inner.startswith("lora_"):
            return module, inner
    return head, tail


def _scale_of(state: dict, module: str, rank: int) -> float:
    """The multiplier a module's alpha asks for.

    Args:
        state: The LoRA's state dict.
        module: Module prefix, as :func:`_module_of` answers it.
        rank: Rank of that module's factors.

    Returns:
        ``alpha / rank``, and 1.0 where no alpha is recorded, which is how a LoRA saved
        without one is already scaled.
    """
    alpha = state.get(f"{module}.alpha")
    if alpha is None or not rank:
        return 1.0
    try:
        return float(alpha.item() if hasattr(alpha, "item") else alpha) / float(rank)
    except (TypeError, ValueError):
        return 1.0


def fuse_state_dicts(loaded: list[tuple[dict, float]]) -> tuple[dict, list[tuple[int, dict]], int]:
    """Combine several weighted LoRA state dicts into one, keeping apart what cannot be joined.

    Args:
        loaded: ``(state_dict, weight)`` per LoRA, in the order they were listed.

    Returns:
        ``(fused, apart, paired)``: the joined state dict, applied at strength 1.0 since every
        weight is folded into it; ``(index, state_dict)`` per LoRA holding layers that were not
        joined, applied at ``loaded[index]``'s weight; and how many modules were joined.
    """
    plain: dict[str, list[tuple[int, dict[str, str]]]] = {}
    apart: dict[int, dict] = {}

    for index, (state, _weight) in enumerate(loaded):
        modules: dict[str, dict[str, str]] = {}
        for key in state:
            module, part = _part_of(key)
            modules.setdefault(module, {})[part] = key
        for module, parts in modules.items():
            if set(parts) <= PLAIN_PARTS and "down" in parts and "up" in parts:
                plain.setdefault(module, []).append((index, parts))
            else:
                held = apart.setdefault(index, {})
                for key in parts.values():
                    held[key] = state[key]

    fused: dict[str, torch.Tensor] = {}
    paired = 0
    for module, sources in plain.items():
        down_parts, up_parts = [], []
        for index, parts in sources:
            state, weight = loaded[index]
            down, up = state[parts["down"]], state[parts["up"]]
            scale = _scale_of(state, module, int(down.shape[0])) * float(weight)
            down_parts.append(down.to(torch.float32))
            up_parts.append(up.to(torch.float32) * scale)
        try:
            down_cat = torch.cat(down_parts, dim=0)
            up_cat = torch.cat(up_parts, dim=1)
        except RuntimeError as error:
            logger.debug("%s could not be joined (%s), each LoRA applies it apart", module, error)
            for index, parts in sources:
                held = apart.setdefault(index, {})
                for key in parts.values():
                    held[key] = loaded[index][0][key]
            continue

        # Kept in float32: the joined factors carry every row's contribution.
        fused[f"{module}.lora_down.weight"] = down_cat
        fused[f"{module}.lora_up.weight"] = up_cat
        # Every scale is folded into up, so an alpha equal to the joined rank leaves it as summed.
        fused[f"{module}.alpha"] = torch.tensor(float(down_cat.shape[0]), dtype=torch.float32)
        paired += 1

    return fused, sorted(apart.items()), paired

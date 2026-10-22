"""Apply a LoRA through ComfyUI, leaving out the parts whose shape does not fit the weight they patch."""

from __future__ import annotations

import math

from ..log import get_logger

logger = get_logger("lora.fit")

__all__ = ["fits", "load_lora_for_models"]

#: Input width of MiniMax H3's time projection, unpruned and in the pruned curve form.
H3_FULL_TIME = 2688
H3_CURVE_TIME = 8


def _patch_shape(patch):
    """The shape a patch's change takes, or None where it cannot be told without computing it."""
    weights = getattr(patch, "weights", None)
    if getattr(patch, "name", None) == "lora" and isinstance(weights, (tuple, list)) and len(weights) >= 2:
        up, down = weights[0], weights[1]
        mid = weights[3] if len(weights) > 3 else None
        reshape = weights[5] if len(weights) > 5 else None
        if mid is not None or reshape is not None or not hasattr(up, "shape") or not hasattr(down, "shape"):
            return None
        return (int(up.shape[0]), int(math.prod(down.shape[1:])))
    if isinstance(patch, tuple) and len(patch) == 2 and patch[0] == "diff":
        tensor = patch[1][0] if patch[1] else None
        return tuple(tensor.shape) if hasattr(tensor, "shape") else None
    return None


def _target_shape(patcher, key: str, offset):
    """The shape of the weight a patch lands on, narrowed by its offset, or None if absent."""
    import comfy.utils

    try:
        weight = comfy.utils.get_attr(patcher.model, key)
    except AttributeError:
        return None
    shape = list(getattr(weight, "shape", ()))
    if not shape:
        return None
    if offset is not None:
        shape[offset[0]] = offset[2]
    return tuple(shape)


def fits(patcher, key: str, patch, offset=None) -> bool:
    """Whether a patch's change has as many values as the weight it lands on.

    Args:
        patcher: The ``ModelPatcher`` holding the weight.
        key: The weight's state dict key.
        patch: The patch, a LoRA adapter or a ``("diff", (tensor,))`` pair.
        offset: The ``(dim, start, length)`` slice the patch lands on, or None.

    Returns:
        False only where both shapes are known and disagree; any other patch is kept.
    """
    change, target = _patch_shape(patch), _target_shape(patcher, key, offset)
    if change is None or target is None:
        return True
    if isinstance(patch, tuple):
        return tuple(change) == tuple(target)
    return math.prod(change) == math.prod(target)


def _drop(before, after) -> list[tuple[str, tuple, tuple]]:
    """Remove the patches ``after`` gained over ``before`` that do not fit, and list them."""
    dropped = []
    for key in list(after.patches):
        entries = after.patches[key]
        start = len(before.patches.get(key, ())) if before is not None else 0
        kept = entries[:start]
        for entry in entries[start:]:
            patch, offset = entry[1], entry[3] if len(entry) > 3 else None
            if fits(after, key, patch, offset):
                kept.append(entry)
            else:
                dropped.append((key, _patch_shape(patch), _target_shape(after, key, offset)))
        if kept:
            after.patches[key] = kept
        else:
            del after.patches[key]
    return dropped


def _explain(name: str, dropped: list) -> str:
    """One line naming what was left out of a LoRA, and why."""
    key, change, target = dropped[0]
    time = [d for d in dropped if d[0].endswith("adaln_proj.linear.weight")]
    if len(time) == len(dropped) and change and target and change[0] == target[0]:
        if change[1] == H3_CURVE_TIME and target[1] == H3_FULL_TIME:
            return (
                f"{name} was trained on a pruned MiniMax H3, whose time projection is "
                f"{H3_CURVE_TIME} wide, and this transformer is unpruned ({H3_FULL_TIME} wide), so "
                f"its {len(time)} time modulation parts are left out and the rest applies. A pruned "
                f"transformer takes all of it.")
        if change[1] == H3_FULL_TIME and target[1] == H3_CURVE_TIME:
            return (
                f"{name} was trained on an unpruned MiniMax H3, whose time projection is "
                f"{H3_FULL_TIME} wide, and this transformer is pruned ({H3_CURVE_TIME} wide), so "
                f"its {len(time)} time modulation parts are left out and the rest applies. An "
                f"unpruned transformer takes all of it.")
    return (
        f"{name}: {len(dropped)} part(s) do not fit this model and are left out, the first "
        f"{key} with a {list(change)} change on a {list(target)} weight. The LoRA was likely "
        f"trained on another model or another form of this one.")


def load_lora_for_models(model, clip, state: dict, strength_model: float, strength_clip: float,
                         name: str = "LoRA"):
    """``comfy.sd.load_lora_for_models``, without the parts that cannot fit.

    Args:
        model: The ``ModelPatcher`` to patch, or None.
        clip: The text encoder to patch, or None.
        state: The LoRA's state dict.
        strength_model: Strength on the model.
        strength_clip: Strength on the text encoder.
        name: The LoRA's name in the message about what was left out.

    Returns:
        The patched model and text encoder, as ComfyUI returns them.
    """
    import comfy.sd

    new_model, new_clip = comfy.sd.load_lora_for_models(model, clip, state, strength_model, strength_clip)
    dropped = []
    if new_model is not None and new_model is not model:
        dropped += _drop(model, new_model)
    if new_clip is not None and new_clip is not clip:
        dropped += _drop(getattr(clip, "patcher", None), new_clip.patcher)
    if dropped:
        logger.warning(_explain(name, dropped))
    return new_model, new_clip

"""Scaling a LoRA's blocks before it is applied.

A module's key names where in the model it sits, ``lora_unet_blocks_12_...`` is block 12.
:func:`reweight_state_dict` returns a new state dict and a count of what it changed.
"""

from __future__ import annotations

import re
from pathlib import PurePath
from typing import Dict, Optional, Tuple

import torch

__all__ = [
    "compute_block_scale",
    "detect_total_blocks_from_lora",
    "detect_total_blocks_from_model",
    "infer_preset",
    "lora_part",
    "parse_block_index",
    "reweight_state_dict",
    "suggest_filename",
]

#: Any ``…blocks.12…``, ``…layers_3…`` or ``…h.5…`` segment, for a file whose family is
#: not one of the known ones.
BLOCK_ANY = re.compile(
    r"(?:^|[._])(?:transformer(?:[._])?)?(?:blocks?|layers?|stages?|resblocks?|h)[._](\d+)(?:[._]|$)",
    re.IGNORECASE,
)

BLOCK_QWEN = re.compile(r"^transformer_blocks\.(\d+)\.", re.IGNORECASE)
BLOCK_WAN = re.compile(r"^lora_unet_blocks_(\d+)_", re.IGNORECASE)
BLOCK_FLUX = re.compile(r"^lora_unet_(?:double|single)_blocks_(\d+)_", re.IGNORECASE)
BLOCK_ZIMG_TURBO = re.compile(r"^diffusion_model\.layers\.(\d+)\.", re.IGNORECASE)
BLOCK_SD_UNET = re.compile(r"^lora_unet_(?:down|up)_blocks_(\d+)_", re.IGNORECASE)
BLOCK_SD_TEXT_ENCODER = re.compile(r"^lora_te_.*?_encoder_layers_(\d+)_", re.IGNORECASE)
BLOCK_SDXL_UNET = re.compile(r"^lora_unet_(?:input|output)_blocks_(\d+)_", re.IGNORECASE)
BLOCK_SDXL_MID = re.compile(r"^lora_unet_middle_block_", re.IGNORECASE)
BLOCK_SDXL_TEXT_ENCODER_1 = re.compile(r"^lora_te1_.*?_encoder_layers_(\d+)_", re.IGNORECASE)
BLOCK_SDXL_TEXT_ENCODER_2 = re.compile(r"^lora_te2_.*?_encoder_layers_(\d+)_", re.IGNORECASE)

#: Key suffixes naming the two halves of a LoRA pair, in every spelling in circulation, and
#: the first and second factors of a LoKr or LoHa, each of which scales its change linearly.
UP_SUFFIXES = (
    "lora.up.weight", "lora_up.weight", "lorab.weight", "lora_b.weight",
    ".lokr_w1", ".lokr_w1_a", ".hada_w1_a",
)
DOWN_SUFFIXES = (
    "lora.down.weight", "lora_down.weight", "loraa.weight", "lora_a.weight",
    ".lokr_w2", ".lokr_w2_a", ".hada_w2_a",
)


def infer_preset(keys) -> str:
    """Guess which model family a LoRA was trained against, from its key names.

    Args:
        keys: The state dict's keys, in any order.

    Returns:
        One of ``zimg-turbo``, ``flux``, ``wan``, ``qwen``, ``sdxl``, ``sd``, or
        ``generic`` where no family's prefix is recognised.
    """
    for key in keys:
        if not isinstance(key, str):
            continue
        if key.startswith("diffusion_model.layers."):
            return "zimg-turbo"
        if key.startswith("lora_unet_double_blocks_") or key.startswith("lora_unet_single_blocks_"):
            return "flux"
        if key.startswith("lora_unet_blocks_"):
            return "wan"
        if key.startswith("transformer_blocks."):
            return "qwen"
        if key.startswith("lora_te1_") or key.startswith("lora_te2_"):
            return "sdxl"
        if key.startswith("lora_unet_"):
            return "sd"
    return "generic"


def parse_block_index(key: str, preset: str) -> Optional[int]:
    """Read the block number out of one key.

    Args:
        key: A state dict key, such as ``lora_unet_blocks_12_self_attn_q_lora_up.weight``.
        preset: The naming scheme to read it under, from :func:`infer_preset`.

    Returns:
        The block number, or ``None`` for a key that names no block, the SDXL middle
        block, an embedding, or anything the scheme does not cover. Such a key is scaled
        by the global factor alone.
    """
    family = (preset or "generic").lower().replace("_", "-")
    try:
        if family == "qwen":
            found = BLOCK_QWEN.search(key)
            return int(found.group(1)) if found else None
        if family == "wan":
            found = BLOCK_WAN.search(key)
            return int(found.group(1)) if found else None
        if family == "flux":
            found = BLOCK_FLUX.search(key)
            return int(found.group(1)) if found else None
        if family == "zimg-turbo":
            found = BLOCK_ZIMG_TURBO.search(key)
            return int(found.group(1)) if found else None
        if family == "sdxl":
            for pattern in (BLOCK_SDXL_TEXT_ENCODER_1, BLOCK_SDXL_TEXT_ENCODER_2, BLOCK_SDXL_UNET):
                found = pattern.search(key)
                if found:
                    return int(found.group(1))
            return None
        if family == "sd":
            found = BLOCK_SD_TEXT_ENCODER.search(key)
            if found:
                return int(found.group(1))
            found = BLOCK_SD_UNET.search(key)
            return int(found.group(1)) if found else None

        found = BLOCK_ANY.search(key)
        return int(found.group(1)) if found else None
    except (AttributeError, TypeError, ValueError):
        return None


def detect_total_blocks_from_model(model) -> int:
    """Count the blocks in the model a LoRA is about to be applied to.

    Args:
        model: A ComfyUI MODEL. Anything without an inner torch module reports 0.

    Returns:
        The highest block number found in the module names, plus one, or 0 where none was
        found. This is the count the front/middle/back split is measured against, so a
        LoRA holding more blocks than the model has can be cut down to fit.
    """
    inner = getattr(model, "model", None)
    if inner is None:
        return 0
    highest = -1
    try:
        for name, _ in inner.named_modules():
            found = BLOCK_ANY.search(name)
            if found:
                index = int(found.group(1))
                if index > highest:
                    highest = index
    except (AttributeError, TypeError, ValueError):
        return 0
    return (highest + 1) if highest >= 0 else 0


def detect_total_blocks_from_lora(state: Dict[str, torch.Tensor], preset: str) -> int:
    """Count the blocks the LoRA itself covers.

    Args:
        state: The LoRA's state dict.
        preset: Naming scheme, from :func:`infer_preset`.

    Returns:
        The highest block number in the keys, plus one, or 0 where none was found. Used
        when the model reports no blocks of its own.
    """
    highest = -1
    for key in state.keys():
        index = parse_block_index(key, preset)
        if index is not None and index > highest:
            highest = index
    return (highest + 1) if highest >= 0 else 0


# Scales a LoRA by block position, with a multiplier of its own for the final block.
def compute_block_scale(
    index: Optional[int],
    total_blocks: int,
    global_scale: float,
    front: float,
    mid: float,
    back: float,
    last: float,
) -> float:
    """Work out the multiplier for one block.

    Args:
        index: Block number, or ``None`` for a key that names no block.
        total_blocks: How many blocks the split is measured over. 0 disables the split.
        global_scale: Multiplier applied to every block.
        front: Extra multiplier for the first third.
        mid: Extra multiplier for the middle third.
        back: Extra multiplier for the last third.
        last: Extra multiplier for the final block, on top of its third's.

    Returns:
        The multiplier this block's tensors are scaled by.
    """
    if total_blocks <= 0 or index is None:
        return global_scale
    third = max(1, total_blocks // 3)
    if index < third:
        scale = global_scale * front
    elif index < 2 * third:
        scale = global_scale * mid
    else:
        scale = global_scale * back
    if index == (total_blocks - 1):
        scale *= last
    return scale


def lora_part(key: str) -> Optional[str]:
    """Say which half of a LoRA pair a key holds, the first or second factor of a LoKr or LoHa.

    Args:
        key: A state dict key.

    Returns:
        ``"up"`` for an up half or first factor, ``"down"`` for a down half or second factor,
        or ``None`` for anything else: an alpha, a magnitude vector, a factor's other half.
    """
    lowered = key.lower()
    if lowered.endswith(UP_SUFFIXES):
        return "up"
    if lowered.endswith(DOWN_SUFFIXES):
        return "down"
    return None


def reweight_state_dict(
    state: Dict[str, torch.Tensor],
    total_blocks: int,
    global_scale: float,
    front: float,
    mid: float,
    back: float,
    last: float,
    scale_target: str,
    filter_by_block_range: bool,
    filter_cutoff_blocks: int,
    preset: str,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, int]]:
    """Scale a LoRA's tensors by block position, dropping blocks the model does not have.

    Args:
        state: The LoRA's state dict. Not modified; a new dict is returned.
        total_blocks: Block count the front/middle/back split is measured over.
        global_scale: Multiplier applied to every block.
        front: Extra multiplier for the first third.
        mid: Extra multiplier for the middle third.
        back: Extra multiplier for the last third.
        last: Extra multiplier for the final block.
        scale_target: ``"up_only"``, ``"down_only"`` or ``"both"``, which half of each
            pair is scaled.
        filter_by_block_range: Whether to drop keys whose block is at or beyond
            ``filter_cutoff_blocks``.
        filter_cutoff_blocks: First block number to drop. 0 drops nothing.
        preset: Naming scheme, from :func:`infer_preset`.

    Returns:
        ``(state, counts)``, the new state dict, and how many tensors were ``changed``,
        ``dropped`` and ``kept``.
    """
    result: Dict[str, torch.Tensor] = {}
    changed = 0
    dropped = 0
    kept = 0
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            result[key] = value
            continue
        index = parse_block_index(key, preset)
        if filter_by_block_range and filter_cutoff_blocks > 0 and index is not None and index >= filter_cutoff_blocks:
            dropped += 1
            continue
        part = lora_part(key)
        scale_this = (
            (scale_target == "up_only" and part == "up")
            or (scale_target == "down_only" and part == "down")
            or (scale_target == "both" and (part == "up" or part == "down"))
        )
        if scale_this:
            scale = compute_block_scale(index, total_blocks, global_scale, front, mid, back, last)
            if scale != 1.0:
                value = value * scale
                changed += 1
        result[key] = value
        kept += 1
    return result, {"changed": changed, "dropped": dropped, "kept": kept}


def suggest_filename(
    source_name: str,
    scale_target: str,
    global_scale: float,
    front: float,
    mid: float,
    back: float,
    last: float,
) -> str:
    """Build a file name recording the settings the reweighted LoRA was made with.

    Args:
        source_name: File name of the LoRA that was reweighted.
        scale_target: Which half of each pair was scaled.
        global_scale: Multiplier applied to every block.
        front: Extra multiplier for the first third.
        mid: Extra multiplier for the middle third.
        back: Extra multiplier for the last third.
        last: Extra multiplier for the final block.

    Returns:
        A name such as ``style.reweighted.up_only.g1.00.f1.0.m1.2.b0.8.L1.0.safetensors``,
        so two runs with different settings never overwrite each other.
    """
    stem = PurePath(source_name).stem
    return (
        f"{stem}.reweighted.{scale_target}.g{global_scale:.2f}"
        f".f{round(front, 2)}.m{round(mid, 2)}.b{round(back, 2)}.L{round(last, 2)}.safetensors"
    )

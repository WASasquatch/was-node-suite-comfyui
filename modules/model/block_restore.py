"""A range of a transformer's blocks put back to another checkpoint of the same model.

Each restored weight is ``(1 - strength) * own + strength * source``, ahead of the model's other
patches. Quantized sources are dequantized.
"""

from __future__ import annotations

import os

import torch

from .. import log

__all__ = ["BLOCK_LISTS", "FOLDER", "block_list", "checkpoints", "restore"]

logger = log.get_logger("model.block_restore")

#: Folder the source checkpoints are picked from.
FOLDER = "diffusion_models"

#: Attribute names a transformer's block list goes by.
BLOCK_LISTS = ("layers", "blocks", "transformer_blocks", "joint_blocks")

#: Prefixes a checkpoint may put in front of the transformer's own key names.
_PREFIXES = ("", "model.diffusion_model.", "diffusion_model.")

#: File extensions read as safetensors.
_SAFETENSORS = (".safetensors", ".sft")


def checkpoints() -> list[str]:
    """Every safetensors file ComfyUI's diffusion model folders offer.

    Returns:
        File names as the folders list them; empty outside ComfyUI.
    """
    try:
        import folder_paths

        return [name for name in folder_paths.get_filename_list(FOLDER)
                if name.lower().endswith(_SAFETENSORS)]
    except Exception as error:
        logger.debug("the %s folder could not be read: %s", FOLDER, error)
        return []


def block_list(model) -> tuple[str, torch.nn.ModuleList]:
    """The transformer's block list.

    Args:
        model: A model patcher.

    Returns:
        ``(name, blocks)``.

    Raises:
        ValueError: The transformer holds no block list, or more than one.
    """
    transformer = model.get_model_object("diffusion_model")
    found = [(name, getattr(transformer, name)) for name in BLOCK_LISTS
             if isinstance(getattr(transformer, name, None), torch.nn.ModuleList)]
    if len(found) != 1:
        named = ", ".join(name for name, _ in found) or "none"
        raise ValueError(
            f"Model Block Restore works on a transformer with one list of blocks, such as "
            f"MOSS-Image's layers, and {type(transformer).__name__} has {named}."
        )
    return found[0]


def _path(name: str) -> str:
    """The full path of a checkpoint picked from :func:`checkpoints`."""
    import folder_paths

    if not str(name).lower().endswith(_SAFETENSORS):
        raise ValueError(
            f"{name!r} is not a .safetensors file. Pick the source checkpoint from the menu, such "
            f"as moss_image_base_int8_convrot.safetensors."
        )
    path = folder_paths.get_full_path(FOLDER, name)
    if not path or not os.path.isfile(path):
        raise ValueError(
            f"ComfyUI's {FOLDER} folder holds no {name!r}. Reload the page after putting it there."
        )
    return path


def _dequantized(handle, base: str, keys: set, shape: tuple) -> torch.Tensor:
    """A quantized linear weight from a checkpoint, dequantized to float32 by ComfyUI's layers.

    Args:
        handle: The open checkpoint.
        base: Key prefix of the layer, ending in ``.``.
        keys: Every key in the checkpoint.
        shape: ``(out_features, in_features)`` the weight must have.

    Returns:
        The weight.
    """
    import comfy.ops

    layer_keys = [key for key in keys if key.startswith(base) and "." not in key[len(base):]]
    operations = comfy.ops.mixed_precision_ops(compute_dtype=torch.float32)
    layer = operations.Linear(int(shape[1]), int(shape[0]), bias=False, device="cpu")
    layer.load_state_dict({key[len(base):]: handle.get_tensor(key) for key in layer_keys
                           if not key.endswith("bias")}, strict=False)
    weight = layer.weight
    weight = weight.dequantize() if hasattr(weight, "dequantize") else weight
    return weight.detach().to(torch.float32)


def restore(model, name: str, first: int, last: int, strength: float):
    """A clone of a model with blocks ``first`` to ``last`` taken from a checkpoint.

    Args:
        model: A model patcher.
        name: The source checkpoint, a name from :func:`checkpoints`.
        first: First block restored, counted from 0.
        last: Last block restored.
        strength: Share of the source weights, 0 to 1.

    Returns:
        ``(model, tensors)``: the patched clone and how many weights were restored.

    Raises:
        ValueError: The blocks are out of range, or the checkpoint does not match the model.
    """
    from safetensors import safe_open

    list_name, blocks = block_list(model)
    if not 0 <= first <= last < len(blocks):
        raise ValueError(
            f"This model has blocks 0 to {len(blocks) - 1}, and blocks {first} to {last} were "
            f"asked for. Set first_block and last_block inside that range."
        )
    path = _path(name)
    sources = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        probe = f"{list_name}.{first}."
        prefix = next((p for p in _PREFIXES if any(key.startswith(p + probe) for key in keys)), None)
        if prefix is None:
            raise ValueError(
                f"{name!r} holds no {list_name}.{first} weights, so it is not a checkpoint of this "
                f"model. Pick the same model before fine-tuning, such as "
                f"moss_image_base_int8_convrot.safetensors for MOSS-Image SFT."
            )
        for index in range(first, last + 1):
            for param, current in blocks[index].named_parameters():
                if not param.endswith(("weight", "bias")):
                    continue
                stored = f"{prefix}{list_name}.{index}.{param}"
                shape = tuple(current.shape)
                base = stored[: -len(param.rsplit(".", 1)[-1])]
                if stored not in keys:
                    raise ValueError(
                        f"{name!r} has no {list_name}.{index}.{param}, which this model holds. Pick "
                        f"a checkpoint of the same model."
                    )
                found = tuple(handle.get_slice(stored).get_shape())
                quantized = f"{base}comfy_quant" in keys and param.endswith("weight")
                if not quantized and found != shape:
                    raise ValueError(
                        f"{name!r} stores {list_name}.{index}.{param} as {found}, and this model "
                        f"needs {shape}. Pick a checkpoint of the same model."
                    )
                tensor = (_dequantized(handle, base, keys, shape) if quantized
                          else handle.get_tensor(stored))
                if tuple(tensor.shape) != shape:
                    raise ValueError(
                        f"{name!r} gives {list_name}.{index}.{param} as {tuple(tensor.shape)}, and "
                        f"this model needs {shape}. Pick a checkpoint of the same model."
                    )
                sources[f"diffusion_model.{list_name}.{index}.{param}"] = (tensor,)

    patched = model.clone()
    added = patched.add_patches(sources, float(strength), 1.0 - float(strength))
    # Restored weights go first, so the patches already on the model apply on top of them.
    for key in added:
        patched.patches[key].insert(0, patched.patches[key].pop())
    logger.info("restored %s %d to %d from %s at strength %.2f: %d weights", list_name, first,
                last, name, strength, len(added))
    return patched, len(added)

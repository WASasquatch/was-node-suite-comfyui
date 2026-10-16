"""A FLUX 3 Image Klein distilled checkpoint's target-time branch, attached to another model.

The branch is the ``ref_time_in`` embedder: two bias-free linear layers embedding the sigma
each step lands on, added to every modulation.
"""

from __future__ import annotations

import os

import torch

from .. import log

__all__ = [
    "BRANCH_KEYS",
    "FOLDER",
    "attach",
    "checkpoints",
    "read_branch",
]

logger = log.get_logger("model.flux3_flow_map")

#: Folder the distilled checkpoints are picked from.
FOLDER = "diffusion_models"

#: Tensors the branch is read from, as a distilled checkpoint names them.
BRANCH_KEYS = ("ref_time_in.in_layer.weight", "ref_time_in.out_layer.weight")

#: Prefixes a checkpoint may put in front of the transformer's own key names.
_PREFIXES = ("", "model.diffusion_model.", "diffusion_model.")

#: File extensions read as safetensors.
_SAFETENSORS = (".safetensors", ".sft")

#: Width of the sinusoidal timestep embedding the branch reads.
_EMBED = 256


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


def read_branch(name: str) -> dict[str, torch.Tensor]:
    """Read only the target-time branch's two weights from a distilled checkpoint.

    Args:
        name: A file name from :func:`checkpoints`.

    Returns:
        ``{"in_layer.weight": ..., "out_layer.weight": ...}`` as stored.

    Raises:
        ValueError: The file is not safetensors, is not in the folder, or holds no branch.
    """
    import folder_paths
    from safetensors import safe_open

    if not str(name).lower().endswith(_SAFETENSORS):
        raise ValueError(
            f"{name!r} is not a .safetensors file. Pick flux3_image_klein_sd_bf16 or "
            f"flux3_image_klein_sd_int8_convrot from the checkpoint menu."
        )
    path = folder_paths.get_full_path(FOLDER, name)
    if not path or not os.path.isfile(path):
        raise ValueError(
            f"ComfyUI's {FOLDER} folder holds no {name!r}. Reload the page after putting it there."
        )
    with safe_open(path, framework="pt", device="cpu") as handle:
        stored = set(handle.keys())
        for prefix in _PREFIXES:
            keys = [prefix + key for key in BRANCH_KEYS]
            if all(key in stored for key in keys):
                return {key.split("ref_time_in.", 1)[1]: handle.get_tensor(key) for key in keys}
    raise ValueError(
        f"{name!r} holds no ref_time_in weights, so it is not a distilled FLUX 3 Image Klein "
        f"checkpoint. Pick one with _sd_ in its name, such as "
        f"flux3_image_klein_sd_bf16.safetensors."
    )


def attach(model, name: str):
    """A clone of a FLUX 3 Image model that samples with a distilled checkpoint's target-time branch.

    Args:
        model: A ``MODEL`` whose transformer is ComfyUI's FLUX 3 Image transformer.
        name: The distilled checkpoint to read the branch from, a name from :func:`checkpoints`.

    Returns:
        The clone, or ``model`` itself where its transformer already carries a branch.

    Raises:
        ValueError: The model is not a FLUX 3 Image model, or the branch does not fit it.
    """
    import comfy.ops
    from comfy.ldm.flux.layers import MLPEmbedder
    from comfy.ldm.flux3.model import Flux3

    transformer = getattr(getattr(model, "model", None), "diffusion_model", None)
    if not isinstance(transformer, Flux3):
        raise ValueError(
            f"Flux 3 Flow Map Patch takes a FLUX 3 Image model, and this one is "
            f"{type(transformer).__name__}. Wire the Load Diffusion Model that loads "
            f"flux3_image_klein_bf16, or a LoRA after it."
        )
    # Read from the checkpoint and this patcher, never the live module, which still carries the
    # object patches of whichever clone was loaded last.
    config = getattr(getattr(model.model, "model_config", None), "unet_config", None) or {}
    if config.get("flow_map") or model.object_patches.get("diffusion_model.flow_map") is True:
        logger.info("the model already samples with a target-time branch; nothing was added")
        return model

    weights = read_branch(name)
    hidden, embed = (int(side) for side in transformer.time_in.in_layer.weight.shape)
    wanted = {"in_layer.weight": (hidden, embed), "out_layer.weight": (hidden, hidden)}
    for key, shape in wanted.items():
        found = tuple(weights[key].shape)
        if found != shape or not weights[key].is_floating_point():
            raise ValueError(
                f"{name!r} has a ref_time_in.{key} of {found} {weights[key].dtype}, and this "
                f"model needs {shape} floating point. Pick the distilled checkpoint that "
                f"matches the base model."
            )

    branch = MLPEmbedder(embed, hidden, bias=False, dtype=weights["in_layer.weight"].dtype,
                         device="cpu", operations=comfy.ops.manual_cast)
    branch.load_state_dict(weights)
    branch.requires_grad_(False)

    # An object patch replaces an existing attribute, and the base transformer has none.
    if not hasattr(transformer, "ref_time_in"):
        transformer.ref_time_in = None
    patched = model.clone()
    patched.add_object_patch("diffusion_model.ref_time_in", branch)
    patched.add_object_patch("diffusion_model.flow_map", True)
    logger.info("added the target-time branch from %s", name)
    return patched

"""The MiniMax H3 learned latent upscaler: checkpoints found or fetched, built and run on video latents.

The network lives in ``modules/vendor/h3_latent_upscaler``. Latents are raw H3 video latents,
``(batch, 24, time, height, width)``, as VAE Encode answers them.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .. import log
from . import (
    NETWORK_FEATURE,
    ModelUnavailable,
    checkpoint_shapes,
    managed_module,
    model_directories,
    model_file_path,
    model_files,
    network_enabled,
)

__all__ = [
    "CHECKPOINTS",
    "FOLDER",
    "available",
    "backend",
    "build",
    "offered",
    "resolve",
    "upscale",
    "working_bytes",
]

logger = log.get_logger("h3_latent_upscaler")

#: ``folder_paths`` key, and the directory it names under ``models``.
FOLDER = "latent_upscale_models"

#: Checkpoints a run can fetch, by the filename each is kept under: the repository holding it
#: and its name there.
CHECKPOINTS = {
    "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors": (
        "LBH-123-AI/Minimax_h3_latent_Upscaler",
        "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors",
    ),
}

#: Extensions a checkpoint carries.
SUFFIXES = (".safetensors",)

#: Feature maps of the network's width held at once while it runs, counted at the output size.
LIVE_MAPS = 6


def _architecture(shapes: dict) -> dict | None:
    """The network's settings a checkpoint's tensor shapes describe, or ``None`` for another network."""
    shapes = {key.removeprefix("upscaler."): shape for key, shape in shapes.items()}
    conv = shapes.get("conv_in.weight")
    if not conv or len(conv) != 5 or conv[1] != 24 or "conv_out.weight" not in shapes:
        return None
    blocks = {"in": set(), "out": set()}
    temporal = None
    for key, shape in shapes.items():
        found = re.match(r"(in|out)_blocks\.(\d+)\.in_layers\.", key)
        if found:
            blocks[found.group(1)].add(int(found.group(2)))
        if key.endswith(".dwconv.weight") and temporal is None:
            temporal = int(shape[2])
    if not blocks["in"] or not blocks["out"]:
        return None
    return {
        "in_channels": int(conv[1]),
        "channels": int(conv[0]),
        "in_blocks": len(blocks["in"]),
        "out_blocks": len(blocks["out"]),
        "temporal_every": 2 if temporal else 0,
        "temporal_kernel": temporal or 5,
    }


def available() -> list[str]:
    """Upscaler checkpoints on disk whose tensors match the network.

    Returns:
        Names relative to whichever model directory holds them, :data:`CHECKPOINTS` first.
    """
    found = []
    for name in model_files(FOLDER, suffixes=SUFFIXES):
        path = model_file_path(FOLDER, name)
        if path is not None and _architecture(checkpoint_shapes(path)):
            found.append(name)
    order = list(CHECKPOINTS)
    return sorted(found, key=lambda name: (order.index(Path(name).name) if Path(name).name in order
                                           else len(order), name))


def offered() -> list[str]:
    """What the checkpoint menu lists: what is on disk, then what a run could fetch.

    Returns:
        Names on disk, followed by the fetchable ones not yet there when ``features.network``
        is on.
    """
    found = available()
    if not network_enabled():
        return found
    on_disk = {Path(name).name for name in found}
    return found + [name for name in CHECKPOINTS if name not in on_disk]


def resolve(name: str) -> Path:
    """Locate a checkpoint, fetching it when it is one of :data:`CHECKPOINTS`.

    Args:
        name: A name from :func:`offered`.

    Returns:
        The path to the file.

    Raises:
        ModelUnavailable: The file is not on disk and could not be fetched.
    """
    found = model_file_path(FOLDER, name)
    if found is not None:
        return found
    base = Path(str(name)).name
    directories = model_directories(FOLDER)
    source = CHECKPOINTS.get(base)
    if source is not None and network_enabled() and directories:
        return _fetch(base, source, directories[0])
    searched = ", ".join(str(directory) for directory in directories) or "no directory at all"
    where = (
        f"{_by_hand(base)} Setting {NETWORK_FEATURE}: true in config.yaml fetches it from "
        "Hugging Face on first use instead."
        if source is not None
        else f"Put it in ComfyUI/models/{FOLDER}."
    )
    raise ModelUnavailable(f"The H3 latent upscaler {name} was not found. {where} Searched: {searched}.")


def _by_hand(name: str) -> str:
    """Where one of :data:`CHECKPOINTS` can be downloaded by hand, as a sentence."""
    repo_id, filename = CHECKPOINTS[name]
    return (
        f"Download {filename} from https://huggingface.co/{repo_id} and save it as "
        f"ComfyUI/models/{FOLDER}/{name}."
    )


def _fetch(name: str, source: tuple[str, str], target: Path) -> Path:
    """Download one checkpoint and keep it under ``name`` in ``target``.

    Raises:
        DependencyError: ``huggingface_hub`` is not importable.
        ModelUnavailable: The download did not complete.
    """
    import shutil

    from .. import deps

    hub = deps.require("huggingface_hub", feature="features.network")
    repo_id, filename = source
    logger.info("fetching %s from %s into %s. This happens once; the file is kept.", name, repo_id, target)
    landing = target / f".{Path(name).stem}.download"
    try:
        target.mkdir(parents=True, exist_ok=True)
        fetched = hub.hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(landing))
        final = target / name
        os.replace(fetched, final)
    except Exception as error:
        raise ModelUnavailable(
            f"The H3 latent upscaler {name} could not be fetched from https://huggingface.co/"
            f"{repo_id} ({type(error).__name__}: {error}). {_by_hand(name)} Then restart ComfyUI."
        ) from error
    finally:
        shutil.rmtree(landing, ignore_errors=True)
    logger.info("%s is at %s", name, final)
    return final


def build(path):
    """Build the network a checkpoint describes and load it.

    Args:
        path: A checkpoint file.

    Returns:
        The network in eval mode on the CPU, half precision where CUDA is present.

    Raises:
        ValueError: The file does not hold the H3 latent upscaler.
    """
    import torch
    from safetensors.torch import load_file

    from ..vendor.h3_latent_upscaler.network import LatentResizer3D

    state = {key.removeprefix("upscaler."): value for key, value in load_file(str(path)).items()}
    settings = _architecture({key: list(value.shape) for key, value in state.items()})
    if settings is None:
        raise ValueError(f"{Path(path).name} does not hold the MiniMax H3 latent upscaler")
    with torch.device("meta"):
        network = LatentResizer3D(**settings)
    network.load_state_dict(state, strict=True, assign=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return network.eval().requires_grad_(False).to(dtype)


def backend(name: str, device: str | None = None):
    """The network for one checkpoint, built once and registered with ComfyUI's model management.

    Args:
        name: A name from :func:`offered`.
        device: Device name for inference, or ``None`` for ComfyUI's compute device.

    Returns:
        A :class:`~modules.model.Backend` named ``"H3 Latent Upscaler"``.

    Raises:
        ModelUnavailable: The file is not on disk and could not be fetched.
    """
    path = resolve(name)
    key = ("h3_latent_upscaler", str(path), path.stat().st_mtime_ns)
    return managed_module(key, lambda: build(path), device=device, name="H3 Latent Upscaler")


def working_bytes(rows: int, height: int, width: int, channels: int = 512) -> int:
    """Device memory one run takes beside the weights, at half precision.

    Args:
        rows: Latent rows resized at once.
        height: Target latent height.
        width: Target latent width.
        channels: The network's feature width.

    Returns:
        Bytes.
    """
    return int(LIVE_MAPS * channels * int(rows) * int(height) * int(width) * 2)


def upscale(name: str, latent, height: int, width: int):
    """Resize an H3 video latent with the learned upscaler.

    Args:
        name: A name from :func:`offered`.
        latent: ``(batch, 24, time, h, w)``.
        height: Target latent height, no smaller than ``h``.
        width: Target latent width, no smaller than ``w``.

    Returns:
        ``(batch, 24, time, height, width)`` on the CPU, in ``latent``'s dtype.

    Raises:
        ModelUnavailable: The checkpoint is not on disk and could not be fetched.
    """
    import torch

    from ..vendor.h3_latent_upscaler.network import LATENTS_MEAN, LATENTS_STD

    net = backend(name)
    rows = int(latent.shape[2])
    device = net.load(memory_required=working_bytes(rows, height, width, net.model.conv_in.out_channels))
    dtype = net.model.conv_in.weight.dtype
    mean = torch.tensor(LATENTS_MEAN, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    std = torch.tensor(LATENTS_STD, dtype=dtype, device=device).view(1, -1, 1, 1, 1)
    scale = (height / int(latent.shape[3]) + width / int(latent.shape[4])) / 2.0
    with torch.inference_mode():
        work = (latent.to(device=device, dtype=dtype) - mean) / std
        out = net.model(work, scale, (rows, int(height), int(width)))
        out = out * std + mean
    return out.to(device="cpu", dtype=latent.dtype)

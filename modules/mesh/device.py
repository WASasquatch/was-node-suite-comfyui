"""Device choice and memory handling for mesh algorithms.

Work runs on ComfyUI's torch device when ComfyUI is loaded and on the CPU otherwise. Sizes are
in bytes.
"""

from __future__ import annotations

__all__ = [
    "chunk_size",
    "compute_device",
    "free_bytes",
    "is_out_of_memory",
    "prepare",
    "release",
    "with_cpu_fallback",
]

import sys
from typing import Callable, TypeVar

import torch

from ..log import get_logger

logger = get_logger("mesh.device")

Result = TypeVar("Result")

#: Labels already reported as falling back to the CPU.
warned: set[str] = set()

#: Working budget assumed for the CPU when no memory figure is available.
CPU_BUDGET = 2 << 30


def _management():
    """ComfyUI's model management module when ComfyUI has loaded it, else None."""
    return sys.modules.get("comfy.model_management")


def compute_device() -> torch.device:
    """The device mesh work runs on.

    Returns:
        ComfyUI's torch device when ComfyUI is loaded and it is a usable CUDA or XPU
        device, the CPU otherwise.
    """
    management = _management()
    if management is None:
        return torch.device("cpu")
    try:
        device = torch.device(management.get_torch_device())
    except Exception:
        return torch.device("cpu")
    if device.type == "cuda" and torch.cuda.is_available():
        return device
    xpu = getattr(torch, "xpu", None)
    if device.type == "xpu" and xpu is not None and xpu.is_available():
        return device
    return torch.device("cpu")


def free_bytes(device: torch.device | str | None = None) -> int:
    """Memory free for new tensors on a device.

    Args:
        device: The device; None means `compute_device`.

    Returns:
        Free bytes, or `CPU_BUDGET` for a CPU whose free memory cannot be read.
    """
    device = torch.device(device) if device is not None else compute_device()
    management = _management()
    if management is not None:
        try:
            return int(management.get_free_memory(device))
        except Exception:
            pass
    if device.type == "cpu":
        try:
            import psutil
        except ImportError:
            return CPU_BUDGET
        return int(psutil.virtual_memory().available)
    if device.type == "cuda" and torch.cuda.is_available():
        try:
            free, _total = torch.cuda.mem_get_info(device)
            return int(free)
        except Exception:
            pass
    return CPU_BUDGET


def prepare(bytes_needed: int, device: torch.device | str | None = None) -> int:
    """Ask ComfyUI to unload models until ``bytes_needed`` is free on a GPU device.

    Args:
        bytes_needed: Size of the coming allocations.
        device: The device; None means `compute_device`. A CPU device is left alone.

    Returns:
        Free bytes on the device afterwards.
    """
    device = torch.device(device) if device is not None else compute_device()
    if device.type == "cpu":
        return free_bytes(device)
    management = _management()
    if management is not None:
        try:
            management.free_memory(int(bytes_needed), device)
        except Exception:
            logger.debug("model management could not free memory", exc_info=True)
        if free_bytes(device) < bytes_needed:
            release(device)
    return free_bytes(device)


def release(device: torch.device | str | None = None) -> None:
    """Return cached allocator blocks on a GPU device to the driver.

    Args:
        device: The device; None means `compute_device`.
    """
    device = torch.device(device) if device is not None else compute_device()
    if device.type == "cpu":
        return
    management = _management()
    if management is not None:
        try:
            management.soft_empty_cache()
            return
        except Exception:
            pass
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()


def chunk_size(
    bytes_per_item: int,
    device: torch.device | str | None = None,
    minimum: int = 1024,
    maximum: int = 1 << 20,
    share: float = 0.25,
) -> int:
    """How many items fit in a share of the free memory on a device.

    Args:
        bytes_per_item: Working memory one item needs.
        device: The device; None means `compute_device`.
        minimum: Smallest chunk returned.
        maximum: Largest chunk returned.
        share: Fraction of the free memory one chunk may use.

    Returns:
        Items per chunk, between ``minimum`` and ``maximum``.
    """
    budget = free_bytes(device) * share
    if torch.device(device if device is not None else compute_device()).type == "cpu":
        budget = min(budget, CPU_BUDGET)
    count = int(budget // max(int(bytes_per_item), 1))
    return max(int(minimum), min(int(maximum), count))


def is_out_of_memory(error: BaseException) -> bool:
    """Whether an error is a device running out of memory.

    Args:
        error: The raised error.

    Returns:
        True for a CUDA or allocator out of memory error.
    """
    if isinstance(error, getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)):
        return True
    management = _management()
    exception = getattr(management, "OOM_EXCEPTION", None) if management is not None else None
    if isinstance(exception, type) and isinstance(error, exception):
        return True
    text = str(error).lower()
    return isinstance(error, RuntimeError) and ("out of memory" in text or "not enough memory" in text)


def with_cpu_fallback(
    work: Callable[[torch.device], Result],
    device: torch.device | str | None = None,
    label: str = "mesh operation",
) -> Result:
    """Run ``work`` on a device and again on the CPU when the device runs out of memory.

    Args:
        work: Called with the device to run on.
        device: The first device; None means `compute_device`.
        label: Name used in the warning, logged once per label.

    Returns:
        What ``work`` returns.

    Raises:
        Exception: Any error from ``work`` other than running out of GPU memory.
    """
    device = torch.device(device) if device is not None else compute_device()
    if device.type == "cpu":
        return work(device)
    try:
        return work(device)
    except Exception as error:
        if not is_out_of_memory(error):
            raise
    release(device)
    if label not in warned:
        warned.add(label)
        logger.warning(
            "%s ran out of memory on %s and ran on the CPU instead, which is slower. Free GPU "
            "memory by unloading models, or lower the mesh size, to keep it on the GPU.",
            label, device,
        )
    return work(torch.device("cpu"))

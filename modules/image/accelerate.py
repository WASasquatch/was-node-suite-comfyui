"""Running an image operation on ComfyUI's compute device.

An IMAGE arrives and leaves on the CPU. :func:`run_on` runs a whole batch on the device,
:func:`run_frames` a run of frames at a time, both falling back to the CPU.
"""

from __future__ import annotations

__all__ = ["RUN_SHARE", "frames_per_run", "run_frames", "run_on", "working_device"]

#: Share of the device's free memory one run of frames may fill.
RUN_SHARE = 0.5

import torch

from .. import log

logger = log.get_logger("image.accelerate")


def working_device(prefer_gpu: bool = True) -> torch.device:
    """Where an image operation should run.

    Args:
        prefer_gpu: Whether to ask for ComfyUI's compute device. False answers the CPU.

    Returns:
        A ``torch.device``.
    """
    if not prefer_gpu:
        return torch.device("cpu")
    try:
        from ..model import compute_device

        return compute_device()
    except Exception:
        logger.debug("no compute device is available, so this runs on the CPU", exc_info=True)
        return torch.device("cpu")


def run_on(images: torch.Tensor, work, prefer_gpu: bool = True) -> torch.Tensor:
    """Run one operation over a batch, on the accelerator where there is one.

    Args:
        images: The batch, as ComfyUI hands it over.
        work: A callable taking the batch and answering a tensor of the same shape.
        prefer_gpu: Whether to move the batch to ComfyUI's compute device first.

    Returns:
        The result, on the device the batch arrived on.
    """
    home = images.device
    device = working_device(prefer_gpu)
    if device == home:
        return work(images)

    try:
        return work(images.to(device)).to(home)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
        logger.warning(
            "%s could not run this image operation (%s), so it ran on the CPU instead",
            device, type(error).__name__,
        )
        logger.debug("the device reported", exc_info=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return work(images.to(home))


def _free_bytes(device: torch.device) -> int:
    """Memory an operation can take on a device, in bytes."""
    try:
        import comfy.model_management

        return int(comfy.model_management.get_free_memory(device))
    except Exception:
        logger.debug("ComfyUI did not report free memory", exc_info=True)
    if device.type == "cuda":
        return int(torch.cuda.mem_get_info(device)[0])
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return 4 << 30


def frames_per_run(images: torch.Tensor, device: torch.device, factor: float) -> int:
    """How many frames of a batch one run of an operation can take on a device.

    Args:
        images: ``(frames, ...)`` batch.
        device: Where the operation runs.
        factor: Float32 copies of a frame the operation holds at its peak.

    Returns:
        1 at the least.
    """
    frame = max(1, int(images[0].numel())) * 4
    return max(1, int(_free_bytes(device) * RUN_SHARE // (frame * max(1.0, float(factor)))))


def _out_of_memory(error: BaseException) -> bool:
    """Whether an error is a device or host running out of memory."""
    if isinstance(error, torch.cuda.OutOfMemoryError):
        return True
    text = str(error).lower()
    return "out of memory" in text or "not enough memory" in text


def _progress(total: int):
    """A callable advancing ComfyUI's progress bar by a count of frames, or doing nothing outside it."""
    try:
        import comfy.utils

        bar = comfy.utils.ProgressBar(total)
        return bar.update
    except Exception:
        return lambda _count: None


def run_frames(images: torch.Tensor, work, prefer_gpu: bool = True, factor: float = 8.0,
               node: str = "", advice: str = "") -> torch.Tensor:
    """Run a per-frame operation over a batch in runs sized to the free memory where it runs.

    Args:
        images: ``(frames, height, width, channels)`` batch, frames first.
        work: A callable taking a run of frames and answering a tensor with one row per frame.
            Each frame's result depends on that frame alone.
        prefer_gpu: Whether to run on ComfyUI's compute device. False runs on the CPU.
        factor: Float32 copies of a frame ``work`` holds at its peak.
        node: Display name of the calling node, for a refusal from scratch memory.
        advice: Closing sentence of that refusal.

    Returns:
        The result on the device the batch arrived on: a batch from
        :func:`modules.image.scratch.allocate` where it took more than one run, otherwise
        what :func:`run_on` answers. A run the device refuses for memory is halved, and a
        single frame it still refuses sends the rest of the batch to the CPU.
    """
    if images.ndim < 4 or int(images.shape[0]) < 2:
        return run_on(images, work, prefer_gpu)
    home = images.device
    device = working_device(prefer_gpu)
    count = int(images.shape[0])
    run = min(count, frames_per_run(images, device, factor))
    if run >= count:
        return run_on(images, work, prefer_gpu)

    from . import scratch

    out = None
    advance = _progress(count)
    start = 0
    while start < count:
        stop = min(count, start + run)
        chunk = images[start:stop]
        try:
            result = work(chunk.to(device)).to(home)
        except (RuntimeError, torch.cuda.OutOfMemoryError) as error:
            if device == home and not (_out_of_memory(error) and run > 1):
                raise
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if _out_of_memory(error) and run > 1:
                run = max(1, run // 2)
                continue
            logger.warning(
                "%s could not run this image operation (%s), so the rest of the batch runs on "
                "the CPU", device, type(error).__name__,
            )
            logger.debug("the device reported", exc_info=True)
            device = home
            run = min(run, frames_per_run(images, home, factor))
            continue
        if out is None:
            out = scratch.allocate((count, *result.shape[1:]), result.dtype, node=node, advice=advice)
        out[start:stop].copy_(result)
        del result
        scratch.trim(out[start:stop])
        scratch.trim(chunk)
        advance(stop - start)
        start = stop
    return out

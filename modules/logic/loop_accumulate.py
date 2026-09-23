"""Collecting a loop's values across its iterations, and joining them when it finishes.

One list per socket. Frames are counted along dimension 0, dimension 2 for a five
dimensional video tensor, and ``samples`` for a latent.
"""

from __future__ import annotations

from typing import Any

from . import loop_meta

__all__ = [
    "append",
    "batch_values",
    "collected",
    "count_frames",
    "finalize",
    "frame_count",
    "total_count",
]


def append(accumulated: dict[str, list] | None, name: str, value: Any) -> dict[str, list]:
    """A copy of ``accumulated`` with ``value`` added to ``name``'s list.

    Args:
        accumulated: The mapping so far, or None on the first iteration.
        name: The socket the value arrived on.
        value: Whatever that socket held this iteration.

    Returns:
        A new mapping. The lists inside it are new too, so the argument is left as it was.
    """
    # Copied rather than appended in place: a token from an earlier iteration stays reachable
    # through the execution cache and must not see a later iteration's contents.
    current = accumulated or {}
    return {**current, name: [*current.get(name, []), value]}


def collected(accumulated: dict[str, list] | None, name: str) -> list:
    """``name``'s collected values, as a list safe for the caller to keep.

    Args:
        accumulated: The mapping a loop finished with, or None when nothing was collected.
        name: The socket to read.

    Returns:
        Every value collected on that socket, in iteration order.
    """
    return list((accumulated or {}).get(name, ()))


def frame_count(value: Any) -> int | None:
    """How many frames one value holds, or None when it holds no countable frames.

    Args:
        value: A tensor, a latent, a nested tensor, a list, or anything else.

    Returns:
        The length along the value's frame dimension, or None for a value that is not a
        sequence of frames, such as a number or a string.
    """
    # Duck typed rather than checked against torch: a readout and a frame count must not be
    # what makes the pack import it.
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return None

    if isinstance(value, dict):
        samples = value.get("samples")
        return frame_count(samples) if samples is not None else None

    # A nested tensor keeps one tensor per resolution and the same number of frames in each,
    # so the first one answers for all of them.
    tensors = getattr(value, "tensors", None)
    if isinstance(tensors, (list, tuple)) and tensors:
        return frame_count(tensors[0])

    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            dims = [int(n) for n in shape]
        except (TypeError, ValueError):
            return None
        if not dims:
            return None
        return dims[2] if len(dims) == 5 else dims[0]

    if isinstance(value, (list, tuple)):
        return len(value)
    return None


def count_frames(values: list) -> int | None:
    """The frames across every value in ``values``, or None when none of them hold frames.

    Args:
        values: One socket's collected values, in iteration order.

    Returns:
        The total, or None when nothing in the list carries a frame count.
    """
    total = 0
    counted = False
    for value in values:
        count = frame_count(value)
        if count is None:
            continue
        total += count
        counted = True
    return total if counted else None


def total_count(accumulated: dict[str, list] | None, names: list[str]) -> int:
    """The frames collected so far, read from the first socket that holds any.

    Args:
        accumulated: The mapping so far, or None when nothing was collected.
        names: The collecting socket names, in the order the node declares them.

    Returns:
        The count, or 0 when nothing has been collected.
    """
    if not accumulated:
        return 0
    fallback = 0
    for name in names:
        values = accumulated.get(name)
        if not values:
            continue
        count = count_frames(values)
        if count is not None:
            return count
        fallback = fallback or len(values)
    return fallback


def _frame_dim(samples) -> int:
    """Which dimension of a tensor counts frames: 2 for a 5D video tensor, 0 otherwise."""
    return 2 if getattr(samples, "ndim", 0) == 5 else 0


def _joinable(shapes: list, dim: int) -> bool:
    """Whether every shape matches apart from its frame dimension."""
    first = shapes[0]
    return all(
        len(shape) == len(first)
        and all(a == b for axis, (a, b) in enumerate(zip(first, shape)) if axis != dim)
        for shape in shapes
    )


def batch_values(values: list):
    """The collected values joined into one batch, or None when they do not join.

    Args:
        values: One socket's collected values, in iteration order.

    Returns:
        One tensor, one latent, one soundtrack, or None.
    """
    if not values:
        return None

    # Imported here rather than at module scope: a loop carrying nothing but numbers must not
    # pay for torch, and the pack's import budget is measured.
    try:
        import torch
    except ImportError:
        return None

    if all(isinstance(value, torch.Tensor) for value in values):
        dim = _frame_dim(values[0])
        if not _joinable([tuple(v.shape) for v in values], dim):
            return None
        return torch.cat(values, dim=dim)

    # Audio clips are joined along time into one track.
    if all(isinstance(value, dict) and isinstance(value.get("waveform"), torch.Tensor)
           for value in values):
        rates = {value.get("sample_rate") for value in values}
        if len(rates) != 1:
            return None
        waves = [value["waveform"] for value in values]
        along = waves[0].ndim - 1
        if not _joinable([tuple(wave.shape) for wave in waves], along):
            return None
        joined = dict(values[0])
        joined["waveform"] = torch.cat(waves, dim=along)
        return joined

    if all(isinstance(value, dict) and isinstance(value.get("samples"), torch.Tensor)
           for value in values):
        samples = [value["samples"] for value in values]
        dim = _frame_dim(samples[0])
        if not _joinable([tuple(s.shape) for s in samples], dim):
            return None
        joined = dict(values[0])
        joined["samples"] = torch.cat(samples, dim=dim)
        # A per-frame mask cannot describe frames it was never measured against, so a joined
        # latent carries one only when every part brought a matching one.
        masks = [value.get("noise_mask") for value in values]
        covers = all(
            mask is not None and getattr(mask, "ndim", None) == sample.ndim
            for mask, sample in zip(masks, samples)
        )
        if covers and _joinable([tuple(m.shape) for m in masks], dim):
            joined["noise_mask"] = torch.cat(masks, dim=dim)
        else:
            joined.pop("noise_mask", None)
        return joined

    return None


def finalize(values: list, accumulate: bool, latest: Any):
    """What one slot's output socket carries once the loop is over.

    Args:
        values: That socket's collected values, in iteration order.
        accumulate: Whether the loop was collecting.
        latest: The value the socket last received.

    Returns:
        ``(value, kind)``, where kind is one of :data:`loop_meta.FINAL`,
        :data:`loop_meta.BATCH` or :data:`loop_meta.LIST`.
    """
    if not accumulate or not values:
        return latest, loop_meta.FINAL
    joined = batch_values(values)
    if joined is not None:
        return joined, loop_meta.BATCH
    return list(values), loop_meta.LIST

"""Frame batches held in memory where they fit, and in a scratch file where they do not.

A scratch file lives in ``paths.scratch`` or ComfyUI's ``temp/`` and is removed once nothing
reads it.
"""

from __future__ import annotations

import math
import mmap
import os
import shutil
import tempfile
import weakref

import torch

from .. import config, log

__all__ = [
    "DISK_RESERVE", "MEMORY_SHARE", "FrameStore", "allocate", "join", "room", "spelled", "stack",
    "trim",
]

logger = log.get_logger("image.scratch")

#: Share of free memory one batch may take before it goes to a scratch file instead.
MEMORY_SHARE = 0.25

#: Bytes left free on a drive after a scratch file is placed on it.
DISK_RESERVE = 4 * 1024 ** 3

#: Name every scratch file starts with.
PREFIX = "was-frames-"

#: Every live mapped batch: its first byte's address, to its length and its mapping.
_MAPPED: dict = {}


def spelled(count: int) -> str:
    """A byte count as gigabytes or megabytes, such as ``48.2 GB``.

    Args:
        count: Bytes.

    Returns:
        The count with one decimal and its unit.
    """
    if count >= 1024 ** 3:
        return f"{count / 1024 ** 3:.1f} GB"
    return f"{count / 1024 ** 2:.1f} MB"


def _available_memory() -> int:
    """Bytes of system memory free now, measured the way ComfyUI's cache measures it."""
    try:
        from comfy import system_memory

        return int(system_memory.virtual_memory_available())
    except (ImportError, AttributeError):
        import psutil

        return int(psutil.virtual_memory().available)


def _directories() -> list[str]:
    """Directories a scratch file may go in, in the order they are tried."""
    found = []
    configured = config.scratch_directory()
    if configured:
        try:
            os.makedirs(configured, exist_ok=True)
            found.append(configured)
        except OSError as error:
            logger.warning("paths.scratch names %s, which cannot be created: %s", configured, error)
    try:
        import folder_paths

        found.append(folder_paths.get_temp_directory())
    except (ImportError, AttributeError):
        pass
    found.append(tempfile.gettempdir())
    unique = []
    for directory in found:
        if os.path.isdir(directory) and os.path.normcase(os.path.abspath(directory)) not in {
            os.path.normcase(os.path.abspath(kept)) for kept in unique
        }:
            unique.append(directory)
    return unique


def _free_disk(directory: str) -> int:
    """Bytes free on the drive holding ``directory``, or -1 where it cannot be read."""
    try:
        return int(shutil.disk_usage(directory).free)
    except OSError:
        return -1


def room() -> int:
    """Bytes one set of frames may take: the share of free memory, or the roomiest scratch drive.

    Returns:
        The larger of the two, never below 0.
    """
    best = _available_memory() * MEMORY_SHARE
    for directory in _directories():
        best = max(best, _free_disk(directory) - DISK_RESERVE)
    return max(0, int(best))


def _mapped(shape: tuple, dtype: torch.dtype, directory: str) -> torch.Tensor:
    """A batch mapped from a temporary file in ``directory``.

    Args:
        shape: Size of each axis.
        dtype: Element type.
        directory: Where the file is created.

    Returns:
        A contiguous CPU tensor whose file is deleted when the last view of it is freed.
    """
    count = math.prod(shape)
    with tempfile.TemporaryFile(dir=directory, prefix=PREFIX, suffix=".bin") as handle:
        if os.name != "nt":
            os.ftruncate(handle.fileno(), count * dtype.itemsize)
        mapped = mmap.mmap(handle.fileno(), count * dtype.itemsize)
    batch = torch.frombuffer(mapped, dtype=dtype, count=count).view(shape)
    start = batch.data_ptr()
    _MAPPED[start] = (count * dtype.itemsize, weakref.ref(mapped))
    weakref.finalize(mapped, _MAPPED.pop, start, None)
    return batch


def trim(view) -> None:
    """Drop the pages under part of a batch from this process's memory; its scratch file keeps them.

    Args:
        view: A CPU tensor. Anything not held in a scratch file is left as it is.
    """
    if not _MAPPED or getattr(view, "device", None) is None or view.device.type != "cpu":
        return
    first = int(view.data_ptr())
    span = sum((int(side) - 1) * int(step) for side, step in zip(view.shape, view.stride()) if side > 0)
    last = first + (span + 1) * view.element_size()
    for start, (length, mapping) in list(_MAPPED.items()):
        if not start <= first < start + length:
            continue
        last = min(last, start + length)
        if os.name == "nt":
            _unlock(first, last - first)
        else:
            mapped = mapping()
            if mapped is not None and hasattr(mapped, "madvise"):
                page = mmap.PAGESIZE
                offset = ((first - start) // page) * page
                mapped.madvise(mmap.MADV_DONTNEED, offset, (last - start) - offset)
        return


def _unlock(address: int, length: int) -> None:
    """Remove the pages of an unlocked address range from the process working set."""
    import ctypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.VirtualUnlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    kernel.VirtualUnlock.restype = ctypes.c_int
    kernel.VirtualUnlock(address, length)


def _refusal(needed: int, free: int, shape: tuple, node: str, advice: str, tried: list) -> MemoryError:
    """The error raised when no memory and no scratch drive can hold ``needed`` bytes."""
    if len(shape) in (3, 4):
        sized = f"{shape[0]} frame(s) at {shape[2]}x{shape[1]}"
    else:
        sized = "x".join(str(side) for side in shape)
    drives = "; ".join(
        f"{directory} has {spelled(room)} free" if room >= 0 else f"{directory} cannot be read"
        for directory, room in tried
    ) or "no scratch directory exists"
    return MemoryError(
        f"{node or 'This node'} would produce {spelled(needed)} ({sized}), more than the "
        f"{MEMORY_SHARE:.0%} of free memory one result may take ({spelled(free)} is free now), "
        f"so it was not started. No scratch drive has room for it either, with "
        f"{spelled(DISK_RESERVE)} kept free on each: {drives}. Set paths.scratch in "
        f"config.yaml to a folder on a drive with at least {spelled(needed + DISK_RESERVE)} "
        f"free, then run it again. {advice}".strip()
    )


def allocate(shape, dtype: torch.dtype = torch.float32, node: str = "", advice: str = ""):
    """An uninitialised CPU batch, in memory where it fits and in a scratch file otherwise.

    Args:
        shape: Size of each axis, frames first: ``(frames, height, width, channels)`` for an
            image, ``(frames, height, width)`` for a mask.
        dtype: Element type.
        node: Display name of the calling node, for the log and the refusal.
        advice: Closing sentence of the refusal, naming what to change on the node.

    Returns:
        A contiguous ``torch.Tensor`` on the CPU.

    Raises:
        MemoryError: Neither free memory nor any scratch drive has room for the batch.
    """
    shape = tuple(int(side) for side in shape)
    count = math.prod(shape)
    needed = count * dtype.itemsize
    free = _available_memory()
    if count == 0 or needed <= free * MEMORY_SHARE:
        return torch.empty(shape, dtype=dtype)

    tried = []
    for directory in _directories():
        room = _free_disk(directory)
        tried.append((directory, room))
        if room - DISK_RESERVE < needed:
            continue
        try:
            batch = _mapped(shape, dtype, directory)
        except (OSError, ValueError) as error:
            logger.warning("a %s scratch file could not be made in %s: %s",
                           spelled(needed), directory, error)
            continue
        logger.info(
            "%s holds its %s result in a scratch file in %s, as %s of memory is free",
            node or "a node", spelled(needed), directory, spelled(free),
        )
        return batch
    raise _refusal(needed, free, shape, node, advice, tried)


class FrameStore:
    """Equal frames written and read one at a time, in memory where they fit and in a scratch file otherwise.

    Args:
        count: Frames held at once.
        shape: Shape of one frame.
        dtype: Element type.
        node: Display name of the calling node, for the log and the refusal.
        advice: Closing sentence of the refusal, naming what to change on the node.
        tensor: An existing ``(count, *shape)`` batch to hold the frames in instead, frame
            ``i`` at row ``i``.
        alongside: Bytes of further frames the caller keeps beside this store, counted
            against the share of free memory and against a scratch drive's room.

    Raises:
        MemoryError: Neither free memory nor any scratch drive has room for the frames.
    """

    def __init__(self, count: int, shape, dtype=torch.float16, node: str = "", advice: str = "",
                 tensor=None, alongside: int = 0):
        self.count = int(count)
        self.shape = tuple(int(side) for side in shape)
        self.dtype = tensor.dtype if tensor is not None else dtype
        self.frame_bytes = math.prod(self.shape) * self.dtype.itemsize
        self.tensor = tensor
        self.direct = tensor is not None
        self.handle = None
        self.slots = {}
        self.spare = []
        self.buffer = None
        if self.direct:
            return
        needed = self.frame_bytes * self.count
        alongside = max(int(alongside), 0)
        free = _available_memory()
        if needed + alongside <= free * MEMORY_SHARE:
            self.tensor = torch.empty((self.count,) + self.shape, dtype=self.dtype)
            return
        tried = []
        for directory in _directories():
            room = _free_disk(directory)
            tried.append((directory, room))
            if room - DISK_RESERVE < needed + alongside:
                continue
            try:
                self.handle = tempfile.TemporaryFile(
                    dir=directory, prefix=PREFIX, suffix=".bin", buffering=0
                )
                # The file's full length is claimed on disk before any frame is written.
                self.handle.truncate(needed)
            except OSError as error:
                if self.handle is not None:
                    self.handle.close()
                    self.handle = None
                logger.warning("a %s scratch file could not be made in %s: %s",
                               spelled(needed), directory, error)
                continue
            logger.info(
                "%s keeps %s of frames in a scratch file in %s, as %s of memory is free",
                node or "a node", spelled(needed), directory, spelled(free),
            )
            return
        raise _refusal(needed + alongside, free, (self.count,) + self.shape, node, advice, tried)

    def _slot(self, index: int, new: bool) -> int:
        """Where frame ``index`` is kept, claiming a free place for a new one."""
        if self.direct:
            return int(index)
        slot = self.slots.get(int(index))
        if slot is None:
            if not new:
                raise KeyError(f"frame {index} was never kept")
            if self.spare:
                slot = self.spare.pop()
            elif len(self.slots) < self.count:
                slot = len(self.slots)
            else:
                raise IndexError(f"all {self.count} places are taken; release a frame first")
            self.slots[int(index)] = slot
        return slot

    def write(self, index: int, frame) -> None:
        """Keep one frame.

        Args:
            index: The frame's number in the clip.
            frame: A tensor on any device. A store in memory takes fewer trailing channels
                than ``shape``; a scratch file takes exactly ``shape``.

        Raises:
            IndexError: Every place is taken by a frame not yet released.
        """
        slot = self._slot(index, True)
        if self.handle is None:
            self.tensor[slot, ..., :frame.shape[-1]].copy_(frame)
            trim(self.tensor[slot])
            return
        data = frame.to(dtype=self.dtype).contiguous().cpu()
        self.handle.seek(slot * self.frame_bytes)
        self.handle.write(memoryview(data.numpy()).cast("B"))

    def read(self, index: int, device=None, channels: int | None = None):
        """One frame as float32.

        Args:
            index: The frame's number in the clip.
            device: Where the answer is placed; the CPU when None.
            channels: Leading channels answered, all when None.

        Returns:
            A tensor shaped like one frame.

        Raises:
            KeyError: The store was never given this frame.
            OSError: The scratch file ends inside the frame.
        """
        slot = self._slot(index, False)
        if self.handle is None:
            frame = self.tensor[slot]
            if channels is not None:
                frame = frame[..., :channels]
            answer = frame.to(device=device, dtype=torch.float32, copy=True)
            trim(self.tensor[slot])
            return answer
        if self.buffer is None:
            self.buffer = torch.empty(self.shape, dtype=self.dtype)
        view = memoryview(self.buffer.numpy()).cast("B")
        self.handle.seek(slot * self.frame_bytes)
        done = 0
        while done < self.frame_bytes:
            got = self.handle.readinto(view[done:])
            if not got:
                raise OSError(f"a scratch file ends {self.frame_bytes - done} byte(s) short of a frame")
            done += got
        frame = self.buffer if channels is None else self.buffer[..., :channels]
        return frame.to(device=device, dtype=torch.float32, copy=True)

    def release(self, index: int) -> None:
        """Let go of one frame, freeing its place for another.

        Args:
            index: The frame's number in the clip.
        """
        if self.direct:
            return
        slot = self.slots.pop(int(index), None)
        if slot is not None:
            self.spare.append(slot)

    def close(self) -> None:
        """Let go of the frames and delete the scratch file."""
        if self.handle is not None:
            self.handle.close()
            self.handle = None
        self.tensor = None
        self.buffer = None


def join(parts: list, node: str = "", advice: str = "", dtype: torch.dtype | None = None):
    """Concatenate batches along the first axis into one batch from :func:`allocate`.

    Args:
        parts: Tensors sharing every axis but the first. Emptied as each is copied.
        node: Display name of the calling node, for the log and the refusal.
        advice: Closing sentence of the refusal, naming what to change on the node.
        dtype: Element type of the result, the first part's when None.

    Returns:
        A contiguous CPU tensor holding every part in order.

    Raises:
        ValueError: ``parts`` is empty or the parts disagree past the first axis.
        MemoryError: Neither free memory nor any scratch drive has room for the result.
    """
    if not parts:
        raise ValueError("join() was given no tensors to join")
    tail = tuple(parts[0].shape[1:])
    for part in parts:
        if tuple(part.shape[1:]) != tail:
            raise ValueError(
                f"join() was given parts shaped {tuple(parts[0].shape)} and "
                f"{tuple(part.shape)}, which differ past the first axis"
            )
    total = sum(int(part.shape[0]) for part in parts)
    batch = allocate((total,) + tail, dtype or parts[0].dtype, node, advice)
    position = 0
    while parts:
        part = parts.pop(0)
        batch[position:position + part.shape[0]].copy_(part)
        position += int(part.shape[0])
        del part
    return batch


def stack(frames: list, node: str = "", advice: str = "", dtype: torch.dtype | None = None):
    """Stack equal frames along a new first axis into one batch from :func:`allocate`.

    Args:
        frames: Tensors of one shape. Emptied as each is copied.
        node: Display name of the calling node, for the log and the refusal.
        advice: Closing sentence of the refusal, naming what to change on the node.
        dtype: Element type of the result, the first frame's when None.

    Returns:
        A contiguous CPU tensor of ``len(frames)`` frames.

    Raises:
        ValueError: ``frames`` is empty or the frames differ in shape.
        MemoryError: Neither free memory nor any scratch drive has room for the result.
    """
    if not frames:
        raise ValueError("stack() was given no frames to stack")
    shape = tuple(frames[0].shape)
    for frame in frames:
        if tuple(frame.shape) != shape:
            raise ValueError(
                f"stack() was given frames shaped {shape} and {tuple(frame.shape)}"
            )
    batch = allocate((len(frames),) + shape, dtype or frames[0].dtype, node, advice)
    index = 0
    while frames:
        batch[index].copy_(frames.pop(0))
        index += 1
    return batch

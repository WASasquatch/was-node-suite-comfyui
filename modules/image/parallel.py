"""Per-frame work spread over CPU threads, answered in order behind ComfyUI's progress bar."""

from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

__all__ = ["MAX_WORKERS", "frame_progress", "map_frames", "thread_count"]

#: The most frames worked on at once.
MAX_WORKERS = 16


def thread_count(count: int, limit: int | None = None) -> int:
    """Threads a run over a number of frames takes.

    Args:
        count: Frames in the run.
        limit: The most frames that fit in memory at once, or None for no bound.

    Returns:
        1 at the least, and never more than the frames, the CPU's cores or
        :data:`MAX_WORKERS`.
    """
    threads = min(int(count), os.cpu_count() or 1, MAX_WORKERS)
    if limit is not None:
        threads = min(threads, int(limit))
    return max(1, threads)


def frame_progress(total: int):
    """A callable advancing the running node's progress bar by one frame.

    Args:
        total: Frames in the batch.

    Returns:
        A callable that also stops a cancelled run. It does nothing outside ComfyUI, and
        draws no bar for a single frame.
    """
    try:
        import comfy.model_management
        import comfy.utils
    except ImportError:
        return lambda: None
    bar = comfy.utils.ProgressBar(total) if total > 1 else None

    def step():
        comfy.model_management.throw_exception_if_processing_interrupted()
        if bar is not None:
            bar.update(1)

    return step


def map_frames(work, items, parallel: bool = True, limit: int | None = None) -> list:
    """Call a function on every frame, several frames at once.

    Args:
        work: Callable taking one item and answering its result.
        items: One item per frame, in batch order.
        parallel: Run on worker threads. False runs one item at a time on this thread.
        limit: The most items in flight at once, or None for the thread count alone.

    Returns:
        What ``work`` answered for each item, in the order of ``items``.

    Raises:
        Exception: Whatever ``work`` raised first, or ComfyUI's interruption when the run is
            cancelled. Items not yet started are dropped either way.
    """
    items = list(items)
    advance = frame_progress(len(items))
    threads = thread_count(len(items), limit) if parallel else 1
    if threads < 2:
        answers = []
        for item in items:
            answers.append(work(item))
            advance()
        return answers

    answers = [None] * len(items)
    queued = iter(enumerate(items))
    pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix="was-frames")
    try:
        # One item is submitted per thread, and another each time one finishes.
        running = {}
        for index, item in queued:
            running[pool.submit(work, item)] = index
            if len(running) >= threads:
                break
        while running:
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                answers[running.pop(future)] = future.result()
                advance()
                for index, item in queued:
                    running[pool.submit(work, item)] = index
                    break
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return answers

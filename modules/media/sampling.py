"""Choosing which frames of a sequence to keep.

The image sampler and the video sampler pick indices the same way. ``head``, ``center``
and ``tail`` answer consecutive frames.
"""

from __future__ import annotations

from .. import deps, log

__all__ = [
    "CONTIGUOUS",
    "STRATEGIES",
    "decode_frames",
    "describe",
    "frame_indices",
    "frame_span",
    "scatter",
    "slice_bounds",
]

logger = log.get_logger("media.sampling")

#: Every strategy the samplers offer, in the order their combo lists them.
STRATEGIES = ("uniform", "head", "center", "tail", "random", "every_nth")

#: The strategies whose frames are consecutive.
CONTIGUOUS = frozenset({"head", "center", "tail"})

#: The multiplier, increment and modulus of the generator ``random`` draws from. Numerical
#: Recipes' 32-bit LCG, which the frame timeline in the browser reproduces exactly.
LCG_MULTIPLIER = 1664525
LCG_INCREMENT = 1013904223
LCG_MODULUS = 0x100000000


def scatter(total: int, count: int, seed: int) -> list[int]:
    """Pick ``count`` distinct frames out of ``total``, the same way every time.

    Args:
        total: Frames available.
        count: How many to pick, at most ``total``.
        seed: Seed for the draw.

    Returns:
        The chosen indices, ascending.
    """
    state = (seed ^ (seed >> 32)) % LCG_MODULUS

    def draw() -> int:
        nonlocal state
        state = (LCG_MULTIPLIER * state + LCG_INCREMENT) % LCG_MODULUS
        return state

    # A partial Fisher-Yates: each step swaps one more frame into place, so no frame is drawn
    # twice and the work is proportional to what is kept rather than to a retry loop.
    pool = list(range(total))
    for position in range(count):
        target = position + draw() % (total - position)
        pool[position], pool[target] = pool[target], pool[position]
    return sorted(pool[:count])


#: How each strategy reads in a sentence, before the step is added to it. ``every_nth`` is not
#: here: it says nothing without its step, so :func:`describe` spells that one out.
PHRASES = {
    "uniform": "evenly spaced",
    "head": "from the start",
    "center": "from the middle",
    "tail": "from the end",
    "random": "at random",
}


def _ordinal(value: int) -> str:
    """``7`` as ``7th``, for a sentence rather than a widget.

    Args:
        value: A positive number.

    Returns:
        The number with its English suffix.
    """
    if 10 <= value % 100 <= 20:
        return f"{value}th"
    return f"{value}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(value % 10, 'th') }"


def describe(strategy: str, nth: int = 1) -> str:
    """How a strategy reads in a sentence, with its step where it has one.

    Args:
        strategy: One of :data:`STRATEGIES`.
        nth: The step between the frames the strategy chose from.

    Returns:
        A phrase such as ``every 7th frame`` or ``from the middle, every 3rd frame``.
    """
    step = int(nth) if nth and nth > 1 else 1
    if strategy == "every_nth":
        return "every frame" if step == 1 else f"every {_ordinal(step)} frame"
    phrase = PHRASES.get(strategy, strategy)
    return phrase if step == 1 else f"{phrase}, every {_ordinal(step)} frame"


def frame_span(total: int, count: int, strategy: str) -> tuple[int, int]:
    """Where a consecutive run starts and how many frames it holds.

    Args:
        total: Frames available.
        count: Frames wanted.
        strategy: One of :data:`CONTIGUOUS`.

    Returns:
        The first frame's index and how many follow it.

    Raises:
        ValueError: The strategy does not answer consecutive frames.
    """
    count = max(1, min(count, total))
    if strategy == "head":
        return 0, count
    if strategy == "tail":
        return total - count, count
    if strategy == "center":
        return (total - count) // 2, count
    raise ValueError(f"{strategy!r} does not answer consecutive frames")


def slice_bounds(total: int, start: int, end: int) -> tuple[int, int]:
    """The half-open range a start and an end name, held inside a sequence.

    Args:
        total: Frames available.
        start: First frame to consider, counting from 0. Negative counts back from the end.
        end: Last frame to consider, inclusive. -1 means the final frame, and negatives
            count back from the end as they do in a python slice.

    Returns:
        ``(first, stop)`` as a half-open range. An end before the start, or a range with
        nothing in it, answers the whole sequence, so a mistyped bound samples everything
        rather than failing a prompt part way through a batch.

    Raises:
        ValueError: ``total`` is not positive, so there is no sequence to bound.
    """
    if total <= 0:
        raise ValueError("there are no frames to bound")
    first = start + total if start < 0 else start
    last = end + total if end < 0 else end
    first = max(0, min(first, total - 1))
    last = max(0, min(last, total - 1))
    if last < first:
        return 0, total
    return first, last + 1


def _picked(total: int, count: int, strategy: str, seed: int) -> list[int]:
    """Which positions of a pool a strategy keeps, ascending.

    Args:
        total: Positions available.
        count: The most to keep.
        strategy: One of :data:`STRATEGIES`.
        seed: Seed for ``random``.

    Returns:
        Positions, ascending, never more than ``count`` of them.
    """
    count = max(1, min(count, total))
    if strategy in CONTIGUOUS:
        start, taken = frame_span(total, count, strategy)
        return list(range(start, start + taken))
    if strategy == "uniform":
        # One frame from the middle rather than from the start, so a single-frame sample of
        # a clip that fades up is not simply black.
        if count == 1:
            return [total // 2]
        return [round(i * (total - 1) / (count - 1)) for i in range(count)]
    if strategy == "random":
        return scatter(total, count, seed)
    return list(range(total))[:count]


def frame_indices(
    total: int, count: int, strategy: str, nth: int = 1, seed: int = 0,
) -> list[int]:
    """Which frames a strategy keeps, ascending.

    Args:
        total: Frames available.
        count: The most frames to keep.
        strategy: One of :data:`STRATEGIES`.
        nth: Step between the frames a strategy may choose from, 1 for all of them.
        seed: Seed for ``random``, so a graph re-runs to the same frames.

    Returns:
        Frame indices, ascending, never more than ``count`` of them.

    Raises:
        ValueError: The strategy is unknown, or there is nothing to sample.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}, expected one of {STRATEGIES}")
    if total <= 0:
        raise ValueError("there are no frames to sample")
    # The step thins the frames first and the strategy then chooses among what is left, so a
    # step means the same thing whichever strategy is reading it: `head` with a step of 2 is
    # the opening of the clip on every other frame.
    pool = list(range(0, total, max(1, int(nth))))
    return [pool[position] for position in _picked(len(pool), count, strategy, seed)]


def decode_frames(video, indices: list[int], name: str = ""):
    """Decode only the listed frames of a video, and answer them as a video.

    Args:
        video: The source, a ComfyUI ``VideoInput``.
        indices: Frame numbers to keep, ascending.
        name: Display name of the calling node, for the log and the refusal.

    Returns:
        A video holding those frames at the source's frame rate.

    Raises:
        DependencyError: PyAV is absent.
        ValueError: The video ended before the last frame asked for.
        MemoryError: Neither free memory nor a scratch drive can hold the frames.
    """
    import torch
    from comfy_api.latest import InputImpl, Types

    from ..image import scratch

    av = deps.require("av")

    slots: dict[int, list[int]] = {}
    for slot, number in enumerate(indices):
        slots.setdefault(number, []).append(slot)
    last = indices[-1]
    images = None
    kept: set[int] = set()
    # One pass in presentation order, stopping at the last frame asked for.
    with av.open(video.get_stream_source(), mode="r") as container:
        for number, frame in enumerate(container.decode(container.streams.video[0])):
            if number in slots:
                plane = torch.from_numpy(frame.to_ndarray(format="rgb24").copy())
                if images is None:
                    images = scratch.allocate(
                        (len(indices),) + tuple(plane.shape),
                        node=name,
                        advice="Lowering num_frames also fits it.",
                    )
                for slot in slots[number]:
                    images[slot].copy_(plane).div_(255.0)
                kept.add(number)
            if number >= last:
                break

    missing = [i for i in indices if i not in kept]
    if missing:
        # A container whose header count is higher than what it actually decodes.
        raise ValueError(
            f"the video ended before frame {missing[0]}: asked for {len(indices)} frame(s) "
            f"and {len(kept)} could be decoded"
        )

    return InputImpl.VideoFromComponents(
        Types.VideoComponents(images=images, frame_rate=video.get_frame_rate())
    )

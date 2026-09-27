"""What one picture holds, and what a node did to it.

:func:`publish` files the measurements under the node's own id. :func:`drift` measures a
picture against the one it came from, in 8-bit codes.
"""

from __future__ import annotations

from .. import log
from . import run_result

__all__ = ["CODES", "drift", "publish"]

logger = log.get_logger("interface.image_report")

#: Levels an 8-bit file holds, which every difference is reported in.
CODES = 255.0

#: Difference below which two pictures are called the same, in codes. Half a code is what
#: rounding to 8 bits can move a sample by on its own.
SAME = 0.5

#: Channel names, in the order an image holds them.
CHANNELS = ("red", "green", "blue")


def drift(before, after) -> dict:
    """Measure one picture against the one it was made from.

    Args:
        before: ``(height, width, channels)`` tensor, the picture as it arrived.
        after: A tensor of the same shape, the picture as it left.

    Returns:
        A mapping of ``worst``, ``mean`` and one signed shift per channel, all in codes, and
        ``moved`` saying whether anything moved further than half a code. Empty when the two
        cannot be compared.
    """
    try:
        import torch

        first = before[..., :3].float()
        second = after[..., :3].float()
        if first.shape != second.shape:
            return {}
        gap = (second - first) * CODES
        shifts = {
            name: float(gap[..., index].mean())
            for index, name in enumerate(CHANNELS)
            if index < gap.shape[-1]
        }
        worst = float(torch.abs(gap).max())
        return {
            "worst": worst,
            "mean": float(torch.abs(gap).mean()),
            "moved": worst > SAME,
            **shifts,
        }
    except Exception as error:
        logger.debug("the drift could not be measured (%s)", error)
        return {}


def _spell(codes: float) -> str:
    """One difference, in codes, written the way a reader compares it with 255."""
    return f"{codes:+.2f}" if abs(codes) < 10 else f"{codes:+.1f}"


def publish(image, facts=None, moved=None, summary="", node_id=None, depth=8) -> bool:
    """Store what one picture holds, for the publishing node's own interface to fetch.

    Never raises, and never touches the values it is given.

    Args:
        image: ``(batch, height, width, channels)`` or one image, on a 0 to 1 scale.
        facts: Named strings shown as rows, merged after the report's own.
        moved: What :func:`drift` answered, or None where nothing was compared. A picture
            that moved for no stated reason draws the panel in the warning colour.
        summary: One line saying what the node did. Left out, the size and the range.
        node_id: The publishing node's graph id.
        depth: Bits a sample the picture was stored at, which the entropy is counted in.

    Returns:
        Whatever :func:`run_result.publish` answered, which is False when no browser is
        connected and the readout is not worth building.
    """
    if not run_result.watching():
        return False
    try:
        from ..image import statistics

        frame = image[0] if getattr(image, "ndim", 0) == 4 else image
        height, width = (int(size) for size in frame.shape[:2])
        channels = int(frame.shape[2]) if frame.ndim > 2 else 1
        measured = statistics.measure(frame.float().clamp(0.0, 1.0), bins=1 << int(depth))
        low, high = float(frame.min()), float(frame.max())

        counts = {
            "mean": round(measured["mean"] * CODES, 1),
            "contrast": round(measured["contrast"] * CODES, 1),
            "clipped %": round(
                (measured["clipped_shadows"] + measured["clipped_highlights"]) * 100.0, 2
            ),
        }
        # run_result keeps eight rows, so the report writes four of its own at most and
        # leaves the rest to the node that knows what else is worth saying.
        rows = {
            "size": (
                f"{width} x {height} "
                f"{ {1: 'grey', 3: 'RGB', 4: 'RGBA'}.get(channels, str(channels) + ' channels') }"
            ),
            "range": f"{low:.4g} to {high:.4g}, {measured['entropy']:.2f} bits of {int(depth)}",
        }
        if moved:
            rows["drift"] = f"worst {_spell(moved['worst'])}, mean {moved['mean']:.2f}"
            rows["by channel"] = "  ".join(
                f"{name[0].upper()} {_spell(moved[name])}"
                for name in CHANNELS if name in moved
            )
        rows.update(facts or {})

        line = summary or f"{width} x {height}, mean {counts['mean']:.0f} of 255"
        return run_result.publish(
            # A picture that moved with nothing to account for it is the reading worth seeing.
            status=run_result.WARNING if moved and moved.get("unexplained") else run_result.OK,
            summary=line,
            counts=counts,
            facts=rows,
            node_id=node_id,
        )
    except Exception as error:
        logger.debug("no image report was published (%s)", error)
        return False

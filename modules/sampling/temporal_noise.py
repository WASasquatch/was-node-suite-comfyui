"""Starting noise whose pattern carries along a video latent's time axis.

A stream shaped ``[B, C, T, H, W]`` is filtered along ``T``. Any other rank is left as
drawn. Hold lengths are in latent frames.
"""

from __future__ import annotations

import math

import torch

#: Longest hold a caller may ask for, in latent frames.
MAX_HOLD = 128.0


def hold_to_rho(hold: float) -> float:
    """Turn a hold length into the one-pole coefficient that produces it.

    Args:
        hold: How many latent frames the pattern carries over, 0.0 or more.

    Returns:
        A coefficient from 0.0, which draws ordinary noise, towards 1.0 as the hold grows.
    """
    hold = float(hold)
    if hold <= 0.0:
        return 0.0
    return math.exp(-1.0 / hold)


def correlate_time(noise: torch.Tensor, rho: float) -> torch.Tensor:
    """Carry each frame of a noise draw forward into the next.

    Args:
        noise: A ``[B, C, T, H, W]`` standard normal draw.
        rho: A coefficient from :func:`hold_to_rho`.

    Returns:
        A tensor of the same shape and dtype, mean 0.0 and variance 1.0 at every frame.
        The draw itself where ``rho`` is 0.0, the stream has no time axis, or it holds one
        frame.
    """
    if rho <= 0.0 or noise.ndim != 5 or noise.shape[2] < 2:
        return noise
    fresh = math.sqrt(1.0 - rho * rho)
    out = noise.to(dtype=torch.float32).clone()
    for frame in range(1, out.shape[2]):
        out[:, :, frame].mul_(fresh).add_(out[:, :, frame - 1], alpha=rho)
    return out.to(dtype=noise.dtype)


def measure_time_stats(noise: torch.Tensor) -> dict:
    """Measure how far a noise stream carries along its time axis.

    Args:
        noise: A ``[B, C, T, H, W]`` tensor, or one of another rank.

    Returns:
        ``correlation`` between neighbouring frames, ``frame_change`` as a share of what an
        independent draw changes by, and ``spread`` as the standard deviation over every
        element.
    """
    work = noise.to(dtype=torch.float32)
    spread = float(work.std())
    if work.ndim != 5 or work.shape[2] < 2:
        return {"correlation": 0.0, "frame_change": 1.0, "spread": spread}
    head = work[:, :, :-1]
    tail = work[:, :, 1:]
    return {
        "correlation": float((head * tail).mean()),
        "frame_change": float((tail - head).pow(2).mean().sqrt() / math.sqrt(2.0)),
        "spread": spread,
    }


def shape_label(tensor: torch.Tensor) -> str:
    """Write a tensor's shape the way the report reads it.

    Args:
        tensor: Any tensor.

    Returns:
        The dimensions joined by ``x``.
    """
    return " x ".join(str(size) for size in tensor.shape)


class TemporalNoiseHold:
    """A noise source whose pattern carries from one video frame to the next.

    Attributes:
        seed: The seed the draw is taken from.
        hold: How many latent frames the pattern carries over.
        node_id: The node any report is published on.
    """

    def __init__(self, seed, hold, node_id=None):
        self.seed = int(seed)
        self.hold = float(hold)
        self.node_id = node_id

    def generate_noise(self, input_latent: dict):
        """Draw the starting noise and carry every video stream along its time axis.

        Args:
            input_latent: A latent carrying ``samples`` and any ``batch_index``.

        Returns:
            A tensor, or a NestedTensor holding one per stream, shaped as the latent is.
        """
        import comfy.sample

        samples = input_latent["samples"]
        batch_index = input_latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(samples, self.seed, batch_index)
        rho = hold_to_rho(self.hold)

        if getattr(noise, "is_nested", False):
            import comfy.nested_tensor

            parts = list(noise.unbind())
            shaped = [correlate_time(part, rho) for part in parts]
            result = comfy.nested_tensor.NestedTensor(tuple(shaped))
        else:
            parts = [noise]
            shaped = [correlate_time(noise, rho)]
            result = shaped[0]

        self._report(rho, parts, shaped)
        return result

    def _report(self, rho: float, drawn: list, shaped: list) -> None:
        """Publish what each stream came out as.

        Args:
            rho: The coefficient the hold worked out to.
            drawn: One tensor per stream as it was drawn.
            shaped: The same streams after filtering.
        """
        from ..interface import run_result

        held = [part for part in drawn if part.ndim == 5 and part.shape[2] > 1]
        facts = {"seed": self.seed}
        for index, (before, after) in enumerate(zip(drawn, shaped)):
            carried = before.ndim == 5 and before.shape[2] > 1
            name = f"stream {index}" if len(drawn) > 1 else "stream"
            facts[name] = f"{shape_label(after)}, {'held' if carried else 'left alone'}"

        counts = {"hold": round(self.hold, 2), "aimed for": round(rho, 4)}
        if held:
            measured = measure_time_stats(shaped[drawn.index(held[0])])
            counts["correlation"] = round(measured["correlation"], 4)
            counts["frame change"] = round(measured["frame_change"], 4)
            counts["spread"] = round(measured["spread"], 4)

        if not held:
            status = run_result.WARNING
            summary = "no time axis: ordinary noise"
        elif rho <= 0.0:
            status = run_result.OK
            summary = f"hold 0: ordinary noise, {len(held)} stream(s)"
        else:
            status = run_result.OK
            summary = (
                f"hold {self.hold:g}, frame change "
                f"{counts.get('frame change', 1.0):.2f}, {len(held)} stream(s)"
            )

        run_result.publish(
            status=status,
            summary=summary,
            counts=counts,
            facts=facts,
            node_id=self.node_id,
        )

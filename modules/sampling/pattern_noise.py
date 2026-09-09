"""Starting noise shaped by an affine mask pattern.

The mask multiplies the draw where it is white and adds an offset there, through
:func:`modules.latent.affine.apply_plane`. Patterns read off a picture are not available.
"""

from __future__ import annotations

import torch

from ..latent import affine
from ..latent import affine_patterns as patterns

#: Patterns a noise draw cannot use: the four read off a picture, a mask wired in from
#: outside, and the one that covers everything.
UNAVAILABLE = (*patterns.CONTENT_PATTERNS, "external_mask", "solid")

#: Patterns the node offers, in the order they appear on the affine nodes.
NOISE_PATTERNS = [name for name in patterns.PATTERNS if name not in UNAVAILABLE]


def unit_variance(noise: torch.Tensor) -> torch.Tensor:
    """Recentre a tensor on zero and rescale it to a standard deviation of one.

    Args:
        noise: Any tensor.

    Returns:
        A tensor of the same shape and dtype, mean 0.0 and standard deviation 1.0. The
        tensor as it was where its values are all the same.
    """
    work = noise.to(dtype=torch.float32)
    spread = float(work.std())
    if not spread > 1e-8:
        return noise
    return ((work - float(work.mean())) / spread).to(dtype=noise.dtype)


class PatternNoise:
    """A noise source whose amplitude and offset follow a generated mask.

    Attributes:
        seed: The seed the draw and the mask are taken from.
        pattern: Which mask the draw is shaped by.
        max_scale: What the draw is multiplied by where the mask is white.
        max_bias: What is added where the mask is white.
        temporal_mode: How the mask varies across the frames of a video latent.
        streams: Which streams of a packed latent are shaped.
        normalize: Whether the result is returned at a standard deviation of one.
        clamp_sigma: Where the result is cut off, or 0.0 to leave it uncut.
        options: Pattern parameters and mask shaping.
        node_id: The node any report is published on.
    """

    def __init__(
        self,
        seed,
        pattern="white_noise",
        max_scale=1.1,
        max_bias=0.0,
        temporal_mode="static",
        streams="video",
        normalize=True,
        clamp_sigma=0.0,
        options=None,
        node_id=None,
    ):
        self.seed = int(seed)
        self.pattern = str(pattern)
        self.max_scale = float(max_scale)
        self.max_bias = float(max_bias)
        self.temporal_mode = str(temporal_mode)
        self.streams = str(streams)
        self.normalize = bool(normalize)
        self.clamp_sigma = float(clamp_sigma)
        self.options = options
        self.node_id = node_id

    @property
    def is_noop(self) -> bool:
        """Whether the settings leave the draw exactly as it was."""
        return abs(self.max_scale - 1.0) < 1e-8 and abs(self.max_bias) < 1e-8

    def generate_noise(self, input_latent: dict):
        """Draw the starting noise and shape the selected streams by the mask.

        Args:
            input_latent: A latent carrying ``samples`` and any ``batch_index``.

        Returns:
            A tensor, or a NestedTensor holding one per stream, shaped as the latent is.
        """
        import comfy.sample

        samples = input_latent["samples"]
        noise = comfy.sample.prepare_noise(
            samples, self.seed, input_latent.get("batch_index", None)
        )

        nested = bool(getattr(noise, "is_nested", False))
        parts = list(noise.unbind()) if nested else [noise]
        wanted = affine.stream_indices(self.streams, len(parts))

        shaped = list(parts)
        touched = []
        for index in wanted:
            if index >= len(parts):
                continue
            part = parts[index]
            if not torch.is_tensor(part) or part.ndim not in (3, 4, 5):
                continue
            if self.is_noop:
                continue
            out, _mask = affine.apply_plane(
                part,
                self.max_scale,
                self.max_bias,
                self.pattern,
                self.temporal_mode,
                self.seed + index,
                options=self.options,
            )
            if self.normalize:
                out = unit_variance(out)
            if self.clamp_sigma > 0.0:
                out = out.clamp(-self.clamp_sigma, self.clamp_sigma)
            shaped[index] = out
            touched.append(index)

        self._report(parts, shaped, touched)
        if not nested:
            return shaped[0]

        import comfy.nested_tensor

        return comfy.nested_tensor.NestedTensor(tuple(shaped))

    def _report(self, drawn: list, shaped: list, touched: list) -> None:
        """Publish what each stream came out as.

        Args:
            drawn: One tensor per stream as it was drawn.
            shaped: The same streams after the mask.
            touched: Indices of the streams the mask reached.
        """
        from ..interface import run_result

        facts = {"seed": self.seed, "pattern": self.pattern}
        for index, after in enumerate(shaped):
            name = f"stream {index}" if len(shaped) > 1 else "stream"
            size = " x ".join(str(dim) for dim in after.shape)
            facts[name] = f"{size}, {'shaped' if index in touched else 'left alone'}"

        counts = {"max scale": round(self.max_scale, 4), "max bias": round(self.max_bias, 4)}
        if touched:
            work = shaped[touched[0]].to(dtype=torch.float32)
            before = drawn[touched[0]].to(dtype=torch.float32)
            counts["spread"] = round(float(work.std()), 4)
            counts["mean"] = round(float(work.mean()), 4)
            counts["changed by %"] = round(
                100.0 * float((work - before).abs().mean() / before.abs().mean().clamp_min(1e-8)),
                2,
            )

        if self.is_noop:
            status = run_result.WARNING
            summary = "max_scale 1.0, max_bias 0.0: ordinary noise"
        elif not touched:
            status = run_result.WARNING
            summary = f"streams '{self.streams}': no stream matched"
        else:
            status = run_result.OK
            summary = (
                f"{self.pattern} on {len(touched)} stream(s), spread "
                f"{counts.get('spread', 1.0):.2f}"
            )

        run_result.publish(
            status=status,
            summary=summary,
            counts=counts,
            facts=facts,
            node_id=self.node_id,
        )

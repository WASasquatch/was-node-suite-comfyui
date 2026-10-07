"""Spectral Progressive Diffusion: when to grow a latent, and by how much.

:func:`sample_speed` runs the early part of a schedule on a smaller grid and grows it
partway through. :func:`fit_power_spectrum` measures ``A`` and ``beta`` from a latent.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch

from . import spectral

__all__ = [
    "align_timestep",
    "conditioned_on_a_picture",
    "flow_shift",
    "packed_shapes",
    "delta_optimal_transitions",
    "fit_power_spectrum",
    "parse_scales",
    "parse_sigmas",
    "spatial_axes_or_refuse",
    "validate_scales",
]

#: Measured ``(A, beta)`` for the two models the paper profiled, offered as starting points.
#: Anything else should be measured with :func:`fit_power_spectrum` rather than guessed at, and
#: these two are only correct for the exact checkpoints named.
PRESETS = {
    "FLUX.1-dev": (203.615097, 1.915461),
    "Wan 2.1 T2V-1.3B": (219.484718, 2.422687),
}


def align_timestep(t: float, r: float) -> float:
    """The flow-matching time to carry on from once the latent has been grown.

    Args:
        t: Time at the transition.
        r: Ratio of the new scale to the old one.

    Returns:
        The realigned time.
    """
    return t * spectral.kappa(t, r)


def activation_time(power: float, delta: float) -> float:
    """The time a frequency stops being buried in noise and starts being worth resolving.

    Args:
        power: The power spectrum at the frequency in question.
        delta: How much residual noise counts as burying the frequency.

    Returns:
        The activation time, between 0 and 1, read against a sigma schedule running down from 1.
        A larger ``delta`` returns a smaller time, which a decreasing schedule reaches later, so
        raising it holds the coarse grid for more steps.

    Raises:
        ValueError: ``delta`` is 1 or above, which asks for a frequency to activate before the
            trajectory carries any signal at all.
    """
    if delta >= 1.0:
        raise ValueError(f"delta has to be below 1, not {delta}.")
    return 1.0 / (1.0 + math.sqrt(delta / (power * (1.0 + power - delta))))


def validate_scales(scales: Sequence[float]) -> None:
    """Check a scale list is usable.

    Args:
        scales: Fractions of the full resolution, increasing, ending at 1.0.

    Raises:
        ValueError: The list is empty, does not end at 1.0, is not strictly increasing, or holds
            a value outside ``(0, 1]``.
    """
    if not scales:
        raise ValueError("scales is empty; it needs at least the final 1.0.")
    if any(not 0.0 < value <= 1.0 for value in scales):
        raise ValueError(f"every scale has to be above 0 and at most 1.0; got {list(scales)}.")
    if scales[-1] != 1.0:
        raise ValueError(
            f"scales has to end at 1.0, so sampling finishes at full resolution; "
            f"got {list(scales)}."
        )
    for earlier, later in zip(scales[:-1], scales[1:]):
        if earlier >= later:
            raise ValueError(f"scales has to increase strictly; got {list(scales)}.")


def parse_scales(text: str) -> list[float]:
    """Read a comma-separated scale list.

    Args:
        text: Something like ``0.5,1.0``.

    Returns:
        The scales.

    Raises:
        ValueError: A value is not a number, or the list fails :func:`validate_scales`.
    """
    try:
        scales = [float(part) for part in text.split(",") if part.strip()]
    except ValueError as bad:
        raise ValueError(f"scales must be comma-separated numbers; got {text!r}.") from bad
    validate_scales(scales)
    return scales


def parse_sigmas(text: str) -> list[float]:
    """Read a comma-separated list of manual transition sigmas.

    Args:
        text: Something like ``0.85`` or ``0.95,0.85``.

    Returns:
        The sigmas.

    Raises:
        ValueError: A value is not a number, sits outside ``(0, 1)``, or the list does not
            decrease strictly.
    """
    try:
        sigmas = [float(part) for part in text.split(",") if part.strip()]
    except ValueError as bad:
        raise ValueError(f"the sigmas must be comma-separated numbers; got {text!r}.") from bad
    if any(not 0.0 < value < 1.0 for value in sigmas):
        raise ValueError(f"every sigma has to be between 0 and 1; got {sigmas}.")
    for earlier, later in zip(sigmas[:-1], sigmas[1:]):
        if earlier <= later:
            raise ValueError(f"the sigmas have to decrease strictly; got {sigmas}.")
    return sigmas


def delta_optimal_transitions(
    scales: Sequence[float],
    delta: float,
    amplitude: float,
    beta: float,
    height: int,
    width: int,
) -> list[float]:
    """The time to grow at, for each step up through ``scales``.

    Args:
        scales: Fractions of full resolution, as :func:`validate_scales` accepts.
        delta: How much residual noise counts as burying a frequency.
        amplitude: The ``A`` of the power spectrum.
        beta: The decay exponent of the power spectrum.
        height: Full-resolution latent height.
        width: Full-resolution latent width.

    Returns:
        One time per transition, so one fewer than there are scales.
    """
    validate_scales(scales)
    # The highest frequency the full grid can represent at all.
    nyquist = min(height, width) / 2.0
    times = []
    for scale in scales[:-1]:
        omega = scale * nyquist
        times.append(activation_time(amplitude * abs(omega) ** -beta, delta))
    return times


def spatial_axes_or_refuse(x: torch.Tensor) -> None:
    """Check a latent is laid out as a grid this can grow.

    Args:
        x: The latent the sampler was handed.

    Raises:
        ValueError: The latent is not 4- or 5-dimensional, or its trailing axes are too small to
            be a spatial grid.
    """
    # Growth is over the trailing two axes, true of an image model's (batch, channel, height,
    # width) and a video model's (batch, channel, frame, height, width). A model handing its
    # sampler a flat token sequence has no grid, and would be reshaped into nonsense untested.
    if x.ndim not in (4, 5):
        raise ValueError(
            f"SPEED grows a latent over its height and width, so it needs a 4-D image latent or "
            f"a 5-D video latent, not one shaped {tuple(x.shape)}. A model whose latent is a "
            f"flat token sequence cannot be sampled this way."
        )
    if min(x.shape[-2], x.shape[-1]) < 2:
        raise ValueError(
            f"the latent's trailing axes are {x.shape[-2]}x{x.shape[-1]}, which is too small to "
            f"be a spatial grid. SPEED needs a latent laid out with height and width last."
        )


def fit_power_spectrum(
    latent: torch.Tensor,
    low: float = 0.05,
    high: float = 0.5,
) -> tuple[float, float]:
    """Measure ``(A, beta)`` from a latent, by fitting a power law to its radial spectrum.

    Args:
        latent: A latent whose trailing two axes are spatial. Any leading axes are averaged
            over, so a batch or a whole video improves the estimate.
        low: Start of the band to fit, as a fraction of the Nyquist frequency. Below this the
            spectrum is dominated by a handful of coefficients and is not a power law.
        high: End of the band to fit, as a fraction of Nyquist. Near Nyquist the spectrum rolls
            off for reasons that have nothing to do with the content.

    Returns:
        The amplitude and decay exponent, as ``P(omega) = A * omega ** -beta``.

    Raises:
        ValueError: The latent has no spatial grid, or the band leaves too few frequencies to
            fit a line through.
    """
    spatial_axes_or_refuse(latent)
    work = latent.detach().to(torch.float32)
    if work.ndim == 5:
        batch, channels, frames, height, width = work.shape
        work = work.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)

    height, width = work.shape[-2], work.shape[-1]
    power = torch.fft.fft2(work, norm="ortho").abs().square()
    power = torch.fft.fftshift(power, dim=(-2, -1)).mean(dim=tuple(range(work.ndim - 2)))

    # Distance of every coefficient from DC, in the same units the transitions use: a radial
    # index running out to min(height, width) / 2 at Nyquist.
    rows = torch.arange(height, device=work.device, dtype=torch.float32) - height // 2
    columns = torch.arange(width, device=work.device, dtype=torch.float32) - width // 2
    radius = torch.sqrt(rows[:, None] ** 2 + columns[None, :] ** 2)

    nyquist = min(height, width) / 2.0
    keep = (radius >= low * nyquist) & (radius <= high * nyquist)
    if int(keep.sum()) < 8:
        raise ValueError(
            f"a {height}x{width} latent leaves only {int(keep.sum())} frequencies between "
            f"{low:g} and {high:g} of Nyquist, which is too few to fit a spectrum through. "
            f"Measure from a larger latent."
        )

    # Averaged into radial bins first, which keeps the fit off the coefficient count of the
    # high frequencies.
    bins = radius[keep].round().long()
    totals = torch.zeros(int(bins.max()) + 1, device=work.device, dtype=torch.float64)
    counts = torch.zeros_like(totals)
    totals.index_add_(0, bins, power[keep].double())
    counts.index_add_(0, bins, torch.ones_like(bins, dtype=torch.float64))
    filled = counts > 0
    omega = torch.arange(totals.numel(), device=work.device, dtype=torch.float64)[filled]
    mean_power = (totals[filled] / counts[filled])

    usable = (omega > 0) & (mean_power > 0)
    if int(usable.sum()) < 3:
        raise ValueError(
            "the latent's spectrum has too few non-empty frequency bands to fit through; "
            "measure from a larger or less uniform latent."
        )

    # A power law is a straight line once both axes are logarithmic, so this is least squares.
    log_omega = torch.log(omega[usable])
    log_power = torch.log(mean_power[usable])
    centred = log_omega - log_omega.mean()
    slope = (centred * (log_power - log_power.mean())).sum() / (centred * centred).sum()
    intercept = log_power.mean() - slope * log_omega.mean()
    return float(torch.exp(intercept)), float(-slope)


def base_sampler_names() -> list[str]:
    """The solvers a SPEED run can be built on.

    Returns:
        The solver names, sorted. Read from the running ComfyUI rather than frozen into the
        source, so a solver added by a later release is offered without a change here.
    """
    import comfy.k_diffusion.sampling as k_diffusion

    # Each segment is a fresh call into the solver, which a stateful solver survives by starting
    # its state again. These three cannot be segmented at all: two choose their own step count
    # from a tolerance instead of following the sigmas handed to them, and one carries a schedule.
    cannot_be_segmented = {"dpm_fast", "dpm_adaptive", "lcm"}
    prefix = "sample_"
    found = [name[len(prefix):] for name in dir(k_diffusion) if name.startswith(prefix)]
    return sorted(name for name in found if name not in cannot_be_segmented)


def flow_shift(model) -> float:
    """The shift the model's sampling applies between flow time and sigma.

    Args:
        model: The denoiser the sampler was handed.

    Returns:
        The shift, or 1.0 when the model does not declare one, which leaves sigma and flow time
        the same thing.
    """
    # A sampler is handed sigma, which for a flow-matching model is the flow time put through
    # shift * t / (1 + (shift - 1) * t). ModelSamplingSD3 sets that shift, commonly 8 for video,
    # where a flow time of 0.78 is a sigma of 0.97. Comparing a transition time against a raw
    # sigma compares two different quantities, which is what this exists to prevent.
    seen = set()
    node = model
    for _ in range(8):
        if node is None or id(node) in seen:
            break
        seen.add(id(node))
        sampling = getattr(node, "model_sampling", None)
        if sampling is not None:
            return float(getattr(sampling, "shift", 1.0) or 1.0)
        node = getattr(node, "inner_model", None)
    return 1.0


def sigma_of(t: float, shift: float) -> float:
    """The sigma a sampler is handed for flow time ``t``, matching ComfyUI's ``time_snr_shift``."""
    if shift == 1.0:
        return t
    return shift * t / (1.0 + (shift - 1.0) * t)


def time_of(sigma: float, shift: float) -> float:
    """The flow time behind a sigma, the inverse of :func:`sigma_of`."""
    if shift == 1.0:
        return sigma
    return sigma / (shift - (shift - 1.0) * sigma)

def packed_shapes(model) -> list | None:
    """The shapes of the latents packed into one, for a model that samples several at once.

    Args:
        model: The denoiser the sampler was handed.

    Returns:
        The live list ComfyUI recorded before packing, or ``None`` when the model samples a
        single latent and nothing was packed.
    """
    # ComfyUI flattens each latent to (batch, 1, -1) and concatenates them, keeping the original
    # shapes in a list it hands to the model and closes the preview callback over. The list
    # object is shared, so growing a latent means editing it in place rather than replacing it,
    # or the previews would keep unpacking against the old sizes.
    seen = set()
    node = model
    for _ in range(8):
        if node is None or id(node) in seen:
            break
        seen.add(id(node))
        shapes = getattr(node, "latent_shapes", None)
        if isinstance(shapes, list) and len(shapes) > 1:
            return shapes
        node = getattr(node, "inner_model", None)
    return None


def unpack(x: torch.Tensor, shapes: list | None) -> list[torch.Tensor]:
    """Split a packed latent back into the latents it carries.

    Args:
        x: The latent the sampler is working on.
        shapes: The shapes from :func:`packed_shapes`, or ``None`` for an unpacked latent.

    Returns:
        One tensor per packed latent, or ``[x]`` when nothing was packed.
    """
    if not shapes:
        return [x]
    import comfy.utils

    return comfy.utils.unpack_latents(x, shapes)


def repack(parts: list[torch.Tensor], shapes: list | None) -> torch.Tensor:
    """Put the latents back together and record their sizes.

    Args:
        parts: The tensors to pack, in the order they were unpacked.
        shapes: The live shape list to update in place, or ``None`` for an unpacked latent.

    Returns:
        The packed latent, or ``parts[0]`` when nothing was packed.
    """
    if not shapes:
        return parts[0]
    import comfy.utils

    packed, sizes = comfy.utils.pack_latents(parts)
    # In place: the preview callback holds this same list.
    shapes[:] = sizes
    return packed

def _transition_steps(
    sigmas: torch.Tensor,
    scales: Sequence[float],
    times: Sequence[float],
    shift: float = 1.0,
) -> list[tuple[int, float, float]]:
    """Turn transition times into the step each one lands on.

    Args:
        sigmas: The full sigma schedule the sampler was handed.
        scales: The scale list, as :func:`validate_scales` accepts.
        times: One transition time per step up through ``scales``.
        shift: The model's flow shift. A threshold in flow time is compared against the flow
            time behind each sigma rather than against the sigma itself. Pass 1.0 for a
            threshold that is already a sigma, which is what manual thresholds are.

    Returns:
        ``(step, scale before, scale after)`` per transition. Where one would land past the end
        of the schedule, it and every one after it land on the last step that did, and where
        none lands the list is empty.
    """
    steps = len(sigmas) - 1
    landed = []
    for before, after, time in zip(scales[:-1], scales[1:], times):
        step = next(
            (index for index in range(steps)
             if time_of(float(sigmas[index]), shift) <= time),
            steps,
        )
        if step >= steps:
            break
        landed.append((step, before, after))
    if landed:
        last = landed[-1][0]
        for before, after in zip(scales[len(landed):-1], scales[len(landed) + 1:]):
            landed.append((last, before, after))
    return landed


def _grow(
    x: torch.Tensor,
    before: float,
    after: float,
    t: float,
    transform: str,
    seed: int,
    height: int,
    width: int,
) -> tuple[torch.Tensor, float]:
    """Grow one latent across a transition and report the time to carry on from.

    Args:
        x: The latent at the transition, 4- or 5-dimensional.
        before: Scale the latent is currently at.
        after: Scale to grow to.
        t: Flow-matching time at the transition.
        transform: The spectral basis, one of :data:`~modules.sampling.spectral.TRANSFORMS`.
        seed: Seed for the noise filling the new frequencies.
        height: Full-resolution latent height.
        width: Full-resolution latent width.

    Returns:
        The grown latent and the realigned time.
    """
    ratio = after / before
    target = (round(after * height), round(after * width))

    if x.ndim == 5:
        # Video latents grow a frame at a time, so the frames join the batch for the transform
        # and are put back afterwards.
        batch, channels, frames = x.shape[0], x.shape[1], x.shape[2]
        planes = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, *x.shape[-2:])
        grown = spectral.expand(planes, target, t, transform, seed)
        grown = grown.reshape(batch, frames, channels, *target).permute(0, 2, 1, 3, 4)
    else:
        grown = spectral.expand(x, target, t, transform, seed)

    return grown * spectral.kappa(t, ratio), align_timestep(t, ratio)


def _rebased(callback, offset: int):
    """Shift a step callback so one segment reports against the whole schedule."""
    if callback is None:
        return None

    def shifted(status):
        status = dict(status)
        status["i"] = status.get("i", 0) + offset
        callback(status)

    return shifted


def _holder(model, attribute: str):
    """The first model in an inner-model chain carrying an attribute, or ``None``."""
    seen = set()
    node = model
    for _ in range(8):
        if node is None or id(node) in seen:
            return None
        seen.add(id(node))
        if getattr(node, attribute, None) is not None:
            return node
        node = getattr(node, "inner_model", None)
    return None


def _area(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """A mask averaged down to ``size`` over its trailing two axes."""
    height, width = mask.shape[-2], mask.shape[-1]
    if (height, width) == tuple(size):
        return mask
    flat = mask.reshape(-1, 1, height, width).to(torch.float32)
    shrunk = torch.nn.functional.interpolate(flat, size=tuple(size), mode="area")
    return shrunk.reshape(*mask.shape[:-2], *size).to(mask.dtype)


#: Conditioning keys a model derives from the denoise mask at the latent's own size.
MASK_CONDS = ("denoise_mask", "audio_denoise_mask")


class _Held:
    """A run's denoise mask, held picture, noise and keyframes, fitted to each grid.

    Attributes:
        active: Whether the run carries a denoise mask or MiniMax H3 keyframes.
        mask: The full-size denoise mask, or ``None``.
        latent: The full-size picture held under the mask, or ``None``.
        noise: The full-size noise held under the mask, or ``None``.
        conds: The full-size conditioning entries, by kind.
    """

    def __init__(self, model, extra_args: dict, shapes: list | None, height: int, width: int):
        self.extra_args = extra_args
        self.shapes = shapes
        self.full_shapes = [tuple(shape) for shape in shapes] if shapes else None
        self.height, self.width = height, width
        self.inpaint = _holder(model, "latent_image")
        self.guider = _holder(model, "conds")
        self.base = getattr(self.guider, "inner_model", None)
        self.mask = extra_args.get("denoise_mask")
        self.latent = getattr(self.inpaint, "latent_image", None)
        self.noise = getattr(self.inpaint, "noise", None)
        conds = getattr(self.guider, "conds", None)
        self.conds = (
            {kind: list(entries or ()) for kind, entries in conds.items()}
            if isinstance(conds, dict) else {}
        )
        keyframed = any(
            entry.get("minimax_keyframes") for entries in self.conds.values() for entry in entries
        )
        self.active = self.mask is not None or keyframed

    def refusal(self) -> str | None:
        """Why the run cannot be fitted to a smaller grid, or ``None`` where it can."""
        if not self.active:
            return None
        missing = []
        if self.mask is not None and self.inpaint is None:
            missing.append("the picture held under the denoise mask")
        if not self.conds or not hasattr(self.base, "extra_conds"):
            missing.append("the conditioning")
        if not missing:
            return None
        return (
            f"SPEED samples the early steps on a smaller latent, and this run's sampler does not "
            f"reach {' or '.join(missing)} to fit it to that size. Use an ordinary sampler for "
            f"it, or set scales to 1.0 to turn growth off and keep the rest of the schedule."
        )

    def _first(self, packed, fit):
        """Apply ``fit`` to the first stream of a tensor packed the way the latent is."""
        if packed is None:
            return None
        if not self.full_shapes:
            return fit(packed)
        import comfy.utils

        parts = unpack(packed, self.full_shapes)
        parts[0] = fit(parts[0])
        return comfy.utils.pack_latents(parts)[0]

    def _set(self, mask, latent, noise) -> None:
        """Hand the sampler a mask, held picture and noise."""
        if self.mask is not None:
            self.extra_args["denoise_mask"] = mask
        if self.inpaint is not None:
            if self.latent is not None:
                self.inpaint.latent_image = latent
            if self.noise is not None:
                self.inpaint.noise = noise

    def _fitted(self, entry: dict, scale: float, ratio: float) -> dict:
        """One conditioning entry with its keyframes on the smaller grid and no mask conds."""
        model_conds = {
            key: value for key, value in (entry.get("model_conds") or {}).items()
            if key not in MASK_CONDS
        }
        fitted = {**entry, "model_conds": model_conds}
        blocks = entry.get("minimax_keyframes")
        if blocks:
            # H3 lays keyframe rows out on the video's own grid.
            resized = []
            for block in blocks:
                latent = block.get("latent") if isinstance(block, dict) else None
                if torch.is_tensor(latent) and tuple(latent.shape[-2:]) == (self.height, self.width):
                    block = {**block, "latent": spectral.downscale(latent, scale) * ratio}
                resized.append(block)
            fitted["minimax_keyframes"] = resized
        return fitted

    def fit(self, scale: float, x: torch.Tensor) -> None:
        """Fit everything held to the grid ``scale`` makes, or restore it at 1.0.

        Args:
            scale: Fraction of the full resolution the latent is at.
            x: The latent at that scale, packed the way the sampler carries it.
        """
        if not self.active:
            return
        if scale >= 1.0:
            self.restore()
            return
        import comfy.samplers

        size = (round(self.height * scale), round(self.width * scale))
        # A clean picture keeps its level on the smaller grid; noise keeps its variance.
        ratio = math.sqrt(size[0] * size[1] / (self.height * self.width))
        mask = self._first(self.mask, lambda part: _area(part, size))
        latent = self._first(self.latent, lambda part: spectral.downscale(part, scale) * ratio)
        noise = self._first(self.noise, lambda part: spectral.downscale(part, scale))
        self._set(mask, latent, noise)

        shapes = self.shapes if self.shapes else [x.shape]
        for kind, entries in self.conds.items():
            self.guider.conds[kind] = comfy.samplers.encode_model_conds(
                self.base.extra_conds,
                [self._fitted(entry, scale, ratio) for entry in entries],
                noise if noise is not None else x,
                x.device,
                kind,
                latent_image=latent,
                denoise_mask=mask,
                seed=self.extra_args.get("seed"),
                latent_shapes=shapes,
            )

    def restore(self) -> None:
        """Hand the sampler back everything held at full size."""
        if not self.active:
            return
        self._set(self.mask, self.latent, self.noise)
        if self.guider is not None:
            for kind, entries in self.conds.items():
                self.guider.conds[kind] = entries


def conditioned_on_a_picture(model) -> list[str]:
    """Conditioning entries that carry a picture at the latent's own size.

    Args:
        model: The denoiser the sampler was handed, which reaches the guider holding the conds.

    Returns:
        The conditioning keys that carry a picture, deduplicated and sorted. Empty for a model
        conditioned only on text, and empty when the conds cannot be reached, since refusing on
        a reading this cannot make would refuse the runs that work.
    """
    import comfy.conds

    guider = model
    conds = None
    for _ in range(8):
        if guider is None:
            break
        found = getattr(guider, "conds", None)
        if isinstance(found, dict):
            conds = found
            break
        guider = getattr(guider, "inner_model", None)
    if not conds:
        return []

    named = set()
    for group in conds.values():
        for entry in group or ():
            for key, value in (entry.get("model_conds") or {}).items():
                # CONDNoiseShape is the conditioning that is a picture: it is narrowed to an
                # area and repeated to a batch, never resampled, so the model concatenates it
                # to the latent at whatever size it was built at.
                if isinstance(value, comfy.conds.CONDNoiseShape):
                    named.add(key)
    return sorted(named)


@torch.no_grad()
def sample_speed(
    model,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    extra_args=None,
    callback=None,
    disable=None,
    *,
    transform: str = "dct",
    base_sampler: str = "euler",
    scales: Sequence[float] = (),
    manual_sigmas: Sequence[float] = (),
    delta: float = 0.01,
    amplitude: float = 203.615097,
    beta: float = 1.915461,
    seed: int = 0,
):
    """Sample ``x``, growing its resolution partway through.

    Args:
        model: The denoiser, as ComfyUI hands it over.
        x: The latent to sample, at full resolution. It is truncated to the first scale here.
        sigmas: The full sigma schedule.
        extra_args: Passed through to the underlying solver.
        callback: Step callback, re-based so its step numbers count across the whole schedule
            rather than restarting at every segment.
        disable: Progress-bar suppression, passed through.
        transform: The spectral basis growth happens in.
        base_sampler: Which solver to run inside each segment.
        scales: Fractions of full resolution to sample at, increasing, ending at 1.0. Fewer than
            two means there is nothing to grow, and the solver is called once, unchanged.
        manual_sigmas: Sigma thresholds to transition at. When empty the thresholds are computed
            from ``delta``, ``amplitude`` and ``beta`` instead.
        delta: How much residual noise counts as burying a frequency.
        amplitude: The ``A`` of the model's power spectrum.
        beta: The decay exponent of the model's power spectrum.
        seed: Seed for the noise filling new frequencies at each transition.

    Returns:
        The sampled latent, at full resolution.

    Raises:
        ValueError: The solver is unknown, the latent has no spatial grid, ``manual_sigmas``
            does not carry one threshold per transition, the model takes a picture as an extra
            input, or a masked run's held picture or conditioning cannot be reached.
    """
    import comfy.k_diffusion.sampling as k_diffusion

    solver = getattr(k_diffusion, "sample_" + base_sampler, None)
    if solver is None:
        raise ValueError(f"there is no sampler called {base_sampler!r} in this ComfyUI.")

    extra_args = {} if extra_args is None else extra_args
    scales = list(scales)
    if len(scales) < 2:
        return solver(
            model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable
        )

    # A model that generates more than one thing at once, such as video with its own audio, is
    # handed every latent flattened into one tensor. Only the first carries a picture; it is
    # the one grown and the rest travel untouched.
    shapes = packed_shapes(model)
    parts = unpack(x, shapes)
    spatial_axes_or_refuse(parts[0])
    height, width = parts[0].shape[-2], parts[0].shape[-1]

    shift = flow_shift(model)
    if manual_sigmas:
        times = list(manual_sigmas)
        if len(times) != len(scales) - 1:
            raise ValueError(
                f"{len(scales)} scales need {len(scales) - 1} transition sigmas, but "
                f"{len(times)} were given."
            )
        # A manual threshold is given as a sigma, so it is already in the schedule's own terms.
        transitions = _transition_steps(sigmas, scales, times, shift=1.0)
    else:
        times = delta_optimal_transitions(scales, delta, amplitude, beta, height, width)
        transitions = _transition_steps(sigmas, scales, times, shift=shift)
    if not transitions:
        # Every transition would land past the end of the schedule, so there is nothing to be
        # gained by segmenting and the latent is never taken off full resolution.
        return solver(
            model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable
        )

    picture_conds = conditioned_on_a_picture(model)
    if picture_conds:
        raise ValueError(
            f"SPEED samples the early steps on a smaller latent, and this run gives the model a "
            f"picture at the full one ({', '.join(picture_conds)}) as an extra input, which is "
            f"built before sampling starts and is never resized to follow. Image to video and "
            f"inpainting models that take the picture this way are not supported. Use an "
            f"ordinary sampler for those, or set scales to 1.0 to turn growth off and keep the "
            f"rest of the schedule."
        )
    held = _Held(model, extra_args, shapes, height, width)
    refusal = held.refusal()
    if refusal:
        raise ValueError(refusal)

    parts[0] = spectral.downscale(parts[0], scales[0])
    x = repack(parts, shapes)
    sigmas = sigmas.clone()

    starts = [0] + [step for step, _, _ in transitions]
    try:
        held.fit(scales[0], x)
        for index, start in enumerate(starts):
            end = transitions[index][0] if index < len(transitions) else len(sigmas) - 1
            segment = sigmas[start : end + 1]
            if len(segment) >= 2:
                x = solver(
                    model, x, segment, extra_args=extra_args,
                    callback=_rebased(callback, start), disable=disable,
                )
            if index >= len(transitions):
                break

            step, before, after = transitions[index]
            # The growth and the realignment are both in flow time, so the sigma is converted
            # in and the answer converted back out.
            at_time = time_of(float(sigmas[step]), shift)
            parts = unpack(x, shapes)
            parts[0], realigned = _grow(
                parts[0], before, after, at_time, transform,
                seed + (index + 1) * 10000, height, width,
            )
            x = repack(parts, shapes)
            held.fit(after, x)
            # Only the sigma at the transition moves: the trajectory either side of it is
            # unchanged, and the realigned time is where the grown latent actually sits.
            sigmas[step] = sigma_of(realigned, shift)
    finally:
        held.restore()

    return x

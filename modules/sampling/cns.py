"""Colored noise sampling: a stochastic sampler's fresh noise spread over frequency bands.

A band's noise scales by ``(1 - gamma / divider) ** power`` times ``exp(alpha * f)``.
Bands are radial rings over a latent's last two axes.
"""

from __future__ import annotations

import inspect
import logging
import math
from collections import OrderedDict
from dataclasses import dataclass

import torch

#: How the allocation is set: the published presets, or by hand.
MODES = ("auto", "manual")

#: Frequency rings a latent is split into by default, and the range offered.
DEFAULT_BANDS = 32
MIN_BANDS = 4
MAX_BANDS = 128

#: Key the patch is filed under on a model.
WRAPPER_KEY = "was_cns"

#: Clean predictions kept for the end of run measurement, and the most bytes they may take.
MAX_KEPT_STEPS = 256
MAX_KEPT_BYTES = 512 * 1024 * 1024

#: Profiles kept at once, and the runs a profile's running mean counts at most.
MAX_PROFILES = 16
MAX_PROFILE_RUNS = 16

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class Allocation:
    """How the resolved share becomes a noise scale per ring.

    Attributes:
        divider: ``gamma`` is divided by this before ``1 - gamma``, so ``1.73`` keeps a band
            at least ``0.42`` of its noise and ``25`` keeps nearly all of it.
        power: Exponent on the residual, ``0.5`` for the square root.
        tilt_start: ``alpha`` of the ``exp(alpha * f)`` tilt at the first step.
        tilt_end: ``alpha`` at the last step.
        sharpness: Exponential easing of the tilt between the two, ``0`` for linear.
        energy: Standard deviation of the coloured noise against the white draw.
    """

    divider: float = 1.73
    power: float = 0.75
    tilt_start: float = 0.15
    tilt_end: float = -0.5
    sharpness: float = 0.75
    energy: float = 0.98


#: The settings behind the paper's best unguided result, FID 6.27 on SiT-XL/2.
UNGUIDED = Allocation()

#: The settings behind the paper's best guided result, FID 1.98 at CFG 1.45.
GUIDED = Allocation(divider=25.0, power=0.5, tilt_start=-0.1, tilt_end=0.03, sharpness=0.0,
                    energy=0.998)


@dataclass(frozen=True)
class Settings:
    """How a patched model colours its noise.

    Attributes:
        mode: An entry of :data:`MODES`.
        bands: Frequency rings.
        allocation: The allocation ``manual`` uses.
    """

    mode: str = MODES[0]
    bands: int = DEFAULT_BANDS
    allocation: Allocation = UNGUIDED

    def resolved(self, cfg: float) -> "Settings":
        """The settings a run uses: ``auto`` takes the published preset for its guidance."""
        if self.mode == "auto":
            return Settings(mode="auto", bands=DEFAULT_BANDS,
                            allocation=GUIDED if float(cfg) > 1.0 else UNGUIDED)
        return self


def band_index(height: int, width: int, bands: int, device) -> torch.Tensor:
    """The ring each coefficient of a real 2D FFT falls in.

    Args:
        height: Rows of the transformed plane.
        width: Columns of the transformed plane.
        bands: Rings, spread evenly from zero frequency to the corner.
        device: Device the index lives on.

    Returns:
        A long tensor shaped ``[height, width // 2 + 1]`` with values in ``[0, bands)``.
    """
    fy = torch.fft.fftfreq(height, device=device)
    fx = torch.fft.rfftfreq(width, device=device)
    radius = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    corner = torch.sqrt(fy.abs().max() ** 2 + fx.abs().max() ** 2).clamp(min=1e-12)
    return (radius / corner * (bands - 1)).long().clamp(0, bands - 1)


def band_mean(values: torch.Tensor, index: torch.Tensor, bands: int) -> torch.Tensor:
    """Mean of a per-coefficient plane over each ring.

    Args:
        values: A plane shaped like ``index``.
        index: From :func:`band_index`.
        bands: Rings.

    Returns:
        A float32 tensor of ``bands`` values.
    """
    flat = index.reshape(-1)
    total = torch.zeros(bands, device=values.device).scatter_add_(0, flat, values.reshape(-1))
    count = torch.zeros(bands, device=values.device).scatter_add_(
        0, flat, torch.ones_like(values).reshape(-1))
    return total / count.clamp(min=1.0)


def band_power(x: torch.Tensor, index: torch.Tensor, bands: int) -> torch.Tensor:
    """Mean power per coefficient in each ring, over every leading axis.

    Args:
        x: A tensor shaped ``[..., H, W]``.
        index: From :func:`band_index` for that plane.
        bands: Rings.

    Returns:
        A float32 tensor of ``bands`` values.
    """
    spectrum = torch.fft.rfft2(x.float(), norm="ortho")
    power = (spectrum.real ** 2 + spectrum.imag ** 2).reshape(-1, *spectrum.shape[-2:])
    return band_mean(power.mean(dim=0), index, bands)


def band_progress(final: torch.Tensor, predicted: torch.Tensor, index: torch.Tensor,
                  bands: int) -> torch.Tensor:
    """Resolved share of each ring, ``1 - |final - predicted|^2 / |final|^2`` per coefficient.

    Args:
        final: The finished sample, ``[..., H, W]``.
        predicted: A clean prediction of the same shape.
        index: From :func:`band_index`.
        bands: Rings.

    Returns:
        A float32 tensor of ``bands`` values in ``[0, 1]``: each coefficient's share clamped
        to ``[0, 1]``, averaged over every leading axis, then over the ring.
    """
    target = torch.fft.rfft2(final.float())
    error = (torch.fft.rfft2(predicted.float()) - target).abs() ** 2
    progress = (1.0 - error / (target.abs() ** 2 + 1e-8)).clamp(0.0, 1.0)
    return band_mean(progress.reshape(-1, *progress.shape[-2:]).mean(dim=0), index, bands)


def noise_ratio(model_sampling, sigma: float) -> float:
    """Noise over signal a latent at ``sigma`` carries, for any model's parameterisation.

    Args:
        model_sampling: The model's ``model_sampling``.
        sigma: A sampler sigma.

    Returns:
        The noise coefficient over the data coefficient, or ``sigma`` where the model scales
        no noise into its latent.
    """
    level = torch.tensor([float(sigma)], dtype=torch.float32)
    ones, zeros = torch.ones(1, 1), torch.zeros(1, 1)
    try:
        noise = float(model_sampling.noise_scaling(level, ones, zeros))
        data = float(model_sampling.noise_scaling(level, zeros, ones))
    except Exception:
        return float(sigma)
    if not math.isfinite(noise) or noise <= 0.0:
        return float(sigma)
    if not math.isfinite(data) or data <= 1e-6:
        return float("inf")
    return noise / data


def wiener_gamma(power: torch.Tensor, ratio: float) -> torch.Tensor:
    """Resolved share of each ring from its signal power and the noise left.

    Args:
        power: Signal power per ring, per coefficient.
        ratio: Noise over signal, from :func:`noise_ratio`.

    Returns:
        ``power / (power + ratio^2)`` per ring.
    """
    if not math.isfinite(ratio):
        return torch.zeros_like(power)
    return power / (power + ratio * ratio)


def min_max(gamma: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """``gamma`` stretched per ring so ``low`` maps to 0 and ``high`` to 1.

    Args:
        gamma: Resolved share per ring.
        low: Each ring's least resolved share over the run.
        high: Each ring's most resolved share over the run.

    Returns:
        The stretched share, clamped to ``[0, 1]``.
    """
    return ((gamma - low) / (high - low + 1e-8)).clamp(0.0, 1.0)


def tilt_at(allocation: Allocation, progress: float) -> float:
    """The tilt ``alpha`` at a fraction of the run.

    Args:
        allocation: The allocation.
        progress: ``0`` at the first step, ``1`` at the last.

    Returns:
        ``alpha``, eased between ``tilt_start`` and ``tilt_end``.
    """
    progress = min(1.0, max(0.0, float(progress)))
    sharp = float(allocation.sharpness)
    if abs(sharp) > 1e-6:
        progress = (math.exp(sharp * progress) - 1.0) / (math.exp(sharp) - 1.0)
    return allocation.tilt_start + progress * (allocation.tilt_end - allocation.tilt_start)


def noise_scale(gamma: torch.Tensor, allocation: Allocation, progress: float) -> torch.Tensor:
    """The noise scale per ring.

    Args:
        gamma: Resolved share per ring, in ``[0, 1]``.
        allocation: The allocation.
        progress: Fraction of the run, for the tilt.

    Returns:
        A scale per ring, before the draw is renormalised.
    """
    residual = 1.0 - gamma / max(float(allocation.divider), 1e-6)
    alpha = tilt_at(allocation, progress)
    if alpha != 0.0:
        f_norm = torch.linspace(0.0, 1.0, gamma.shape[-1], device=gamma.device)
        residual = residual * torch.exp(alpha * f_norm)
    return residual.clamp(min=0.0) ** float(allocation.power)


def coloured(noise: torch.Tensor, scale: torch.Tensor, index: torch.Tensor,
             energy: float) -> torch.Tensor:
    """Noise with each ring scaled, renormalised per sample to ``energy`` times the draw.

    Args:
        noise: White noise shaped ``[B, ..., H, W]``.
        scale: A scale per ring.
        index: From :func:`band_index`.
        energy: Standard deviation of the result against the draw's.

    Returns:
        Noise of the same shape and dtype.
    """
    height, width = noise.shape[-2:]
    spectrum = torch.fft.rfft2(noise.float(), norm="ortho") * scale.to(noise.device)[index]
    shaped = torch.fft.irfft2(spectrum, s=(height, width), norm="ortho")
    axes = tuple(range(1, noise.ndim))
    target = noise.float().std(dim=axes, keepdim=True) * float(energy)
    shaped = shaped / shaped.std(dim=axes, keepdim=True).clamp(min=1e-8) * target
    return shaped.to(noise.dtype)


def takes_noise_sampler(function) -> bool:
    """Whether a sampler function injects noise through a ``noise_sampler`` argument.

    Args:
        function: A sampler's ``sampler_function``.

    Returns:
        True where the signature names ``noise_sampler``.
    """
    try:
        return "noise_sampler" in inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False


def base_noise_sampler(function, x: torch.Tensor, sigmas: torch.Tensor, seed):
    """The noise sampler a sampler function builds for itself when handed none.

    Args:
        function: A sampler's ``sampler_function``.
        x: A tensor of the latent's shape, dtype and device.
        sigmas: The run's sigmas.
        seed: The run's seed.

    Returns:
        ``noise(sigma, sigma_next)``.
    """
    from comfy.k_diffusion import sampling

    name = getattr(function, "__name__", "")
    if "dpmpp" in name and "sde" in name:
        positive = sigmas[sigmas > 0]
        return sampling.BrownianTreeNoiseSampler(
            x, positive.min(), sigmas.max(), seed=seed, cpu=not name.endswith("_gpu"))
    return sampling.default_noise_sampler(x, seed=seed)


class Run:
    """One sampling run on a patched model: the predictions seen and the noise coloured.

    Attributes:
        settings: The resolved settings.
        sigmas: The run's sigmas.
        shapes: Shapes a packed latent unpacks to, or None for a plain latent.
        ratio: ``noise_ratio`` bound to the model.
        profile: ``(sigmas, gamma)`` measured on earlier runs of the same model and size, or
            None.
        current: The ``gamma`` for the noise injected after the step just seen.
        kept: Clean predictions kept for the end of run measurement, by step.
        estimates: The ``gamma`` used after each step, with that step.
        injections: Noise draws coloured.
    """

    def __init__(self, settings: Settings, sigmas: torch.Tensor, shapes, ratio, profile=None):
        self.settings = settings
        self.sigmas = sigmas.detach().float().cpu()
        self.shapes = shapes if shapes and len(shapes) > 1 else None
        self.ratio = ratio
        self.profile = profile
        self.current: torch.Tensor | None = None
        self.kept: dict[int, torch.Tensor] = {}
        self.kept_bytes = 0
        self.estimates: list[tuple[int, torch.Tensor]] = []
        self.injections = 0
        self.index_cache: dict = {}

    def index(self, plane: torch.Tensor) -> torch.Tensor:
        """The ring index for a plane's size, built once."""
        key = (plane.shape[-2], plane.shape[-1], plane.device)
        if key not in self.index_cache:
            self.index_cache[key] = band_index(
                plane.shape[-2], plane.shape[-1], self.settings.bands, plane.device)
        return self.index_cache[key]

    def primary(self, x: torch.Tensor) -> torch.Tensor:
        """The latent the colouring acts on: the first of a packed latent, or the latent."""
        if getattr(x, "is_nested", False):
            return x.unbind()[0]
        return x

    def seen(self, step: int, denoised: torch.Tensor) -> None:
        """Record one step's clean prediction and the ``gamma`` its noise is coloured by."""
        latest = self.primary(denoised)
        if latest.ndim < 3:
            return
        index = self.index(latest)
        sigma = float(self.sigmas[min(int(step), len(self.sigmas) - 1)])
        self.current = self.gamma(sigma, index, latest)
        self.estimates.append((int(step), self.current.detach().cpu()))
        size = latest.numel() * 2
        if len(self.kept) < MAX_KEPT_STEPS and self.kept_bytes + size <= MAX_KEPT_BYTES:
            self.kept[int(step)] = latest.detach().to("cpu", torch.float16)
            self.kept_bytes += size

    def step_of(self, sigma: float) -> int:
        """The step a sigma opens."""
        return int(torch.argmin((self.sigmas - float(sigma)).abs()))

    def progress_of(self, step: int) -> float:
        """A step as a fraction of the run's sigmas."""
        return int(step) / max(1, len(self.sigmas) - 1)

    def gamma(self, sigma: float, index: torch.Tensor, latest: torch.Tensor) -> torch.Tensor:
        """The resolved share per ring at ``sigma``, stretched per ring over the run."""
        if self.profile is not None:
            return profile_at(self.profile, sigma).to(index.device)
        power = band_power(latest, index, self.settings.bands)
        positive = self.sigmas[self.sigmas > 0]
        low = wiener_gamma(power, self.ratio(float(self.sigmas.max())))
        high = wiener_gamma(power, self.ratio(float(positive.min()) if len(positive) else 0.0))
        return min_max(wiener_gamma(power, self.ratio(sigma)), low, high)

    def colour(self, noise: torch.Tensor, sigma: float, sigma_next: float) -> torch.Tensor:
        """One noise draw, coloured by the ``gamma`` of the step that draws it."""
        if self.current is None:
            return noise
        parts = self.unpacked(noise)
        head = parts[0]
        if head.ndim < 3:
            return noise
        allocation = self.settings.allocation
        scale = noise_scale(self.current.to(head.device), allocation,
                            self.progress_of(self.step_of(sigma)))
        parts[0] = coloured(head, scale, self.index(head), allocation.energy)
        self.injections += 1
        return self.packed(parts, noise)

    def unpacked(self, x: torch.Tensor) -> list[torch.Tensor]:
        """A packed latent's parts, or the latent alone."""
        if self.shapes is None:
            return [x]
        import comfy.utils

        return list(comfy.utils.unpack_latents(x, self.shapes))

    def packed(self, parts: list[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
        """Parts packed back to the sampler's layout."""
        if self.shapes is None:
            return parts[0]
        import comfy.utils

        return comfy.utils.pack_latents(parts)[0].to(like.dtype)

    def measured(self) -> dict[int, torch.Tensor]:
        """Each kept prediction's resolved share against the run's last one, stretched per ring."""
        if len(self.kept) < 2:
            return {}
        steps = sorted(self.kept)
        final = self.kept[steps[-1]].float()
        index = band_index(final.shape[-2], final.shape[-1], self.settings.bands, "cpu")
        raw = torch.stack([band_progress(final, self.kept[step].float(), index,
                                         self.settings.bands) for step in steps])
        scaled = min_max(raw, raw.min(dim=0).values, raw.max(dim=0).values)
        return dict(zip(steps, scaled))

    def learned(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """``(sigmas, gamma)`` this run measured, one row per kept prediction."""
        measured = self.measured()
        if not measured:
            return None
        steps = sorted(measured)
        return self.sigmas[steps], torch.stack([measured[step] for step in steps])

    def agreement(self) -> float | None:
        """Mean distance between the ``gamma`` used and the share measured at the end."""
        measured = self.measured()
        gaps = [float((gamma - measured[step]).abs().mean())
                for step, gamma in self.estimates if step in measured]
        if not gaps:
            return None
        return sum(gaps) / len(gaps)


#: Measured profiles by model, latent size and ring count, most recently used last. Each
#: holds ``(sigmas, gamma, runs)``, ``gamma`` a running mean over ``runs`` runs.
PROFILES: "OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor, int]]" = OrderedDict()


def profile_key(model_patcher, latent: torch.Tensor, bands: int, cfg: float) -> tuple:
    """The key a model, latent size and guidance file their measured profile under.

    Args:
        model_patcher: The sampling model's patcher.
        latent: The latent the colouring acts on.
        bands: Rings.
        cfg: The guidance scale.

    Returns:
        A hashable key.
    """
    inner = getattr(model_patcher, "model", model_patcher)
    return (id(inner), type(inner).__name__, tuple(latent.shape[1:]), int(bands),
            round(float(cfg), 2))


def profile_at(profile, sigma: float) -> torch.Tensor:
    """A measured profile's row for a sigma, interpolated between its neighbours.

    Args:
        profile: ``(sigmas, gamma, ...)`` with sigmas falling.
        sigma: The sigma wanted.

    Returns:
        The resolved share per ring.
    """
    sigmas, gamma = profile[0], profile[1]
    sigma = float(sigma)
    if sigma >= float(sigmas[0]):
        return gamma[0]
    if sigma <= float(sigmas[-1]):
        return gamma[-1]
    after = int(torch.nonzero(sigmas <= sigma)[0])
    high, low = float(sigmas[after - 1]), float(sigmas[after])
    weight = (high - sigma) / max(high - low, 1e-12)
    return gamma[after - 1] * (1.0 - weight) + gamma[after] * weight


def remember(key: tuple, learned: tuple[torch.Tensor, torch.Tensor]) -> None:
    """Fold one run's measured profile into the running mean kept for its key.

    Args:
        key: From :func:`profile_key`.
        learned: ``(sigmas, gamma)`` from :meth:`Run.learned`.
    """
    sigmas, gamma = learned
    previous = PROFILES.get(key)
    runs = 1
    if previous is not None:
        runs = min(previous[2] + 1, MAX_PROFILE_RUNS)
        earlier = torch.stack([profile_at(previous, float(sigma)) for sigma in sigmas])
        gamma = earlier + (gamma - earlier) / runs
    PROFILES[key] = (sigmas, gamma, runs)
    PROFILES.move_to_end(key)
    while len(PROFILES) > MAX_PROFILES:
        PROFILES.popitem(last=False)


def generator_registries() -> list[dict]:
    """RES4LYF's noise generator registries, where RES4LYF is loaded.

    Returns:
        Each distinct ``NOISE_GENERATOR_CLASSES`` dictionary found, empty without RES4LYF.
    """
    import sys

    found, seen = [], set()
    for name, module in list(sys.modules.items()):
        if module is None or "res4lyf" not in name.lower():
            continue
        for attribute in ("NOISE_GENERATOR_CLASSES", "NOISE_GENERATOR_CLASSES_SIMPLE"):
            registry = getattr(module, attribute, None)
            if isinstance(registry, dict) and id(registry) not in seen:
                seen.add(id(registry))
                found.append(registry)
    return found


def coloured_generator(base, run: Run):
    """A noise generator class whose draws ``run`` colours.

    Args:
        base: A RES4LYF generator class, built as ``base(x=..., seed=..., ...)`` and called as
            ``generator(sigma=..., sigma_next=...)``.
        run: The run colouring the draws.

    Returns:
        A class built and called the same way, forwarding every attribute to ``base``.
    """
    class Coloured:
        def __init__(self, *args, **kwargs):
            object.__setattr__(self, "_inner", base(*args, **kwargs))

        def __call__(self, *args, **kwargs):
            noise = self._inner(*args, **kwargs)
            sigma = kwargs.get("sigma", args[0] if args else None)
            sigma_next = kwargs.get("sigma_next", args[1] if len(args) > 1 else None)
            if sigma is None or sigma_next is None or not torch.is_tensor(noise):
                return noise
            return run.colour(noise, float(sigma), float(sigma_next))

        def __getattr__(self, name):
            return getattr(object.__getattribute__(self, "_inner"), name)

        def __setattr__(self, name, value):
            setattr(object.__getattribute__(self, "_inner"), name, value)

    Coloured.__name__ = getattr(base, "__name__", "Coloured")
    return Coloured


def wrapper(settings: Settings):
    """A ``sampler_sample`` wrapper colouring the noise of whatever sampler runs.

    Args:
        settings: How to colour.

    Returns:
        The wrapper.
    """
    def sample(executor, guider, sigmas, extra_args, callback, noise, latent_image=None,
               denoise_mask=None, disable_pbar=False):
        sampler = executor.class_obj
        function = getattr(sampler, "sampler_function", None)
        model_sampling = guider.inner_model.model_sampling
        cfg = float(getattr(guider, "cfg", 1.0) or 1.0)
        resolved = settings.resolved(cfg)
        shapes = getattr(guider.inner_model, "latent_shapes", None)
        run = Run(resolved, sigmas, shapes, lambda sigma: noise_ratio(model_sampling, sigma))
        key = profile_key(guider.model_patcher, run.unpacked(noise)[0], resolved.bands, cfg)
        if key in PROFILES:
            PROFILES.move_to_end(key)
            run.profile = PROFILES[key]

        def watched(step, x0, x, total):
            run.seen(step, x0)
            if callback is not None:
                return callback(step, x0, x, total)
            return None

        options = getattr(sampler, "extra_options", None)
        stochastic = function is not None and takes_noise_sampler(function)
        if stochastic and isinstance(options, dict):
            base = options.get("noise_sampler") or base_noise_sampler(
                function, noise, sigmas, extra_args.get("seed"))

            def noise_sampler(sigma, sigma_next):
                return run.colour(base(sigma, sigma_next), sigma, sigma_next)

            sampler.extra_options = {**options, "noise_sampler": noise_sampler}
        # RES4LYF builds its own generators by name; each is wrapped for this run only.
        swapped: list[tuple[dict, dict]] = []
        if not stochastic and "res4lyf" in getattr(function, "__module__", "").lower():
            for registry in generator_registries():
                original = dict(registry)
                registry.update({name: coloured_generator(value, run)
                                 for name, value in original.items() if callable(value)})
                swapped.append((registry, original))
        try:
            return executor(guider, sigmas, extra_args, watched, noise, latent_image,
                            denoise_mask, disable_pbar)
        finally:
            if stochastic and isinstance(options, dict):
                sampler.extra_options = options
            for registry, original in swapped:
                registry.clear()
                registry.update(original)
            report(run, resolved, key, function, stochastic or bool(swapped), cfg)

    return sample


def report(run: Run, settings: Settings, key: tuple, function, reachable: bool,
           cfg: float) -> None:
    """Keep the run's measured profile and log what the run did.

    Args:
        run: The finished run.
        settings: Its resolved settings.
        key: Its profile key.
        function: The sampler function.
        reachable: Whether the sampler's noise could be coloured.
        cfg: The guidance scale.
    """
    name = getattr(function, "__name__", "the sampler").replace("sample_", "")
    agreement = run.agreement()
    source = ("a profile measured on earlier runs" if run.profile is not None
              else "a live estimate")
    measured = ("" if agreement is None
                else f"; {source}, within {agreement:.3f} of the share measured at the end")
    learned = run.learned()
    if learned is not None:
        remember(key, learned)
    preset = ""
    if settings.mode == "auto":
        preset = f", {'guided' if cfg > 1.0 else 'unguided'} preset"
    if run.injections:
        LOG.info("CNS: %s, %s mode%s, coloured %d noise draw(s)%s", name, settings.mode, preset,
                 run.injections, measured)
    elif reachable:
        LOG.info("CNS: %s drew no noise this run, so nothing was coloured%s", name, measured)
    else:
        LOG.info("CNS: %s takes no noise sampler, so nothing was coloured%s", name, measured)


def patched(model, settings: Settings):
    """A clone of a model whose sampling runs colour their noise.

    Args:
        model: A ``ModelPatcher``.
        settings: How to colour.

    Returns:
        The patched clone.
    """
    import comfy.patcher_extension

    clone = model.clone()
    clone.add_wrapper_with_key(
        comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE, WRAPPER_KEY, wrapper(settings))
    return clone

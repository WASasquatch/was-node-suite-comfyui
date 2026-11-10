"""A sampler for MiniMax H3 Image: core's Euler and lcm steps, optionally starting small.

Progressive sampling runs the first ``grow_at`` steps on a smaller grid and grows the noisy latent
in the cosine basis.
"""

from __future__ import annotations

import math

import torch

from . import spectral

__all__ = ["sample", "start_size"]


def start_size(height: int, width: int, scale: float) -> tuple[int, int]:
    """The even latent size a progressive run starts at.

    Args:
        height: Full latent height.
        width: Full latent width.
        scale: Share of each side to start at.

    Returns:
        ``(height, width)``, each even, at least 2 and no larger than the full size.
    """
    def side(full):
        return max(2, min(full, int(round(full * scale / 2.0)) * 2))
    return side(height), side(width)


def _renoised(sigmas, renoise_steps: int) -> set[int]:
    """Indices of the steps that take fresh noise."""
    above = [i for i in range(len(sigmas) - 1) if float(sigmas[i + 1]) > 0]
    return set(above[max(0, len(above) - int(renoise_steps)):]) if renoise_steps > 0 else set()


def _step(x, denoised, sigma, sigma_next, fresh, noise_sampler, noise_scale, model_sampling):
    """One step: core's lcm step when ``fresh``, else core's Euler step."""
    import comfy.k_diffusion.sampling as k_diffusion

    if fresh:
        noise = noise_sampler(sigma, sigma_next)
        if noise_scale != 1.0:
            noise = noise * noise_scale
        return model_sampling.noise_scaling(sigma_next, noise, denoised)
    return x + k_diffusion.to_d(x, sigma, denoised) * (sigma_next - sigma)


@torch.no_grad()
def sample(model, x, sigmas, extra_args=None, callback=None, disable=None, renoise_steps: int = 1,
           noise_scale: float = 1.0, start_scale: float = 1.0, grow_at: int = 0):
    """Sample a latent on ``sigmas``, the last ``renoise_steps`` steps above zero as lcm steps.

    Args:
        model: Core's denoiser for the sampling run.
        x: The noisy latent.
        sigmas: Sigmas ending in 0.
        renoise_steps: Steps ending above zero, counted from the last, that take fresh noise.
            ``0`` is core's euler; every step is core's lcm.
        noise_scale: Multiplier on the fresh noise of a renoised step; ``1`` is core's lcm step.
        start_scale: Share of each side the first ``grow_at`` steps run at; ``1`` is off.
        grow_at: Steps run on the smaller grid; ``0`` is off.

    Returns:
        The sampled latent.

    Raises:
        ValueError: A progressive run is given an inpainting mask.
    """
    import comfy.k_diffusion.sampling as k_diffusion
    from tqdm.auto import trange

    extra_args = {} if extra_args is None else extra_args
    model_sampling = model.inner_model.model_patcher.get_model_object("model_sampling")
    steps = len(sigmas) - 1
    renoised = _renoised(sigmas, renoise_steps)
    if start_scale < 1.0 and 0 < grow_at < steps:
        return _progressive(model, x, sigmas, extra_args, callback, disable, renoised, noise_scale,
                            start_scale, int(grow_at), model_sampling)

    noise_sampler = k_diffusion.default_noise_sampler(x, seed=extra_args.get("seed", None))
    s_in = x.new_ones([x.shape[0]])
    for i in trange(steps, disable=disable):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})
        x = _step(x, denoised, sigma, sigma_next, i in renoised, noise_sampler, noise_scale, model_sampling)
    return x


def _progressive(model, x, sigmas, extra_args, callback, disable, renoised, noise_scale,
                 start_scale, grow_at, model_sampling):
    """The first ``grow_at`` steps on a smaller grid, the rest at full size."""
    import comfy.k_diffusion.sampling as k_diffusion
    from tqdm.auto import tqdm

    if extra_args.get("denoise_mask") is not None:
        raise ValueError(
            "H3 Image Sampler's progressive mode samples part of the run at a smaller size and "
            "cannot follow an inpainting mask. Set mode to standard for a masked latent.")
    seed = extra_args.get("seed", None)
    full = (x.shape[-2], x.shape[-1])
    small = start_size(*full, start_scale)
    ratio = small[0] / full[0], small[1] / full[1]
    noise = getattr(model, "noise", None)
    latent = getattr(model, "latent_image", None)
    if noise is not None and latent is not None and noise.shape == x.shape and latent.shape == x.shape:
        # Truncation keeps noise at unit strength and raises a picture's, so the picture is scaled back.
        picture = spectral.downscale(latent, 1.0, small) * math.sqrt(ratio[0] * ratio[1])
        x = model_sampling.noise_scaling(sigmas[0], spectral.downscale(noise, 1.0, small), picture)
    else:
        x = spectral.downscale(x, 1.0, small)

    noise_sampler = k_diffusion.default_noise_sampler(x, seed=seed)
    s_in = x.new_ones([x.shape[0]])
    steps = len(sigmas) - 1
    bar = tqdm(total=steps, disable=disable)
    i = 0
    while i < steps:
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})
        bar.update(1)
        if i == grow_at - 1:
            grown_seed = None if seed is None else seed + 1
            noise_sampler = k_diffusion.default_noise_sampler(x.new_empty((*x.shape[:-2], *full)), seed=grown_seed)
            x, i = _grow(x, denoised, sigmas, i, full, grown_seed, noise_sampler)
            s_in = x.new_ones([x.shape[0]])
            continue
        x = _step(x, denoised, sigma, sigma_next, i in renoised, noise_sampler, noise_scale, model_sampling)
        i += 1
    bar.close()
    return x


def _grow(x, denoised, sigmas, i, full, seed, noise_sampler):
    """Cross from the small grid to ``full`` at step ``i``, topping up with ``noise_sampler``.

    Returns:
        The full-size latent and the index of the step to run next.
    """
    import comfy.k_diffusion.sampling as k_diffusion

    sigma, sigma_next = sigmas[i], sigmas[i + 1]
    area = math.sqrt(full[0] * full[1] / (x.shape[-2] * x.shape[-1]))
    # The stepped latent grows with noise in its new frequencies and lands on a higher time; noise
    # is then added to reach the nearest sigma on the schedule at or above it.
    stepped = x + k_diffusion.to_d(x, sigma, denoised) * (sigma_next - sigma)
    t = float(sigma_next)
    grown = spectral.expand(stepped, full, t, "dct", 0 if seed is None else seed) * spectral.kappa(t, area)
    landed = t * spectral.kappa(t, area)
    j = i + 1
    while j > 0 and float(sigmas[j]) < landed:
        j -= 1
    target = float(sigmas[j])
    keep = (1.0 - target) / (1.0 - landed)
    extra = max(target * target - (keep * landed) ** 2, 0.0) ** 0.5
    return grown * keep + noise_sampler(sigmas[j], sigmas[j]) * extra, j

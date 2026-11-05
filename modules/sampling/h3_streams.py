"""A sampler stepping MiniMax H3's picture and sound by separate rules from one model call.

Sound steps on its own sigmas from the model's shifts; each step is one of core's ``lcm``,
``euler`` or ``euler_ancestral``.
"""

from __future__ import annotations

import torch

__all__ = ["RULES", "sample"]

#: Step rules a stream can take, each one of core's samplers.
RULES = ("lcm", "euler", "euler_ancestral")

#: Rules that draw fresh noise.
_NOISY = ("lcm", "euler_ancestral")


class _Given:
    """A denoiser returning a prediction already made, so core's samplers can take a single step."""

    def __init__(self, inner_model, denoised: torch.Tensor):
        self.inner_model = inner_model
        self.denoised = denoised

    def __call__(self, x, sigma, **kwargs):
        return self.denoised


def _step(rule: str, inner_model, x, denoised, sigma, sigma_next, noise, eta: float) -> torch.Tensor:
    """One step of core's sampler ``rule`` from ``sigma`` to ``sigma_next``, given its prediction and noise."""
    import comfy.k_diffusion.sampling as k_diffusion

    given = _Given(inner_model, denoised)
    pair = torch.stack([sigma, sigma_next]).to(x.device, torch.float32)
    if rule == "lcm":
        return k_diffusion.sample_lcm(given, x, pair, disable=True, noise_sampler=lambda s, n: noise)
    if rule == "euler":
        return k_diffusion.sample_euler(given, x, pair, disable=True)
    return k_diffusion.sample_euler_ancestral(given, x, pair, disable=True, eta=eta,
                                              noise_sampler=lambda s, n: noise)


@torch.no_grad()
def sample(model, x, sigmas, extra_args=None, callback=None, disable=None,
           video: str = "lcm", audio: str = "euler_ancestral", eta: float = 1.0):
    """Sample a packed MiniMax H3 latent, the picture by ``video`` and the sound by ``audio``.

    Args:
        model: Core's denoiser for the sampling run.
        x: The packed noisy latent.
        sigmas: Video sigmas, ending in 0.
        video: The picture's step rule, one of :data:`RULES`.
        audio: The sound's step rule, one of :data:`RULES`.
        eta: Share of fresh noise in an ``euler_ancestral`` step.

    Returns:
        The packed sampled latent.

    Raises:
        ValueError: The latent is not MiniMax H3's picture and sound, or a rule is unknown.
    """
    import comfy.k_diffusion.sampling as k_diffusion
    import comfy.ldm.minimax.model as minimax
    import comfy.utils
    from tqdm.auto import trange

    for rule in (video, audio):
        if rule not in RULES:
            raise ValueError(f"H3 Stream Sampler has no step rule {rule!r}. Choose one of: {', '.join(RULES)}.")
    extra_args = {} if extra_args is None else extra_args
    guider = model.inner_model
    shapes = getattr(guider.inner_model, "latent_shapes", None)
    sampling = guider.model_patcher.get_model_object("model_sampling")
    audio_shift = getattr(sampling, "audio_shift", None)
    if shapes is None or len(shapes) != 2 or audio_shift is None:
        raise ValueError(
            "H3 Stream Sampler samples MiniMax H3's picture and sound together, and this model or "
            "latent has no separate sound stream. Use another sampler for it.")
    shift, audio_shift = float(sampling.shift), float(audio_shift)
    scale = shift / audio_shift
    noise_sampler = k_diffusion.default_noise_sampler(x, seed=extra_args.get("seed", None))
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=disable):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = model(x, sigma * s_in, **extra_args)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})
        picture, sound = comfy.utils.unpack_latents(x, shapes)
        picture_x0, sound_x0 = comfy.utils.unpack_latents(denoised, shapes)
        fresh = (None, None)
        if sigma_next > 0 and (video in _NOISY or audio in _NOISY):
            fresh = comfy.utils.unpack_latents(noise_sampler(sigma, sigma_next), shapes)
        picture = _step(video, guider, picture, picture_x0, sigma, sigma_next, fresh[0], eta)
        # The sampler carries sound as (sigma / sound sigma) * sound, its clean value scaled by shift / audio_shift.
        level = minimax.time_shift_sigma(sigma, shift, audio_shift)
        level_next = minimax.time_shift_sigma(sigma_next, shift, audio_shift)
        own = _step(audio, guider, sound * (level / sigma), sound_x0 / scale, level, level_next, fresh[1], eta)
        sound = own * (sigma_next / level_next) if sigma_next > 0 else own * scale
        x = comfy.utils.pack_latents([picture, sound])[0]
    return x

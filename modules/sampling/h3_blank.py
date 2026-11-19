"""Blank-picture guidance for MiniMax H3 Image.

A guided step runs the model on the prompt alone and on the prompt with a blank picture, and
answers ``plain + strength * (blank - plain)``.
"""

from __future__ import annotations

import torch

import comfy.samplers

__all__ = ["BLANK", "BlankGuider", "GRID", "MODES", "SUBTRACTION", "encode", "empty_latent"]

#: MiniMax H3 Image's six-step sigma grid.
GRID = (1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0)

#: Guidance modes: the blank on every step, or the difference on the first steps.
BLANK = "blank"
SUBTRACTION = "subtraction"
MODES = (BLANK, SUBTRACTION)

#: Side of the blank picture, in pixels, as the text encoder and the VAE read a reference.
BLANK_SIDE = 1024

#: How the text encoder labels a picture in the prompt.
PICTURE_LABEL = "Picture %d: "

#: Latent channels and pixels per latent cell of MiniMax H3 Image.
LATENT_CHANNELS = 24
LATENT_SCALE = 16


def encode(clip, vae, prompt: str, colour: tuple) -> tuple:
    """The prompt encoded alone and with a flat picture.

    Args:
        clip: The MiniMax H3 Image text encoder.
        vae: The MiniMax H3 Image VAE, which turns the picture into reference rows.
        prompt: The prompt.
        colour: ``(r, g, b)`` of the picture in 0 to 1.

    Returns:
        ``(plain, blank)`` CONDITIONING.
    """
    picture = torch.tensor(colour, dtype=torch.float32).view(1, 1, 1, 3).expand(1, BLANK_SIDE, BLANK_SIDE, 3).contiguous()
    plain = clip.encode_from_tokens_scheduled(clip.tokenize(prompt))
    blank = clip.encode_from_tokens_scheduled(clip.tokenize(prompt, images=[picture], picture_label=PICTURE_LABEL))
    z = vae.encode(picture)
    block = {"kind": "image", "latent_h": z.shape[-2], "latent_w": z.shape[-1], "latent": z}
    blank = [[embedding, {**extras, "minimax_refs": list(extras.get("minimax_refs") or []) + [block]}]
             for embedding, extras in blank]
    return plain, blank


def empty_latent(width: int, height: int, device) -> dict:
    """An empty MiniMax H3 Image latent for a picture of ``width`` x ``height``."""
    shape = [1, LATENT_CHANNELS, 1, max(1, int(height) // LATENT_SCALE), max(1, int(width) // LATENT_SCALE)]
    return {"samples": torch.zeros(shape, device=device)}


class BlankGuider(comfy.samplers.CFGGuider):
    """Mixes the prompt alone with the prompt and a blank picture on the guided steps."""

    def __init__(self, model_patcher, mode: str, strength: float, steps: int):
        super().__init__(model_patcher)
        self.strength = float(strength)
        self.until = 0.0 if mode == BLANK else GRID[max(0, min(len(GRID) - 1, int(steps)))]

    def set_conds(self, positive, blank):
        """Set the prompt alone and the prompt with the blank picture."""
        self.inner_set_conds({"positive": positive, "blank": blank})

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        """Both runs mixed on a guided step; one run where the mix needs only one side."""
        sigma = float(timestep.flatten()[0])
        guided = sigma > self.until + 1e-3
        if not guided or self.strength == 0.0:
            return comfy.samplers.calc_cond_batch(self.inner_model, [self.conds["positive"]], x, timestep, model_options)[0]
        if self.strength == 1.0:
            return comfy.samplers.calc_cond_batch(self.inner_model, [self.conds["blank"]], x, timestep, model_options)[0]
        plain, blank = comfy.samplers.calc_cond_batch(
            self.inner_model, [self.conds["positive"], self.conds["blank"]], x, timestep, model_options
        )
        return plain + self.strength * (blank - plain)

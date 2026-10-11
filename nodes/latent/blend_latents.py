"""Blend two latent tensors with one of twelve mix operations."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ...modules.compat.sockets import require_input

OPERATIONS = [
    "add",
    "multiply",
    "divide",
    "subtract",
    "overlay",
    "hard_light",
    "soft_light",
    "screen",
    "linear_dodge",
    "difference",
    "exclusion",
    "random",
]


def normalize(latent: torch.Tensor) -> torch.Tensor:
    """Rescale a tensor so its minimum is 0.0 and its maximum is 1.0.

    Args:
        latent: Tensor to rescale.

    Returns:
        The tensor spanning 0.0-1.0. A tensor whose values are all the same spans nothing
        and comes back as zeros, that being where its single value sits on the scale.
    """
    low = latent.min()
    span = latent.max() - low
    if span == 0:
        return latent - low
    return (latent - low) / span


def _overlay(latent1, latent2, blend_factor):
    low = 2 * latent1 * latent2
    high = 1 - 2 * (1 - latent1) * (1 - latent2)
    return (latent1 * blend_factor) * low + (latent2 * blend_factor) * high


def _screen(latent1, latent2, blend_factor):
    inverted_latent1 = 1 - latent1
    inverted_latent2 = 1 - latent2
    return 1 - (inverted_latent1 * inverted_latent2 * (1 - blend_factor))


def _difference(latent1, latent2, blend_factor):
    return abs(latent1 - latent2) * blend_factor


def _exclusion(latent1, latent2, blend_factor):
    return (latent1 + latent2 - 2 * latent1 * latent2) * blend_factor


def _hard_light(latent1, latent2, blend_factor):
    return torch.where(
        latent2 < 0.5, 2 * latent1 * latent2, 1 - 2 * (1 - latent1) * (1 - latent2)
    ) * blend_factor


def _linear_dodge(latent1, latent2, blend_factor):
    return torch.clamp(latent1 + latent2, 0, 1) * blend_factor


def _soft_light(latent1, latent2, blend_factor):
    low = 2 * latent1 * latent2 + latent1 ** 2 - 2 * latent1 * latent2 * latent1
    high = 2 * latent1 * (1 - latent2) + torch.sqrt(latent1) * (2 * latent2 - 1)
    return (latent1 * blend_factor) * low + (latent2 * blend_factor) * high


def _random_noise(latent1, latent2, blend_factor):
    noise1 = torch.randn_like(latent1)
    noise2 = torch.randn_like(latent2)
    noise1 = (noise1 - noise1.min()) / (noise1.max() - noise1.min())
    noise2 = (noise2 - noise2.min()) / (noise2.max() - noise2.min())
    blended_noise = (latent1 * blend_factor) * noise1 + (latent2 * blend_factor) * noise2
    return torch.clamp(blended_noise, 0, 1)


#: Operations that read a single blend factor. ``add``, ``multiply``, ``divide`` and
#: ``subtract`` are not here: they weight both operands, one with ``blend`` and the other
#: with ``1 - blend``, so they are written out in :func:`blend_latents`.
BLEND_FUNCTIONS = {
    "overlay": _overlay,
    "screen": _screen,
    "difference": _difference,
    "exclusion": _exclusion,
    "hard_light": _hard_light,
    "linear_dodge": _linear_dodge,
    "soft_light": _soft_light,
    "random": _random_noise,
}


def blend_latents(latent1, latent2, mode="add", blend_percentage=0.5):
    """Mix two latent tensors and renormalise the result.

    Args:
        latent1: First operand.
        latent2: Second operand.
        mode: One of :data:`OPERATIONS`.
        blend_percentage: Weight given to ``latent1``. The four arithmetic modes give
            ``latent2`` the complement, ``1 - blend_percentage``; every other mode ignores
            it and weights both operands by ``blend_percentage``.

    Returns:
        The blended tensor, rescaled to 0.0-1.0. A blend that comes out uniform, two equal
        latents, or a weight that cancels one of them, has no range to rescale and comes
        back as zeros.

    Raises:
        ValueError: ``mode`` is not one of :data:`OPERATIONS`.
    """
    blend_factor1 = blend_percentage
    blend_factor2 = 1 - blend_percentage

    if mode == "add":
        blended_latent = (latent1 * blend_factor1) + (latent2 * blend_factor2)
    elif mode == "multiply":
        blended_latent = (latent1 * blend_factor1) * (latent2 * blend_factor2)
    elif mode == "divide":
        blended_latent = (latent1 * blend_factor1) / (latent2 * blend_factor2)
    elif mode == "subtract":
        blended_latent = (latent1 * blend_factor1) - (latent2 * blend_factor2)
    elif mode in BLEND_FUNCTIONS:
        blended_latent = BLEND_FUNCTIONS[mode](latent1, latent2, blend_factor1)
    else:
        raise ValueError(
            "Unsupported blending mode {!r}. Choose one of: {}.".format(
                mode, ", ".join(OPERATIONS)
            )
        )

    return normalize(blended_latent)


class BlendLatents(io.ComfyNode):
    """Blend two LATENT tensors with a selectable mix operation."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Blend Latents",
            display_name="Blend Latents",
            search_aliases=["Blend Latents", "latent blend", "mix latents", "latent math"],
            category="WAS Suite/Latent",
            description=(
                "Blend two latents with an arithmetic or photographic mix, such as `overlay` "
                "or `screen`. The result is rescaled so its values run from 0.0 to 1.0."
            ),
            inputs=[
                io.Latent.Input(
                    "latent_a",
                    tooltip=(
                        "First latent. On the arithmetic operations this is the one blend "
                        "weights; on subtract and divide it is the left-hand side."
                    ),
                ),
                io.Latent.Input(
                    "latent_b",
                    tooltip=(
                        "Second latent. On the arithmetic operations it takes the remaining "
                        "weight, 1 - blend; on subtract and divide it is what latent_a is "
                        "reduced by."
                    ),
                ),
                io.Combo.Input(
                    "operation",
                    options=OPERATIONS,
                    tooltip=(
                        "`add`, `multiply`, `divide`, `subtract` = weighted arithmetic; "
                        "`overlay`, `hard_light`, `soft_light` = darks stay dark and lights "
                        "light, each gentler than the last; `screen`, `linear_dodge` = brighten "
                        "only; `difference`, `exclusion` = keep what the two disagree on; "
                        "`random` = mix through fresh noise each run."
                    ),
                ),
                io.Float.Input(
                    "blend",
                    default=0.5,
                    min=0.01,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Weight of latent_a. On `add`, `multiply`, `divide` and `subtract`, "
                        "latent_b gets the rest: 0.5 = even mix; 1.0 = latent_a only. Every "
                        "other operation scales both latents by it, as an overall strength."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output(
                    tooltip=(
                        "The blended latent, rescaled so its lowest value is 0.0 and its "
                        "highest 1.0."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, latent_a, latent_b, operation, blend) -> io.NodeOutput:
        """Blend the two latents.

        Raises:
            ValueError: Nothing is connected to one of the two latent inputs.
        """
        for value, socket in ((latent_a, "latent_a"), (latent_b, "latent_b")):
            require_input(
                value,
                "Blend Latents",
                socket,
                "latent",
                "latent source such as Empty Latent Image or VAE Encode",
                "LATENT",
            )

        blended = blend_latents(latent_a["samples"], latent_b["samples"], operation, blend)
        return io.NodeOutput({"samples": blended})

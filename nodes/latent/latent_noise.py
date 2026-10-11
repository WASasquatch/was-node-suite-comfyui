"""Add gaussian noise to a latent."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ...modules.compat.sockets import require_input


class LatentNoiseInjection(io.ComfyNode):
    """Add zero-mean gaussian noise of a given standard deviation to a LATENT."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Latent Noise Injection",
            display_name="Latent Noise Injection",
            search_aliases=["Latent Noise Injection", "latent noise", "add noise", "jitter latent"],
            category="WAS Suite/Latent/Generate",
            description=(
                "A copy of the latent with random noise mixed in, so that resampling it "
                "brings out new variation or extra detail."
            ),
            inputs=[
                io.Latent.Input(
                    "samples",
                    tooltip="The latent the noise is added to. It is left untouched itself.",
                ),
                io.Float.Input(
                    "noise_std",
                    default=0.1,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Standard deviation of the added noise: 0.0 = none, latent passed "
                        "through; 0.1 = a light dusting a low-denoise resample turns into "
                        "detail; 1.0 = as strong as the latent itself, leaving little of the "
                        "original."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output(
                    tooltip="The latent with the noise added, ready to resample.",
                ),
            ],
        )

    @classmethod
    def execute(cls, samples, noise_std) -> io.NodeOutput:
        """Add the noise.

        Raises:
            ValueError: Nothing is connected to the samples input.
        """
        require_input(
            samples,
            "Latent Noise Injection",
            "samples",
            "latent",
            "latent source such as Empty Latent Image or VAE Encode",
            "LATENT",
        )
        noised = samples.copy()
        noised["samples"] = noised["samples"] + torch.randn_like(noised["samples"]) * noise_std
        return io.NodeOutput(noised)

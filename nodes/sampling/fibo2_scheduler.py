"""The sampling schedule Fibo 2.0 was released with."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ...modules.compat.limits import max_resolution
from ...modules.model.fibo2 import prompts

NODE_NAME = "Fibo 2 Scheduler"


class Fibo2Scheduler(io.ComfyNode):
    """Fibo 2.0's released sigmas for a step count and an image size."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASFibo2Scheduler",
            display_name=NODE_NAME,
            search_aliases=["WASFibo2Scheduler", NODE_NAME, "fibo", "fibo 2", "bria", "sigmas", "scheduler"],
            category="WAS Suite/Sampling",
            description=(
                "The sampling schedule Fibo 2.0 was released with, shifted for the image size. Hands "
                "sigmas to SamplerCustomAdvanced; Dev runs 50 steps at CFG 5, Edit Turbo 4 at CFG 1."
            ),
            inputs=[
                io.Int.Input(
                    "steps", default=50, min=1, max=4096,
                    tooltip="Denoising steps: 50 for Dev, 30 for a faster draft, 4 for Edit Turbo.",
                ),
                io.Int.Input(
                    "width", default=1024, min=16, max=max_resolution(), step=16,
                    tooltip="Image width in pixels, as 1024 or 1216. Ignored when latent is connected.",
                ),
                io.Int.Input(
                    "height", default=1024, min=16, max=max_resolution(), step=16,
                    tooltip="Image height in pixels, as 1024 or 832. Ignored when latent is connected.",
                ),
                io.Latent.Input(
                    "latent", optional=True,
                    tooltip="The latent being sampled, as Text Encode Fibo 2 hands on; its size replaces width and height.",
                ),
            ],
            outputs=[io.Sigmas.Output(tooltip="steps + 1 sigmas from 1 down to 0.")],
        )

    @classmethod
    def execute(cls, steps, width, height, latent=None) -> io.NodeOutput:
        if latent is not None:
            samples = latent["samples"]
            tokens = int(samples.shape[-1]) * int(samples.shape[-2])
        else:
            tokens = (int(width) // 16) * (int(height) // 16)
        return io.NodeOutput(torch.tensor(prompts.sigmas(int(steps), tokens), dtype=torch.float32))

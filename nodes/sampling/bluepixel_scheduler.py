"""The sampling schedule BluePixel was released with."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ...modules.interface import run_result
from ...modules.model.bluepixel import schedule

NODE_NAME = "BluePixel Scheduler"


class BluePixelScheduler(io.ComfyNode):
    """BluePixel's released sigmas for a step count and the latent being edited."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASBluePixelScheduler",
            display_name=NODE_NAME,
            search_aliases=["WASBluePixelScheduler", NODE_NAME, "bluepixel", "blue pixel", "sigmas", "scheduler"],
            category="WAS Suite/Sampling",
            description=(
                "The sampling schedule BluePixel was released with, shifted for the latent's size. Hands "
                "sigmas to SamplerCustomAdvanced with euler and a CFG Guider at 3."
            ),
            inputs=[
                io.Int.Input(
                    "steps", default=4, min=1, max=4096,
                    tooltip="Denoising steps: 4, the count BluePixel-Turbo was distilled for.",
                ),
                io.Latent.Input(
                    "latent",
                    tooltip="The latent being edited, as Text Encode BluePixel hands on; its size sets the shift.",
                ),
            ],
            outputs=[io.Sigmas.Output(tooltip="steps + 1 sigmas from 1 down to 0.")],
        )

    @classmethod
    def execute(cls, steps, latent) -> io.NodeOutput:
        tokens = schedule.tokens_of(latent)
        sigmas = schedule.sigmas(int(steps), tokens)
        run_result.publish(
            status=run_result.OK,
            summary=f"{int(steps)} steps for {tokens:,} image tokens",
            counts={"tokens": tokens, "shift (mu)": round(schedule.mu(tokens, int(steps)), 4)},
            facts={"sigmas": ", ".join(f"{value:.4f}" for value in sigmas)},
        )
        return io.NodeOutput(torch.tensor(sigmas, dtype=torch.float32))

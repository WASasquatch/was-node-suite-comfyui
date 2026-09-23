"""Taking one clip's prompt and its own empty latent out of a MiniMax H3 run."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import H3_PROMPTS
from ...modules.latent import h3_conditioning

INDEX_HINT = (
    "Which clip to take, from `0`. Wire a While Loop Open's index in to step through "
    "them one per iteration."
)


class ThreeH3ClipSelect(io.ComfyNode):
    """Hand one clip its prompt and the empty latent it samples into."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASMiniMaxH3ClipSelect",
            display_name="MiniMax H3 Clip Select",
            search_aliases=[
                "WASMiniMaxH3ClipSelect",
                "MiniMax H3 Clip Select",
                "minimax h3 clip",
                "select clip",
                "separate clips",
            ],
            category="WAS Suite/Latent/Video",
            description=(
                "Take one clip out of a MiniMax H3 run: its prompt, and the empty latent "
                "it samples into. Every clip has a latent of its own, so a loop wrapped "
                "round this samples each one from fresh and nothing passes between them. "
                "Send the latent to a sampler and the sampled result to a decode inside "
                "the loop, so each clip arrives as its own frames."
            ),
            inputs=[
                H3_PROMPTS.Input(
                    "prompts",
                    tooltip="Every clip's prompt and length, from MiniMax H3 Conditioning.",
                ),
                io.Int.Input(
                    "index",
                    default=0,
                    min=0,
                    max=999,
                    tooltip=INDEX_HINT,
                ),
            ],
            outputs=[
                io.Conditioning.Output(
                    display_name="positive",
                    tooltip="That clip's prompt, for the guider that samples it.",
                ),
                io.Latent.Output(
                    display_name="latent",
                    tooltip="That clip's own empty latent, for the sampler's latent input.",
                ),
                io.Int.Output(
                    display_name="frames",
                    tooltip="Frames that clip runs for, as `124`.",
                ),
                io.String.Output(
                    display_name="report",
                    tooltip="Which clip was taken and how long it is.",
                ),
            ],
        )

    @classmethod
    def execute(cls, prompts, index) -> io.NodeOutput:
        """Read one clip's entry.

        Raises:
            ValueError: Nothing arrived, the index is outside it, or that clip carries
                no latent of its own.
        """
        positive, frames, _, _ = h3_conditioning.pick(prompts, index)
        latent = h3_conditioning.latent_at(prompts, index)
        report = f"clip {int(index) + 1} of {len(prompts)}, {frames} frames"
        return io.NodeOutput(positive, latent, frames, report)

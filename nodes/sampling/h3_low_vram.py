"""A MiniMax H3 model set up to sample long clips in less VRAM."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.model import h3_low_vram


class H3LowVRAM(io.ComfyNode):
    """Run a MiniMax H3 model's blocks in token slices with its resident weights budgeted."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASH3LowVRAM",
            display_name="H3 Low VRAM",
            search_aliases=[
                "WASH3LowVRAM",
                "H3 Low VRAM",
                "low vram",
                "streamed blocks",
                "minimax h3",
                "out of memory",
                "head chunks",
                "query chunks",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Set a MiniMax H3 model up to sample long clips in less VRAM, with the same "
                "result. Each block runs over the clip in token slices, and only the weights that "
                "fit beside the clip stay on the card while the rest stream in as each block runs. "
                "head_chunks and query_chunks split attention, the largest memory a block holds, "
                "into smaller calls, and the room they free keeps more weights on the card. The "
                "log names what was applied when sampling starts. Place it after the model "
                "loaders and any LoRA so every H3 sampler in the graph uses it."
            ),
            inputs=[
                io.Model.Input("model", tooltip="The MiniMax H3 model, after any LoRA."),
                io.Int.Input(
                    "head_chunks",
                    default=0, min=0, max=56,
                    tooltip=(
                        "0 = off; 4 = attention runs over 14 of the 56 heads at a time; 8 = 7 at "
                        "a time. Every head is computed as before, so the result is unchanged. "
                        "Frees roughly a third of the memory attention holds on a long clip."
                    ),
                ),
                io.Int.Input(
                    "query_chunks",
                    default=0, min=0, max=64,
                    tooltip=(
                        "0 = off; 4 = attention answers the clip's tokens in 4 chunks against keys "
                        "and values worked out once. Each token is computed as before, so the "
                        "result is unchanged. Frees about half the memory attention holds, at the "
                        "cost of one extra attention projection per block. Combines with head_chunks."
                    ),
                ),
                io.Int.Input(
                    "token_slice",
                    default=8192, min=1024, max=65536, step=1024,
                    tooltip=(
                        "8192 = norms, projections and feed-forward run 8192 tokens at a time; "
                        "4096 holds about half the working memory there. The result matches up "
                        "to float rounding. Lower it when memory is still short after "
                        "head_chunks and query_chunks."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="model",
                    tooltip="The model for every H3 sampler in the graph.",
                ),
            ],
        )

    @classmethod
    def execute(cls, model, head_chunks=0, query_chunks=0,
                token_slice=8192) -> io.NodeOutput:
        """Wrap the model, or a clone of one that already carries the setup, with the settings."""
        options = h3_low_vram.Options(head_chunks, query_chunks, token_slice)
        return io.NodeOutput(h3_low_vram.configure(model, options))

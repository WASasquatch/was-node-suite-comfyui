"""A MiniMax H3 model whose every call runs over overlapping tiles."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.model import h3_tiles


def _tiling_options() -> list:
    """The ``auto`` and ``manual`` tilings with their settings."""
    return [
        io.DynamicCombo.Option(key="auto", inputs=[
            io.Int.Input(
                "max_tokens", default=h3_tiles.AUTO_TOKENS, min=4000, max=1048000, step=1000,
                tooltip="Most tokens one tile holds, as `48000`, about 200 frames at 0.8 MP, "
                        "or `24000` on a 12 to 16 GB card.",
            ),
        ]),
        io.DynamicCombo.Option(key="manual", inputs=[
            io.Int.Input(
                "tile_width", default=1024, min=64, max=16384, step=32,
                tooltip="Pixels a tile spans across the frame, as `1024`.",
            ),
            io.Int.Input(
                "tile_height", default=576, min=64, max=16384, step=32,
                tooltip="Pixels a tile spans down the frame, as `576`.",
            ),
            io.Int.Input(
                "window_frames", default=0, min=0, max=100000, step=1,
                tooltip="Frames a time window spans, as `141`, or `0` for the whole clip.",
            ),
            io.Int.Input(
                "overlap", default=128, min=32, max=4096, step=32,
                tooltip="Pixels neighbouring tiles share, as `128`.",
            ),
            io.Int.Input(
                "window_overlap", default=17, min=1, max=1000, step=1,
                tooltip="Frames neighbouring time windows share, as `17`.",
            ),
        ]),
    ]


class H3Tiles(io.ComfyNode):
    """Run each MiniMax H3 model call over overlapping tiles in time and space, blended per step."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASH3Tiles",
            display_name="H3 Tiles",
            search_aliases=[
                "WASH3Tiles",
                "H3 Tiles",
                "tiled diffusion",
                "tiles",
                "windows",
                "minimax h3",
                "upscale",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Run every MiniMax H3 model call in overlapping tiles across the frame and "
                "overlapping windows along the clip, blended where they meet at each step so the "
                "tiles stay one clip. Any sampler then refines a long or upscaled H3 latent in "
                "tiles that each fit the card, run as H3 Low VRAM runs them. Upscale the latent "
                "first, or decode, upscale and encode, and sample it at a partial denoise."
            ),
            inputs=[
                io.Model.Input("model", tooltip="The MiniMax H3 model, after any LoRA."),
                io.DynamicCombo.Input(
                    "tiling", options=_tiling_options(), display_name="tiling",
                    tooltip="`auto` splits the clip into windows along time under a token budget, "
                            "and across the frame only where a window will not fit; "
                            "`manual` sets tile and window sizes.",
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="model",
                    tooltip="The model for the sampler that refines the latent.",
                ),
            ],
        )

    @classmethod
    def execute(cls, model, tiling) -> io.NodeOutput:
        """Wrap the model with the chosen tiling."""
        return io.NodeOutput(h3_tiles.tiled_model(model, h3_tiles.tiling_settings(tiling)))

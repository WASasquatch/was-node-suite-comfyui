"""A model patch that denoises any sampler's latent over overlapping tiles."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.sampling import tile_refine, tiled_diffusion
from .tiled_captioned_sampler import TILE_SIZE_INPUTS, TILING_HINT


class TiledDiffusionModelPatch(io.ComfyNode):
    """Make a model predict over overlapping tiles inside any sampler."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASTiledDiffusionModelPatch",
            display_name="Tiled Diffusion Model Patch",
            search_aliases=[
                "WASTiledDiffusionModelPatch",
                "Tiled Diffusion Model Patch",
                "tiled diffusion",
                "multidiffusion",
                "mixture of diffusers",
                "tile",
                "detail",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Make a model denoise the latent as overlapping tiles inside any sampler, each "
                "tile a picture the model's native size, blended where they meet. Background and "
                "small detail get the attention a whole frame would. Start it partway in to keep "
                "the whole frame's composition; a shift set before it lays out only those "
                "whole-frame steps."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="The diffusion model to patch; MODEL, such as from Load Diffusion Model.",
                ),
                io.DynamicCombo.Input(
                    "tiling",
                    options=[
                        io.DynamicCombo.Option(key="auto", inputs=[]),
                        io.DynamicCombo.Option(key="manual", inputs=TILE_SIZE_INPUTS),
                    ],
                    tooltip=TILING_HINT,
                ),
                io.Float.Input(
                    "start_percent",
                    default=0.3,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Share of the model's own noise schedule laid out whole before the "
                    "tiles take over. 0 = tiled from the first step; 0.3 = the first 6 of 20 "
                    "steps on simple; 0.5 = from sigma 0.87 on Flux 3 Klein.",
                ),
                io.Float.Input(
                    "end_percent",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Share of the model's own noise schedule where the tiles stop. 1 = "
                    "tiled to the last step; 0.9 = the last 2 of 20 steps on simple run whole.",
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="model",
                    tooltip="The patched model, for KSampler or any sampler; MODEL. A latent no "
                    "bigger than one tile is not tiled.",
                ),
            ],
        )

    @classmethod
    def execute(cls, model, tiling, start_percent=0.3, end_percent=1.0) -> io.NodeOutput:
        """Patch a clone of ``model`` to predict over the tiles ``tiling`` plans.

        Raises:
            ValueError: ``end_percent`` comes before ``start_percent``.
        """
        if end_percent < start_percent:
            raise ValueError(
                f"Tiled Diffusion Model Patch ends at {end_percent} before it starts at "
                f"{start_percent}. Set end_percent to {start_percent} or more."
            )
        tile, overlap = tile_refine.tile_settings(tiling, model)
        return io.NodeOutput(tiled_diffusion.patch(model, tile, overlap, start_percent, end_percent))

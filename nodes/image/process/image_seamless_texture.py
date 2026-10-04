"""Make an image tile against itself, optionally as a grid of those tiles."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.image.seamless import LARGEST_BLEND, make_seamless
from ....modules.interface import size_report


class ImageSeamlessTexture(io.ComfyNode):
    """Blend an image's opposite edges together so copies of it tile without a visible join."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Seamless Texture",
            display_name="Image Seamless Texture",
            search_aliases=["Image Seamless Texture", "tileable", "seamless tile", "texture"],
            category="WAS Suite/Image/Process",
            description=(
                "Turn an image into a seamlessly tiling texture by blending its opposite "
                "edges into each other, and optionally show it repeated as a grid. The "
                "tile is smaller than the source by the blended fraction on each side."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The images to make tileable. Each one is processed separately.",
                ),
                io.Float.Input(
                    "blending",
                    default=0.4,
                    max=LARGEST_BLEND,
                    min=0.0,
                    step=0.01,
                    tooltip=(
                        "How much of each side the cross-fade between edges spans, as a "
                        "fraction. 0.0 answers the image unchanged; 0.4 fades the outer 40 "
                        "percent, which suits most textures; 0.5 is the most the edges leave "
                        "room for. The answer is smaller by this fraction on each side."
                    ),
                ),
                io.Boolean.Input(
                    "tiled",
                    default=True,
                    tooltip=(
                        "`on` = the tile repeated into a grid, which is how the join is checked; "
                        "`off` = the single tile, which is what to feed onward."
                    ),
                ),
                io.Int.Input(
                    "tiles",
                    default=2,
                    max=6,
                    min=2,
                    step=2,
                    tooltip=(
                        "How many copies along each side of the grid when tiled is on. 2 "
                        "gives a 2x2 preview of four copies; 4 gives sixteen. Ignored when tiled "
                        "is off."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip=(
                        "The tileable images, or the grids of them, one per image in. A grid is "
                        "tiles times larger on each side than the tile."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, images, blending, tiled, tiles) -> io.NodeOutput:
        tiled_images = make_seamless(
            images,
            blending,
            tiled,
            tiles,
            node="Image Seamless Texture",
            advice="Turning tiled off, lowering tiles or passing fewer frames also fits it.",
        )
        size_report.publish(
            images,
            tiled_images,
            action="made seamless",
            facts={"tiling": f"{tiles} by {tiles}" if tiled else "off"},
        )
        return io.NodeOutput(tiled_images)

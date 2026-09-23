"""Cutting an image into a grid of tiles and reordering them."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.image import dynamic
from ....modules.convert.tensors import image_planes, stack_images, tensor2pil
from ....modules.interface import size_report

REQUIRES = "extras"


class ImageTileShuffle(io.ComfyNode):
    """Cut each image into a grid of tiles and lay them back down in a shuffled order."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASNSImageTileShuffle",
            display_name="Image Tile Shuffle",
            search_aliases=["WASNSImageTileShuffle", "tile", "shuffle", "scramble", "mosaic"],
            category="WAS Suite/Image/Transform",
            description=(
                "Cut each picture into a grid of equal tiles and lay them back down in a "
                "shuffled order, optionally with a coloured gap between them. Useful for "
                "puzzle and collage looks, and for building a scrambled reference that a "
                "model cannot read as a coherent scene."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The pictures to cut up. Every frame of a batch is cut the same way "
                        "and shuffled with the same seed, so a sequence stays consistent."
                    ),
                ),
                io.Int.Input(
                    "max_tiles", default=4, min=2, max=64, step=2,
                    tooltip=(
                        "How many tiles the picture is cut into. The grid is the squarest "
                        "arrangement of that many: 4 gives 2 by 2, 12 gives 3 by 4. Rows and "
                        "columns divide the picture exactly, so any leftover pixels on the "
                        "right and bottom edges are dropped."
                    ),
                ),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF,
                    tooltip=(
                        "Seed for the shuffle. The same seed always produces the same tile "
                        "order; change it to get a different arrangement of the same tiles. "
                        "Any whole number; `0` is as good a seed as any."
                    ),
                ),
                io.Int.Input(
                    "border_width", default=0, min=0, max=100, step=1,
                    tooltip=(
                        "Gap in pixels drawn between neighbouring tiles, in border_color. "
                        "0 butts the tiles together with no gap, which keeps the output the "
                        "same size as the input."
                    ),
                ),
                io.String.Input(
                    "border_color", default="#FFFFFF",
                    tooltip=(
                        "Colour of the gap between tiles, as a hex string such as #FFFFFF "
                        "for white or #000000 for black. The leading # is optional, and an "
                        "unreadable value falls back to white."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip=(
                        "The reassembled pictures. Larger than the input when border_width "
                        "is above 0, since the gaps are added between the tiles."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, images, max_tiles, seed, border_width, border_color) -> io.NodeOutput:
        folded = dynamic.fold(images)
        images = folded.images
        import numpy as np
        from PIL import Image

        from ....modules.image.tiles import compute_grid, parse_hex_color

        border_rgb = parse_hex_color(border_color)
        rows, columns = compute_grid(max_tiles)

        results = []
        for plane in image_planes(images):
            source = tensor2pil(plane)
            width, height = source.size
            tile_w = width // columns
            tile_h = height // rows

            tiles = [
                source.crop((column * tile_w, row * tile_h,
                             column * tile_w + tile_w, row * tile_h + tile_h))
                for row in range(rows)
                for column in range(columns)
            ]

            order = np.random.default_rng(seed).permutation(len(tiles)).tolist()
            canvas = Image.new(
                "RGB",
                (columns * tile_w + (columns - 1) * border_width,
                 rows * tile_h + (rows - 1) * border_width),
                border_rgb,
            )
            for position, index in enumerate(order):
                canvas.paste(
                    tiles[index],
                    ((position % columns) * (tile_w + border_width),
                     (position // columns) * (tile_h + border_width)),
                )
            results.append(canvas)

        shuffled = stack_images(results)
        size_report.publish(
            images,
            shuffled,
            action="shuffled",
            facts={"grid": f"{columns} by {rows}"},
        )
        return io.NodeOutput(dynamic.unfold(shuffled, folded))

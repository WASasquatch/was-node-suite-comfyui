"""Splitting each image into its four quadrants."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.image import dynamic
from ....modules.convert.tensors import image_planes, stack_images, tensor2pil
from ....modules.interface import size_report

REQUIRES = "extras"


class ImageTileExtract(io.ComfyNode):
    """Split each image into four quadrants, one per output."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASNSImageTileExtract",
            display_name="Image Tile Extract (Quadrants)",
            search_aliases=[
                "WASNSImageTileExtract", "Image Tile Extract",
                "tile", "quadrant", "quarters", "split", "crop", "four tiles",
            ],
            category="WAS Suite/Image/Transform",
            description=(
                "Split each picture into its four quadrants and send each one to its own "
                "output. Handy for upscaling or re-rendering a large frame in four pieces, "
                "and for feeding four separate crops into a comparison grid."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The pictures to split. Every frame of a batch is split the same "
                        "way, so each output carries one quadrant of every frame."
                    ),
                ),
                io.Int.Input(
                    "border_width", default=0, min=0, max=100, step=1,
                    tooltip=(
                        "Border in pixels drawn around each quadrant, in border_color. The "
                        "quadrant is shrunk to fit inside it, so the output tile stays the "
                        "same size. 0 leaves the quadrant at its own resolution with no "
                        "resampling at all."
                    ),
                ),
                io.String.Input(
                    "border_color", default="#FFFFFF",
                    tooltip=(
                        "Colour of the border, as a hex string such as #FFFFFF for white or "
                        "#000000 for black. The leading # is optional, and an unreadable "
                        "value falls back to white. Ignored when border_width is 0."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="top_left",
                    tooltip="The upper-left quarter of each picture.",
                ),
                io.Image.Output(
                    display_name="top_right",
                    tooltip="The upper-right quarter of each picture.",
                ),
                io.Image.Output(
                    display_name="bottom_left",
                    tooltip="The lower-left quarter of each picture.",
                ),
                io.Image.Output(
                    display_name="bottom_right",
                    tooltip="The lower-right quarter of each picture.",
                ),
            ],
        )

    @classmethod
    def execute(cls, images, border_width, border_color) -> io.NodeOutput:
        folded = dynamic.fold(images)
        images = folded.images
        from PIL import Image

        from ....modules.image.tiles import parse_hex_color

        border_rgb = parse_hex_color(border_color)
        corners: list[list[Image.Image]] = [[], [], [], []]

        for plane in image_planes(images):
            source = tensor2pil(plane)
            width, height = source.size
            tile_w = width // 2
            tile_h = height // 2

            quadrants = [
                source.crop((0, 0, tile_w, tile_h)),
                source.crop((tile_w, 0, tile_w * 2, tile_h)),
                source.crop((0, tile_h, tile_w, tile_h * 2)),
                source.crop((tile_w, tile_h, tile_w * 2, tile_h * 2)),
            ]

            for index, quadrant in enumerate(quadrants):
                if border_width > 0:
                    inner = quadrant.resize(
                        (max(tile_w - border_width * 2, 1), max(tile_h - border_width * 2, 1)),
                        Image.LANCZOS,
                    )
                    framed = Image.new("RGB", (tile_w, tile_h), border_rgb)
                    framed.paste(inner, (border_width, border_width))
                    corners[index].append(framed)
                else:
                    corners[index].append(quadrant)

        frame = size_report.frame_size(images)
        if frame is not None:
            size_report.publish(
                frame,
                (frame[0] // 2, frame[1] // 2),
                action="quartered",
                facts={"tiles": "4 of 2 by 2"},
            )
        return io.NodeOutput(
            *(dynamic.unfold(stack_images(corner), folded) for corner in corners)
        )

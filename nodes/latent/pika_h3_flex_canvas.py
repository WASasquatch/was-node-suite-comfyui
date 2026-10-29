"""The canvas sizes and clip lengths Pika H3 Flex was trained at."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.interface import run_result
from ...modules.model import pika_h3

NODE_NAME = "Pika H3 Flex Canvas"


class PikaH3FlexCanvas(io.ComfyNode):
    """A Pika H3 Flex canvas: width, height and frame count."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASPikaH3FlexCanvas",
            display_name=NODE_NAME,
            search_aliases=[
                "WASPikaH3FlexCanvas", NODE_NAME,
                "pika",
                "pika h3 flex",
                "canvas",
                "resolution",
                "aspect ratio",
                "frame count",
            ],
            category="WAS Suite/Latent/Video",
            description=(
                "Pick a Pika H3 Flex canvas by resolution, aspect and length, for the width, height "
                "and length of MiniMax H3 Image To Video or Reference To Video."
            ),
            inputs=[
                io.Combo.Input(
                    "resolution",
                    options=list(pika_h3.CANVAS),
                    default="480p",
                    tooltip="`480p` = 832x480 at 16:9; `768p` = 1344x768 at 16:9.",
                ),
                io.Combo.Input(
                    "aspect",
                    options=list(pika_h3.ASPECTS),
                    default="16:9",
                    tooltip="Width to height: at 480p, `16:9` = 832x480, `9:16` = 480x832, `1:1` = 480x480.",
                ),
                io.Int.Input(
                    "seconds",
                    default=pika_h3.MIN_SECONDS,
                    min=pika_h3.MIN_SECONDS,
                    max=pika_h3.MAX_SECONDS,
                    tooltip="Clip length: `5` = 124 frames, `10` = 243, `15` = 362, at 24 fps.",
                ),
            ],
            outputs=[
                io.Int.Output(display_name="width", tooltip="Canvas width in pixels, as `832`."),
                io.Int.Output(display_name="height", tooltip="Canvas height in pixels, as `480`."),
                io.Int.Output(
                    display_name="length",
                    tooltip="Frames at 24 fps on the 17k+5 grid, as `124` for 5 seconds.",
                ),
            ],
        )

    @classmethod
    def execute(cls, resolution="480p", aspect="16:9", seconds=pika_h3.MIN_SECONDS) -> io.NodeOutput:
        """Look up the canvas.

        Raises:
            ValueError: The resolution or aspect is not one the release has.
        """
        sizes = pika_h3.CANVAS.get(resolution)
        if sizes is None or aspect not in sizes:
            raise ValueError(
                f"{NODE_NAME} has no {resolution} {aspect} canvas. Choose a resolution from "
                f"{', '.join(pika_h3.CANVAS)} and an aspect from {', '.join(pika_h3.ASPECTS)}.")
        width, height = sizes[aspect]
        seconds = min(pika_h3.MAX_SECONDS, max(pika_h3.MIN_SECONDS, int(seconds)))
        length = pika_h3.frames_for_seconds(seconds)
        run_result.publish(
            summary=f"{width}x{height}, {length} frames",
            counts={"width": width, "height": height, "frames": length},
        )
        return io.NodeOutput(width, height, length)

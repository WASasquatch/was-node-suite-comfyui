"""Read a layer stack out as data: the canvas, the counts and a row per layer."""

from __future__ import annotations

from comfy_api.latest import io, ui

from ....modules.compat.types import DICT, LIST
from ....modules.image import layer_ops
from ....modules.log import get_logger

logger = get_logger("nodes.image.layers")


class LayersInfo(io.ComfyNode):
    """Report what a ``LAYERS`` document holds, as data and as a printed table."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLayersInfo",
            display_name="Layers Info",
            search_aliases=[
                "WASLayersInfo",
                "Layers Info",
                "layer count",
                "inspect layers",
                "layer names",
                "canvas size",
                "compositor",
            ],
            category="WAS Suite/Image/Layers",
            description=(
                "Read a layer stack out as data: the canvas, the layer count and how many are "
                "visible, then a row per layer with its index, name, placement, size, angle, "
                "opacity, blend mode and mask. The table shows on the node and comes out as a "
                "dictionary and a list for a switch or a caption."
            ),
            inputs=[
                io.Layers.Input(
                    "layers",
                    tooltip=(
                        "The stack to read. Wire in Add Layer, Layer Edit or anything else "
                        "answering a LAYERS document. Nothing is changed."
                    ),
                ),
            ],
            outputs=[
                DICT.Output(
                    display_name="document",
                    tooltip=(
                        "The whole document as one dictionary: canvas_width, canvas_height, "
                        "layers, visible, hidden and names. Feeds Text Dictionary Get or "
                        "Dictionary to Console."
                    ),
                ),
                LIST.Output(
                    display_name="layer_info",
                    tooltip=(
                        "One dictionary per layer, lowest in the stack first, carrying index, "
                        "name, x, y, width, height, frames, rotation in degrees, opacity, "
                        "blend_mode, visible, flip_h, flip_v, has_mask and z_index."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many layers the document holds, for a switch that handles none.",
                ),
                io.Int.Output(
                    display_name="canvas_width",
                    tooltip=(
                        "How wide the canvas is in pixels, for an Image Blank the stack is "
                        "composited over."
                    ),
                ),
                io.Int.Output(
                    display_name="canvas_height",
                    tooltip="How tall the canvas is in pixels, read the same way.",
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, layers) -> io.NodeOutput:
        summary, rows = layer_ops.described(layers)
        report = "\n".join(cls.lines(summary, rows))
        logger.info(
            "Layers Info read %d layer(s) off a %dx%d canvas",
            summary["layers"],
            summary["canvas_width"],
            summary["canvas_height"],
        )
        return io.NodeOutput(
            summary,
            rows,
            summary["layers"],
            summary["canvas_width"],
            summary["canvas_height"],
            ui=ui.PreviewText(report),
        )

    @staticmethod
    def lines(summary: dict, rows: list) -> list[str]:
        """One line naming the canvas, then one line per layer.

        Args:
            summary: The canvas size and the counts.
            rows: One dictionary per layer.

        Returns:
            The lines of the table, in stack order.
        """
        head = (
            f"canvas {summary['canvas_width']}x{summary['canvas_height']}, "
            f"{summary['layers']} layer(s), {summary['visible']} visible"
        )
        written = [head]
        for row in rows:
            marks = []
            if row["frames"] > 1:
                marks.append(f"{row['frames']} frames")
            if row["rotation"]:
                marks.append(f"turned {row['rotation']:g} deg")
            if row["flip_h"] or row["flip_v"]:
                marks.append("mirrored")
            if row["has_mask"]:
                marks.append("masked")
            if not row["visible"]:
                marks.append("hidden")
            written.append(
                f"{row['index']}: '{row['name']}' at {row['x']},{row['y']} "
                f"{row['width']}x{row['height']} {row['blend_mode']} "
                f"{round(row['opacity'] * 100)}%" + ("".join(f", {mark}" for mark in marks))
            )
        return written

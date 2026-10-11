"""Pin the canvas a layer stack is drawn on, so an effect cannot resize the picture."""

from __future__ import annotations

from comfy_api.latest import io, ui

from ....modules.image import layer_ops
from ....modules.log import get_logger

logger = get_logger("nodes.image.layers")

NODE_NAME = "Layers Canvas"

#: Where a stack is held when the canvas is smaller or larger than the layers reach.
ANCHORS = (
    "top left", "top centre", "top right",
    "middle left", "centre", "middle right",
    "bottom left", "bottom centre", "bottom right",
)


class LayersCanvas(io.ComfyNode):
    """Set the canvas of a ``LAYERS`` document, moving its layers to suit."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLayersCanvas",
            display_name=NODE_NAME,
            search_aliases=[
                "WASLayersCanvas",
                "Layers Canvas",
                "canvas size",
                "document size",
                "pin canvas",
                "resize canvas",
                "crop layers",
            ],
            category="WAS Suite/Image/Layers",
            description=(
                "Give a layer stack a fixed canvas size instead of the size its layers happen "
                "to reach. Strokes, glows and shadows then run off the edge as they would in "
                "an image editor, and the picture stays at the size it is meant to be."
            ),
            inputs=[
                io.Layers.Input(
                    "layers",
                    tooltip="The stack to set the canvas on; LAYERS.",
                ),
                io.Int.Input(
                    "width",
                    default=0,
                    min=0,
                    max=16384,
                    tooltip=(
                        "Canvas width in pixels; INT. 0 keeps whatever width the stack "
                        "already names, or the width its layers reach where it names none."
                    ),
                ),
                io.Int.Input(
                    "height",
                    default=0,
                    min=0,
                    max=16384,
                    tooltip=(
                        "Canvas height in pixels; INT. 0 keeps whatever height the stack "
                        "already names, or the height its layers reach where it names none."
                    ),
                ),
                io.Combo.Input(
                    "anchor",
                    options=list(ANCHORS),
                    default="top left",
                    tooltip=(
                        "Where the layers sit when the canvas is not the size they reach. "
                        "`top left` leaves every placement as it is; `centre` moves them all "
                        "so the middle of what they cover is the middle of the canvas."
                    ),
                ),
                io.Image.Input(
                    "match",
                    optional=True,
                    tooltip=(
                        "A picture to take the canvas from instead of typing it; IMAGE. Wire "
                        "the plate being composited onto and the canvas becomes its size."
                    ),
                ),
            ],
            outputs=[
                io.Layers.Output(
                    display_name="layers",
                    tooltip="The same stack, carrying the canvas it is drawn on; LAYERS.",
                ),
                io.Int.Output(
                    display_name="width",
                    tooltip="The canvas width the stack now names; INT.",
                ),
                io.Int.Output(
                    display_name="height",
                    tooltip="The canvas height the stack now names; INT.",
                ),
            ],
        )

    @classmethod
    def execute(cls, layers, width=0, height=0, anchor="top left", match=None) -> io.NodeOutput:
        """Set the document's canvas and move its layers to the anchor.

        Args:
            layers: A ``LAYERS`` document.
            width: Canvas width, or 0 to keep the one already in force.
            height: Canvas height, or 0 to keep the one already in force.
            anchor: Where the layers sit within the canvas.
            match: A picture to take the canvas from.

        Returns:
            The document with a canvas, and that canvas as two numbers.
        """
        entries = layer_ops.entries(layers)
        held_width, held_height = layer_ops.size_of(layers)

        if match is not None and getattr(match, "ndim", 0) == 4:
            width, height = int(match.shape[2]), int(match.shape[1])

        wanted_width = int(width) if width else held_width
        wanted_height = int(height) if height else held_height

        moved = [dict(entry) for entry in entries]
        if anchor != "top left" and moved:
            dx, dy = cls.offset(moved, wanted_width, wanted_height, anchor)
            for entry in moved:
                entry["x"] = int(entry.get("x", 0)) + dx
                entry["y"] = int(entry.get("y", 0)) + dy

        document = layer_ops.rebuilt(layers, moved)
        document["canvas"] = (wanted_width, wanted_height)

        logger.debug(
            "%s set the canvas to %dx%d over %d layer(s)",
            NODE_NAME, wanted_width, wanted_height, len(moved),
        )
        layer_ops.report(
            NODE_NAME,
            f"canvas pinned to {wanted_width}x{wanted_height} over {len(moved)} layer(s)",
            document,
            counts={"width": wanted_width, "height": wanted_height},
            facts={"anchored": anchor},
        )
        return io.NodeOutput(
            document, wanted_width, wanted_height,
            ui=ui.PreviewText(
                f"canvas {wanted_width}x{wanted_height}\n"
                f"{len(moved)} layer(s), anchored {anchor}"
            ),
        )

    @classmethod
    def box(cls, entry):
        """One layer's placement and drawn size.

        Args:
            entry: A layer as a dictionary.

        Returns:
            ``(x, y, width, height)`` in pixels, before any rotation.
        """
        picture = entry.get("image")
        shape = getattr(picture, "shape", None)
        natural_w = int(shape[-2]) if shape is not None and len(shape) >= 3 else 1
        natural_h = int(shape[-3]) if shape is not None and len(shape) >= 3 else 1
        width = int(entry.get("w") or 0) or natural_w
        height = int(entry.get("h") or 0) or natural_h
        return int(entry.get("x", 0)), int(entry.get("y", 0)), max(1, width), max(1, height)

    @classmethod
    def offset(cls, entries, width, height, anchor):
        """How far to move every layer so the stack sits at the anchor.

        Args:
            entries: The layers, as dictionaries.
            width: Canvas width.
            height: Canvas height.
            anchor: One of :data:`ANCHORS`.

        Returns:
            ``(dx, dy)`` in pixels.
        """
        boxes = [cls.box(entry) for entry in entries]
        left = min(box[0] for box in boxes)
        top = min(box[1] for box in boxes)
        right = max(box[0] + box[2] for box in boxes)
        bottom = max(box[1] + box[3] for box in boxes)
        spare_x, spare_y = width - (right - left), height - (bottom - top)

        across, down = anchor.split(" ")[-1], anchor.split(" ")[0]
        if anchor == "centre":
            across, down = "centre", "middle"
        dx = {"left": 0, "centre": spare_x // 2, "right": spare_x}.get(across, 0) - left
        dy = {"top": 0, "middle": spare_y // 2, "bottom": spare_y}.get(down, 0) - top
        return dx, dy


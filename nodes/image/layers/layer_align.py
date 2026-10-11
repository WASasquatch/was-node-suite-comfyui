"""Move one layer to an anchor on the canvas or against another layer."""

from __future__ import annotations

from comfy_api.latest import io, ui

from ....modules.compat import limits
from ....modules.image import layer_ops, sizing
from ....modules.log import get_logger

logger = get_logger("nodes.image.layers")

#: What a layer is aligned against, in menu order.
TARGETS = ("the canvas", "another layer")

#: Which layers the node moves, in menu order.
SCOPES = ("one layer", "every layer")


class LayerAlign(io.ComfyNode):
    """Place layers of a ``LAYERS`` document at an anchor rather than by coordinate."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLayerAlign",
            display_name="Layer Align",
            search_aliases=[
                "WASLayerAlign",
                "Layer Align",
                "align layer",
                "centre layer",
                "center layer",
                "compositor",
            ],
            category="WAS Suite/Image/Layers",
            description=(
                "Place a layer by naming a position instead of typing coordinates: centred "
                "on the canvas, flush to an edge, or lined up with another layer. A rotated "
                "layer is aligned by the box it draws in, so a turned title still sits flush."
            ),
            inputs=[
                io.Layers.Input(
                    "layers",
                    tooltip=(
                        "The stack holding the layer to move. Wire in Add Layer, Layer Edit "
                        "or anything else answering a LAYERS document."
                    ),
                ),
                io.Combo.Input(
                    "align",
                    options=list(sizing.ALIGNMENT_NAMES),
                    default=sizing.DEFAULT_ALIGNMENT,
                    tooltip=(
                        "Where the layer sits inside what it is aligned against. `middle "
                        "center` centres it, `bottom right` puts its bottom right corner on "
                        "the target's."
                    ),
                ),
                io.Combo.Input(
                    "scope",
                    options=list(SCOPES),
                    tooltip=(
                        "How many layers move. `one layer` moves the one index or layer_name "
                        "picks; `every layer` moves them all to the same anchor, which stacks "
                        "a set of crops into one spot."
                    ),
                ),
                io.Int.Input(
                    "index",
                    default=-1,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Which layer moves, counting 0 from the back of the stack. -1 is the "
                        "front layer. Ignored while layer_name names one, or on `every layer`."
                    ),
                ),
                io.String.Input(
                    "layer_name",
                    default="",
                    tooltip=(
                        "Name of the layer to move instead of index. Blank uses index. 'sky' "
                        "finds a layer called Sky, and finds Sky Backdrop where nothing is "
                        "called exactly Sky."
                    ),
                ),
                io.Combo.Input(
                    "against",
                    options=list(TARGETS),
                    tooltip=(
                        "What the layer is aligned inside. `the canvas` is the document's own "
                        "size, or the box its layers reach where it names none; `another "
                        "layer` is the box the layer target_index or target_name draws in."
                    ),
                ),
                io.Int.Input(
                    "target_index",
                    default=0,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Which layer is aligned against on `another layer`, counting 0 from "
                        "the back. 0 is the back layer, which is usually the plate."
                    ),
                ),
                io.String.Input(
                    "target_name",
                    default="",
                    tooltip=(
                        "Name of the layer to align against instead of target_index. Blank "
                        "uses target_index. Read only on `another layer`."
                    ),
                ),
                io.Int.Input(
                    "offset_x",
                    default=0,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Pixels added right of where the anchor put it. 0 = flush, 24 = 24px "
                        "in from a left edge, -24 = 24px in from a right one."
                    ),
                ),
                io.Int.Input(
                    "offset_y",
                    default=0,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Pixels added below where the anchor put it. 0 = flush, 24 = 24px "
                        "down from a top edge, -24 = 24px up from a bottom one."
                    ),
                ),
            ],
            outputs=[
                io.Layers.Output(
                    display_name="layers",
                    tooltip=(
                        "The stack with the layers moved, for Create Layered Image or another "
                        "edit."
                    ),
                ),
                io.Int.Output(
                    display_name="x",
                    tooltip="Where the last layer moved landed, as its left edge in pixels.",
                ),
                io.Int.Output(
                    display_name="y",
                    tooltip="Where the last layer moved landed, as its top edge in pixels.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls, layers, align=sizing.DEFAULT_ALIGNMENT, scope=SCOPES[0], index=-1,
        layer_name="", against=TARGETS[0], target_index=0, target_name="", offset_x=0,
        offset_y=0,
    ) -> io.NodeOutput:
        stack = layer_ops.entries(layers)
        if not stack:
            raise ValueError(
                "Layer Align was handed a stack with no layer in it, so there is nothing to "
                "move. Wire in a document that Add Layer or Layers From Bounding Boxes has "
                "put a layer into."
            )

        anchor = sizing.ALIGNMENTS.get(align, sizing.ALIGNMENTS[sizing.DEFAULT_ALIGNMENT])
        if against == TARGETS[1]:
            other = layer_ops.found(stack, target_index, target_name, where="Layer Align")
            box = layer_ops.box_of(stack[other])
        else:
            other = -1
            width, height = layer_ops.size_of(layers)
            box = (0, 0, width, height)

        if scope == SCOPES[1]:
            moving = [place for place in range(len(stack)) if place != other]
        else:
            moving = [layer_ops.found(stack, index, layer_name, where="Layer Align")]

        x = y = 0
        for place in moving:
            entry = dict(stack[place])
            x, y = layer_ops.aligned(entry, box, anchor)
            x, y = x + int(offset_x), y + int(offset_y)
            entry["x"], entry["y"] = x, y
            stack[place] = entry

        document = layer_ops.rebuilt(layers, stack)
        line = (
            f"{len(moving)} layer(s) aligned {align} against {against}, "
            f"last at {x}, {y}"
        )
        layer_ops.report(
            "Layer Align", line, document,
            counts={"moved": len(moving), "x": x, "y": y},
            facts={
                "align": align,
                "against": f"{box[2]}x{box[3]} at {box[0]}, {box[1]}",
                "offset": f"{offset_x:+d}, {offset_y:+d}",
            },
        )
        logger.info("Layer Align %s", line)
        return io.NodeOutput(document, x, y, ui=ui.PreviewText(line))

"""Scale one layer to reach the canvas or a box, without resampling its picture."""

from __future__ import annotations

from comfy_api.latest import io, ui

from ....modules.compat import limits
from ....modules.image import layer_ops, sizing
from ....modules.log import get_logger

logger = get_logger("nodes.image.layers")

#: What a layer is fitted to, in menu order.
TARGETS = ("the canvas", "a size", "another layer")


class LayerFit(io.ComfyNode):
    """Set the drawn size of one layer of a ``LAYERS`` document."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLayerFit",
            display_name="Layer Fit",
            search_aliases=[
                "WASLayerFit",
                "Layer Fit",
                "scale layer",
                "fit layer",
                "cover canvas",
                "compositor",
            ],
            category="WAS Suite/Image/Layers",
            description=(
                "Scale a layer to the canvas, a typed size or another layer, and place it at "
                "the anchor, so a plate covers a frame or a badge sits at a fixed height. "
                "The layer keeps its full-resolution picture until Create Layered Image "
                "flattens the stack."
            ),
            inputs=[
                io.Layers.Input(
                    "layers",
                    tooltip=(
                        "The stack holding the layer to scale. Wire in Add Layer, Layer Edit "
                        "or anything else answering a LAYERS document."
                    ),
                ),
                io.Combo.Input(
                    "resize_mode",
                    options=list(layer_ops.FITS),
                    tooltip=(
                        "How the layer reaches the box. `fit inside` keeps the whole layer "
                        "visible with space left over; `fill and overflow` covers the box and "
                        "runs past its edges; `stretch` takes the box exactly and distorts."
                    ),
                ),
                io.Combo.Input(
                    "against",
                    options=list(TARGETS),
                    tooltip=(
                        "What the layer is fitted to. `the canvas` is the document's own "
                        "size; `a size` is the width and height below; `another layer` is the "
                        "box the layer target_index or target_name draws in."
                    ),
                ),
                io.Int.Input(
                    "index",
                    default=-1,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Which layer is scaled, counting 0 from the back of the stack. -1 is "
                        "the front layer. Ignored while layer_name names one."
                    ),
                ),
                io.String.Input(
                    "layer_name",
                    default="",
                    tooltip=(
                        "Name of the layer to scale instead of index. Blank uses index. 'sky' "
                        "finds a layer called Sky, and finds Sky Backdrop where nothing is "
                        "called exactly Sky."
                    ),
                ),
                io.Int.Input(
                    "width",
                    default=1024,
                    min=1,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Width of the box on `a size`, in pixels. 1024 fits the layer into a "
                        "1024-wide box. Ignored by the other two targets."
                    ),
                ),
                io.Int.Input(
                    "height",
                    default=1024,
                    min=1,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Height of the box on `a size`, in pixels. 1024 fits the layer into a "
                        "1024-tall box. Ignored by the other two targets."
                    ),
                ),
                io.Combo.Input(
                    "align",
                    options=list(sizing.ALIGNMENT_NAMES),
                    default=sizing.DEFAULT_ALIGNMENT,
                    tooltip=(
                        "Where the scaled layer sits inside the box. `middle center` centres "
                        "it, `top left` puts its corner on the box's."
                    ),
                ),
                io.Float.Input(
                    "scale",
                    default=1.0,
                    min=0.01,
                    max=16.0,
                    step=0.01,
                    tooltip=(
                        "Multiplies the fitted size afterwards. 1.0 = exactly the fit, 0.9 = "
                        "a tenth smaller with a margin round it, 1.1 = a tenth larger."
                    ),
                ),
                io.Int.Input(
                    "target_index",
                    default=0,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Which layer is fitted to on `another layer`, counting 0 from the "
                        "back. 0 is the back layer, which is usually the plate."
                    ),
                ),
                io.String.Input(
                    "target_name",
                    default="",
                    tooltip=(
                        "Name of the layer to fit to instead of target_index. Blank uses "
                        "target_index. Read only on `another layer`."
                    ),
                ),
            ],
            outputs=[
                io.Layers.Output(
                    display_name="layers",
                    tooltip=(
                        "The stack with the layer scaled and placed, for Create Layered Image "
                        "or another edit."
                    ),
                ),
                io.Int.Output(
                    display_name="width",
                    tooltip="The width the layer is now drawn at, in pixels.",
                ),
                io.Int.Output(
                    display_name="height",
                    tooltip="The height the layer is now drawn at, in pixels.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls, layers, resize_mode=layer_ops.FITS[0], against=TARGETS[0], index=-1, layer_name="",
        width=1024, height=1024, align=sizing.DEFAULT_ALIGNMENT, scale=1.0,
        target_index=0, target_name="",
    ) -> io.NodeOutput:
        stack = layer_ops.entries(layers)
        if not stack:
            raise ValueError(
                "Layer Fit was handed a stack with no layer in it, so there is nothing to "
                "scale. Wire in a document that Add Layer or Layers From Bounding Boxes has "
                "put a layer into."
            )

        position = layer_ops.found(stack, index, layer_name, where="Layer Fit")
        if against == TARGETS[2]:
            other = layer_ops.found(stack, target_index, target_name, where="Layer Fit")
            if other == position:
                raise ValueError(
                    "Layer Fit cannot fit a layer to itself. Point target_index or "
                    "target_name at a different layer, or set against to 'the canvas'."
                )
            box = layer_ops.box_of(stack[other])
        elif against == TARGETS[1]:
            box = (0, 0, int(width), int(height))
        else:
            canvas_w, canvas_h = layer_ops.size_of(layers)
            box = (0, 0, canvas_w, canvas_h)

        entry = dict(stack[position])
        held_w, held_h = layer_ops.drawn_size(entry)
        new_w, new_h = layer_ops.fitted(entry, box[2], box[3], resize_mode)
        new_w = max(1, int(round(new_w * float(scale))))
        new_h = max(1, int(round(new_h * float(scale))))
        entry["w"], entry["h"] = new_w, new_h

        anchor = sizing.ALIGNMENTS.get(align, sizing.ALIGNMENTS[sizing.DEFAULT_ALIGNMENT])
        entry["x"], entry["y"] = layer_ops.aligned(entry, box, anchor)

        stack[position] = entry
        document = layer_ops.rebuilt(layers, stack)
        line = (
            f"layer {position} of {len(stack)} drawn at {new_w}x{new_h}, was "
            f"{held_w}x{held_h}, {resize_mode} in {box[2]}x{box[3]}"
        )
        layer_ops.report(
            "Layer Fit", line, document,
            counts={"layer": position, "width": new_w, "height": new_h},
            facts={
                "name": str(entry.get("name") or "unnamed"),
                "was": f"{held_w}x{held_h}",
                "box": f"{box[2]}x{box[3]} at {box[0]}, {box[1]}",
                "placement": f"{entry['x']}, {entry['y']}",
            },
        )
        logger.info("Layer Fit %s", line)
        return io.NodeOutput(document, new_w, new_h, ui=ui.PreviewText(line))

"""Cut the empty band off a layer's picture and move its placement back to match."""

from __future__ import annotations

from comfy_api.latest import io, ui

from ....modules.compat import limits
from ....modules.image import layer_ops
from ....modules.log import get_logger

logger = get_logger("nodes.image.layers")

#: Which layers the node trims, in menu order.
SCOPES = ("one layer", "every layer")


class LayerTrim(io.ComfyNode):
    """Crop the transparent margin off layers of a ``LAYERS`` document."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLayerTrim",
            display_name="Layer Trim",
            search_aliases=[
                "WASLayerTrim",
                "Layer Trim",
                "trim layer",
                "crop transparent",
                "autocrop layer",
                "compositor",
            ],
            category="WAS Suite/Image/Layers",
            description=(
                "Cut the empty band off a layer's picture and move x and y in by the same "
                "amount, so nothing appears to shift. A cut-out on a full transparent frame "
                "becomes a layer the size of the subject, so Layer Align centres the subject "
                "and effects skip the empty pixels."
            ),
            inputs=[
                io.Layers.Input(
                    "layers",
                    tooltip=(
                        "The stack holding the layer to trim. Wire in Add Layer, Layer Mask "
                        "or anything else answering a LAYERS document."
                    ),
                ),
                io.Combo.Input(
                    "scope",
                    options=list(SCOPES),
                    tooltip=(
                        "How many layers are trimmed. `one layer` trims the one index or "
                        "layer_name picks; `every layer` trims them all, which tidies a whole "
                        "set of cut-outs in one node."
                    ),
                ),
                io.Int.Input(
                    "index",
                    default=-1,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Which layer is trimmed, counting 0 from the back of the stack. -1 is "
                        "the front layer. Ignored while layer_name names one, or on `every "
                        "layer`."
                    ),
                ),
                io.String.Input(
                    "layer_name",
                    default="",
                    tooltip=(
                        "Name of the layer to trim instead of index. Blank uses index. 'sky' "
                        "finds a layer called Sky, and finds Sky Backdrop where nothing is "
                        "called exactly Sky."
                    ),
                ),
                io.Float.Input(
                    "threshold",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Coverage at or below which a pixel counts as empty. 0.0 keeps every "
                        "pixel that is not fully transparent, 0.05 also drops the faintest "
                        "fringe a soft cut-out leaves, 0.5 cuts into the edge itself."
                    ),
                ),
                io.Int.Input(
                    "padding",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Pixels of the empty band kept on every side. 0 = trimmed flush, 16 = "
                        "a 16px border left round the subject, which gives a later glow or "
                        "shadow room before it grows the layer."
                    ),
                ),
            ],
            outputs=[
                io.Layers.Output(
                    display_name="layers",
                    tooltip=(
                        "The stack with the layers trimmed, for Create Layered Image or "
                        "another edit."
                    ),
                ),
                io.Int.Output(
                    display_name="width",
                    tooltip="The last trimmed layer's picture width in pixels.",
                ),
                io.Int.Output(
                    display_name="height",
                    tooltip="The last trimmed layer's picture height in pixels.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls, layers, scope=SCOPES[0], index=-1, layer_name="", threshold=0.0, padding=0
    ) -> io.NodeOutput:
        stack = layer_ops.entries(layers)
        if not stack:
            raise ValueError(
                "Layer Trim was handed a stack with no layer in it, so there is nothing to "
                "trim. Wire in a document that Add Layer or Layers From Bounding Boxes has "
                "put a layer into."
            )

        if scope == SCOPES[1]:
            trimming = list(range(len(stack)))
        else:
            trimming = [layer_ops.found(stack, index, layer_name, where="Layer Trim")]

        width = height = 0
        cut = 0
        for place in trimming:
            before = stack[place]["image"]
            held_h, held_w = int(before.shape[-3]), int(before.shape[-2])
            entry, box = layer_ops.trimmed(stack[place], float(threshold))
            if int(padding) > 0:
                entry, box = cls.padded(stack[place], box, int(padding), held_w, held_h)
            stack[place] = entry
            width, height = box[2] - box[0], box[3] - box[1]
            if (width, height) != (held_w, held_h):
                cut += 1

        document = layer_ops.rebuilt(layers, stack)
        line = (
            f"{cut} of {len(trimming)} layer(s) trimmed, last now {width}x{height}"
            if cut else f"no empty band to cut off {len(trimming)} layer(s)"
        )
        layer_ops.report(
            "Layer Trim", line, document,
            counts={"trimmed": cut, "width": width, "height": height},
            facts={
                "threshold": f"{threshold:g}",
                "padding": f"{padding}px",
                "scope": scope,
            },
        )
        logger.info("Layer Trim %s", line)
        return io.NodeOutput(document, width, height, ui=ui.PreviewText(line))

    @staticmethod
    def padded(entry, box, padding: int, held_w: int, held_h: int):
        """One layer trimmed to a box widened by a border, held inside its own picture.

        Args:
            entry: The layer dictionary before trimming.
            box: The ``(left, top, right, bottom)`` the trim found.
            padding: Pixels of the empty band kept on every side.
            held_w: The layer picture's width in pixels.
            held_h: The layer picture's height in pixels.

        Returns:
            ``(entry, box)`` as :func:`layer_ops.trimmed` answers them.
        """
        left = max(0, box[0] - padding)
        top = max(0, box[1] - padding)
        right = min(held_w, box[2] + padding)
        bottom = min(held_h, box[3] + padding)
        pictures = entry["image"]
        stacked = pictures if pictures.ndim == 4 else pictures.unsqueeze(0)

        grown = dict(entry)
        grown["image"] = stacked[:, top:bottom, left:right]
        veil = entry.get("mask")
        if veil is not None and hasattr(veil, "ndim"):
            planes = veil if veil.ndim == 3 else veil.unsqueeze(0)
            if int(planes.shape[1]) == held_h and int(planes.shape[2]) == held_w:
                grown["mask"] = planes[:, top:bottom, left:right]

        drawn_w, drawn_h = layer_ops.drawn_size(entry)
        scale_x, scale_y = drawn_w / held_w, drawn_h / held_h
        grown["x"] = int(entry.get("x", 0)) + int(round(left * scale_x))
        grown["y"] = int(entry.get("y", 0)) + int(round(top * scale_y))
        grown["w"] = max(1, int(round((right - left) * scale_x)))
        grown["h"] = max(1, int(round((bottom - top) * scale_y)))
        return grown, (left, top, right, bottom)

"""Swap the picture inside one layer of a stack, keeping where it sits."""

from __future__ import annotations

import torch
from comfy_api.latest import io, ui

from ....modules.compat import limits
from ....modules.image import layer_ops
from ....modules.log import get_logger

logger = get_logger("nodes.image.layers")

#: What the replaced layer is drawn at, in menu order.
SIZES = ("keep the drawn size", "take the new picture's size")


class LayerReplaceImage(io.ComfyNode):
    """Put a different picture into one layer of a ``LAYERS`` document."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLayerReplaceImage",
            display_name="Layer Replace Image",
            search_aliases=[
                "WASLayerReplaceImage",
                "Layer Replace Image",
                "replace layer",
                "layer content",
                "round trip",
                "compositor",
            ],
            category="WAS Suite/Image/Layers",
            description=(
                "Replace one layer's picture with another, leaving its placement, angle, "
                "opacity, blend mode and name alone. Take a layer out with Layers to Image "
                "Batch, filter it, and put the result back where it came from."
            ),
            inputs=[
                io.Layers.Input(
                    "layers",
                    tooltip=(
                        "The stack holding the layer to change. Wire in Add Layer, Layer Edit "
                        "or anything else answering a LAYERS document."
                    ),
                ),
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The picture that goes in. A batch is carried whole, so the layer "
                        "draws one picture per frame the way Add Layer does."
                    ),
                ),
                io.Int.Input(
                    "index",
                    default=-1,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Which layer is replaced, counting 0 from the back of the stack. -1 "
                        "is the front layer. Ignored while layer_name names one."
                    ),
                ),
                io.String.Input(
                    "layer_name",
                    default="",
                    tooltip=(
                        "Name of the layer to replace instead of index. Blank uses index. "
                        "'sky' finds a layer called Sky, and finds Sky Backdrop where nothing "
                        "is called exactly Sky."
                    ),
                ),
                io.Combo.Input(
                    "size",
                    options=list(SIZES),
                    tooltip=(
                        "What the layer is drawn at afterwards. `keep the drawn size` = the "
                        "box it already filled, so a resized picture lands in the same place; "
                        "`take the new picture's size` = its own pixels, growing or shrinking "
                        "the layer."
                    ),
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip=(
                        "What the layer covers, white where it paints, as every mask in this "
                        "pack reads. Left unwired the layer keeps the mask it already had; "
                        "wire a fully white mask to clear one."
                    ),
                ),
                io.String.Input(
                    "name",
                    default="",
                    optional=True,
                    tooltip=(
                        "What the layer is called afterwards. Blank keeps the name it has, so "
                        "a later Layer Edit or Layer Select still finds it; 'sky graded' "
                        "renames it."
                    ),
                ),
            ],
            outputs=[
                io.Layers.Output(
                    display_name="layers",
                    tooltip=(
                        "The stack with the new picture in it, for Create Layered Image or "
                        "another edit."
                    ),
                ),
                io.Int.Output(
                    display_name="index",
                    tooltip="Which layer was replaced, counting 0 from the back.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls, layers, image, index=-1, layer_name="", size=SIZES[0], mask=None, name=""
    ) -> io.NodeOutput:
        stack = layer_ops.entries(layers)
        if not stack:
            raise ValueError(
                "Layer Replace Image was handed a stack with no layer in it, so there is "
                "nothing to replace. Wire in a document that Add Layer or Layers From "
                "Bounding Boxes has put a layer into."
            )

        position = layer_ops.found(stack, index, layer_name, where="Layer Replace Image")
        entry = dict(stack[position])
        held_w, held_h = layer_ops.drawn_size(entry)

        pictures = image if image.ndim == 4 else image.unsqueeze(0)
        entry["image"] = pictures
        if size == SIZES[0]:
            entry["w"], entry["h"] = held_w, held_h
        else:
            entry["w"], entry["h"] = int(pictures.shape[2]), int(pictures.shape[1])
        if mask is not None:
            planes = mask if mask.ndim == 3 else mask.unsqueeze(0)
            # A document mask is 1 where the layer is cut away, the opposite of a MASK input.
            entry["mask"] = torch.clamp(1.0 - planes.to(dtype=torch.float32), 0.0, 1.0)
        if name.strip():
            entry["name"] = name.strip()

        stack[position] = entry
        document = layer_ops.rebuilt(layers, stack)
        line = (
            f"layer {position} of {len(stack)} now carries a "
            f"{int(pictures.shape[2])}x{int(pictures.shape[1])} picture, drawn at "
            f"{entry['w']}x{entry['h']}"
        )
        layer_ops.report(
            "Layer Replace Image", line, document,
            counts={"layer": position, "frames": int(pictures.shape[0])},
            facts={
                "name": str(entry.get("name") or "unnamed"),
                "drawn": f"{entry['w']}x{entry['h']}",
                "placement": f"{entry.get('x', 0)}, {entry.get('y', 0)}",
            },
        )
        logger.info("Layer Replace Image %s", line)
        return io.NodeOutput(document, position, ui=ui.PreviewText(line))

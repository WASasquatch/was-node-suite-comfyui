"""Flatten a run of layers, or a whole document, into one raster layer."""

from __future__ import annotations

from comfy_api.latest import io, ui

from ....modules.compat import limits
from ....modules.image import layer_ops
from ....modules.log import get_logger

logger = get_logger("nodes.image.layers")

#: What the merged layer is called where the name is left blank.
MERGED = "merged"


class LayersMerge(io.ComfyNode):
    """Composite a run of a ``LAYERS`` document down to a single layer."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLayersMerge",
            display_name="Layers Merge",
            search_aliases=[
                "WASLayersMerge",
                "Layers Merge",
                "flatten layers",
                "merge down",
                "merge visible",
                "bake layer",
                "compositor",
            ],
            category="WAS Suite/Image/Layers",
            description=(
                "Flatten a run of layers into one, baking in each layer's opacity, blend mode, "
                "mask, placement, size, angle and mirroring. The merged layer takes the lowest "
                "place in the run at full opacity, and its picture also comes out on the image "
                "and mask sockets. Hidden layers in the run are dropped."
            ),
            inputs=[
                io.Layers.Input(
                    "layers",
                    tooltip=(
                        "The stack to flatten. Wire in Add Layer, Layer Edit or anything else "
                        "answering a LAYERS document."
                    ),
                ),
                io.Int.Input(
                    "first_index",
                    default=0,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Lowest layer of the run, counting 0 from the back of the stack. -1 is "
                        "the front layer, -2 the one under it."
                    ),
                ),
                io.Int.Input(
                    "last_index",
                    default=-1,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Highest layer of the run, counted the same way. -1 = the front of the "
                        "stack, so 0 and -1 together flatten everything."
                    ),
                ),
                io.String.Input(
                    "name",
                    default=MERGED,
                    tooltip=(
                        "What the merged layer is called. Blank calls it 'merged'. 'plate' "
                        "names it plate, which Layer Select and Layer Order then match on."
                    ),
                ),
            ],
            outputs=[
                io.Layers.Output(
                    display_name="layers",
                    tooltip=(
                        "The document with the run replaced by the one merged layer, numbered "
                        "from 0 at the back, for Create Layered Image or another edit."
                    ),
                ),
                io.Image.Output(
                    display_name="image",
                    tooltip=(
                        "The merged picture on its own, cropped to what the run covered. Wire "
                        "it to Image Preview to see the flatten without rendering the document."
                    ),
                ),
                io.Mask.Output(
                    display_name="mask",
                    tooltip=(
                        "What the merged picture leaves clear, white where nothing was drawn "
                        "and black where it is solid. Feeds Image Paste Crop or a sampler's "
                        "inpaint mask."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many layers were folded into the one, 1 while a single layer is baked.",
                ),
            ],
        )

    @classmethod
    def execute(cls, layers, first_index=0, last_index=-1, name=MERGED) -> io.NodeOutput:
        stack = layer_ops.entries(layers)
        if not stack:
            raise ValueError(
                "Layers Merge was handed a stack with no layer in it, so there is nothing to "
                "flatten. Wire in a document that Add Layer or Layers From Bounding Boxes has "
                "put a layer into."
            )

        total = len(stack)
        low = int(first_index) if first_index >= 0 else total + int(first_index)
        high = int(last_index) if last_index >= 0 else total + int(last_index)
        if low > high:
            low, high = high, low
        low = max(0, min(total - 1, low))
        high = max(0, min(total - 1, high))
        run = stack[low : high + 1]

        width, height = layer_ops.size_of(layers)
        entry, image, mask = layer_ops.merged(run, width, height, name.strip() or MERGED)
        document = layer_ops.rebuilt(layers, stack[:low] + [entry] + stack[high + 1 :])

        drawn = sum(1 for layer in run if bool(layer.get("visible", True)))
        line = (
            f"merged {len(run)} layer(s) at index {low} to {high}, {drawn} drawn, into "
            f"'{entry['name']}' at {entry['w']}x{entry['h']}"
        )
        if drawn:
            logger.info("Layers Merge %s", line)
        else:
            logger.warning(
                "Layers Merge found no visible layer between index %d and %d, so the merged "
                "layer is empty", low, high
            )
        layer_ops.report(
            "Layers Merge", line, document,
            counts={"merged": len(run), "drawn": drawn},
            facts={
                "name": str(entry["name"]),
                "merged size": f"{entry['w']}x{entry['h']}",
                "placement": f"{entry['x']}, {entry['y']}",
            },
        )
        return io.NodeOutput(document, image, mask, len(run), ui=ui.PreviewText(line))

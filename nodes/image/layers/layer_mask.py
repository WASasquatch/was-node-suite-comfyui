"""Set, combine, soften or clear the mask on one layer of a stack."""

from __future__ import annotations

import torch
from comfy_api.latest import io, ui

from ....modules.compat import limits
from ....modules.image import layer_fx, layer_ops
from ....modules.log import get_logger

logger = get_logger("nodes.image.layers")

#: How a wired mask meets the one the layer already carries, in menu order.
OPERATIONS = ("replace", "add", "subtract", "intersect", "invert", "remove")


class LayerMask(io.ComfyNode):
    """Change what one layer of a ``LAYERS`` document covers."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLayerMask",
            display_name="Layer Mask",
            search_aliases=[
                "WASLayerMask",
                "Layer Mask",
                "mask layer",
                "feather layer",
                "cut out layer",
                "compositor",
            ],
            category="WAS Suite/Image/Layers",
            description=(
                "Give one layer a mask, combine one with the mask it already has, soften its "
                "edge or take it away, to cut a layer to shape once it is in a stack. Masks "
                "are white where the layer paints."
            ),
            inputs=[
                io.Layers.Input(
                    "layers",
                    tooltip=(
                        "The stack holding the layer to mask. Wire in Add Layer, Layer Edit "
                        "or anything else answering a LAYERS document."
                    ),
                ),
                io.Int.Input(
                    "index",
                    default=-1,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Which layer is masked, counting 0 from the back of the stack. -1 is "
                        "the front layer. Ignored while layer_name names one."
                    ),
                ),
                io.String.Input(
                    "layer_name",
                    default="",
                    tooltip=(
                        "Name of the layer to mask instead of index. Blank uses index. 'sky' "
                        "finds a layer called Sky, and finds Sky Backdrop where nothing is "
                        "called exactly Sky."
                    ),
                ),
                io.Combo.Input(
                    "operation",
                    options=list(OPERATIONS),
                    tooltip=(
                        "What happens to the layer's mask. `replace` takes the wired mask as "
                        "it is; `add` widens what the layer covers; `subtract` cuts the wired "
                        "mask out of it; `intersect` keeps only what both cover; `invert` and "
                        "`remove` need no mask wired."
                    ),
                ),
                io.Float.Input(
                    "feather",
                    default=0.0,
                    min=0.0,
                    max=512.0,
                    step=0.5,
                    tooltip=(
                        "Pixels the edge is blurred over afterwards. 0 = a hard edge, 8 = a "
                        "soft one, 64 = a wide gradient a composite fades across."
                    ),
                ),
                io.Int.Input(
                    "expand",
                    default=0,
                    min=-512,
                    max=512,
                    step=1,
                    tooltip=(
                        "Pixels the covered area grows by before feathering. 0 = as it is, "
                        "8 = 8px wider all round, -8 = 8px tighter, which pulls a halo off a "
                        "cut-out edge."
                    ),
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip=(
                        "What the layer covers, white where it paints. Stretched to the "
                        "layer's own pixels. Not read on `invert` or `remove`."
                    ),
                ),
            ],
            outputs=[
                io.Layers.Output(
                    display_name="layers",
                    tooltip=(
                        "The stack with the layer's mask changed, for Create Layered Image "
                        "or another edit."
                    ),
                ),
                io.Mask.Output(
                    display_name="mask",
                    tooltip=(
                        "What the layer covers afterwards, white where it paints, at the "
                        "layer's own size."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls, layers, index=-1, layer_name="", operation=OPERATIONS[0], feather=0.0,
        expand=0, mask=None,
    ) -> io.NodeOutput:
        stack = layer_ops.entries(layers)
        if not stack:
            raise ValueError(
                "Layer Mask was handed a stack with no layer in it, so there is nothing to "
                "mask. Wire in a document that Add Layer or Layers From Bounding Boxes has "
                "put a layer into."
            )
        if mask is None and operation not in (OPERATIONS[4], OPERATIONS[5]):
            raise ValueError(
                f"Layer Mask on '{operation}' needs a mask wired into its mask input. Wire "
                f"one, or set operation to 'invert' or 'remove', which need none."
            )

        position = layer_ops.found(stack, index, layer_name, where="Layer Mask")
        entry = dict(stack[position])
        pictures = entry["image"]
        height = int(pictures.shape[-3])
        width = int(pictures.shape[-2])
        held = cls.covered(entry, height, width)

        if operation == OPERATIONS[5]:
            entry.pop("mask", None)
            cover = torch.ones_like(held)
        else:
            cover = cls.combined(operation, held, mask, height, width)
            cover = layer_fx.grow(cover, expand) if expand > 0 else cover
            cover = layer_fx.shrink(cover, -expand) if expand < 0 else cover
            cover = layer_fx.blur(cover, feather).clamp(0.0, 1.0)
            # A document mask is 1 where the layer is cut away, the opposite of a MASK.
            entry["mask"] = (1.0 - cover).unsqueeze(0)

        stack[position] = entry
        document = layer_ops.rebuilt(layers, stack)
        share = float(cover.mean())
        line = (
            f"{operation} on layer {position} of {len(stack)}, covering "
            f"{share * 100:.1f}% of its {width}x{height} picture"
        )
        layer_ops.report(
            "Layer Mask", line, document,
            counts={"layer": position, "coverage %": round(share * 100, 1)},
            facts={
                "name": str(entry.get("name") or "unnamed"),
                "operation": operation,
                "edge": f"feather {feather:g}px, expand {expand:+d}px",
            },
        )
        logger.info("Layer Mask %s", line)
        return io.NodeOutput(document, cover.unsqueeze(0), ui=ui.PreviewText(line))

    @staticmethod
    def covered(entry, height: int, width: int):
        """What a layer covers today, white where it paints, at its own size.

        Args:
            entry: The layer dictionary.
            height: The layer picture's height in pixels.
            width: The layer picture's width in pixels.

        Returns:
            A ``(height, width)`` tensor in 0 to 1.
        """
        pictures = entry["image"]
        blank = torch.ones(
            (height, width), dtype=torch.float32, device=pictures.device
        )
        veil = entry.get("mask")
        if not isinstance(veil, torch.Tensor):
            return blank
        plane = veil
        while plane.ndim > 2:
            plane = plane[0]
        return 1.0 - LayerMask.stretched(plane, height, width).clamp(0.0, 1.0)

    @staticmethod
    def stretched(plane, height: int, width: int):
        """One mask plane at another size.

        Args:
            plane: A ``(height, width)`` tensor.
            height: Height to reach in pixels.
            width: Width to reach in pixels.

        Returns:
            A ``(height, width)`` float tensor.
        """
        flat = plane.to(dtype=torch.float32)
        while flat.ndim > 2:
            flat = flat[0]
        if tuple(flat.shape) == (height, width):
            return flat
        return torch.nn.functional.interpolate(
            flat.unsqueeze(0).unsqueeze(0), size=(height, width),
            mode="bilinear", align_corners=False,
        )[0, 0]

    @staticmethod
    def combined(operation: str, held, mask, height: int, width: int):
        """The layer's coverage after one operation against a wired mask.

        Args:
            operation: One of :data:`OPERATIONS`, other than ``remove``.
            held: ``(height, width)`` coverage the layer already had.
            mask: The MASK wired in, or None on ``invert``.
            height: The layer picture's height in pixels.
            width: The layer picture's width in pixels.

        Returns:
            A ``(height, width)`` tensor in 0 to 1.
        """
        if operation == OPERATIONS[4]:
            return (1.0 - held).clamp(0.0, 1.0)
        given = LayerMask.stretched(mask, height, width).clamp(0.0, 1.0)
        given = given.to(device=held.device)
        if operation == OPERATIONS[1]:
            return torch.maximum(held, given)
        if operation == OPERATIONS[2]:
            return (held * (1.0 - given)).clamp(0.0, 1.0)
        if operation == OPERATIONS[3]:
            return torch.minimum(held, given)
        return given

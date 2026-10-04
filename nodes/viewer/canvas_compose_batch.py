"""Combining image batches of different sizes into one, on a transparent canvas."""

from __future__ import annotations

import functools

from comfy_api.latest import io

from ...modules import log
from ...modules.image import fit
from ...modules.image.batching import IMAGE_SLOTS, MAX_SLOTS, frames, image_slot_template
from ...modules.interface import size_report
from ...modules.util.slots import connected_in_order

REQUIRES = "viewer"

logger = log.get_logger("nodes.viewer")

#: Size of the transparent stand-in returned when no slot is fed.
EMPTY_SIZE = 64


class WASCanvasComposeBatch(io.ComfyNode):
    """Join any number of image batches into one, padding every image to the largest."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASCanvasComposeBatch",
            display_name="CV Canvas Compose Batch",
            search_aliases=[
                "WASCanvasComposeBatch",
                "CV Canvas Compose Batch",
                "canvas compose",
                "combine batches",
                "pad images",
            ],
            category="WAS Suite/View",
            description=(
                "Join any number of image batches into one, centring every image on a "
                "transparent canvas the size of the largest, so batches that differ in "
                "size combine where a plain batch node refuses them."
            ),
            inputs=[
                io.Autogrow.Input(
                    "images",
                    template=image_slot_template(),
                    tooltip=(
                        "The batches to join, in slot order. The list grows as slots are "
                        f"filled, up to {MAX_SLOTS}. Slots do not have to match each other "
                        "in size or in count, and an unconnected slot contributes nothing."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip=(
                        "Every image from every slot, each centred on a transparent canvas "
                        "as wide and as tall as the largest one. With no slot fed, a single "
                        f"transparent {EMPTY_SIZE}x{EMPTY_SIZE} image."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip=(
                        "How many frames the batch holds, which is the total across the "
                        "slots rather than the number of slots."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, images=None) -> io.NodeOutput:
        """Pad every connected frame onto one canvas and join them.

        Raises:
            ValueError: A slot holds a tensor that is not an image.
            MemoryError: Neither free memory nor any scratch drive has room for the batch.
        """
        import torch

        slots = images or {}
        names = connected_in_order(slots, IMAGE_SLOTS)
        per_slot = [frames(slots[name], name) for name in names]
        flat = [frame for slot in per_slot for frame in slot]
        if not flat:
            return io.NodeOutput(torch.zeros((1, EMPTY_SIZE, EMPTY_SIZE, fit.RGBA)), 0)

        # The canvas takes the tallest and the widest frame across every slot.
        height = max(int(frame.shape[1]) for frame in flat)
        width = max(int(frame.shape[2]) for frame in flat)
        logger.debug("padding %s image(s) to %sx%s", len(flat), width, height)

        from ...modules.image import scratch

        shape = (len(flat), height, width, fit.RGBA)
        dtype = functools.reduce(torch.promote_types, (frame.dtype for frame in flat))
        if all(frame.device.type == "cpu" for frame in flat):
            stacked = scratch.allocate(
                shape,
                dtype,
                node="CV Canvas Compose Batch",
                advice="Connecting fewer frames, or a smaller largest frame, also fits it.",
            )
        else:
            stacked = torch.empty(shape, dtype=dtype, device=flat[0].device)
        for index, frame in enumerate(flat):
            fit.pad_to(frame, height, width, transparent=True, out=stacked[index:index + 1])
        # The size report pairs the smallest frame with the canvas it was padded onto.
        smallest = min(flat, key=lambda frame: int(frame.shape[1]) * int(frame.shape[2]))
        size_report.publish(
            smallest,
            stacked,
            action="padded",
            facts={"frames": " plus ".join(str(len(slot)) for slot in per_slot)},
        )
        return io.NodeOutput(stacked, int(stacked.shape[0]))

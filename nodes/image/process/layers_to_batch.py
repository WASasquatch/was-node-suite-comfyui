"""Read a layer stack apart into frames, coverage, placement and names."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ....modules.compat.lists import require_values
from ....modules.compat.types import IMAGE_BOUNDS, LIST
from ....modules.image import layer_ops
from ....modules.interface import batch_report
from ....modules.log import get_logger

logger = get_logger("nodes.image.process")

#: Widget values deciding how big each frame comes out.
PLACEMENTS = ("on the canvas", "at its own size")

#: Channels every frame comes out with, so the batch stacks whatever the layers held.
CHANNELS = 3


class LayersToImageBatch(io.ComfyNode):
    """Split a ``LAYERS`` document into one frame per layer, with its placement beside it."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLayersToImageBatch",
            display_name="Layers to Image Batch",
            search_aliases=[
                "WASLayersToImageBatch",
                "Layers to Image Batch",
                "explode layers",
                "unstack layers",
                "layer stack",
                "compositor",
            ],
            category="WAS Suite/Image/Process",
            description=(
                "Take a layer stack apart: one frame per layer, its coverage as a mask, "
                "where it sits as bounds, and its name. Every layer of a document or a "
                "detector's regions can then go through the rest of the pack one frame at a "
                "time."
            ),
            inputs=[
                io.Layers.Input(
                    "layers",
                    tooltip=(
                        "The stack to read. Wire in Add Layer, Layers From Bounding Boxes or "
                        "anything else answering a LAYERS document."
                    ),
                ),
                io.Combo.Input(
                    "placement",
                    options=list(PLACEMENTS),
                    tooltip=(
                        "`on the canvas` pads every layer out to the document's canvas at "
                        "the position it sits, so the frames line up and can be recombined; "
                        "`at its own size` crops each to its own pixels, which suits sending "
                        "one layer through a filter."
                    ),
                ),
                io.Boolean.Input(
                    "hidden_layers",
                    default=False,
                    tooltip=(
                        "Whether a layer the compositor has switched off is read too. `off` "
                        "skips it, which matches what Create Layered Image draws; `on` reads "
                        "the whole document."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip=(
                        "One frame per picture, lowest in the stack first, drawn at the "
                        "size, angle and flip the layer carries. A layer holding a batch "
                        "answers one frame per picture."
                    ),
                ),
                io.Mask.Output(
                    display_name="masks",
                    tooltip=(
                        "What each layer covers, white where it paints. A layer with no mask "
                        "of its own covers its whole rectangle."
                    ),
                ),
                IMAGE_BOUNDS.Output(
                    display_name="bounds",
                    tooltip=(
                        "Where each layer sits on the canvas, one row per layer, for Bounded "
                        "Image Crop, Bounds to Mask or Draw Image Bounds."
                    ),
                ),
                LIST.Output(
                    display_name="names",
                    tooltip="What the compositor calls each layer, in the same order.",
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many frames came out, for a switch that handles none.",
                ),
            ],
        )

    @classmethod
    def execute(cls, layers, placement=PLACEMENTS[0], hidden_layers=False) -> io.NodeOutput:
        found = layer_ops.drawn(layers)
        if not hidden_layers:
            found = [frame for frame in found if frame.visible]
        require_values(
            found,
            "Layers to Image Batch was handed a stack with no layer in it to read. Wire a "
            "stack that Add Layer or Layers From Bounding Boxes has put a layer into, and "
            "switch hidden_layers on where every layer is hidden.",
        )

        width, height = layer_ops.size_of(layers)
        frames, masks, rows, names = [], [], [], []
        for frame in found:
            if placement == PLACEMENTS[0]:
                picture, cover = layer_ops.placed(frame, width, height, CHANNELS)
            else:
                picture, cover = frame.image, frame.coverage
            frames.append(picture)
            masks.append(cover)
            rows.append(
                (
                    frame.y,
                    frame.y + int(frame.image.shape[0]) - 1,
                    frame.x,
                    frame.x + int(frame.image.shape[1]) - 1,
                )
            )
            names.append(frame.name)

        # Layers read at their own size differ frame to frame, so they cannot be one tensor.
        if placement != PLACEMENTS[0] and len({frame.shape for frame in frames}) > 1:
            raise ValueError(
                f"Layers to Image Batch cannot stack {len(frames)} layers of different sizes "
                f"into one batch. Set placement to '{PLACEMENTS[0]}', which pads every layer "
                f"out to the canvas."
            )

        images = torch.stack(frames, dim=0)
        coverage = torch.stack(masks, dim=0)
        size, mode = batch_report.describe_images(images)
        batch_report.publish(
            len(frames), len(found), size, mode, images.numel() * images.element_size()
        )
        logger.info("Layers to Image Batch read %d frame(s) off a %dx%d canvas",
                    len(frames), width, height)
        return io.NodeOutput(images, coverage, rows, names, len(frames))

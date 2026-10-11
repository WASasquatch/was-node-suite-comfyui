"""Build a layer stack out of a batch of pictures, with placement and names beside it."""

from __future__ import annotations

import torch
from comfy_api.latest import io, ui

from ....modules.compat import limits
from ....modules.compat.types import IMAGE_BOUNDS, LIST
from ....modules.image import layer_ops
from ....modules.log import get_logger

logger = get_logger("nodes.image.layers")

#: How a frame with no bounds of its own is placed, in menu order.
PLACEMENTS = ("all at 0, 0", "stepped by offset", "spread across the canvas")


class LayersFromImageBatch(io.ComfyNode):
    """Turn a batch of pictures into a ``LAYERS`` document, one layer per frame."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLayersFromImageBatch",
            display_name="Layers from Image Batch",
            search_aliases=[
                "WASLayersFromImageBatch",
                "Layers from Image Batch",
                "batch to layers",
                "stack images",
                "implode layers",
                "compositor",
            ],
            category="WAS Suite/Image/Layers",
            description=(
                "Build a layer stack from a batch: one layer per frame, lowest first. Wire "
                "bounds in to place each frame at its rectangle, so a batch that Layers to "
                "Image Batch took apart, or a set of detector crops, goes back together."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The frames to stack, lowest in the stack first. One layer is made "
                        "per frame."
                    ),
                ),
                io.Combo.Input(
                    "placement",
                    options=list(PLACEMENTS),
                    tooltip=(
                        "Where a frame goes when no bounds are wired. `all at 0, 0` piles "
                        "them up; `stepped by offset` moves each one offset_x and offset_y "
                        "past the one below, which fans a batch out; `spread across the "
                        "canvas` lays them left to right and wraps."
                    ),
                ),
                io.Int.Input(
                    "offset_x",
                    default=0,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Pixels each frame sits right of the one below it on `stepped by "
                        "offset`. 0 = straight on top, 24 = 24px right each time."
                    ),
                ),
                io.Int.Input(
                    "offset_y",
                    default=0,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Pixels each frame sits below the one below it on `stepped by "
                        "offset`. 0 = straight on top, 24 = 24px down each time."
                    ),
                ),
                io.Int.Input(
                    "canvas_width",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Width the stack is drawn on. 0 = no canvas of its own, so Create "
                        "Layered Image sizes it to whatever the layers reach; 1920 pins it."
                    ),
                ),
                io.Int.Input(
                    "canvas_height",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Height the stack is drawn on. 0 = no canvas of its own; 1080 pins "
                        "it. Read only alongside a canvas_width above 0."
                    ),
                ),
                io.String.Input(
                    "name_prefix",
                    default="layer",
                    tooltip=(
                        "What each layer is called when no names are wired, with its number "
                        "after it: 'layer' gives layer 1, layer 2. Blank leaves them unnamed."
                    ),
                ),
                io.Layers.Input(
                    "layers",
                    optional=True,
                    tooltip=(
                        "A stack the new layers are added on top of. Left unwired the batch "
                        "becomes a stack of its own."
                    ),
                ),
                IMAGE_BOUNDS.Input(
                    "bounds",
                    optional=True,
                    tooltip=(
                        "Where each frame sits, one row per frame, as Layers to Image Batch "
                        "and Mask to Bounds answer them. Fewer rows than frames and the rest "
                        "fall back to placement."
                    ),
                ),
                io.Mask.Input(
                    "masks",
                    optional=True,
                    tooltip=(
                        "What each frame covers, white where it paints, as every mask in this "
                        "pack reads. One mask is used for every frame; a batch is paired "
                        "frame by frame."
                    ),
                ),
                LIST.Input(
                    "names",
                    optional=True,
                    tooltip=(
                        "What each layer is called, in the same order as the frames, as "
                        "Layers to Image Batch answers them. Overrides name_prefix."
                    ),
                ),
            ],
            outputs=[
                io.Layers.Output(
                    display_name="layers",
                    tooltip=(
                        "The stack, numbered from 0 at the back, for Create Layered Image, "
                        "Layers Merge or any of the effects."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many layers the stack holds once the batch was added.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls, images, placement=PLACEMENTS[0], offset_x=0, offset_y=0, canvas_width=0,
        canvas_height=0, name_prefix="layer", layers=None, bounds=None, masks=None,
        names=None,
    ) -> io.NodeOutput:
        frames = images if images.ndim == 4 else images.unsqueeze(0)
        if int(frames.shape[0]) == 0:
            raise ValueError(
                "Layers from Image Batch was handed an empty batch, so there is no frame to "
                "make a layer out of. Wire in a batch holding at least one image."
            )

        stack = layer_ops.entries(layers) if layers is not None else []
        rows = list(bounds or [])
        titles = [str(value) for value in (names or [])]
        width = int(canvas_width) if canvas_width > 0 and canvas_height > 0 else 0
        height = int(canvas_height) if width else 0

        placed = 0
        for index in range(int(frames.shape[0])):
            picture = frames[index : index + 1]
            frame_h, frame_w = int(picture.shape[1]), int(picture.shape[2])
            x, y = cls.corner(index, rows, placement, offset_x, offset_y,
                              frame_w, frame_h, width, height)
            entry = {
                "image": picture,
                "type": "raster",
                "x": x,
                "y": y,
                "w": frame_w,
                "h": frame_h,
                "opacity": 1.0,
                "blend_mode": "normal",
                "visible": True,
                "flip_h": False,
                "flip_v": False,
                "rotation": 0.0,
            }
            title = titles[index] if index < len(titles) else ""
            if not title and name_prefix.strip():
                title = f"{name_prefix.strip()} {index + 1}"
            if title:
                entry["name"] = title
            plane = cls.veil(masks, index)
            if plane is not None:
                entry["mask"] = plane
            stack.append(entry)
            placed += 1

        document = layer_ops.rebuilt(layers if isinstance(layers, dict) else {}, stack)
        if width:
            document["canvas"] = (width, height)

        line = f"{placed} frame(s) added, {len(stack)} layer(s) in the stack"
        layer_ops.report(
            "Layers from Image Batch", line, document,
            counts={"added": placed},
            facts={
                "placed by": "bounds" if rows else placement,
                "named by": "names" if titles else (name_prefix.strip() or "nothing"),
            },
        )
        logger.info("Layers from Image Batch %s", line)
        return io.NodeOutput(document, len(stack), ui=ui.PreviewText(line))

    @staticmethod
    def corner(index, rows, placement, offset_x, offset_y, frame_w, frame_h, width, height):
        """Where one frame's top left corner lands on the canvas.

        Args:
            index: Which frame of the batch, counting 0.
            rows: The bounds rows wired in, which may be shorter than the batch.
            placement: One of :data:`PLACEMENTS`, read when no row covers this frame.
            offset_x: Pixels each frame steps right of the one before it.
            offset_y: Pixels each frame steps below the one before it.
            frame_w: The frame's width in pixels.
            frame_h: The frame's height in pixels.
            width: Canvas width, or 0 where the stack names none.
            height: Canvas height, or 0 where the stack names none.

        Returns:
            ``(x, y)`` in pixels.
        """
        if index < len(rows):
            row = rows[index]
            if isinstance(row, (tuple, list)) and len(row) >= 4:
                return int(row[2]), int(row[0])
        if placement == PLACEMENTS[1]:
            return int(offset_x) * index, int(offset_y) * index
        if placement == PLACEMENTS[2] and width and frame_w:
            across = max(1, width // frame_w)
            return (index % across) * frame_w, (index // across) * frame_h
        return 0, 0

    @staticmethod
    def veil(masks, index):
        """One frame's document mask, 1 where the layer is cut away, or None.

        Args:
            masks: The MASK wired in, white where the layer paints, or None.
            index: Which frame of the batch, counting 0.

        Returns:
            A ``(1, height, width)`` tensor, or None where no mask covers this frame.
        """
        if not isinstance(masks, torch.Tensor):
            return None
        planes = masks if masks.ndim == 3 else masks.unsqueeze(0)
        if int(planes.shape[0]) == 0:
            return None
        slot = 0 if int(planes.shape[0]) == 1 else index
        if slot >= int(planes.shape[0]):
            return None
        plane = planes[slot : slot + 1].to(dtype=torch.float32)
        return torch.clamp(1.0 - plane, 0.0, 1.0)

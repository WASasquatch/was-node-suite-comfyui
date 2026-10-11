"""Convert between ComfyUI's bounding boxes and the pack's image bounds."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.types import IMAGE_BOUNDS
from ....modules.image import boxes
from ....modules.image.bounds import rows as bounds_rows
from ....modules.interface import run_result
from ....modules.log import get_logger

logger = get_logger("nodes.image.bounds")


def _report(node: str, count: int, first) -> None:
    """Report what was converted to the node's own panel. Never raises."""
    try:
        if not run_result.watching():
            return
        run_result.publish(
            status=run_result.OK if count else run_result.WARNING,
            summary=f"{count} region(s)" if count else "nothing to convert",
            counts={"regions": count},
            facts={"first": str(first) if first is not None else "none"},
        )
    except Exception as error:
        logger.debug("%s published no report (%s)", node, error)


class BoundingBoxesToBounds(io.ComfyNode):
    """Read a detector's bounding boxes as an ``IMAGE_BOUNDS`` value."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASBoundingBoxesToBounds",
            display_name="Bounding Boxes to Bounds",
            search_aliases=[
                "WASBoundingBoxesToBounds",
                "Bounding Boxes to Bounds",
                "bbox to bounds",
                "detection to bounds",
                "sam3",
                "rt-detr",
            ],
            category="WAS Suite/Image/Bound",
            description=(
                "Turn bounding boxes into bounds, so a region found by a detector reaches "
                "the cropping, blending and masking nodes in this pack. Reads one box, a list "
                "of them, or a detector's per-frame lists, and hands on one bounds row per box."
            ),
            inputs=[
                io.MultiType.Input(
                    "bounding_boxes",
                    [io.BoundingBox, io.BoundingBoxes, io.String],
                    tooltip=(
                        "The boxes to convert. Wire in SAM3 Detect, Run Real-Time Detection, "
                        "SDPose Face Bounding Boxes or Create Bounding Boxes. JSON text "
                        "holding the same boxes is read too."
                    ),
                ),
            ],
            outputs=[
                IMAGE_BOUNDS.Output(
                    display_name="bounds",
                    tooltip=(
                        "One row per box, ready for Bounded Image Crop, Bounds to Mask, "
                        "Inset Image Bounds or Draw Image Bounds."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many regions came out, for a switch that handles none.",
                ),
            ],
        )

    @classmethod
    def execute(cls, bounding_boxes) -> io.NodeOutput:
        rows = boxes.boxes_to_rows(bounding_boxes)
        if not rows:
            logger.warning(
                "Bounding Boxes to Bounds was handed no box with any area, so it answers no "
                "bounds. Check the detector found something before this node."
            )
        _report("Bounding Boxes to Bounds", len(rows), rows[0] if rows else None)
        return io.NodeOutput(rows, len(rows))


class BoundsToBoundingBoxes(io.ComfyNode):
    """Emit an ``IMAGE_BOUNDS`` value as ComfyUI bounding boxes."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASBoundsToBoundingBoxes",
            display_name="Bounds to Bounding Boxes",
            search_aliases=[
                "WASBoundsToBoundingBoxes",
                "Bounds to Bounding Boxes",
                "bounds to bbox",
                "region to bbox",
            ],
            category="WAS Suite/Image/Bound",
            description=(
                "Turn bounds into bounding boxes, so a region measured in this pack reaches "
                "ComfyUI's own box nodes: Crop By Bounding Boxes, Draw BBoxes and Layers "
                "From Bounding Boxes. Each bounds row becomes one box with its origin at the "
                "top left corner, which is the same rectangle counted the other way."
            ),
            inputs=[
                IMAGE_BOUNDS.Input(
                    "bounds",
                    tooltip=(
                        "The regions to convert. Wire in Mask to Bounds, Image Bounds or "
                        "Inset Image Bounds."
                    ),
                ),
                io.String.Input(
                    "label",
                    default="",
                    tooltip=(
                        "A description carried on every box, such as `face` or `product`. "
                        "Draw BBoxes prints it beside the rectangle. Empty attaches none."
                    ),
                ),
            ],
            outputs=[
                io.BoundingBox.Output(
                    display_name="bounding_boxes",
                    tooltip=(
                        "One box per bounds row, ready for Crop By Bounding Boxes, Draw "
                        "BBoxes or Layers From Bounding Boxes."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many boxes came out, for a switch that handles none.",
                ),
            ],
        )

    @classmethod
    def execute(cls, bounds, label="") -> io.NodeOutput:
        rows = bounds_rows(bounds)
        metadata = {"desc": label} if label else None
        found = boxes.rows_to_boxes(rows, metadata)
        if not found:
            logger.warning(
                "Bounds to Bounding Boxes was handed no bounds, so it answers no boxes. "
                "Check the node before it measured a region."
            )
        _report("Bounds to Bounding Boxes", len(found), found[0] if found else None)
        return io.NodeOutput(found, len(found))

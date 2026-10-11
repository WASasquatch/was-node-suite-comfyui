"""Thin, grow, order and cap the regions a detector found."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat import limits
from ....modules.convert.tensors import image_planes
from ....modules.image import boxes
from ....modules.interface import run_result
from ....modules.log import get_logger

logger = get_logger("nodes.image.bounds")

#: Overlap share at which nothing is thinned out.
KEEP_EVERYTHING = 1.0


class BoundingBoxesFilter(io.ComfyNode):
    """Reduce a detector's boxes to the ones worth acting on."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASBoundingBoxesFilter",
            display_name="Bounding Boxes Filter",
            search_aliases=[
                "WASBoundingBoxesFilter",
                "Bounding Boxes Filter",
                "nms",
                "non maximum suppression",
                "filter detections",
                "sort boxes",
                "bbox",
            ],
            category="WAS Suite/Image/Bound",
            description=(
                "Reduce a detector's regions to a shortlist: drop the ones too small to "
                "matter and the duplicates over one subject, pad them out for a crop, put them "
                "in order and keep the first few. Hands on the regions kept, with counts of "
                "what was kept and dropped."
            ),
            inputs=[
                io.MultiType.Input(
                    "bounding_boxes",
                    [io.BoundingBox, io.BoundingBoxes, io.String],
                    tooltip=(
                        "The regions to reduce. Wire in SAM3 Detect, Run Real-Time "
                        "Detection, MediaPipe's landmarker or Bounds to Bounding Boxes. "
                        "JSON text holding the same boxes is read too."
                    ),
                ),
                io.Combo.Input(
                    "order",
                    options=list(boxes.ORDERS),
                    tooltip=(
                        "`as found` = the detector's own order; `area, largest first` = the "
                        "main subject at index 0, for an index switch; `left to right` and "
                        "`top to bottom` = a row of faces or a contact sheet."
                    ),
                ),
                io.Int.Input(
                    "keep",
                    default=0,
                    min=0,
                    max=4096,
                    step=1,
                    tooltip=(
                        "How many to keep after ordering. 0 = all, 1 = only the first, 5 = "
                        "the first five. Set order first, since this counts from the top of "
                        "that order."
                    ),
                ),
                io.Float.Input(
                    "overlap",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How much two regions may share before the smaller is dropped. 1.0 = "
                        "keep everything, 0.5 = drop a region sharing half its area with a "
                        "larger one, 0.0 = drop anything touching. Use it where a detector "
                        "found the same subject twice."
                    ),
                ),
                io.Int.Input(
                    "min_width",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Narrowest region kept, in pixels. 0 = keep every width, 64 = drop "
                        "anything under 64 across, which clears the specks a low threshold "
                        "picks up."
                    ),
                ),
                io.Int.Input(
                    "min_height",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Shortest region kept, in pixels. 0 = keep every height, 64 = drop "
                        "anything under 64 tall."
                    ),
                ),
                io.Int.Input(
                    "expand",
                    default=0,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Pixels every edge moves out by, applied after the size test. 0 = as "
                        "found, 32 = 32px of room on every side for a crop, -8 = pulled in, "
                        "which trims a detector's habit of framing loosely."
                    ),
                ),
                io.Image.Input(
                    "image",
                    optional=True,
                    tooltip=(
                        "The picture the regions were found on. Connected, every region is "
                        "held inside the frame, so an expanded one cannot run off the edge. "
                        "Unconnected, a region may sit partly outside the picture."
                    ),
                ),
            ],
            outputs=[
                io.BoundingBox.Output(
                    display_name="bounding_boxes",
                    tooltip="The regions kept, in the chosen order.",
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many were kept, for a switch that handles none.",
                ),
                io.Int.Output(
                    display_name="dropped",
                    tooltip="How many the settings removed, so a filter that took too much shows.",
                ),
            ],
        )

    @classmethod
    def execute(cls, bounding_boxes, order=boxes.ORDERS[0], keep=0, overlap=KEEP_EVERYTHING,
                min_width=0, min_height=0, expand=0, image=None) -> io.NodeOutput:
        found = boxes.normalise(bounding_boxes)
        offered = len(found)

        kept = [
            box
            for box in found
            if int(box.get("width", 0)) >= min_width and int(box.get("height", 0)) >= min_height
        ]
        if overlap < KEEP_EVERYTHING:
            kept = boxes.suppressed(kept, overlap)
        if expand:
            kept = [boxes.grown(box, expand) for box in kept]

        frame = None
        if image is not None:
            planes = image_planes(image)
            if planes:
                frame = (int(planes[0].shape[1]), int(planes[0].shape[0]))
                kept = [boxes.clamped(box, frame[0], frame[1]) for box in kept]

        kept = [box for box in boxes.ordered(kept, order) if boxes.area(box) > 0]
        if keep:
            kept = kept[:keep]

        cls.report(offered, kept, frame)
        logger.info(
            "Bounding Boxes Filter kept %d of %d region(s)", len(kept), offered
        )
        return io.NodeOutput(kept, len(kept), max(0, offered - len(kept)))

    @classmethod
    def report(cls, offered: int, kept, frame) -> None:
        """Draw what the settings removed on the node. Never raises."""
        try:
            if not run_result.watching():
                return
            largest = max((boxes.area(box) for box in kept), default=0)
            run_result.publish(
                status=run_result.OK if kept else run_result.WARNING,
                summary=(
                    f"{len(kept)} of {offered} region(s) kept"
                    if kept
                    else f"every one of {offered} region(s) was filtered out"
                ),
                counts={"kept": len(kept), "offered": offered, "largest px": largest},
                facts={"frame": f"{frame[0]}x{frame[1]}" if frame else "not connected"},
            )
        except Exception as error:
            logger.debug("Bounding Boxes Filter published no report (%s)", error)

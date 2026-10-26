"""Draw layout boxes on a frame and hand them on as boxes, prompt elements, bounds and masks."""

from __future__ import annotations

from comfy_api.latest import io
from comfy_execution.graph_utils import ExecutionBlocker

from ....modules.compat import limits
from ....modules.compat.types import IMAGE_BOUNDS
from ....modules.convert.tensors import image_planes, plane_shape
from ....modules.image import box_layout, boxes
from ....modules.interface import channel, preview


class BoundingBoxEditor(io.ComfyNode):
    """Layout boxes edited on the node, for layout prompts and region nodes."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASBoundingBoxEditor",
            display_name="Bounding Box Editor",
            search_aliases=[
                "WASBoundingBoxEditor",
                "Bounding Box Editor",
                "bbox",
                "bounding boxes",
                "layout boxes",
                "regions",
                "MOSS layout",
                "FLUX 3 layout",
                "Create Bounding Boxes",
            ],
            category="WAS Suite/Image/Bound",
            description=(
                "Draw layout boxes on a picture or a blank frame and give each a kind, caption, "
                "text and colour palette. Hands them on as bounding boxes, prompt elements, "
                "bounds and masks for MOSS-Image and FLUX 3 layout prompts and the cropping and "
                "masking nodes. A layout drawn at one size is refitted to another without stretching."
            ),
            is_output_node=True,
            inputs=[
                io.Int.Input(
                    "width",
                    default=1024,
                    min=16,
                    max=limits.max_resolution(),
                    step=8,
                    tooltip=(
                        "Frame width in pixels when no image is wired. `1024` = 1024px wide. A "
                        "wired image's own width replaces it."
                    ),
                ),
                io.Int.Input(
                    "height",
                    default=1024,
                    min=16,
                    max=limits.max_resolution(),
                    step=8,
                    tooltip=(
                        "Frame height in pixels when no image is wired. `1024` = 1024px tall. A "
                        "wired image's own height replaces it."
                    ),
                ),
                io.String.Input(
                    "layout",
                    default="",
                    multiline=True,
                    dynamic_prompts=False,
                    advanced=True,
                    tooltip=(
                        "The boxes as JSON the panel writes, such as `{\"frame\": [1344, 768], "
                        "\"boxes\": [...]}`. Empty = no boxes. Paste a layout from another "
                        "Bounding Box Editor, a Create Bounding Boxes box list or a prompt "
                        "element list."
                    ),
                ),
                io.Combo.Input(
                    "on_resize",
                    options=list(box_layout.RESIZE_MODES),
                    default=box_layout.RESIZE_MODES[0],
                    advanced=True,
                    tooltip=(
                        "Used when the layout was drawn at another frame size. `fit` = whole "
                        "layout inside, unstretched; `fill` = covers the frame, edges may be "
                        "cut; `stretch` = scaled to the frame, shapes change; `keep size` = "
                        "pixel size kept, centred."
                    ),
                ),
                io.Int.Input(
                    "grid",
                    default=16,
                    min=1,
                    max=256,
                    step=1,
                    advanced=True,
                    tooltip=(
                        "Pixels box edges snap to while drawing in the panel and when a layout "
                        "is refitted. `1` = no snapping; `16` = 16px steps; `64` = coarse blocks."
                    ),
                ),
                io.Image.Input(
                    "image",
                    optional=True,
                    tooltip=(
                        "Optional picture to lay boxes over. Its size replaces width and height, "
                        "and its first frame is drawn behind the boxes in the panel and preview."
                    ),
                ),
                io.MultiType.Input(
                    "bboxes",
                    [io.BoundingBox, io.BoundingBoxes, io.Array, io.String],
                    optional=True,
                    tooltip=(
                        "Boxes to start from: SAM3 Detect or Run Real-Time Detection results, a "
                        "Create Bounding Boxes output, prompt elements or JSON text, in the "
                        "frame's pixels. bboxes_mode decides how they join the layout."
                    ),
                ),
                io.Combo.Input(
                    "bboxes_mode",
                    options=list(box_layout.BBOXES_MODES),
                    default=box_layout.BBOXES_MODES[0],
                    advanced=True,
                    tooltip=(
                        "`add` = incoming boxes come after the layout's own; `replace` = only "
                        "the incoming boxes are output; `ignore` = the input is not read."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="preview",
                    tooltip=(
                        "The boxes numbered and captioned over the image, or over a dark frame, "
                        "at most 1024px on the long edge."
                    ),
                ),
                io.BoundingBox.Output(
                    display_name="bboxes",
                    tooltip=(
                        "Every box in pixels, carrying its kind, caption, text and palette, for "
                        "Crop By Bounding Boxes, Draw BBoxes, SAM3 Detect or another editor."
                    ),
                ),
                io.Array.Output(
                    display_name="elements",
                    tooltip=(
                        "Prompt elements for MOSS-Image Layout Prompt and FLUX 3 Image Layout "
                        "Prompt, numbered back to front as the preview shows."
                    ),
                ),
                io.Int.Output(
                    display_name="width",
                    tooltip=(
                        "Width of the frame the boxes are in: the image's when one is wired, "
                        "otherwise the width widget. Wire it into the layout prompt's width."
                    ),
                ),
                io.Int.Output(
                    display_name="height",
                    tooltip=(
                        "Height of the frame the boxes are in: the image's when one is wired, "
                        "otherwise the height widget. Wire it into the layout prompt's height."
                    ),
                ),
                io.String.Output(
                    display_name="character_boxes",
                    tooltip=(
                        "Numbers of the `character` boxes, such as `2, 5`, for MOSS-Image Layout "
                        "Prompt's character_boxes input. Empty when there are none."
                    ),
                ),
                IMAGE_BOUNDS.Output(
                    display_name="bounds",
                    tooltip=(
                        "One bounds row per box, for Bounded Image Crop, Bounds to Mask or Draw "
                        "Image Bounds."
                    ),
                ),
                io.Mask.Output(
                    display_name="masks",
                    tooltip=(
                        "One mask per box, white inside it, in box order. No boxes = one black "
                        "mask the size of the frame."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, width, height, layout="", on_resize=box_layout.RESIZE_MODES[0], grid=16,
                image=None, bboxes=None, bboxes_mode=box_layout.BBOXES_MODES[0]) -> io.NodeOutput:
        """Read the layout, refit it to the frame, join the incoming boxes and build every output.

        Raises:
            ValueError: The layout or the incoming boxes are unreadable or break a ceiling.
        """
        if bboxes_mode not in box_layout.BBOXES_MODES:
            raise ValueError(
                f"bboxes_mode is {bboxes_mode!r}; it takes one of "
                f"{', '.join(box_layout.BBOXES_MODES)}."
            )
        planes = image_planes(image) if image is not None else []
        if planes:
            frame_height, frame_width, _ = plane_shape(planes[0])
            size = (int(frame_width), int(frame_height))
        else:
            size = (int(width), int(height))

        drawn, layout_frame = box_layout.parse_layout(layout, size)
        shown = [box for box in drawn if not box["hidden"]]
        placed = box_layout.refit(shown, layout_frame or size, size, on_resize, grid)
        incoming = (
            [] if bboxes_mode == "ignore"
            else box_layout.incoming_boxes(bboxes, size, len(drawn))
        )
        found = incoming if bboxes_mode == "replace" else placed + incoming
        if len(found) > box_layout.MAX_BOXES:
            raise ValueError(
                f"The layout's {len(placed)} boxes and the {len(incoming)} wired into bboxes make "
                f"{len(found)}, past the {box_layout.MAX_BOXES} the node outputs. Delete boxes in "
                f"the panel, thin the incoming ones with Bounding Boxes Filter, or set "
                f"bboxes_mode to replace or ignore."
            )

        bbox_frame = box_layout.to_bbox_frame(found)
        refusal = box_layout.mask_refusal(len(found), size)
        masks = ExecutionBlocker(refusal) if refusal else box_layout.masks(found, size)
        if planes:
            preview.publish(image, slot="image")
        report = {
            "frame": list(size),
            "source": "image" if planes else "size",
            "layout_frame": list(layout_frame) if layout_frame else None,
            "refit": on_resize if layout_frame and tuple(layout_frame) != size else None,
            "count": len(found),
            "hidden": len(drawn) - len(shown),
            "incoming": [box_layout.report_box(box) for box in incoming],
            "workflow": channel.executing_workflow_id() or "",
        }
        return io.NodeOutput(
            box_layout.render_preview(found, size, image if planes else None),
            bbox_frame,
            box_layout.to_elements(found, size),
            size[0],
            size[1],
            box_layout.character_numbers(found),
            boxes.boxes_to_rows(bbox_frame),
            masks,
            ui={"was_box_layout": [report]},
        )

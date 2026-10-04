"""Draw a bounds row onto the image it describes."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.types import IMAGE_BOUNDS
from ....modules.convert.tensors import image_planes, pil2mask, pil2tensor, tensor2pil
from ....modules.image import bounds, draw


class DrawImageBounds(io.ComfyNode):
    """Draw an ``IMAGE_BOUNDS`` value as rectangles over the image, and as a mask."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASDrawImageBounds",
            display_name="Draw Image Bounds",
            search_aliases=[
                "WASDrawImageBounds", "Draw Image Bounds",
                "bounding box",
                "draw box",
                "visualise bounds",
                "annotate",
                "rectangle",
            ],
            category="WAS Suite/Image/Bound",
            description=(
                "Draw a bounds value as rectangles on the image, with an optional fill and "
                "label, and return the same rectangles as a mask."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The images to draw on. A single image gets every row of the bounds "
                        "drawn on it, which is how a whole set of detected regions is seen "
                        "at once; a batch gets one row each, in order."
                    ),
                ),
                IMAGE_BOUNDS.Input(
                    "image_bounds",
                    tooltip=(
                        "The regions to draw, from Image Bounds, Inset Image Bounds, Mask "
                        "Crop Region or any other node with a bounds output."
                    ),
                ),
                io.String.Input(
                    "color",
                    default="#00FF00",
                    multiline=False,
                    tooltip=(
                        "Colour of the outline, as #RRGGBB, #RRGGBBAA or a name. Green shows "
                        "up on most photographs; magenta is the better choice over foliage."
                    ),
                ),
                io.Int.Input(
                    "thickness",
                    default=3,
                    min=0,
                    max=256,
                    step=1,
                    tooltip=(
                        "Width of the outline in pixels. Scale it with the image, 3 is "
                        "clear at 1024 and invisible at 4096. 0 draws no outline, which "
                        "leaves the fill as the only mark and turns the region into a solid "
                        "block."
                    ),
                ),
                io.Float.Input(
                    "fill_opacity",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How solidly the inside of each rectangle is filled, in the same "
                        "colour. 0.0 leaves it open, which is what an inspection overlay "
                        "wants; around 0.25 tints the region while leaving the picture "
                        "readable underneath."
                    ),
                ),
                io.String.Input(
                    "label",
                    default="",
                    multiline=False,
                    tooltip=(
                        "Caption drawn inside the top left corner of each rectangle. Empty "
                        "draws none. {index} becomes the row number, {width} and {height} "
                        "the size of that region in pixels, 'region {index}: {width}x"
                        "{height}' labels a set of crops with what each one will produce."
                    ),
                ),
                io.Int.Input(
                    "label_size",
                    default=20,
                    min=1,
                    max=512,
                    step=1,
                    tooltip="Height of the label text in points. Read only when a label is set.",
                ),
                io.Float.Input(
                    "opacity",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How much of the whole overlay shows. Applied to the outlines, fills "
                        "and labels together, so nothing drifts out of step when it is faded."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip="The images with the bounds drawn over them.",
                ),
                io.Mask.Output(
                    tooltip=(
                        "The rectangles as a mask, white wherever the overlay was drawn. "
                        "With thickness 0 and fill_opacity above 0 this is a solid region "
                        "mask for the bounds."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        image_bounds,
        color="#00FF00",
        thickness=3,
        fill_opacity=0.0,
        label="",
        label_size=20,
        opacity=1.0,
    ) -> io.NodeOutput:
        from PIL import ImageOps

        from ....modules.image import scratch

        rows = bounds.rows(image_bounds)
        if not rows:
            raise ValueError(
                "Draw Image Bounds was given a bounds value holding no regions. Check the "
                "node feeding image_bounds: a region search that matched nothing produces "
                "this."
            )

        outline = draw.parse_color(color)
        fill = (outline[0], outline[1], outline[2], int(round(255 * max(0.0, fill_opacity))))
        font = draw.load_font(label_size) if label else None

        planes = image_planes(image)
        grouped = [rows] if len(planes) == 1 else [[rows[index % len(rows)]] for index in range(len(planes))]

        drawn = masks = None
        for index, (plane, group) in enumerate(zip(planes, grouped)):
            picture = tensor2pil(plane).convert("RGB")
            boxes = [(left, top, right, bottom) for top, bottom, left, right in group]
            labels = [cls.caption(label, index, box) for index, box in enumerate(boxes)] if label else None
            layer = draw.draw_boxes_layer(
                picture.size,
                boxes,
                outline,
                thickness=thickness,
                fill=fill,
                labels=labels,
                font=font,
                label_color=outline,
            )
            frame = pil2tensor(draw.composite(picture, layer, opacity))
            # The coverage channel is inverted so the mask is white where the overlay is.
            coverage = pil2mask(ImageOps.invert(draw.layer_mask(layer)))
            if drawn is None:
                advice = "Passing fewer or smaller frames also fits it."
                drawn = scratch.allocate(
                    (len(planes),) + tuple(frame.shape[1:]), node="Draw Image Bounds", advice=advice
                )
                masks = scratch.allocate(
                    (len(planes),) + tuple(coverage.shape), node="Draw Image Bounds", advice=advice
                )
            drawn[index] = frame[0]
            masks[index] = coverage

        if drawn is None:
            raise ValueError(
                "Draw Image Bounds was handed an empty image batch. Wire in at least one image."
            )
        return io.NodeOutput(drawn, masks)

    @staticmethod
    def caption(template: str, index: int, box: tuple[int, int, int, int]) -> str:
        """Fill a label template for one rectangle.

        Args:
            template: The label as typed, which may hold ``{index}``, ``{width}`` and
                ``{height}``.
            index: Row number, counted from 0.
            box: ``(left, top, right, bottom)`` of the rectangle.

        Returns:
            The finished label. A template naming a field that does not exist is returned
            as written rather than raising.
        """
        left, top, right, bottom = box
        try:
            return template.format(
                index=index, width=abs(right - left) + 1, height=abs(bottom - top) + 1
            )
        except (KeyError, IndexError, ValueError):
            return template

"""Pick a width and a height from an aspect ratio and one measurement."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.image import resolution


class ResolutionSelector(io.ComfyNode):
    """Turn an aspect ratio and one measurement into a width and a height."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASResolutionSelector",
            display_name="Resolution Selector (Advanced)",
            search_aliases=[
                "WASResolutionSelector",
                "Resolution Selector",
                "aspect ratio",
                "resolution",
                "image size",
                "width height",
            ],
            category="WAS Suite/Number",
            description=(
                "Pick a shape and one measurement, and get the width and height that match. "
                "Size by an edge in pixels when you know the resolution you want, such as "
                "1024 across, and by megapixels when you are working to a budget instead. "
                "Both sides land on a whole step, so the pair is one a model will take."
            ),
            inputs=[
                io.Combo.Input(
                    "aspect_ratio",
                    options=list(resolution.RATIOS),
                    default="1:1",
                    tooltip=(
                        "The shape, widest side first. 16:9 is widescreen, 3:2 is a stills "
                        "camera, 1:1 is square. Orientation below decides which way round it "
                        "is applied, so each shape is listed once."
                    ),
                ),
                io.Combo.Input(
                    "orientation",
                    options=list(resolution.ORIENTATIONS),
                    default="landscape",
                    tooltip=(
                        "Which way round the shape goes. `portrait` swaps the two sides, and "
                        "`square` ignores the ratio entirely."
                    ),
                ),
                io.Combo.Input(
                    "size_by",
                    options=list(resolution.DRIVERS),
                    default="long edge",
                    tooltip=(
                        "Which measurement you are giving. `long edge` and `short edge` fix "
                        "whichever side is longer or shorter; `width` and `height` fix that "
                        "side whatever the orientation; `megapixels` fixes the area instead "
                        "and reads the megapixels widget rather than size."
                    ),
                ),
                io.Int.Input(
                    "size",
                    default=1024,
                    min=1,
                    max=resolution.MAX_EDGE,
                    step=8,
                    tooltip=(
                        "The measurement in pixels, read by every option but `megapixels`. "
                        "1024 on the long edge of 16:9 is 1024 by 576."
                    ),
                ),
                io.Float.Input(
                    "megapixels",
                    default=1.0,
                    min=0.01,
                    max=64.0,
                    step=0.01,
                    optional=True,
                    tooltip=(
                        "The area in millions of pixels, read only when size_by is "
                        "`megapixels`. 1.0 at 16:9 is about 1344 by 768."
                    ),
                ),
                io.Int.Input(
                    "multiple_of",
                    default=64,
                    min=1,
                    max=256,
                    tooltip=(
                        "Step both sides land on. 8 = the least a latent takes; 64 = what most "
                        "model families were trained on. Snapping moves a side by up to half "
                        "a step, so the ratio output may differ slightly from the one asked "
                        "for."
                    ),
                ),
            ],
            outputs=[
                io.Int.Output(
                    display_name="width",
                    tooltip="Width in pixels, a whole number of steps.",
                ),
                io.Int.Output(
                    display_name="height",
                    tooltip="Height in pixels, a whole number of steps.",
                ),
                io.Float.Output(
                    display_name="ratio",
                    tooltip=(
                        "Width divided by height as it came out, after the step was applied. "
                        "Compare it with the shape you asked for to see what the step cost."
                    ),
                ),
                io.Float.Output(
                    display_name="megapixels",
                    tooltip=(
                        "Area of the pair in millions of pixels, whichever way it was sized. "
                        "Read it to keep two different shapes to the same cost."
                    ),
                ),
                io.String.Output(
                    display_name="label",
                    tooltip=(
                        "The pair written out, as '1024 x 576, 16:9, 0.59 MP'. Feed it to a "
                        "filename prefix or a caption so a render records its own size."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls, aspect_ratio, orientation="landscape", size_by="long edge", size=1024,
        megapixels=1.0, multiple_of=64,
    ) -> io.NodeOutput:
        """Work out the pair and describe it.

        Args:
            aspect_ratio: The shape, as an entry from the widget's list.
            orientation: Which way round the shape goes.
            size_by: Which measurement is being given.
            size: The measurement in pixels.
            megapixels: The measurement in millions of pixels.
            multiple_of: Step both sides land on.

        Returns:
            The width and height, the ratio and area they came to, and a label.
        """
        width, height = resolution.resolve(
            aspect_ratio, orientation, size_by, size, megapixels, multiple_of
        )
        area = width * height / 1_000_000.0
        label = f"{width} x {height}, {aspect_ratio}, {area:.2f} MP"
        return io.NodeOutput(width, height, width / height, area, label)

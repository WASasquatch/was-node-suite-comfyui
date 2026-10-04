"""Draw text onto an image."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.image import dynamic
from ....modules.compat import limits
from ....modules.convert.tensors import image_planes, pil2mask, pil2tensor, tensor2pil
from ....modules.data import paths
from ....modules.image import draw
from ....modules.interface import preview


class ImageDrawText(io.ComfyNode):
    """Draw text over every image in a batch, and return its coverage as a mask."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageDrawText",
            display_name="Image Draw Text",
            search_aliases=[
                "WASImageDrawText", "Image Draw Text",
                "caption",
                "watermark",
                "label",
                "title",
                "annotate",
                "text on image",
            ],
            category="WAS Suite/Image/Process",
            description=(
                (
                    (
                        "Draw text over an image, with wrapping, alignment, an outline and a "
                        "background panel. Returns the picture and the text as a mask. The "
                        "font list is built from disk: drop .ttf, .otf or .ttc files into the "
                        "fonts folder beside config.yaml, at "
                        "ComfyUI/user/was-node-suite/fonts, and they appear in the menu named "
                        "after the file. Put them there rather than inside the pack, which an "
                        "update overwrites. The DejaVu faces cover the most ground and are the "
                        "safe choice for text not known in advance; the Liberation faces match "
                        "Arial, Times New Roman and Courier New width for width, so a caption "
                        "composed against one of those breaks its lines in the same places. A "
                        "font that cannot be opened falls back to the picked one, then to a "
                        "small built-in bitmap face that ignores font_size."
                    )
                )
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The images to draw on. Every image in the batch gets the same text, "
                        "so a caption applies across a whole sequence in one node."
                    ),
                ),
                io.String.Input(
                    "text",
                    multiline=True,
                    tooltip=(
                        "The text to draw. Line breaks are kept. Tokens such as `[time]` and "
                        "`[hostname]` resolve before drawing, which is how a frame carries the "
                        "date it was rendered."
                    ),
                ),
                io.Int.Input(
                    "font_size",
                    default=32,
                    min=1,
                    max=1024,
                    step=1,
                    tooltip=(
                        "Height of the text in points. A caption on a 1024-pixel image reads "
                        "at around 24 to 40; a title wants considerably more."
                    ),
                ),
                io.String.Input(
                    "text_color",
                    default="#FFFFFF",
                    multiline=False,
                    tooltip=(
                        "Colour of the glyphs, as #RRGGBB, #RRGGBBAA or a name such as "
                        "'white'. The eight-digit form carries its own transparency, which "
                        "is how a watermark is made faint without fading the outline with it."
                    ),
                ),
                io.Combo.Input(
                    "position",
                    options=list(draw.ANCHORS),
                    tooltip=(
                        "Where the block of text sits on the image. The corners and edges "
                        "hold back by the margin below; 'middle center' ignores it."
                    ),
                ),
                io.Combo.Input(
                    "align",
                    options=["left", "center", "right"],
                    tooltip=(
                        "How the lines line up with each other inside the block. Separate "
                        "from position, which places the block: a block anchored bottom "
                        "right can still be left-aligned inside itself."
                    ),
                ),
                io.Int.Input(
                    "offset_x",
                    default=0,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Pixels to move the text right of where position put it. Negative "
                        "moves it left. For nudging a block off an anchor rather than "
                        "placing it from scratch."
                    ),
                ),
                io.Int.Input(
                    "offset_y",
                    default=0,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip="Pixels to move the text down. Negative moves it up.",
                ),
                io.Int.Input(
                    "margin",
                    default=16,
                    min=0,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Space held back from every edge, so an edge-anchored caption does "
                        "not touch the border. Ignored on whichever axis the position "
                        "centres on."
                    ),
                ),
                io.Float.Input(
                    "line_spacing",
                    default=1.0,
                    min=0.1,
                    max=5.0,
                    step=0.05,
                    tooltip=(
                        "Multiplier on the font's own line height. 1.0 is single spaced; "
                        "1.2 to 1.5 is easier to read for a paragraph of several lines."
                    ),
                ),
                io.Int.Input(
                    "wrap_width",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Width in pixels to wrap the text at. 0 turns wrapping off and only "
                        "the line breaks already in the text are honoured. A word wider than "
                        "this on its own is left whole and overhangs rather than being split."
                    ),
                ),
                io.Int.Input(
                    "stroke_width",
                    default=0,
                    min=0,
                    max=64,
                    step=1,
                    tooltip=(
                        "Width of an outline drawn around every glyph. 1 or 2 is what keeps "
                        "a caption legible over a picture whose brightness changes underneath "
                        "it. 0 draws no outline."
                    ),
                ),
                io.String.Input(
                    "stroke_color",
                    default="#000000",
                    multiline=False,
                    tooltip=(
                        "Colour of the outline. Read only when stroke_width is 1 or more."
                    ),
                ),
                io.String.Input(
                    "background_color",
                    default="",
                    multiline=False,
                    tooltip=(
                        "Colour of a panel drawn behind the text. Empty draws no panel. A "
                        "half-transparent #000000B4 is the usual subtitle treatment, and is "
                        "more readable than an outline over a busy picture."
                    ),
                ),
                io.Int.Input(
                    "background_padding",
                    default=8,
                    min=0,
                    max=512,
                    step=1,
                    tooltip=(
                        "How far the panel extends past the text on every side. Read only "
                        "when background_color is set."
                    ),
                ),
                io.Float.Input(
                    "opacity",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How much of the finished text shows, 0.0 to 1.0. Applied to the "
                        "text, its outline and its panel together, so a faint watermark "
                        "stays consistent instead of the outline surviving the fade."
                    ),
                ),
                io.Combo.Input(
                    "font",
                    options=list(paths.font_names()),
                    optional=True,
                    tooltip=(
                        "Which typeface to draw with. The DejaVu faces cover the most ground; "
                        "`Mono` gives every character the same width, which stops a frame "
                        "counter jittering."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip="The images with the text drawn on them.",
                ),
                io.Mask.Output(
                    tooltip=(
                        "The text as a mask, white where a glyph, outline or panel was "
                        "drawn. Feeds an inpaint region or a blend factor without drawing "
                        "the text twice."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        text="",
        font_size=32,
        text_color="#FFFFFF",
        position="bottom center",
        align="center",
        offset_x=0,
        offset_y=0,
        margin=16,
        line_spacing=1.0,
        wrap_width=0,
        stroke_width=0,
        stroke_color="#000000",
        background_color="",
        background_padding=8,
        opacity=1.0,
        font=draw.DEFAULT_FONT,
    ) -> io.NodeOutput:
        from PIL import ImageOps

        from ....modules.image import scratch

        folded = dynamic.fold(image)
        image = folded.images

        # The picture the overlay lays the text over is published before anything is drawn.
        preview.publish(image)

        resolved_font = None
        typeface = draw.load_font(font_size, resolved_font, font)
        fill = draw.parse_color(text_color)
        stroke = draw.parse_color(stroke_color)
        panel = draw.parse_color(background_color, draw.TRANSPARENT)

        planes = image_planes(image)
        drawn = masks = None
        for index, plane in enumerate(planes):
            picture = tensor2pil(plane).convert("RGB")
            layer = draw.draw_text_layer(
                picture.size,
                text,
                typeface,
                fill,
                position=position,
                align=align,
                offset=(offset_x, offset_y),
                margin=margin,
                line_spacing=line_spacing,
                wrap_width=wrap_width,
                stroke_width=stroke_width,
                stroke_color=stroke,
                background=panel,
                background_padding=background_padding,
            )
            frame = pil2tensor(draw.composite(picture, layer, opacity))
            # The coverage channel is inverted so the mask is white where the text is.
            coverage = pil2mask(ImageOps.invert(draw.layer_mask(layer)))
            if drawn is None:
                advice = "Passing fewer or smaller frames also fits it."
                drawn = scratch.allocate(
                    (len(planes),) + tuple(frame.shape[1:]), node="Image Draw Text", advice=advice
                )
                masks = scratch.allocate(
                    (len(planes),) + tuple(coverage.shape), node="Image Draw Text", advice=advice
                )
            drawn[index] = dynamic.unfold(frame[0], folded)
            masks[index] = coverage

        if drawn is None:
            raise ValueError(
                "Image Draw Text was handed an empty image batch. Wire in at least one image."
            )
        return io.NodeOutput(drawn, masks)

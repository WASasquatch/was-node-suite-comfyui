"""Light a slope built from one layer's edge, so it reads as raised or pressed in."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.image import layer_fx
from ....modules.log import get_logger

logger = get_logger("nodes.image.layerfx")

#: What the node calls itself in a message.
NODE_NAME = "Layer Bevel"


class LayerBevel(io.ComfyNode):
    """Bake a lit bevel or emboss into one layer of a stack."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLayerBevel",
            display_name="Layer Bevel",
            search_aliases=[
                "WASLayerBevel",
                "Layer Bevel",
                "bevel and emboss",
                "emboss",
                "layer effect",
                "layer style",
                "compositor",
            ],
            category="WAS Suite/Image/Layers",
            description=(
                "Build a slope from the edge of one layer in a stack and light it, so the "
                "layer reads as raised off the plate or pressed into it. Gives a title, a "
                "badge or a cut-out subject a physical edge, and the layer keeps its place "
                "on the canvas."
            ),
            inputs=[
                io.Layers.Input(
                    "layers",
                    tooltip=(
                        "The stack holding the layer to bevel. Wire in Add Layer, Layers "
                        "From Bounding Boxes, or another layer effect to stack effects up."
                    ),
                ),
                io.Int.Input(
                    "layer_index",
                    default=-1,
                    min=-1000,
                    max=1000,
                    step=1,
                    tooltip=(
                        "Which layer to bevel when layer_name is empty. -1 = the top of the "
                        "stack, -2 = the one below it, 0 = the bottom, 1 = the next one up."
                    ),
                ),
                io.String.Input(
                    "layer_name",
                    default="",
                    tooltip=(
                        "Name of the layer to bevel, matched ignoring case and spare space. "
                        "Empty reads layer_index instead. 'subject' picks the layer Add "
                        "Layer was given that name."
                    ),
                ),
                io.Combo.Input(
                    "style",
                    options=list(layer_fx.BEVEL_STYLES),
                    default=layer_fx.BEVEL_STYLES[0],
                    tooltip=(
                        "Where the slope sits. `inner` runs inside the edge and grows "
                        "nothing, `outer` runs outside it and grows the layer, `emboss` "
                        "straddles the edge and grows the layer."
                    ),
                ),
                io.Float.Input(
                    "depth",
                    default=1.0,
                    min=0.0,
                    max=10.0,
                    step=0.05,
                    tooltip=(
                        "How steep the slope reads. 0.5 = a soft swell, 1.0 = a clear edge, "
                        "4.0 = a hard metallic lip, 0.0 = flat."
                    ),
                ),
                io.Combo.Input(
                    "direction",
                    options=list(layer_fx.BEVEL_DIRECTIONS),
                    default=layer_fx.BEVEL_DIRECTIONS[0],
                    tooltip=(
                        "Which way the slope faces. `up` reads as raised off the plate, "
                        "`down` swaps the lit and unlit sides and reads as stamped into it."
                    ),
                ),
                io.Int.Input(
                    "size",
                    default=8,
                    min=0,
                    max=128,
                    step=1,
                    tooltip=(
                        "How far the slope runs from the edge, in pixels. 2 = a crisp lip, "
                        "8 = a readable bevel, 40 = a broad dome."
                    ),
                ),
                io.Int.Input(
                    "soften",
                    default=2,
                    min=0,
                    max=64,
                    step=1,
                    tooltip=(
                        "Extra blur on the slope, in pixels. 0 = every corner sharp, 2 = "
                        "smooth, 16 = the shape's detail is lost into a swell."
                    ),
                ),
                io.Float.Input(
                    "angle",
                    default=135.0,
                    min=-360.0,
                    max=360.0,
                    step=1.0,
                    tooltip=(
                        "Degrees the light comes from, counted counter-clockwise from "
                        "pointing right. 135 = from above and to the left, 90 = straight "
                        "down from the top, 0 = from the right."
                    ),
                ),
                io.Float.Input(
                    "altitude",
                    default=30.0,
                    min=0.0,
                    max=90.0,
                    step=1.0,
                    tooltip=(
                        "Degrees the light sits above the surface. 0 = grazing, which is "
                        "the strongest, 30 = a normal key light, 90 = straight on, which "
                        "flattens the bevel away."
                    ),
                ),
                io.String.Input(
                    "highlight_color",
                    default="#ffffff",
                    tooltip=(
                        "Colour of the lit side, as hexadecimal digits. #ffffff = white, "
                        "#ffe0a0 = warm metal. Three digits such as #fff work too."
                    ),
                ),
                io.Float.Input(
                    "highlight_opacity",
                    default=0.75,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How strongly the lit side is laid down. 0.75 = a clear highlight, "
                        "1.0 = a hard specular, 0.0 = none."
                    ),
                ),
                io.Combo.Input(
                    "highlight_blend_mode",
                    options=list(layer_fx.BLEND_MODES),
                    default="screen",
                    tooltip=(
                        "How the lit side mixes with the layer's own pixels. `screen` "
                        "lightens without flattening, `normal` paints the colour flat, "
                        "`linear-dodge` blows it out."
                    ),
                ),
                io.String.Input(
                    "shadow_color",
                    default="#000000",
                    tooltip=(
                        "Colour of the unlit side, as hexadecimal digits. #000000 = black, "
                        "#201040 = a cool dark violet. Three digits such as #124 work too."
                    ),
                ),
                io.Float.Input(
                    "shadow_opacity",
                    default=0.6,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How strongly the unlit side is laid down. 0.6 = a natural "
                        "shadowed edge, 1.0 = a hard black lip, 0.0 = none."
                    ),
                ),
                io.Combo.Input(
                    "shadow_blend_mode",
                    options=list(layer_fx.BLEND_MODES),
                    default="multiply",
                    tooltip=(
                        "How the unlit side mixes with the layer's own pixels. `multiply` "
                        "darkens, `normal` paints the colour flat, `color-burn` deepens the "
                        "edge hard."
                    ),
                ),
            ],
            outputs=[
                io.Layers.Output(
                    display_name="layers",
                    tooltip=(
                        "The stack with the bevel baked into that layer's picture and "
                        "transparency. Wire it into Create Layered Image or the next effect."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, layers, layer_index=-1, layer_name="",
                style=layer_fx.BEVEL_STYLES[0], depth=1.0,
                direction=layer_fx.BEVEL_DIRECTIONS[0], size=8, soften=2, angle=135.0,
                altitude=30.0, highlight_color="#ffffff", highlight_opacity=0.75,
                highlight_blend_mode="screen", shadow_color="#000000",
                shadow_opacity=0.6, shadow_blend_mode="multiply") -> io.NodeOutput:
        found = layer_fx.stack(layers)
        place, entry = layer_fx.chosen(found, layer_index, layer_name, NODE_NAME)
        warm = layer_fx.colour(highlight_color, "highlight_color")
        cool = layer_fx.colour(shadow_color, "shadow_color")
        work, keep = layer_fx.bevel_margin(style, size, soften)

        def render(colours, alpha):
            return layer_fx.bevel(colours, alpha, style, depth, direction, size, soften,
                                  angle, altitude, warm, highlight_opacity,
                                  highlight_blend_mode, cool, shadow_opacity,
                                  shadow_blend_mode)

        document, grown = layer_fx.applied(layers, entry, render, work, keep)
        layer_fx.report(NODE_NAME, document, grown, place, len(found), keep)
        logger.info(
            "%s lit a %dpx %s bevel on layer %d of %d", NODE_NAME, size, style, place,
            len(found),
        )
        return io.NodeOutput(document)

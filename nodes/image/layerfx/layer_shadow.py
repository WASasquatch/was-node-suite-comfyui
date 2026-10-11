"""Cast a shadow behind one layer of a stack or inside its edge."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.image import layer_fx
from ....modules.log import get_logger

logger = get_logger("nodes.image.layerfx")

#: What the node calls itself in a message.
NODE_NAME = "Layer Shadow"


class LayerShadow(io.ComfyNode):
    """Bake a drop or inner shadow into one layer of a stack."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLayerShadow",
            display_name="Layer Shadow",
            search_aliases=[
                "WASLayerShadow",
                "Layer Shadow",
                "drop shadow",
                "inner shadow",
                "layer effect",
                "layer style",
                "compositor",
            ],
            category="WAS Suite/Image/Layers",
            description=(
                "Cast a shadow off one layer in a stack. A drop shadow sits behind the "
                "layer and lifts a cut-out subject off the plate below it; an inner shadow "
                "sits inside the edge and sinks the layer into it. The layer keeps its place "
                "on the canvas."
            ),
            inputs=[
                io.Layers.Input(
                    "layers",
                    tooltip=(
                        "The stack holding the layer to shade. Wire in Add Layer, Layers "
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
                        "Which layer to shade when layer_name is empty. -1 = the top of the "
                        "stack, -2 = the one below it, 0 = the bottom, 1 = the next one up."
                    ),
                ),
                io.String.Input(
                    "layer_name",
                    default="",
                    tooltip=(
                        "Name of the layer to shade, matched ignoring case and spare space. "
                        "Empty reads layer_index instead. 'subject' picks the layer Add "
                        "Layer was given that name."
                    ),
                ),
                io.Combo.Input(
                    "shadow_type",
                    options=list(layer_fx.SHADOW_MODES),
                    default=layer_fx.SHADOW_MODES[0],
                    tooltip=(
                        "Which side of the edge the shadow falls on. `drop` sits behind the "
                        "layer and grows it, `inner` sits inside the layer's own coverage "
                        "on the side the light does not reach and grows nothing."
                    ),
                ),
                io.Float.Input(
                    "angle",
                    default=315.0,
                    min=-360.0,
                    max=360.0,
                    step=1.0,
                    tooltip=(
                        "Degrees the shadow is thrown at, counted counter-clockwise from "
                        "pointing right. 0 = right, 90 = up, 180 = left, 315 = down and to "
                        "the right. An inner shadow lands on the opposite side."
                    ),
                ),
                io.Int.Input(
                    "distance",
                    default=12,
                    min=0,
                    max=256,
                    step=1,
                    tooltip=(
                        "Pixels the shadow moves from the layer. 0 = directly behind it, 12 "
                        "= a small lift, 60 = thrown well clear."
                    ),
                ),
                io.Float.Input(
                    "spread",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Share of size spent hardening the shadow's edge rather than "
                        "blurring it. 0.0 = a soft falloff, 0.5 = a solid core with a soft "
                        "rim, 1.0 = a hard silhouette."
                    ),
                ),
                io.Int.Input(
                    "size",
                    default=12,
                    min=0,
                    max=128,
                    step=1,
                    tooltip=(
                        "Blur radius in pixels. 0 = a hard-edged copy, 12 = a soft contact "
                        "shadow, 64 = a broad haze."
                    ),
                ),
                io.String.Input(
                    "color",
                    default="#000000",
                    tooltip=(
                        "Colour of the shadow, as hexadecimal digits. #000000 = black, "
                        "#1a0033 = a cool dark violet. Three digits such as #103 work too."
                    ),
                ),
                io.Float.Input(
                    "opacity",
                    default=0.6,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How strongly the shadow is laid down. 1.0 = solid, 0.6 = a natural "
                        "cast, 0.0 = nothing."
                    ),
                ),
                io.Combo.Input(
                    "blend_mode",
                    options=list(layer_fx.BLEND_MODES),
                    default=layer_fx.BLEND_MODES[0],
                    tooltip=(
                        "How the shadow mixes with the layer's own pixels where the two "
                        "overlap. `normal` lays the colour down as it is, `multiply` "
                        "darkens, `screen` lightens."
                    ),
                ),
            ],
            outputs=[
                io.Layers.Output(
                    display_name="layers",
                    tooltip=(
                        "The stack with the shadow baked into that layer's picture and "
                        "transparency. Wire it into Create Layered Image or the next effect."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, layers, layer_index=-1, layer_name="",
                shadow_type=layer_fx.SHADOW_MODES[0], angle=315.0, distance=12, spread=0.0,
                size=12, color="#000000", opacity=0.6,
                blend_mode=layer_fx.BLEND_MODES[0]) -> io.NodeOutput:
        found = layer_fx.stack(layers)
        place, entry = layer_fx.chosen(found, layer_index, layer_name, NODE_NAME)
        tint = layer_fx.colour(color, "color")
        work, keep = layer_fx.shadow_margin(shadow_type, angle, distance, size)

        def render(colours, alpha):
            return layer_fx.shadow(colours, alpha, shadow_type, angle, distance, spread, size,
                                   tint, opacity, blend_mode)

        document, grown = layer_fx.applied(layers, entry, render, work, keep)
        layer_fx.report(NODE_NAME, document, grown, place, len(found), keep)
        logger.info(
            "%s cast a %s shadow %dpx at %.0f degrees on layer %d of %d", NODE_NAME, shadow_type,
            distance, angle, place, len(found),
        )
        return io.NodeOutput(document)

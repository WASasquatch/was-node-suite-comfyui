"""Spread light out from one layer of a stack or in from its edge."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.image import layer_fx
from ....modules.log import get_logger

logger = get_logger("nodes.image.layerfx")

#: What the node calls itself in a message.
NODE_NAME = "Layer Glow"


class LayerGlow(io.ComfyNode):
    """Bake an outer or inner glow into one layer of a stack."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLayerGlow",
            display_name="Layer Glow",
            search_aliases=[
                "WASLayerGlow",
                "Layer Glow",
                "outer glow",
                "inner glow",
                "halo",
                "layer effect",
                "layer style",
                "compositor",
            ],
            category="WAS Suite/Image/Layers",
            description=(
                "Spread light off one layer in a stack. An outer glow sits behind the "
                "layer as a halo that separates a title or cut-out subject from a busy "
                "plate; an inner glow burns in from the edge as a rim light. The layer keeps "
                "its place on the canvas."
            ),
            inputs=[
                io.Layers.Input(
                    "layers",
                    tooltip=(
                        "The stack holding the layer to light. Wire in Add Layer, Layers "
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
                        "Which layer to light when layer_name is empty. -1 = the top of the "
                        "stack, -2 = the one below it, 0 = the bottom, 1 = the next one up."
                    ),
                ),
                io.String.Input(
                    "layer_name",
                    default="",
                    tooltip=(
                        "Name of the layer to light, matched ignoring case and spare space. "
                        "Empty reads layer_index instead. 'subject' picks the layer Add "
                        "Layer was given that name."
                    ),
                ),
                io.Combo.Input(
                    "glow_type",
                    options=list(layer_fx.GLOW_MODES),
                    default=layer_fx.GLOW_MODES[0],
                    tooltip=(
                        "Which side of the edge the light spreads over. `outer` sits behind "
                        "the layer as a halo and grows it, `inner` burns in from the edge "
                        "and grows nothing."
                    ),
                ),
                io.Int.Input(
                    "size",
                    default=16,
                    min=0,
                    max=128,
                    step=1,
                    tooltip=(
                        "How far the light reaches, in pixels. 4 = a tight rim, 16 = a "
                        "readable halo, 64 = a broad bloom, 0 = nothing drawn."
                    ),
                ),
                io.Float.Input(
                    "spread",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Share of size spent hardening the glow rather than blurring it. "
                        "0.0 = a smooth falloff, 0.5 = a bright core with a soft rim, 1.0 = "
                        "a hard band."
                    ),
                ),
                io.String.Input(
                    "color",
                    default="#ffcc66",
                    tooltip=(
                        "Colour of the light, as hexadecimal digits. #ffffff = white, "
                        "#ffcc66 = warm, #66ccff = cool. Three digits such as #fc6 work too."
                    ),
                ),
                io.Float.Input(
                    "opacity",
                    default=0.75,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How strongly the light is laid down. 1.0 = solid, 0.75 = a clear "
                        "halo, 0.0 = nothing."
                    ),
                ),
                io.Combo.Input(
                    "blend_mode",
                    options=list(layer_fx.BLEND_MODES),
                    default=layer_fx.BLEND_MODES[0],
                    tooltip=(
                        "How the light mixes with the layer's own pixels where the two "
                        "overlap. `normal` lays the colour down as it is, `screen` lightens "
                        "without flattening, `linear-dodge` blows the edge out."
                    ),
                ),
            ],
            outputs=[
                io.Layers.Output(
                    display_name="layers",
                    tooltip=(
                        "The stack with the glow baked into that layer's picture and "
                        "transparency. Wire it into Create Layered Image or the next effect."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, layers, layer_index=-1, layer_name="", glow_type=layer_fx.GLOW_MODES[0],
                size=16, spread=0.0, color="#ffcc66", opacity=0.75,
                blend_mode=layer_fx.BLEND_MODES[0]) -> io.NodeOutput:
        found = layer_fx.stack(layers)
        place, entry = layer_fx.chosen(found, layer_index, layer_name, NODE_NAME)
        tint = layer_fx.colour(color, "color")
        work, keep = layer_fx.glow_margin(glow_type, size)

        def render(colours, alpha):
            return layer_fx.glow(colours, alpha, glow_type, size, spread, tint, opacity,
                                 blend_mode)

        document, grown = layer_fx.applied(layers, entry, render, work, keep)
        layer_fx.report(NODE_NAME, document, grown, place, len(found), keep)
        logger.info(
            "%s spread a %dpx %s glow on layer %d of %d", NODE_NAME, size, glow_type, place,
            len(found),
        )
        return io.NodeOutput(document)

"""A flat, unlit surface for a Three.js mesh."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_MATERIAL, THREE_TEXTURE
from ...modules.threejs.spec import compact_deps, create_spec

REQUIRES = "threejs"


class ThreeBasicMaterial(io.ComfyNode):
    """Build a basic material descriptor."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeBasicMaterial",
            display_name="Three Basic Material",
            search_aliases=[
                "WASThreeBasicMaterial",
                "Three Basic Material",
                "unlit",
                "flat",
                "material",
            ],
            category="WAS Suite/Three",
            description=(
                "A surface that ignores every light and draws its colour flat, so a sphere reads "
                "as a circle. Right for a background card, a UI panel, a wireframe overlay or a "
                "texture that must arrive unaltered, and it shows even in a scene with no light."
            ),
            inputs=[
                io.String.Input(
                    "color",
                    default="#ffffff",
                    multiline=False,
                    tooltip=(
                        "Flat colour as hexadecimal. `#ffffff` is white and leaves a texture's own "
                        "colours alone."
                    ),
                ),
                io.Float.Input(
                    "opacity",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How solid the surface is, once transparent is on. 1.0 is opaque, 0.35 "
                        "is a ghost."
                    ),
                ),
                io.Boolean.Input(
                    "transparent",
                    default=False,
                    tooltip="`true` honours opacity and an alpha map; `false` draws the surface fully solid.",
                ),
                io.Boolean.Input(
                    "wireframe",
                    default=False,
                    tooltip="`true` draws the triangle edges instead of filled faces; `false` draws solid faces.",
                ),
                io.Combo.Input(
                    "side",
                    options=["front", "back", "double"],
                    default="front",
                    tooltip=(
                        "Which faces are drawn. 'front' for closed shapes, 'double' for planes "
                        "seen from behind."
                    ),
                ),
                THREE_TEXTURE.Input(
                    "map",
                    optional=True,
                    tooltip="The picture to draw. Its colour replaces the colour swatch per pixel.",
                ),
                THREE_TEXTURE.Input(
                    "alpha_map",
                    optional=True,
                    tooltip=(
                        "Opacity per pixel, read as greyscale. Needs transparent on to have "
                        "any effect."
                    ),
                ),
            ],
            outputs=[
                THREE_MATERIAL.Output(
                    display_name="material",
                    tooltip="The surface, for the material socket on Three Mesh.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls, color, opacity, transparent, wireframe, side, map=None, alpha_map=None
    ) -> io.NodeOutput:
        """Describe the surface."""
        return io.NodeOutput(
            create_spec(
                "material",
                "MeshBasicMaterial",
                params={
                    "color": color,
                    "opacity": float(opacity),
                    "transparent": bool(transparent),
                    "wireframe": bool(wireframe),
                    "side": side,
                },
                deps=compact_deps(map=map, alphaMap=alpha_map),
            )
        )

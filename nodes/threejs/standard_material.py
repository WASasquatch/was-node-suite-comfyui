"""A physically based surface for a Three.js mesh."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_MATERIAL, THREE_TEXTURE
from ...modules.threejs.spec import compact_deps, create_spec

REQUIRES = "threejs"


class ThreeStandardMaterial(io.ComfyNode):
    """Build a standard material descriptor."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeStandardMaterial",
            display_name="Three Standard Material",
            search_aliases=[
                "WASThreeStandardMaterial",
                "Three Standard Material",
                "pbr",
                "material",
                "metalness",
            ],
            category="WAS Suite/Three",
            description=(
                "A physically based surface lit by the scene's lights, set by colour, roughness "
                "and metalness, with any wired map overriding its channel per pixel. Metalness 0 "
                "and roughness near 0.5 suit plastic, paint and cloth; for clearcoat, "
                "transmission or sheen use Three Physical Material."
            ),
            inputs=[
                io.String.Input(
                    "color",
                    default="#ffffff",
                    multiline=False,
                    tooltip=(
                        "Base colour as hexadecimal. #ffffff is white, #d8b24a gold, #4aa3d8 "
                        "sky blue."
                    ),
                ),
                io.Float.Input(
                    "roughness",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How scattered reflections are. 0.0 is a mirror, 0.25 polished metal, "
                        "1.0 chalk."
                    ),
                ),
                io.Float.Input(
                    "metalness",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How metallic the surface reads. 0.0 for plastic and cloth, 1.0 for "
                        "bare metal."
                    ),
                ),
                io.String.Input(
                    "emissive",
                    default="#000000",
                    multiline=False,
                    tooltip=(
                        "Colour the surface gives off on its own, as hexadecimal. #000000 "
                        "emits nothing."
                    ),
                ),
                io.Float.Input(
                    "emissive_intensity",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=0.01,
                    tooltip=(
                        "How strongly the emissive colour shows. 1.0 matches it, 5.0 blooms "
                        "under tone mapping."
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
                        "is glassy."
                    ),
                ),
                io.Boolean.Input(
                    "transparent",
                    default=False,
                    tooltip=(
                        "`true` honours opacity and an alpha map; `false` draws the surface fully solid."
                    ),
                ),
                io.Boolean.Input(
                    "wireframe",
                    default=False,
                    tooltip=(
                        "`true` draws the triangle edges instead of filled faces; `false` draws solid faces."
                    ),
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
                io.Float.Input(
                    "normal_scale",
                    default=1.0,
                    min=-10.0,
                    max=10.0,
                    step=0.01,
                    tooltip=(
                        "How strongly the normal map bends the surface. 1.0 is as authored, "
                        "0.0 flat, -1.0 inverted."
                    ),
                ),
                io.Float.Input(
                    "bump_scale",
                    default=1.0,
                    min=-10.0,
                    max=10.0,
                    step=0.01,
                    tooltip="How deep the bump map reads. 1.0 is as authored, 0.2 is a subtle grain.",
                ),
                io.Float.Input(
                    "displacement_scale",
                    default=0.1,
                    min=-100.0,
                    max=100.0,
                    step=0.01,
                    tooltip=(
                        "How far the displacement map moves vertices, in scene units. 0.1 is "
                        "gentle relief."
                    ),
                ),
                io.Float.Input(
                    "displacement_bias",
                    default=0.0,
                    min=-100.0,
                    max=100.0,
                    step=0.01,
                    tooltip=(
                        "Shifts the whole displacement. -0.05 with a scale of 0.1 centres the "
                        "movement on the original surface."
                    ),
                ),
                io.Float.Input(
                    "ao_intensity",
                    default=1.0,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    tooltip="How strongly the ambient occlusion map darkens. 1.0 is as authored, 0.0 off.",
                ),
                THREE_TEXTURE.Input(
                    "map",
                    optional=True,
                    tooltip="Albedo texture. Its colour replaces the colour swatch per pixel.",
                ),
                THREE_TEXTURE.Input(
                    "normal_map",
                    optional=True,
                    tooltip=(
                        "Tangent space normals, as the usual blue-violet image. Fakes surface "
                        "detail without geometry."
                    ),
                ),
                THREE_TEXTURE.Input(
                    "roughness_map",
                    optional=True,
                    tooltip=(
                        "Roughness per pixel, read from the green channel. Black is a mirror, "
                        "white is chalk."
                    ),
                ),
                THREE_TEXTURE.Input(
                    "metalness_map",
                    optional=True,
                    tooltip=(
                        "Metalness per pixel, read from the blue channel. Black is dielectric, "
                        "white is metal."
                    ),
                ),
                THREE_TEXTURE.Input(
                    "emissive_map",
                    optional=True,
                    tooltip=(
                        "Where the surface glows. Multiplied by the emissive colour and its "
                        "intensity."
                    ),
                ),
                THREE_TEXTURE.Input(
                    "alpha_map",
                    optional=True,
                    tooltip=(
                        "Opacity per pixel, read as greyscale. Needs transparent on to have "
                        "any effect."
                    ),
                ),
                THREE_TEXTURE.Input(
                    "bump_map",
                    optional=True,
                    tooltip=(
                        "Height as greyscale, faked in the shading alone. Cheaper than a normal "
                        "map and softer."
                    ),
                ),
                THREE_TEXTURE.Input(
                    "displacement_map",
                    optional=True,
                    tooltip=(
                        "Height as greyscale, moving real vertices. Needs a geometry with "
                        "segments to move, such as 64 by 64."
                    ),
                ),
                THREE_TEXTURE.Input(
                    "ao_map",
                    optional=True,
                    tooltip=(
                        "Baked shadow in creases, read from the red channel. Black is fully "
                        "occluded, white is open."
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
        cls,
        color,
        roughness,
        metalness,
        emissive,
        emissive_intensity,
        opacity,
        transparent,
        wireframe,
        side,
        normal_scale,
        bump_scale,
        displacement_scale,
        displacement_bias,
        ao_intensity,
        map=None,
        normal_map=None,
        roughness_map=None,
        metalness_map=None,
        emissive_map=None,
        alpha_map=None,
        bump_map=None,
        displacement_map=None,
        ao_map=None,
    ) -> io.NodeOutput:
        """Describe the surface."""
        return io.NodeOutput(
            create_spec(
                "material",
                "MeshStandardMaterial",
                params={
                    "color": color,
                    "roughness": float(roughness),
                    "metalness": float(metalness),
                    "emissive": emissive,
                    "emissiveIntensity": float(emissive_intensity),
                    "opacity": float(opacity),
                    "transparent": bool(transparent),
                    "wireframe": bool(wireframe),
                    "side": side,
                    "normalScale": float(normal_scale),
                    "bumpScale": float(bump_scale),
                    "displacementScale": float(displacement_scale),
                    "displacementBias": float(displacement_bias),
                    "aoMapIntensity": float(ao_intensity),
                },
                deps=compact_deps(
                    map=map,
                    normalMap=normal_map,
                    roughnessMap=roughness_map,
                    metalnessMap=metalness_map,
                    emissiveMap=emissive_map,
                    alphaMap=alpha_map,
                    bumpMap=bump_map,
                    displacementMap=displacement_map,
                    aoMap=ao_map,
                ),
            )
        )

"""A physically based surface with glass and coating controls."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_MATERIAL, THREE_TEXTURE
from ...modules.threejs.spec import compact_deps, create_spec

REQUIRES = "threejs"


class ThreePhysicalMaterial(io.ComfyNode):
    """Build a physical material descriptor."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreePhysicalMaterial",
            display_name="Three Physical Material",
            search_aliases=[
                "WASThreePhysicalMaterial",
                "Three Physical Material",
                "glass",
                "clearcoat",
                "transmission",
                "iridescence",
            ],
            category="WAS Suite/Three",
            description=(
                "Three Standard Material with the extra layers glass, car paint, velvet and soap "
                "bubbles need: transmission, clearcoat, sheen and iridescence. Emission counts as "
                "a light in Three Path Trace Render. With every extra at 0.0 it looks exactly "
                "like a standard material, at a higher cost."
            ),
            inputs=[
                io.String.Input(
                    "color",
                    default="#ffffff",
                    multiline=False,
                    tooltip="Base colour as hexadecimal. `#ffffff` is white, `#d8b24a` gold.",
                ),
                io.Float.Input(
                    "roughness",
                    default=0.25,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="How scattered reflections are. 0.0 is a mirror, 0.25 lacquer, 1.0 chalk.",
                ),
                io.Float.Input(
                    "metalness",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="How metallic the surface reads. 0.0 for glass and paint, 1.0 for bare metal.",
                ),
                io.Float.Input(
                    "clearcoat",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Strength of a glossy layer over the base. 0.0 is off, 1.0 is full lacquer.",
                ),
                io.Float.Input(
                    "clearcoat_roughness",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="How scattered the clearcoat is. 0.0 is glassy, 0.3 reads as worn lacquer.",
                ),
                io.Float.Input(
                    "transmission",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How much light passes through, by refraction. 0.0 is solid, 1.0 is "
                        "clear glass; set thickness and ior for a solid block, "
                        "attenuation_color for a tint."
                    ),
                ),
                io.Float.Input(
                    "thickness",
                    default=0.0,
                    min=0.0,
                    max=100000.0,
                    step=0.01,
                    tooltip="How deep the glass is, in scene units. 0.0 is a thin shell, 1.0 a solid block.",
                ),
                io.Float.Input(
                    "ior",
                    default=1.5,
                    min=1.0,
                    max=2.333,
                    step=0.001,
                    tooltip="Index of refraction. 1.5 is glass, 1.33 water, 2.42 diamond, 1.0 bends nothing.",
                ),
                io.Float.Input(
                    "dispersion",
                    default=0.0,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    tooltip="How far colours split through the glass. 0.0 is off, 1.0 gives a prism edge.",
                ),
                io.Float.Input(
                    "iridescence",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="Strength of an angle-shifting film. 0.0 is off, 1.0 is a soap bubble.",
                ),
                io.Float.Input(
                    "anisotropy",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="How far highlights stretch, as on brushed metal. 0.0 is round, 1.0 fully stretched.",
                ),
                io.Float.Input(
                    "opacity",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip="How solid the surface is, once transparent is on. 1.0 is opaque, 0.35 is glassy.",
                ),
                io.Boolean.Input(
                    "transparent",
                    default=False,
                    tooltip=(
                        "`true` honours opacity; `false` draws solid. Transmission works either "
                        "way and usually looks better with this off."
                    ),
                ),
                io.Combo.Input(
                    "side",
                    options=["front", "back", "double"],
                    default="front",
                    tooltip=(
                        "Which faces are drawn. 'front' for closed shapes, 'double' for glass "
                        "seen through both walls."
                    ),
                ),
                io.Float.Input(
                    "specular_intensity",
                    default=1.0,
                    min=0.0,
                    max=10.0,
                    step=0.01,
                    tooltip=(
                        "Strength of the non-metal highlight. 1.0 is normal, 0.0 kills the "
                        "sheen entirely."
                    ),
                ),
                io.String.Input(
                    "specular_color",
                    default="#ffffff",
                    multiline=False,
                    tooltip="Tint of that highlight, as hexadecimal. `#ffffff` leaves it uncoloured.",
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
                    "transmission_map",
                    optional=True,
                    tooltip=(
                        "How much light passes through, per pixel. Black stays solid, white is "
                        "fully clear."
                    ),
                ),
                THREE_TEXTURE.Input(
                    "specular_intensity_map",
                    optional=True,
                    tooltip=(
                        "Highlight strength per pixel, read from the alpha channel. Black is "
                        "matte, white is full sheen."
                    ),
                ),
                THREE_TEXTURE.Input(
                    "specular_color_map",
                    optional=True,
                    tooltip="Highlight tint per pixel, read as colour. Multiplied by specular_color.",
                ),
                THREE_TEXTURE.Input(
                    "clearcoat_map",
                    optional=True,
                    tooltip=(
                        "Clearcoat strength per pixel, read from the red channel. Black is bare, "
                        "white is fully lacquered."
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
                io.String.Input(
                    "emissive",
                    default="#000000",
                    multiline=False,
                    tooltip=(
                        "Light the surface gives off, as hexadecimal, on top of what falls on "
                        "it. `#000000` gives off none, `#ff5522` a hot ember."
                    ),
                ),
                io.Float.Input(
                    "emissive_intensity",
                    default=1.0,
                    min=0.0,
                    max=1000.0,
                    step=0.1,
                    tooltip=(
                        "How strongly the emissive colour reads. 1.0 matches it, 8.0 makes the "
                        "surface a light bright enough to lift what is around it."
                    ),
                ),
                io.Float.Input(
                    "sheen",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Strength of a soft rim of light at grazing angles, as on velvet and "
                        "brushed cloth. 0.0 is off, 1.0 is full."
                    ),
                ),
                io.Float.Input(
                    "sheen_roughness",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How scattered the sheen is. 1.0 is a broad cloth rim, 0.3 a tight "
                        "silky one."
                    ),
                ),
                io.String.Input(
                    "sheen_color",
                    default="#ffffff",
                    multiline=False,
                    tooltip=(
                        "Colour of the sheen, as hexadecimal. `#ffffff` keeps the base colour, "
                        "`#8899ff` gives cloth a cool rim."
                    ),
                ),
                io.String.Input(
                    "attenuation_color",
                    default="#ffffff",
                    multiline=False,
                    tooltip=(
                        "Colour left after light has travelled through the glass, as "
                        "hexadecimal. `#88ccaa` is bottle green. Needs transmission above 0.0."
                    ),
                ),
                io.Float.Input(
                    "attenuation_distance",
                    default=0.0,
                    min=0.0,
                    max=10000.0,
                    step=0.1,
                    tooltip=(
                        "How far light travels through the glass before it takes on the whole "
                        "attenuation colour. 0.0 absorbs nothing, 0.5 tints a thick pane deeply."
                    ),
                ),
                THREE_TEXTURE.Input(
                    "emissive_map",
                    optional=True,
                    tooltip=(
                        "Emitted light per pixel, multiplying the emissive colour. Black gives "
                        "off none."
                    ),
                ),
                THREE_TEXTURE.Input(
                    "alpha_map",
                    optional=True,
                    tooltip=(
                        "Opacity as greyscale, read from the green channel. Black is clear, "
                        "white is solid. Needs transparent on."
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
        clearcoat,
        clearcoat_roughness,
        transmission,
        thickness,
        ior,
        dispersion,
        iridescence,
        anisotropy,
        opacity,
        transparent,
        side,
        specular_intensity,
        specular_color,
        normal_scale,
        bump_scale,
        displacement_scale,
        ao_intensity,
        emissive,
        emissive_intensity,
        sheen,
        sheen_roughness,
        sheen_color,
        attenuation_color,
        attenuation_distance,
        map=None,
        normal_map=None,
        roughness_map=None,
        metalness_map=None,
        transmission_map=None,
        specular_intensity_map=None,
        specular_color_map=None,
        clearcoat_map=None,
        bump_map=None,
        displacement_map=None,
        ao_map=None,
        emissive_map=None,
        alpha_map=None,
    ) -> io.NodeOutput:
        """Describe the surface."""
        return io.NodeOutput(
            create_spec(
                "material",
                "MeshPhysicalMaterial",
                params={
                    "color": color,
                    "roughness": float(roughness),
                    "metalness": float(metalness),
                    "clearcoat": float(clearcoat),
                    "clearcoatRoughness": float(clearcoat_roughness),
                    "transmission": float(transmission),
                    "thickness": float(thickness),
                    "ior": float(ior),
                    "dispersion": float(dispersion),
                    "iridescence": float(iridescence),
                    "anisotropy": float(anisotropy),
                    "opacity": float(opacity),
                    "transparent": bool(transparent),
                    "side": side,
                    "specularIntensity": float(specular_intensity),
                    "specularColor": specular_color,
                    "normalScale": float(normal_scale),
                    "bumpScale": float(bump_scale),
                    "displacementScale": float(displacement_scale),
                    "aoMapIntensity": float(ao_intensity),
                    "emissive": emissive,
                    "emissiveIntensity": float(emissive_intensity),
                    "sheen": float(sheen),
                    "sheenRoughness": float(sheen_roughness),
                    "sheenColor": sheen_color,
                    "attenuationColor": attenuation_color,
                    "attenuationDistance": float(attenuation_distance),
                },
                deps=compact_deps(
                    map=map,
                    normalMap=normal_map,
                    roughnessMap=roughness_map,
                    metalnessMap=metalness_map,
                    transmissionMap=transmission_map,
                    specularIntensityMap=specular_intensity_map,
                    specularColorMap=specular_color_map,
                    clearcoatMap=clearcoat_map,
                    bumpMap=bump_map,
                    displacementMap=displacement_map,
                    aoMap=ao_map,
                    emissiveMap=emissive_map,
                    alphaMap=alpha_map,
                ),
            )
        )

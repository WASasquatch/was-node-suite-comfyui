"""An IMAGE carried into a Three.js material as a texture."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_TEXTURE
from ...modules.threejs.spec import create_spec
from ...modules.threejs.textures import COLOR_SPACES, WRAP_MODES, texture_url

REQUIRES = "threejs"


class ThreeTextureImage(io.ComfyNode):
    """Turn an image into a texture descriptor."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeTextureImage",
            display_name="Three Texture From Image",
            search_aliases=[
                "WASThreeTextureImage",
                "Three Texture From Image",
                "texture",
                "albedo",
                "uv",
            ],
            category="WAS Suite/Three",
            description=(
                "Carry the first frame of an image batch into any of a material's map sockets, "
                "tiled by repeat and offset. Tag colour maps 'srgb' and data maps such as normal "
                "or roughness 'linear-srgb', or they look wrong."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="The picture to use. A batch is read on its first frame alone.",
                ),
                io.Combo.Input(
                    "color_space",
                    options=list(COLOR_SPACES),
                    default="srgb",
                    tooltip=(
                        "'srgb' for a colour map such as albedo; 'linear-srgb' for normal, "
                        "roughness, metalness or alpha."
                    ),
                ),
                io.Combo.Input(
                    "wrap_s",
                    options=list(WRAP_MODES),
                    default="clamp",
                    tooltip=(
                        "What happens past the horizontal edge. 'clamp' stretches the edge "
                        "pixel, 'repeat' tiles."
                    ),
                ),
                io.Combo.Input(
                    "wrap_t",
                    options=list(WRAP_MODES),
                    default="clamp",
                    tooltip=(
                        "What happens past the vertical edge. 'clamp' stretches the edge pixel, "
                        "'repeat' tiles."
                    ),
                ),
                io.Float.Input(
                    "repeat_x",
                    default=1.0,
                    min=-1024.0,
                    max=1024.0,
                    step=0.01,
                    tooltip="How many times it tiles across. 1.0 fits once, 4.0 tiles four times.",
                ),
                io.Float.Input(
                    "repeat_y",
                    default=1.0,
                    min=-1024.0,
                    max=1024.0,
                    step=0.01,
                    tooltip="How many times it tiles down. 1.0 fits once, 4.0 tiles four times.",
                ),
                io.Float.Input(
                    "offset_x",
                    default=0.0,
                    min=-1024.0,
                    max=1024.0,
                    step=0.01,
                    tooltip="Slide across, in tiles. 0.5 moves it half a tile sideways.",
                ),
                io.Float.Input(
                    "offset_y",
                    default=0.0,
                    min=-1024.0,
                    max=1024.0,
                    step=0.01,
                    tooltip="Slide down, in tiles. 0.5 moves it half a tile vertically.",
                ),
                io.Boolean.Input(
                    "flip_y",
                    default=True,
                    tooltip=(
                        "`true` matches how image files are stored against how UVs are read; "
                        "`false` turns it upside down."
                    ),
                ),
                io.Int.Input(
                    "anisotropy",
                    default=1,
                    min=1,
                    max=64,
                    tooltip="Sharpness at a grazing angle. 1 is off, 16 keeps a floor crisp into the distance.",
                ),
            ],
            outputs=[
                THREE_TEXTURE.Output(
                    display_name="texture",
                    tooltip="The texture, for any map socket on a Three material node.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image,
        color_space,
        wrap_s,
        wrap_t,
        repeat_x,
        repeat_y,
        offset_x,
        offset_y,
        flip_y,
        anisotropy,
    ) -> io.NodeOutput:
        """Describe the texture.

        Raises:
            ValueError: The image is missing, oddly shaped, or has an unusable channel count.
        """
        return io.NodeOutput(
            create_spec(
                "texture",
                "TextureURL",
                params={
                    "url": texture_url(image),
                    "colorSpace": color_space,
                    "wrapS": wrap_s,
                    "wrapT": wrap_t,
                    "repeat": [float(repeat_x), float(repeat_y)],
                    "offset": [float(offset_x), float(offset_y)],
                    "rotation": 0.0,
                    "flipY": bool(flip_y),
                    "anisotropy": int(anisotropy),
                },
                meta={"source": "image"},
            )
        )

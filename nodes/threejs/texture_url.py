"""A texture loaded in the browser from an address."""

from __future__ import annotations

import math
from urllib.parse import urlsplit

from comfy_api.latest import io

from ...modules.compat.types import THREE_TEXTURE
from ...modules.interface import three_asset
from ...modules.model import NETWORK_FEATURE, network_enabled
from ...modules.threejs.spec import create_spec
from ...modules.threejs.textures import COLOR_SPACES, WRAP_MODES

REQUIRES = "threejs"

#: Schemes a web address may use once the network is allowed.
WEB_SCHEMES = ("http", "https")


def held_address(address: str) -> bool:
    """Whether an address carries its own bytes or names an asset this server holds.

    Args:
        address: The address typed into the node, already stripped.

    Returns:
        True for a ``data:`` URL or the pack's asset route, False for anything else.
    """
    parts = urlsplit(address)
    if parts.scheme.lower() == "data":
        return True
    return not parts.scheme and not parts.netloc and parts.path == three_asset.ROUTE


class ThreeTextureURL(io.ComfyNode):
    """Turn an address into a texture descriptor."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeTextureURL",
            display_name="Three Texture URL",
            search_aliases=[
                "WASThreeTextureURL",
                "Three Texture URL",
                "texture",
                "url",
                "data url",
            ],
            category="WAS Suite/Three",
            description=(
                "A texture the browser reads for itself, from a data URL already holding the "
                "bytes, or from a web address when features.network is on in config.yaml. A "
                "remote address has to allow cross-origin reads or the texture arrives blank. "
                "To use a picture from the graph, reach for Three Texture From Image instead."
            ),
            inputs=[
                io.String.Input(
                    "url",
                    default="",
                    multiline=False,
                    tooltip=(
                        "Where to read from, as a `data:image/png;base64,` string, or "
                        "`https://example.com/wood.jpg` with features.network on."
                    ),
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
                io.Float.Input(
                    "rotation",
                    default=0.0,
                    min=-360.0,
                    max=360.0,
                    step=0.1,
                    tooltip="Turn the texture on the surface, in degrees. 0.0 leaves it square, 45.0 tilts it.",
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
        url,
        color_space,
        wrap_s,
        wrap_t,
        repeat_x,
        repeat_y,
        offset_x,
        offset_y,
        rotation,
        flip_y,
        anisotropy,
    ) -> io.NodeOutput:
        """Describe the texture.

        Raises:
            ValueError: No address was given, a web address was given with
                ``features.network`` off, or the address is not ``http`` or ``https``.
        """
        address = str(url).strip()
        if not address:
            raise ValueError(
                "Three Texture URL has no address to fetch. Paste a data:image/png;base64, "
                "string, or use Three Texture From Image to take a picture from the graph "
                "instead."
            )
        remote = not held_address(address)
        if remote and not network_enabled():
            raise ValueError(
                f"Three Texture URL will not fetch {address[:80]} while {NETWORK_FEATURE} is "
                f"off. Turn {NETWORK_FEATURE} on in config.yaml to let the browser fetch web "
                "addresses, paste a data:image/png;base64, string instead, or use Three "
                "Texture From Image to take a picture from the graph."
            )
        if remote and urlsplit(address).scheme.lower() not in WEB_SCHEMES:
            raise ValueError(
                f"Three Texture URL cannot fetch {address[:80]}. Give an http:// or https:// "
                "address, or a data:image/png;base64, string."
            )
        return io.NodeOutput(
            create_spec(
                "texture",
                "TextureURL",
                params={
                    "url": address,
                    "network": remote,
                    "colorSpace": color_space,
                    "wrapS": wrap_s,
                    "wrapT": wrap_t,
                    "repeat": [float(repeat_x), float(repeat_y)],
                    "offset": [float(offset_x), float(offset_y)],
                    "rotation": math.radians(float(rotation)),
                    "flipY": bool(flip_y),
                    "anisotropy": int(anisotropy),
                },
            )
        )

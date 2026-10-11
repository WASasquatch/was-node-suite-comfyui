"""Convert a CSS HSL(A) colour string to hex."""

from __future__ import annotations

import re

from comfy_api.latest import io

#: ``hsl(0, 100%, 50%)`` and ``hsla(0, 100%, 50%, 0.5)``. Percent signs are optional and
#: the alpha group is absent on the three-component form.
HSL_PATTERN = re.compile(r"hsla?\(\s*(\d+),\s*(\d+)%?,\s*(\d+)%?(?:,\s*([\d.]+))?\s*\)")


class HSLToHex(io.ComfyNode):
    """Turn a CSS ``hsl()`` or ``hsla()`` string into ``#RRGGBB``.

    Raises:
        ValueError: The string is not in ``hsl()``/``hsla()`` form.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="HSL to Hex",
            display_name="HSL to Hex",
            search_aliases=["HSL to Hex", "colour", "color", "hex"],
            category="WAS Suite/Utilities",
            description="Convert a CSS hsl() or hsla() colour string into a hex colour string.",
            inputs=[
                io.String.Input(
                    "hsl_color",
                    default="hsl(0, 100%, 50%)",
                    tooltip=(
                        "A CSS colour: hue 0 to 359 degrees, then saturation and lightness as "
                        "percentages. 'hsl(0, 100%, 50%)' = pure red; 'hsla(0, 100%, 50%, "
                        "0.5)' adds opacity, 0.0 to 1.0. Percent signs are optional. A hue of "
                        "360 comes out black; use 0."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="hex_color",
                    tooltip=(
                        "The same colour as '#RRGGBB', e.g. '#FF0000'. An opacity below 1.0 "
                        "adds a fourth pair, giving '#RRGGBBAA'."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, hsl_color) -> io.NodeOutput:
        match = HSL_PATTERN.match(hsl_color)
        if not match:
            raise ValueError("Invalid HSL(A) color format")

        h, s, l = map(int, match.groups()[:3])
        a = float(match.groups()[3]) if match.groups()[3] else 1.0

        s /= 100
        l /= 100

        c = (1 - abs(2 * l - 1)) * s
        x = c * (1 - abs((h / 60) % 2 - 1))
        m = l - c / 2

        if 0 <= h < 60:
            r, g, b = c, x, 0
        elif 60 <= h < 120:
            r, g, b = x, c, 0
        elif 120 <= h < 180:
            r, g, b = 0, c, x
        elif 180 <= h < 240:
            r, g, b = 0, x, c
        elif 240 <= h < 300:
            r, g, b = x, 0, c
        elif 300 <= h < 360:
            r, g, b = c, 0, x
        else:
            r, g, b = 0, 0, 0

        r = int((r + m) * 255)
        g = int((g + m) * 255)
        b = int((b + m) * 255)
        alpha = int(a * 255)

        hex_color = f"#{r:02X}{g:02X}{b:02X}"
        if a < 1:
            hex_color += f"{alpha:02X}"

        return io.NodeOutput(hex_color)

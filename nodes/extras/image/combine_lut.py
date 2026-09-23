"""Mixing two colour lookup tables into one."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.sockets import require_input
from ....modules.compat.types import LUT
from ....modules.image import lut as tables
from ....modules.interface import lut_report
from ....modules.image.lut_blend import BLEND_MODES, blend

REQUIRES = "extras"


class CombineLUT(io.ComfyNode):
    """Blend two colour lookup tables into a single table."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASNSCombineLUT",
            display_name="LUT Blender",
            search_aliases=[
                "WASNSCombineLUT", "WAS LUT Blender", "lut", "blend", "mix", "grade",
            ],
            category="WAS Suite/Image/LUT",
            description=(
                "Mix two colour lookup tables into one, in whichever colour space suits the "
                "pair. Use it to dial a strong film look back towards neutral, to cross-fade "
                "between two grades, or to stack a corrective table under a creative one. "
                "Both tables are resampled to a common size first, so a 17-point table and a "
                "65-point one mix without trouble. 'linear' is a straight average, 'cosine' "
                "and 'smoothstep' ease that mix, 'slerp' turns hue the short way round the "
                "wheel instead of through grey, 'lab' and 'oklab' mix in a perceptual space "
                "which keeps midway grades believable, 'hsv' mixes hue, saturation and "
                "brightness separately, 'auto' picks 'slerp' where the two colours differ a "
                "lot and 'linear' where they agree, and 'multiply' darkens, 'screen' "
                "brightens and 'overlay' does both."
            ),
            inputs=[
                LUT.Input(
                    "lut_a",
                    tooltip=(
                        "The base table. At a strength of 0.0 this is what comes out "
                        "unchanged."
                    ),
                ),
                LUT.Input(
                    "lut_b",
                    tooltip=(
                        "The table mixed in. At a strength of 1.0 this is what comes out, "
                        "except in the multiply, screen and overlay modes, which combine the "
                        "two rather than replace one with the other."
                    ),
                ),
                io.Combo.Input(
                    "mode", options=list(BLEND_MODES),
                    tooltip=(
                        "How the two tables are mixed. 'linear' is a straight average and "
                        "the place to start; other modes ease it, mix perceptually, or "
                        "combine the tables like layers."
                    ),
                ),
                io.Float.Input(
                    "strength", default=0.5, min=0.0, max=1.0, step=0.01,
                    tooltip=(
                        "How far the mix travels from lut_a to lut_b. 0.0 keeps lut_a, 0.5 "
                        "is an even mix, 1.0 reaches lut_b."
                    ),
                ),
                io.Int.Input(
                    "output_size", default=33, min=17, max=65, step=2,
                    tooltip=(
                        "Edge length of the resulting cube, in samples. Both inputs are "
                        "resampled to it before mixing. 33 is the industry-standard size; "
                        "raise it towards 65 only when banding shows on a steep grade."
                    ),
                ),
            ],
            outputs=[
                LUT.Output(
                    display_name="lut",
                    tooltip="The mixed table, for Apply LUT or Save LUT.",
                ),
            ],
        )

    @classmethod
    def execute(cls, lut_a, lut_b, mode, strength, output_size) -> io.NodeOutput:
        """Mix the two tables.

        Raises:
            ValueError: Nothing is connected to one of the two table inputs, or one of them
                holds no table to resample.
        """
        for value, socket in ((lut_a, "lut_a"), (lut_b, "lut_b")):
            require_input(
                value, "LUT Blender", socket, "table", "Load LUT or another LUT Blender", "lut"
            )

        table_a = tables.convert_to_3d(lut_a, output_size).table_3d
        table_b = tables.convert_to_3d(lut_b, output_size).table_3d
        mixed = blend(table_a, table_b, mode, strength)
        title = f"{lut_a.title}+{lut_b.title}"
        blended = tables.LUT(title, (0, 0, 0), (1, 1, 1), None, mixed)
        lut_report.publish(blended, detail=f"{mode} at {strength:g}")
        return io.NodeOutput(blended)

"""Mixing two colour lookup tables into one."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.sockets import require_input
from ....modules.compat.types import LUT
from ....modules.image import lut as tables
from ....modules.interface import lut_report
from ....modules.image.lut_blend import BLEND_MODES, blend

REQUIRES = "extras"

#: Smallest and largest edge of the mixed cube, in samples.
MIN_OUTPUT_SIZE = 17
MAX_OUTPUT_SIZE = 65


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
                "Mix two colour lookup tables into one, to dial a strong film look back "
                "towards neutral, cross-fade between two grades, or stack a corrective table "
                "under a creative one. A 17-point and a 65-point table mix without trouble."
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
                        "'linear' = straight average; 'cosine', 'smoothstep' = eased; 'slerp' "
                        "= hue the short way, not through grey; 'lab', 'oklab' = perceptual; "
                        "'hsv' = hue, saturation, value apart; 'auto' = slerp where colours "
                        "differ, else linear; 'multiply' = darkens; 'screen' = brightens; "
                        "'overlay' = both."
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
                    "output_size", default=33, min=MIN_OUTPUT_SIZE, max=MAX_OUTPUT_SIZE,
                    step=2,
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

        output_size = max(MIN_OUTPUT_SIZE, min(int(output_size), MAX_OUTPUT_SIZE))
        table_a = tables.convert_to_3d(lut_a, output_size).table_3d
        table_b = tables.convert_to_3d(lut_b, output_size).table_3d
        mixed = blend(table_a, table_b, mode, strength)
        title = f"{lut_a.title}+{lut_b.title}"
        blended = tables.LUT(title, (0, 0, 0), (1, 1, 1), None, mixed)
        lut_report.publish(blended, detail=f"{mode} at {strength:g}")
        return io.NodeOutput(blended)

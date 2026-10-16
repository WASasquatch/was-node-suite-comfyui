"""Bake the grade between two images into a reusable colour lookup table."""

from __future__ import annotations

import numpy as np
import torch
from comfy_api.latest import io

from ....modules.compat.types import LUT
from ....modules.image import color_match, lut
from ....modules.interface import lut_report

#: Cube edge lengths offered. 33 is what most grading software writes and what `.cube`
#: readers are surest of; 17 is smaller and coarser, 65 is finer and eight times the size.
SIZES = ("17", "33", "65")

DEFAULT_SIZE = "33"


class LUTFromReference(io.ComfyNode):
    """Measure the grade from one image to another and write it as a 3D lookup table."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLUTFromReference",
            display_name="LUT from Reference",
            search_aliases=[
                'WASLUTFromReference',
                "LUT from Reference",
                "make lut",
                "bake lut",
                "match to lut",
                "grade to lut",
                "steal a look",
            ],
            category="WAS Suite/Image/LUT",
            description=(
                "Work out the colour grade that takes one image to another and bake it "
                "into a LUT. Image Color Match applies that grade to one batch; this "
                "captures it so it can be saved, blended and reused on anything."
            ),
            inputs=[
                io.Image.Input(
                    "source",
                    tooltip=(
                        "The ungraded look, usually a frame straight out of your pipeline. "
                        "The LUT maps colours from here to the reference."
                    ),
                ),
                io.Image.Input(
                    "reference",
                    tooltip=(
                        "The look to capture. A batch is pooled into one distribution, so "
                        "several frames describe a target grade rather than one shot."
                    ),
                ),
                io.Combo.Input(
                    "method",
                    options=list(color_match.METHODS),
                    default="mkl",
                    tooltip=(
                        "How the grade is measured. `mkl` matches the full colour covariance "
                        "and suits most looks. `reinhard` matches mean and spread per channel, "
                        "which is gentler. `histogram` matches the whole distribution and is "
                        "the strongest and the most likely to posterise."
                    ),
                ),
                io.Combo.Input(
                    "color_space",
                    options=list(color_match.COLOR_SPACES),
                    default="RGB",
                    tooltip=(
                        "Where the match is measured. `RGB` is direct. `Lab` separates "
                        "lightness from colour, which usually holds skin tones better."
                    ),
                ),
                io.Combo.Input(
                    "size",
                    options=list(SIZES),
                    default=DEFAULT_SIZE,
                    tooltip=(
                        "Edge length of the cube. 33 is the usual choice and what most "
                        "grading software writes. 65 is finer and eight times the data; 17 "
                        "is coarser and can band on a smooth sky."
                    ),
                ),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How much of the measured grade to bake in. 1.0 is the full match, "
                        "0.5 bakes it at half power, and 0.0 writes the identity LUT."
                    ),
                ),
                io.String.Input(
                    "title",
                    default="",
                    optional=True,
                    tooltip=(
                        "Name written into the LUT's TITLE line, which is what Save LUT "
                        "(.cube) puts in the file. Empty names it after the method."
                    ),
                ),
            ],
            outputs=[
                LUT.Output(
                    display_name="lut",
                    tooltip=(
                        "The grade as a 3D lookup table, for Apply LUT, LUT Blender or "
                        "Save LUT (.cube)."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls, source, reference, method="mkl", color_space="RGB", size=DEFAULT_SIZE,
        strength=1.0, title="",
    ) -> io.NodeOutput:
        """Measure the grade from source to reference and bake it into a cube.

        Raises:
            ValueError: ``size`` is not one of :data:`SIZES`.
        """
        if str(size).strip() not in SIZES:
            raise ValueError(
                f"LUT from Reference was given a size of {size!r}, and the cube is built at "
                f"{', '.join(SIZES[:-1])} or {SIZES[-1]} samples. Set size to one of those; "
                f"33 is the usual choice."
            )
        edge = int(str(size).strip())
        rgb_source = source[..., :3]
        rgb_reference = reference[..., :3]

        working_source = color_match.to_space(rgb_source, color_space)
        working_reference = color_match.to_space(rgb_reference, color_space)
        channels = working_source.shape[-1]
        source_stats = color_match.measure(working_source.reshape(-1, channels), method)
        target_stats = color_match.measure(working_reference.reshape(-1, channels), method)

        # The cube is carried through the same transform an image would be, so whatever the
        # match does to a colour is what the table records for that colour.
        cube = lut.identity_cube(edge)
        working_cube = color_match.to_space(cube.reshape(1, edge * edge, edge, 3), color_space)
        mapped = color_match.map_pixels(
            working_cube.reshape(-1, channels), source_stats, target_stats, method, float(strength)
        )
        graded = color_match.from_space(
            mapped.reshape(1, edge * edge, edge, channels), color_space
        )

        table = torch.clamp(graded, 0.0, 1.0).reshape(edge, edge, edge, 3)
        built = lut.LUT(
            title=str(title).strip() or f"{method} from reference",
            table_3d=np.ascontiguousarray(table.cpu().numpy().astype(np.float32)),
        )
        lut_report.publish(built, strip=False, detail=f"{method} in {color_space}")
        return io.NodeOutput(built)

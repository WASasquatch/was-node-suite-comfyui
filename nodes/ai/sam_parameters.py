"""Point prompts for Segment Anything."""

from __future__ import annotations

import numpy as np
from comfy_api.latest import io

from ...modules.compat.types import SAM_PARAMETERS

REQUIRES = "sam"


def parse_matrix(text: str) -> np.ndarray:
    """Parse a semicolon-separated matrix literal into an array.

    Args:
        text: The widget value.

    Returns:
        A two-dimensional array, one row per ``;``-separated group.

    Raises:
        ValueError: The rows are not all the same length, or a value is not a number
            literal.
    """
    # numpy.matrix is the one numpy entry point that reads a matrix literal out of a string.
    return np.asarray(np.matrix(text))


class SamParameters(io.ComfyNode):
    """Build the points and labels `SAM Image Mask` segments on."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="SAM Parameters",
            display_name="SAM Parameters",
            search_aliases=["SAM Parameters", "sam points", "segment anything points"],
            category="WAS Suite/Image/Masking",
            description=(
                "Describe which parts of an image Segment Anything should select, as a list "
                "of points and a matching list of keep/drop labels. Enable features.sam to "
                "load this node."
            ),
            inputs=[
                io.String.Input(
                    "points",
                    default="[128, 128]; [0, 0]",
                    multiline=False,
                    tooltip=(
                        "The points to segment from, as x and y pixels from the top left, "
                        "separated by semicolons: '[128, 128]; [0, 0]' is two points, one "
                        "128 pixels in from the corner and one on the corner itself."
                    ),
                ),
                io.String.Input(
                    "labels",
                    default="[1, 0]",
                    multiline=False,
                    tooltip=(
                        "One number per point, in the same order: 1 means 'the thing I want "
                        "is here', 0 means 'this is background, leave it out'. '[1, 0]' "
                        "keeps whatever sits under the first point and pushes the mask away "
                        "from the second."
                    ),
                ),
            ],
            outputs=[
                SAM_PARAMETERS.Output(
                    tooltip=(
                        "The points and labels, for the sam_parameters input of SAM Image "
                        "Mask or either input of SAM Parameters Combine."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, points, labels) -> io.NodeOutput:
        parameters = {
            "points": parse_matrix(points),
            "labels": parse_matrix(labels)[0],
        }
        return io.NodeOutput(parameters)

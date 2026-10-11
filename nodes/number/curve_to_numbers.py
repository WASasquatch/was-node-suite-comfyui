"""Turn a drawn curve into numbers across a range."""

from __future__ import annotations

import math

from comfy_api.latest import io, ui

from ...modules.compat.types import LIST, NUMBER
from ...modules.image import curve_numbers


class CurveToNumbers(io.ComfyNode):
    """Read a curve as numbers, walking a range by a fixed step."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASCurveToNumbers",
            display_name="Curve to Numbers",
            search_aliases=[
                "WASCurveToNumbers",
                "Curve to Numbers",
                "curve points",
                "curve to list",
                "tone curve numbers",
                "schedule from curve",
                "ramp",
                "envelope",
            ],
            category="WAS Suite/Number/Operations",
            description=(
                "Read a hand-drawn curve as numbers, walking minimum to maximum by step, so "
                "its shape can drive a strength ramp, a denoise schedule or a camera move. "
                "Bend the plot on the node or type the points. Every value comes out on one "
                "wire and one per run, for a list node or a For Loop."
            ),
            inputs=[
                io.String.Input(
                    "curve_points",
                    default="",
                    multiline=True,
                    tooltip=(
                        "The curve as '0,0;128,200;255,255' on a 0 to 255 scale, lowest input "
                        "first. Drag the plot to draw it, or paste an Image Curves node's "
                        "curve_points (its composite RGB curve is read). Empty = a straight "
                        "line, an even ramp."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input(
                        "minimum",
                        default=0.0,
                        min=-curve_numbers.MAX_RANGE,
                        max=curve_numbers.MAX_RANGE,
                        step=0.01,
                        round=curve_numbers.MIN_STEP,
                    ),
                    [io.Float, NUMBER, io.Int],
                    tooltip=(
                        "The bottom of the range, and the value the curve reads out at its "
                        "lowest. 0.0 for a strength or a denoise, -1.0 with a maximum of 1.0 "
                        "for a move either side of centre."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input(
                        "maximum",
                        default=1.0,
                        min=-curve_numbers.MAX_RANGE,
                        max=curve_numbers.MAX_RANGE,
                        step=0.01,
                        round=curve_numbers.MIN_STEP,
                    ),
                    [io.Float, NUMBER, io.Int],
                    tooltip=(
                        "The top of the range, and the value the curve reads out at its "
                        "highest. 1.0 for a blend factor, 255 for a colour level. Below "
                        "minimum stops with an error."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input(
                        "step",
                        default=0.1,
                        min=curve_numbers.MIN_STEP,
                        max=curve_numbers.MAX_RANGE,
                        step=0.01,
                        round=curve_numbers.MIN_STEP,
                    ),
                    [io.Float, NUMBER, io.Int],
                    tooltip=(
                        "The increment from one position to the next, which is what decides "
                        "how many values there are. 0.1 over 0.0 to 1.0 gives 11 of them. "
                        "Maximum is included only where the step divides the range exactly, "
                        "so a step of 0.3 stops at 0.9."
                    ),
                ),
                io.Int.Input(
                    "decimals",
                    default=6,
                    min=0,
                    max=curve_numbers.MAX_DECIMALS,
                    step=1,
                    tooltip=(
                        "How many decimal places each number is rounded to. This is what "
                        "keeps 0.30000000000000004 out of a filename or a log line. 0 rounds "
                        "to whole numbers."
                    ),
                ),
            ],
            outputs=[
                LIST.Output(
                    display_name="values",
                    tooltip=(
                        "What the curve reads at each position, on one wire and on the "
                        "minimum to maximum scale, for Number List Statistics, Text List Get "
                        "and Text List Length. The straight line answers the positions back."
                    ),
                ),
                LIST.Output(
                    display_name="positions",
                    tooltip=(
                        "Where each value sits along the range, on one wire and in the same "
                        "order: 0, 0.1, 0.2 and so on for 0.0 to 1.0 by 0.1. Entry 3 here is "
                        "the position of value 3, which is what plots the pair."
                    ),
                ),
                io.Float.Output(
                    display_name="value",
                    is_output_list=True,
                    tooltip=(
                        "The same values one per run, so the graph below runs once for each: "
                        "wire it into a sampler's denoise or a blend factor to render the "
                        "whole curve as a series."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip=(
                        "How many values there are, which is how many times the graph below "
                        "the 'value' output runs. 11 for a range of 0.0 to 1.0 by 0.1."
                    ),
                ),
                io.String.Output(
                    display_name="text",
                    tooltip=(
                        "The values on one line, separated by commas, as '0, 0.5, 1'. Wire it "
                        "into Text to Console or Save Text File to keep the numbers a curve "
                        "produced, or into a text input that takes a list of weights."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        curve_points="",
        minimum=0.0,
        maximum=1.0,
        step=0.1,
        decimals=6,
    ) -> io.NodeOutput:
        """Read the curve across the range.

        Returns:
            The values as a list and one per run, the positions they sit at, how many there
            are, and the values written out on one line.

        Raises:
            ValueError: A bound is not a real number, ``maximum`` is below ``minimum``, or
                the walk asks for more values than the node will make.
        """
        low = float(minimum)
        high = float(maximum)
        size = max(abs(float(step)), curve_numbers.MIN_STEP)
        if not all(math.isfinite(number) for number in (low, high, size)):
            raise ValueError(
                f"Curve to Numbers was given a minimum of {low}, a maximum of {high} and a "
                f"step of {size}, and all three have to be real numbers. Check what is wired "
                f"into them, or type the range in, such as 0.0 to 1.0 by 0.1."
            )
        if high < low:
            raise ValueError(
                f"Curve to Numbers was given a maximum of {high:g} below its minimum of "
                f"{low:g}. The curve is read upwards from minimum to maximum, so raise "
                f"maximum to {low:g} or above, or lower minimum."
            )
        count = curve_numbers.count_of(low, high, size)
        if count > curve_numbers.MAX_VALUES:
            fits = (high - low) / (curve_numbers.MAX_VALUES - 1)
            raise ValueError(
                f"Curve to Numbers was asked for {count} values, walking {low:g} to {high:g} "
                f"by {size:g}, and makes at most {curve_numbers.MAX_VALUES}. Raise step to "
                f"{fits:.6g} or above, or narrow the range."
            )

        positions, values = curve_numbers.read(curve_points, low, high, size, int(decimals))
        written = curve_numbers.text_of(values)
        return io.NodeOutput(
            list(values),
            list(positions),
            list(values),
            len(values),
            written,
            ui=ui.PreviewText(f"{len(values)} value(s), {low:g} to {high:g} by {size:g}\n{written}"),
        )

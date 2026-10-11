"""Generate a series of numbers between two values."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import LIST, NUMBER
from ...modules.util.easing import EASING_NAMES, ease
from ...modules.util.numbers import whole_steps


class NumberRange(io.ComfyNode):
    """Generate a series of numbers between two values, by count or by step.

    Every numeric output is a list.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASNumberRange",
            display_name="Number Range",
            search_aliases=[
                "WASNumberRange", "Number Range",
                "sequence",
                "linspace",
                "series",
                "frames",
                "schedule",
                "arange",
                "number list",
                "list of numbers",
            ],
            category="WAS Suite/Number/Operations",
            description=(
                "Generate a series of numbers between two values, by count or by step, shaped "
                "by an easing curve. Every value comes out as a list, so the graph below runs "
                "once per number, for a camera move, a strength ramp or a frame sweep."
            ),
            inputs=[
                io.MultiType.Input(
                    io.Float.Input("start", default=0.0, min=-1e9, max=1e9, step=0.01),
                    [io.Float, NUMBER, io.Int],
                    tooltip="The first value of the series.",
                ),
                io.MultiType.Input(
                    io.Float.Input("stop", default=1.0, min=-1e9, max=1e9, step=0.01),
                    [io.Float, NUMBER, io.Int],
                    tooltip=(
                        "The value the series runs to. Lower than start counts downwards, "
                        "which is how a fade-out is written."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=["count", "step"],
                    tooltip=(
                        "`count` = exactly that many values spread across the span, when the "
                        "frame count is fixed; `step` = walks start towards stop by a fixed "
                        "gap, as many as fit, when the spacing matters."
                    ),
                ),
                io.Int.Input(
                    "count",
                    default=10,
                    min=1,
                    max=10000,
                    step=1,
                    tooltip=(
                        "How many values to produce, read in `count` mode. A count of 1 "
                        "gives start alone. Every node below this one runs this many times, "
                        "so a large count is a large queue."
                    ),
                ),
                io.Float.Input(
                    "step",
                    default=0.1,
                    min=-1e9,
                    max=1e9,
                    step=0.01,
                    tooltip=(
                        "The gap between values in `step` mode; its sign follows start to "
                        "stop, so 0.1 and -0.1 match. The series ends on stop or the last "
                        "whole step before it. 0 stops with an error."
                    ),
                ),
                io.Combo.Input(
                    "easing",
                    options=list(EASING_NAMES),
                    tooltip=(
                        "In `count` mode only. `linear` = even spacing; `ease_in` = starts "
                        "slow, accelerates; `ease_out` = the reverse; `ease_in_out` = both; "
                        "`back`, `elastic` = overshoot past start and stop."
                    ),
                ),
                io.Boolean.Input(
                    "endpoint",
                    default=True,
                    tooltip=(
                        "Whether the last value is exactly stop, in `count` mode. Turn it "
                        "off for a seamless loop: the final frame of a loop is the first "
                        "frame of the next pass, so emitting both repeats it."
                    ),
                ),
                io.Int.Input(
                    "decimals",
                    default=6,
                    min=0,
                    max=12,
                    step=1,
                    tooltip=(
                        "How many decimal places each value is rounded to. This is what "
                        "keeps 0.30000000000000004 out of a filename or a log line. 0 rounds "
                        "to whole numbers."
                    ),
                ),
            ],
            outputs=[
                LIST.Output(
                    tooltip=(
                        "Every value on one wire, for Number Easing, Text List Get, Text "
                        "List Length and the other list nodes."
                    ),
                ),
                NUMBER.Output(
                    is_output_list=True,
                    tooltip=(
                        "One NUMBER per value. Because this is a list, a node reading it "
                        "runs once for each. This is the socket that drives a series of "
                        "renders."
                    ),
                ),
                io.Float.Output(
                    display_name="floats",
                    is_output_list=True,
                    tooltip="The same values as decimals, one per run.",
                ),
                io.Int.Output(
                    display_name="ints",
                    is_output_list=True,
                    tooltip=(
                        "The same values as whole numbers, cut off rather than rounded, one "
                        "per run. For a step count, a frame number or a seed."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip=(
                        "How many values were produced, which is how many times the graph "
                        "below this node runs."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        start=0.0,
        stop=1.0,
        mode="count",
        count=10,
        step=0.1,
        easing="linear",
        endpoint=True,
        decimals=6,
    ) -> io.NodeOutput:
        values = cls.series(float(start), float(stop), mode, int(count), float(step), easing, endpoint)
        values = [round(value, int(decimals)) for value in values]
        whole = [int(value) for value in values]
        # A list of its own per slot, so a node that edits the one it was handed does not
        # change what the other slots emit or how many times the graph below them runs.
        return io.NodeOutput(values, list(values), list(values), whole, len(values))

    @staticmethod
    def series(
        start: float,
        stop: float,
        mode: str,
        count: int,
        step: float,
        easing: str,
        endpoint: bool,
    ) -> list[float]:
        """Build the series.

        Args:
            start: First value.
            stop: Value the series runs towards.
            mode: ``count`` for a fixed number of values, ``step`` for a fixed spacing.
            count: Values to produce in ``count`` mode.
            step: Spacing in ``step`` mode. Its sign is ignored; direction comes from
                ``start`` and ``stop``.
            easing: Curve shaping the distribution in ``count`` mode.
            endpoint: Whether the last value of a ``count`` series is exactly ``stop``.

        Returns:
            The values, in order. A ``step`` series ends on ``stop`` when the step divides
            the span, and on the last whole step before ``stop`` when it does not.

        Raises:
            ValueError: ``step`` is 0 in ``step`` mode, which never reaches ``stop``.
        """
        if mode == "step":
            if step == 0:
                raise ValueError(
                    "Number Range was given a step of 0, which never reaches stop. Use a "
                    "non-zero step, or switch mode to 'count'."
                )
            span = stop - start
            size = abs(step) * (1 if span >= 0 else -1)
            steps = whole_steps(abs(span), abs(step)) + 1
            return [start + size * index for index in range(steps)]

        if count <= 1:
            return [start]
        divisor = (count - 1) if endpoint else count
        return [start + ease(easing, index / divisor) * (stop - start) for index in range(count)]

"""Remap numbers through an easing curve, one value or a whole list."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import LIST, NUMBER
from ...modules.util.easing import EASING_NAMES, ease
from ...modules.util.numbers import UNREADABLE, as_numbers, split_values, whole

#: Display name, written into the messages this node raises.
NAME = "Number Easing"


class NumberEasing(io.ComfyNode):
    """Convert numbers from one range to another through an easing curve.

    Every numeric output is a list.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASNumberEasing",
            display_name="Number Easing",
            search_aliases=[
                "WASNumberEasing", "Number Easing",
                "map range",
                "remap",
                "lerp",
                "interpolate",
                "curve",
                "normalize",
                "number list",
                "list of numbers",
                "scale",
                "offset",
                "schedule",
                "ramp",
            ],
            category="WAS Suite/Number/Operations",
            description=(
                "Convert numbers from one range to another through an easing curve. Type one "
                "number or a list, or wire in a list such as Number Range's, and every value "
                "comes out as a list, so the graph below runs once per number. `linear` is a "
                "plain remap; the others also shape how values accelerate."
            ),
            inputs=[
                io.MultiType.Input(
                    io.String.Input(
                        "value",
                        default="",
                        multiline=True,
                        placeholder="0, 0.25, 0.5, 0.75, 1 (or one number to a line)",
                    ),
                    [io.String, NUMBER, io.Float, io.Int, LIST],
                    tooltip=(
                        "The numbers to convert, one to a line or separated by commas. A "
                        "comma is a separator rather than a thousands mark, so 1,000 reads as "
                        "two values."
                    ),
                ),
                io.Combo.Input(
                    "easing",
                    options=list(EASING_NAMES),
                    tooltip=(
                        "`linear` = a plain range conversion; `ease_in_*` = starts slow; "
                        "`ease_out_*` = finishes slow; `ease_in_out_*` = both; `*_back`, "
                        "`*_elastic` = overshoot the output range; `*_bounce` = settles in "
                        "shrinking hops, never leaving it."
                    ),
                ),
                io.Float.Input(
                    "input_min",
                    default=0.0,
                    min=-1e9,
                    max=1e9,
                    step=0.01,
                    tooltip=(
                        "Start of the input range, shared by every value; values outside it "
                        "are pulled to the nearest end. Match the source: 0 and 23 for a "
                        "24-frame counter, or the start and stop of the Number Range feeding "
                        "it."
                    ),
                ),
                io.Float.Input(
                    "input_max",
                    default=1.0,
                    min=-1e9,
                    max=1e9,
                    step=0.01,
                    tooltip=(
                        "The value that counts as the end of the input range. Equal to "
                        "input_min stops with an error, since a range of no width cannot say "
                        "where a value sits in it."
                    ),
                ),
                io.Float.Input(
                    "output_min",
                    default=0.0,
                    min=-1e9,
                    max=1e9,
                    step=0.01,
                    tooltip=(
                        "What the start of the input range becomes. Larger than output_max "
                        "runs the result backwards, which is how a fade-out is written "
                        "without touching the curve."
                    ),
                ),
                io.Float.Input(
                    "output_max",
                    default=1.0,
                    min=-1e9,
                    max=1e9,
                    step=0.01,
                    tooltip="What the end of the input range becomes.",
                ),
                io.Boolean.Input(
                    "clamp",
                    default=True,
                    tooltip=(
                        "On = each result held inside the output range; off = `back` "
                        "overshoots about a tenth of the span and `elastic` almost a whole "
                        "one. Keep it on where the result feeds a hard limit such as denoise."
                    ),
                ),
                io.Combo.Input(
                    "unreadable",
                    options=list(UNREADABLE),
                    tooltip=(
                        "What an entry that is not a number does. `skip` = left out; `zero` = "
                        "0 in its place, keeping every position; `error` = stops the prompt "
                        "and names the entry."
                    ),
                ),
            ],
            outputs=[
                LIST.Output(
                    tooltip=(
                        "Every converted value on one wire, for Text List Get, Text List "
                        "Length and the other list nodes."
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
                        "The same values as whole numbers, one per run, for a step count or a "
                        "pixel size. Cut off rather than rounded, so 7.6 arrives as 7."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip=(
                        "How many values were converted, which is how many times the graph "
                        "below this node runs."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        value="",
        easing="linear",
        input_min=0.0,
        input_max=1.0,
        output_min=0.0,
        output_max=1.0,
        clamp=True,
        unreadable="skip",
    ) -> io.NodeOutput:
        from ...modules.compat.lists import require_values

        entries = split_values(value)
        values = as_numbers(entries, unreadable, NAME)
        require_values(values, cls.nothing_to_convert(len(entries), isinstance(value, str)))

        eased = cls.convert(
            values,
            easing,
            float(input_min),
            float(input_max),
            float(output_min),
            float(output_max),
            clamp,
        )
        whole_numbers = [whole(number) for number in eased]
        # A list of its own per slot, so a node that edits the one it was handed does not
        # change what the other slots emit or how many times the graph below them runs.
        return io.NodeOutput(eased, list(eased), list(eased), whole_numbers, len(eased))

    @staticmethod
    def nothing_to_convert(entries: int, typed: bool) -> str:
        """The message for a run that found no number to convert.

        Args:
            entries: How many entries the value input held.
            typed: Whether the value arrived as text, which is what an empty box looks like
                and what a wire carrying a number or a list never does.

        Returns:
            The exception text for whichever of the three ways a run finds nothing
            happened: an empty box, a wire that delivered no values, or entries that hold
            no number.
        """
        if entries:
            return (
                f"{NAME} read no numbers out of the {entries} entr"
                f"{'y' if entries == 1 else 'ies'} it was given, so there is nothing to "
                f"convert and the graph below it cannot be run. Check that they are figures "
                f"rather than words, or set unreadable to 'zero' to keep a value in every "
                f"position."
            )
        if typed:
            return (
                f"{NAME} was given an empty value, so there is nothing to convert: either "
                f"the box is empty with nothing wired into it, or a text wire delivered no "
                f"characters. Type the numbers into the box, one to a line or separated by "
                f"commas, or wire a number or a list in, such as the LIST or NUMBER output "
                f"of Number Range."
            )
        return (
            f"{NAME} received nothing on its value input, so there is nothing to convert. "
            f"The node wired into value produced no values, so that node is where the "
            f"empty result comes from and where to look first. Disconnect the wire to type "
            f"the numbers into the box instead."
        )

    @staticmethod
    def convert(
        values: list[float],
        easing: str,
        input_min: float,
        input_max: float,
        output_min: float,
        output_max: float,
        clamp: bool,
    ) -> list[float]:
        """Convert every value through the curve.

        Args:
            values: The numbers to convert.
            easing: Curve applied to each value once it is normalised.
            input_min: Value that counts as the start of the input range.
            input_max: Value that counts as the end of it.
            output_min: What ``input_min`` becomes.
            output_max: What ``input_max`` becomes.
            clamp: Whether each result is held between ``output_min`` and ``output_max``.

        Returns:
            The converted values, in the order they were given.

        Raises:
            ValueError: ``input_min`` and ``input_max`` are equal, so no value can be placed
                in the input range.
        """
        span = input_max - input_min
        if span == 0:
            raise ValueError(
                f"{NAME} was given an input_min and input_max that are equal, so there is "
                f"no range to place the values in. Set them to the lowest and highest value "
                f"the source produces."
            )

        low, high = sorted((output_min, output_max))
        converted = []
        for value in values:
            eased = ease(easing, (value - input_min) / span)
            result = output_min + eased * (output_max - output_min)
            converted.append(min(high, max(low, result)) if clamp else result)
        return converted

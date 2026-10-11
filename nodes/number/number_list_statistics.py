"""Sum, mean, min, max, median and range over a list of numbers."""

from __future__ import annotations

import math
import statistics

from comfy_api.latest import io

from ...modules.compat.types import LIST, NUMBER
from ...modules.util.numbers import UNREADABLE, as_numbers, split_values

#: Display name, written into the messages this node raises.
NAME = "Number List Statistics"

#: Fewest and most decimal places the figures are rounded to.
MIN_DECIMALS = 0
MAX_DECIMALS = 12


class NumberListStatistics(io.ComfyNode):
    """Measure a list of numbers onto one figure per output."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASNumberListStatistics",
            display_name="Number List Statistics",
            search_aliases=[
                "WASNumberListStatistics",
                "Number List Statistics",
                "sum",
                "total",
                "average",
                "mean",
                "median",
                "minimum",
                "maximum",
                "range",
                "spread",
                "normalize",
                "number list",
                "list of numbers",
                "schedule",
                "statistics",
            ],
            category="WAS Suite/Number/Operations",
            description=(
                "Measure a whole list of numbers at once: sum, mean, min, max, median, range, "
                "count and a summary line. Each figure is a single value, so the graph below "
                "runs once however long the list is. Divide by max to normalise a schedule, "
                "or by sum to turn weights into shares."
            ),
            inputs=[
                io.MultiType.Input(
                    io.String.Input(
                        "values",
                        default="",
                        multiline=True,
                        placeholder="0, 0.25, 0.5, 0.75, 1 (or one number to a line)",
                    ),
                    [io.String, LIST, NUMBER, io.Float, io.Int],
                    tooltip=(
                        "The numbers to measure, one to a line or separated by commas, or a "
                        "LIST wired in from Number Range or Text List. A comma separates "
                        "values rather than marking thousands, so 1,000 reads as two values."
                    ),
                ),
                io.Combo.Input(
                    "unreadable",
                    options=list(UNREADABLE),
                    tooltip=(
                        "What an entry that is not a number does. `skip` leaves it out and "
                        "lowers count, `zero` counts it as 0 and pulls the mean down, "
                        "`error` stops the prompt and names the entry."
                    ),
                ),
                io.Int.Input(
                    "decimals",
                    default=4,
                    min=MIN_DECIMALS,
                    max=MAX_DECIMALS,
                    step=1,
                    tooltip=(
                        "Decimal places every figure is rounded to, on the outputs and in "
                        "the summary. 0 = whole numbers; 2 = 0.33; 6 = 0.333333. Raise it "
                        "where sum or mean feeds further maths."
                    ),
                ),
            ],
            outputs=[
                io.Float.Output(
                    display_name="sum",
                    tooltip=(
                        "Every value added together. 1, 2, 3 gives 6. Divide a value by "
                        "this to get its share of the whole."
                    ),
                ),
                io.Float.Output(
                    display_name="mean",
                    tooltip=(
                        "The average, which is sum divided by count. 1, 2, 6 gives 3. A "
                        "single far-off value drags it, so read it beside median."
                    ),
                ),
                io.Float.Output(
                    display_name="min",
                    tooltip=(
                        "The smallest value. 4, 1, 9 gives 1. Subtract it from each value "
                        "and divide by range to spread a schedule across 0 to 1."
                    ),
                ),
                io.Float.Output(
                    display_name="max",
                    tooltip=(
                        "The largest value. 4, 1, 9 gives 9. Divide each value by this to "
                        "normalise a schedule against its own peak."
                    ),
                ),
                io.Float.Output(
                    display_name="median",
                    tooltip=(
                        "The middle value once sorted, or the average of the middle two "
                        "when count is even. 1, 2, 90 gives 2, where mean gives 31."
                    ),
                ),
                io.Float.Output(
                    display_name="range",
                    tooltip=(
                        "max minus min, the width the values cover. 4, 1, 9 gives 8. 0 "
                        "means every value is the same, so dividing by it to normalise "
                        "would fail."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip=(
                        "How many values were read. `skip` leaves an entry holding no number "
                        "out, so 4 entries with one word among them gives 3, where `zero` "
                        "gives 4. Compare it with the entries given to see how many held no "
                        "number."
                    ),
                ),
                io.String.Output(
                    display_name="summary",
                    tooltip=(
                        "Every figure on one line, rounded to decimals: `count 3, sum "
                        "6.0000, mean 2.0000, min 1.0000, max 3.0000, median 2.0000, range "
                        "2.0000`. Wire it to a text preview or into a filename."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, values="", unreadable="skip", decimals=4) -> io.NodeOutput:
        """Measure the values and answer each figure.

        Args:
            values: Numbers as text, as a ``LIST``, or as one number.
            unreadable: What an entry holding no number does.
            decimals: Decimal places every figure is rounded to.

        Returns:
            Sum, mean, min, max, median, range, count and the summary line.

        Raises:
            ValueError: No entry held a number.
        """
        entries = split_values(values)
        numbers = as_numbers(entries, unreadable, NAME)
        if not numbers:
            raise ValueError(cls.nothing_to_measure(len(entries), isinstance(values, str)))

        places = max(MIN_DECIMALS, min(MAX_DECIMALS, int(decimals)))
        figures = [round(figure, places) for figure in cls.measure(numbers)]
        summary = cls.summarise(figures, len(numbers), places)
        return io.NodeOutput(*figures, len(numbers), summary)

    @staticmethod
    def measure(numbers: list[float]) -> list[float]:
        """Work the six figures out of the values.

        Args:
            numbers: The values read, at least one.

        Returns:
            Sum, mean, min, max, median and range, in that order.
        """
        total = math.fsum(numbers)
        low = min(numbers)
        high = max(numbers)
        return [
            total,
            total / len(numbers),
            low,
            high,
            float(statistics.median(numbers)),
            high - low,
        ]

    @staticmethod
    def summarise(figures: list[float], count: int, places: int) -> str:
        """Write the figures out as one line.

        Args:
            figures: Sum, mean, min, max, median and range, already rounded.
            count: How many values were read.
            places: Decimal places each figure is written to.

        Returns:
            The line, naming count first and then each figure.
        """
        names = ("sum", "mean", "min", "max", "median", "range")
        written = ", ".join(
            f"{name} {figure:.{places}f}" for name, figure in zip(names, figures)
        )
        return f"count {count}, {written}"

    @staticmethod
    def nothing_to_measure(entries: int, typed: bool) -> str:
        """The message for a run that found no number to measure.

        Args:
            entries: How many entries the values input held.
            typed: Whether the values arrived as text.

        Returns:
            The exception text for whichever way the run found nothing.
        """
        if entries:
            return (
                f"{NAME} read no numbers out of the {entries} entr"
                f"{'y' if entries == 1 else 'ies'} it was given, so there is nothing to "
                f"measure. Check that they are figures rather than words, or set unreadable "
                f"to 'zero' to count every entry as 0."
            )
        if typed:
            return (
                f"{NAME} was given an empty value, so there is nothing to measure: either "
                f"the box is empty with nothing wired into it, or a text wire delivered no "
                f"characters. Type the numbers into the box, one to a line or separated by "
                f"commas, or wire a list in, such as the LIST output of Number Range."
            )
        return (
            f"{NAME} received nothing on its values input, so there is nothing to measure. "
            f"The node wired into values produced no values, so that node is where the "
            f"empty result comes from and where to look first. Disconnect the wire to type "
            f"the numbers into the box instead."
        )

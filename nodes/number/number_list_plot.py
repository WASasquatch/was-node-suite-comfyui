"""Draw lists of numbers as lines on the node."""

from __future__ import annotations

import math

from comfy_api.latest import io

from ...modules.compat.types import LIST, NUMBER
from ...modules.util.numbers import as_number

#: Display name, written into the messages this node raises.
NAME = "Number List Plot"

#: The key the node's run output carries its series under, read by its interface.
UI_KEY = "was_number_plot"

#: The series inputs, in the order they are drawn.
SERIES = ("values", "values_b", "values_c", "values_d")

#: Most entries one series sends to the node's plot. The rest are counted, not drawn.
MAX_DRAWN = 10000

#: Tooltip on every series past the first.
MORE_TIP = (
    "Another list drawn over the first in its own colour, to compare two schedules, as the "
    "strengths of two LoRAs."
)


class NumberListPlot(io.ComfyNode):
    """Plot up to four lists of numbers on the node."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASNumberListPlot",
            display_name="Number List Plot",
            search_aliases=[
                "WASNumberListPlot",
                "Number List Plot",
                "visualizer",
                "visualize numbers",
                "graph",
                "chart",
                "plot",
                "preview numbers",
                "schedule preview",
                "number list",
                "list of numbers",
            ],
            category="WAS Suite/Number/Operations",
            description=(
                "Draw up to four lists of numbers as lines on the node, with each list's "
                "length, lowest and highest value, to check a schedule, a curve or a sweep "
                "before it drives a render. Takes a LIST, a per-run list or one number."
            ),
            inputs=[
                io.MultiType.Input(
                    "values",
                    [LIST, NUMBER, io.Float, io.Int],
                    tooltip=(
                        "The list to draw: a LIST such as Number Sequence's values, a per-run "
                        "output such as its floats, or one number. Entry 1 is drawn on the "
                        "left."
                    ),
                ),
                io.MultiType.Input(
                    "values_b", [LIST, NUMBER, io.Float, io.Int], optional=True, tooltip=MORE_TIP
                ),
                io.MultiType.Input(
                    "values_c", [LIST, NUMBER, io.Float, io.Int], optional=True, tooltip=MORE_TIP
                ),
                io.MultiType.Input(
                    "values_d", [LIST, NUMBER, io.Float, io.Int], optional=True, tooltip=MORE_TIP
                ),
            ],
            outputs=[],
            is_input_list=True,
            is_output_node=True,
        )

    @classmethod
    def execute(cls, values=None, values_b=None, values_c=None, values_d=None) -> io.NodeOutput:
        """Read each wired list and send it to the node's plot.

        Returns:
            No outputs; the series ride on the run output under :data:`UI_KEY`.

        Raises:
            ValueError: The first list holds no number.
        """
        from ...modules.interface.channel import executing_workflow_id

        drawn = []
        for name, wired in zip(SERIES, (values, values_b, values_c, values_d)):
            if wired is None:
                continue
            numbers, skipped = cls.numbers(wired)
            if name == SERIES[0] and not numbers:
                given = f" among the {skipped} entr{'y' if skipped == 1 else 'ies'} it was given"
                raise ValueError(
                    f"{NAME} found no number on values{given if skipped else ''}. Wire in a "
                    f"list of numbers, such as Number Sequence's values."
                )
            drawn.append({
                "name": name,
                "values": numbers[:MAX_DRAWN],
                "total": len(numbers),
                "skipped": skipped,
            })
        report = {"workflow": executing_workflow_id() or "", "series": drawn}
        return io.NodeOutput(ui={UI_KEY: [report]})

    @staticmethod
    def numbers(wired) -> tuple[list[float], int]:
        """The numbers one input carries, a list nested one deep read flat.

        Args:
            wired: What the input received, as a list: one entry per run, or one entry holding
                a whole list.

        Returns:
            The finite numbers in order, and how many entries were not numbers.
        """
        entries = []
        for item in wired if isinstance(wired, (list, tuple)) else [wired]:
            if isinstance(item, (list, tuple)):
                entries.extend(item)
            else:
                entries.append(item)
        numbers = []
        skipped = 0
        for entry in entries:
            number = as_number(entry)
            if number is None or not math.isfinite(number):
                skipped += 1
                continue
            numbers.append(number)
        return numbers, skipped

"""Compare two values of any kind and answer true or false."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.types import NUMBER
from ....modules.logic.compare import COMPARISONS, compare


class CompareAny(io.ComfyNode):
    """Compare two values and answer a boolean."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASCompareAny",
            display_name="Compare",
            search_aliases=[
                "WASCompareAny",
                "Compare",
                "if",
                "condition",
                "equals",
                "greater than",
                "less than",
            ],
            category="WAS Suite/Logic/Boolean",
            description=(
                "Compare two values and answer true or false, whatever they are. Numbers "
                "compare as numbers and everything else as text, so '10' is greater than "
                "'9' rather than sorting before it. Feed the answer to any switch."
            ),
            inputs=[
                io.MultiType.Input(
                    io.String.Input("value_a", default="", multiline=False),
                    [io.String, io.Int, io.Float, NUMBER, io.Boolean],
                    tooltip="Left-hand value. Text, a number or a switch.",
                ),
                io.Combo.Input(
                    "comparison",
                    options=list(COMPARISONS),
                    default="equals",
                    tooltip=(
                        "The test. `equals` and the four orderings read both sides as "
                        "numbers where they can: 10 > 9. `contains`, `starts with`, `ends "
                        "with` and `matches regex` read them as text. `is empty` ignores "
                        "value_b."
                    ),
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "value_b", default="", multiline=False,
                        tooltip=(
                            "The right-hand value, as `10` or `cat`. For `matches regex` it is "
                            "the pattern, as `^cat|dog$`, stopped with an error past 1 second. "
                            "Patterns run in the regex package's re-compatible mode."
                        ),
                    ),
                    [io.String, io.Int, io.Float, NUMBER, io.Boolean],
                    tooltip="Right-hand value. Ignored by `is empty`.",
                ),
            ],
            outputs=[
                io.Boolean.Output(
                    display_name="boolean",
                    tooltip="true when the test holds. Wire it to any switch's boolean.",
                ),
                io.String.Output(
                    display_name="comparison_text",
                    tooltip="The test written out, as `10 greater than 9 = true`.",
                ),
            ],
        )

    @classmethod
    def execute(cls, value_a="", comparison="equals", value_b="") -> io.NodeOutput:
        """Answer whether the comparison holds.

        Args:
            value_a: Left-hand value.
            comparison: One of the offered tests.
            value_b: Right-hand value.

        Returns:
            The answer, and the test written out.

        Raises:
            ValueError: The comparison is unknown or a regex could not be read.
        """
        answer = compare(value_a, value_b, comparison)
        if comparison == "is empty":
            text = f"{value_a!r} is empty = {str(answer).lower()}"
        else:
            text = f"{value_a} {comparison} {value_b} = {str(answer).lower()}"
        return io.NodeOutput(answer, text)

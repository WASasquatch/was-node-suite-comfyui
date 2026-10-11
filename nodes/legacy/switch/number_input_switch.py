"""Route one of two numbers onward."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.types import NUMBER

REQUIRES = "switches"


class NumberInputSwitch(io.ComfyNode):
    """Select between two numbers with a boolean."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Number Input Switch",
            display_name="Number Input Switch",
            search_aliases=["Number Input Switch", "number switch", "boolean switch"],
            category="WAS Suite/Logic/Switch",
            description=(
                "Deprecated: use Any Input Switch instead. Passes number_a on when boolean is "
                "true and number_b when it is false, also as a float and a whole number."
            ),
            inputs=[
                io.MultiType.Input(
                    "number_a",
                    [NUMBER, io.Int, io.Float],
                    tooltip="The value sent on when boolean is true.",
                ),
                io.MultiType.Input(
                    "number_b",
                    [NUMBER, io.Int, io.Float],
                    tooltip="The value sent on when boolean is false.",
                ),
                io.Boolean.Input(
                    "boolean",
                    default=True,
                    tooltip=(
                        "Which input passes; BOOLEAN. true = number_a, false = "
                        "number_b. Toggle it, or wire it from Logic Boolean."
                    ),
                ),
            ],
            outputs=[
                NUMBER.Output(
                    tooltip=(
                        "The selected value, keeping the type it arrived as: a whole number "
                        "stays whole, 2.5 stays 2.5."
                    ),
                ),
                io.Float.Output(
                    tooltip="The same value as a float, so 8 leaves here as 8.0.",
                ),
                io.Int.Output(
                    tooltip=(
                        "The same value as a whole number, cut off rather than rounded, so "
                        "2.9 leaves here as 2."
                    ),
                ),
            ],
            is_deprecated=True,
        )

    @classmethod
    def execute(cls, number_a, number_b, boolean=True) -> io.NodeOutput:
        number = number_a if boolean else number_b
        return io.NodeOutput(number, float(number), int(number))

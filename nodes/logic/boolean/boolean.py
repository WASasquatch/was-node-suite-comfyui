"""A true/false switch read from a toggle, a number or a line of text."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.types import NUMBER
from ....modules.logic.compare import to_boolean


class LogicBoolean(io.ComfyNode):
    """Read a toggle, a number or a line of text as true or false."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Logic Boolean",
            display_name="Logic Boolean",
            search_aliases=["Logic Boolean", "boolean", "bool", "toggle"],
            category="WAS Suite/Logic/Boolean",
            description=(
                "One true/false switch for the Input Switch nodes and any node with a toggle "
                "or reset input, also given as 1 or 0. Wire a number or text into it to "
                "decide from that instead: 0 is false and any other number true, so a loop "
                "index reads false on its first pass only."
            ),
            inputs=[
                io.MultiType.Input(
                    io.Boolean.Input("boolean", default=True),
                    [io.Int, io.Float, NUMBER, io.String],
                    tooltip=(
                        "Ticked is true. A wired number is true unless it is 0, so index 0 is "
                        "false and every later index true. Text reads `true`, `yes`, `on` or a "
                        "non-zero number as true; `false`, `no`, `off`, `0`, empty or any other "
                        "word is false."
                    ),
                ),
            ],
            outputs=[
                io.Boolean.Output(
                    tooltip="True or false, for the boolean input of an Input Switch node.",
                ),
                NUMBER.Output(
                    tooltip="The same answer as 1 or 0, for a NUMBER input such as a reset.",
                ),
                io.Int.Output(tooltip="The same answer as 1 or 0, on an INT socket."),
                io.Float.Output(tooltip="The same answer as 1.0 or 0.0, on a FLOAT socket."),
            ],
        )

    @classmethod
    def execute(cls, boolean=True) -> io.NodeOutput:
        """Answer the toggle, or what is wired into it, as true or false.

        Args:
            boolean: The toggle's state, or the number or text wired in its place.

        Returns:
            The answer as a boolean, a NUMBER, an INT and a FLOAT.
        """
        answer = to_boolean(boolean, default=False)
        return io.NodeOutput(answer, int(answer), int(answer), float(answer))

"""Read a number, a switch or a line of text as true or false."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.types import NUMBER
from ....modules.logic.compare import to_boolean


class ToBoolean(io.ComfyNode):
    """Convert an INT, FLOAT, STRING, NUMBER or BOOLEAN to a BOOLEAN."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASToBoolean",
            display_name="To Boolean",
            search_aliases=[
                "WASToBoolean",
                "To Boolean",
                "int to boolean",
                "text to boolean",
                "truthy",
                "convert boolean",
            ],
            category="WAS Suite/Logic/Boolean",
            description=(
                "Read any value as true or false, so a count, a size or a line of text can "
                "drive a switch. Every switch in this pack keys on a boolean, and only one "
                "other node answers one from a number, so this is usually what stands "
                "between a measurement and a branch."
            ),
            inputs=[
                io.MultiType.Input(
                    io.String.Input("value", default="", multiline=False),
                    [io.String, io.Int, io.Float, NUMBER, io.Boolean],
                    tooltip=(
                        "What to read. A number is true when it is not 0. Text is read as a "
                        "word, then as a number, so `true`, `yes`, `on` and `2` are true and "
                        "`false`, `no`, `off`, `0` and empty are false."
                    ),
                ),
                io.Boolean.Input(
                    "unreadable",
                    default=False,
                    tooltip=(
                        "Answer for text that is neither, such as `maybe`. false treats it "
                        "as false; true treats any unrecognised word as true."
                    ),
                ),
                io.Boolean.Input(
                    "invert",
                    default=False,
                    tooltip="Flip the answer, saving a Logic NOT after it.",
                ),
            ],
            outputs=[
                io.Boolean.Output(
                    display_name="boolean",
                    tooltip="The value as true or false. Wire it to any switch's boolean.",
                ),
                io.Int.Output(
                    display_name="int",
                    tooltip="The same answer as 1 or 0, for arithmetic that counts branches.",
                ),
            ],
        )

    @classmethod
    def execute(cls, value="", unreadable=False, invert=False) -> io.NodeOutput:
        """Answer the value as a boolean.

        Args:
            value: What to read.
            unreadable: Answer for a word that reads as neither.
            invert: Flip the answer.

        Returns:
            The boolean, and the same answer as 1 or 0.
        """
        answer = to_boolean(value, default=bool(unreadable))
        if invert:
            answer = not answer
        return io.NodeOutput(answer, int(answer))

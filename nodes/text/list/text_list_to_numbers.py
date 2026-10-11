"""Read a LIST as numbers, onto the NUMBER, FLOAT and INT sockets, as lists."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.types import LIST, NUMBER
from ....modules.util.numbers import UNREADABLE, as_numbers

#: Display name, written into the messages this node raises.
NAME = "Text List to Numbers"


class TextListToNumbers(io.ComfyNode):
    """Read a ``LIST`` as numbers onto the NUMBER, FLOAT and INT sockets.

    All three are declared ``is_output_list``, so the graph below runs once per value.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASTextListToNumbers",
            display_name="Text List to Numbers",
            search_aliases=[
                "WASTextListToNumbers", "Text List to Numbers",
                "list to numbers",
                "parse numbers",
                "string to float list",
                "output is list",
                "number list",
                "list of numbers",
                "schedule",
                "sum",
                "average",
                "mean",
            ],
            category="WAS Suite/Text/List",
            description=(
                "Read a LIST as numbers and emit them on the NUMBER, FLOAT and INT sockets "
                "as lists, so every node downstream runs once per value. Entries that are "
                "not numbers, nan and infinity included, are skipped, zeroed or stop the "
                "prompt, as `unreadable` chooses."
            ),
            inputs=[
                LIST.Input(
                    "text_list",
                    extra_dict={"forceInput": True},
                    tooltip=(
                        "The list to read. Entries may be numbers already or text holding "
                        "one; surrounding space, a leading + or - and a decimal point are "
                        "all read, so '  -1.5 ' arrives as -1.5."
                    ),
                ),
                io.Combo.Input(
                    "unreadable",
                    options=list(UNREADABLE),
                    tooltip=(
                        "What an entry that is not a number does. `skip` = left out, for a "
                        "column with a heading; `zero` = 0 in its place, so entry 7 stays "
                        "entry 7; `error` = stop the prompt, naming the entry."
                    ),
                ),
                io.Boolean.Input(
                    "round_to_int",
                    default=False,
                    tooltip=(
                        "Whether the INT socket rounds to the nearest whole number instead "
                        "of cutting the decimal off. Off, 1.9 arrives as 1; on, it arrives "
                        "as 2. The NUMBER and FLOAT sockets keep the decimal either way."
                    ),
                ),
            ],
            outputs=[
                NUMBER.Output(
                    is_output_list=True,
                    tooltip=(
                        "One NUMBER per entry, for the maths and counter nodes; a node "
                        "reading this runs once per value. A list with no readable number "
                        "stops the prompt; set unreadable to `zero` to keep a value in "
                        "every position."
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
                        "The same values as whole numbers, one per run, for a step count or "
                        "a seed."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip=(
                        "How many values came through, which is how many times the graph "
                        "below this node runs."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, text_list, unreadable="skip", round_to_int=False) -> io.NodeOutput:
        from ....modules.compat.lists import as_list, require_values

        entries = as_list(text_list)
        values = as_numbers(entries, unreadable, NAME)

        require_values(
            values,
            f"{NAME} read no numbers out of the {len(entries)} entr"
            f"{'y' if len(entries) == 1 else 'ies'} it was given, so the graph below it "
            f"cannot be run. Set unreadable to 'zero' to keep a value in every position, "
            f"or check that the list holds figures rather than words.",
        )
        whole = [round(value) if round_to_int else int(value) for value in values]
        # A list of its own per slot, so a node that edits the one it was handed does not
        # change what the other slots emit or how many times the graph below them runs.
        return io.NodeOutput(values, list(values), whole, len(values))

"""Take a run of entries out of a list."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.compat.types import LIST

logger = log.get_logger("nodes.text.list")


class TextListSlice(io.ComfyNode):
    """Take the entries from ``start`` to ``end`` of a ``LIST``, both included."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASTextListSlice",
            display_name="Text List Slice",
            search_aliases=[
                "WASTextListSlice", "Text List Slice",
                "slice",
                "range of lines",
                "take from list",
                "sublist",
                "every nth",
                "number list",
                "list of numbers",
                "trim schedule",
            ],
            category="WAS Suite/Text/List",
            description=(
                "Take a run of entries out of a list, from start to end with both ends "
                "included, optionally every second or third one. The run comes out as one "
                "LIST and as a STRING list that runs everything downstream once per entry. "
                "A range that selects nothing stops the prompt."
            ),
            inputs=[
                LIST.Input(
                    "text_list",
                    extra_dict={"forceInput": True},
                    tooltip=(
                        "The list to take from, such as the lines output of Load Text Line or "
                        "the LIST output of Text Split to List, Text List or Text Dictionary "
                        "Keys."
                    ),
                ),
                io.Int.Input(
                    "start",
                    default=0,
                    min=-99999999,
                    max=99999999,
                    step=1,
                    tooltip=(
                        "The first entry taken, counting from 0, and it is taken. -1 is the "
                        "last entry, -2 the one before it. A start before the beginning of "
                        "the list begins at the first entry."
                    ),
                ),
                io.Int.Input(
                    "end",
                    default=-1,
                    min=-99999999,
                    max=99999999,
                    step=1,
                    tooltip=(
                        "The last entry taken, and it is taken as well: start 9 and end 19 "
                        "give 11 entries, the 10th to the 20th. -1 = the last entry; an end "
                        "past the list stops at its last entry."
                    ),
                ),
                io.Int.Input(
                    "step",
                    default=1,
                    min=1,
                    max=99999999,
                    step=1,
                    tooltip=(
                        "How far to move between entries taken. 1 takes every entry, 2 takes "
                        "every second one starting at start, 3 every third. The last entry is "
                        "taken only when the stepping lands on it."
                    ),
                ),
            ],
            outputs=[
                LIST.Output(
                    tooltip=(
                        "The entries taken, on one wire, for Text List Get, Text List "
                        "Concatenate and Text List to Text."
                    ),
                ),
                io.String.Output(
                    display_name="strings",
                    is_output_list=True,
                    tooltip=(
                        "The same entries as a STRING list: a node reading it runs once per "
                        "entry, so wired into a sampler's prompt it renders every line of the "
                        "range in turn. An entry that is not text is converted to it."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many entries the range took.",
                ),
            ],
        )

    @classmethod
    def execute(cls, text_list, start=0, end=-1, step=1) -> io.NodeOutput:
        from ....modules.compat.lists import as_list, require_values

        entries = as_list(text_list)
        length = len(entries)
        first, last = cls.bounds(start, end, length)
        taken = entries[first : last + 1 : max(1, int(step))] if last >= first else []
        strings = [str(entry) for entry in taken]
        logger.debug(
            "Text List Slice took %d of %d entries, %d to %d by %d",
            len(taken), length, first, last, step,
        )
        require_values(
            strings,
            f"Text List Slice was asked for entries {start} to {end} of a list holding "
            f"{length}, which selects nothing, so there is no list to hand on and the graph "
            f"below it cannot be run. start counts from 0 and end is included, so the last "
            f"entry of this list is {length - 1} and end must not be before start. Check "
            f"what feeds text_list, and that its length is what the range expects.",
        )
        return io.NodeOutput(taken, strings, len(taken))

    @staticmethod
    def bounds(start: int, end: int, length: int) -> tuple[int, int]:
        """The requested range as positions inside the list.

        Args:
            start: The first entry requested.
            end: The last entry requested, included.
            length: How many entries the list holds.

        Returns:
            ``(first, last)``, both inside the list, or a pair with ``last`` below ``first``
            where the range selects nothing: an empty list, an end before the start, a start
            past the last entry, or an end before the first.
        """
        first = int(start) + length if start < 0 else int(start)
        last = int(end) + length if end < 0 else int(end)
        return max(0, first), min(length - 1, last)

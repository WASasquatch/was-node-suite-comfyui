"""Read one entry out of a list by its position."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.types import LIST, NUMBER


class TextListGet(io.ComfyNode):
    """Read the entry of a ``LIST`` at ``index``.

    A negative index counts back from the end. ``out_of_range`` decides what happens past
    either end.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASTextListGet",
            display_name="Text List Get",
            search_aliases=[
                "WASTextListGet", "Text List Get",
                "list index",
                "list item",
                "select from list",
                "nth",
                "number list",
                "list of numbers",
                "schedule",
            ],
            category="WAS Suite/Text/List",
            description=(
                "Read one entry out of a list by position, with negative positions counting "
                "from the end. An index past either end wraps, clamps, comes back empty or "
                "stops, so a counter that only climbs can cycle a list forever. Hands on the "
                "entry as text, the position read and the list's length."
            ),
            inputs=[
                LIST.Input(
                    "text_list",
                    extra_dict={"forceInput": True},
                    tooltip=(
                        "The list to read from, such as the LIST output of Text Split to "
                        "List, Text List or Text Dictionary Keys."
                    ),
                ),
                io.MultiType.Input(
                    io.Int.Input("index", default=0, min=-99999999, max=99999999, step=1),
                    [io.Int, NUMBER, io.Float],
                    tooltip=(
                        "Which entry to take, counting from 0. -1 is the last entry, -2 the "
                        "one before it. Wire a Number Counter in to step through the list "
                        "one entry per run. A decimal value is cut down to a whole number."
                    ),
                ),
                io.Combo.Input(
                    "out_of_range",
                    options=["wrap", "clamp", "empty", "error"],
                    tooltip=(
                        "What an index past the end does: `wrap` counts round again, "
                        "`clamp` sticks at the first or last entry, `empty` returns "
                        "nothing, `error` stops the prompt."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="text",
                    tooltip=(
                        "The entry at that position, as text. An entry that is not text is "
                        "converted to it, so a list of numbers reads out as numerals."
                    ),
                ),
                io.Int.Output(
                    display_name="resolved_index",
                    tooltip=(
                        "The position actually read, after wrapping or clamping. Worth "
                        "watching when a counter drives the index, since it is the entry "
                        "number the result really came from."
                    ),
                ),
                io.Int.Output(
                    display_name="length",
                    tooltip="How many entries the list holds.",
                ),
            ],
        )

    @classmethod
    def execute(cls, text_list, index=0, out_of_range="wrap") -> io.NodeOutput:
        entries = list(text_list) if isinstance(text_list, (list, tuple)) else [text_list]
        length = len(entries)
        if length == 0:
            if out_of_range == "error":
                raise ValueError(
                    "Text List Get was given an empty list. Set out_of_range to 'empty' to "
                    "let an empty list pass through as an empty string."
                )
            return io.NodeOutput("", 0, 0)

        position = cls.resolve(int(index), length, out_of_range)
        if position is None:
            return io.NodeOutput("", 0, length)
        return io.NodeOutput(str(entries[position]), position, length)

    @staticmethod
    def resolve(index: int, length: int, out_of_range: str) -> int | None:
        """Turn a requested index into a real one.

        Args:
            index: The requested position. Negative counts back from the end.
            length: How many entries there are, which is one or more.
            out_of_range: ``wrap``, ``clamp``, ``empty`` or ``error``.

        Returns:
            A position inside the list, or ``None`` when the index is outside it and
            ``out_of_range`` is ``empty``.

        Raises:
            IndexError: The index is outside the list and ``out_of_range`` is ``error``.
        """
        position = index + length if index < 0 else index
        if 0 <= position < length:
            return position
        if out_of_range == "wrap":
            return position % length
        if out_of_range == "clamp":
            return 0 if position < 0 else length - 1
        if out_of_range == "error":
            raise IndexError(
                f"Text List Get was asked for entry {index} of a list holding {length}. "
                f"Set out_of_range to 'wrap', 'clamp' or 'empty' to allow it."
            )
        return None

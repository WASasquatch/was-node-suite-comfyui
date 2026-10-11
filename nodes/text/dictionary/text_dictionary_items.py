"""Open a dictionary into parallel key and value lists."""

from __future__ import annotations

from collections.abc import Mapping

from comfy_api.latest import io

from ....modules.compat.sockets import require_input
from ....modules.compat.types import DICT, LIST

#: Keep the entries in the order they were added.
NONE = "none"

#: Order the entries by their key.
BY_KEY = "key"

#: Order the entries by their value, read as text.
BY_VALUE = "value"

#: How the entries may be ordered, in the order the widget lists them.
SORTS = [NONE, BY_KEY, BY_VALUE]


def sort_key(value) -> tuple[str, str]:
    """Sort key reading a dictionary key or value as text.

    Args:
        value: One key or one value out of the dictionary.

    Returns:
        The text folded for case, then the text as it stands.
    """
    text = value if isinstance(value, str) else str(value)
    return text.casefold(), text


def ordered(dictionary: Mapping, sort: str) -> list[tuple]:
    """The dictionary's entries in the requested order.

    Args:
        dictionary: The mapping to read.
        sort: One of :data:`SORTS`. An unrecognised name keeps insertion order.

    Returns:
        ``(key, value)`` pairs.
    """
    entries = list(dictionary.items())
    if sort == BY_KEY:
        return sorted(entries, key=lambda entry: sort_key(entry[0]))
    if sort == BY_VALUE:
        return sorted(entries, key=lambda entry: sort_key(entry[1]))
    return entries


class DictionaryItems(io.ComfyNode):
    """Split a dictionary into a list of keys and a matching list of values."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASTextDictionaryItems",
            display_name="Text Dictionary Items",
            search_aliases=[
                "WASTextDictionaryItems",
                "Text Dictionary Items",
                "dictionary values",
                "dictionary items",
                "dict entries",
                "key value pairs",
                "loop over dictionary",
            ],
            category="WAS Suite/Text/Dictionary",
            description=(
                "Open a dictionary into two lists that line up: the keys, and their values. "
                "Entry 0 of one belongs with entry 0 of the other, so a For Loop stepping "
                "one index through both reads a key and its value together. The entries "
                "also come out written as text, and counted for the loop to run on."
            ),
            inputs=[
                DICT.Input(
                    "dictionary",
                    tooltip="The dictionary whose entries are wanted, keys and values both.",
                ),
                io.Combo.Input(
                    "sort",
                    options=SORTS,
                    default=NONE,
                    tooltip=(
                        "Order of both lists. `none` = the order the entries were added; "
                        "`key` = A to Z by name; `value` = A to Z by the value as text. "
                        "Case is ignored and digits sort as text: '10' before '9'."
                    ),
                ),
            ],
            outputs=[
                LIST.Output(
                    display_name="keys",
                    tooltip=(
                        "The entry names on one wire, such as ['subject', 'style']. Feed it "
                        "to Text List Get with a For Loop's index to read one name per "
                        "iteration."
                    ),
                ),
                LIST.Output(
                    display_name="values",
                    tooltip=(
                        "What each name is stored against, in step with keys. An entry "
                        "holding several alternatives stays a list, so ['a cat', 'a wolf'] "
                        "arrives whole rather than as writing."
                    ),
                ),
                io.String.Output(
                    display_name="pairs",
                    tooltip=(
                        "Every entry written as `subject: a cat`, one to a line. For a "
                        "preview, a caption or a saved text file. A value carrying line "
                        "breaks of its own spans several lines. An empty dictionary gives "
                        "an empty string."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip=(
                        "How many entries the dictionary holds, 0 when it holds none. Feed "
                        "it to a For Loop's iteration count to run the graph once per entry."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, dictionary, sort=NONE) -> io.NodeOutput:
        """Answer the keys, the values, the entries as text, and how many there are.

        Args:
            dictionary: The dictionary to open.
            sort: One of :data:`SORTS`.

        Returns:
            The keys, the values, the entries one per line, and the entry count.

        Raises:
            ValueError: The dictionary input carries nothing, or carries something that is
                not a dictionary.
        """
        require_input(
            dictionary,
            "Text Dictionary Items",
            "dictionary",
            "dictionary",
            "Text Dictionary New or Text Dictionary Convert",
            "DICT",
        )
        if not isinstance(dictionary, Mapping):
            raise ValueError(
                f"Text Dictionary Items was handed a {type(dictionary).__name__} on its "
                f"dictionary input rather than a dictionary, so it has no entries to open. "
                f"Text Dictionary Convert emits whatever its text describes, so check that "
                f"text is written with braces and colons, as {{'subject': 'a cat'}}."
            )
        entries = ordered(dictionary, sort)
        keys = [key for key, _ in entries]
        values = [value for _, value in entries]
        pairs = "\n".join(f"{key}: {value}" for key, value in entries)
        return io.NodeOutput(keys, values, pairs, len(entries))

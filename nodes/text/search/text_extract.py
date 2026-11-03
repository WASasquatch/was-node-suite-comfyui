"""Pull every match of a regular expression, or one of its groups, out of a text."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.types import LIST
from ....modules.prompt import text_extract, text_lists

#: The node and input a pattern comes from, for messages.
WHERE = "Text Extract's pattern"


class TextExtract(io.ComfyNode):
    """Find every match of a pattern in a text and hand back the matches or a group of them."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASTextExtract",
            display_name="Text Extract",
            search_aliases=[
                "WASTextExtract", "Text Extract",
                "regex extract",
                "capture group",
                "find all",
                "pull number",
                "parse value",
            ],
            category="WAS Suite/Text/Search",
            description=(
                "Pull every match of a regular expression out of a text, or the part one "
                "group captures, such as the number in `30 steps`. The matches come out as "
                "one LIST, as the first match and its count, and as a STRING list that runs "
                "the nodes below once per match."
            ),
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    placeholder="Eg: Steps: 30, CFG scale: 7",
                    tooltip=(
                        "The text searched, as `Steps: 30, CFG scale: 7`. Wired from a STRING "
                        "list, the node runs once per entry."
                    ),
                ),
                io.String.Input(
                    "pattern",
                    default="",
                    multiline=False,
                    tooltip=(
                        "A regular expression, as `(\\d+) steps` or `Steps: (\\d+)`. Empty "
                        "finds nothing. A search past 1 second stops with an error."
                    ),
                ),
                io.Int.Input(
                    "group",
                    default=0,
                    min=0,
                    max=99,
                    step=1,
                    tooltip=(
                        "0 = the whole match; 1 = the first parenthesised group, so "
                        "`(\\d+) steps` gives `30`; 2 = the second. A group the pattern "
                        "lacks stops the run."
                    ),
                ),
                io.Boolean.Input(
                    "ignore_case",
                    default=True,
                    tooltip="On = `steps` matches `Steps`; off = case must match.",
                ),
            ],
            outputs=[
                LIST.Output(
                    display_name="matches",
                    tooltip=(
                        "Every match in order, on one wire for Text List Get, Text List "
                        "Filter and the other list nodes. Empty when nothing matched."
                    ),
                ),
                io.String.Output(
                    display_name="first",
                    tooltip="The first match, or an empty string when nothing matched.",
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many matches were found. 0 = none.",
                ),
                io.String.Output(
                    display_name="match",
                    is_output_list=True,
                    tooltip=(
                        "Every match as a STRING list, so a node reading it runs once per "
                        "match. No match stops the nodes reading it."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, text, pattern="", group=0, ignore_case=True) -> io.NodeOutput:
        from ....modules.compat.lists import block_if_empty

        found = text_extract.extract(text or "", pattern or "", group, ignore_case, WHERE)
        reason = (
            f"Text Extract found no match for pattern {text_lists.quoted(pattern)}, so the "
            f"nodes reading its match output have nothing to run on. Check the pattern "
            f"against the text, or read count first."
            if pattern else
            "Text Extract has no pattern, so the nodes reading its match output have nothing "
            "to run on. Type a pattern, as (\\d+) steps."
        )
        return io.NodeOutput(
            found,
            found[0] if found else "",
            len(found),
            block_if_empty(list(found), reason),
        )

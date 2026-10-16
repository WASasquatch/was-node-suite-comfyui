"""Test whether a string contains a substring or matches a regular expression."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.util import patterns


class TextFind(io.ComfyNode):
    """Report whether ``text`` holds ``substring``, or matches ``pattern``."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text Find",
            display_name="Text Find",
            search_aliases=["Text Find", "search", "contains", "regex", "match"],
            category="WAS Suite/Text/Search",
            description=(
                "Search text for a plain substring, or for a regular expression when the "
                "substring field is left empty."
            ),
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    placeholder="Eg: a cat on a mat",
                    tooltip=(
                        "Text to search; STRING, as `a tabby cat`. Read only, never changed."
                    ),
                ),
                io.String.Input(
                    "substring",
                    default="",
                    multiline=False,
                    tooltip=(
                        "Exact text to look for, taken literally and matched case "
                        "sensitively, so 'cat' does not find 'Cat'. Filled in, it is what "
                        "gets searched for and pattern is ignored; leave it empty to search "
                        "with pattern instead."
                    ),
                ),
                io.String.Input(
                    "pattern",
                    default="",
                    multiline=False,
                    tooltip=(
                        "Regular expression searched for when substring is empty, as "
                        "'cat|dog' for either word or '^photo' for one at the start. Empty "
                        "matches everything. A search past 1 second stops with an error. "
                        "Patterns run in the regex package's re-compatible mode."
                    ),
                ),
            ],
            outputs=[
                io.Boolean.Output(
                    display_name="found",
                    tooltip=(
                        "True when the substring or the pattern was found anywhere in the "
                        "text."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, text, substring, pattern) -> io.NodeOutput:
        if substring:
            return io.NodeOutput(substring in text)

        return io.NodeOutput(bool(patterns.search(pattern, text, "Text Find's pattern")))

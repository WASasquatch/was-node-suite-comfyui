"""Tidy a comma-separated prompt: split it, drop the repeats, put it back together."""

from __future__ import annotations

from comfy_api.latest import io, ui

from ...modules.compat.types import LIST
from ...modules.prompt import tags as tag_tools


class PromptTagCleanup(io.ComfyNode):
    """Split a prompt into tags, remove duplicates and empties, and rejoin it."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASPromptTagCleanup",
            display_name="Prompt Tag Cleanup",
            search_aliases=[
                "WASPromptTagCleanup", "Prompt Tag Cleanup",
                "dedupe",
                "unique tags",
                "clean prompt",
                "tidy prompt",
                "remove duplicates",
            ],
            category="WAS Suite/Text/Operations",
            description=(
                "Split a prompt into tags, drop the duplicates and the empty ones, "
                "optionally sort and cap the count, and join it back up. Emphasis such as "
                "(tag:1.4) is recognised as the same tag as the plain spelling."
            ),
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    tooltip=(
                        "The prompt to tidy, as `a tabby cat,, a tabby cat`. Typed here, or wired in from "
                        "whatever built it: Text Concatenate, a style selector, or a caption node."
                    ),
                ),
                io.String.Input(
                    "delimiter",
                    default=",",
                    multiline=False,
                    tooltip=(
                        "What separates one tag from the next in the incoming text. Leave it "
                        "empty to treat every word as its own tag. Type \\n to split a "
                        "prompt written one tag to a line."
                    ),
                ),
                io.String.Input(
                    "join_with",
                    default=", ",
                    multiline=False,
                    tooltip=(
                        "What is put between the tags on the way out. The default ', ' is "
                        "the ordinary prompt spelling; type \\n to get one tag per line, "
                        "which is easier to read in a saved text file."
                    ),
                ),
                io.Boolean.Input(
                    "dedupe",
                    default=True,
                    tooltip=(
                        "Whether a tag appearing more than once is reduced to one. The "
                        "survivor keeps the position of the first occurrence, so tidying "
                        "does not reshuffle the prompt."
                    ),
                ),
                io.Boolean.Input(
                    "ignore_case",
                    default=True,
                    tooltip=(
                        "Whether 'Neon Glow' and 'neon glow' count as the same tag. Off, "
                        "both survive, which is only useful where a downstream tool treats "
                        "capitalisation as meaningful."
                    ),
                ),
                io.Boolean.Input(
                    "ignore_emphasis",
                    default=True,
                    tooltip=(
                        "Whether '(neon glow:1.4)' counts as the same tag as 'neon glow'. On "
                        "with keep set to 'last' is the combination that collapses a prompt "
                        "onto its weighted spellings, which is normally the intended one, "
                        "the plain duplicate is usually what a second source contributed."
                    ),
                ),
                io.Combo.Input(
                    "keep",
                    options=["first", "last"],
                    tooltip=(
                        "Which of a set of duplicates survives. `first` keeps the earliest "
                        "spelling, `last` the latest. Position is the first occurrence "
                        "either way, so keeping the last spelling does not move the tag to "
                        "the end of the prompt."
                    ),
                ),
                io.Boolean.Input(
                    "remove_empty",
                    default=True,
                    tooltip=(
                        "Whether tags holding nothing are dropped. This is what clears the "
                        "run of bare commas an unconnected input leaves behind, which "
                        "otherwise reaches the text encoder as it stands."
                    ),
                ),
                io.Boolean.Input(
                    "collapse_whitespace",
                    default=True,
                    tooltip=(
                        "Whether runs of spaces, tabs and line breaks inside a tag become a "
                        "single space. This is what removes the line breaks a multi-line "
                        "prompt box leaves in the middle of a tag."
                    ),
                ),
                io.Combo.Input(
                    "sort",
                    options=["none", "a-z", "z-a", "shortest first", "longest first"],
                    tooltip=(
                        "How the surviving tags are ordered. `none` = as written, keeping the "
                        "weight early tags carry in most encoders; `a-z` and `z-a` = "
                        "alphabetical, for comparing prompts by eye; `longest first` = "
                        "phrases ahead of single words; `shortest first` = the reverse."
                    ),
                ),
                io.Int.Input(
                    "limit",
                    default=0,
                    min=0,
                    max=9999,
                    step=1,
                    tooltip=(
                        "Keep at most this many tags, counted after everything else has run. "
                        "0 keeps all of them. Useful for trimming a caption model's output "
                        "to the few tags worth keeping."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="text",
                    tooltip="The tidied prompt, joined back up with join_with.",
                ),
                LIST.Output(
                    display_name="tags",
                    tooltip=(
                        "The surviving tags as one LIST, for Text List Get and the other "
                        "list nodes."
                    ),
                ),
                io.String.Output(
                    display_name="tag_strings",
                    is_output_list=True,
                    tooltip=(
                        "The same tags as a STRING list, so a node reading this runs once "
                        "per tag, such as one render per tag. A prompt that tidies down to "
                        "no tags stops the nodes reading this socket; the text output is "
                        "still delivered."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many tags survived.",
                ),
                io.Int.Output(
                    display_name="removed",
                    tooltip=(
                        "How many entries the tidy-up took out, counting duplicates, empties "
                        "and anything past the limit. 0 means the prompt was already clean."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        text,
        delimiter=",",
        join_with=", ",
        dedupe=True,
        ignore_case=True,
        ignore_emphasis=True,
        keep="first",
        remove_empty=True,
        collapse_whitespace=True,
        sort="none",
        limit=0,
    ) -> io.NodeOutput:
        from ...modules.compat.lists import block_if_empty

        cleaned, removed = tag_tools.clean(
            text,
            delimiter=cls.unescape(delimiter),
            dedupe=dedupe,
            ignore_case=ignore_case,
            ignore_emphasis=ignore_emphasis,
            keep=keep,
            remove_empty=remove_empty,
            collapse_whitespace=collapse_whitespace,
            order=sort,
            limit=limit,
        )
        joined = cls.unescape(join_with).join(cleaned)
        return io.NodeOutput(
            joined,
            cleaned,
            # A list of its own per slot, so a node that edits the LIST it was handed does
            # not change how many times the graph under tag_strings runs.
            block_if_empty(
                list(cleaned),
                "Prompt Tag Cleanup tidied the prompt down to no tags at all, so the nodes "
                "reading its tag_strings output have nothing to run on. The text input is "
                "empty or held only separators and duplicates.",
            ),
            len(cleaned),
            removed,
            ui=ui.PreviewText(joined),
        )

    @staticmethod
    def unescape(value: str) -> str:
        """Read the two-character escapes a single-line widget has to stand in with.

        Args:
            value: A delimiter as typed.

        Returns:
            The same string with ``\\n`` and ``\\t`` replaced by the characters they name.
        """
        return value.replace("\\n", "\n").replace("\\t", "\t")

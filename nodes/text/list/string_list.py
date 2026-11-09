"""A list of strings typed one per line."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log

logger = log.get_logger("nodes.text.list")


class StringList(io.ComfyNode):
    """One entry per line, handed on as a ``STRING`` list."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASStringList",
            display_name="String List",
            search_aliases=[
                "WASStringList", "String List",
                "prompt list",
                "prompts per line",
                "batch prompts",
                "lines to list",
                "multiline list",
                "list of strings",
            ],
            category="WAS Suite/Text/List",
            description=(
                "Type one prompt per line and hand them on as a STRING list, so everything "
                "downstream runs once per line. Each line can be switched off without deleting "
                "it, and # or // lines are notes for organising the list. A STRING wired into "
                "text replaces the typed lines."
            ),
            inputs=[
                io.String.Input(
                    "lines",
                    multiline=True,
                    default="",
                    dynamic_prompts=False,
                    placeholder="One prompt per line",
                    tooltip=(
                        "One entry per line, as `a red fox in snow`. A line starting with ~ is "
                        "switched off; # and // start a note, and /* */ wraps several lines of "
                        "notes. Ignored while a STRING is wired into text."
                    ),
                ),
                io.Boolean.Input(
                    "skip_blank_lines",
                    default=True,
                    tooltip=(
                        "`on` = empty lines are left out; `off` = each empty line is an empty "
                        "entry, for a list whose positions have to line up with another list's."
                    ),
                ),
                io.Boolean.Input(
                    "skip_comment_lines",
                    default=True,
                    tooltip=(
                        "`on` = notes (#, // and /* */ lines) are left out; `off` = each note "
                        "line is handed on as written, marker included."
                    ),
                ),
                io.String.Input(
                    "text",
                    optional=True,
                    force_input=True,
                    tooltip=(
                        "A STRING to use instead of the typed lines, such as `a fox\\na heron`, "
                        "read the same way: one entry per line, ~ lines off, # and // notes. "
                        "Unwired, the typed lines are used."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="strings",
                    is_output_list=True,
                    tooltip=(
                        "One entry per line as a STRING list: a node reading it runs once per "
                        "entry, so wired into a prompt it renders each line in turn. A list "
                        "with nothing in it stops the prompt."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many entries the list holds.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls, lines="", skip_blank_lines=True, skip_comment_lines=True, text=None
    ) -> io.NodeOutput:
        from ....modules.compat.lists import require_values
        from ....modules.prompt.prompt_lines import list_lines

        source = lines if text is None else text
        entries = list_lines(source, skip_blank_lines, skip_comment_lines)
        logger.debug("String List handed on %d entries", len(entries))
        require_values(
            entries,
            "String List has no lines to hand on, so nothing below it can run. Every line is "
            "switched off, a note or blank, or the text wired in is empty. Switch a line on, or "
            "turn skip_blank_lines or skip_comment_lines off to hand those lines on.",
        )
        return io.NodeOutput(entries, len(entries))

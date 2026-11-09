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
                "dynamic prompts",
                "prompt variations",
            ],
            category="WAS Suite/Text/List",
            description=(
                "Type one prompt per line and hand them on as a STRING list, so everything "
                "downstream runs once per line. Lines can be switched off, kept as notes, grouped "
                "under == headings, given versions that take turns between runs, and use $terms "
                "and {a|b} choices."
            ),
            inputs=[
                io.String.Input(
                    "lines",
                    multiline=True,
                    default="",
                    dynamic_prompts=False,
                    placeholder="One prompt per line",
                    tooltip=(
                        "One entry per line. `~` switches a line off, `#` starts a note, `== name` "
                        "a heading, `| text` a version of the line above, `$name = value` a term; "
                        "`{a|b}` draws one option. Ignored while a STRING is wired into text."
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
                        "read the same way. Unwired, the typed lines are used."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFF,
                    control_after_generate=True,
                    advanced=True,
                    tooltip=(
                        "Which option each `{a|b}` choice draws: the same seed and lines give the "
                        "same entries. A new seed is drawn after a run only while a line holds a "
                        "choice."
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
        cls, lines="", skip_blank_lines=True, skip_comment_lines=True, text=None, seed=0
    ) -> io.NodeOutput:
        from ....modules.compat.lists import require_values
        from ....modules.prompt.prompt_lines import list_lines

        source = lines if text is None else text
        entries = list_lines(source, skip_blank_lines, skip_comment_lines, seed)
        logger.debug("String List handed on %d entries", len(entries))
        require_values(
            entries,
            "String List has no lines to hand on, so nothing below it can run. Every line is "
            "switched off, a note, a term, a heading or blank, or the text wired in is empty. "
            "Switch a line on, or turn skip_blank_lines or skip_comment_lines off to hand those "
            "lines on.",
        )
        return io.NodeOutput(entries, len(entries))

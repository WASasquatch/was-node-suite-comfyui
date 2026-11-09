"""Several lists of strings, each typed in its own box, put together into one list."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log

logger = log.get_logger("nodes.text.list")

#: One empty list, what a new node starts with.
FIRST_LIST = '[{"name": "List 1", "mode": "all", "pick": 1, "on": true, "lines": ""}]'


class StringListSections(io.ComfyNode):
    """Lists combined line by line, or as every combination, into one ``STRING`` list."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASStringListSections",
            display_name="String List (Sections)",
            search_aliases=[
                "WASStringListSections", "String List (Sections)",
                "prompt sections",
                "combine lists",
                "zip lists",
                "style prompt background",
                "prompt matrix",
                "every combination",
                "sublists",
            ],
            category="WAS Suite/Text/List",
            description=(
                "Type or paste several lists, each in its own box, such as a style, the prompts "
                "and the backgrounds, and hand on one STRING per entry with a line from each list "
                "joined. A shorter list starts over, and a list can give one fixed, stepped or "
                "random line every run."
            ),
            inputs=[
                io.String.Input(
                    "lists",
                    multiline=True,
                    default=FIRST_LIST,
                    dynamic_prompts=False,
                    tooltip=(
                        "The lists as JSON, each `{\"name\": \"Style\", \"mode\": \"all\", "
                        "\"pick\": 1, \"on\": true, \"lines\": \"watercolour\\nink\"}`; mode is "
                        "`all`, `fixed`, `increment`, `decrement` or `random`. Plain text is read "
                        "as one list."
                    ),
                ),
                io.Combo.Input(
                    "combine",
                    options=["cycle", "every combination"],
                    default="cycle",
                    tooltip=(
                        "`cycle` = entry 1 takes line 1 of each list, a shorter list starting "
                        "over, as many entries as the longest list; `every combination` = each "
                        "line with every line of the other lists, up to `10000` entries."
                    ),
                ),
                io.Combo.Input(
                    "join_with",
                    options=["comma", "space", "line break"],
                    default="comma",
                    tooltip=(
                        "How one entry's lines are joined: `comma` = `watercolour, a fox, snow`; "
                        "`space` = one line with spaces; `line break` = one line each."
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
                        "Which option each `{a|b}` choice draws: the same seed and lists give the "
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
                        "One entry per line of the longest list, or per combination, as a STRING "
                        "list: a node reading it runs once per entry. A list with nothing in it "
                        "stops the prompt."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many entries the list holds.",
                ),
            ],
        )

    @classmethod
    def execute(cls, lists=FIRST_LIST, combine="cycle", join_with="comma", seed=0) -> io.NodeOutput:
        from ....modules.compat.lists import require_values
        from ....modules.compat.tokens import expand
        from ....modules.prompt.prompt_lines import combine as combine_lists
        from ....modules.prompt.prompt_lines import list_entries, read_lists

        found = read_lists(lists)
        for entry in found:
            entry["lines"] = expand({"lines": entry["lines"]}, ("lines",))["lines"]
        entries = combine_lists(list_entries(found, seed), combine, join_with)
        logger.debug(
            "String List (Sections) put %d list(s) together into %d entries", len(found), len(entries)
        )
        require_values(
            entries,
            "String List (Sections) has no lines to hand on, so nothing below it can run. Every "
            "list is empty or switched off, or every line in them is off, a note or a term. Type "
            "a line into a list, or switch a list on.",
        )
        return io.NodeOutput(entries, len(entries))

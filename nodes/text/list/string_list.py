"""A list of strings typed one per line, in one list or several."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log

logger = log.get_logger("nodes.text.list")

#: The socket that feeds each list in place of its typed lines: ``text`` for the first, then
#: ``text_2`` onward, one per list a node can hold.
TEXT_SOCKETS = ("text",) + tuple(f"text_{number}" for number in range(2, 17))

LIST_SOCKET_HINT = (
    "A STRING used instead of one added list's typed lines, `text_2` for list 2 and on, such as "
    "`a fox\\na heron`, read the same way. Shown once that list is added."
)


class StringList(io.ComfyNode):
    """One entry per line, handed on as a ``STRING`` list."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASStringList",
            display_name="Multiline Text Lists",
            search_aliases=[
                "WASStringList", "Multiline Text Lists", "String List",
                "prompt list",
                "prompts per line",
                "batch prompts",
                "lines to list",
                "multiline list",
                "list of strings",
                "dynamic prompts",
                "prompt variations",
                "combine lists",
                "sublists",
            ],
            category="WAS Suite/Text/List",
            description=(
                "Type one prompt per line and hand them on as a STRING list, so everything "
                "downstream runs once per line. Lines can be switched off, kept as notes, given "
                "versions and use $terms and {a|b} choices; added lists hand on after the first, "
                "or everything joins into single strings."
            ),
            inputs=[
                io.String.Input(
                    "lines",
                    multiline=True,
                    default="",
                    dynamic_prompts=False,
                    placeholder="One prompt per line",
                    tooltip=(
                        "One entry per line: `~` switches a line off, `#` starts a note, `| text` "
                        "adds a version, `$name = value` a term, `{a|b}` one option. Several lists "
                        "are kept as JSON. Ignored while text is wired."
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
                        "A STRING used instead of the first list's typed lines, such as "
                        "`a fox\\na heron`, read the same way. Unwired, the typed lines are used."
                    ),
                ),
                io.String.Input("text_2", optional=True, force_input=True, tooltip=LIST_SOCKET_HINT),
                io.String.Input("text_3", optional=True, force_input=True, tooltip=LIST_SOCKET_HINT),
                io.String.Input("text_4", optional=True, force_input=True, tooltip=LIST_SOCKET_HINT),
                io.String.Input("text_5", optional=True, force_input=True, tooltip=LIST_SOCKET_HINT),
                io.String.Input("text_6", optional=True, force_input=True, tooltip=LIST_SOCKET_HINT),
                io.String.Input("text_7", optional=True, force_input=True, tooltip=LIST_SOCKET_HINT),
                io.String.Input("text_8", optional=True, force_input=True, tooltip=LIST_SOCKET_HINT),
                io.String.Input("text_9", optional=True, force_input=True, tooltip=LIST_SOCKET_HINT),
                io.String.Input("text_10", optional=True, force_input=True, tooltip=LIST_SOCKET_HINT),
                io.String.Input("text_11", optional=True, force_input=True, tooltip=LIST_SOCKET_HINT),
                io.String.Input("text_12", optional=True, force_input=True, tooltip=LIST_SOCKET_HINT),
                io.String.Input("text_13", optional=True, force_input=True, tooltip=LIST_SOCKET_HINT),
                io.String.Input("text_14", optional=True, force_input=True, tooltip=LIST_SOCKET_HINT),
                io.String.Input("text_15", optional=True, force_input=True, tooltip=LIST_SOCKET_HINT),
                io.String.Input("text_16", optional=True, force_input=True, tooltip=LIST_SOCKET_HINT),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFF,
                    control_after_generate=io.ControlAfterGenerate.fixed,
                    advanced=True,
                    tooltip=(
                        "Which option each `{a|b}` choice draws: the same seed and lines give the "
                        "same entries. The control starts `fixed`; `randomize` draws again after "
                        "each run while a line holds a choice."
                    ),
                ),
                io.Combo.Input(
                    "combine",
                    options=["concatenate", "precompile", "every combination"],
                    default="concatenate",
                    tooltip=(
                        "`concatenate` = one entry per line, the lists one after another; "
                        "`precompile` = every line joined into one string, `num_items` of them; "
                        "`every combination` = each line joined with every line of the other "
                        "lists, up to `10000`."
                    ),
                ),
                io.Combo.Input(
                    "join_with",
                    options=["comma", "space", "line break"],
                    default="comma",
                    tooltip=(
                        "How `precompile` and `every combination` join lines: `comma` = "
                        "`watercolour, a fox, snow`; `space` = one line with spaces; "
                        "`line break` = one line each."
                    ),
                ),
                io.Int.Input(
                    "num_items",
                    default=1,
                    min=1,
                    max=1000,
                    tooltip=(
                        "With `precompile`, how many strings to make, as a list: `10` = ten "
                        "strings, each a turn on, so stepped and random lists and `{a|b}` choices "
                        "change between them."
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
        cls,
        lines="",
        skip_blank_lines=True,
        skip_comment_lines=True,
        text=None,
        seed=0,
        combine="concatenate",
        join_with="comma",
        num_items=1,
        **wired_lists,
    ) -> io.NodeOutput:
        from ....modules.compat.lists import require_values
        from ....modules.compat.tokens import expand
        from ....modules.prompt.prompt_lines import PRECOMPILE, list_entries, precompiled, read_lists
        from ....modules.prompt.prompt_lines import combine as combine_lists

        found = read_lists(lines)
        for entry in found:
            entry["lines"] = expand({"lines": entry["lines"]}, ("lines",))["lines"]
        wired = {"text": text, **wired_lists}
        for index, socket in enumerate(TEXT_SOCKETS):
            value = wired.get(socket)
            if value is None:
                continue
            while len(found) <= index:
                found.append({"name": "", "mode": "all", "pick": 1, "start": 1, "on": True, "lines": ""})
            found[index]["lines"] = value
        if combine == PRECOMPILE:
            entries = precompiled(found, seed, num_items, skip_blank_lines, skip_comment_lines, join_with)
        else:
            entries = combine_lists(
                list_entries(found, seed, skip_blank_lines, skip_comment_lines), combine, join_with
            )
        logger.debug("String List handed on %d entries from %d list(s)", len(entries), len(found))
        require_values(
            entries,
            "String List has no lines to hand on, so nothing below it can run. Every line is "
            "switched off, a note, a term, a heading or blank, every list is off, or the text "
            "wired in is empty. Switch a line on, or turn skip_blank_lines or skip_comment_lines "
            "off to hand those lines on.",
        )
        return io.NodeOutput(entries, len(entries))

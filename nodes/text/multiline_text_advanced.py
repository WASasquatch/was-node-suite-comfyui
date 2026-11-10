"""A multiline text built from lines that can be switched off, versioned or kept as notes."""

from __future__ import annotations

from comfy_api.latest import io


class MultilineTextAdvanced(io.ComfyNode):
    """One text from the lines that are kept, joined."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASMultilineTextAdvanced",
            display_name="Multiline Text (Advanced)",
            search_aliases=[
                "WASMultilineTextAdvanced", "Multiline Text (Advanced)",
                "prompt builder",
                "optional prompt parts",
                "prompt with comments",
                "text box with comments",
                "toggle lines",
                "dynamic prompts",
                "prompt variables",
            ],
            category="WAS Suite/Text",
            description=(
                "Write a prompt one part per line and hand on the kept lines as one text, joined "
                "by line breaks, spaces or commas. Lines can be switched off, kept as notes, "
                "given versions that take turns between runs, and use $terms and {a|b} choices."
            ),
            inputs=[
                io.String.Input(
                    "lines",
                    multiline=True,
                    default="",
                    dynamic_prompts=False,
                    placeholder="One part per line",
                    tooltip=(
                        "One part per line. `~` switches a line off, `#` starts a note, `== name` "
                        "a heading, `| text` a version of the line above, `$name = value` a term; "
                        "`{a|b}` draws one option. Ignored while a STRING is wired into text."
                    ),
                ),
                io.Boolean.Input(
                    "skip_blank_lines",
                    default=False,
                    tooltip=(
                        "`off` = empty lines stay as paragraph breaks when joined by line "
                        "breaks; `on` = they are left out. Joined by spaces or commas, empty "
                        "lines are always left out."
                    ),
                ),
                io.Boolean.Input(
                    "skip_comment_lines",
                    default=True,
                    tooltip=(
                        "`on` = notes (#, // and /* */ lines) are left out; `off` = each note "
                        "line is kept as written, marker included."
                    ),
                ),
                io.Combo.Input(
                    "join_with",
                    options=["line break", "space", "comma"],
                    default="line break",
                    tooltip=(
                        "`line break` = lines as written, indents kept; `space` = one line of "
                        "text; `comma` = a tag prompt, `red fox,` and `snow` giving "
                        "`red fox, snow`."
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
                    control_after_generate=io.ControlAfterGenerate.fixed,
                    advanced=True,
                    tooltip=(
                        "Which option each `{a|b}` choice draws: the same seed and lines give the "
                        "same text. The control starts `fixed`; `randomize` draws again after each "
                        "run while a line holds a choice."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    tooltip="The kept lines as one text, empty when every line is off or a note.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        lines="",
        skip_blank_lines=False,
        skip_comment_lines=True,
        join_with="line break",
        text=None,
        seed=0,
    ) -> io.NodeOutput:
        from ...modules.prompt.prompt_lines import join_lines

        source = lines if text is None else text
        return io.NodeOutput(join_lines(source, skip_blank_lines, skip_comment_lines, join_with, seed))

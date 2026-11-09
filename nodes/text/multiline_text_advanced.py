"""A multiline text built from lines that can be switched off or kept as notes."""

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
            ],
            category="WAS Suite/Text",
            description=(
                "Write a prompt one part per line and hand on the lines that are kept as one "
                "text, joined by line breaks, spaces or commas. Each line can be switched off "
                "without deleting it, and # or // lines are notes for organising the prompt."
            ),
            inputs=[
                io.String.Input(
                    "lines",
                    multiline=True,
                    default="",
                    dynamic_prompts=False,
                    placeholder="One part per line",
                    tooltip=(
                        "One part per line, as `a red fox in snow`. A line starting with ~ is "
                        "switched off; # and // start a note, and /* */ wraps several lines of "
                        "notes. Ignored while a STRING is wired into text."
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
                        "read the same way: one part per line, ~ lines off, # and // notes. "
                        "Unwired, the typed lines are used."
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
        cls, lines="", skip_blank_lines=False, skip_comment_lines=True, join_with="line break", text=None
    ) -> io.NodeOutput:
        from ...modules.prompt.prompt_lines import join_lines

        source = lines if text is None else text
        return io.NodeOutput(join_lines(source, skip_blank_lines, skip_comment_lines, join_with))

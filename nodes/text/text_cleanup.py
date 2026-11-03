"""Tidy a caption or prompt: letter case, spacing, line breaks, punctuation, ending and quotes."""

from __future__ import annotations

from comfy_api.latest import io, ui

from ...modules.prompt import cleanup


class TextCleanup(io.ComfyNode):
    """Tidy the case, spacing, punctuation, ending and quotes of one text."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASTextCleanup",
            display_name="Text Cleanup",
            search_aliases=[
                "WASTextCleanup",
                "Text Cleanup",
                "tidy text",
                "clean caption",
                "sentence case",
                "title case",
                "fix punctuation",
                "straight quotes",
            ],
            category="WAS Suite/Text/Operations",
            description=(
                "Tidy a caption or prompt: change its letter case, collapse stray spaces and "
                "line breaks, fix the spacing around punctuation, settle how it ends and "
                "straighten curly quotes. Bracketed tokens such as [trigger] keep their case, "
                "and with every setting off the text passes through untouched."
            ),
            is_output_node=True,
            inputs=[
                io.String.Input(
                    "text",
                    default="",
                    multiline=True,
                    placeholder="a woman ,stands  on a beach,,",
                    tooltip=(
                        "The caption or prompt to tidy, as `a woman ,stands  on a beach,,`. "
                        "Wired from a STRING list, the node runs once per text."
                    ),
                ),
                io.Combo.Input(
                    "case",
                    options=list(cleanup.CASES),
                    default="keep",
                    tooltip=(
                        "keep = as written; lower; upper; sentence = a capital starts each "
                        "sentence; title = a capital starts each word. Bracketed tokens such "
                        "as `[trigger]` keep their case, and sentence and title leave words "
                        "like NASA or iPhone alone."
                    ),
                ),
                io.Boolean.Input(
                    "whitespace",
                    default=True,
                    tooltip=(
                        "true = runs of spaces and tabs become one space and every line is "
                        "trimmed, as `a  woman ` to `a woman`; false = spacing as written."
                    ),
                ),
                io.Combo.Input(
                    "line_breaks",
                    options=list(cleanup.LINE_BREAKS),
                    default="keep",
                    tooltip=(
                        "`keep` = lines as written; `one line` = every line joined with a "
                        "space and blank lines dropped, for a caption file or an encoder that "
                        "wants a single line."
                    ),
                ),
                io.Boolean.Input(
                    "punctuation",
                    default=True,
                    tooltip=(
                        "true = no space before , . ; : ! ?, one space after , ; : before a "
                        "word, doubled commas and `. .` collapsed, and separators at either "
                        "end of a line removed; false = punctuation as written."
                    ),
                ),
                io.Combo.Input(
                    "ending",
                    options=list(cleanup.ENDINGS),
                    default="keep",
                    tooltip=(
                        "`keep` = as written; `period` = ends with a full stop unless it "
                        "already ends in . ! or ?; `none` = closing full stops, commas and "
                        "semicolons removed."
                    ),
                ),
                io.Boolean.Input(
                    "quotes",
                    default=False,
                    tooltip=(
                        "true = curly quotes and apostrophes become straight ones, as "
                        "“it’s” to \"it's\"; false = kept as written."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="text",
                    tooltip="The tidied text.",
                ),
                io.Boolean.Output(
                    display_name="changed",
                    tooltip=(
                        "true when a setting changed the text; false when it came through "
                        "exactly as it went in."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        text,
        case="keep",
        whitespace=True,
        line_breaks="keep",
        punctuation=True,
        ending="keep",
        quotes=False,
    ) -> io.NodeOutput:
        result = cleanup.clean(
            text,
            case=case,
            whitespace=whitespace,
            line_breaks=line_breaks,
            punctuation=punctuation,
            ending=ending,
            quotes=quotes,
        )
        return io.NodeOutput(result, result != text, ui=ui.PreviewText(result))

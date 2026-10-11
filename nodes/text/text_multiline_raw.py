"""A multiline text box that leaves its text exactly as typed."""

from __future__ import annotations

from comfy_api.latest import io


class TextMultilineRaw(io.ComfyNode):
    """A text box with dynamic prompts off and no comment stripping."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text Multiline (Code Compatible)",
            display_name="Text Multiline (Code Compatible)",
            search_aliases=["Text Multiline (Code Compatible)", "raw text", "code text"],
            category="WAS Suite/Text",
            description=(
                "A multiline text box for code, JSON, YAML or anything else that has to "
                "arrive exactly as typed. Every line is kept, including one starting with "
                "'#', and a {red|blue} alternation is passed through as written."
            ),
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    dynamic_prompts=False,
                    tooltip=(
                        "The text to emit, kept exactly as typed. A line starting with `#` "
                        "survives and a `{red|blue}` alternation is passed through literally, "
                        "which is what makes this box safe for code and for JSON. Tokens such "
                        "as [time] and [user] are still replaced."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    tooltip="The text as typed, with only its tokens replaced.",
                ),
            ],
        )

    @classmethod
    def execute(cls, text) -> io.NodeOutput:
        return io.NodeOutput(text)

"""A multiline text box with comment stripping and token substitution."""

from __future__ import annotations

from comfy_api.latest import io



class TextMultiline(io.ComfyNode):
    """A prompt-sized text box whose lines may be commented out."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text Multiline",
            display_name="Text Multiline",
            search_aliases=["Text Multiline", "prompt", "text box"],
            category="WAS Suite/Text",
            description=(
                "A multiline text box. Lines starting with # are dropped, tokens such as "
                "[time] and [user] are substituted, and a {red|blue} alternation picks one "
                "option at random unless dynamic_prompts is off. Text whose # lines must "
                "survive, such as code, belongs in Text Multiline (Code Compatible)."
            ),
            inputs=[
                io.String.Input(
                    "text",
                    multiline=True,
                    default="",
                    dynamic_prompts=True,
                    tooltip=(
                        "The text to emit. A line whose first non-blank character is # is "
                        "left out, so part of a prompt can be parked; tokens such as [time], "
                        "[user] and [hostname] are replaced; {red|blue} picks one option at "
                        "random unless dynamic_prompts is off."
                    ),
                ),
                io.Boolean.Input(
                    "dynamic_prompts",
                    default=True,
                    tooltip=(
                        "Whether a {red|blue} alternation picks one option at random. `on` = "
                        "prompt behaviour; `off` = braces kept as typed, as JSON needs. # "
                        "lines are dropped either way; code belongs in Text Multiline (Code "
                        "Compatible). Text sent straight to the API is never rewritten."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    tooltip="The text with # lines removed and every token replaced.",
                ),
            ],
        )

    @classmethod
    def execute(cls, text, dynamic_prompts=True) -> io.NodeOutput:
        # StringIO rather than splitlines(): it splits on \n, \r\n and \r only, where
        # splitlines() also breaks on form feed and the unicode line separators, which
        # would cut a line a prompt box treats as one.
        from io import StringIO

        kept = []
        for line in StringIO(text):
            if not line.strip().startswith("#"):
                kept.append(line.replace("\n", ""))
        return io.NodeOutput("\n".join(kept))

"""Define custom ``[token]`` substitutions from a block of ``name: value`` lines."""

from __future__ import annotations

import json

from comfy_api.latest import io, ui

from ....modules import log

logger = log.get_logger("nodes.text.tokens")


class TextAddTokens(io.ComfyNode):
    """Store one custom token per line of the ``tokens`` widget."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text Add Tokens",
            display_name="Text Add Tokens",
            search_aliases=["Text Add Tokens", "tokens", "variables", "define"],
            category="WAS Suite/Text/Tokens",
            description=(
                "Define custom tokens, one 'name: value' pair per line, for Text Parse "
                "Tokens and every other node that expands [tokens]. Only the first colon "
                "splits a line, and a line without one is skipped. The name is matched "
                "exactly as written, so the square brackets are a convention, not a "
                "requirement."
            ),
            inputs=[
                io.String.Input(
                    "tokens",
                    default="[hello]: world",
                    multiline=True,
                    tooltip=(
                        "One token per line, written as 'name: value', so '[hello]: world' "
                        "turns [hello] into world in every later node that expands tokens."
                    ),
                ),
                io.Boolean.Input(
                    "print_current_tokens",
                    default=False,
                    tooltip=(
                        "`on` logs every custom token now defined and shows them on the node, "
                        "which is how to check what a previous run left behind; `off` "
                        "stores them silently."
                    ),
                ),
            ],
            outputs=[],
            is_output_node=True,
            not_idempotent=True,
        )

    @classmethod
    def execute(cls, tokens, print_current_tokens=False) -> io.NodeOutput:
        from ....modules.state.tokens import TextTokens

        tk = TextTokens()

        for line in tokens.splitlines():
            parts = line.split(":", 1)
            if len(parts) < 2:
                # A line carrying no colon, which is what a blank line in the widget is.
                continue
            tk.addToken(parts[0].strip(), parts[1].strip())

        if print_current_tokens:
            dump = json.dumps(tk.custom_tokens, indent=4)
            logger.info("current custom tokens:\n%s", dump)
            return io.NodeOutput(ui=ui.PreviewText(dump))

        return io.NodeOutput()

"""Define one custom ``[token]`` from linked name and value strings."""

from __future__ import annotations

import json

from comfy_api.latest import io, ui

from ....modules import log

logger = log.get_logger("nodes.text.tokens")


class TextAddTokenByInput(io.ComfyNode):
    """Store one custom token whose name and value both arrive as links."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text Add Token by Input",
            display_name="Text Add Token by Input",
            search_aliases=["Text Add Token by Input", "tokens", "variables", "define"],
            category="WAS Suite/Text/Tokens",
            description=(
                "Define a single custom token from a linked name and a linked value, for "
                "Text Parse Tokens and every other node that expands [tokens]."
            ),
            inputs=[
                io.String.Input(
                    "token_name",
                    multiline=True,
                    placeholder="Eg: season",
                    tooltip=(
                        "Token name, no brackets; STRING. Expands as [name] in every "
                        "text node. Empty adds nothing. Eg: season"
                    ),
                ),
                io.String.Input(
                    "token_value",
                    multiline=True,
                    placeholder="Eg: late autumn",
                    tooltip=(
                        "What the token expands to; STRING. Empty expands to nothing. "
                        "Eg: late autumn"
                    ),
                ),
                io.Boolean.Input(
                    "print_current_tokens",
                    default=False,
                    tooltip=(
                        "`on` shows every custom token now defined on the node, with its "
                        "value, and logs their names only, which is how to check what a "
                        "previous run left behind; `off` stores the token silently."
                    ),
                ),
            ],
            outputs=[],
            is_output_node=True,
            not_idempotent=True,
        )

    @classmethod
    def execute(cls, token_name, token_value, print_current_tokens=False) -> io.NodeOutput:
        from ....modules.state.tokens import TextTokens

        if token_name.strip() == "":
            logger.error(
                "a token_name is required for a token; the token name provided is empty, so "
                "no token was added"
            )
            return io.NodeOutput()

        # The widget asks for a bare name and promises it expands as [name], so the
        # brackets are put on here. A legacy entry stored without them is dropped, since it
        # would go on matching the bare word anywhere in a prompt.
        name = token_name.strip()
        bracketed = name if name.startswith("[") and name.endswith("]") else f"[{name}]"

        tk = TextTokens()
        if bracketed != name and name in tk.custom_tokens:
            tk.removeToken(name)
        tk.addToken(bracketed, token_value)

        if print_current_tokens:
            logger.info("custom tokens defined: %s", ", ".join(tk.custom_tokens) or "none")
            return io.NodeOutput(ui=ui.PreviewText(json.dumps(tk.custom_tokens, indent=4)))

        return io.NodeOutput()

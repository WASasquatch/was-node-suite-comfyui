"""Count the tokens in a text, by a model's own tokenizer where it has one."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM
from ...modules.interface import run_result
from ...modules.llm import generate, runtime, wires

REQUIRES = "llm"

logger = log.get_logger("llm.generate")

#: The node's name in messages.
NODE = "LLM Token Count"

#: The nodes that make a language model, for messages.
LLM_HINT = "LLM From CLIP, LLM GGUF Loader or LLM OpenAI"


class LLMTokenCount(io.ComfyNode):
    """How many tokens a text costs a model."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMTokenCount",
            display_name="LLM Token Count",
            search_aliases=[
                "WASLLMTokenCount",
                "LLM Token Count",
                "count tokens",
                "tokenizer",
                "tiktoken",
                "context length",
            ],
            category="WAS Suite/LLM/Generate",
            is_output_node=True,
            description=(
                "Count how many tokens a text costs a language model, such as to check that a "
                "document fits before sending it. The panel names the tokenizer that counted and "
                "how much of the model's context window the text fills."
            ),
            inputs=[
                io.String.Input(
                    "text", multiline=True, default="",
                    tooltip=(
                        "The text to count, typed here or wired in, such as `A lighthouse in a "
                        "storm, oil painting` or a whole document."
                    ),
                ),
                LLM.Input(
                    "llm", optional=True,
                    tooltip=(
                        "The model to count for. A model on this machine counts with its own "
                        "tokenizer; a hosted one, or none, counts with a bundled GPT tokenizer, "
                        "which is close for most models."
                    ),
                ),
            ],
            outputs=[
                io.Int.Output(
                    display_name="count",
                    tooltip="Tokens in the text. A token is about three quarters of an English word.",
                ),
            ],
        )

    @classmethod
    def execute(cls, text, llm=None) -> io.NodeOutput:
        """Count the tokens.

        Args:
            text: The text to count.
            llm: The ``LanguageModel`` to count for, or ``None``.

        Returns:
            The count.

        Raises:
            ValueError: The input is not a model, or the text is too long.
            RuntimeError: No tokenizer is available.
        """
        model = None if llm is None else wires.expect(llm, wires.LanguageModel, NODE, "llm", LLM_HINT)
        text = str(text or "")
        with runtime.guarded():
            count, tokenizer, note = generate.count_tokens(model if model is not None and model.local else None, text)
        if model is not None and not model.local:
            note = "a hosted model is counted on this machine, with the bundled tokenizer"
        facts = {"tokenizer": tokenizer}
        counts = {"tokens": count, "characters": len(text)}
        summary = f"{count:,} token{'s' if count != 1 else ''}, counted by {tokenizer}."
        if model is not None:
            facts["model"] = model.label
            window = int(model.context_window or 0)
            if window > 0:
                share = count / window
                counts["context window"] = window
                counts["percent of context"] = round(100.0 * share, 2)
                summary = f"{count:,} tokens, {100.0 * share:.1f}% of {model.label}'s {window:,}-token context."
        if note:
            facts["note"] = note
        logger.debug("%s counted %d tokens with %s", NODE, count, tokenizer)
        over = model is not None and model.context_window and count > model.context_window
        run_result.publish(
            status=run_result.WARNING if over else run_result.OK,
            summary=summary,
            counts=counts,
            facts=facts,
        )
        return io.NodeOutput(count)

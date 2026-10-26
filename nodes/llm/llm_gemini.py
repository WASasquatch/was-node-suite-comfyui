"""Pick a Google Gemini model for the LLM nodes."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import LLM
from ...modules.llm import providers, runtime

REQUIRES = "llm"

#: The node's name in messages.
NODE = "LLM Google Gemini"

#: The most tokens one Gemini answer may be held to.
MAX_OUTPUT = 65536

#: The largest context window the node takes.
MAX_CONTEXT = 4194304


class LLMGemini(io.ComfyNode):
    """A Gemini chat model on the LLM wire."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMGemini",
            display_name="LLM Google Gemini",
            search_aliases=[
                "WASLLMGemini",
                "LLM Google Gemini",
                "Gemini",
                "Google",
                "GoogleGenAI",
                "hosted llm",
                "api model",
            ],
            category="WAS Suite/LLM/Models",
            description=(
                "Choose a Google Gemini model such as gemini-3.8-flash for the LLM nodes to write "
                "with. Prompts go to Google's servers, so features.network must be on and "
                "GOOGLE_API_KEY or GEMINI_API_KEY set where ComfyUI starts. The panel shows the "
                "model and what it can do."
            ),
            inputs=[
                io.String.Input(
                    "model", default="gemini-3.8-flash",
                    tooltip=(
                        "The Gemini model id: `gemini-4-argon` = flagship; `gemini-3.8-flash` = "
                        "fast; `gemini-3.5-flash-lite` = cheapest; `gemini-3.1-pro-preview` also "
                        "works."
                    ),
                ),
                io.Combo.Input(
                    "thinking_level", options=providers.GEMINI_THINKING, default="default",
                    tooltip=(
                        "How much Gemini thinks before answering: `default` = the model's own; "
                        "`minimal` = least and fastest; `high` = most and slowest."
                    ),
                ),
                io.Boolean.Input(
                    "show_thinking", default=True,
                    tooltip=(
                        "`true` = summaries of Gemini's thinking come back on the thinking "
                        "output; `false` = the answer alone."
                    ),
                ),
                io.Int.Input(
                    "max_tokens", default=16000, min=1, max=MAX_OUTPUT,
                    tooltip=(
                        "The most tokens one answer may take, thinking included. `16000` suits "
                        "most prompts."
                    ),
                ),
                io.Float.Input(
                    "temperature", default=1.0, min=0.0, max=2.0, step=0.05,
                    tooltip=(
                        "`1.0` = the model's default, which Google recommends for Gemini 3 and "
                        "later; `0.3` = steadier; `1.5` = looser."
                    ),
                ),
                io.Int.Input(
                    "timeout", default=600, min=5, max=3600, optional=True, advanced=True,
                    tooltip=(
                        "Seconds one request may run before it fails: `600` = ten minutes. Raise "
                        "it for long answers at a high thinking level."
                    ),
                ),
                io.Int.Input(
                    "max_retries", default=3, min=0, max=10, optional=True, advanced=True,
                    tooltip=(
                        "Times a failed or rate-limited request is tried again, waiting longer "
                        "each time. `0` = fail at once."
                    ),
                ),
                io.Int.Input(
                    "context_window", default=0, min=0, max=MAX_CONTEXT, optional=True,
                    advanced=True,
                    tooltip=(
                        "Tokens the model reads at once, used to fit documents and chat history. "
                        "`0` = 1048576, what current Gemini models read."
                    ),
                ),
            ],
            outputs=[
                LLM.Output(
                    display_name="llm",
                    tooltip="The model, for LLM Complete, LLM Chat, agents and the other LLM nodes.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls, model, thinking_level, show_thinking, max_tokens, temperature, timeout=600,
        max_retries=3, context_window=0,
    ) -> io.NodeOutput:
        """Describe the Gemini model the wire carries. No request is sent.

        Args:
            model: The model id.
            thinking_level: How much Gemini thinks first.
            show_thinking: Return the thinking summaries.
            max_tokens: Tokens one answer may take.
            temperature: Randomness, 1.0 for the model's default.
            timeout: Seconds one request may run.
            max_retries: Retries after a failed request.
            context_window: Tokens it reads at once, 0 for 1048576.

        Returns:
            The model.

        Raises:
            ValueError: The model id is not one.
            RuntimeError: ``features.network`` is off or no Gemini key is set.
        """
        with runtime.guarded():
            built = providers.gemini_model(
                model, thinking_level, bool(show_thinking), max_tokens, temperature, timeout,
                max_retries, context_window, node=NODE,
            )
        providers.publish(
            built, providers.GEMINI_HOST,
            {
                "thinking": f"{thinking_level}, {'shown' if show_thinking else 'hidden'}",
                "max tokens": f"{int(max_tokens):,}",
            },
        )
        return io.NodeOutput(built)

"""Pick an OpenAI chat model for the LLM nodes."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import LLM
from ...modules.llm import providers, runtime

REQUIRES = "llm"

#: The node's name in messages.
NODE = "LLM OpenAI"

#: The most tokens one answer may be held to.
MAX_OUTPUT = 128000

#: The largest context window the node takes.
MAX_CONTEXT = 4194304


class LLMOpenAI(io.ComfyNode):
    """An OpenAI chat model on the LLM wire."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMOpenAI",
            display_name="LLM OpenAI",
            search_aliases=[
                "WASLLMOpenAI",
                "LLM OpenAI",
                "OpenAI",
                "GPT",
                "ChatGPT",
                "gpt-6",
                "hosted llm",
                "api model",
            ],
            category="WAS Suite/LLM/Models",
            description=(
                "Choose an OpenAI model such as gpt-6.1-sol for the LLM nodes to write with. "
                "Prompts go to OpenAI's servers, so features.network must be on and OPENAI_API_KEY "
                "set where ComfyUI starts. The panel shows the model, its context window and what "
                "it can do."
            ),
            inputs=[
                io.String.Input(
                    "model", default="gpt-6.1-sol",
                    tooltip=(
                        "The OpenAI model id: `gpt-6-astra` = flagship; `gpt-6.1-sol` = balanced; "
                        "`gpt-6-luna` = low cost. Any id from OpenAI's models page works."
                    ),
                ),
                io.Combo.Input(
                    "reasoning_effort", options=providers.OPENAI_EFFORTS, default="default",
                    tooltip=(
                        "How long the model reasons before answering: `default` = the model's own; "
                        "`low` = quickest; `max` = deepest and slowest. `none` turns reasoning off, "
                        "on gpt-6-luna only."
                    ),
                ),
                io.Int.Input(
                    "max_tokens", default=16000, min=1, max=MAX_OUTPUT,
                    tooltip=(
                        "The most tokens one answer may take, reasoning included. `16000` suits "
                        "most prompts; raise it for long writing at a high effort."
                    ),
                ),
                io.Float.Input(
                    "temperature", default=1.0, min=0.0, max=2.0, step=0.05,
                    tooltip=(
                        "Randomness, for models that take it such as gpt-4.1: `0.2` = steady; "
                        "`1.0` = default; `1.5` = loose. Reasoning models such as gpt-6.1-sol "
                        "ignore it."
                    ),
                ),
                io.Int.Input(
                    "timeout", default=600, min=5, max=3600, optional=True, advanced=True,
                    tooltip=(
                        "Seconds one request may run before it fails: `600` = ten minutes. Raise "
                        "it for long answers at a high effort."
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
                        "`0` = the known figure for the model, or 128000 for a name not known yet."
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
        cls, model, reasoning_effort, max_tokens, temperature, timeout=600, max_retries=3,
        context_window=0,
    ) -> io.NodeOutput:
        """Describe the OpenAI model the wire carries. No request is sent.

        Args:
            model: The model id.
            reasoning_effort: How long the model reasons first.
            max_tokens: Tokens one answer may take.
            temperature: Randomness, for models that take it.
            timeout: Seconds one request may run.
            max_retries: Retries after a failed request.
            context_window: Tokens it reads at once, 0 for the known figure.

        Returns:
            The model.

        Raises:
            ValueError: The model id is not one.
            RuntimeError: ``features.network`` is off or ``OPENAI_API_KEY`` is not set.
        """
        with runtime.guarded():
            built = providers.openai_model(
                model, reasoning_effort, max_tokens, temperature, timeout, max_retries,
                context_window, node=NODE,
            )
        providers.publish(
            built, "api.openai.com",
            {"reasoning effort": reasoning_effort, "max tokens": f"{int(max_tokens):,}"},
        )
        return io.NodeOutput(built)

"""Pick a Claude model from Anthropic for the LLM nodes."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import LLM
from ...modules.llm import providers, runtime

REQUIRES = "llm"

#: The node's name in messages.
NODE = "LLM Anthropic"

#: The most tokens one answer may be held to.
MAX_OUTPUT = 128000

#: The largest context window the node takes.
MAX_CONTEXT = 4194304


class LLMAnthropic(io.ComfyNode):
    """A Claude chat model on the LLM wire."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMAnthropic",
            display_name="LLM Anthropic",
            search_aliases=[
                "WASLLMAnthropic",
                "LLM Anthropic",
                "Anthropic",
                "Claude",
                "claude-opus-5-5",
                "hosted llm",
                "api model",
            ],
            category="WAS Suite/LLM/Models",
            description=(
                "Choose a Claude model such as claude-opus-5-5 for the LLM nodes to write with. "
                "Prompts go to Anthropic's servers, so features.network must be on and "
                "ANTHROPIC_API_KEY set where ComfyUI starts. The panel shows the model, its "
                "context window and what it can do."
            ),
            inputs=[
                io.String.Input(
                    "model", default="claude-opus-5-5",
                    tooltip=(
                        "The Claude model id, with no date: `claude-opus-5-5` = most capable; "
                        "`claude-sonnet-5-5` = balanced; `claude-haiku-5-5` = fastest; "
                        "`claude-fable-5-1` also works."
                    ),
                ),
                io.Combo.Input(
                    "effort", options=providers.ANTHROPIC_EFFORTS, default="default",
                    tooltip=(
                        "How much effort Claude puts in: `default` = the model's own, medium on "
                        "Opus 5.5; `low` = quick and cheap; `max` = most thorough and slowest."
                    ),
                ),
                io.Combo.Input(
                    "thinking_display", options=providers.THINKING_DISPLAYS, default="summarized",
                    tooltip=(
                        "`summarized` = Claude thinks as it needs and the thinking output gets a "
                        "summary; `omitted` = the thinking stays hidden and answers start sooner; "
                        "`default` = no thinking setting, for older models such as "
                        "claude-haiku-4-5."
                    ),
                ),
                io.Int.Input(
                    "max_tokens", default=16000, min=1, max=MAX_OUTPUT,
                    tooltip=(
                        "The most tokens one answer may take, thinking included. `16000` suits "
                        "most prompts; raise it for long documents."
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
                        "Times a failed or overloaded request is tried again, waiting longer "
                        "each time. `0` = fail at once."
                    ),
                ),
                io.Int.Input(
                    "context_window", default=0, min=0, max=MAX_CONTEXT, optional=True,
                    advanced=True,
                    tooltip=(
                        "Tokens the model reads at once, used to fit documents and chat history. "
                        "`0` = the known figure for the model, or 200000 for a name not known yet."
                    ),
                ),
                io.Boolean.Input(
                    "cache_prompt", default=False, optional=True, advanced=True,
                    tooltip=(
                        "`true` asks Anthropic to cache each request, so the next call sharing "
                        "its long system prompt or documents within a few minutes is read "
                        "faster and costs less."
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
        cls, model, effort, thinking_display, max_tokens, timeout=600, max_retries=3,
        context_window=0, cache_prompt=False,
    ) -> io.NodeOutput:
        """Describe the Claude model the wire carries. No request is sent.

        Args:
            model: The model id.
            effort: How much effort Claude puts in.
            thinking_display: How the thinking comes back.
            max_tokens: Tokens one answer may take.
            timeout: Seconds one request may run.
            max_retries: Retries after a failed request.
            context_window: Tokens it reads at once, 0 for the known figure.
            cache_prompt: Mark each request for prompt caching.

        Returns:
            The model.

        Raises:
            ValueError: The model id is not one.
            RuntimeError: ``features.network`` is off.
        """
        with runtime.guarded():
            built = providers.anthropic_model(
                model, effort, thinking_display, max_tokens, timeout, max_retries,
                context_window, bool(cache_prompt), node=NODE,
            )
        providers.publish(
            built, providers.anthropic_host(built),
            {
                "effort": effort,
                "thinking": thinking_display,
                "prompt caching": "on" if cache_prompt else "off",
            },
        )
        return io.NodeOutput(built)

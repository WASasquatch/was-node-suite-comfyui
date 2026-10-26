"""Pick a model on another hosted service, a server on this machine, or a configured endpoint."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import LLM
from ...modules.llm import providers, runtime

REQUIRES = "llm"

#: The node's name in messages.
NODE = "LLM Provider"

#: The most tokens one answer may be held to.
MAX_OUTPUT = 128000

#: The largest context window the node takes.
MAX_CONTEXT = 4194304


class LLMProvider(io.ComfyNode):
    """A chat model on a built-in provider, a local server or an ``llm.endpoints`` entry."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMProvider",
            display_name="LLM Provider",
            search_aliases=[
                "WASLLMProvider",
                "LLM Provider",
                "OpenAILike",
                "Ollama",
                "LM Studio",
                "vLLM",
                "llama.cpp server",
                "Groq",
                "DeepSeek",
                "Mistral",
                "OpenRouter",
                "openai compatible",
            ],
            category="WAS Suite/LLM/Models",
            description=(
                "Choose a model on a hosted service such as Groq, DeepSeek or Mistral, a server on "
                "this machine such as Ollama or LM Studio, or an endpoint named in config.yaml. "
                "Hosted services need features.network on and their own key variable. The panel "
                "shows where prompts go."
            ),
            inputs=[
                io.Combo.Input(
                    "provider", options=providers.options(), default="ollama",
                    tooltip=(
                        "Where the model runs. `groq`, `deepseek`, `mistral` and the other "
                        "services each read their own key variable, such as GROQ_API_KEY; "
                        "`ollama`, `lmstudio`, `vllm` and `llama.cpp` are servers on this machine; "
                        "`endpoint: name` entries come from llm.endpoints in config.yaml."
                    ),
                ),
                io.String.Input(
                    "model", default="",
                    tooltip=(
                        "The model id the provider uses, such as `llama3.2:3b` on Ollama or "
                        "`openai/gpt-oss-120b` on Groq. Empty = the provider's default; a server on "
                        "this machine has none, so type the model it serves."
                    ),
                ),
                io.Float.Input(
                    "temperature", default=0.7, min=0.0, max=2.0, step=0.05,
                    tooltip="Randomness: `0.2` = steady and factual; `0.7` = balanced; `1.2` = loose.",
                ),
                io.Int.Input(
                    "max_tokens", default=4096, min=1, max=MAX_OUTPUT,
                    tooltip=(
                        "The most tokens one answer may take. `4096` is about 3000 words; a "
                        "reasoning model needs more, since its thinking counts too."
                    ),
                ),
                io.Int.Input(
                    "context_window", default=0, min=0, max=MAX_CONTEXT,
                    tooltip=(
                        "Tokens the model reads at once. `0` = the provider's usual figure, 8192 "
                        "for a server on this machine; set the model's real figure to fit longer "
                        "documents and chats."
                    ),
                ),
                io.Combo.Input(
                    "function_calling", options=providers.FUNCTION_CALLING, default="auto",
                    tooltip=(
                        "Whether agents call tools through the API: `auto` = what the provider "
                        "usually supports; `on` = always; `off` = agents write tool calls as text "
                        "instead, which any model can do."
                    ),
                ),
                io.Int.Input(
                    "timeout", default=600, min=5, max=3600, optional=True, advanced=True,
                    tooltip=(
                        "Seconds one request may run before it fails: `600` = ten minutes. A "
                        "server on this machine loading a large model may need longer at first."
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
        cls, provider, model, temperature, max_tokens, context_window, function_calling,
        timeout=600,
    ) -> io.NodeOutput:
        """Describe the provider's model the wire carries. No request is sent.

        Args:
            provider: A built-in provider or ``endpoint: <name>``.
            model: The model id, empty for the provider's default.
            temperature: Randomness.
            max_tokens: Tokens one answer may take.
            context_window: Tokens it reads at once, 0 for the provider's figure.
            function_calling: ``auto``, ``on`` or ``off``.
            timeout: Seconds one request may run.

        Returns:
            The model.

        Raises:
            ValueError: The provider or model id is not valid, or a local server has no model.
            RuntimeError: A hosted provider with ``features.network`` off, or its key unset.
        """
        with runtime.guarded():
            built, target = providers.provider_model(
                provider, model, temperature, max_tokens, context_window, function_calling,
                timeout, node=NODE,
            )
        providers.publish(
            built, target.host,
            {"provider": target.title, "max tokens": f"{int(max_tokens):,}"},
        )
        return io.NodeOutput(built)

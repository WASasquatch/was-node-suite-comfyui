"""Hosted and server language models, and the provider presets the model nodes share.

:data:`PRESETS` holds each built-in provider's fixed address and key variable. :func:`openai_model`,
:func:`anthropic_model`, :func:`gemini_model` and :func:`provider_model` return
:class:`~.wires.LanguageModel` objects without sending a request.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from urllib.parse import urlsplit

from .. import deps, log
from . import FEATURE, runtime, wires

__all__ = [
    "ANTHROPIC_EFFORTS",
    "FUNCTION_CALLING",
    "GEMINI_THINKING",
    "OPENAI_EFFORTS",
    "PRESETS",
    "Preset",
    "THINKING_DISPLAYS",
    "TokenCounter",
    "anthropic_host",
    "anthropic_model",
    "check_model",
    "construct",
    "gemini_model",
    "key_value",
    "openai_model",
    "options",
    "preset",
    "provider_model",
    "publish",
]

logger = log.get_logger("llm.models")

#: Where LLM OpenAI sends requests.
OPENAI_URL = "https://api.openai.com/v1"

#: The variable LLM OpenAI reads its key from.
OPENAI_KEY = "OPENAI_API_KEY"

#: The variable an Anthropic authentication failure names.
ANTHROPIC_KEY = "ANTHROPIC_API_KEY"

#: The variables LLM Google Gemini reads its key from, in order.
GEMINI_KEYS = ("GOOGLE_API_KEY", "GEMINI_API_KEY")

#: The host Gemini requests go to.
GEMINI_HOST = "generativelanguage.googleapis.com"

#: Context window assumed for a model name no table lists.
FALLBACK_CONTEXT = 128000

#: Context window assumed for a Claude model name no table lists.
CLAUDE_FALLBACK_CONTEXT = 200000

#: Context window every current Gemini model reads.
GEMINI_CONTEXT = 1048576

#: The reasoning effort choices for OpenAI, ``default`` sending none.
OPENAI_EFFORTS = ["default", "none", "low", "medium", "high", "xhigh", "max"]

#: The effort choices for Claude, ``default`` sending none.
ANTHROPIC_EFFORTS = ["default", "low", "medium", "high", "xhigh", "max"]

#: How Claude's thinking comes back, ``default`` sending no thinking setting.
THINKING_DISPLAYS = ["summarized", "omitted", "default"]

#: The thinking levels for Gemini, ``default`` sending none.
GEMINI_THINKING = ["default", "minimal", "low", "medium", "high"]

#: The function_calling choices of LLM Provider.
FUNCTION_CALLING = ["auto", "on", "off"]

#: The key handed to a server that takes none.
NO_KEY = "not-needed"

#: Label prefix of a server named in ``llm.endpoints``.
ENDPOINT_PREFIX = "endpoint: "

#: Characters of an endpoint name shown in the menu.
MAX_ENDPOINT_NAME = 64

#: Endpoints read from ``llm.endpoints``.
MAX_ENDPOINTS = 64

#: One model id: letters, digits and ``._:/@~+-``, as ``openai/gpt-oss-120b`` or ``llama3.2:3b``.
_MODEL_ID = re.compile(r"[A-Za-z0-9~][A-Za-z0-9._:/@~+\-]{0,255}")

#: One environment variable name.
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}")

#: OpenAI model names that reason before answering and take no temperature.
_REASONING = re.compile(r"(?:o\d|gpt-(?:[5-9]|[1-9]\d))(?:[.\-]|$)")

_classes: dict = {}


@dataclass(frozen=True)
class Preset:
    """One provider LLM Provider and LLM Embedding Model offer.

    Attributes:
        name: The menu entry, such as ``groq`` or ``endpoint: lab``.
        title: The provider's name in labels, such as ``Groq``.
        url: The base address requests go to.
        api: ``openai`` for an OpenAI-compatible server, ``ollama`` for Ollama's own API.
        key_env: The environment variable holding the key, ``""`` for a server that takes none.
        local: True when the server runs on this machine.
        chat_model: The model used when the node's model is empty, ``""`` for none.
        embed_model: The embedding model used when the node's model is empty, ``""`` for none.
        context_window: Tokens assumed when the node's context_window is 0.
        function_calling: What ``auto`` means for tool calling.
    """

    name: str
    title: str
    url: str
    api: str = "openai"
    key_env: str = ""
    local: bool = False
    chat_model: str = ""
    embed_model: str = ""
    context_window: int = FALLBACK_CONTEXT
    function_calling: bool = True

    @property
    def host(self) -> str:
        """The host requests reach, such as ``api.groq.com``."""
        return urlsplit(self.url).hostname or self.url


#: Every built-in provider, in menu order.
PRESETS: dict[str, Preset] = {
    preset.name: preset
    for preset in (
        Preset("groq", "Groq", "https://api.groq.com/openai/v1", key_env="GROQ_API_KEY",
               chat_model="openai/gpt-oss-120b", context_window=131072),
        Preset("deepseek", "DeepSeek", "https://api.deepseek.com", key_env="DEEPSEEK_API_KEY",
               chat_model="deepseek-flash"),
        Preset("xai", "xAI", "https://api.x.ai/v1", key_env="XAI_API_KEY", chat_model="grok-4.7"),
        Preset("mistral", "Mistral", "https://api.mistral.ai/v1", key_env="MISTRAL_API_KEY",
               chat_model="mistral-large-latest", embed_model="mistral-embed"),
        Preset("openrouter", "OpenRouter", "https://openrouter.ai/api/v1",
               key_env="OPENROUTER_API_KEY", chat_model="~openai/gpt-sol-latest"),
        Preset("together", "Together AI", "https://api.together.ai/v1",
               key_env="TOGETHER_API_KEY", chat_model="openai/gpt-oss-120b"),
        Preset("fireworks", "Fireworks AI", "https://api.fireworks.ai/inference/v1",
               key_env="FIREWORKS_API_KEY", chat_model="fireworks/gpt-oss-120b",
               embed_model="fireworks/qwen3-embedding-8b"),
        Preset("perplexity", "Perplexity", "https://api.perplexity.ai",
               key_env="PERPLEXITY_API_KEY", chat_model="sonar", function_calling=False),
        Preset("cerebras", "Cerebras", "https://api.cerebras.ai/v1", key_env="CEREBRAS_API_KEY",
               chat_model="gpt-oss-120b", context_window=65536),
        Preset("nvidia", "NVIDIA", "https://integrate.api.nvidia.com/v1",
               key_env="NVIDIA_API_KEY", chat_model="openai/gpt-oss-20b"),
        Preset("ollama", "Ollama", "http://127.0.0.1:11434", api="ollama", local=True,
               context_window=8192),
        Preset("lmstudio", "LM Studio", "http://127.0.0.1:1234/v1", local=True,
               context_window=8192),
        Preset("vllm", "vLLM", "http://127.0.0.1:8000/v1", local=True, context_window=8192,
               function_calling=False),
        Preset("llama.cpp", "llama.cpp server", "http://127.0.0.1:8080/v1", local=True,
               context_window=8192, function_calling=False),
    )
}


class TokenCounter:
    """Token counts from llama-index's bundled tokenizer, read without a download."""

    def __init__(self, encode):
        self.encode = encode


def options() -> list[str]:
    """Every provider a menu offers: the built-in ones, then each ``llm.endpoints`` entry."""
    return [*PRESETS, *(ENDPOINT_PREFIX + name for name in _endpoints())]


def preset(choice: str, node: str) -> Preset:
    """The provider a menu entry names.

    Args:
        choice: An entry from :func:`options`.
        node: The node's display name, for the message.

    Returns:
        The built-in preset, or one built from its ``llm.endpoints`` entry.

    Raises:
        ValueError: The entry names no built-in provider and no valid endpoint.
    """
    text = str(choice or "").strip()
    if text in PRESETS:
        return PRESETS[text]
    if text.startswith(ENDPOINT_PREFIX):
        found = _endpoints().get(text[len(ENDPOINT_PREFIX):])
        if found is not None:
            return found
        raise ValueError(
            f"{node} was given `{text[:80]}`, which llm.endpoints in config.yaml does not name "
            f"with a valid http or https url. Add it there and restart ComfyUI, or pick another "
            f"provider."
        )
    raise ValueError(
        f"{node} does not know the provider `{text[:80]}`. Pick one from the provider menu: "
        f"{', '.join(PRESETS)}, or an endpoint from llm.endpoints in config.yaml."
    )


def _endpoints() -> dict[str, Preset]:
    """Every valid ``llm.endpoints`` entry from the loaded configuration, by name."""
    from ..config import load_config

    try:
        block = load_config().get("llm", {}).get("endpoints", {})
    except AttributeError:
        return {}
    if not isinstance(block, dict):
        return {}
    found: dict[str, Preset] = {}
    for raw, entry in list(block.items())[:MAX_ENDPOINTS]:
        name = str(raw).strip()[:MAX_ENDPOINT_NAME]
        made = _endpoint(name, entry) if name else None
        if made is None:
            logger.debug("llm.endpoints entry %r was skipped: it needs an http or https url", name)
            continue
        found[name] = made
    return found


def _endpoint(name: str, entry) -> Preset | None:
    """A preset for one ``llm.endpoints`` entry, or ``None`` when it is not usable."""
    if not isinstance(entry, dict):
        return None
    url = str(entry.get("url") or "").strip().rstrip("/")
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    api = str(entry.get("api") or "openai").strip().lower()
    if api not in ("openai", "ollama"):
        return None
    key_env = str(entry.get("key_env") or "").strip()
    if key_env and not _ENV_NAME.fullmatch(key_env):
        return None
    return Preset(
        name=ENDPOINT_PREFIX + name, title=name, url=url, api=api, key_env=key_env,
        local=runtime.is_loopback(url), context_window=8192,
    )


def check_model(name, node: str, fallback: str = "") -> str:
    """A model id from a widget, checked to be an id and not a path or an address.

    Args:
        name: The widget's value.
        node: The node's display name, for the message.
        fallback: The id used when the value is empty, ``""`` for none.

    Returns:
        The id, stripped.

    Raises:
        ValueError: The value is empty with no fallback, or is not shaped like a model id.
    """
    text = str(name or "").strip() or fallback
    if not text:
        raise ValueError(
            f"{node} needs a model name for this provider, such as `llama3.2:3b` for Ollama. Type "
            f"the name the server lists in the model box."
        )
    if not _MODEL_ID.fullmatch(text) or ".." in text or "//" in text:
        raise ValueError(
            f"{node} was given `{text[:80]}` as a model, which is not a model name. Use the "
            f"provider's own id, such as `gpt-6.1-sol` or `openai/gpt-oss-120b`."
        )
    return text


def foreign_headers() -> dict[str, str]:
    """Blank values for the headers the OpenAI client adds from its own environment variables.

    Returns:
        ``OpenAI-Organization``, ``OpenAI-Project`` and every header ``OPENAI_CUSTOM_HEADERS``
        names, each empty, for a server other than OpenAI.
    """
    names = ["OpenAI-Organization", "OpenAI-Project"]
    for line in os.environ.get("OPENAI_CUSTOM_HEADERS", "").splitlines()[:64]:
        if ":" in line:
            names.append(line.split(":", 1)[0].strip())
    return {name: "" for name in names if name}


def key_value(variables, node: str) -> str:
    """The first set key among environment variables.

    Args:
        variables: One variable name, or several tried in order.
        node: The node's display name, for the message.

    Returns:
        The key. It is never logged or shown.

    Raises:
        RuntimeError: None of the variables is set.
    """
    names = (variables,) if isinstance(variables, str) else tuple(variables)
    for name in names:
        value = os.environ.get(name, "").strip()
        if value:
            runtime.hide(value)
            return value
    shown = " or ".join(names)
    raise RuntimeError(
        f"{node} found no API key. Set {shown} in the environment ComfyUI starts from, then "
        f"restart ComfyUI."
    )


def construct(node: str, build):
    """Build a library object, re-raising a library failure with keys hidden.

    Args:
        node: The node's display name, for the message.
        build: A callable taking no argument.

    Returns:
        What ``build`` returns.

    Raises:
        DependencyError: A package ``build`` needs is missing.
        RuntimeError: ``build`` failed.
    """
    try:
        return build()
    except deps.DependencyError:
        raise
    except Exception as error:
        if type(error).__name__ == "InterruptProcessingException":
            raise
        raise RuntimeError(f"{node} could not set up the model: {runtime.describe_error(error)}") from None


def _known_context(table: dict, name: str) -> int:
    """The context window a table lists for a name or its longest listed prefix, 0 for none."""
    if name in table:
        return int(table[name])
    best = ""
    for key in table:
        if name.startswith(key + "-") and len(key) > len(best):
            best = key
    return int(table[best]) if best else 0


def _openai_vision(name: str) -> bool:
    """Whether an OpenAI model name reads pictures."""
    if name.startswith(("o1-mini", "o1-preview", "o3-mini", "gpt-3.5")):
        return False
    return name.startswith(("gpt-4o", "gpt-4.1", "gpt-4-turbo", "o1", "o3", "o4")) or bool(
        _REASONING.match(name)
    )


def _openai_class():
    """The OpenAI chat model class that reads any model name and its own context window."""
    if "openai" in _classes:
        return _classes["openai"]
    runtime.core()
    deps.require("llama_index.llms.openai", feature=FEATURE)
    from llama_index.core.base.llms.types import LLMMetadata, MessageRole
    from llama_index.core.bridge.pydantic import Field
    from llama_index.llms.openai import OpenAI
    from llama_index.llms.openai import utils as openai_utils

    class HostedOpenAI(OpenAI):
        """OpenAI chat model with a context window for names the integration does not list."""

        context_size: int = Field(default=0, ge=0, description="Context window, 0 for the known one.")

        @classmethod
        def class_name(cls) -> str:
            return "WAS_OpenAI"

        @property
        def metadata(self) -> LLMMetadata:
            name = self._get_model_name()
            known = _known_context(openai_utils.ALL_AVAILABLE_MODELS, name)
            listed = name in openai_utils.ALL_AVAILABLE_MODELS
            return LLMMetadata(
                context_window=self.context_size or known or FALLBACK_CONTEXT,
                num_output=self.max_tokens or -1,
                is_chat_model=openai_utils.is_chat_model(name) if listed else True,
                is_function_calling_model=openai_utils.is_function_calling_model(name),
                model_name=self.model,
                system_role=MessageRole.USER if self.model in openai_utils.O1_MODELS else MessageRole.SYSTEM,
            )

        @property
        def _tokenizer(self):
            from llama_index.core.utils import get_tokenizer

            return TokenCounter(get_tokenizer())

        def _get_model_kwargs(self, **kwargs):
            values = super()._get_model_kwargs(**kwargs)
            name = self._get_model_name()
            if name in openai_utils.O1_MODELS or _REASONING.match(name):
                values.pop("temperature", None)
                limit = values.pop("max_tokens", None)
                if limit is not None:
                    values.setdefault("max_completion_tokens", limit)
            return values

    _classes["openai"] = HostedOpenAI
    return HostedOpenAI


def openai_model(
    model: str,
    reasoning_effort: str = "default",
    max_tokens: int = 16000,
    temperature: float = 1.0,
    timeout: float = 600.0,
    max_retries: int = 3,
    context_window: int = 0,
    node: str = "LLM OpenAI",
) -> wires.LanguageModel:
    """An OpenAI chat model on ``api.openai.com``.

    Args:
        model: The model id, such as ``gpt-6.1-sol``.
        reasoning_effort: One of :data:`OPENAI_EFFORTS`.
        max_tokens: Tokens written at most, reasoning included.
        temperature: Sampling temperature, sent only to models that take one.
        timeout: Seconds one request may take.
        max_retries: Retries after a failed request.
        context_window: Tokens it reads at once, 0 for the known figure.
        node: The node's display name, for messages.

    Returns:
        The wire value. No request is sent.

    Raises:
        ValueError: The model id or effort is not valid.
        RuntimeError: ``features.network`` is off or ``OPENAI_API_KEY`` is not set.
    """
    name = check_model(model, node)
    if reasoning_effort not in OPENAI_EFFORTS:
        raise ValueError(f"{node} does not know the reasoning effort `{str(reasoning_effort)[:40]}`.")
    label = f"{name} on OpenAI"
    runtime.require_network(label)
    key = key_value(OPENAI_KEY, node)
    extra = {} if reasoning_effort == "default" else {"reasoning_effort": reasoning_effort}
    cls = _openai_class()
    llm = construct(node, lambda: cls(
        model=name, temperature=float(temperature), max_tokens=int(max_tokens),
        additional_kwargs=extra, max_retries=int(max_retries), timeout=float(timeout),
        api_key=key, api_base=OPENAI_URL, context_size=int(context_window),
    ))
    metadata = llm.metadata
    return wires.LanguageModel(
        llm=llm, label=label, provider="openai", local=False,
        function_calling=bool(metadata.is_function_calling_model), vision=_openai_vision(name),
        context_window=int(metadata.context_window),
    )


def _anthropic_failure(error) -> RuntimeError | None:
    """The message for an Anthropic credentials failure, or ``None`` for any other error."""
    try:
        import anthropic
    except ImportError:
        return None
    missing = isinstance(error, TypeError) and "authentication method" in str(error)
    if not missing and not isinstance(error, anthropic.AuthenticationError):
        return None
    return RuntimeError(
        f"Anthropic did not accept the credentials ({runtime.describe_error(error)}). Set "
        f"{ANTHROPIC_KEY} in the environment ComfyUI starts from, then restart ComfyUI."
    )


def _refusal(stop_reason, details) -> None:
    """Raise on a response in which Claude declined the request.

    Raises:
        RuntimeError: ``stop_reason`` is ``refusal``.
    """
    if stop_reason != "refusal":
        return
    if isinstance(details, dict):
        category, explanation = details.get("category"), details.get("explanation")
    else:
        category = getattr(details, "category", None)
        explanation = getattr(details, "explanation", None)
    reason = f" in the {category} category" if category else ""
    told = f" Claude said: {runtime.describe_error(explanation)}" if explanation else ""
    raise RuntimeError(
        f"Claude declined this request{reason}.{told} Rephrase the request, or send it to "
        f"another model."
    )


def _anthropic_class():
    """The Claude chat model class that sends no sampling settings and never forces a tool."""
    if "anthropic" in _classes:
        return _classes["anthropic"]
    runtime.core()
    deps.require("llama_index.llms.anthropic", feature=FEATURE)
    from llama_index.core.base.llms.types import LLMMetadata
    from llama_index.core.bridge.pydantic import Field
    from llama_index.llms.anthropic import Anthropic
    from llama_index.llms.anthropic import utils as anthropic_utils

    def stopped(chunk) -> None:
        """Raise on a streamed chunk carrying a refusal."""
        extra = getattr(getattr(chunk, "message", None), "additional_kwargs", None) or {}
        if extra.get("stop_reason") == "refusal":
            delta = (getattr(chunk, "raw", None) or {}).get("delta")
            _refusal("refusal", getattr(delta, "stop_details", None))

    def answered(response) -> None:
        """Raise on a whole response carrying a refusal."""
        raw = getattr(response, "raw", None) or {}
        _refusal(raw.get("stop_reason"), raw.get("stop_details"))

    class HostedAnthropic(Anthropic):
        """Claude chat model sending no temperature, top_p or top_k, and never a forced tool."""

        context_size: int = Field(default=0, ge=0, description="Context window, 0 for the known one.")

        def __init__(self, context_size: int = 0, **kwargs):
            super().__init__(**kwargs)
            self.context_size = max(0, int(context_size))

        @classmethod
        def class_name(cls) -> str:
            return "WAS_Anthropic"

        @property
        def metadata(self) -> LLMMetadata:
            known = _known_context(anthropic_utils.ANTHROPIC_MODELS, self.model)
            return LLMMetadata(
                context_window=self.context_size or known or CLAUDE_FALLBACK_CONTEXT,
                num_output=self.max_tokens,
                is_chat_model=True,
                model_name=self.model,
                is_function_calling_model=True,
            )

        def _get_all_kwargs(self, **kwargs):
            values = super()._get_all_kwargs(**kwargs)
            for name in ("temperature", "top_p", "top_k"):
                values.pop(name, None)
            choice = values.get("tool_choice")
            if isinstance(choice, dict) and choice.get("type") in ("any", "tool"):
                kept = {k: v for k, v in choice.items() if k == "disable_parallel_tool_use"}
                values["tool_choice"] = {**kept, "type": "auto"}
            return values

        def _map_tool_choice_to_anthropic(self, tool_required, allow_parallel_tool_calls):
            return {"disable_parallel_tool_use": not allow_parallel_tool_calls, "type": "auto"}

        def chat(self, messages, **kwargs):
            try:
                response = super().chat(messages, **kwargs)
            except Exception as error:
                failure = _anthropic_failure(error)
                if failure is None:
                    raise
                raise failure from None
            answered(response)
            return response

        def stream_chat(self, messages, **kwargs):
            try:
                chunks = super().stream_chat(messages, **kwargs)
            except Exception as error:
                failure = _anthropic_failure(error)
                if failure is None:
                    raise
                raise failure from None

            def checked():
                for chunk in chunks:
                    stopped(chunk)
                    yield chunk

            return checked()

        async def achat(self, messages, **kwargs):
            try:
                response = await super().achat(messages, **kwargs)
            except Exception as error:
                failure = _anthropic_failure(error)
                if failure is None:
                    raise
                raise failure from None
            answered(response)
            return response

        async def astream_chat(self, messages, **kwargs):
            try:
                chunks = await super().astream_chat(messages, **kwargs)
            except Exception as error:
                failure = _anthropic_failure(error)
                if failure is None:
                    raise
                raise failure from None

            async def checked():
                async for chunk in chunks:
                    stopped(chunk)
                    yield chunk

            return checked()

    _classes["anthropic"] = HostedAnthropic
    return HostedAnthropic


def anthropic_model(
    model: str,
    effort: str = "default",
    thinking_display: str = "summarized",
    max_tokens: int = 16000,
    timeout: float = 600.0,
    max_retries: int = 3,
    context_window: int = 0,
    cache_prompt: bool = False,
    node: str = "LLM Anthropic",
) -> wires.LanguageModel:
    """A Claude chat model.

    Args:
        model: The model id, such as ``claude-opus-5-5``.
        effort: One of :data:`ANTHROPIC_EFFORTS`.
        thinking_display: One of :data:`THINKING_DISPLAYS`.
        max_tokens: Tokens written at most, thinking included.
        timeout: Seconds one request may take.
        max_retries: Retries after a failed request.
        context_window: Tokens it reads at once, 0 for the known figure.
        cache_prompt: Mark the request for prompt caching.
        node: The node's display name, for messages.

    Returns:
        The wire value. No request is sent.

    Raises:
        ValueError: The model id, effort or thinking display is not valid.
        RuntimeError: ``features.network`` is off.
    """
    name = check_model(model, node)
    if effort not in ANTHROPIC_EFFORTS:
        raise ValueError(f"{node} does not know the effort `{str(effort)[:40]}`.")
    if thinking_display not in THINKING_DISPLAYS:
        raise ValueError(f"{node} does not know the thinking display `{str(thinking_display)[:40]}`.")
    label = f"{name} on Anthropic"
    runtime.require_network(label)
    extra: dict = {}
    if effort != "default":
        extra["output_config"] = {"effort": effort}
    if cache_prompt:
        extra["cache_control"] = {"type": "ephemeral"}
    thinking = (
        None if thinking_display == "default"
        else {"type": "adaptive", "display": thinking_display}
    )
    cls = _anthropic_class()
    llm = construct(node, lambda: cls(
        model=name, max_tokens=int(max_tokens), timeout=float(timeout),
        max_retries=int(max_retries), additional_kwargs=extra, thinking_dict=thinking,
        context_size=int(context_window),
    ))
    return wires.LanguageModel(
        llm=llm, label=label, provider="anthropic", local=False, function_calling=True,
        vision=True, context_window=int(llm.metadata.context_window),
    )


def anthropic_host(model: wires.LanguageModel) -> str:
    """The host a Claude model's requests reach, such as ``api.anthropic.com``."""
    try:
        return urlsplit(str(model.llm._client.base_url)).hostname or "api.anthropic.com"
    except Exception:
        return "api.anthropic.com"


def _gemini_class():
    """The Gemini chat model class."""
    if "gemini" in _classes:
        return _classes["gemini"]
    runtime.core()
    deps.require("llama_index.llms.google_genai", feature=FEATURE)
    from llama_index.llms.google_genai import GoogleGenAI

    class HostedGemini(GoogleGenAI):
        """Gemini chat model."""

        @classmethod
        def class_name(cls) -> str:
            return "WAS_Gemini"

    _classes["gemini"] = HostedGemini
    return HostedGemini


def gemini_model(
    model: str,
    thinking_level: str = "default",
    show_thinking: bool = True,
    max_tokens: int = 16000,
    temperature: float = 1.0,
    timeout: float = 600.0,
    max_retries: int = 3,
    context_window: int = 0,
    node: str = "LLM Google Gemini",
) -> wires.LanguageModel:
    """A Gemini chat model.

    Args:
        model: The model id, such as ``gemini-3.8-flash``.
        thinking_level: One of :data:`GEMINI_THINKING`.
        show_thinking: Return the model's thought summaries.
        max_tokens: Tokens written at most, thinking included.
        temperature: Sampling temperature, ``1.0`` sending none.
        timeout: Seconds one request may take.
        max_retries: Retries after a failed request.
        context_window: Tokens it reads at once, 0 for :data:`GEMINI_CONTEXT`.
        node: The node's display name, for messages.

    Returns:
        The wire value. No request is sent.

    Raises:
        ValueError: The model id or thinking level is not valid.
        RuntimeError: ``features.network`` is off or no Gemini key is set.
    """
    name = check_model(model, node)
    if thinking_level not in GEMINI_THINKING:
        raise ValueError(f"{node} does not know the thinking level `{str(thinking_level)[:40]}`.")
    label = f"{name} on Google Gemini"
    runtime.require_network(label)
    key = key_value(GEMINI_KEYS, node)
    cls = _gemini_class()
    import google.genai.types as types

    thinking = None
    if thinking_level != "default" or show_thinking:
        thinking = types.ThinkingConfig(
            include_thoughts=bool(show_thinking),
            thinking_level=None if thinking_level == "default" else thinking_level.upper(),
        )
    sent = None if abs(float(temperature) - 1.0) < 1e-9 else float(temperature)
    config = types.GenerateContentConfig(
        temperature=sent, max_output_tokens=int(max_tokens), thinking_config=thinking,
    )
    context = int(context_window) or GEMINI_CONTEXT
    llm = construct(node, lambda: cls(
        model=name, api_key=key, temperature=sent, max_tokens=int(max_tokens),
        context_window=context, max_retries=int(max_retries), generation_config=config,
        http_options=types.HttpOptions(timeout=int(float(timeout) * 1000)),
    ))
    return wires.LanguageModel(
        llm=llm, label=label, provider="gemini", local=False, function_calling=True,
        vision=True, context_window=context,
    )


def provider_model(
    choice: str,
    model: str = "",
    temperature: float = 0.7,
    max_tokens: int = 4096,
    context_window: int = 0,
    function_calling: str = "auto",
    timeout: float = 600.0,
    node: str = "LLM Provider",
) -> tuple[wires.LanguageModel, Preset]:
    """A chat model on a built-in provider, a local server or an ``llm.endpoints`` entry.

    Args:
        choice: An entry from :func:`options`.
        model: The model id, empty for the provider's default.
        temperature: Sampling temperature.
        max_tokens: Tokens written at most.
        context_window: Tokens it reads at once, 0 for the provider's default.
        function_calling: One of :data:`FUNCTION_CALLING`.
        timeout: Seconds one request may take.
        node: The node's display name, for messages.

    Returns:
        The wire value and the provider it reaches. No request is sent.

    Raises:
        ValueError: The provider, model id or tool setting is not valid.
        RuntimeError: The server is on another machine with ``features.network`` off, or its
            key variable is not set.
    """
    target = preset(choice, node)
    if function_calling not in FUNCTION_CALLING:
        raise ValueError(f"{node} does not know the function_calling choice `{str(function_calling)[:40]}`.")
    name = check_model(model, node, target.chat_model)
    label = f"{name} on {target.title}"
    if not target.local:
        runtime.require_network(label)
    key = key_value(target.key_env, node) if target.key_env else ""
    tools = target.function_calling if function_calling == "auto" else function_calling == "on"
    context = int(context_window) or target.context_window
    if target.api == "ollama":
        deps.require("llama_index.llms.ollama", feature=FEATURE)
        from llama_index.llms.ollama import Ollama

        llm = construct(node, lambda: Ollama(
            model=name, base_url=target.url, temperature=float(temperature),
            context_window=context, request_timeout=float(timeout),
            is_function_calling_model=tools, additional_kwargs={"num_predict": int(max_tokens)},
            headers={"Authorization": f"Bearer {key or NO_KEY}"},
        ))
        family = "ollama"
    else:
        runtime.core()
        deps.require("llama_index.llms.openai_like", feature=FEATURE)
        from llama_index.llms.openai_like import OpenAILike

        llm = construct(node, lambda: OpenAILike(
            model=name, api_base=target.url, api_key=key or NO_KEY, is_chat_model=True,
            is_function_calling_model=tools, context_window=context,
            max_tokens=int(max_tokens), temperature=float(temperature), timeout=float(timeout),
            max_retries=2, default_headers=foreign_headers(),
        ))
        family = "provider"
    wired = wires.LanguageModel(
        llm=llm, label=label, provider=family, local=target.local, function_calling=tools,
        vision=False, context_window=context,
    )
    return wired, target


def publish(model: wires.LanguageModel, host: str, facts: dict | None = None) -> None:
    """Publish what a model node built, for its panel.

    Args:
        model: The model the node hands on.
        host: Where its requests go, such as ``api.openai.com``.
        facts: Further named settings to show, such as the effort.
    """
    from ..interface import run_result

    shown = {
        "model": model.label,
        "context window": f"{model.context_window:,} tokens",
        "calls tools": "yes" if model.function_calling else "no",
        "reads pictures": "yes" if model.vision else "no",
        "text goes": f"to {host}" if not model.local else f"to {host}, on this machine",
    }
    shown.update(facts or {})
    run_result.publish(
        status=run_result.OK,
        summary=(
            f"{model.label} is ready. Prompts are sent to {host}, off this machine."
            if not model.local
            else f"{model.label} is ready. Prompts stay on this machine."
        ),
        counts={"context window": model.context_window},
        facts=shown,
    )
    logger.info(
        "%s ready: context %d tokens, tools %s, pictures %s",
        model.label, model.context_window, model.function_calling, model.vision,
    )

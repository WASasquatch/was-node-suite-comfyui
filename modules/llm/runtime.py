"""What every LLM node needs around a llama-index call.

The import, the guard holding llama-index to wired models, the network gate, interrupts, text
streamed to the running node, and error text with keys hidden.
"""

from __future__ import annotations

import contextlib
import functools
import ipaddress
import logging
import os
import re
import threading
import time
from urllib.parse import urlsplit

from .. import deps, log
from . import FEATURE

__all__ = [
    "NO_EMBEDDING",
    "NO_LLM",
    "TextStream",
    "check_interrupt",
    "core",
    "describe_error",
    "guarded",
    "is_loopback",
    "require_network",
]

logger = log.get_logger("llm")

#: Raised by a llama-index step that reached for a language model no node supplied.
NO_LLM = (
    "This step needs a language model and none was connected. Connect an LLM output, such as "
    "LLM From CLIP, LLM GGUF Loader or LLM OpenAI, to this node's llm input."
)

#: Raised by a llama-index step that reached for an embedding model no node supplied.
NO_EMBEDDING = (
    "This step needs an embedding model and none was connected. Connect LLM Embedding Model to "
    "this node's embedding input."
)

#: Seconds between two updates of the text drawn under a running node.
STREAM_INTERVAL = 0.2

#: Characters of the newest text drawn under a running node.
STREAM_TAIL = 4000

#: Characters of an error message handed on.
ERROR_CHARS = 2000

#: Shapes of API keys and bearer tokens, replaced in any error text a node hands on.
_SECRET = re.compile(
    r"(sk-[A-Za-z0-9_\-*]{8,}|sk_[A-Za-z0-9_\-*]{8,}|AIza[0-9A-Za-z_\-]{20,}|gsk_[A-Za-z0-9]{16,}"
    r"|xai-[A-Za-z0-9]{16,}|hf_[A-Za-z0-9]{16,}|tvly-[A-Za-z0-9\-]{12,}"
    r"|(?i:bearer)\s+[A-Za-z0-9._\-]{12,}|(?i:api[_-]?key)[\"'=: ]+[A-Za-z0-9._\-]{12,})"
)

#: Loggers held at WARNING, whose INFO lines carry prompts, queries and request addresses.
QUIET_LOGGERS = ("llama_index", "httpx", "httpx2", "httpcore", "mcp")

#: Shortest value :func:`hide` records.
_HIDDEN_MIN_CHARS = 6

#: Values most held at once, newest kept.
_HIDDEN_MAX = 256

_hidden_lock = threading.Lock()
_hidden: dict[str, None] = {}

_prepared = False

_guard_lock = threading.Lock()
_guard_depth = 0
_guard_saved: tuple | None = None
_refusers: tuple | None = None


def core():
    """``llama_index.core``, with llama-index's own loggers held at WARNING.

    Raises:
        DependencyError: llama-index is not installed, naming the command that installs it.
    """
    module = deps.require("llama_index.core", feature=FEATURE)
    for name in QUIET_LOGGERS:
        library = logging.getLogger(name)
        if library.level == logging.NOTSET or library.level < logging.WARNING:
            library.setLevel(logging.WARNING)
    global _prepared
    if not _prepared:
        _prepared = True
        _bundled_data()
    return module


def _bundled_data() -> None:
    """Point llama-index's sentence and token data at the copies it ships when the environment names another folder."""
    try:
        import llama_index.core as library
        from llama_index.core.utils import globals_helper, set_global_tokenizer

        if os.environ.get("TIKTOKEN_CACHE_DIR") and library.global_tokenizer is None:
            from . import generate

            found = generate.bundled_encoding("cl100k_base")
            if found is not None:
                set_global_tokenizer(functools.partial(found[1].encode, allowed_special="all"))
        if os.environ.get("NLTK_DATA") and globals_helper._punkt_tokenizer is None:
            from nltk.corpus import stopwords
            from nltk.data import path as searched
            from nltk.tokenize import PunktSentenceTokenizer

            bundled = os.path.join(os.path.dirname(os.path.abspath(library.__file__)), "_static", "nltk_cache")
            if os.path.isdir(bundled) and bundled not in searched:
                searched.append(bundled)
            globals_helper._stopwords = stopwords.words("english")
            globals_helper._punkt_tokenizer = PunktSentenceTokenizer()
    except Exception as error:
        logger.debug("llama-index's bundled sentence and token data could not be set (%s)", error)


@contextlib.contextmanager
def guarded():
    """Make llama-index's global models raise :data:`NO_LLM` and :data:`NO_EMBEDDING` inside the block.

    Yields:
        llama-index's ``Settings``. Nested blocks share one installation.
    """
    global _guard_depth, _guard_saved
    settings = core().Settings
    llm, embedding = _refusing_models()
    with _guard_lock:
        if _guard_depth == 0:
            _guard_saved = (getattr(settings, "_llm", None), getattr(settings, "_embed_model", None))
            settings._llm = llm
            settings._embed_model = embedding
        _guard_depth += 1
    try:
        yield settings
    finally:
        with _guard_lock:
            _guard_depth -= 1
            if _guard_depth == 0 and _guard_saved is not None:
                settings._llm, settings._embed_model = _guard_saved
                _guard_saved = None


def _refusing_models() -> tuple:
    """A language model and an embedding model that raise when either is used."""
    global _refusers
    if _refusers is not None:
        return _refusers
    core()
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.llms import CustomLLM, LLMMetadata

    class NoLanguageModel(CustomLLM):
        """Stands in for a language model no node supplied."""

        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(model_name="none")

        def complete(self, prompt, formatted=False, **kwargs):
            raise RuntimeError(NO_LLM)

        def stream_complete(self, prompt, formatted=False, **kwargs):
            raise RuntimeError(NO_LLM)

    class NoEmbeddingModel(BaseEmbedding):
        """Stands in for an embedding model no node supplied."""

        def _get_query_embedding(self, query):
            raise RuntimeError(NO_EMBEDDING)

        async def _aget_query_embedding(self, query):
            raise RuntimeError(NO_EMBEDDING)

        def _get_text_embedding(self, text):
            raise RuntimeError(NO_EMBEDDING)

    _refusers = (NoLanguageModel(), NoEmbeddingModel(model_name="none"))
    return _refusers


def require_network(what: str) -> None:
    """Refuse a call that sends text off this machine unless ``features.network`` is on.

    Args:
        what: What would be sent and where, for the message, such as ``"LLM OpenAI"``.

    Raises:
        RuntimeError: ``features.network`` is off.
    """
    from ..model import network_enabled

    if network_enabled():
        return
    raise RuntimeError(
        f"{what} sends text to a server on another machine, and features.network is off. Set "
        f"features.network to true in config.yaml, in ComfyUI/user/was-node-suite, and restart "
        f"ComfyUI to allow it, or use a model that runs on this machine."
    )


def is_loopback(url: str) -> bool:
    """Whether an address names this machine.

    Args:
        url: An ``http`` or ``https`` address.

    Returns:
        True for ``localhost`` and loopback addresses, False for anything else, including an
        address that does not parse.
    """
    try:
        host = urlsplit(str(url)).hostname or ""
    except ValueError:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_interrupt() -> None:
    """Stop here when the queue was cancelled from the browser.

    Raises:
        InterruptProcessingException: ComfyUI was asked to stop the run.
    """
    try:
        import comfy.model_management as management
    except ImportError:
        return
    management.throw_exception_if_processing_interrupted()


def hide(*values) -> None:
    """Record secrets read from the environment or the configuration, replaced in every error text.

    Args:
        values: Keys, tokens, passwords or addresses; one shorter than six characters is skipped.
    """
    with _hidden_lock:
        for value in values:
            text = str(value or "").strip()
            if len(text) < _HIDDEN_MIN_CHARS:
                continue
            _hidden.pop(text, None)
            _hidden[text] = None
            while len(_hidden) > _HIDDEN_MAX:
                _hidden.pop(next(iter(_hidden)))


def describe_error(error: BaseException | str) -> str:
    """An error's text with keys and tokens replaced, cut to :data:`ERROR_CHARS`.

    Args:
        error: The exception, or its text.

    Returns:
        Text safe to put in an exception a node raises.
    """
    text = str(error).strip() or type(error).__name__
    text = _SECRET.sub("[key hidden]", text)
    with _hidden_lock:
        held = sorted(_hidden, key=len, reverse=True)
    for value in held:
        if value in text:
            text = text.replace(value, "[hidden]")
    if _refused(error, text):
        text += (
            ". Nothing answered at the model's address. For a server on this machine, such as "
            "Ollama, LM Studio or vLLM, start it and load the model; for another, check its "
            "address in llm.endpoints."
        )
    return text if len(text) <= ERROR_CHARS else text[: ERROR_CHARS - 3] + "..."


#: How a refused or unreachable connection reads across libraries and systems.
_REFUSED = re.compile(r"10061|connection refused|actively refused|failed to establish a new connection", re.I)


def _refused(error, text: str) -> bool:
    """Whether an error is a connection nothing answered."""
    kinds = {type(link).__name__ for link in _chain(error)} if isinstance(error, BaseException) else set()
    if kinds & {"ConnectionRefusedError", "ConnectError", "APIConnectionError"}:
        return True
    return bool(_REFUSED.search(text))


def _chain(error: BaseException, depth: int = 8):
    """An exception and the ones it was raised from, nearest first."""
    seen = 0
    while error is not None and seen < depth:
        yield error
        error = error.__cause__ or error.__context__
        seen += 1


class TextStream:
    """Text drawn under the executing node while a model writes it, for the tab that queued the run."""

    def __init__(self, label: str = "", node_id: str | None = None):
        """Start a stream for the node ComfyUI is executing.

        Args:
            label: A line drawn above the text, such as the model's name.
            node_id: The node to draw on. Left out, the executing node.
        """
        from ..interface import channel

        self.node_id = node_id if node_id is not None else channel.executing_node_id()
        self.client = channel.executing_client_id()
        self.label = label
        self.text = ""
        self._sent_at = 0.0
        self._dirty = False

    def write(self, delta: str) -> None:
        """Add text and redraw when the interval has passed.

        Args:
            delta: Text the model wrote since the last call.
        """
        if not delta:
            return
        self.text += str(delta)
        self._dirty = True
        now = time.monotonic()
        if now - self._sent_at >= STREAM_INTERVAL:
            self._send()
            self._sent_at = now

    def note(self, line: str) -> None:
        """Replace the label line, such as with the tool an agent is calling.

        Args:
            line: The new label.
        """
        self.label = str(line or "")
        self._dirty = True
        self._send()

    def finish(self) -> None:
        """Draw whatever is still unsent."""
        if self._dirty:
            self._send()

    def _send(self) -> None:
        """Send the label and the tail of the text."""
        self._dirty = False
        if not self.node_id:
            return
        tail = self.text[-STREAM_TAIL:]
        body = f"{self.label}\n{tail}" if self.label else tail
        try:
            from server import PromptServer

            PromptServer.instance.send_progress_text(body, self.node_id, self.client or None)
        except Exception:
            logger.debug("no progress text is available", exc_info=True)

"""Generation, chat turns, token counts and prompt templates for the LLM nodes.

:func:`reply` and :func:`areply` stream an answer under the running node, :func:`count_tokens`
counts with a model's own tokenizer or the one bundled with llama-index, and :func:`fill`
substitutes ``{name}`` fields.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass

from .. import deps, log
from . import FEATURE, messages, runtime

__all__ = [
    "MAX_SYSTEM_CHARS",
    "MAX_TEMPLATE_CHARS",
    "MAX_TOKEN_TEXT_CHARS",
    "Reply",
    "Turn",
    "areply",
    "bundled_encoding",
    "converse",
    "count_tokens",
    "digest",
    "discard",
    "fill",
    "merge_system",
    "new_messages",
    "pictures_in",
    "reply",
    "require_vision",
    "split_reply",
    "split_system",
    "text_value",
    "with_max_tokens",
    "with_system_prompt",
]

logger = log.get_logger("llm.generate")

#: Characters a filled template holds at most.
MAX_TEMPLATE_CHARS = 1_000_000

#: Characters of a system prompt kept.
MAX_SYSTEM_CHARS = 100_000

#: Characters of text one token count reads at most.
MAX_TOKEN_TEXT_CHARS = 50_000_000

#: A ``{name}`` field, or a doubled brace standing for a literal one.
FIELD = re.compile(r"\{\{|\}\}|\{([A-Za-z_][A-Za-z0-9_]{0,63})\}")

#: The tokenizers bundled with llama-index, best first: download address, SHA-256 of the file,
#: split pattern and special tokens.
_BUNDLED = {
    "o200k_base": (
        "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
        "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
        "|".join(
            [
                r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
                r"""[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?""",
                r"""\p{N}{1,3}""",
                r""" ?[^\s\p{L}\p{N}]+[\r\n/]*""",
                r"""\s*[\r\n]+""",
                r"""\s+(?!\S)""",
                r"""\s+""",
            ]
        ),
        {"<|endoftext|>": 199999, "<|endofprompt|>": 200018},
    ),
    "cl100k_base": (
        "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
        "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
        r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s""",
        {
            "<|endoftext|>": 100257,
            "<|fim_prefix|>": 100258,
            "<|fim_middle|>": 100259,
            "<|fim_suffix|>": 100260,
            "<|endofprompt|>": 100276,
        },
    ),
}

_encoding_lock = threading.Lock()
_encodings: dict = {}


@dataclass(frozen=True)
class Reply:
    """What a model answered.

    Attributes:
        text: The answer, thinking removed.
        thinking: What a reasoning model thought first, ``""`` otherwise.
        seconds: Time from the request to the last word.
        limited: True when a requested token limit reached the model, False when it was not
            asked for or the model takes none.
    """

    text: str
    thinking: str
    seconds: float
    limited: bool = False


def with_max_tokens(llm, limit: int):
    """A copy of a llama-index LLM that writes at most ``limit`` tokens.

    Args:
        llm: The llama-index LLM.
        limit: Tokens to write at most, 0 for the model's own setting.

    Returns:
        ``(llm, applied)``: the copy and True, or the model unchanged and False when ``limit``
        is 0 or the model takes no limit.
    """
    if not limit or limit <= 0:
        return llm, False
    fields = getattr(type(llm), "model_fields", {}) or {}
    for name in ("max_tokens", "max_new_tokens", "num_predict"):
        if name in fields:
            try:
                return llm.model_copy(update={name: int(limit)}), True
            except Exception as error:
                logger.debug("the token limit could not be set on %s (%s)", type(llm).__name__, error)
                return llm, False
    config = getattr(llm, "_generation_config", None)
    if isinstance(config, dict):
        try:
            copy = llm.model_copy()
            copy._generation_config = {**config, "max_output_tokens": int(limit)}
            if hasattr(copy, "_max_tokens"):
                copy._max_tokens = int(limit)
            return copy, True
        except Exception as error:
            logger.debug("the token limit could not be set on %s (%s)", type(llm).__name__, error)
    return llm, False


def with_system_prompt(llm, text: str):
    """A copy of a llama-index LLM that opens every call with a system message.

    Args:
        llm: The llama-index LLM.
        text: The instructions, cut to :data:`MAX_SYSTEM_CHARS`. Empty leaves the model as it is.

    Returns:
        The copy, or the model unchanged when ``text`` is empty.

    Raises:
        ValueError: The model class takes no system prompt.
    """
    wanted = str(text or "").strip()[:MAX_SYSTEM_CHARS]
    if not wanted:
        return llm
    if "system_prompt" not in (getattr(type(llm), "model_fields", {}) or {}):
        raise ValueError(f"{type(llm).__name__} takes no system prompt")
    return llm.model_copy(update={"system_prompt": wanted})


def split_reply(response, streamed: str = "") -> tuple[str, str]:
    """The answer and the thinking in a response.

    Args:
        response: A llama-index ``ChatResponse``, or ``None``.
        streamed: Every delta streamed, read in place of a shorter text when the response
            carries no thinking block.

    Returns:
        ``(answer, thinking)``, both stripped.
    """
    blocks = messages.thinking_of(response) if response is not None else ""
    text = messages.text_of(response) if response is not None else ""
    if not blocks and len(streamed) > len(text):
        text = streamed
    opened = "</think>" in text and "<think>" not in text
    answer, inline = messages.split_thinking(text, opened=opened)
    if blocks and inline and inline.strip() != blocks.strip():
        return answer, f"{blocks}\n\n{inline}"
    return answer, blocks or inline


def _interrupted(error: BaseException) -> bool:
    """Whether an exception is ComfyUI stopping the run."""
    return type(error).__name__ == "InterruptProcessingException"


def reply(model, chat, node: str, max_tokens: int = 0) -> Reply:
    """Stream a model's answer to a conversation under the running node.

    Args:
        model: The ``LanguageModel`` to ask.
        chat: llama-index ``ChatMessage`` objects, oldest first.
        node: The node's display name, for errors.
        max_tokens: Tokens to write at most, 0 for the model's own setting.

    Returns:
        The answer.

    Raises:
        RuntimeError: The model failed, its message cleared of keys.
    """
    llm, limited = with_max_tokens(model.llm, max_tokens)
    stream = runtime.TextStream(model.label)
    started = time.monotonic()
    last = None
    deltas: list[str] = []
    try:
        try:
            chunks = llm.stream_chat(list(chat))
        except NotImplementedError:
            chunks = None
        if chunks is None:
            last = llm.chat(list(chat))
            stream.write(messages.text_of(last))
        else:
            for chunk in chunks:
                runtime.check_interrupt()
                last = chunk
                delta = getattr(chunk, "delta", None) or ""
                if delta:
                    deltas.append(delta)
                    stream.write(delta)
    except Exception as error:
        if _interrupted(error):
            raise
        raise RuntimeError(f"{node} failed: {runtime.describe_error(error)}") from None
    finally:
        stream.finish()
    text, thinking = split_reply(last, "".join(deltas))
    return Reply(text, thinking, time.monotonic() - started, limited)


async def areply(model, chat, node: str, max_tokens: int = 0) -> Reply:
    """Stream a model's answer to a conversation under the running node, asynchronously.

    Args:
        model: The ``LanguageModel`` to ask.
        chat: llama-index ``ChatMessage`` objects, oldest first.
        node: The node's display name, for errors.
        max_tokens: Tokens to write at most, 0 for the model's own setting.

    Returns:
        The answer.

    Raises:
        RuntimeError: The model failed, its message cleared of keys.
    """
    llm, limited = with_max_tokens(model.llm, max_tokens)
    stream = runtime.TextStream(model.label)
    started = time.monotonic()
    last = None
    deltas: list[str] = []
    try:
        try:
            chunks = await llm.astream_chat(list(chat))
        except NotImplementedError:
            chunks = None
        if chunks is None:
            last = await llm.achat(list(chat))
            stream.write(messages.text_of(last))
        else:
            async for chunk in chunks:
                runtime.check_interrupt()
                last = chunk
                delta = getattr(chunk, "delta", None) or ""
                if delta:
                    deltas.append(delta)
                    stream.write(delta)
    except Exception as error:
        if _interrupted(error):
            raise
        raise RuntimeError(f"{node} failed: {runtime.describe_error(error)}") from None
    finally:
        stream.finish()
    text, thinking = split_reply(last, "".join(deltas))
    return Reply(text, thinking, time.monotonic() - started, limited)


def require_vision(model, node: str, pictures: int) -> None:
    """Refuse pictures for a model that does not read them.

    Args:
        model: The ``LanguageModel``.
        node: The node's display name, for the message.
        pictures: How many pictures would be sent.

    Raises:
        ValueError: Pictures would be sent and the model does not read them.
    """
    if pictures and not model.vision:
        raise ValueError(
            f"{node} was given {pictures} picture{'s' if pictures != 1 else ''}, and "
            f"{model.label} does not read pictures. Connect a model that does, such as LLM From "
            f"CLIP with Qwen3 VL or a vision model from LLM OpenAI, or disconnect the image."
        )


def pictures_in(chat) -> int:
    """How many pictures llama-index messages carry."""
    from llama_index.core.llms import ImageBlock

    return sum(isinstance(block, ImageBlock) for item in chat or () for block in item.blocks)


def split_system(chat) -> tuple[list, list]:
    """Messages split into the system ones and the rest, each in order.

    Returns:
        ``(system, others)``.
    """
    system, others = [], []
    for item in chat or ():
        (system if getattr(item.role, "value", item.role) == "system" else others).append(item)
    return system, others


def merge_system(chat, system) -> list:
    """A conversation led by one system message holding ``system`` and any it already had.

    Args:
        chat: llama-index ``ChatMessage`` objects, oldest first.
        system: System messages to put first.

    Returns:
        The conversation, unchanged when ``system`` is empty.
    """
    chat = list(chat or ())
    if not system:
        return chat
    from llama_index.core.llms import ChatMessage, MessageRole, TextBlock

    blocks = []
    for item in system:
        if blocks:
            blocks.append(TextBlock(text="\n\n"))
        blocks.extend(item.blocks)
    if chat and getattr(chat[0].role, "value", chat[0].role) == "system":
        blocks = [*blocks, TextBlock(text="\n\n"), *chat[0].blocks]
        chat = chat[1:]
    return [ChatMessage(role=MessageRole.SYSTEM, blocks=blocks), *chat]


def _identity(item) -> tuple:
    """What tells two messages apart: role, words and picture count."""
    from llama_index.core.llms import ImageBlock

    role = getattr(item.role, "value", str(item.role))
    pictures = sum(isinstance(block, ImageBlock) for block in item.blocks)
    return role, messages.text_of(item).strip(), pictures


def new_messages(given, stored) -> list:
    """The messages in ``given`` that ``stored`` does not already hold.

    Args:
        given: llama-index ``ChatMessage`` objects to add.
        stored: The messages already kept.

    Returns:
        ``given`` without the ones matching a stored message by role, words and pictures.
    """
    held = {_identity(item) for item in stored or ()}
    return [item for item in given or () if _identity(item) not in held]


def digest(chat) -> str:
    """A short hex digest of messages, the same for the same conversation."""
    data = json.dumps(messages.to_dicts(chat), sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(data.encode("ascii")).hexdigest()[:32]


@dataclass(frozen=True)
class Turn:
    """One chat turn.

    Attributes:
        reply: The answer, thinking removed.
        thinking: What a reasoning model thought first.
        conversation: llama-index ``ChatMessage`` objects, oldest first, the reply last.
        seconds: Time the model took, 0 when it was not asked.
        source: ``model`` when the model answered; ``stored`` when a saved reply was handed back;
            ``none`` when nothing was asked and nothing was saved.
        stored: What :func:`memory.save_memory` counted, empty without a memory.
    """

    reply: str
    thinking: str
    conversation: tuple
    seconds: float
    source: str
    stored: dict


async def converse(model, given, text: str, image, spec, node: str) -> Turn:
    """Answer a new turn after a conversation, keeping it in a memory when one is given.

    Args:
        model: The ``LanguageModel``.
        given: llama-index ``ChatMessage`` objects already in the conversation.
        text: The new message, ``""`` for none.
        image: Pictures sent with it, or ``None``.
        spec: A ``MemorySpec``, or ``None``.
        node: The node's display name, for errors.

    Returns:
        The turn.

    Raises:
        ValueError: There is nothing to send.
        RuntimeError: The model failed.
    """
    from llama_index.core.llms import ChatMessage, MessageRole

    from . import memory as memories

    given = list(given or ())
    system, earlier = split_system(given)
    asked = bool(str(text or "").strip()) or image is not None
    request = messages.message("user", text, images=image) if asked else None
    key = ""
    if spec is not None and spec.session:
        key = memories.turn_key(text, image, extra=f"{model.label}\0{digest(given)}")
        previous = memories.last_turn(spec)
        if not asked or (previous is not None and previous.get("key") == key):
            history = await memories.history(spec)
            held = previous or {}
            return Turn(
                str(held.get("reply") or ""), str(held.get("thinking") or ""),
                tuple(system + history), 0.0, "stored" if previous else "none", {},
            )
    elif not asked and not earlier:
        raise ValueError(
            f"{node} has nothing to send. Type a message, or connect LLM Message to its messages input."
        )
    if spec is None:
        chat = given + ([request] if request is not None else [])
        answer = await areply(model, chat, node)
        said = ChatMessage(role=MessageRole.ASSISTANT, content=answer.text)
        return Turn(answer.text, answer.thinking, tuple(chat + [said]), answer.seconds, "model", {})
    memory = await memories.open_memory(spec, model)
    try:
        kept = await memory.aget_all()
        added = new_messages(earlier, kept) if spec.session else earlier
        if added:
            await memory.aput_messages(added)
        if request is not None:
            await memory.aput(request)
        seen = merge_system(await memory.aget(), system)
        answer = await areply(model, seen, node)
        await memory.aput(ChatMessage(role=MessageRole.ASSISTANT, content=answer.text))
        conversation = system + list(await memory.aget_all())
    except BaseException:
        await discard(memory)
        raise
    turn = {"key": key, "reply": answer.text, "thinking": answer.thinking} if spec.session else None
    counts = await memories.save_memory(spec, memory, turn=turn)
    return Turn(answer.text, answer.thinking, tuple(conversation), answer.seconds, "model", counts)


async def discard(memory) -> None:
    """Close a memory without saving it, releasing its database engine."""
    from . import memory as memories

    await memories.close_memory(memory)


def bundled_encoding(only: str = ""):
    """The best tokenizer bundled with llama-index, read from disk with no download.

    Args:
        only: One tokenizer's name, such as ``cl100k_base``; blank takes the best one bundled.

    Returns:
        ``(name, encoding)``, or ``None`` when tiktoken or both bundled files are missing.
    """
    tiktoken = deps.optional("tiktoken")
    if tiktoken is None:
        return None
    for name in (only,) if only in _BUNDLED else _BUNDLED:
        with _encoding_lock:
            if name not in _encodings:
                _encodings[name] = _load_encoding(tiktoken, name)
            encoding = _encodings[name]
        if encoding is not None:
            return name, encoding
    return None


def _load_encoding(tiktoken, name: str):
    """One bundled tokenizer, or ``None`` when its file is missing or altered."""
    import base64

    runtime.core()
    import llama_index.core as core

    url, expected, pattern, special = _BUNDLED[name]
    key = hashlib.sha1(url.encode("ascii")).hexdigest()
    path = os.path.join(os.path.dirname(os.path.abspath(core.__file__)), "_static", "tiktoken_cache", key)
    if not os.path.isfile(path):
        logger.debug("the bundled %s tokenizer is not installed", name)
        return None
    with open(path, "rb") as handle:
        data = handle.read()
    if hashlib.sha256(data).hexdigest() != expected:
        logger.warning("the bundled %s tokenizer file does not match its checksum and was skipped", name)
        return None
    ranks = {}
    for line in data.splitlines():
        if line:
            token, rank = line.split()
            ranks[base64.b64decode(token)] = int(rank)
    return tiktoken.Encoding(name=name, pat_str=pattern, mergeable_ranks=ranks, special_tokens=special)


def count_tokens(model, text: str) -> tuple[int, str, str]:
    """Tokens in a text, by the model's own tokenizer where it has one.

    Args:
        model: A ``LanguageModel``, or ``None``.
        text: The text to count.

    Returns:
        ``(count, tokenizer, note)``: the count, what counted it, and why the model's own
        tokenizer was passed over, ``""`` when it was not.

    Raises:
        ValueError: The text is longer than :data:`MAX_TOKEN_TEXT_CHARS`.
        RuntimeError: No tokenizer is available.
    """
    text = str(text or "")
    if len(text) > MAX_TOKEN_TEXT_CHARS:
        raise ValueError(
            f"The text is {len(text):,} characters and LLM Token Count reads at most "
            f"{MAX_TOKEN_TEXT_CHARS:,}. Count it in smaller pieces."
        )
    note = ""
    counter = getattr(getattr(model, "llm", None), "count_tokens", None) if model is not None else None
    if callable(counter):
        try:
            count = counter(text)
            if isinstance(count, (list, tuple)):
                count = len(count)
            return int(count), f"{model.label}'s own", ""
        except Exception as error:
            note = f"its own tokenizer failed: {runtime.describe_error(error)}"
            logger.debug("count_tokens failed on %s (%s)", model.label, error)
    elif model is not None:
        note = "it carries no tokenizer of its own"
    found = bundled_encoding()
    if found is None:
        raise RuntimeError(
            "LLM Token Count found no tokenizer. Connect an llm whose model counts its own tokens, "
            "such as LLM From CLIP or LLM GGUF Loader, or reinstall llama-index-core, whose "
            "bundled tokenizer files are missing."
        )
    name, encoding = found
    return len(encoding.encode_ordinary(text)), name, note


def text_value(value) -> str | None:
    """A template value as text: lists joined with commas, dictionaries as JSON.

    Returns:
        The text, or ``None`` for no value.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        parts = (text_value(item) for item in value)
        return ", ".join(part for part in parts if part is not None)
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def fill(template: str, values, limit: int = MAX_TEMPLATE_CHARS) -> tuple[str, list[str], list[str]]:
    """Substitute ``{name}`` fields in a template; ``{{`` and ``}}`` stay single braces.

    Args:
        template: The text holding fields.
        values: Field name to value, read through :func:`text_value`; a name absent here, or
            holding ``None``, is left in place.
        limit: Characters the result holds at most.

    Returns:
        ``(text, filled, missing)``: the result and the field names filled and left empty, each
        once, in the order they first appear.

    Raises:
        ValueError: The result would be longer than ``limit``.
    """
    template = str(template or "")
    pieces: list[str] = []
    size = 0
    filled: list[str] = []
    missing: list[str] = []
    position = 0
    converted: dict[str, str | None] = {}

    def add(piece: str) -> None:
        nonlocal size
        size += len(piece)
        if size > limit:
            raise ValueError(
                f"The filled template would be longer than {limit:,} characters. Shorten the "
                f"template or the values put into it."
            )
        pieces.append(piece)

    for match in FIELD.finditer(template):
        add(template[position:match.start()])
        token = match.group(0)
        if token == "{{":
            add("{")
        elif token == "}}":
            add("}")
        else:
            name = match.group(1)
            if name not in converted:
                converted[name] = text_value(values.get(name))
            value = converted[name]
            if value is None:
                add(token)
                if name not in missing:
                    missing.append(name)
            else:
                add(value)
                if name not in filled:
                    filled.append(name)
        position = match.end()
    add(template[position:])
    return "".join(pieces), filled, missing

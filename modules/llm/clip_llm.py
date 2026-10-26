"""Local language models as llama-index chat models.

:func:`local_base` builds the class every local model derives from; :func:`clip_model` wraps the
language model inside a ComfyUI ``CLIP``, rendered in its own chat format.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field

from .. import log
from . import messages, runtime

__all__ = [
    "Family",
    "MAX_CONTEXT",
    "MAX_PICTURES",
    "clip_model",
    "context_of",
    "conversation",
    "empty_cache",
    "family_of",
    "free_bytes",
    "generator_of",
    "local_base",
    "pieces",
    "release_clip",
    "split",
]

logger = log.get_logger("llm.models")

#: The most tokens a local model is offered as its context window.
MAX_CONTEXT = 32768

#: The context window assumed where a model's config names none.
DEFAULT_CONTEXT = 8192

#: Pictures handed to a model across one conversation, the newest kept.
MAX_PICTURES = 16

#: Seconds between two redraws of the text a CLIP model is writing.
LIVE_INTERVAL = 0.25

#: Characters per delta when a finished generation is replayed as a stream.
PIECE_CHARS = 48

#: The role names a conversation is reduced to.
USER, ASSISTANT, TOOL = "user", "assistant", "tool"

#: How a CLIP's tokenizer is driven: a full ChatML conversation, a Mistral instruction, or the
#: model's own template around one user turn.
CHATML, MISTRAL, TEMPLATE = "chatml", "mistral", "template"

#: One picture slot in a Qwen user turn, filled in order with the pictures handed over.
VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"

#: The empty thinking block that turns reasoning off in a hybrid Qwen3 or Qwen3.5 chat.
EMPTY_THINK = "<think>\n\n</think>\n\n"

#: Control strings removed from message text, by tokenizer family.
CHATML_CONTROLS = (
    "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>",
    "<|video_pad|>", "<|endoftext|>",
)
GEMMA3_CONTROLS = (
    "<start_of_turn>", "<end_of_turn>", "<image_soft_token>", "<start_of_image>",
    "<end_of_image>", "<bos>", "<eos>",
)
GEMMA4_CONTROLS = (
    "<|turn>", "<turn|>", "<|image>", "<image|>", "<|image|>", "<|video|>", "<|audio>",
    "<audio|>", "<|audio|>", "<|think|>", "<|channel>", "<channel|>", "<bos>", "<eos>",
)
MISTRAL_CONTROLS = (
    "[INST]", "[/INST]", "[SYSTEM_PROMPT]", "[/SYSTEM_PROMPT]", "[IMG]", "[IMG_BREAK]",
    "[IMG_END]", "<s>", "</s>",
)

#: Attributes of a CLIP's generating transformer that hold a vision tower.
VISION_PARTS = ("visual", "vision_model", "vision_tower", "multi_modal_projector")

_classes: dict = {}


@dataclass(frozen=True)
class Family:
    """How one CLIP's tokenizer turns a conversation into tokens.

    Attributes:
        kind: :data:`CHATML`, :data:`MISTRAL` or :data:`TEMPLATE`.
        think: ``"open"`` where thinking opens the block in the prompt, ``"hybrid"`` where the
            model opens it, ``""`` where the model has no thinking switch.
        vision: True when the model reads pictures.
        many: True when more than one picture can be handed over at once.
        name: The tokenizer's model name, such as ``"qwen3_4b"``.
        controls: Control strings removed from message text.
    """

    kind: str
    think: str = ""
    vision: bool = False
    many: bool = False
    name: str = ""
    controls: tuple = field(default_factory=tuple)


@dataclass
class Turn:
    """One message of a conversation, reduced to what a local model reads.

    Attributes:
        role: :data:`USER`, :data:`ASSISTANT` or :data:`TOOL`.
        text: The words, thinking left out.
        pictures: The ``ImageBlock`` objects it carries.
    """

    role: str
    text: str
    pictures: list = field(default_factory=list)


def conversation(items, controls: tuple = ()) -> tuple[str, list[Turn]]:
    """The system text and the turns of a llama-index conversation.

    Args:
        items: llama-index ``ChatMessage`` objects, oldest first.
        controls: Strings removed from every text, such as a chat format's turn markers.

    Returns:
        ``(system, turns)``. Pictures past the newest :data:`MAX_PICTURES` are replaced by a
        ``[picture]`` note.
    """
    from llama_index.core.llms import ImageBlock

    system_parts: list[str] = []
    turns: list[Turn] = []
    for item in items or ():
        role = getattr(item.role, "value", str(item.role))
        text = _clean(messages.text_of(item), controls)
        if role in ("system", "developer"):
            if text.strip():
                system_parts.append(text.strip())
            continue
        pictures = [
            block for block in getattr(item, "blocks", ()) or ()
            if isinstance(block, ImageBlock) and block.image
        ]
        if role in ("tool", "function"):
            turns.append(Turn(TOOL, text))
        elif role in ("assistant", "model", "chatbot"):
            turns.append(Turn(ASSISTANT, text))
        else:
            turns.append(Turn(USER, text, pictures))
    total = sum(len(turn.pictures) for turn in turns)
    excess = total - MAX_PICTURES
    for turn in turns:
        if excess <= 0:
            break
        dropped = min(excess, len(turn.pictures))
        if dropped:
            turn.pictures = turn.pictures[dropped:]
            turn.text = f"{'[picture] ' * dropped}{turn.text}".strip()
            excess -= dropped
    return "\n\n".join(system_parts), turns


def _clean(text: str, controls: tuple) -> str:
    """Text with every control string removed."""
    text = str(text or "")
    for control in controls:
        if control in text:
            text = text.replace(control, "")
    return text


def transcript_prompt(turns: list[Turn], seen: bool) -> str:
    """One user message holding a conversation's earlier turns as a short transcript.

    Args:
        turns: The conversation, oldest first.
        seen: True when the model is handed the pictures, False to note them in the text.

    Returns:
        The last user message alone when there is no earlier turn.
    """
    names = {USER: "User", ASSISTANT: "Assistant", TOOL: "Tool result"}
    bodies = []
    for turn in turns:
        note = "" if seen or not turn.pictures else "[picture] " * len(turn.pictures)
        bodies.append(f"{note}{turn.text}".strip())
    if len(turns) == 1 and turns[0].role == USER:
        return bodies[0]
    lines = [f"{names[turn.role]}: {body}" for turn, body in zip(turns, bodies)]
    return "The conversation so far:\n\n" + "\n\n".join(lines) if lines else ""


def split(text: str, opened: bool = False) -> tuple[str, str]:
    """An answer and the thinking before it.

    Args:
        text: What the model wrote.
        opened: True when the prompt opened the thinking block.

    Returns:
        ``(answer, thinking)``, both stripped.
    """
    text = str(text or "")
    opened = opened or ("</think>" in text and "<think>" not in text)
    return messages.split_thinking(text, opened=opened)


def pieces(text: str, size: int = PIECE_CHARS):
    """A finished text cut into deltas that join back into it.

    Args:
        text: The text.
        size: The most characters per piece.

    Yields:
        Consecutive slices, cut after a space or a line break where one is near.
    """
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            cut = max(text.rfind(" ", start, end), text.rfind("\n", start, end))
            if cut > start:
                end = cut + 1
        yield text[start:end]
        start = end


def local_base():
    """The llama-index chat model class every local model here derives from.

    Returns:
        A ``CustomLLM`` subclass. A subclass supplies ``_stream`` and ``count_tokens``.
    """
    if "base" in _classes:
        return _classes["base"]
    runtime.core()
    from llama_index.core.base.llms.types import ThinkingBlock
    from llama_index.core.llms import (
        ChatMessage,
        ChatResponse,
        CompletionResponse,
        CustomLLM,
        LLMMetadata,
        MessageRole,
        TextBlock,
    )
    from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback

    class LocalLLM(CustomLLM):
        """A language model running inside this ComfyUI process."""

        model_label: str = "local"
        context: int = DEFAULT_CONTEXT
        max_tokens: int = 1024
        draws: bool = True

        @classmethod
        def class_name(cls) -> str:
            return "was_local_llm"

        @property
        def metadata(self) -> LLMMetadata:
            return LLMMetadata(
                context_window=int(self.context),
                num_output=int(self.max_tokens),
                is_chat_model=True,
                is_function_calling_model=False,
                model_name=self.model_label,
            )

        def count_tokens(self, text: str) -> int:
            """Tokens in a text, by the model's own tokenizer."""
            raise NotImplementedError

        def _stream(self, items):
            """Raw text deltas, then ``(opened, stats)`` as the generator's return value."""
            raise NotImplementedError

        def _reply(self, raw: str, opened: bool, stats: dict) -> ChatResponse:
            """The final response: thinking in a ``ThinkingBlock``, the answer in a ``TextBlock``."""
            answer, thinking = split(raw, opened)
            blocks = [ThinkingBlock(content=thinking)] if thinking else []
            blocks.append(TextBlock(text=answer))
            return ChatResponse(
                message=ChatMessage(role=MessageRole.ASSISTANT, blocks=blocks),
                raw=dict(stats or {}),
                delta="",
            )

        @llm_chat_callback()
        def chat(self, messages, **kwargs) -> ChatResponse:
            stream = runtime.TextStream(self.model_label) if self.draws else None
            generator = self._stream(list(messages or ()))
            parts: list[str] = []
            while True:
                try:
                    delta = next(generator)
                except StopIteration as stop:
                    opened, stats = stop.value or (False, {})
                    break
                parts.append(delta)
                if stream is not None:
                    stream.write(delta)
            if stream is not None:
                stream.finish()
            return self._reply("".join(parts), opened, stats)

        @llm_chat_callback()
        def stream_chat(self, messages, **kwargs):
            generator = self._stream(list(messages or ()))

            def replies():
                parts: list[str] = []
                while True:
                    try:
                        delta = next(generator)
                    except StopIteration as stop:
                        opened, stats = stop.value or (False, {})
                        break
                    parts.append(delta)
                    yield ChatResponse(
                        message=ChatMessage(role=MessageRole.ASSISTANT, content="".join(parts)),
                        delta=delta,
                    )
                yield self._reply("".join(parts), opened, stats)

            return replies()

        @llm_completion_callback()
        def complete(self, prompt: str, formatted: bool = False, **kwargs) -> CompletionResponse:
            reply = self.chat([ChatMessage(role=MessageRole.USER, content=str(prompt))])
            return CompletionResponse(
                text=messages.text_of(reply.message),
                raw=reply.raw,
                additional_kwargs={"thinking": messages.thinking_of(reply.message)},
            )

        @llm_completion_callback()
        def stream_complete(self, prompt: str, formatted: bool = False, **kwargs):
            replies = self.stream_chat([ChatMessage(role=MessageRole.USER, content=str(prompt))])

            def completions():
                text = ""
                for reply in replies:
                    if reply.delta:
                        text += reply.delta
                        yield CompletionResponse(text=text, delta=reply.delta)
                    else:
                        yield CompletionResponse(
                            text=messages.text_of(reply.message),
                            delta="",
                            raw=reply.raw,
                            additional_kwargs={"thinking": messages.thinking_of(reply.message)},
                        )

            return completions()

    _classes["base"] = LocalLLM
    return LocalLLM


def family_of(clip) -> Family:
    """How a CLIP's tokenizer is driven, read from the tokenizer and the model.

    Args:
        clip: A ComfyUI ``CLIP``.

    Returns:
        The :class:`Family`.
    """
    tokenizer = getattr(clip, "tokenizer", None)
    name = str(getattr(tokenizer, "clip_name", "") or "")
    lowered = name.lower()
    template = str(getattr(tokenizer, "llama_template", "") or "")
    sees = _model_sees(clip)
    if "<|im_start|>" in template or lowered.startswith("qwen"):
        if lowered.startswith("qwen35"):
            think = "open"
        elif lowered.startswith("qwen3") and not lowered.startswith("qwen3vl"):
            think = "hybrid"
        else:
            think = ""
        vision = sees and hasattr(tokenizer, "llama_template_images")
        return Family(CHATML, think, vision, True, name, CHATML_CONTROLS)
    if lowered.startswith(("mistral", "ministral")):
        return Family(MISTRAL, "", False, False, name, MISTRAL_CONTROLS)
    if lowered.startswith("gemma4"):
        return Family(TEMPLATE, "", sees, True, name, GEMMA4_CONTROLS)
    if lowered.startswith("gemma"):
        return Family(TEMPLATE, "", sees, False, name, GEMMA3_CONTROLS)
    return Family(TEMPLATE, "", False, False, name, ())


def _inner(tokenizer):
    """The tokenizer object inside a ComfyUI ``SD1Tokenizer``, or ``None``."""
    key = getattr(tokenizer, "clip", None)
    return getattr(tokenizer, key, None) if isinstance(key, str) else None


def generator_of(clip):
    """The transformer whose ``generate`` writes text inside a CLIP, or ``None``.

    Args:
        clip: A ComfyUI ``CLIP``.

    Returns:
        The transformer module.
    """
    from ..model import text_decode

    transformer, _ = text_decode.decoder_of(clip)
    if transformer is not None:
        return transformer
    cond = getattr(clip, "cond_stage_model", None)
    for child in getattr(cond, "_modules", {}).values():
        inner = getattr(child, "transformer", None)
        if inner is not None and callable(getattr(inner, "generate", None)):
            return inner
    return None


def _model_sees(clip) -> bool:
    """Whether the CLIP's generating transformer carries a vision tower."""
    transformer = generator_of(clip)
    return transformer is not None and any(
        getattr(transformer, part, None) is not None for part in VISION_PARTS
    )


def context_of(clip) -> int:
    """The context window offered for a CLIP's language model.

    Args:
        clip: A ComfyUI ``CLIP``.

    Returns:
        The model's own length from its config, at most :data:`MAX_CONTEXT`.
    """
    transformer = generator_of(clip)
    config = getattr(getattr(transformer, "model", None), "config", None)
    length = getattr(config, "max_position_embeddings", None)
    if not isinstance(length, int) or length <= 0:
        return DEFAULT_CONTEXT
    return min(length, MAX_CONTEXT)


def _stop_token(transformer):
    """The token that ends a generation, from the model's config, or ``None``."""
    config = getattr(getattr(transformer, "model", None), "config", None)
    tokens = getattr(config, "stop_tokens", None)
    if isinstance(tokens, (list, tuple)) and tokens:
        return int(tokens[-1])
    return None


@contextlib.contextmanager
def _no_embeddings(tokenizer):
    """Turn textual inversion lookups off on a CLIP tokenizer for the length of a block."""
    inner = _inner(tokenizer)
    had = inner is not None and hasattr(inner, "embedding_directory")
    saved = getattr(inner, "embedding_directory", None) if had else None
    if had:
        inner.embedding_directory = None
    try:
        yield
    finally:
        if had:
            inner.embedding_directory = saved


@contextlib.contextmanager
def _watch(transformer, clip, label: str):
    """Watch each sampled token: draw the text so far and end the generation on an interrupt.

    Args:
        transformer: The generating transformer, or ``None``.
        clip: The CLIP, whose tokenizer decodes the tokens.
        label: The line drawn above the text.

    Yields:
        A dict whose ``interrupted`` turns True when the run was cancelled.
    """
    state = {"ids": [], "interrupted": False, "shown": ""}
    sample = getattr(transformer, "sample_token", None) if transformer is not None else None
    if sample is None:
        yield state
        return
    try:
        import comfy.model_management as management

        interrupted = management.processing_interrupted
    except ImportError:
        def interrupted():
            return False
    stop = _stop_token(transformer)
    stream = runtime.TextStream(label)
    marks = {"drawn": 0.0}

    def draw() -> None:
        text = clip.decode(state["ids"])
        if text.startswith(state["shown"]):
            stream.write(text[len(state["shown"]):])
            state["shown"] = text

    def watched(*args, **kwargs):
        token = sample(*args, **kwargs)
        if interrupted():
            state["interrupted"] = True
            if stop is not None:
                token.fill_(stop)
            return token
        state["ids"].append(int(token.reshape(-1)[0].item()))
        now = time.monotonic()
        if now - marks["drawn"] >= LIVE_INTERVAL:
            marks["drawn"] = now
            draw()
        return token

    had = "sample_token" in transformer.__dict__
    saved = transformer.__dict__.get("sample_token")
    setattr(transformer, "sample_token", watched)
    try:
        yield state
    finally:
        if had:
            setattr(transformer, "sample_token", saved)
        else:
            transformer.__dict__.pop("sample_token", None)
        if not state["interrupted"] and state["ids"]:
            try:
                draw()
            except Exception as error:
                logger.debug("the last text could not be drawn (%s)", error)
        stream.finish()


def _strip_padding(tokens, tokenizer) -> None:
    """Drop pad tokens a tokenizer appended after the prompt, in place."""
    inner = _inner(tokenizer)
    pad = getattr(inner, "pad_token", None)
    if pad is None or getattr(inner, "pad_left", False) or not isinstance(tokens, dict):
        return
    for batches in tokens.values():
        for batch in batches:
            while len(batch) > 1 and isinstance(batch[-1][0], int) and batch[-1][0] == pad:
                batch.pop()


def _fitted(pictures: list, many: bool):
    """Pictures as one batch at the newest picture's size, or the newest alone.

    Args:
        pictures: ``(1, height, width, 3)`` tensors.
        many: True to keep every picture, False for the newest only.

    Returns:
        A ``(batch, height, width, 3)`` tensor, or ``None`` for no pictures.
    """
    if not pictures:
        return None
    if not many:
        return pictures[-1]
    import torch

    height, width = pictures[-1].shape[1:3]
    fitted = []
    for picture in pictures:
        if picture.shape[1:3] != (height, width):
            import comfy.utils

            picture = comfy.utils.common_upscale(
                picture.movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)
        fitted.append(picture)
    return torch.cat(fitted, dim=0)


def render_chatml(system: str, turns: list[Turn], family: Family, thinking: bool) -> tuple[str, bool]:
    """A conversation as Qwen ChatML text, ready for the model to answer.

    Args:
        system: The system text, empty for none.
        turns: The conversation, oldest first.
        family: The tokenizer family.
        thinking: True to let a reasoning model think first.

    Returns:
        ``(text, opened)``, ``opened`` True when the text ends inside a thinking block.
    """
    parts = [f"<|im_start|>system\n{system}<|im_end|>\n"] if system else []
    for turn in turns:
        if turn.role == ASSISTANT:
            parts.append(f"<|im_start|>assistant\n{turn.text}<|im_end|>\n")
        elif turn.role == TOOL:
            parts.append(f"<|im_start|>user\n<tool_response>\n{turn.text}\n</tool_response><|im_end|>\n")
        else:
            if family.vision:
                slots = VISION_BLOCK * len(turn.pictures)
            else:
                slots = "[picture] " * len(turn.pictures)
            parts.append(f"<|im_start|>user\n{slots}{turn.text}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    opened = False
    if family.think == "open":
        parts.append("<think>\n" if thinking else EMPTY_THINK)
        opened = thinking
    elif family.think == "hybrid" and not thinking:
        parts.append(EMPTY_THINK)
    return "".join(parts), opened


def render_mistral(system: str, turns: list[Turn]) -> str:
    """A conversation as one Mistral instruction, earlier turns as a transcript.

    Args:
        system: The system text, empty for none.
        turns: The conversation, oldest first.

    Returns:
        The text.
    """
    head = f"[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT]" if system else ""
    return f"{head}[INST]{transcript_prompt(turns, seen=False)}[/INST]"


def clip_model():
    """The llama-index chat model class around a ComfyUI CLIP.

    Returns:
        A :func:`local_base` subclass built with ``ClipLLM(clip, ...)``.
    """
    if "clip" in _classes:
        return _classes["clip"]
    base = local_base()
    from pydantic import PrivateAttr

    class ClipLLM(base):
        """The language model inside a ComfyUI CLIP."""

        fast_decode: bool = True
        thinking: bool = False
        sampling: dict = {}
        _clip = PrivateAttr(default=None)
        _family = PrivateAttr(default=None)

        def __init__(self, clip, **data):
            data.setdefault("draws", False)
            super().__init__(**data)
            self._clip = clip
            self._family = family_of(clip)

        @classmethod
        def class_name(cls) -> str:
            return "was_clip_llm"

        @property
        def family(self) -> Family:
            """How this CLIP's tokenizer is driven."""
            return self._family

        def count_tokens(self, text: str) -> int:
            """Tokens in a text, by the CLIP's own tokenizer."""
            inner = _inner(getattr(self._clip, "tokenizer", None))
            try:
                return len(inner.tokenizer(str(text or ""))["input_ids"])
            except Exception as error:
                logger.debug("the raw tokenizer failed, counting through tokenize (%s)", error)
            with _no_embeddings(self._clip.tokenizer):
                tokens = self._clip.tokenize(str(text or ""), llama_template="{}", skip_template=True, min_length=1)
            batches = next(iter(tokens.values())) if isinstance(tokens, dict) else tokens
            return sum(len(batch) for batch in batches)

        def tokens(self, items) -> tuple:
            """Tokenize a conversation the way this CLIP's model reads one.

            Args:
                items: llama-index ``ChatMessage`` objects.

            Returns:
                ``(tokens, opened)``: what ``clip.tokenize`` answered, and whether the prompt
                ends inside a thinking block.
            """
            family = self._family
            system, turns = conversation(items, family.controls)
            if not any(turn.role == USER for turn in turns):
                turns.append(Turn(USER, ""))
            pictures = [
                tensor for turn in turns for tensor in messages.image_tensors(turn.pictures)
            ] if family.vision else []
            clip = self._clip
            with _no_embeddings(clip.tokenizer):
                if family.kind == CHATML:
                    text, opened = render_chatml(system, turns, family, self.thinking)
                    tokens = clip.tokenize(
                        text, llama_template="{}", images=pictures, skip_template=True,
                        min_length=1, thinking=self.thinking,
                    )
                    _strip_padding(tokens, clip.tokenizer)
                    return tokens, opened
                if family.kind == MISTRAL:
                    text = render_mistral(system, turns)
                    return clip.tokenize(text, llama_template="{}", min_length=1), False
                prompt = transcript_prompt(turns, seen=family.vision and family.many)
                tokens = clip.tokenize(
                    prompt, image=_fitted(pictures, family.many), skip_template=False,
                    system_prompt=system, thinking=self.thinking, min_length=1,
                )
                return tokens, False

        def generate(self, items) -> tuple[str, bool, dict]:
            """Write a reply to a conversation.

            Args:
                items: llama-index ``ChatMessage`` objects.

            Returns:
                ``(raw, opened, stats)``: the decoded text, whether the prompt opened a thinking
                block, and figures describing the generation.

            Raises:
                InterruptProcessingException: The run was cancelled.
                RuntimeError: The model failed.
            """
            from ..model import text_decode

            clip = self._clip
            tokens, opened = self.tokens(items)
            sampling = dict(self.sampling or {})
            options = {
                "do_sample": bool(sampling.get("on", False)),
                "temperature": float(sampling.get("temperature", 1.0)),
                "top_k": int(sampling.get("top_k", 50)),
                "top_p": float(sampling.get("top_p", 1.0)),
                "min_p": float(sampling.get("min_p", 0.0)),
                "repetition_penalty": float(sampling.get("repetition_penalty", 1.0)),
                "presence_penalty": float(sampling.get("presence_penalty", 0.0)),
                "seed": sampling.get("seed"),
            }
            runtime.check_interrupt()
            transformer = generator_of(clip)
            with _watch(transformer, clip, self.model_label) as watch:
                try:
                    ids, report = text_decode.generate(
                        clip, tokens, int(self.max_tokens), fast=bool(self.fast_decode), **options
                    )
                except BaseException:
                    _release_graphs()
                    raise
            if watch["interrupted"]:
                runtime.check_interrupt()
            raw = clip.decode(ids)
            stats = {
                "tokens": report.new_tokens,
                "prompt_tokens": report.prompt_tokens,
                "seconds": round(report.seconds, 3),
                "tokens_per_second": round(report.tokens_per_second, 2),
                "decode": report.path,
            }
            if report.reason:
                stats["why_not_fast"] = report.reason
            logger.info(
                "%s wrote %d token(s) in %.2fs on %s",
                self.model_label, report.new_tokens, report.seconds, report.path,
            )
            return raw, opened, stats

        def _stream(self, items):
            """Generate, then replay the finished text as deltas."""
            raw, opened, stats = self.generate(items)
            yield from pieces(raw)
            return opened, stats

    _classes["clip"] = ClipLLM
    return ClipLLM


def _release_graphs() -> None:
    """Drop the decode graphs a generation left behind, ignoring an install without them."""
    try:
        from ..model import text_decode

        text_decode.release_graphs()
    except Exception as error:
        logger.debug("the decode graphs could not be released (%s)", error)


def release_clip(clip) -> None:
    """Move a CLIP's weights off the GPU and hand the cached memory back.

    Args:
        clip: A ComfyUI ``CLIP``.
    """
    _release_graphs()
    try:
        import comfy.model_management as management
    except ImportError:
        return
    patcher = getattr(clip, "patcher", None)
    if patcher is not None:
        management.unload_model_and_clones(patcher)
    management.soft_empty_cache()


def free_bytes() -> int | None:
    """Free memory on ComfyUI's device, or ``None`` outside ComfyUI."""
    try:
        import comfy.model_management as management

        return int(management.get_free_memory(management.get_torch_device()))
    except Exception as error:
        logger.debug("free memory is unknown (%s)", error)
        return None


def empty_cache() -> None:
    """Hand the device memory torch keeps cached back to the driver."""
    try:
        import comfy.model_management as management
    except ImportError:
        return
    management.soft_empty_cache()

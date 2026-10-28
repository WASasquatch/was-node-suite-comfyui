"""Conversions between ComfyUI values and llama-index chat messages.

A picture becomes one PNG ``ImageBlock`` per frame of an ``IMAGE`` batch, and sound a WAV
``AudioBlock``. Thinking is carried in ``ThinkingBlock`` objects or between ``<think>`` tags.
"""

from __future__ import annotations

import base64
import io
import re
import wave

from . import runtime

__all__ = [
    "MAX_IMAGES",
    "MAX_IMAGE_EDGE",
    "ROLES",
    "audio_block",
    "closes_only",
    "from_dicts",
    "image_blocks",
    "image_tensors",
    "message",
    "split_thinking",
    "storable",
    "text_of",
    "thinking_of",
    "to_dicts",
    "transcript",
]

#: The roles a message node offers.
ROLES = ("user", "system", "assistant")

#: Longest edge, in pixels, of a picture sent to a model.
MAX_IMAGE_EDGE = 2048

#: Longest edge, in pixels, of a picture kept in a saved conversation.
STORED_IMAGE_EDGE = 1024

#: Pictures one message carries at most.
MAX_IMAGES = 64

#: Seconds of sound one message carries at most.
MAX_AUDIO_SECONDS = 600

#: A thinking block a reasoning model writes inline, and an unclosed one at the start.
_THINK = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_OPEN_THINK = re.compile(r"^\s*<think>(.*)$", re.DOTALL)
_CLOSE_ONLY = re.compile(r"^(.*?)</think>", re.DOTALL)


def image_blocks(image, max_edge: int = MAX_IMAGE_EDGE) -> list:
    """One PNG ``ImageBlock`` per frame of an ``IMAGE`` batch.

    Args:
        image: A tensor shaped ``(batch, height, width, channels)`` in ``[0, 1]``, or ``None``.
        max_edge: Longest edge a frame is reduced to, keeping its shape.

    Returns:
        The blocks, empty for ``None``.

    Raises:
        ValueError: The batch holds more than :data:`MAX_IMAGES` frames.
    """
    if image is None:
        return []
    import numpy as np
    from PIL import Image

    runtime.core()
    from llama_index.core.llms import ImageBlock

    frames = image if image.ndim == 4 else image[None]
    if frames.shape[0] > MAX_IMAGES:
        raise ValueError(
            f"{frames.shape[0]} pictures were connected and one message carries at most "
            f"{MAX_IMAGES}. Pick fewer frames, such as with Image Batch Get."
        )
    blocks = []
    for frame in frames:
        values = (frame.detach().float().clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
        if values.shape[-1] == 1:
            picture = Image.fromarray(values[..., 0], mode="L").convert("RGB")
        else:
            picture = Image.fromarray(values[..., :3], mode="RGB")
        if max(picture.size) > max_edge:
            picture.thumbnail((max_edge, max_edge), Image.LANCZOS)
        buffer = io.BytesIO()
        picture.save(buffer, format="PNG")
        blocks.append(ImageBlock(image=buffer.getvalue(), image_mimetype="image/png"))
    return blocks


def image_tensors(blocks) -> list:
    """The pictures in content blocks, each as a ``(1, height, width, 3)`` tensor.

    Args:
        blocks: llama-index content blocks; only ``ImageBlock`` objects holding bytes are read.

    Returns:
        Tensors in the order the blocks hold them.
    """
    import numpy as np
    import torch
    from PIL import Image

    from llama_index.core.llms import ImageBlock

    pictures = []
    for block in blocks or ():
        if not isinstance(block, ImageBlock) or not block.image:
            continue
        data = block.resolve_image(as_base64=False).read()
        with Image.open(io.BytesIO(data)) as opened:
            picture = opened.convert("RGB")
        values = np.asarray(picture, dtype=np.float32) / 255.0
        pictures.append(torch.from_numpy(values)[None])
    return pictures


def audio_block(audio):
    """A WAV ``AudioBlock`` from ComfyUI ``AUDIO``, the first clip of a batch.

    Args:
        audio: ``{"waveform": (batch, channels, samples), "sample_rate": int}``, or ``None``.

    Returns:
        The block, or ``None`` for no sound.

    Raises:
        ValueError: The clip is longer than :data:`MAX_AUDIO_SECONDS`.
    """
    if not audio:
        return None
    import numpy as np

    runtime.core()
    from llama_index.core.llms import AudioBlock

    waveform = audio["waveform"]
    rate = int(audio["sample_rate"])
    clip = waveform[0] if waveform.ndim == 3 else waveform
    if clip.shape[-1] > MAX_AUDIO_SECONDS * rate:
        raise ValueError(
            f"The sound is {clip.shape[-1] / rate:.0f} seconds long and one message carries at "
            f"most {MAX_AUDIO_SECONDS}. Trim it first, such as with Audio Split."
        )
    channels = min(int(clip.shape[0]), 2)
    samples = (clip[:channels].detach().float().clamp(-1, 1).cpu().numpy() * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(samples.T.tobytes())
    return AudioBlock(audio=buffer.getvalue(), format="wav")


def message(role: str, text: str = "", images=None, audio=None):
    """A llama-index ``ChatMessage`` from text, pictures and sound.

    Args:
        role: One of :data:`ROLES`.
        text: The message's words.
        images: An ``IMAGE`` batch, or ``None``.
        audio: ComfyUI ``AUDIO``, or ``None``.

    Returns:
        The message, its text first, then each picture, then the sound.
    """
    runtime.core()
    from llama_index.core.llms import ChatMessage, MessageRole, TextBlock

    blocks = []
    if text:
        blocks.append(TextBlock(text=str(text)))
    blocks.extend(image_blocks(images))
    sound = audio_block(audio)
    if sound is not None:
        blocks.append(sound)
    if not blocks:
        blocks.append(TextBlock(text=""))
    return ChatMessage(role=MessageRole(role), blocks=blocks)


def text_of(value) -> str:
    """The words of a message, a response or a block list, thinking left out."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    inner = getattr(value, "message", None)
    if inner is not None:
        return text_of(inner)
    blocks = getattr(value, "blocks", None)
    if blocks is not None:
        from llama_index.core.llms import TextBlock

        return "".join(block.text for block in blocks if isinstance(block, TextBlock))
    text = getattr(value, "text", None)
    if isinstance(text, str):
        return text
    response = getattr(value, "response", None)
    return response if isinstance(response, str) else str(value)


def thinking_of(value) -> str:
    """The thinking a response or message carries in ``ThinkingBlock`` objects."""
    inner = getattr(value, "message", None)
    if inner is not None:
        value = inner
    from llama_index.core.base.llms.types import ThinkingBlock

    parts = [block.content or "" for block in getattr(value, "blocks", ()) or () if isinstance(block, ThinkingBlock)]
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def closes_only(text: str) -> bool:
    """Whether a text closes a thinking block it never opened, as after a prompt that opened it.

    Args:
        text: What the model wrote.

    Returns:
        True when ``</think>`` appears and ``<think>`` does not.
    """
    text = str(text or "")
    return "</think>" in text and "<think>" not in text


def split_thinking(text: str, opened: bool = False) -> tuple[str, str]:
    """An answer and the thinking a model wrote inline between ``<think>`` tags.

    Args:
        text: What the model wrote.
        opened: True when the prompt already opened the thinking block, so the text starts
            inside it and only its close is written.

    Returns:
        ``(answer, thinking)``, both stripped.
    """
    text = str(text or "")
    if opened:
        closed = _CLOSE_ONLY.match(text)
        if closed:
            return text[closed.end():].strip(), closed.group(1).strip()
        return "", text.strip()
    thoughts = [match.strip() for match in _THINK.findall(text)]
    answer = _THINK.sub("", text)
    unclosed = _OPEN_THINK.match(answer)
    if unclosed:
        thoughts.append(unclosed.group(1).strip())
        answer = ""
    return answer.strip(), "\n\n".join(part for part in thoughts if part)


def transcript(messages, limit: int = 0) -> str:
    """A conversation as plain text, one ``role: text`` paragraph per message.

    Args:
        messages: llama-index ``ChatMessage`` objects.
        limit: Keep only the newest this many messages, 0 for all.

    Returns:
        The text, pictures and sound noted in brackets.
    """
    from llama_index.core.llms import AudioBlock, ImageBlock

    items = list(messages or ())
    if limit > 0:
        items = items[-limit:]
    lines = []
    for item in items:
        role = getattr(item.role, "value", str(item.role))
        notes = []
        pictures = sum(isinstance(block, ImageBlock) for block in item.blocks)
        if pictures:
            notes.append(f"[{pictures} picture{'s' if pictures != 1 else ''}]")
        if any(isinstance(block, AudioBlock) for block in item.blocks):
            notes.append("[sound]")
        body = text_of(item).strip()
        lines.append(f"{role}: {' '.join([*notes, body]).strip()}")
    return "\n\n".join(lines)


def to_dicts(messages) -> list[dict]:
    """Messages as JSON-ready dictionaries."""
    return [item.model_dump(mode="json") for item in messages or ()]


def from_dicts(rows) -> list:
    """Messages read back from :func:`to_dicts` rows, skipping any that do not parse."""
    runtime.core()
    from llama_index.core.llms import ChatMessage

    found = []
    for row in rows or ():
        if not isinstance(row, dict):
            continue
        try:
            found.append(ChatMessage.model_validate(row))
        except Exception as error:
            runtime.logger.debug("a saved message did not parse (%s)", error)
    return found


def storable(messages) -> list:
    """Messages with every picture reduced to a JPEG no larger than :data:`STORED_IMAGE_EDGE`."""
    from PIL import Image

    from llama_index.core.llms import ImageBlock

    kept = []
    for item in messages or ():
        blocks = []
        changed = False
        for block in item.blocks:
            if isinstance(block, ImageBlock) and block.image:
                try:
                    data = base64.b64decode(block.image)
                    with Image.open(io.BytesIO(data)) as opened:
                        picture = opened.convert("RGB")
                    if max(picture.size) > STORED_IMAGE_EDGE:
                        picture.thumbnail((STORED_IMAGE_EDGE, STORED_IMAGE_EDGE), Image.LANCZOS)
                    buffer = io.BytesIO()
                    picture.save(buffer, format="JPEG", quality=90)
                    blocks.append(ImageBlock(image=buffer.getvalue(), image_mimetype="image/jpeg"))
                    changed = True
                    continue
                except Exception as error:
                    runtime.logger.debug("a picture was kept as it was (%s)", error)
            blocks.append(block)
        kept.append(item.model_copy(update={"blocks": blocks}) if changed else item)
    return kept

"""GGUF language models through llama-cpp-python, as llama-index chat models.

:func:`read_header` reads a GGUF file's metadata without loading it, :func:`load` keeps at most
:data:`CACHE_LIMIT` models loaded, and :func:`gguf_model` builds the chat model class.
"""

from __future__ import annotations

import gc
import os
import struct
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from .. import deps, log
from . import runtime
from .clip_llm import ASSISTANT, TOOL, conversation, local_base

__all__ = [
    "AUTO",
    "CACHE_LIMIT",
    "CHAT_FORMATS",
    "LoadSpec",
    "Loaded",
    "cache_bytes",
    "gguf_model",
    "load",
    "locate",
    "menu_name",
    "model_choices",
    "projector_choices",
    "read_header",
    "release",
    "require",
]

logger = log.get_logger("llm.models")

#: The chat format read from the GGUF file itself.
AUTO = "auto"

#: Text chat formats llama-cpp-python 0.3.20 registers, the tool-calling ones left out.
TEXT_FORMATS = (
    "alpaca", "baichuan", "baichuan-2", "chatglm3", "chatml", "gemma", "intel", "llama-2",
    "llama-3", "mistral-instruct", "mistrallite", "oasst_llama", "open-orca", "openbuddy",
    "openchat", "phind", "pygmalion", "qwen", "redpajama-incite", "saiga", "snoozy", "vicuna",
    "zephyr",
)

#: Picture chat formats, each the llama-cpp-python 0.3.20 handler class that reads an mmproj.
VISION_FORMATS = {
    "llava-1-5": "Llava15ChatHandler",
    "llava-1-6": "Llava16ChatHandler",
    "moondream2": "MoondreamChatHandler",
    "nanollava": "NanoLlavaChatHandler",
    "llama-3-vision-alpha": "Llama3VisionAlphaChatHandler",
    "minicpm-v-2.6": "MiniCPMv26ChatHandler",
    "qwen2.5-vl": "Qwen25VLChatHandler",
    "obsidian": "ObsidianChatHandler",
}

#: Every choice the chat_format widget offers.
CHAT_FORMATS = [AUTO, *TEXT_FORMATS, *VISION_FORMATS]

#: The model folders searched for ``.gguf`` files, by ``folder_paths`` key.
FOLDERS = ("LLM", "text_encoders")

#: The ``folder_paths`` key listing ``.gguf`` files across :data:`FOLDERS`.
LISTING_KEY = "was_llm_gguf"

#: Models kept loaded before the least recently used one is closed.
CACHE_LIMIT = 2

#: Keys one GGUF header may declare.
MAX_KEYS = 100_000

#: Bytes of one string a GGUF header may declare.
MAX_STRING = 16 * 1024 * 1024

#: Characters of a header string kept in the metadata answered.
KEPT_STRING = 4096

#: Entries of a numeric header array kept in the metadata answered.
KEPT_ARRAY = 4096

#: The key of a user message listing the picture addresses its text holds, in order.
PICTURE_PART = "was_pictures"

_SCALARS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q",
    11: "<q", 12: "<d",
}
_STRING, _ARRAY = 8, 9

_loaded: OrderedDict = OrderedDict()
_classes: dict = {}


def require():
    """``llama_cpp``.

    Raises:
        DependencyError: llama-cpp-python is not installed or will not import.
    """
    module = deps.optional("llama_cpp")
    if module is None:
        raise deps.DependencyError(
            "LLM GGUF Loader needs llama-cpp-python, which is not installed in the python that "
            "runs ComfyUI. Install a build that matches this machine: a CUDA build for an NVIDIA "
            "GPU, picked to match the installed CUDA version from the prebuilt wheels listed in "
            "the llama-cpp-python README, or the CPU build. features.llm turned this node on."
        )
    return module


def _register() -> bool:
    """Point :data:`LISTING_KEY` at every folder of :data:`FOLDERS`, ``.gguf`` files only."""
    try:
        import folder_paths
    except ImportError:
        return False
    from ..model import model_directories

    model_directories("LLM")
    paths: list[str] = []
    for name in FOLDERS:
        if name in folder_paths.folder_names_and_paths:
            for path in folder_paths.get_folder_paths(name):
                if path not in paths:
                    paths.append(path)
    current = folder_paths.folder_names_and_paths.get(LISTING_KEY)
    if current is None or list(current[0]) != paths:
        folder_paths.folder_names_and_paths[LISTING_KEY] = (paths, {".gguf"})
    return True


def _names() -> list[str]:
    """Every ``.gguf`` file ComfyUI sees across :data:`FOLDERS`, by name."""
    if not _register():
        return []
    import folder_paths

    return list(folder_paths.get_filename_list(LISTING_KEY))


def model_choices() -> list[str]:
    """The ``.gguf`` language models a menu lists, projector files left out."""
    return [name for name in _names() if "mmproj" not in os.path.basename(name).lower()]


def projector_choices() -> list[str]:
    """The ``.gguf`` picture projector files a menu lists, each with ``mmproj`` in its name."""
    return [name for name in _names() if "mmproj" in os.path.basename(name).lower()]


def locate(name: str) -> str | None:
    """The file a menu name stands for.

    Args:
        name: A name from :func:`model_choices` or :func:`projector_choices`.

    Returns:
        The full path, or ``None`` when no folder holds it or the name leaves its folder.
    """
    from ..util.sandbox import leaves_folder

    if not name or leaves_folder(name) is not None or not name.lower().endswith(".gguf"):
        return None
    if not _register():
        return None
    import folder_paths

    return folder_paths.get_full_path(LISTING_KEY, name)


def menu_name(value) -> str:
    """The name the menus list a ``.gguf`` file under, from a menu name or a wired full path.

    Args:
        value: A name from :func:`model_choices` or :func:`projector_choices`, or a full path.

    Returns:
        The value itself when it is a name, otherwise the path below the folder holding it.

    Raises:
        PathNotAllowed: The path is not a ``.gguf`` file inside the LLM or text_encoders
            folders, or a file of that name in an earlier folder is the one that loads.
    """
    from ..io import picker
    from ..util.sandbox import PathNotAllowed

    _register()
    try:
        return picker.model_name(value, (LISTING_KEY,))
    except PathNotAllowed as error:
        # A refusal names the .gguf files and the LLM folder rather than the listing key.
        text = str(error).replace(f"a {LISTING_KEY} model", "a .gguf model")
        raise PathNotAllowed(text.replace(f"`{LISTING_KEY}`", "`LLM`")) from None


class _Reader:
    """Bounded reads from an open GGUF file."""

    def __init__(self, handle, size: int):
        self.handle = handle
        self.size = size

    def take(self, count: int) -> bytes:
        """The next ``count`` bytes, refused past the end of the file."""
        if count < 0 or self.handle.tell() + count > self.size:
            raise ValueError("the file ends before its header does")
        return self.handle.read(count)

    def skip(self, count: int) -> None:
        """Move past ``count`` bytes, refused past the end of the file."""
        if count < 0 or self.handle.tell() + count > self.size:
            raise ValueError("the file ends before its header does")
        self.handle.seek(count, os.SEEK_CUR)

    def number(self, kind: int):
        """One scalar of a GGUF value type."""
        layout = _SCALARS[kind]
        return struct.unpack(layout, self.take(struct.calcsize(layout)))[0]

    def string(self, keep: bool = True):
        """One length-prefixed string, or ``None`` when it is longer than :data:`KEPT_STRING`."""
        length = self.number(10)
        if length > MAX_STRING:
            raise ValueError("the header declares a string longer than the file allows")
        if not keep or length > KEPT_STRING * 4:
            self.skip(length)
            return None
        return self.take(length).decode("utf-8", errors="replace")

    def value(self, kind: int):
        """One value of any GGUF type, long strings and long arrays skipped as ``None``."""
        if kind in _SCALARS:
            return self.number(kind)
        if kind == _STRING:
            return self.string()
        if kind != _ARRAY:
            raise ValueError(f"the header holds a value of unknown type {kind}")
        inner = self.number(4)
        count = self.number(10)
        if count > self.size:
            raise ValueError("the header declares an array longer than the file")
        if inner in _SCALARS:
            width = struct.calcsize(_SCALARS[inner])
            if count > KEPT_ARRAY:
                self.skip(width * count)
                return None
            return [self.number(inner) for _ in range(count)]
        if inner == _STRING:
            for _ in range(count):
                self.string(keep=False)
            return None
        for _ in range(count):
            self.value(inner)
        return None


def read_header(path: str) -> dict:
    """The metadata a GGUF file declares, read without loading the model.

    Args:
        path: The ``.gguf`` file.

    Returns:
        Key to value. Strings past :data:`KEPT_STRING` characters and arrays past
        :data:`KEPT_ARRAY` entries are ``None``. Reading stops at the tokenizer keys once the
        architecture's own keys are in.

    Raises:
        ValueError: The file is not GGUF, or its header claims more than the file holds.
    """
    size = os.path.getsize(path)
    meta: dict = {}
    with open(path, "rb") as handle:
        reader = _Reader(handle, size)
        if reader.take(4) != b"GGUF":
            raise ValueError("the file does not start with the GGUF signature")
        version = reader.number(4)
        if version not in (2, 3):
            raise ValueError(f"GGUF version {version} is not one this reader knows")
        reader.number(10)
        count = reader.number(10)
        if count > MAX_KEYS:
            raise ValueError("the header declares more keys than any model carries")
        for _ in range(count):
            key = reader.string()
            if key is None:
                raise ValueError("the header holds a key longer than any model uses")
            arch = meta.get("general.architecture")
            if key.startswith("tokenizer.") and arch and f"{arch}.block_count" in meta:
                break
            meta[key] = reader.value(reader.number(4))
    return meta


def cache_bytes(meta: dict, n_ctx: int) -> int:
    """Bytes the attention cache takes at a context length, at 16 bits a value.

    Args:
        meta: What :func:`read_header` answered.
        n_ctx: The context length in tokens.

    Returns:
        The estimate, 0 when the header lacks the figures.
    """
    arch = meta.get("general.architecture")
    if not arch:
        return 0

    def figure(name):
        value = meta.get(f"{arch}.{name}")
        if isinstance(value, list):
            value = max((item for item in value if isinstance(item, (int, float))), default=None)
        return int(value) if isinstance(value, (int, float)) and value > 0 else None

    layers, heads, embed = figure("block_count"), figure("attention.head_count"), figure("embedding_length")
    kv_heads = figure("attention.head_count_kv") or heads
    if not layers or not heads or not kv_heads:
        return 0
    key_length = figure("attention.key_length") or (embed // heads if embed else None)
    if not key_length:
        return 0
    value_length = figure("attention.value_length") or key_length
    return int(layers) * int(n_ctx) * int(kv_heads) * (key_length + value_length) * 2


@dataclass(frozen=True)
class LoadSpec:
    """What one loaded GGUF model is built from.

    Attributes:
        name: The model's menu name.
        path: The model file.
        projector_name: The projector's menu name, empty for none.
        projector: The projector file, empty for none.
        n_ctx: The context length in tokens.
        gpu_layers: Layers offloaded to the GPU, -1 for every one, 0 for none.
        flash_attn: True to use flash attention.
    """

    name: str
    path: str
    projector_name: str = ""
    projector: str = ""
    n_ctx: int = 8192
    gpu_layers: int = 0
    flash_attn: bool = False

    @property
    def key(self) -> tuple:
        """The cache key: the files and the settings that change the loaded model."""
        return (self.path, self.n_ctx, self.gpu_layers, self.projector, self.flash_attn)


@dataclass
class Loaded:
    """One GGUF model held in memory.

    Attributes:
        llama: The ``llama_cpp.Llama``.
        spec: What it was built from.
        meta: Its header metadata.
        auto_format: The chat format llama-cpp-python chose from the file.
        handlers: Picture chat handlers built for it, by chat format.
        seconds: How long loading took.
    """

    llama: object
    spec: LoadSpec
    meta: dict
    auto_format: str
    handlers: dict = field(default_factory=dict)
    seconds: float = 0.0


def _shown(text: str, spec: LoadSpec) -> str:
    """Error text with the files' full paths replaced by their menu names."""
    text = runtime.describe_error(text)
    for path, name in ((spec.path, spec.name), (spec.projector, spec.projector_name)):
        if path:
            text = text.replace(path, name).replace(path.replace("\\", "/"), name)
    return text


def _close(loaded: Loaded) -> None:
    """Free a loaded model and its picture handlers."""
    for handler in list(loaded.handlers.values()):
        stack = getattr(handler, "_exit_stack", None)
        try:
            if stack is not None:
                stack.close()
        except Exception as error:
            logger.debug("a picture handler did not close cleanly (%s)", error)
    loaded.handlers.clear()
    close = getattr(loaded.llama, "close", None)
    try:
        if callable(close):
            close()
    except Exception as error:
        logger.debug("a GGUF model did not close cleanly (%s)", error)
    loaded.llama = None
    gc.collect()


def release(key: tuple) -> bool:
    """Close one loaded model.

    Args:
        key: A :attr:`LoadSpec.key`.

    Returns:
        True when a model was loaded under the key.
    """
    loaded = _loaded.pop(key, None)
    if loaded is None:
        return False
    _close(loaded)
    logger.info("closed GGUF model %s", loaded.spec.name)
    return True


def _available_memory(offload: bool) -> int:
    """Bytes free for the attention cache: system memory, plus free VRAM when offloading."""
    total = 0
    try:
        import psutil

        total += int(psutil.virtual_memory().available)
    except Exception as error:
        logger.debug("free system memory is unknown (%s)", error)
        return 0
    if offload:
        try:
            import comfy.model_management as management

            total += int(management.get_free_memory(management.get_torch_device()))
        except Exception as error:
            logger.debug("free VRAM is unknown (%s)", error)
    return total


def load(spec: LoadSpec, seed: int = 0) -> tuple[Loaded, bool]:
    """The loaded model for a spec, loading it when it is not held.

    Args:
        spec: What to load.
        seed: The seed the model starts with.

    Returns:
        ``(loaded, fresh)``, ``fresh`` True when this call loaded it.

    Raises:
        DependencyError: llama-cpp-python is missing.
        ValueError: The file is unreadable, or its attention cache would not fit in memory.
        RuntimeError: llama-cpp-python failed to load it.
    """
    held = _loaded.pop(spec.key, None)
    if held is not None:
        _loaded[spec.key] = held
        return held, False
    llama_cpp = require()
    try:
        meta = read_header(spec.path)
    except (OSError, ValueError, struct.error) as error:
        raise ValueError(f"{spec.name} cannot be read as a GGUF model: {error}") from None
    offload = spec.gpu_layers != 0
    need = cache_bytes(meta, spec.n_ctx)
    free = _available_memory(offload)
    if need and free and need > free:
        raise ValueError(
            f"n_ctx {spec.n_ctx} needs about {need / 2**30:.1f} GB for {spec.name}'s attention "
            f"cache, and {free / 2**30:.1f} GB of memory is free. Lower n_ctx, such as to 8192."
        )
    while len(_loaded) >= CACHE_LIMIT:
        _, oldest = _loaded.popitem(last=False)
        logger.info("closing GGUF model %s to stay under %d loaded", oldest.spec.name, CACHE_LIMIT)
        _close(oldest)
    if offload:
        _make_room(spec, need)
    started = time.perf_counter()
    try:
        llama = llama_cpp.Llama(
            model_path=spec.path,
            n_ctx=int(spec.n_ctx),
            n_batch=min(512, int(spec.n_ctx)),
            n_gpu_layers=int(spec.gpu_layers),
            flash_attn=bool(spec.flash_attn),
            seed=int(seed) & 0xFFFFFFFF,
            verbose=False,
        )
    except Exception as error:
        raise RuntimeError(f"LLM GGUF Loader could not load {spec.name}: {_shown(error, spec)}") from None
    loaded = Loaded(
        llama=llama, spec=spec, meta=meta, auto_format=str(getattr(llama, "chat_format", "") or ""),
        seconds=time.perf_counter() - started,
    )
    _loaded[spec.key] = loaded
    logger.info(
        "loaded GGUF model %s in %.2fs (n_ctx %d, gpu layers %d)",
        spec.name, loaded.seconds, spec.n_ctx, spec.gpu_layers,
    )
    return loaded, True


def _make_room(spec: LoadSpec, cache: int) -> None:
    """Ask ComfyUI to free VRAM for a model about to be offloaded to the GPU."""
    try:
        import comfy.model_management as management
    except ImportError:
        return
    size = os.path.getsize(spec.path) + (os.path.getsize(spec.projector) if spec.projector else 0)
    try:
        management.free_memory(int(size + cache), management.get_torch_device())
        management.soft_empty_cache()
    except Exception as error:
        logger.debug("ComfyUI could not free VRAM ahead of the GGUF model (%s)", error)


def _template_handler(base, template: str):
    """A picture handler class that renders with the GGUF file's own chat template."""

    class TemplateChatHandler(base):
        """Reads pictures through an mmproj, prompts in the model's own chat format."""

        DEFAULT_SYSTEM_MESSAGE = None
        CHAT_FORMAT = template

        @staticmethod
        def get_image_urls(messages):
            found = []
            for message in messages:
                if isinstance(message, dict) and message.get("role") == "user":
                    found.extend(message.get(PICTURE_PART) or ())
            return found

    return TemplateChatHandler


def _data_uri(block) -> str:
    """A picture block as a ``data:`` address, the only kind a handler is handed."""
    data = block.resolve_image(as_base64=True).read()
    mimetype = block.image_mimetype or "image/png"
    if isinstance(data, bytes):
        data = data.decode("ascii")
    if not mimetype.startswith("image/"):
        mimetype = "image/png"
    return f"data:{mimetype};base64,{data}"


def gguf_model():
    """The llama-index chat model class around a GGUF model.

    Returns:
        A local model class built with ``GGUFLLM(spec, ...)``.
    """
    if "gguf" in _classes:
        return _classes["gguf"]
    base = local_base()
    from pydantic import PrivateAttr

    class GGUFLLM(base):
        """A GGUF language model running through llama-cpp-python."""

        chat_format: str = AUTO
        temperature: float = 0.7
        top_p: float = 0.95
        top_k: int = 40
        min_p: float = 0.05
        repeat_penalty: float = 1.1
        seed: int = 0
        _spec = PrivateAttr(default=None)

        def __init__(self, spec: LoadSpec, **data):
            super().__init__(**data)
            self._spec = spec

        @classmethod
        def class_name(cls) -> str:
            return "was_gguf_llm"

        @property
        def spec(self) -> LoadSpec:
            """What the model is loaded from."""
            return self._spec

        def loaded(self) -> Loaded:
            """The loaded model, loading it again after a release."""
            return load(self._spec, self.seed)[0]

        def count_tokens(self, text: str) -> int:
            """Tokens in a text, by the model's own tokenizer."""
            llama = self.loaded().llama
            return len(llama.tokenize(str(text or "").encode("utf-8"), add_bos=False, special=False))

        def payload(self, items, pictures: bool) -> tuple[list, int]:
            """A conversation as the message dictionaries llama-cpp-python takes.

            Args:
                items: llama-index ``ChatMessage`` objects.
                pictures: True to hand pictures over, False to note them in the text.

            Returns:
                ``(messages, count)``, ``count`` the pictures handed over.
            """
            system, turns = conversation(items)
            rows = [{"role": "system", "content": system}] if system else []
            count = 0
            template = self.chat_format == AUTO
            for turn in turns:
                if turn.role == ASSISTANT:
                    rows.append({"role": "assistant", "content": turn.text})
                elif turn.role == TOOL:
                    rows.append({"role": "user", "content": f"Tool result:\n{turn.text}"})
                elif turn.pictures and pictures:
                    addresses = [_data_uri(block) for block in turn.pictures]
                    count += len(addresses)
                    if template:
                        # Each address stands in the text where its picture goes.
                        text = "".join(f"{address}\n" for address in addresses) + turn.text
                        rows.append({"role": "user", "content": text, PICTURE_PART: addresses})
                    else:
                        parts = [{"type": "image_url", "image_url": {"url": address}} for address in addresses]
                        parts.append({"type": "text", "text": turn.text})
                        rows.append({"role": "user", "content": parts})
                else:
                    note = "[picture] " * len(turn.pictures)
                    rows.append({"role": "user", "content": f"{note}{turn.text}"})
            if not any(row["role"] == "user" for row in rows):
                rows.append({"role": "user", "content": ""})
            return rows, count

        def _handler(self, loaded: Loaded):
            """The picture handler for this chat format, built once per loaded model."""
            fmt = self.chat_format
            if fmt in loaded.handlers:
                return loaded.handlers[fmt]
            formats = require().llama_chat_format
            if fmt == AUTO:
                template = (getattr(loaded.llama, "metadata", {}) or {}).get("tokenizer.chat_template")
                if not template:
                    raise ValueError(
                        f"{loaded.spec.name} carries no chat template, so chat_format 'auto' cannot "
                        "place pictures. Pick the picture format the model was trained with, such "
                        "as llava-1-5 or qwen2.5-vl."
                    )
                handler_class = _template_handler(formats.Llava15ChatHandler, template)
            else:
                handler_class = getattr(formats, VISION_FORMATS[fmt])
            try:
                handler = handler_class(clip_model_path=loaded.spec.projector, verbose=False)
            except Exception as error:
                raise RuntimeError(
                    f"The mmproj {loaded.spec.projector_name} could not be read: "
                    f"{_shown(error, loaded.spec)}"
                ) from None
            loaded.handlers[fmt] = handler
            return handler

        def _stream(self, items):
            """Generate through ``create_chat_completion``, yielding text as it is written."""
            loaded = self.loaded()
            spec = loaded.spec
            sees = bool(spec.projector) and self.chat_format not in TEXT_FORMATS
            rows, count = self.payload(items, pictures=sees)
            llama = loaded.llama
            if count:
                llama.chat_handler = self._handler(loaded)
                llama.chat_format = loaded.auto_format
            else:
                llama.chat_handler = None
                llama.chat_format = (
                    self.chat_format if self.chat_format in TEXT_FORMATS else loaded.auto_format
                )
            runtime.check_interrupt()
            started = time.perf_counter()
            written = 0
            chunks = None
            try:
                chunks = llama.create_chat_completion(
                    messages=rows,
                    temperature=float(self.temperature),
                    top_p=float(self.top_p),
                    top_k=int(self.top_k),
                    min_p=float(self.min_p),
                    repeat_penalty=float(self.repeat_penalty),
                    seed=int(self.seed) & 0xFFFFFFFF,
                    max_tokens=int(self.max_tokens),
                    stream=True,
                )
                for chunk in chunks:
                    runtime.check_interrupt()
                    choices = chunk.get("choices") or [{}]
                    delta = (choices[0].get("delta") or {}).get("content") or ""
                    if delta:
                        written += 1
                        yield delta
            except (ValueError, RuntimeError, OSError) as error:
                raise RuntimeError(f"{spec.name} failed: {_shown(error, spec)}") from None
            finally:
                close = getattr(chunks, "close", None)
                if callable(close):
                    close()
            seconds = time.perf_counter() - started
            total = int(getattr(llama, "n_tokens", 0) or 0)
            stats = {
                "tokens": written,
                "prompt_tokens": max(0, total - written),
                "pictures": count,
                "seconds": round(seconds, 3),
                "tokens_per_second": round(written / seconds, 2) if seconds > 0 else 0.0,
            }
            logger.info("%s wrote %d token(s) in %.2fs", self.model_label, written, seconds)
            return False, stats

    _classes["gguf"] = GGUFLLM
    return GGUFLLM

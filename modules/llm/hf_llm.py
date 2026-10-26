"""Hugging Face transformers language models, as llama-index chat models.

:func:`choices` lists the model folders under ``models/LLM`` and the suggested repositories,
:func:`load` builds one under ComfyUI's model management and :func:`hf_model` the chat model class.
"""

from __future__ import annotations

import gc
import io
import os
import threading
import time
from dataclasses import dataclass

from .. import deps, log
from . import FEATURE, runtime
from .clip_llm import ASSISTANT, MAX_CONTEXT, TOOL, conversation, local_base

__all__ = [
    "ATTENTION",
    "DTYPES",
    "FOLDER",
    "Loaded",
    "QUANTIZE",
    "SUGGESTED",
    "Spec",
    "choices",
    "hf_model",
    "load",
    "local_folders",
    "locate",
    "release",
]

logger = log.get_logger("llm.models")

#: The ``folder_paths`` key whose folders hold transformers model folders.
FOLDER = "LLM"

#: Repositories offered for download, each Apache or MIT, ungated, and loadable without remote code.
SUGGESTED = (
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-VL-4B-Instruct",
    "HuggingFaceTB/SmolLM3-3B",
    "microsoft/Phi-4-mini-instruct",
    "ibm-granite/granite-4.0-micro",
)

#: The weight types offered, ``auto`` picking one for the device.
DTYPES = ["auto", "bfloat16", "float16", "float32"]

#: The quantizations offered.
QUANTIZE = ["none", "8-bit", "4-bit"]

#: The attention implementations offered.
ATTENTION = ["sdpa", "eager"]

#: The file that marks a folder as a transformers model.
MARKER = "config.json"

#: Seconds a folder listing is reused before the folders are read again.
LISTING_TTL = 30.0

#: Directory entries read per model folder root while listing.
MAX_EXAMINED = 4000

#: Bytes set aside beside the attention cache when a model is moved onto the GPU.
WORKING_BYTES = 512 * 1024 * 1024

_listing: dict = {"until": 0.0, "names": (), "downloaded": ()}
_held: dict = {}
_classes: dict = {}


def _folders_in(root) -> list[str]:
    """Model folders one or two levels under a root, links listed but never descended."""
    from ..util.file_listing import linked

    found: list[str] = []
    examined = 0
    try:
        outer = sorted(os.scandir(root), key=lambda entry: entry.name.lower())
    except OSError:
        return found
    for entry in outer:
        examined += 1
        if examined > MAX_EXAMINED:
            break
        if entry.name.startswith((".", "models--")):
            continue
        try:
            if not entry.is_dir():
                continue
        except OSError:
            continue
        if os.path.isfile(os.path.join(entry.path, MARKER)):
            found.append(entry.name)
            continue
        if linked(entry):
            continue
        try:
            inner = sorted(os.scandir(entry.path), key=lambda item: item.name.lower())
        except OSError:
            continue
        for item in inner:
            examined += 1
            if examined > MAX_EXAMINED:
                break
            if item.name.startswith("."):
                continue
            try:
                if item.is_dir() and os.path.isfile(os.path.join(item.path, MARKER)):
                    found.append(f"{entry.name}/{item.name}")
            except OSError:
                continue
    return found


def _read_listing() -> tuple[tuple, tuple]:
    """``(folders, downloaded)``, reused for :data:`LISTING_TTL` seconds.

    Returns:
        The model folder names, sorted, and the suggested repositories already downloaded into a
        model folder in Hugging Face cache layout.
    """
    now = time.monotonic()
    if now < _listing["until"]:
        return _listing["names"], _listing["downloaded"]
    from ..model import model_directories

    roots = model_directories(FOLDER)
    names: list[str] = []
    for root in roots:
        for name in _folders_in(root):
            if name not in names:
                names.append(name)
    names.sort(key=str.lower)
    downloaded = tuple(
        repo_id for repo_id in SUGGESTED
        if any(
            os.path.isdir(os.path.join(root, f"models--{repo_id.replace('/', '--')}", "snapshots"))
            for root in roots
        )
    )
    _listing.update(until=time.monotonic() + LISTING_TTL, names=tuple(names), downloaded=downloaded)
    return tuple(names), downloaded


def local_folders() -> list[str]:
    """Model folders under the ``LLM`` model folders, as ``name`` or ``owner/name``, sorted."""
    return list(_read_listing()[0])


def choices() -> list[str]:
    """What the model menu lists: local folders, then the suggested repositories.

    Returns:
        A suggestion is listed when ``features.network`` is on or it was downloaded already.
    """
    from ..model import network_enabled

    folders, downloaded = _read_listing()
    names = list(folders)
    online = network_enabled()
    for repo_id in SUGGESTED:
        if repo_id not in names and (online or repo_id in downloaded):
            names.append(repo_id)
    return names


def locate(label: str) -> tuple[str, str | None]:
    """Where a menu entry's weights are, or the repository to download them from.

    Args:
        label: A name from :func:`choices`.

    Returns:
        ``(pretrained, cache_dir)`` for ``from_pretrained``.

    Raises:
        ValueError: The name is not a model folder and not a suggested repository.
        ModelUnavailable: A suggested repository is not on disk and ``features.network`` is off.
    """
    from ..model import model_directories, repository_id, resolve

    label = repository_id(str(label or ""))
    for root in model_directories(FOLDER):
        candidate = os.path.join(root, *label.split("/"))
        if os.path.isfile(os.path.join(candidate, MARKER)):
            return candidate, None
    if label in SUGGESTED:
        return resolve(FOLDER, label, feature=FEATURE)
    raise ValueError(
        f"{label} is not a model folder under ComfyUI/models/LLM and not one of the suggested "
        "repositories. Put a Hugging Face model folder holding config.json and .safetensors "
        "weights there, then pick it from the model menu."
    )


@dataclass(frozen=True)
class Spec:
    """What one transformers model is built from.

    Attributes:
        label: The model's menu name.
        dtype: One of :data:`DTYPES`.
        quantize: One of :data:`QUANTIZE`.
        attention: One of :data:`ATTENTION`.
    """

    label: str
    dtype: str = "auto"
    quantize: str = "none"
    attention: str = "sdpa"

    @property
    def key(self) -> tuple:
        """The cache key."""
        return ("was_llm_transformers", self.label, self.dtype, self.quantize, self.attention)

    @property
    def short(self) -> str:
        """The model's own name, the owner left out."""
        return self.label.rsplit("/", 1)[-1]


@dataclass
class Loaded:
    """One transformers model held under ComfyUI's model management.

    Attributes:
        backend: The ``modules.model.Backend`` holding the weights.
        tokenizer: The tokenizer.
        processor: The picture processor of a picture model, ``None`` for a text model.
        spec: What it was built from.
        entry: Its key in the shared model cache.
        context: Its context window, at most :data:`MAX_CONTEXT`.
        cache_per_token: Bytes its attention cache grows by per token.
        parameters: Its parameter count.
        seconds: How long loading took.
    """

    backend: object
    tokenizer: object
    processor: object
    spec: Spec
    entry: tuple
    context: int
    cache_per_token: int
    parameters: int
    seconds: float

    @property
    def vision(self) -> bool:
        """True for a picture model."""
        return self.processor is not None

    @property
    def quantized(self) -> bool:
        """True when the weights are 8-bit or 4-bit and fixed to their device."""
        return self.spec.quantize != "none"


def _dtype(name: str, device):
    """The torch dtype a :data:`DTYPES` choice means on a device."""
    import torch

    if name == "auto":
        if device.type == "cpu":
            return torch.float32
        try:
            import comfy.model_management as management

            return torch.bfloat16 if management.should_use_bf16(device) else torch.float16
        except ImportError:
            return torch.float16
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _figures(config) -> tuple[int, int]:
    """``(context, cache_per_token)`` read from a model config, defaults where it is silent."""
    text = getattr(config, "text_config", None) or config
    context = getattr(text, "max_position_embeddings", None)
    context = min(int(context), MAX_CONTEXT) if isinstance(context, int) and context > 0 else 8192
    layers = getattr(text, "num_hidden_layers", 0) or 0
    heads = getattr(text, "num_attention_heads", 0) or 0
    kv_heads = getattr(text, "num_key_value_heads", None) or heads
    hidden = getattr(text, "hidden_size", 0) or 0
    head_dim = getattr(text, "head_dim", None) or (hidden // heads if heads else 0)
    return context, int(2 * layers * kv_heads * head_dim * 2)


def _friendly(error: BaseException, path: str, label: str) -> str:
    """A loading error's text, the folder named by its menu name, plain for the common refusals."""
    text = runtime.describe_error(error)
    if path and os.path.isabs(path):
        text = text.replace(path, label).replace(path.replace("\\", "/"), label)
    if "trust_remote_code" in text:
        return (
            "this model needs program code from its own repository, which this node does not "
            "run. Pick a model transformers supports on its own."
        )
    if "safetensors" in text.lower() and ("no file" in text.lower() or "not found" in text.lower()):
        return "the folder holds no .safetensors weights, and this node reads no other kind."
    return text


def load(spec: Spec) -> tuple[Loaded, bool]:
    """The loaded model for a spec, loading it when it is not held.

    Args:
        spec: What to load.

    Returns:
        ``(loaded, fresh)``, ``fresh`` True when this call loaded it.

    Raises:
        ValueError: The model is not on disk, or quantizing was asked for without a CUDA GPU.
        ModelUnavailable: A suggested repository is missing and ``features.network`` is off.
        RuntimeError: transformers failed to load it.
    """
    from ..model import Backend, cached, compute_device, managed

    held = _held.get(spec.key)
    if held is not None:
        return held, False
    pretrained, cache_dir = locate(spec.label)
    transformers = deps.require("transformers", feature=FEATURE)
    device = compute_device()
    dtype = _dtype(spec.dtype, device)
    weights = {"dtype": dtype, "attn_implementation": spec.attention}
    if spec.quantize != "none":
        if device.type != "cuda":
            raise ValueError(
                f"quantize {spec.quantize} needs an NVIDIA GPU, and ComfyUI runs on {device.type}. "
                "Set quantize to none."
            )
        deps.require("bitsandbytes", feature=FEATURE)
        deps.require("accelerate", feature=FEATURE)
        if spec.quantize == "8-bit":
            quantization = transformers.BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization = transformers.BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
            )
        weights.update(quantization_config=quantization, device_map={"": device})
    found: dict = {}

    def build():
        config = transformers.AutoConfig.from_pretrained(
            pretrained, cache_dir=cache_dir, token=False, trust_remote_code=False,
        )
        found["config"] = config
        if getattr(config, "vision_config", None) is not None:
            processor = transformers.AutoProcessor.from_pretrained(
                pretrained, cache_dir=cache_dir, token=False, trust_remote_code=False,
            )
            model = transformers.AutoModelForImageTextToText.from_pretrained(
                pretrained, cache_dir=cache_dir, token=False, trust_remote_code=False,
                use_safetensors=True, **weights,
            )
        else:
            processor = transformers.AutoTokenizer.from_pretrained(
                pretrained, cache_dir=cache_dir, token=False, trust_remote_code=False,
            )
            model = transformers.AutoModelForCausalLM.from_pretrained(
                pretrained, cache_dir=cache_dir, token=False, trust_remote_code=False,
                use_safetensors=True, **weights,
            )
        return processor, model

    started = time.perf_counter()
    name = f"LLM {spec.short}"
    try:
        if spec.quantize == "none":
            backend = managed(spec.key, build, name=name)
            entry = (*spec.key, str(backend.load_device))
        else:
            def quantized():
                processor, model = build()
                model.eval()
                return Backend(processor, model, device, device, None, key=spec.key, name=name)

            backend = cached(spec.key, quantized)
            entry = spec.key
    except deps.DependencyError:
        raise
    except Exception as error:
        raise RuntimeError(
            f"LLM Transformers Loader could not load {spec.label}: "
            f"{_friendly(error, pretrained, spec.label)}"
        ) from None
    config = found.get("config") or getattr(backend.model, "config", None)
    context, per_token = _figures(config)
    processor = backend.processor
    vision = hasattr(processor, "image_processor")
    tokenizer = getattr(processor, "tokenizer", None) if vision else processor
    loaded = Loaded(
        backend=backend,
        tokenizer=tokenizer,
        processor=processor if vision else None,
        spec=spec,
        entry=entry,
        context=context,
        cache_per_token=per_token,
        parameters=sum(parameter.numel() for parameter in backend.model.parameters()),
        seconds=time.perf_counter() - started,
    )
    _held[spec.key] = loaded
    logger.info("loaded transformers model %s in %.2fs", spec.label, loaded.seconds)
    return loaded, True


def release(key: tuple) -> bool:
    """Drop one loaded model from memory, VRAM and system memory both.

    Args:
        key: A :attr:`Spec.key`.

    Returns:
        True when a model was held under the key.
    """
    from .. import model as store

    loaded = _held.pop(key, None)
    if loaded is None:
        return False
    entry = loaded.entry
    loaded.backend = None
    loaded.tokenizer = None
    loaded.processor = None
    if not store.forget(entry):
        gc.collect()
    logger.info("released transformers model %s", loaded.spec.label)
    return True


def _picture(block):
    """A PIL picture from an ``ImageBlock`` holding bytes."""
    from PIL import Image

    data = block.resolve_image(as_base64=False).read()
    with Image.open(io.BytesIO(data)) as opened:
        return opened.convert("RGB")


def hf_model():
    """The llama-index chat model class around a transformers model.

    Returns:
        A local model class built with ``TransformersLLM(spec, ...)``.
    """
    if "hf" in _classes:
        return _classes["hf"]
    base = local_base()
    from pydantic import PrivateAttr

    class TransformersLLM(base):
        """A Hugging Face transformers language model under ComfyUI's model management."""

        thinking: bool = False
        temperature: float = 0.7
        top_p: float = 0.95
        top_k: int = 20
        min_p: float = 0.0
        repetition_penalty: float = 1.05
        seed: int = 0
        _spec = PrivateAttr(default=None)

        def __init__(self, spec: Spec, **data):
            super().__init__(**data)
            self._spec = spec

        @classmethod
        def class_name(cls) -> str:
            return "was_transformers_llm"

        @property
        def spec(self) -> Spec:
            """What the model is loaded from."""
            return self._spec

        def loaded(self) -> Loaded:
            """The loaded model, loading it again after a release."""
            return load(self._spec)[0]

        def count_tokens(self, text: str) -> int:
            """Tokens in a text, by the model's own tokenizer."""
            tokenizer = self.loaded().tokenizer
            return len(tokenizer(str(text or ""), add_special_tokens=False)["input_ids"])

        def prompt(self, items, loaded: Loaded) -> tuple[str, list]:
            """A conversation rendered by the model's chat template.

            Args:
                items: llama-index ``ChatMessage`` objects.
                loaded: The loaded model.

            Returns:
                ``(text, pictures)``: the prompt, and the PIL pictures it places, in order.
            """
            system, turns = conversation(items)
            sees = loaded.vision
            pictures: list = []
            rows: list = []

            def content(text):
                return [{"type": "text", "text": text}] if sees else text

            for turn in turns:
                if turn.role == ASSISTANT:
                    rows.append({"role": "assistant", "content": content(turn.text)})
                elif turn.role == TOOL:
                    rows.append({"role": "user", "content": content(f"Tool result:\n{turn.text}")})
                elif sees and turn.pictures:
                    parts = [{"type": "image"} for _ in turn.pictures]
                    parts.append({"type": "text", "text": turn.text})
                    pictures.extend(_picture(block) for block in turn.pictures)
                    rows.append({"role": "user", "content": parts})
                else:
                    note = "[picture] " * len(turn.pictures)
                    rows.append({"role": "user", "content": content(f"{note}{turn.text}")})
            if not any(row["role"] == "user" for row in rows):
                rows.append({"role": "user", "content": content("")})
            owner = loaded.processor if sees else loaded.tokenizer
            attempts = [([{"role": "system", "content": content(system)}] if system else []) + rows]
            if system:
                folded = [dict(row) for row in rows]
                first = next(row for row in folded if row["role"] == "user")
                if sees:
                    first["content"] = [{"type": "text", "text": system}, *first["content"]]
                else:
                    first["content"] = f"{system}\n\n{first['content']}"
                attempts.append(folded)
            failure = None
            if getattr(owner, "chat_template", None) or getattr(loaded.tokenizer, "chat_template", None):
                for conversation_rows in attempts:
                    try:
                        text = owner.apply_chat_template(
                            conversation_rows, tokenize=False, add_generation_prompt=True,
                            enable_thinking=bool(self.thinking),
                        )
                        return str(text), pictures
                    except Exception as error:
                        failure = error
                        logger.debug("the chat template refused the conversation (%s)", error)
            if sees:
                raise RuntimeError(
                    f"{loaded.spec.label}'s chat template could not place this conversation: "
                    f"{runtime.describe_error(failure or 'no chat template')}"
                )
            names = {"system": "System", "user": "User", "assistant": "Assistant"}
            lines = [f"System: {system}"] if system else []
            lines += [f"{names[row['role']]}: {row['content']}" for row in rows]
            return "\n\n".join(lines) + "\n\nAssistant:", pictures

        def _options(self, tokenizer) -> dict:
            """Keyword arguments for ``generate`` from the sampling settings."""
            sampled = float(self.temperature) > 0
            options = {
                "max_new_tokens": int(self.max_tokens),
                "do_sample": sampled,
                "repetition_penalty": float(self.repetition_penalty),
            }
            if sampled:
                options.update(temperature=float(self.temperature), top_p=float(self.top_p), top_k=int(self.top_k))
                if float(self.min_p) > 0:
                    options["min_p"] = float(self.min_p)
            pad = getattr(tokenizer, "pad_token_id", None)
            if pad is None:
                pad = getattr(tokenizer, "eos_token_id", None)
            if pad is not None:
                options["pad_token_id"] = pad
            return options

        def _stream(self, items):
            """Generate on a worker thread, yielding text as the streamer receives it."""
            import torch

            transformers = deps.require("transformers", feature=FEATURE)
            loaded = self.loaded()
            text, pictures = self.prompt(items, loaded)
            opened = text.rstrip().endswith("<think>")
            tokenizer = loaded.tokenizer
            if loaded.vision:
                inputs = loaded.processor(text=[text], images=pictures or None, return_tensors="pt")
            else:
                inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            prompt_tokens = int(inputs["input_ids"].shape[-1])
            if prompt_tokens + int(self.max_tokens) > loaded.context and prompt_tokens >= loaded.context:
                raise ValueError(
                    f"The conversation is {prompt_tokens} tokens and {loaded.spec.label} reads at most "
                    f"{loaded.context}. Shorten it, such as with a smaller memory token limit."
                )
            model = loaded.backend.model
            if loaded.quantized:
                device = next(model.parameters()).device
            else:
                memory = (prompt_tokens + int(self.max_tokens)) * loaded.cache_per_token + WORKING_BYTES
                device = loaded.backend.load(memory)
            dtype = next(model.parameters()).dtype
            moved = {}
            for name, value in dict(inputs).items():
                if hasattr(value, "to"):
                    value = value.to(device)
                    if name == "pixel_values" and value.is_floating_point():
                        value = value.to(dtype)
                moved[name] = value
            streamer = transformers.TextIteratorStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            halt = threading.Event()
            failure: list = []
            produced: list = []
            try:
                import comfy.model_management as management

                interrupted = management.processing_interrupted
            except ImportError:
                def interrupted():
                    return False

            class Halt(transformers.StoppingCriteria):
                """Ends the generation when the run is cancelled or the reader stops."""

                def __call__(self, input_ids, scores, **kwargs):
                    stop = halt.is_set() or interrupted()
                    return torch.full((input_ids.shape[0],), stop, dtype=torch.bool, device=input_ids.device)

            options = self._options(tokenizer)
            seed = int(self.seed) & 0xFFFFFFFFFFFFFFFF
            devices = [device.index or 0] if device.type == "cuda" else []

            def run():
                try:
                    with torch.random.fork_rng(devices=devices):
                        torch.manual_seed(seed)
                        with torch.inference_mode():
                            output = model.generate(
                                **moved, streamer=streamer,
                                stopping_criteria=transformers.StoppingCriteriaList([Halt()]),
                                **options,
                            )
                    produced.append(int(output.shape[-1]) - prompt_tokens)
                except BaseException as error:
                    failure.append(error)
                    streamer.end()

            started = time.perf_counter()
            worker = threading.Thread(target=run, name="was-llm-generate", daemon=True)
            worker.start()
            try:
                for piece in streamer:
                    if interrupted():
                        halt.set()
                        break
                    if piece:
                        yield piece
            finally:
                halt.set()
                worker.join()
            if failure:
                raise RuntimeError(f"{loaded.spec.label} failed: {runtime.describe_error(failure[0])}") from None
            runtime.check_interrupt()
            seconds = time.perf_counter() - started
            written = produced[0] if produced else 0
            stats = {
                "tokens": written,
                "prompt_tokens": prompt_tokens,
                "pictures": len(pictures),
                "seconds": round(seconds, 3),
                "tokens_per_second": round(written / seconds, 2) if seconds > 0 else 0.0,
                "device": str(device),
            }
            logger.info("%s wrote %d token(s) in %.2fs on %s", self.model_label, written, seconds, device)
            return opened, stats

    _classes["hf"] = TransformersLLM
    return TransformersLLM

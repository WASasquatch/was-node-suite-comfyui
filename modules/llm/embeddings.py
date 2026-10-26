"""Embedding models for the LLM nodes: transformers folders, GGUF files and hosted services.

:func:`local_choices` lists the folders a menu offers. Each ``*_embedding`` function returns an
:class:`~.wires.EmbeddingModel`; the local ones load their weights and measure the vector length.
"""

from __future__ import annotations

import inspect
import json
import math
import os
import re
import time
from itertools import islice

from .. import deps, log
from . import FEATURE, providers, runtime, wires

__all__ = [
    "GEMINI_DIMENSIONS",
    "NO_GGUF",
    "OPENAI_DIMENSIONS",
    "PICTURE_TYPES",
    "POOLINGS",
    "SUGGESTED",
    "bounded",
    "family_instructions",
    "folder_settings",
    "gemini_embedding",
    "gguf_choices",
    "gguf_embedding",
    "local_choices",
    "local_embedding",
    "model_type",
    "openai_embedding",
    "pool",
    "provider_embedding",
    "transformers_embedding",
]

logger = log.get_logger("llm.models")

#: The model folder local embedding checkpoints sit in, by ``folder_paths`` key.
FOLDER = "LLM"

#: The file every transformers checkpoint folder holds.
MARKER = "config.json"

#: Repository ids offered beside the local folders, fetched only with ``features.network`` on.
SUGGESTED = (
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "Qwen/Qwen3-Embedding-0.6B",
    "intfloat/multilingual-e5-large",
    "sentence-transformers/all-MiniLM-L6-v2",
    "mixedbread-ai/mxbai-embed-large-v1",
)

#: Files fetched from a suggested repository: configuration, tokenizer and safetensors weights.
DOWNLOAD_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "sentencepiece.bpe.model",
    "tokenizer.model",
    "spiece.model",
    "modules.json",
    "sentence_bert_config.json",
    "config_sentence_transformers.json",
    "1_Pooling/config.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "model-*.safetensors",
)

#: The pooling choices.
POOLINGS = ["auto", "cls", "mean", "last token"]

#: Seconds a folder listing is reused.
LISTING_TTL = 5.0

#: Folders a listing holds at most.
MAX_FOLDERS = 500

#: Entries read from one folder while listing.
MAX_ENTRIES = 10000

#: Bytes of a settings file read from a checkpoint folder.
MAX_SETTINGS_BYTES = 1_000_000

#: Characters of an instruction kept.
MAX_INSTRUCTION = 1000

#: Text embedded once to measure a local model's vector length.
PROBE = "dimension probe"

#: The entry a GGUF menu shows when no ``.gguf`` file is found.
NO_GGUF = "no .gguf file in models/LLM or models/text_encoders"

#: Vector length of OpenAI's embedding models at their native size.
OPENAI_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

#: Vector length of Gemini's embedding models at their native size.
GEMINI_DIMENSIONS = {"gemini-embedding-001": 3072}

#: The query instruction BGE English models and mxbai were trained with.
BGE_QUERY = "Represent this sentence for searching relevant passages: "

#: The query instruction Qwen3-Embedding was trained with.
QWEN3_QUERY = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
)

#: Model types that read left to right and pool on their last token.
DECODER_TYPES = frozenset({"qwen2", "qwen3", "llama", "mistral", "gemma", "gemma2", "gemma3", "phi3"})

_E5 = re.compile(r"(?:^|[/_\-])(?:multilingual-)?e5(?:[\-_]|$)")

_listing: tuple[float, tuple, tuple] = (0.0, (), ())
_classes: dict = {}


def bounded(value, default: int, low: int, high: int) -> int:
    """An integer setting held to its range.

    Args:
        value: What arrived, of any type.
        default: The answer when ``value`` is not a number.
        low: The smallest answer.
        high: The largest answer.

    Returns:
        The number, clamped.
    """
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return max(low, min(high, number))


def family_instructions(name: str) -> tuple[str, str]:
    """The query and text instructions a model family was trained with.

    Args:
        name: A repository id, folder or file name, such as ``BAAI/bge-small-en-v1.5``.

    Returns:
        ``(query, text)``, each ``""`` where the family uses none.
    """
    low = str(name).lower()
    if "qwen3-embedding" in low:
        return QWEN3_QUERY, ""
    if _E5.search(low):
        return "query: ", "passage: "
    if ("bge" in low and "m3" not in low) or "mxbai-embed" in low:
        return BGE_QUERY, ""
    return "", ""


def _read_json(path: str):
    """A small JSON file's contents, or ``None`` when it is missing, large or unreadable."""
    try:
        if os.path.getsize(path) > MAX_SETTINGS_BYTES:
            return None
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


def _pooling_of(settings) -> str:
    """The pooling a sentence-transformers pooling config names, ``""`` for none."""
    if not isinstance(settings, dict):
        return ""
    for key, mode in (
        ("pooling_mode_cls_token", "cls"),
        ("pooling_mode_lasttoken", "last token"),
        ("pooling_mode_mean_tokens", "mean"),
        ("pooling_mode_max_tokens", "max"),
    ):
        if settings.get(key) is True:
            return mode
    return ""


def folder_settings(folder: str) -> dict:
    """What a checkpoint folder declares about pooling and prompts.

    Args:
        folder: A transformers checkpoint folder.

    Returns:
        ``pooling`` (``""`` when undeclared), ``query`` and ``text`` prompts (``""`` when
        undeclared) and ``decoder``, True for a model read left to right.
    """
    from ..util.sandbox import leaves_folder

    found = {"pooling": "", "query": "", "text": "", "decoder": False}
    modules = _read_json(os.path.join(folder, "modules.json"))
    if isinstance(modules, list):
        for entry in modules[:16]:
            if not isinstance(entry, dict):
                continue
            path = str(entry.get("path") or "")
            if str(entry.get("type") or "").endswith("Pooling") and path and leaves_folder(path) is None:
                found["pooling"] = _pooling_of(_read_json(os.path.join(folder, path, "config.json")))
    sentence = _read_json(os.path.join(folder, "config_sentence_transformers.json"))
    prompts = sentence.get("prompts") if isinstance(sentence, dict) else None
    if isinstance(prompts, dict):
        query = prompts.get("query")
        text = next((prompts.get(k) for k in ("document", "passage", "text") if prompts.get(k)), "")
        found["query"] = query[:MAX_INSTRUCTION] if isinstance(query, str) else ""
        found["text"] = text[:MAX_INSTRUCTION] if isinstance(text, str) else ""
    config = _read_json(os.path.join(folder, MARKER))
    if isinstance(config, dict):
        names = config.get("architectures") or []
        causal = any(str(name).endswith("ForCausalLM") for name in names if isinstance(name, str))
        found["decoder"] = causal or str(config.get("model_type") or "") in DECODER_TYPES
    return found


def _roots() -> list:
    """The directories the ``LLM`` model folder searches."""
    from ..model import model_directories

    return model_directories(FOLDER)


def _subfolders(path: str) -> list:
    """The folders directly inside ``path``, links, junctions and hidden names left out."""
    from ..util.file_listing import linked

    try:
        with os.scandir(path) as listing:
            entries = list(islice(listing, MAX_ENTRIES))
    except OSError:
        return []
    kept = []
    for entry in entries:
        if entry.name.startswith(".") or linked(entry):
            continue
        try:
            if entry.is_dir(follow_symlinks=False):
                kept.append(entry)
        except OSError:
            continue
    return kept


def local_folders() -> list[str]:
    """Checkpoint folders in the ``LLM`` model folder, at most two levels deep.

    Returns:
        Names relative to their search directory, ``/`` separated, such as
        ``BAAI/bge-small-en-v1.5``. The listing is reused for :data:`LISTING_TTL` seconds.
    """
    global _listing
    roots = _roots()
    key = tuple(str(root) for root in roots)
    until, held, names = _listing
    if held == key and time.monotonic() < until:
        return list(names)
    found: list[str] = []
    for root in roots:
        for entry in _subfolders(str(root)):
            if len(found) >= MAX_FOLDERS:
                break
            if os.path.isfile(os.path.join(entry.path, MARKER)):
                found.append(entry.name)
                continue
            found.extend(
                f"{entry.name}/{inner.name}" for inner in _subfolders(entry.path)
                if os.path.isfile(os.path.join(inner.path, MARKER))
            )
    names = tuple(sorted(dict.fromkeys(found), key=str.lower)[:MAX_FOLDERS])
    _listing = (time.monotonic() + LISTING_TTL, key, names)
    return list(names)


#: ``model_type`` values of checkpoints that embed pictures, offered by LLM Image Embedding Model.
PICTURE_TYPES = frozenset({"clip", "siglip", "siglip2"})


def model_type(folder: str) -> str:
    """The ``model_type`` a checkpoint folder's ``config.json`` names, or ``""``."""
    try:
        data = _read_json(os.path.join(folder, MARKER))
    except Exception:
        return ""
    return str(data.get("model_type", "")) if isinstance(data, dict) else ""


def local_choices() -> list[str]:
    """What the local model menu offers: text folders on disk, then suggested ids not on disk."""
    found = [label for label in local_folders() if model_type(_local_folder(label) or "") not in PICTURE_TYPES]
    return found + [name for name in SUGGESTED if name not in found]


def gguf_choices() -> list[str]:
    """The ``.gguf`` files the GGUF model menu offers, or :data:`NO_GGUF` alone."""
    from .gguf import model_choices

    return model_choices() or [NO_GGUF]


def _local_folder(label: str) -> str | None:
    """The folder a local menu label names on disk, or ``None``."""
    from ..util.sandbox import leaves_folder

    text = str(label or "").strip()
    if not text or "\\" in text or text.count("/") > 1 or leaves_folder(text) is not None:
        return None
    for root in _roots():
        candidate = os.path.join(str(root), *text.split("/"))
        if os.path.isfile(os.path.join(candidate, MARKER)):
            return candidate
    return None


def _fetched(repo_id: str, patterns: tuple = DOWNLOAD_FILES) -> str:
    """The folder holding a suggested repository, downloading it when that is allowed.

    Args:
        repo_id: The repository.
        patterns: The files fetched from it.

    Raises:
        ModelUnavailable: It is not on this machine and ``features.network`` is off.
    """
    from ..model import ModelUnavailable, model_directories, network_enabled, resolve

    pretrained, cache_dir = resolve(FOLDER, repo_id, feature=FEATURE)
    if cache_dir is None:
        return pretrained
    if not network_enabled():
        raise ModelUnavailable(
            f"{repo_id} is not in models/LLM, and features.network is off. Set features.network to "
            f"true in config.yaml to fetch it, or put the folder in models/LLM."
        )
    hub = deps.require("huggingface_hub")
    roots = model_directories(FOLDER)
    owner, _, name = repo_id.partition("/")
    target = os.path.join(str(roots[0]), owner, name) if roots else None
    logger.info("downloading the embedding model %s", repo_id)
    return hub.snapshot_download(
        repo_id, local_dir=target, allow_patterns=list(patterns), token=False,
    )


def _require_safetensors(folder: str, label: str, node: str) -> None:
    """Refuse a folder holding no safetensors weights.

    Raises:
        ValueError: Neither ``model.safetensors`` nor its index is in the folder.
    """
    for name in ("model.safetensors", "model.safetensors.index.json"):
        if os.path.isfile(os.path.join(folder, name)):
            return
    raise ValueError(
        f"{node}: {label} holds no model.safetensors. Only safetensors weights are read; pick "
        f"another model, or put a safetensors copy of this one in models/LLM."
    )


def pool(hidden, mask, mode: str):
    """One vector per sequence from a model's last hidden states.

    Args:
        hidden: Hidden states shaped ``(batch, tokens, width)``.
        mask: The attention mask shaped ``(batch, tokens)``, 1 on real tokens.
        mode: ``cls``, ``mean``, ``max`` or ``last token``.

    Returns:
        A tensor shaped ``(batch, width)``.
    """
    import torch

    if mode == "cls":
        return hidden[:, 0]
    if mode == "last token":
        positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
        last = (positions * mask.long()).max(dim=1).values
        return hidden[torch.arange(hidden.shape[0], device=hidden.device), last.to(hidden.device)]
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    if mode == "max":
        return hidden.masked_fill(weights == 0, float("-inf")).max(dim=1).values
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-9)


def _backend(folder: str, device):
    """The managed tokenizer and model of a checkpoint folder, built on first use."""
    from ..model import compute_device, managed

    def build():
        import torch

        transformers = deps.require("transformers", feature=FEATURE)
        target = compute_device(device)
        if target.type == "cpu":
            dtype = torch.float32
        elif target.type == "cuda" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            folder, token=False, trust_remote_code=False, local_files_only=True,
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        network = transformers.AutoModel.from_pretrained(
            folder, token=False, trust_remote_code=False, use_safetensors=True,
            local_files_only=True, dtype=dtype,
        )
        return tokenizer, network

    key = ("llm-embedding", os.path.normcase(os.path.abspath(folder)))
    return managed(key, build, device=device, name="LLMEmbeddingModel")


def _forward_names(network) -> frozenset[str]:
    """The named arguments a model's forward takes."""
    try:
        parameters = inspect.signature(network.forward).parameters.values()
    except (TypeError, ValueError):
        return frozenset({"input_ids", "attention_mask"})
    return frozenset(
        p.name for p in parameters
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    )


def _transformers_class():
    """The embedding class over a transformers checkpoint folder."""
    if "transformers" in _classes:
        return _classes["transformers"]
    runtime.core()
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.bridge.pydantic import Field

    class TransformersEmbedding(BaseEmbedding):
        """Sentence vectors from a transformers checkpoint folder, pooled and optionally normalised."""

        folder: str = Field(description="The checkpoint folder.")
        pooling: str = Field(default="mean", description="cls, mean, max or last token.")
        normalize: bool = Field(default=True, description="Scale every vector to length 1.")
        query_instruction: str = Field(default="", description="Placed before every query.")
        text_instruction: str = Field(default="", description="Placed before every text.")
        max_length: int = Field(default=512, gt=0, description="Tokens read per text at most.")
        device: str | None = Field(default=None, description="Device name, None for ComfyUI's.")

        @classmethod
        def class_name(cls) -> str:
            return "WAS_TransformersEmbedding"

        def backend(self):
            """The managed tokenizer and model."""
            return _backend(self.folder, self.device)

        def vectors(self, texts: list[str]) -> list[list[float]]:
            """Embed texts in batches of ``embed_batch_size``.

            Args:
                texts: Texts, instructions already placed.

            Returns:
                One vector per text.
            """
            import torch

            backend = self.backend()
            tokenizer, network = backend.processor, backend.model
            accepted = _forward_names(network)
            limit = self.max_length
            sane = getattr(tokenizer, "model_max_length", 0) or 0
            if 0 < sane < 1_000_000:
                limit = min(limit, int(sane))
            positions = getattr(network.config, "max_position_embeddings", 0) or 0
            if positions > 0:
                limit = min(limit, int(positions))
            found: list[list[float]] = []
            for start in range(0, len(texts), self.embed_batch_size):
                runtime.check_interrupt()
                device = backend.load()
                batch = texts[start:start + self.embed_batch_size]
                encoded = tokenizer(
                    batch, padding=True, truncation=True, max_length=limit, return_tensors="pt",
                )
                feed = {name: value.to(device) for name, value in encoded.items() if name in accepted}
                with torch.inference_mode():
                    hidden = network(**feed).last_hidden_state.float()
                pooled = pool(hidden, encoded["attention_mask"].to(hidden.device), self.pooling)
                if self.normalize:
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
                found.extend(pooled.cpu().tolist())
            return found

        def _get_query_embedding(self, query: str) -> list[float]:
            return self.vectors([self.query_instruction + query])[0]

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return self._get_query_embedding(query)

        def _get_text_embedding(self, text: str) -> list[float]:
            return self.vectors([self.text_instruction + text])[0]

        def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
            return self.vectors([self.text_instruction + text for text in texts])

    _classes["transformers"] = TransformersEmbedding
    return TransformersEmbedding


def transformers_embedding(
    folder: str,
    label: str,
    pooling: str = "auto",
    normalize: bool = True,
    query_instruction: str = "",
    text_instruction: str = "",
    max_length: int = 512,
    batch_size: int = 32,
    device: str | None = None,
    node: str = "LLM Embedding Model",
) -> tuple[wires.EmbeddingModel, dict]:
    """An embedding model over a transformers checkpoint folder, loaded and measured.

    Args:
        folder: The checkpoint folder.
        label: What panels call it.
        pooling: One of :data:`POOLINGS`.
        normalize: Scale every vector to length 1.
        query_instruction: Placed before every query, blank for the family's own.
        text_instruction: Placed before every text, blank for the family's own.
        max_length: Tokens read per text at most.
        batch_size: Texts embedded together.
        device: Device name, ``None`` for ComfyUI's.
        node: The node's display name, for messages.

    Returns:
        The wire value, and ``pooling``, ``device``, ``query`` and ``text`` as used.

    Raises:
        RuntimeError: The folder would not load or embed.
    """
    settings = folder_settings(folder)
    family_query, family_text = family_instructions(label)
    if pooling == "auto":
        pooling = settings["pooling"] or (
            "last token" if settings["decoder"] else "cls" if "bge" in label.lower() else "mean"
        )
    query = str(query_instruction or "")[:MAX_INSTRUCTION] or settings["query"] or family_query
    text = str(text_instruction or "")[:MAX_INSTRUCTION] or settings["text"] or family_text
    embedder = _transformers_class()(
        model_name=label, folder=folder, pooling=pooling, normalize=bool(normalize),
        query_instruction=query, text_instruction=text, max_length=int(max_length),
        embed_batch_size=int(batch_size), device=device,
    )
    started = time.monotonic()
    dimensions = len(providers.construct(node, lambda: embedder.get_text_embedding(PROBE)))
    used = str(embedder.backend().load_device)
    logger.info(
        "embedding model %s ready: %d dimensions, %s pooling, on %s, %.2fs",
        label, dimensions, pooling, used, time.monotonic() - started,
    )
    wired = wires.EmbeddingModel(
        model=embedder, label=label, provider="transformers", local=True, dimensions=dimensions,
    )
    return wired, {"pooling": pooling, "device": used, "query": query, "text": text}


def local_embedding(
    choice: str,
    pooling: str = "auto",
    normalize: bool = True,
    query_instruction: str = "",
    text_instruction: str = "",
    max_length: int = 512,
    batch_size: int = 32,
    node: str = "LLM Embedding Model",
) -> tuple[wires.EmbeddingModel, dict]:
    """An embedding model from a local folder or a suggested repository.

    Args:
        choice: An entry of :func:`local_choices`.
        pooling: One of :data:`POOLINGS`.
        normalize: Scale every vector to length 1.
        query_instruction: Placed before every query, blank for the family's own.
        text_instruction: Placed before every text, blank for the family's own.
        max_length: Tokens read per text at most.
        batch_size: Texts embedded together.
        node: The node's display name, for messages.

    Returns:
        What :func:`transformers_embedding` returns.

    Raises:
        ValueError: The choice names nothing on disk or in :data:`SUGGESTED`, or the folder
            holds no safetensors weights.
        ModelUnavailable: A suggested repository is not here and ``features.network`` is off.
    """
    label = str(choice or "").strip()
    folder = _local_folder(label)
    if folder is None:
        if label not in SUGGESTED:
            raise ValueError(
                f"{node} cannot find `{label[:80]}` in models/LLM. Pick a model from the menu, or "
                f"put a Hugging Face checkpoint folder holding config.json in models/LLM."
            )
        folder = _fetched(label)
    _require_safetensors(folder, label, node)
    if pooling not in POOLINGS:
        raise ValueError(f"{node} does not know the pooling `{str(pooling)[:40]}`. Pick one of {', '.join(POOLINGS)}.")
    return transformers_embedding(
        folder, label, pooling, normalize, query_instruction, text_instruction, max_length,
        batch_size, node=node,
    )


def _llama_cpp():
    """``llama_cpp``.

    Raises:
        DependencyError: llama-cpp-python is not installed or will not import.
    """
    module = deps.optional("llama_cpp")
    if module is None:
        raise deps.DependencyError(
            "LLM Embedding Model needs llama-cpp-python to read a .gguf file, and it is not "
            "installed in the python that runs ComfyUI. Install a build that matches this machine: "
            "a CUDA build picked to match the installed CUDA version from the prebuilt wheels the "
            "llama-cpp-python README lists, or the CPU build. features.llm turned this node on."
        )
    return module


def _open_gguf(path: str, context: int, pooling: str):
    """A llama.cpp model opened for embeddings."""
    llama_cpp = _llama_cpp()
    kinds = {
        "auto": llama_cpp.LLAMA_POOLING_TYPE_UNSPECIFIED,
        "mean": llama_cpp.LLAMA_POOLING_TYPE_MEAN,
        "cls": llama_cpp.LLAMA_POOLING_TYPE_CLS,
        "last token": llama_cpp.LLAMA_POOLING_TYPE_LAST,
    }
    offload = -1 if llama_cpp.llama_supports_gpu_offload() else 0
    return llama_cpp.Llama(
        model_path=path, embedding=True, n_ctx=context, n_batch=context, n_ubatch=context,
        pooling_type=kinds.get(pooling, llama_cpp.LLAMA_POOLING_TYPE_UNSPECIFIED),
        n_gpu_layers=offload, verbose=False,
    )


def _one_vector(result, pooling: str) -> list[float]:
    """One vector from what llama.cpp answered, pooling per-token vectors where it gave those."""
    if result and isinstance(result[0], list):
        if pooling == "last token":
            return [float(value) for value in result[-1]]
        if pooling == "cls":
            return [float(value) for value in result[0]]
        width = len(result[0])
        return [sum(row[i] for row in result) / len(result) for i in range(width)]
    return [float(value) for value in result]


def _gguf_class():
    """The embedding class over a GGUF file."""
    if "gguf" in _classes:
        return _classes["gguf"]
    runtime.core()
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.bridge.pydantic import Field

    class GGUFEmbedding(BaseEmbedding):
        """Sentence vectors from a GGUF file through llama.cpp."""

        file: str = Field(description="The .gguf file.")
        context: int = Field(default=512, gt=0, description="Tokens read per text at most.")
        pooling: str = Field(default="auto", description="auto, cls, mean or last token.")
        normalize: bool = Field(default=True, description="Scale every vector to length 1.")
        query_instruction: str = Field(default="", description="Placed before every query.")
        text_instruction: str = Field(default="", description="Placed before every text.")

        @classmethod
        def class_name(cls) -> str:
            return "WAS_GGUFEmbedding"

        def llama(self):
            """The opened llama.cpp model, shared through the model cache."""
            from ..model import cached

            key = ("llm-embedding-gguf", os.path.normcase(self.file), self.context, self.pooling)
            return cached(key, lambda: _open_gguf(self.file, self.context, self.pooling))

        def vectors(self, texts: list[str]) -> list[list[float]]:
            """Embed texts one at a time.

            Args:
                texts: Texts, instructions already placed.

            Returns:
                One vector per text.
            """
            model = self.llama()
            found = []
            for text in texts:
                runtime.check_interrupt()
                vector = _one_vector(model.embed(text, normalize=False, truncate=True), self.pooling)
                if self.normalize:
                    length = math.sqrt(sum(value * value for value in vector)) or 1.0
                    vector = [value / length for value in vector]
                found.append(vector)
            return found

        def _get_query_embedding(self, query: str) -> list[float]:
            return self.vectors([self.query_instruction + query])[0]

        async def _aget_query_embedding(self, query: str) -> list[float]:
            return self._get_query_embedding(query)

        def _get_text_embedding(self, text: str) -> list[float]:
            return self.vectors([self.text_instruction + text])[0]

        def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
            return self.vectors([self.text_instruction + text for text in texts])

    _classes["gguf"] = GGUFEmbedding
    return GGUFEmbedding


def gguf_embedding(
    choice: str,
    pooling: str = "auto",
    normalize: bool = True,
    query_instruction: str = "",
    text_instruction: str = "",
    max_length: int = 512,
    node: str = "LLM Embedding Model",
    path: str | None = None,
) -> tuple[wires.EmbeddingModel, dict]:
    """An embedding model over a GGUF file, loaded and measured.

    Args:
        choice: An entry of :func:`gguf_choices`.
        pooling: One of :data:`POOLINGS`.
        normalize: Scale every vector to length 1.
        query_instruction: Placed before every query, blank for the family's own.
        text_instruction: Placed before every text, blank for the family's own.
        max_length: Tokens read per text at most.
        node: The node's display name, for messages.
        path: The file itself, in place of a menu entry.

    Returns:
        The wire value, and ``pooling``, ``device``, ``query`` and ``text`` as used.

    Raises:
        ValueError: The choice names no ``.gguf`` file the menu lists.
        DependencyError: llama-cpp-python is not installed.
    """
    label = str(choice or "").strip()
    if path is None:
        from .gguf import locate

        path = locate(label) if label != NO_GGUF else None
    if not path:
        raise ValueError(
            f"{node} cannot find `{label[:80]}`. Put a .gguf embedding model in models/LLM or "
            f"models/text_encoders and pick it from the menu."
        )
    if pooling not in POOLINGS:
        raise ValueError(f"{node} does not know the pooling `{str(pooling)[:40]}`. Pick one of {', '.join(POOLINGS)}.")
    offload = bool(_llama_cpp().llama_supports_gpu_offload())
    family_query, family_text = family_instructions(label)
    query = str(query_instruction or "")[:MAX_INSTRUCTION] or family_query
    text = str(text_instruction or "")[:MAX_INSTRUCTION] or family_text
    embedder = _gguf_class()(
        model_name=label, file=str(path), context=int(max_length), pooling=pooling,
        normalize=bool(normalize), query_instruction=query, text_instruction=text,
    )
    started = time.monotonic()
    dimensions = len(providers.construct(node, lambda: embedder.get_text_embedding(PROBE)))
    logger.info(
        "embedding model %s ready: %d dimensions, %s pooling, %.2fs",
        label, dimensions, pooling, time.monotonic() - started,
    )
    wired = wires.EmbeddingModel(
        model=embedder, label=label, provider="gguf", local=True, dimensions=dimensions,
    )
    device = "gpu, through llama.cpp" if offload else "cpu, through llama.cpp"
    return wired, {"pooling": pooling, "device": device, "query": query, "text": text}


def openai_embedding(
    model: str = "text-embedding-3-small", dimensions: int = 0, node: str = "LLM Embedding Model",
) -> wires.EmbeddingModel:
    """An OpenAI embedding model on ``api.openai.com``.

    Args:
        model: The model id, empty for ``text-embedding-3-small``.
        dimensions: Vector length asked for, 0 for the model's own.
        node: The node's display name, for messages.

    Returns:
        The wire value. No request is sent.

    Raises:
        ValueError: The model id is not valid.
        RuntimeError: ``features.network`` is off or ``OPENAI_API_KEY`` is not set.
    """
    name = providers.check_model(model, node, "text-embedding-3-small")
    label = f"{name} on OpenAI"
    runtime.require_network(label)
    key = providers.key_value(providers.OPENAI_KEY, node)
    runtime.core()
    deps.require("llama_index.embeddings.openai", feature=FEATURE)
    from llama_index.embeddings.openai import OpenAIEmbedding

    embedder = providers.construct(node, lambda: OpenAIEmbedding(
        model_name=name, api_key=key, api_base=providers.OPENAI_URL,
        dimensions=int(dimensions) or None, max_retries=3, timeout=60.0,
    ))
    return wires.EmbeddingModel(
        model=embedder, label=label, provider="openai", local=False,
        dimensions=int(dimensions) or OPENAI_DIMENSIONS.get(name, 0),
    )


def gemini_embedding(
    model: str = "gemini-embedding-001", dimensions: int = 0, node: str = "LLM Embedding Model",
) -> wires.EmbeddingModel:
    """A Gemini embedding model.

    Args:
        model: The model id, empty for ``gemini-embedding-001``.
        dimensions: Vector length asked for, 0 for the model's own.
        node: The node's display name, for messages.

    Returns:
        The wire value. No request is sent.

    Raises:
        ValueError: The model id is not valid.
        RuntimeError: ``features.network`` is off or no Gemini key is set.
    """
    name = providers.check_model(model, node, "gemini-embedding-001")
    label = f"{name} on Google Gemini"
    runtime.require_network(label)
    key = providers.key_value(providers.GEMINI_KEYS, node)
    runtime.core()
    deps.require("llama_index.embeddings.google_genai", feature=FEATURE)
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

    config = {"output_dimensionality": int(dimensions)} if int(dimensions) > 0 else None
    embedder = providers.construct(
        node, lambda: GoogleGenAIEmbedding(model_name=name, api_key=key, embedding_config=config),
    )
    return wires.EmbeddingModel(
        model=embedder, label=label, provider="gemini", local=False,
        dimensions=int(dimensions) or GEMINI_DIMENSIONS.get(name, 0),
    )


def provider_embedding(
    choice: str, model: str = "", dimensions: int = 0, node: str = "LLM Embedding Model",
) -> tuple[wires.EmbeddingModel, providers.Preset]:
    """An embedding model on a built-in provider, a local server or an ``llm.endpoints`` entry.

    Args:
        choice: An entry of :func:`.providers.options`.
        model: The model id, empty for the provider's default.
        dimensions: Vector length asked for, 0 for the model's own.
        node: The node's display name, for messages.

    Returns:
        The wire value and the provider it reaches. No request is sent.

    Raises:
        ValueError: The provider or model id is not valid.
        RuntimeError: The server is on another machine with ``features.network`` off, or its
            key variable is not set.
    """
    target = providers.preset(choice, node)
    name = providers.check_model(model, node, target.embed_model)
    label = f"{name} on {target.title}"
    if not target.local:
        runtime.require_network(label)
    key = providers.key_value(target.key_env, node) if target.key_env else ""
    runtime.core()
    if target.api == "ollama":
        deps.require("llama_index.embeddings.ollama", feature=FEATURE)
        from llama_index.embeddings.ollama import OllamaEmbedding

        embedder = providers.construct(node, lambda: OllamaEmbedding(
            model_name=name, base_url=target.url,
            client_kwargs={"headers": {"Authorization": f"Bearer {key or providers.NO_KEY}"}},
        ))
        family = "ollama"
    else:
        deps.require("llama_index.embeddings.openai", feature=FEATURE)
        from llama_index.embeddings.openai import OpenAIEmbedding

        embedder = providers.construct(node, lambda: OpenAIEmbedding(
            model_name=name, api_key=key or providers.NO_KEY, api_base=target.url,
            dimensions=int(dimensions) or None, max_retries=3, timeout=60.0,
            default_headers=providers.foreign_headers(),
        ))
        family = "provider"
    wired = wires.EmbeddingModel(
        model=embedder, label=label, provider=family, local=target.local,
        dimensions=int(dimensions),
    )
    return wired, target

"""Cross-encoder reranking on transformers, as a llama-index postprocessor.

:func:`choices` lists checkpoint folders under the ``LLM`` model folder beside the suggested
repositories, :func:`load` loads one through ComfyUI's model management and :func:`reranker`
wraps it.
"""

from __future__ import annotations

import os
from typing import Any

from .. import deps, log
from ..model import ModelUnavailable, managed, model_directories, model_files, resolve
from ..util.sandbox import leaves_folder
from . import FEATURE, runtime

__all__ = ["FOLDER", "SUGGESTED", "Reranker", "choices", "load", "reranker", "score"]

logger = log.get_logger("llm.rerank")

#: The model folder rerankers are read from, ``ComfyUI/models/LLM``.
FOLDER = "LLM"

#: Cross-encoders offered beside the local folders, downloaded only with ``features.network`` on.
SUGGESTED = (
    "BAAI/bge-reranker-v2-m3",
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-large",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

#: The file a transformers checkpoint folder holds.
MARKER = "config.json"

_class = None


class Reranker:
    """A loaded cross-encoder.

    Attributes:
        backend: The :class:`~modules.model.Backend` holding the tokenizer and the model.
        label: The choice it was loaded from, a folder label or a repository id.
        max_length: The most tokens one question and chunk pair can hold.
        parameters: Weights in the model.
    """

    def __init__(self, backend, label: str, max_length: int, parameters: int):
        self.backend = backend
        self.label = label
        self.max_length = max_length
        self.parameters = parameters


def choices() -> list[str]:
    """Local checkpoint folders under the ``LLM`` model folder, then the suggested repositories.

    Returns:
        Folder labels relative to the model folder, a downloaded repository by its id, sorted,
        followed by :data:`SUGGESTED` entries not already listed.
    """
    found: list[str] = []
    for name in model_files(FOLDER):
        parts = name.replace("\\", "/").split("/")
        if parts[-1] != MARKER or len(parts) < 2:
            continue
        folder = parts[:-1]
        if len(folder) >= 3 and folder[-2] == "snapshots" and folder[-3].startswith("models--"):
            label = folder[-3][len("models--"):].replace("--", "/")
        else:
            label = "/".join(folder)
        if label not in found:
            found.append(label)
    found.sort(key=str.lower)
    return found + [repo for repo in SUGGESTED if repo not in found]


def _folder(label: str) -> str | None:
    """The local checkpoint folder a label names, or ``None``."""
    if not label or leaves_folder(label) is not None:
        return None
    for root in model_directories(FOLDER):
        candidate = os.path.join(str(root), *label.split("/"))
        if os.path.isfile(os.path.join(candidate, MARKER)):
            return candidate
    return None


def _weights_are_safetensors(folder: str) -> bool:
    """Whether a checkpoint folder holds ``.safetensors`` weights."""
    try:
        return any(name.lower().endswith(".safetensors") for name in os.listdir(folder))
    except OSError:
        return False


def load(choice: str, device: str | None = None) -> Reranker:
    """Load a cross-encoder and its tokenizer.

    Args:
        choice: A label from :func:`choices`.
        device: Device name, or ``None`` for ComfyUI's compute device.

    Returns:
        The :class:`Reranker`, its weights resting on the offload device until scored.

    Raises:
        ValueError: The choice names no checkpoint, holds no safetensors weights, or is not a
            cross-encoder.
        ModelUnavailable: A suggested repository is not on disk and ``features.network`` is off.
        DependencyError: transformers is not importable.
    """
    transformers = deps.require("transformers", feature=FEATURE)
    label = str(choice or "").strip()
    local = _folder(label)
    if local is not None:
        if not _weights_are_safetensors(local):
            raise ValueError(
                f"`{label}` in models/LLM holds no .safetensors weights, and LLM Rerank loads "
                f"safetensors only. Put the checkpoint's model.safetensors beside its config.json."
            )
        pretrained, cache_dir, offline = local, None, True
    elif label in SUGGESTED:
        pretrained, cache_dir = resolve(FOLDER, label, feature=FEATURE)
        offline = cache_dir is None
    else:
        try:
            pretrained, cache_dir = resolve(FOLDER, label, feature=FEATURE)
        except (ValueError, ModelUnavailable):
            pretrained, cache_dir = None, None
        if pretrained is None or cache_dir is not None:
            raise ValueError(
                f"`{label[:120]}` is neither a checkpoint folder in models/LLM nor one of the suggested "
                f"rerankers. Pick a model from the list, or put a cross-encoder folder holding "
                f"config.json and model.safetensors in ComfyUI/models/LLM."
            )
        offline = True
    try:
        config = transformers.AutoConfig.from_pretrained(
            pretrained, cache_dir=cache_dir, token=False, trust_remote_code=False, local_files_only=offline
        )
    except Exception as error:
        raise RuntimeError(f"LLM Rerank could not read `{label}`: {runtime.describe_error(error)}") from None
    architectures = [str(name) for name in (getattr(config, "architectures", None) or [])]
    if not any(name.endswith("ForSequenceClassification") for name in architectures):
        raise ValueError(
            f"`{label}` is not a cross-encoder reranker: it is a {', '.join(architectures) or 'model of unknown kind'}. "
            f"Pick a reranker such as BAAI/bge-reranker-base."
        )
    if int(getattr(config, "num_labels", 1)) not in (1, 2):
        raise ValueError(
            f"`{label}` sorts text into {config.num_labels} classes, and a reranker scores one relevance. "
            f"Pick a reranker such as BAAI/bge-reranker-base."
        )

    def build():
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained, cache_dir=cache_dir, token=False, trust_remote_code=False, local_files_only=offline
        )
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained, cache_dir=cache_dir, token=False, trust_remote_code=False, use_safetensors=True,
            local_files_only=offline,
        )
        return tokenizer, model

    try:
        backend = managed(("llm-rerank", str(pretrained)), build, device=device, name="CrossEncoderReranker")
    except Exception as error:
        if type(error).__name__ == "InterruptProcessingException":
            raise
        raise RuntimeError(f"LLM Rerank could not load `{label}`: {runtime.describe_error(error)}") from None
    limits = [getattr(config, "max_position_embeddings", None), getattr(backend.processor, "model_max_length", None)]
    longest = min([int(value) for value in limits if isinstance(value, int) and 0 < value < 1_000_000] or [512])
    parameters = sum(int(weight.numel()) for weight in backend.model.parameters())
    logger.info("LLM Rerank loaded %s, %.1fM parameters", label, parameters / 1e6)
    return Reranker(backend, label, longest, parameters)


def score(model: Reranker, query: str, texts: list[str], batch_size: int = 16, max_length: int = 512) -> list[float]:
    """Relevance of each text to the question, 0 to 1.

    Args:
        model: The loaded :class:`Reranker`.
        query: The question.
        texts: The chunks' texts.
        batch_size: Pairs scored at once.
        max_length: The most tokens one pair holds, cut to what the model reads.

    Returns:
        One score per text, in order.
    """
    import torch

    if not texts:
        return []
    backend = model.backend
    device = backend.load()
    tokenizer, network = backend.processor, backend.model
    limit = max(8, min(int(max_length), model.max_length))
    found: list[float] = []
    with torch.inference_mode():
        for start in range(0, len(texts), max(1, int(batch_size))):
            runtime.check_interrupt()
            batch = texts[start:start + max(1, int(batch_size))]
            encoded = tokenizer(
                [query] * len(batch), batch, padding=True, truncation=True, max_length=limit, return_tensors="pt"
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = network(**encoded).logits.float()
            if logits.shape[-1] == 1:
                values = torch.sigmoid(logits[:, 0])
            else:
                values = torch.softmax(logits, dim=-1)[:, -1]
            found.extend(float(value) for value in values.cpu().tolist())
    return found


def reranker(model: Reranker, top_n: int = 5, batch_size: int = 16, max_length: int = 512):
    """A llama-index postprocessor reordering chunks by the cross-encoder's score.

    Args:
        model: The loaded :class:`Reranker`.
        top_n: Chunks kept, best first.
        batch_size: Pairs scored at once.
        max_length: The most tokens one pair holds.

    Returns:
        A ``BaseNodePostprocessor``.
    """
    return _postprocessor_class()(model, top_n=int(top_n), batch_size=int(batch_size), max_length=int(max_length))


def _postprocessor_class():
    """The cross-encoder postprocessor class, defined on first use."""
    global _class
    if _class is not None:
        return _class
    runtime.core()
    from llama_index.core.bridge.pydantic import PrivateAttr
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.schema import MetadataMode, NodeWithScore

    class CrossEncoderRerank(BaseNodePostprocessor):
        """Scores each chunk against the question with a cross-encoder and keeps the best."""

        top_n: int = 5
        batch_size: int = 16
        max_length: int = 512
        _model: Any = PrivateAttr()

        def __init__(self, model, **fields):
            super().__init__(**fields)
            self._model = model

        @classmethod
        def class_name(cls) -> str:
            return "WASCrossEncoderRerank"

        def _postprocess_nodes(self, nodes, query_bundle=None):
            if query_bundle is None:
                raise ValueError("LLM Rerank scores chunks against a question, and none was given.")
            if not nodes:
                return []
            texts = [item.node.get_content(metadata_mode=MetadataMode.EMBED) for item in nodes]
            values = score(self._model, query_bundle.query_str, texts, self.batch_size, self.max_length)
            order = sorted(range(len(nodes)), key=lambda row: values[row], reverse=True)[: self.top_n]
            return [NodeWithScore(node=nodes[row].node, score=values[row]) for row in order]

    _class = CrossEncoderRerank
    return _class

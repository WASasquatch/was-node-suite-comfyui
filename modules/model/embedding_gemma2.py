"""EmbeddingGemma 2 run through ComfyUI's model management: captions, prompts and pictures in one space.

A checkpoint folder holds ``config.json``, ``model.safetensors`` and ``tokenizer.json``.
"""

from __future__ import annotations

import os

from .. import log

__all__ = ["MODEL_TYPE", "QUERY", "TEXT", "is_folder", "picture_vectors", "text_vectors"]

logger = log.get_logger("model.embedding_gemma2")

#: The ``model_type`` a checkpoint folder's ``config.json`` names.
MODEL_TYPE = "embedding_gemma2"

#: Placed before a prompt searched with, and before a caption searched in.
QUERY = "task: search result | query: "
TEXT = "title: none | text: "

#: Most tokens of one text read, BOS and EOS included.
MAX_TOKENS = 512


def is_folder(folder: str) -> bool:
    """Whether a checkpoint folder holds EmbeddingGemma 2."""
    from ..llm import embeddings

    return embeddings.model_type(folder) == MODEL_TYPE


def _backend(folder: str, device=None):
    """The managed tokenizer and model of a checkpoint folder, built on first use."""
    from . import compute_device, managed

    def build():
        import torch
        from tokenizers import Tokenizer

        from ..vendor.embedding_gemma2.model import EmbeddingGemma2

        target = compute_device(device)
        dtype = torch.float32 if target.type == "cpu" else torch.bfloat16
        tokenizer = Tokenizer.from_file(os.path.join(folder, "tokenizer.json"))
        return tokenizer, EmbeddingGemma2.load(folder, dtype)

    key = ("embedding-gemma2", os.path.normcase(os.path.abspath(folder)))
    return managed(key, build, device=device, name="EmbeddingGemma2")


def text_vectors(folder: str, texts: list[str], prefix: str, batch_size: int = 32, device=None):
    """Unit vectors for texts, each read after ``prefix``.

    Args:
        folder: The checkpoint folder.
        texts: The texts.
        prefix: :data:`QUERY` for a prompt, :data:`TEXT` for a caption.
        batch_size: Texts embedded together.
        device: Device name, ``None`` for ComfyUI's.

    Returns:
        A float32 CPU tensor shaped ``(len(texts), 768)``.
    """
    import torch

    from ..llm.runtime import check_interrupt

    backend = _backend(folder, device)
    tokenizer, model = backend.processor, backend.model
    found = []
    step = max(1, int(batch_size))
    for start in range(0, len(texts), step):
        check_interrupt()
        load_device = backend.load()
        ids = []
        for text in texts[start:start + step]:
            row = tokenizer.encode(prefix + text).ids
            ids.append(row if len(row) <= MAX_TOKENS else row[:MAX_TOKENS - 1] + row[-1:])
        found.append(model.embed_ids(ids, load_device).float().cpu())
    return torch.cat(found) if found else torch.zeros((0, 0))


def picture_vectors(folder: str, pictures, batch_size: int = 8, device=None, opener=None):
    """Unit vectors for pictures.

    Args:
        folder: The checkpoint folder.
        pictures: PIL pictures, or anything ``opener`` turns into one.
        batch_size: Pictures embedded together.
        device: Device name, ``None`` for ComfyUI's.
        opener: Called on each item to make an RGB PIL picture.

    Returns:
        A float32 CPU tensor shaped ``(len(pictures), 768)``.
    """
    from concurrent.futures import ThreadPoolExecutor

    import torch

    from ..llm.runtime import check_interrupt
    from ..vendor.embedding_gemma2.model import picture_patches

    backend = _backend(folder, device)
    model = backend.model
    items = list(pictures)
    step = max(1, int(batch_size))
    found = []

    def prepared(item):
        return picture_patches(opener(item) if opener is not None else item)

    with ThreadPoolExecutor(max_workers=min(8, step)) as workers:
        for start in range(0, len(items), step):
            check_interrupt()
            batch = list(workers.map(prepared, items[start:start + step]))
            load_device = backend.load()
            pixels = torch.stack([pixels for pixels, _ in batch])
            positions = torch.stack([positions for _, positions in batch])
            found.append(model.embed_pictures(pixels, positions, load_device).float().cpu())
    return torch.cat(found) if found else torch.zeros((0, 0))

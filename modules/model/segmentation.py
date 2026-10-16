"""Semantic segmentation, on the transformers SegFormer models trained on ADE20K.

:func:`load` returns a backend whose ``processor`` is an ``AutoImageProcessor`` and whose
``model`` is a ``SegformerForSemanticSegmentation`` answering 150 ADE20K classes.
"""

from __future__ import annotations

from .. import deps
from . import managed, resolve

__all__ = ["MODELS", "load"]

#: Config key of the feature group these nodes are gated on.
FEATURE = "features.preprocessors"

#: ``folder_paths`` model folder searched for the checkpoints.
FOLDER = "segmentation"

#: Widget option -> repository. SegFormer alone.
MODELS = {
    "SegFormer B0 ADE20K": "nvidia/segformer-b0-finetuned-ade-512-512",
    "SegFormer B2 ADE20K": "nvidia/segformer-b2-finetuned-ade-512-512",
    "SegFormer B4 ADE20K": "nvidia/segformer-b4-finetuned-ade-512-512",
}


def load(name: str = "SegFormer B0 ADE20K", device: str | None = None):
    """Load a semantic-segmentation image processor and model.

    Args:
        name: One of the keys of :data:`MODELS`.
        device: Device name, or ``None`` for ComfyUI's compute device.

    Returns:
        A :class:`~modules.model.Backend` whose ``processor`` and ``model`` are the pair.

    Raises:
        ValueError: ``name`` is not a key of :data:`MODELS`.
        DependencyError: transformers is not importable.
        ModelUnavailable: No local checkpoint, and ``features.network`` is off.
    """
    if name not in MODELS:
        raise ValueError(
            f"Segmentation model must be one of {', '.join(MODELS)}, not {name!r}"
        )
    repo_id = MODELS[name]

    transformers = deps.require("transformers", feature=FEATURE)
    pretrained, cache_dir = resolve(FOLDER, repo_id, feature=FEATURE)

    def build():
        processor = transformers.AutoImageProcessor.from_pretrained(
            pretrained, cache_dir=cache_dir, token=False
        )
        model = transformers.SegformerForSemanticSegmentation.from_pretrained(
            pretrained, cache_dir=cache_dir, token=False
        )
        return processor, model

    return managed(("segmentation", pretrained), build, device=device)

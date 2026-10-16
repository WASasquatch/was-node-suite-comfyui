"""Segment Anything masking, on transformers.

Each of the ``model_size`` widget's three options maps to the matching
``facebook/sam-vit-*`` repository.
"""

from __future__ import annotations

from .. import deps
from . import managed, resolve

__all__ = ["load"]

#: Config key of the feature group these nodes are gated on.
FEATURE = "features.sam"

#: ``folder_paths`` model folders searched, in order. ``sams`` is where ComfyUI's other
#: masking packs keep theirs; ``sam`` is where v2 put its download.
FOLDERS = ("sams", "sam")

#: v2 ``model_size`` option -> (repository, the v2-era checkpoint it replaces). The keys
#: are the widget's options, so renaming one stops saved workflows validating.
MODELS = {
    "ViT-H": ("facebook/sam-vit-huge", "sam_vit_h_4b8939.pth"),
    "ViT-L": ("facebook/sam-vit-large", "sam_vit_l_0b3195.pth"),
    "ViT-B": ("facebook/sam-vit-base", "sam_vit_b_01ec64.pth"),
}


def load(model_size: str = "ViT-H", device: str | None = None):
    """Load a SAM processor and model.

    Args:
        model_size: One of the ``model_size`` widget's options: ``"ViT-H"``, ``"ViT-L"``
            or ``"ViT-B"``.
        device: Device name, or ``None`` for ComfyUI's compute device.

    Returns:
        A :class:`~modules.model.Backend` whose ``processor`` and ``model`` are the SAM
        pair, resting on the offload device until ``Backend.load()`` is called.

    Raises:
        ValueError: ``model_size`` is not a key of :data:`MODELS`.
        DependencyError: transformers is not importable.
        ModelUnavailable: No local checkpoint, and ``features.network`` is off.
    """
    if model_size not in MODELS:
        raise ValueError(f"SAM model_size must be one of {', '.join(MODELS)}, not {model_size!r}")
    repo_id, legacy_name = MODELS[model_size]

    transformers = deps.require("transformers", feature=FEATURE)
    pretrained, cache_dir = resolve(FOLDERS, repo_id, legacy=(legacy_name,), feature=FEATURE)

    def build():
        processor = transformers.SamProcessor.from_pretrained(
            pretrained, cache_dir=cache_dir, token=False
        )
        model = transformers.SamModel.from_pretrained(pretrained, cache_dir=cache_dir, token=False)
        return processor, model

    return managed(("sam", pretrained), build, device=device)

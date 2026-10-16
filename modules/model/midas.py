"""MiDaS depth estimation, on the transformers DPT models.

:func:`load` returns a backend whose ``processor`` is an ``AutoImageProcessor`` and whose
``model`` is an ``AutoModelForDepthEstimation``.
"""

from __future__ import annotations

from .. import deps
from . import managed, resolve

__all__ = ["load"]

#: Config key of the feature group these nodes are gated on.
FEATURE = "features.midas"

#: ``folder_paths`` model folder searched for DPT checkpoints.
FOLDER = "midas"

#: v2 widget option -> (repository, the torch.hub checkpoints it replaces).
MODELS = {
    "DPT_Large": ("Intel/dpt-large", ("checkpoints/dpt_large_384.pt",)),
    "DPT_Hybrid": ("Intel/dpt-hybrid-midas", ("checkpoints/dpt_hybrid_384.pt",)),
    # MiDaS v2.1 small, which the legacy checkpoint holds, has no transformers release, so
    # this option points at the smallest DPT there is and its depth is not comparable.
    "DPT_Small": ("Intel/dpt-swinv2-tiny-256", ("checkpoints/midas_v21_small_256.pt",)),
}


def load(midas_model: str = "DPT_Large", device: str | None = None):
    """Load a depth-estimation image processor and model.

    Args:
        midas_model: One of the widget's options: ``"DPT_Large"``, ``"DPT_Hybrid"`` or
            ``"DPT_Small"``.
        device: Device name, or ``None`` for ComfyUI's compute device.

    Returns:
        A :class:`~modules.model.Backend` whose ``processor`` and ``model`` are the DPT
        pair, resting on the offload device until ``Backend.load()`` is called.

    Raises:
        ValueError: ``midas_model`` is not a key of :data:`MODELS`.
        DependencyError: transformers is not importable.
        ModelUnavailable: No local checkpoint, and ``features.network`` is off.
    """
    if midas_model not in MODELS:
        raise ValueError(f"MiDaS model must be one of {', '.join(MODELS)}, not {midas_model!r}")
    repo_id, legacy_names = MODELS[midas_model]

    transformers = deps.require("transformers", feature=FEATURE)
    pretrained, cache_dir = resolve(FOLDER, repo_id, legacy=legacy_names, feature=FEATURE)

    def build():
        processor = transformers.AutoImageProcessor.from_pretrained(
            pretrained, cache_dir=cache_dir, token=False
        )
        model = transformers.AutoModelForDepthEstimation.from_pretrained(
            pretrained, cache_dir=cache_dir, token=False
        )
        return processor, model

    return managed(("midas", pretrained), build, device=device)

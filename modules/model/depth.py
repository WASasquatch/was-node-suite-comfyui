"""Depth estimation, on the transformers depth conversions.

:func:`load` returns a backend whose ``processor`` is an ``AutoImageProcessor`` and whose
``model`` is an ``AutoModelForDepthEstimation``. Every answer grows with nearness.
"""

from __future__ import annotations

from .. import deps
from . import managed, resolve

__all__ = ["MODELS", "load"]

#: Config key of the feature group these nodes are gated on.
FEATURE = "features.preprocessors"

#: ``folder_paths`` model folders searched for the checkpoints, in order.
FOLDERS = ("depth", "depth_anything")

#: Widget option -> repository. The published conversions, which carry the config and the
#: preprocessor settings the raw ``.pth`` releases do not.
MODELS = {
    "Depth Anything V2 Small": "depth-anything/Depth-Anything-V2-Small-hf",
    "Depth Anything V2 Base": "depth-anything/Depth-Anything-V2-Base-hf",
    "Depth Anything V2 Large": "depth-anything/Depth-Anything-V2-Large-hf",
    "DPT SwinV2 Tiny": "Intel/dpt-swinv2-tiny-256",
    "DPT Large": "Intel/dpt-large",
}


def load(name: str = "Depth Anything V2 Small", device: str | None = None):
    """Load a depth-estimation image processor and model.

    Args:
        name: One of the keys of :data:`MODELS`.
        device: Device name, or ``None`` for ComfyUI's compute device.

    Returns:
        A :class:`~modules.model.Backend` whose ``processor`` and ``model`` are the pair,
        resting on the offload device until ``Backend.load()`` is called.

    Raises:
        ValueError: ``name`` is not a key of :data:`MODELS`.
        DependencyError: transformers is not importable.
        ModelUnavailable: No local checkpoint, and ``features.network`` is off.
    """
    if name not in MODELS:
        raise ValueError(
            f"Depth model must be one of {', '.join(MODELS)}, not {name!r}"
        )
    repo_id = MODELS[name]

    transformers = deps.require("transformers", feature=FEATURE)
    pretrained, cache_dir = resolve(FOLDERS, repo_id, feature=FEATURE)

    def build():
        processor = transformers.AutoImageProcessor.from_pretrained(
            pretrained, cache_dir=cache_dir, token=False
        )
        model = transformers.AutoModelForDepthEstimation.from_pretrained(
            pretrained, cache_dir=cache_dir, token=False
        )
        return processor, model

    return managed(("depth", pretrained), build, device=device)

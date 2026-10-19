"""Clothing and body-part labels for people, on a transformers SegFormer.

:func:`load` returns a backend whose ``model`` answers 18 classes per pixel, named in its
``config.id2label``: background, hat, hair, sunglasses, clothing pieces, face, arms, legs,
shoes, bag and scarf.
"""

from __future__ import annotations

from .. import deps
from . import managed, resolve

__all__ = ["REPO", "load"]

#: Config key of the feature group these nodes are gated on.
FEATURE = "features.preprocessors"

#: ``folder_paths`` model folder searched for the checkpoint.
FOLDER = "segmentation"

#: Repository holding the checkpoint.
REPO = "sayeed99/segformer_b3_clothes"


def load(device: str | None = None):
    """Load the body-part segmentation model.

    Args:
        device: Device name, or ``None`` for ComfyUI's compute device.

    Returns:
        A :class:`~modules.model.Backend` whose ``model`` is a
        ``SegformerForSemanticSegmentation`` and whose ``processor`` is ``None``.

    Raises:
        DependencyError: transformers is not importable.
        ModelUnavailable: No local checkpoint, and ``features.network`` is off.
    """
    transformers = deps.require("transformers", feature=FEATURE)
    pretrained, cache_dir = resolve(FOLDER, REPO, feature=FEATURE)

    def build():
        model = transformers.SegformerForSemanticSegmentation.from_pretrained(
            pretrained, cache_dir=cache_dir, token=False
        )
        return None, model

    return managed(("body_parts", pretrained), build, device=device, name="SegFormer Body Parts")

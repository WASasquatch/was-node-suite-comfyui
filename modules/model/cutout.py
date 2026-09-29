"""The cutout models, listed from the ``birefnet`` and ``ben2`` model folders.

:func:`offered` lists every usable checkpoint as ``family/file``; :func:`load` answers a
:class:`Cutout` carrying the network beside the side it reads at.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from . import ben2, birefnet, model_file_path, model_files, network_enabled

__all__ = ["Cutout", "DEFAULT", "FAMILIES", "LEGACY", "PUBLISHED", "load", "offered"]

#: Config key of the feature group these nodes are gated on.
FEATURE = "features.preprocessors"

#: Model folder -> the module building its network.
FAMILIES = {"birefnet": birefnet, "ben2": ben2}

#: Every published checkpoint as a menu name, in the order the menu lists them.
PUBLISHED = tuple(f"birefnet/{name}" for name in birefnet.MODELS.values()) + (
    f"ben2/{ben2.FILENAME}",
)

#: The checkpoint a new node selects.
DEFAULT = PUBLISHED[0]

#: Names the model menu offered before it listed files -> the menu name now.
LEGACY = {label: f"birefnet/{name}" for label, name in birefnet.MODELS.items()}
LEGACY["BEN2"] = f"ben2/{ben2.FILENAME}"


@dataclass(frozen=True)
class Cutout:
    """A built cutout network and what a caller must know to drive it.

    Attributes:
        backend: The ``Backend`` holding the network.
        name: Menu name the network was built from, ``family/file``.
        family: Model folder it came from, ``birefnet`` or ``ben2``.
        side: Square side the frame is read at.
    """

    backend: object
    name: str
    family: str
    side: int


def _listed(folder: str) -> list[str]:
    """Checkpoints in one model folder whose header matches its network."""
    module = FAMILIES[folder]
    names = []
    for name in model_files(folder, suffixes=(".safetensors",)):
        parts = name.replace("\\", "/").split("/")
        if any(part.startswith(("models--", ".")) for part in parts[:-1]):
            continue
        path = model_file_path(folder, name)
        if path is not None and module.fits(path):
            names.append(f"{folder}/{name.replace(os.sep, '/')}")
    return names


def offered() -> list[str]:
    """What the model menu lists: usable files on disk, then published files a run can fetch.

    Returns:
        Menu names, ``family/file``, published files first in their preferred order. Empty
        when neither folder holds a usable file and ``features.network`` is off.
    """
    found = _listed("birefnet") + _listed("ben2")
    order = {name: index for index, name in enumerate(PUBLISHED)}
    found.sort(key=lambda name: (order.get(name, len(order)), name.lower()))
    if network_enabled():
        found += [name for name in PUBLISHED if name not in found]
    return found


def load(model: str = DEFAULT) -> Cutout:
    """Build or return the cached cutout network for one checkpoint.

    Args:
        model: A name from :func:`offered`, or a key of :data:`LEGACY`.

    Returns:
        A :class:`Cutout`, its weights resting until ``backend.load()`` is called.

    Raises:
        ValueError: ``model`` names no model folder, or the file is not that folder's network.
        ModelUnavailable: The file is not on disk and cannot be fetched.
    """
    model = LEGACY.get(model, model).replace("\\", "/")
    family, _, name = model.partition("/")
    module = FAMILIES.get(family)
    if module is None or not name:
        raise ValueError(
            f"Cutout model {model!r} is not a file in models/birefnet or models/ben2. "
            "Pick one from the model menu."
        )
    if module is ben2:
        return Cutout(ben2.load(name), model, family, ben2.TRAINED_SIDE)
    return Cutout(birefnet.load(name), model, family, birefnet.side(name))

"""Driving the vendored EMA-VFI network: finding its weights, building it, and running a pair.

The network itself lives in ``modules/vendor/ema_vfi``. :func:`resolve` fetches the
checkpoint from :data:`REPO_ID` or raises :class:`~..model.ModelUnavailable`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .. import log
from . import (
    NETWORK_FEATURE,
    ModelUnavailable,
    managed,
    model_directories,
    model_file_path,
    model_files,
    network_enabled,
)

__all__ = [
    "CHECKPOINTS",
    "Network",
    "DOWNLOAD_PAGE",
    "FETCHABLE",
    "FOLDER_KEY",
    "FOLDER_LOCATION",
    "NETWORK_FEATURE",
    "PAD_MULTIPLE",
    "REPO_ID",
    "SUFFIXES",
    "available",
    "backend",
    "interpolate",
    "offered",
    "resolve",
    "spec_for",
]

logger = log.get_logger("frame_interpolation")

#: ``folder_paths`` key, and the directory it names under ``models``. The key is lowercased so it
#: reads like every other model folder in an ``extra_model_paths`` file; the directory keeps the
#: project's own capitalisation, which is what its documentation tells people to create.
FOLDER_KEY = "ema_vfi"
FOLDER_LOCATION = "EMA-VFI"

#: Repository holding the four released checkpoints as safetensors, which is what a run fetches
#: from when ``features.network`` is on. Upstream releases them as pickles inside a Drive folder
#: that cannot be linked to a single file, so they are mirrored rather than linked.
REPO_ID = "WAS/EMA-VFI"

#: Where the weights come from, named in the error a missing checkpoint raises.
DOWNLOAD_PAGE = "https://huggingface.co/" + REPO_ID

#: The four released checkpoints, in both the format they are served in and the format upstream
#: released, by filename. ``features`` is the width multiplier and ``depths`` the block counts the
#: network is built with, both of which differ between the full and small models; ``any_timestep``
#: records whether the weights were trained for a timestep other than the midpoint, which is what
#: makes a multiplier above 2 worth offering. Safetensors come first, so a directory holding both
#: formats of one checkpoint offers the one that cannot execute code as it loads.
CHECKPOINTS = {
    "ours_t.safetensors": {"features": 32, "depths": (2, 2, 2, 4, 4), "any_timestep": True},
    "ours.safetensors": {"features": 32, "depths": (2, 2, 2, 4, 4), "any_timestep": False},
    "ours_small_t.safetensors": {"features": 16, "depths": (2, 2, 2, 2, 2), "any_timestep": True},
    "ours_small.safetensors": {"features": 16, "depths": (2, 2, 2, 2, 2), "any_timestep": False},
    "ours_t.pkl": {"features": 32, "depths": (2, 2, 2, 4, 4), "any_timestep": True},
    "ours.pkl": {"features": 32, "depths": (2, 2, 2, 4, 4), "any_timestep": False},
    "ours_small_t.pkl": {"features": 16, "depths": (2, 2, 2, 2, 2), "any_timestep": True},
    "ours_small.pkl": {"features": 16, "depths": (2, 2, 2, 2, 2), "any_timestep": False},
}

#: Which of :data:`CHECKPOINTS` can be fetched, being the ones :data:`REPO_ID` carries. The
#: pickles load and are never offered for download: they are upstream's own release format and
#: this pack mirrors only what it converted and checked itself.
FETCHABLE = tuple(name for name in CHECKPOINTS if name.endswith(".safetensors"))

#: Extensions a checkpoint carries, so a model folder holding notes or a readme lists neither.
SUFFIXES = tuple(sorted({Path(name).suffix for name in CHECKPOINTS}))

#: Window side the attention blocks use, upstream's ``W``.
WINDOW = 7

#: Frames are padded up to a whole multiple of this before the network sees them. The backbone
#: pads its own feature maps to whole attention windows, so an image whose size does not divide
#: cleanly arrives at the flow heads a few pixels wider than the flow does and the concatenation
#: fails. 16 is enough in practice and 32 is what upstream's own demo scripts use, so 32 it is.
PAD_MULTIPLE = 32

#: Keys in a released pickle that are not parameters. ``attn_mask`` and ``HW`` are shifted
#: window buffers the backbone registers during ``forward`` from the padded size it is given, so
#: a checkpoint's pair belongs to whatever resolution it was trained at and would be wrong at any
#: other. Upstream's own loader drops them for the same reason.
DERIVED_KEYS = ("attn_mask", "HW")


def _architecture(features: int, depths: tuple[int, ...]) -> tuple[dict, dict]:
    """The two keyword sets the backbone and the flow network are built from.

    Args:
        features: Width multiplier, upstream's ``F``.
        depths: Block count per stage, upstream's ``depth``.

    Returns:
        ``(backbone_kwargs, flow_kwargs)``.
    """
    from torch import nn

    from functools import partial

    width = features
    common = {
        "embed_dims": [width, 2 * width, 4 * width, 8 * width, 16 * width],
        "motion_dims": [0, 0, 0, 8 * width // depths[-2], 16 * width // depths[-1]],
        "depths": list(depths),
        "num_heads": [8 * width // 32, 16 * width // 32],
        "window_sizes": [WINDOW, WINDOW],
    }
    backbone = dict(
        common,
        mlp_ratios=[4, 4],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    flow = dict(
        common,
        scales=[4, 8, 16],
        hidden_dims=[4 * width, 4 * width],
        c=width,
    )
    return backbone, flow


@dataclass(frozen=True)
class Network:
    """A built EMA-VFI network and the checkpoint it was built from.

    Attributes:
        backend: The loaded backend, whose ``model`` is the network.
        name: Filename of the checkpoint the weights came from.
    """

    backend: object
    name: str


def available() -> list[str]:
    """Which recognised checkpoints are actually on disk, in the order they are preferred.

    Returns:
        Names present in a model directory, most preferred first, each relative to whichever
        directory holds it. Empty when none is, which a node turns into a combo holding only a
        placeholder rather than an empty list ComfyUI cannot render.
    """
    order = list(CHECKPOINTS)
    found = [
        name
        for name in model_files(FOLDER_KEY, FOLDER_LOCATION, SUFFIXES)
        if Path(name).name in CHECKPOINTS
    ]
    return sorted(found, key=lambda name: (order.index(Path(name).name), name))


def spec_for(name: str) -> dict:
    """The architecture a checkpoint name selects.

    Args:
        name: A name from :func:`available`, which may carry a subdirectory.

    Returns:
        The entry in :data:`CHECKPOINTS` its filename matches, or an empty dict when the name
        is not a checkpoint this pack knows.
    """
    return CHECKPOINTS.get(Path(str(name)).name, {})


def offered() -> list[str]:
    """What the checkpoint menu lists: what is on disk, and what a run could fetch.

    Returns:
        Every checkpoint on disk in the order they are preferred, followed by the ones
        ``features.network`` allows a run to fetch and which are not there yet. Empty when
        nothing is on disk and the feature is off, which a node turns into a combo holding a
        placeholder rather than an empty list ComfyUI cannot render.
    """
    found = available()
    if not network_enabled():
        return found
    return found + [name for name in FETCHABLE if name not in found]


def resolve(name: str) -> Path:
    """Locate a checkpoint by filename.

    Args:
        name: One of :data:`CHECKPOINTS`.

    Returns:
        The path to the file.

    Raises:
        ValueError: ``name`` is not a checkpoint this pack knows.
        ModelUnavailable: The file is in none of the model directories and could not be
            fetched, with ``features.network`` off or the download incomplete. The message
            names every directory searched and where to download it.
    """
    if not spec_for(name):
        raise ValueError(
            f"{name!r} is not an EMA-VFI checkpoint. Expected one of: "
            f"{', '.join(CHECKPOINTS)}."
        )
    found = model_file_path(FOLDER_KEY, name, FOLDER_LOCATION)
    if found is not None:
        return found
    # A name carrying a subdirectory still fetches under its own filename, into the top of the
    # first search directory, since nothing else knows what that subdirectory was for.
    base = Path(str(name)).name
    directories = model_directories(FOLDER_KEY, FOLDER_LOCATION)
    if network_enabled() and base in FETCHABLE and directories:
        return _fetch(base, directories[0])
    searched = ", ".join(str(directory) for directory in directories) or "no directory at all"
    fetchable = base in FETCHABLE
    raise ModelUnavailable(
        f"EMA-VFI's {name} was not found. Download it from {DOWNLOAD_PAGE} and put it in "
        f"ComfyUI/models/{FOLDER_LOCATION}. Searched: {searched}."
        + (
            f" Setting {NETWORK_FEATURE}: true in config.yaml lets a run fetch it once instead."
            if fetchable
            else f" That file is upstream's own release format, so it is not fetched for you;"
            f" the ones a run can fetch are {', '.join(FETCHABLE)}."
        )
    )


def _fetch(name: str, target: Path) -> Path:
    """Download one checkpoint into a model directory.

    Args:
        name: One of :data:`FETCHABLE`.
        target: Directory the file lands in, which is the first one searched.

    Returns:
        The downloaded file.

    Raises:
        DependencyError: ``huggingface_hub`` is not importable.
        ModelUnavailable: The download did not complete. The message names what stopped it
            and the directory to place the file in by hand.
    """
    from .. import deps

    # The key is written out here rather than passed as NETWORK_FEATURE so the group a
    # package belongs to can be read from the call, which is how requirements/ is checked
    # against the pack without importing it.
    hub = deps.require("huggingface_hub", feature="features.network")
    # Said at info: a first run fetches a quarter of a gigabyte.
    logger.info(
        "fetching EMA-VFI's %s from %s into %s. This happens once; the file is kept.",
        name, REPO_ID, target,
    )
    try:
        target.mkdir(parents=True, exist_ok=True)
        path = hub.hf_hub_download(repo_id=REPO_ID, filename=name, local_dir=str(target))
    except Exception as error:
        raise ModelUnavailable(
            f"EMA-VFI's {name} could not be fetched from {DOWNLOAD_PAGE} "
            f"({type(error).__name__}: {error}). Download it from that page by hand and put it "
            f"in {target}, then restart ComfyUI."
        ) from error
    logger.info("EMA-VFI's %s is at %s", name, path)
    return Path(path)


def _build(path: Path, name: str):
    """Build the network and load ``path`` into it.

    Args:
        path: The checkpoint file.
        name: Its filename, which selects the architecture size.

    Returns:
        ``(None, net)``, the shape :func:`~..model.managed` wants. There is no processor; a
        frame pair is prepared by :func:`interpolate` instead.
    """
    import torch

    from ..vendor.ema_vfi import feature_extractor as extractor
    from ..vendor.ema_vfi import flow_estimation

    spec = spec_for(name)
    backbone_kwargs, flow_kwargs = _architecture(spec["features"], spec["depths"])
    net = flow_estimation.MultiScaleFlow(
        extractor.feature_extractor(**backbone_kwargs), **flow_kwargs
    )

    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        # Stored as the network wants it: no wrapper prefix, and the resolution-dependent
        # buffers left out rather than saved at a size they do not belong to.
        state = load_file(str(path))
    else:
        raw = torch.load(path, map_location="cpu", weights_only=True)
        # The released pickles were saved from a DataParallel wrapper, so every key is prefixed.
        state = {
            key.replace("module.", "", 1): value
            for key, value in raw.items()
            if key.startswith("module.") and not any(part in key for part in DERIVED_KEYS)
        }
    net.load_state_dict(state, strict=True)
    logger.debug("%s loaded, %d parameter tensors", name, len(state))
    return None, net


def backend(name: str, device: str | None = None):
    """The network for one checkpoint, built once and kept.

    Args:
        name: One of :data:`CHECKPOINTS`.
        device: Device name for inference, or ``None`` for ComfyUI's compute device.

    Returns:
        A :class:`~..model.Backend`; its ``model`` is the network and ``load()`` brings it to
        the compute device.

    Raises:
        ValueError: ``name`` is not a checkpoint this pack knows.
        ModelUnavailable: The file is not on disk.
    """
    path = resolve(name)
    # Modification time is in the key so replacing a checkpoint in place is picked up rather
    # than serving the old weights for the rest of the session.
    key = ("ema_vfi", str(path), path.stat().st_mtime_ns)
    return managed(key, lambda: _build(path, name), device=device, name="EMA-VFI")


def interpolate(net, first, second, timestep: float = 0.5):
    """One frame between two, at ``timestep`` along the way.

    Args:
        net: The network, already on the device the frames are on.
        first: The earlier frame, ``(1, 3, height, width)`` in ``[0, 1]``.
        second: The later frame, the same shape.
        timestep: Where between them to land, 0 being ``first`` and 1 being ``second``.

    Returns:
        A frame shaped like the inputs.

    Raises:
        ValueError: The two frames are not the same shape.
    """
    import torch

    if first.shape != second.shape:
        raise ValueError(
            f"the two frames differ in shape, {tuple(first.shape)} against "
            f"{tuple(second.shape)}; interpolation needs them identical"
        )
    height, width = int(first.shape[2]), int(first.shape[3])
    tall = (height + PAD_MULTIPLE - 1) // PAD_MULTIPLE * PAD_MULTIPLE
    wide = (width + PAD_MULTIPLE - 1) // PAD_MULTIPLE * PAD_MULTIPLE
    pair = torch.cat([first, second], dim=1)
    if (tall, wide) != (height, width):
        # Edges held rather than filled, so the flow heads do not see an invented black border
        # and put motion there.
        pair = torch.nn.functional.pad(
            pair, (0, wide - width, 0, tall - height), mode="replicate"
        )
    with torch.no_grad():
        predicted = net(pair, timestep=float(timestep))[3]
    return predicted[:, :, :height, :width]

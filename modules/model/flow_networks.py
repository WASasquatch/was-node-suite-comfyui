"""Learned optical flow: SEA-RAFT and FlowSeek checkpoints, built and run on frame sequences.

The networks live in ``modules/vendor/sea_raft`` and ``modules/vendor/flowseek``. Frames are
``(frames, 3, height, width)`` in ``[0, 255]``; flows are ``(pairs, 2, height, width)`` in
pixels, x first.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .. import log
from . import (
    NETWORK_FEATURE,
    ModelUnavailable,
    checkpoint_shapes,
    managed_module,
    model_directories,
    model_file_path,
    model_files,
    network_enabled,
)

__all__ = [
    "CHECKPOINTS",
    "FOLDER",
    "FlowNetwork",
    "available",
    "backend",
    "build",
    "flows",
    "offered",
    "resolve",
    "select",
]

logger = log.get_logger("flow_networks")

#: ``folder_paths`` key, and the directory it names under ``models``.
FOLDER = "optical_flow"

#: Checkpoints a run can fetch, by the filename each is kept under, most preferred first: the
#: repository holding it and its name there.
CHECKPOINTS = {
    "sea_raft_m_spring.safetensors": ("MemorySlices/Tartan-C-T-TSKH-spring540x960-M", "model.safetensors"),
    "sea_raft_s_spring.safetensors": ("MemorySlices/Tartan-C-T-TSKH-spring540x960-S", "model.safetensors"),
    "sea_raft_m_ct.safetensors": ("MemorySlices/Tartan-C-T432x960-M", "model.safetensors"),
    "flowseek_t_ct.safetensors": ("WAS/was-node-suite-weights", "optical_flow/flowseek_t_ct.safetensors"),
    "flowseek_t_tskh.safetensors": ("WAS/was-node-suite-weights", "optical_flow/flowseek_t_tskh.safetensors"),
}

#: Upstream's own release of a checkpoint, for downloading by hand: its filename and the page
#: publishing it.
RELEASES = {
    "flowseek_t_ct.safetensors": ("flowseek_T_CT.pth", "https://drive.google.com/file/d/1COOQFkMulzpBm4zMoWsaRGk7E3YcVr2I/view"),
    "flowseek_t_tskh.safetensors": ("flowseek_T_TartanCT_TSKH.pth", "https://drive.google.com/file/d/1IQoyY5PpKSadtiGuhWwVCqvgD3y8CyFd/view"),
}

#: Extensions a checkpoint carries.
SUFFIXES = (".pt", ".pth", ".safetensors")

#: How upstream's own pickled releases are named, lowercased. A pickle is listed only when its
#: name starts with one, so other networks kept in the shared folder stay out of the menu.
RELEASE_PREFIXES = ("tartan", "flowseek_")

#: Share of free device memory one batch of pairs may take.
SPARE = 0.6

#: Pairs run together at most.
MOST_PAIRS = 16

#: Frames encoded together at most.
MOST_FRAMES = 8

#: Pairs whose working memory is asked of ComfyUI before a run.
WORKING_PAIRS = 8

#: Bytes per pixel one frame takes while it is encoded, by network class.
FRAME_BYTES = {"RAFT": 256, "FlowSeek": 660}

#: Bytes per pixel one pair's activations take beside its correlation volume.
PAIR_BYTES = 300

#: Levels in the correlation pyramid.
LEVELS = 4

#: Shortest side, in pixels, a network runs at; smaller frames are enlarged to it.
SHORTEST_SIDE = 128


@dataclass
class FlowNetwork:
    """A built flow network and how it is run.

    Attributes:
        backend: The loaded :class:`~modules.model.Backend`; its ``model`` is the network.
        name: Filename of the checkpoint the weights came from.
        family: ``"SEA-RAFT"`` or ``"FlowSeek"``.
        iterations: Refinement passes per pair.
    """

    backend: object
    name: str
    family: str
    iterations: int

    def describe(self) -> str:
        """One line naming the network and its passes."""
        return f"{self.family} ({Path(self.name).stem}, {self.iterations} passes)"


def _signature(keys) -> str | None:
    """``"flowseek"``, ``"raft"`` or ``None`` for a set of tensor names."""
    keys = set(keys)
    if "fnet.conv1.weight" not in keys or "update_block.refine.0.dwconv.weight" not in keys:
        return None
    return "flowseek" if "dav2.pretrained.cls_token" in keys else "raft"


def available() -> list[str]:
    """Flow checkpoints on disk: safetensors files whose tensors match, and upstream's pickles.

    Returns:
        Names relative to whichever model directory holds them, :data:`CHECKPOINTS` first in
        its order, then the rest by name.
    """
    found = []
    for name in model_files(FOLDER, suffixes=SUFFIXES):
        if not name.lower().endswith(".safetensors"):
            if Path(name).name.lower().startswith(RELEASE_PREFIXES):
                found.append(name)
            continue
        path = model_file_path(FOLDER, name)
        if path is not None and _signature(checkpoint_shapes(path)):
            found.append(name)
    order = list(CHECKPOINTS)
    return sorted(found, key=lambda name: (order.index(Path(name).name) if Path(name).name in order else len(order), name))


def offered() -> list[str]:
    """What the checkpoint menu lists: what is on disk, then what a run could fetch.

    Returns:
        Names on disk, followed by the fetchable ones not yet there when ``features.network``
        is on.
    """
    found = available()
    if not network_enabled():
        return found
    on_disk = {Path(name).name for name in found}
    return found + [name for name in CHECKPOINTS if name not in on_disk]


def resolve(name: str) -> Path:
    """Locate a checkpoint, fetching it when it is one of :data:`CHECKPOINTS`.

    Args:
        name: A name from :func:`offered`.

    Returns:
        The path to the file.

    Raises:
        ModelUnavailable: The file is not on disk and could not be fetched.
    """
    found = model_file_path(FOLDER, name)
    if found is not None:
        return found
    base = Path(str(name)).name
    directories = model_directories(FOLDER)
    source = CHECKPOINTS.get(base)
    if source is not None and network_enabled() and directories:
        return _fetch(base, source, directories[0])
    searched = ", ".join(str(directory) for directory in directories) or "no directory at all"
    where = (
        f"{_by_hand(base)} Setting {NETWORK_FEATURE}: true in config.yaml fetches it from "
        "Hugging Face on first use instead."
        if source is not None
        else f"Put it in ComfyUI/models/{FOLDER}."
    )
    raise ModelUnavailable(f"The flow checkpoint {name} was not found. {where} Searched: {searched}.")


def _by_hand(name: str) -> str:
    """Where one of :data:`CHECKPOINTS` can be downloaded by hand, as a sentence."""
    repo_id, filename = CHECKPOINTS[name]
    text = (
        f"Download {filename} from https://huggingface.co/{repo_id} and save it as "
        f"ComfyUI/models/{FOLDER}/{name}"
    )
    release = RELEASES.get(name)
    if release is not None:
        text += f", or download upstream's {release[0]} from {release[1]} into ComfyUI/models/{FOLDER}"
    return text + "."


def _fetch(name: str, source: tuple[str, str], target: Path) -> Path:
    """Download one checkpoint and keep it under ``name`` in ``target``.

    Raises:
        DependencyError: ``huggingface_hub`` is not importable.
        ModelUnavailable: The download did not complete.
    """
    from .. import deps

    hub = deps.require("huggingface_hub", feature="features.network")
    repo_id, filename = source
    logger.info("fetching %s from %s into %s. This happens once; the file is kept.", name, repo_id, target)
    landing = target / f".{Path(name).stem}.download"
    try:
        target.mkdir(parents=True, exist_ok=True)
        fetched = hub.hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(landing))
        final = target / name
        os.replace(fetched, final)
    except Exception as error:
        raise ModelUnavailable(
            f"The flow checkpoint {name} could not be fetched from https://huggingface.co/"
            f"{repo_id} ({type(error).__name__}: {error}). {_by_hand(name)} Then restart ComfyUI."
        ) from error
    finally:
        import shutil

        shutil.rmtree(landing, ignore_errors=True)
    logger.info("%s is at %s", name, final)
    return final


def _state(path: Path) -> dict:
    """Every tensor in a checkpoint, wrapper prefixes removed."""
    import torch

    if path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path))
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
        for wrapper in ("model", "state_dict"):
            if isinstance(state, dict) and isinstance(state.get(wrapper), dict):
                state = state[wrapper]
    return {key.removeprefix("module."): value for key, value in state.items()}


def _architecture(state: dict) -> tuple[str, dict]:
    """The network class and the keyword set a checkpoint's tensors describe.

    Raises:
        ValueError: The tensors are neither SEA-RAFT's nor FlowSeek's.
    """
    family = _signature(state)
    if family is None:
        raise ValueError("it holds neither SEA-RAFT nor FlowSeek weights")
    dim = int(state["init_conv.weight"].shape[0]) // 2
    taps = int(state["update_block.encoder.convc1.weight"].shape[1]) // 4
    radius = (round(taps ** 0.5) - 1) // 2
    blocks = len({key.split(".")[2] for key in state if key.startswith("update_block.refine.")})
    kwargs = dict(
        pretrain="resnet34" if "fnet.layer1.2.conv1.weight" in state else "resnet18",
        dim=dim,
        radius=radius,
        num_blocks=blocks,
        initial_dim=int(state["fnet.conv1.weight"].shape[0]),
        block_dims=tuple(int(state[f"fnet.layer{i}.0.conv1.weight"].shape[0]) for i in (1, 2, 3)),
    )
    if family == "flowseek":
        width = int(state["dav2.pretrained.cls_token"].shape[-1])
        sizes = {384: "vits", 768: "vitb"}
        if width not in sizes:
            raise ValueError(f"its depth network is {width} wide, and only 384 and 768 are known")
        kwargs["da_size"] = sizes[width]
    return family, kwargs


def build(path):
    """Build the network a checkpoint describes and load it.

    Args:
        path: A checkpoint file.

    Returns:
        The network in eval mode, on the CPU.

    Raises:
        ValueError: The file does not hold SEA-RAFT or FlowSeek weights.
    """
    path = Path(path)
    state = _state(path)
    try:
        family, kwargs = _architecture(state)
    except (KeyError, ValueError) as error:
        raise ValueError(f"{path.name} is not a flow checkpoint this pack can build: {error}.") from error
    if family == "flowseek":
        from ..vendor.flowseek.flowseek import FlowSeek as network_class
    else:
        from ..vendor.sea_raft.raft import RAFT as network_class
    net = network_class(**kwargs)
    wanted = net.state_dict()
    # A shared BatchNorm is saved under one of its two names; both read the same tensors.
    for key in wanted:
        if key not in state:
            for one, other in ((".downsample.1.", ".bn3."), (".bn3.", ".downsample.1.")):
                twin = key.replace(one, other)
                if one in key and twin in state:
                    state[key] = state[twin]
    net.load_state_dict(state, strict=True)
    logger.debug("%s built as %s %s", path.name, family, kwargs)
    return net.eval()


def _family(path: Path) -> str:
    """``"SEA-RAFT"`` or ``"FlowSeek"`` for a checkpoint, from its tensor names or its release name."""
    if path.suffix.lower() == ".safetensors":
        found = _signature(checkpoint_shapes(path))
    else:
        found = "flowseek" if path.name.lower().startswith("flowseek") else "raft"
    return "FlowSeek" if found == "flowseek" else "SEA-RAFT"


def backend(name: str, device: str | None = None):
    """The network for one checkpoint, built once and registered with ComfyUI's model management.

    Args:
        name: A name from :func:`offered`.
        device: Device name for inference, or ``None`` for ComfyUI's compute device.

    Returns:
        A :class:`~modules.model.Backend`, named ``"SEA-RAFT"`` or ``"FlowSeek"``.

    Raises:
        ModelUnavailable: The file is not on disk and could not be fetched.
        ValueError: The file does not hold SEA-RAFT or FlowSeek weights.
    """
    path = resolve(name)
    key = ("flow_network", str(path), path.stat().st_mtime_ns)
    return managed_module(key, lambda: build(path), device=device, name=_family(path))


def select(features: dict, index) -> dict:
    """The entries of an encoding at ``index``, a list of frame positions."""
    return {key: value[index] if hasattr(value, "shape") else value for key, value in features.items()}


def _memory(device) -> tuple[int, int] | None:
    """``(free, total)`` bytes on ``device`` as ComfyUI counts them, or ``None`` where nothing can say."""
    try:
        import comfy.model_management as management
    except ImportError:
        return None
    if device.type == "cpu":
        return None
    return int(management.get_free_memory(device)), int(management.get_total_memory(device))


def _volume(height: int, width: int, local: int) -> int:
    """Bytes one pair's correlation volume takes with its ``local`` finest levels computed on demand."""
    cells = ((height + 7) // 8) * ((width + 7) // 8)
    return sum(4 * cells * cells // 4 ** level for level in range(local, LEVELS))


def flows(network: FlowNetwork, images, progress=None):
    """Forward and backward flow between every pair of neighbouring frames.

    Args:
        network: The :class:`FlowNetwork` to run.
        images: ``(frames, 3, height, width)`` in ``[0, 255]``, two or more frames.
        progress: Optional callable taking a count of ordered pairs finished.

    Returns:
        ``(ahead, behind)``, each ``(frames - 1, 2, height, width)`` on the network's device:
        frame ``i`` onto ``i + 1``, and frame ``i + 1`` onto ``i``.
    """
    import torch
    import torch.nn.functional as F

    from ..image.optical_flow import resize_flow

    count, _, shown_height, shown_width = (int(v) for v in images.shape)
    scale = max(1.0, SHORTEST_SIDE / min(shown_height, shown_width))
    height, width = round(shown_height * scale), round(shown_width * scale)
    if scale > 1.0:
        images = F.interpolate(images, size=(height, width), mode="bilinear", align_corners=False)
    from ..vendor.sea_raft.corr import LOCAL_CHUNK_BYTES

    net = network.backend.model
    frame_bytes = height * width * FRAME_BYTES.get(type(net).__name__, max(FRAME_BYTES.values()))
    pair_bytes = height * width * PAIR_BYTES
    before = _memory(network.backend.load_device)
    local = 0
    while before is not None and local < LEVELS and (
        pair_bytes + _volume(height, width, local) + (2 * LOCAL_CHUNK_BYTES if local else 0)
        > SPARE * before[1]
    ):
        local += 1
    per_pair = pair_bytes + _volume(height, width, local)
    overhead = 2 * LOCAL_CHUNK_BYTES if local else 0
    wanted = max(MOST_FRAMES * frame_bytes, WORKING_PAIRS * per_pair + overhead)
    if before is not None:
        wanted = min(wanted, SPARE * before[1])
    device = network.backend.load(memory_required=wanted)
    after = _memory(device)
    budget = None if after is None else int(after[0] * SPARE)
    at_once_frames = MOST_FRAMES if budget is None else max(1, min(MOST_FRAMES, budget // frame_bytes))
    if local:
        logger.info(
            "flow at %dx%d correlates its %d finest level(s) on demand: the full volume needs %.1f GB",
            width, height, local, _volume(height, width, 0) / 1e9,
        )
    with torch.inference_mode():
        parts = []
        for start in range(0, count, at_once_frames):
            chunk = images[start:start + at_once_frames].to(device=device, dtype=torch.float32)
            parts.append(net.encode(chunk))
        encoded = {
            key: torch.cat([part[key] for part in parts]) if hasattr(parts[0][key], "shape") else parts[0][key]
            for key in parts[0]
        }
        del parts
        left = _memory(device)
        budget = None if left is None else int(left[0] * SPARE) - overhead
        at_once_pairs = MOST_PAIRS if budget is None else max(1, min(MOST_PAIRS, budget // per_pair))
        pairs = count - 1
        sources = list(range(pairs)) + list(range(1, count))
        targets = list(range(1, count)) + list(range(pairs))
        answers = []
        for start in range(0, len(sources), at_once_pairs):
            first = select(encoded, sources[start:start + at_once_pairs])
            second = select(encoded, targets[start:start + at_once_pairs])
            answers.append(net.estimate(first, second, network.iterations, local=local))
            if progress is not None:
                progress(len(answers[-1]))
        found = resize_flow(torch.cat(answers), shown_height, shown_width)
    return found[:pairs], found[pairs:]

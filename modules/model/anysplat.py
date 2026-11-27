"""AnySplat: unposed pictures to Gaussian splats and cameras in one pass; checkpoints found or fetched.

Splats come out in the splat frame, the first picture's camera at the origin, in the model's units.
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
    "FRAMINGS",
    "MAX_PICTURES",
    "Reconstruction",
    "available",
    "backend",
    "build",
    "offered",
    "prepare",
    "reconstruct",
    "resolve",
]

logger = log.get_logger("anysplat")

#: ``folder_paths`` key, shared with core's depth and geometry estimators.
FOLDER = "geometry_estimation"

#: Checkpoints a run can fetch, by the filename each is kept under: the repository holding it
#: and its name there.
CHECKPOINTS = {"anysplat.safetensors": ("lhjiang/anysplat", "model.safetensors")}

#: Extensions a checkpoint carries.
SUFFIXES = (".safetensors",)

#: Tensor that marks an AnySplat checkpoint, and its shape.
MARKER = ("encoder.gaussian_param_head.scratch.output_conv2.2.weight", [84, 128, 1, 1])

#: How pictures meet the network: kept whole with each side trimmed to a whole number of patches,
#: or cropped to a centred square.
FRAMINGS = ("whole picture", "square crop")

#: Most pictures one pass takes.
MAX_PICTURES = 64

#: Side of a fusion voxel, in the model's scene units.
VOXEL = 0.002

#: Gaussians fainter than this are dropped.
FAINTEST = 1.0 / 255.0

#: Harmonic degree kept, the highest core's splat files and renderers read.
KEPT_DEGREE = 3

#: Rows of per-pixel Gaussians fused at once.
FUSE_ROWS = 1 << 20


def available() -> list[str]:
    """AnySplat checkpoints on disk, :data:`CHECKPOINTS` first."""
    found = []
    name, shape = MARKER
    for entry in model_files(FOLDER, suffixes=SUFFIXES):
        path = model_file_path(FOLDER, entry)
        if path is not None and checkpoint_shapes(path).get(name) == shape:
            found.append(entry)
    order = list(CHECKPOINTS)
    return sorted(found, key=lambda entry: (order.index(Path(entry).name) if Path(entry).name in order
                                            else len(order), entry))


def offered() -> list[str]:
    """What the checkpoint menu lists: what is on disk, then what a run could fetch."""
    found = available()
    if not network_enabled():
        return found or list(CHECKPOINTS)
    on_disk = {Path(entry).name for entry in found}
    return found + [entry for entry in CHECKPOINTS if entry not in on_disk]


def _by_hand(name: str) -> str:
    repo_id, filename = CHECKPOINTS[name]
    return (f"Download {filename} from https://huggingface.co/{repo_id} and save it as "
            f"ComfyUI/models/{FOLDER}/{name}.")


def resolve(name: str) -> Path:
    """Locate a checkpoint, fetching it when it is one of :data:`CHECKPOINTS`.

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
    where = (f"{_by_hand(base)} Setting {NETWORK_FEATURE}: true in config.yaml fetches it on first use instead."
             if source is not None else f"Put it in ComfyUI/models/{FOLDER}.")
    raise ModelUnavailable(f"The AnySplat checkpoint {name} was not found. {where} Searched: {searched}.")


def _fetch(name: str, source: tuple[str, str], target: Path) -> Path:
    """Download one checkpoint and keep it under ``name`` in ``target``.

    Raises:
        DependencyError: ``huggingface_hub`` is not importable.
        ModelUnavailable: The download did not complete.
    """
    import shutil

    from .. import deps

    hub = deps.require("huggingface_hub", feature="features.network")
    repo_id, filename = source
    logger.info("fetching %s from %s into %s. This happens once; the file is kept.", name, repo_id, target)
    landing = target / f".{Path(name).stem}.download"
    try:
        target.mkdir(parents=True, exist_ok=True)
        fetched = hub.hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(landing), token=False)
        final = target / name
        os.replace(fetched, final)
    except Exception as error:
        raise ModelUnavailable(
            f"The AnySplat checkpoint {name} could not be fetched from https://huggingface.co/{repo_id} "
            f"({type(error).__name__}: {error}). {_by_hand(name)} Then restart ComfyUI.") from error
    finally:
        shutil.rmtree(landing, ignore_errors=True)
    logger.info("%s is at %s", name, final)
    return final


def build(path):
    """Build the network and load a checkpoint into it.

    Returns:
        The network in eval mode on the CPU, its transformer in bfloat16 as published.

    Raises:
        ValueError: The file does not hold AnySplat.
    """
    import torch
    from safetensors.torch import load_file

    from .anysplat_network import AnySplatNetwork

    state = {key.removeprefix("encoder."): value for key, value in load_file(str(path)).items()}
    if MARKER[0].removeprefix("encoder.") not in state:
        raise ValueError(f"{Path(path).name} does not hold AnySplat")
    with torch.device("meta"):
        network = AnySplatNetwork()
    network.load_state_dict(state, strict=True, assign=True)
    if not torch.cuda.is_available():
        network = network.float()
    return network.eval().requires_grad_(False)


def backend(name: str, device: str | None = None):
    """The network for one checkpoint, built once and registered with ComfyUI's model management.

    Raises:
        ModelUnavailable: The file is not on disk and could not be fetched.
    """
    path = resolve(name)
    key = ("anysplat", str(path), path.stat().st_mtime_ns)
    return managed_module(key, lambda: build(path), device=device, name="AnySplat")


def prepare(pictures, size: int, framing: str):
    """Pictures ``(N, H, W, 3)`` resized for the network, sides whole numbers of patches.

    Args:
        pictures: Colours in ``[0, 1]``.
        size: Shorter side in pixels, rounded down to a whole number of patches.
        framing: One of :data:`FRAMINGS`.

    Returns:
        ``(N, h, w, 3)`` float32.
    """
    import torch.nn.functional as F

    from .anysplat_network import PATCH

    side = max(PATCH * 4, int(size) // PATCH * PATCH)
    height, width = int(pictures.shape[1]), int(pictures.shape[2])
    scale = side / min(height, width)
    new = (max(side, round(height * scale)), max(side, round(width * scale)))
    frames = pictures[..., :3].float().permute(0, 3, 1, 2)
    frames = F.interpolate(frames, size=new, mode="bilinear", align_corners=False, antialias=True).clamp(0, 1)
    if framing == "square crop":
        keep = (side, side)
    else:
        keep = (new[0] // PATCH * PATCH, new[1] // PATCH * PATCH)
    top, left = (new[0] - keep[0]) // 2, (new[1] - keep[1]) // 2
    return frames[:, :, top:top + keep[0], left:left + keep[1]].permute(0, 2, 3, 1).contiguous()


@dataclass
class Reconstruction:
    """What one pass made.

    Attributes:
        gaussians: The splats, on the CPU.
        cameras: Each picture's camera as ``camera_info``.
        pictures: The pictures as the network saw them, ``(N, h, w, 3)``.
        fused: Per-pixel Gaussians before fusion.
        points: Each pixel's world position, ``(N, h, w, 3)`` on the CPU, matching ``pictures``.
    """

    gaussians: object
    cameras: list[dict]
    pictures: object
    fused: int
    points: object = None


def _matrices(quaternions):
    """Rotation matrices ``(N, 3, 3)`` of xyzw quaternions ``(N, 4)``."""
    import torch

    x, y, z, w = quaternions.unbind(-1)
    two = 2.0 / (quaternions * quaternions).sum(-1)
    return torch.stack([
        1 - two * (y * y + z * z), two * (x * y - z * w), two * (x * z + y * w),
        two * (x * y + z * w), 1 - two * (x * x + z * z), two * (y * z - x * w),
        two * (x * z - y * w), two * (y * z + x * w), 1 - two * (x * x + y * y),
    ], -1).reshape(-1, 3, 3)


def _fuse(points, features, confidence):
    """Confidence-weighted means of points and features per voxel of :data:`VOXEL`.

    Returns:
        ``(points (V, 3), features (V, C))``.
    """
    import torch

    cells = torch.round(points / VOXEL).long()
    cells = cells - cells.min(0).values
    spans = cells.max(0).values + 1
    keys = (cells[:, 0] * spans[1] + cells[:, 1]) * spans[2] + cells[:, 2]
    _, inverse = torch.unique(keys, return_inverse=True)
    count = int(inverse.max()) + 1
    peak = torch.full((count,), -torch.inf, device=points.device).scatter_reduce(0, inverse, confidence, "amax")
    weight = torch.exp(confidence - peak[inverse])
    total = torch.zeros(count, device=points.device).index_add_(0, inverse, weight)
    weight = weight / (total[inverse] + 1e-6)
    fused_points = torch.zeros(count, 3, device=points.device)
    fused = torch.zeros(count, features.shape[1], device=points.device)
    for start in range(0, points.shape[0], FUSE_ROWS):
        part = slice(start, start + FUSE_ROWS)
        fused_points.index_add_(0, inverse[part], points[part] * weight[part, None])
        fused.index_add_(0, inverse[part], features[part] * weight[part, None])
    return fused_points, fused


def reconstruct(name: str, pictures, size: int = 448, framing: str = FRAMINGS[0]) -> Reconstruction:
    """Gaussian splats and a camera per picture from unposed pictures.

    Args:
        name: A checkpoint from :func:`offered`.
        pictures: ``(N, H, W, 3)`` colours in ``[0, 1]``, 2 to :data:`MAX_PICTURES` of them.
        size: Shorter side the network sees, in pixels.
        framing: One of :data:`FRAMINGS`.

    Returns:
        The reconstruction.

    Raises:
        ModelUnavailable: The checkpoint is not on disk and could not be fetched.
        ValueError: Too few or too many pictures.
    """
    import torch
    import torch.nn.functional as F

    from ..world import cameras
    from ..world.gaussians import Gaussians
    from .anysplat_network import GAUSSIAN_CHANNELS, SH_BANDS

    count = int(pictures.shape[0])
    if count > MAX_PICTURES:
        raise ValueError(f"{count} pictures is more than the {MAX_PICTURES} one pass takes; pick fewer frames")
    seen = prepare(pictures, size, framing)
    height, width = int(seen.shape[1]), int(seen.shape[2])
    net = backend(name)
    frames_bytes = count * (height // 14) * (width // 14) * 2048 * 4 * 8
    device = net.load(memory_required=frames_bytes + 4 * 1024 ** 3)
    with torch.inference_mode():
        images = seen.permute(0, 3, 1, 2).to(device)
        poses, depth, _, raw = net.model(images)
        poses = poses.float()
        rotations = _matrices(poses[:, 3:7])
        shifts = poses[:, :3]
        if height <= width:
            fy = (height / 2.0) / (torch.tan(poses[:, 7] / 2.0) + 1e-3)
        else:
            fy = (width / 2.0) / (torch.tan(poses[:, 8] / 2.0) + 1e-3)
        fx = fy
        rows = torch.arange(height, device=device, dtype=torch.float32)[None, :, None]
        cols = torch.arange(width, device=device, dtype=torch.float32)[None, None, :]
        local = torch.stack(((cols - width / 2.0) * depth / fx[:, None, None],
                             (rows - height / 2.0) * depth / fy[:, None, None], depth), -1)
        world = torch.einsum("fij,fhwj->fhwi", rotations.transpose(1, 2), local - shifts[:, None, None, :])
        features = raw[:, :GAUSSIAN_CHANNELS].permute(0, 2, 3, 1).reshape(-1, GAUSSIAN_CHANNELS)
        confidence = raw[:, GAUSSIAN_CHANNELS].reshape(-1)
        points, fused = _fuse(world.reshape(-1, 3), features, confidence)
        opacity = torch.sigmoid(fused[:, 0])
        scales = (0.001 * F.softplus(fused[:, 1:4])).clamp_max(0.3)
        quaternion = fused[:, 4:8] / (fused[:, 4:8].norm(dim=-1, keepdim=True) + 1e-8)
        quaternion = torch.cat((quaternion[:, 3:4], quaternion[:, :3]), -1)
        mask = torch.ones(SH_BANDS, device=device)
        for degree in range(1, 5):
            mask[degree ** 2:(degree + 1) ** 2] = 0.1 * 0.25 ** degree
        sh = (fused[:, 8:8 + 3 * SH_BANDS].reshape(-1, 3, SH_BANDS) * mask).transpose(1, 2)
        sh = sh[:, :(KEPT_DEGREE + 1) ** 2].contiguous()
        keep = opacity >= FAINTEST
        splats = Gaussians(points[keep], scales[keep], quaternion[keep], opacity[keep], sh[keep]).to("cpu")
        found = []
        for index in range(count):
            turn = rotations[index].double().cpu()
            eye = -(turn.T @ shifts[index].double().cpu())
            found.append(cameras.from_rows(eye, turn, cameras.fov_for_focal(float(fy[index]), width, height)))
    return Reconstruction(splats, found, seen, count * height * width, world.float().cpu())

"""One subject picked on a key frame and followed through a batch with SAM 3, as a soft matte.

Frames are ``(batch, height, width, channels)`` picture codes; mattes are ``(batch, height,
width)`` floats in 0 to 1.
"""

from __future__ import annotations

import json
import math

import torch
import torch.nn.functional as F

from .. import log
from . import edit_regions
from .develop import soft_blur
from .guided import filter_with_guide

__all__ = ["TRACK_MODES", "Pick", "finish", "parse_points", "pick_subject", "track"]

logger = log.get_logger("image.track")

#: Which way a subject is followed from its key frame.
TRACK_MODES = ("both ways", "forward", "backward")

#: Most points one pick may carry.
MAX_POINTS = 32

#: Least confidence a text find keeps.
FIND_THRESHOLD = 0.4

#: Guided filter radius that snaps a matte to the picture's edges, as a share of the short side.
EDGE_RADIUS = 0.004


class Pick:
    """The subject on its key frame and how it was chosen.

    Attributes:
        mask: ``(height, width)`` float mask.
        source: ``points``, ``text`` or ``mask``.
        candidates: Text finds as ``{"index", "box", "score"}``, largest first.
        picked: Index of the find used, or -1.
    """

    def __init__(self, mask, source, candidates=(), picked=-1):
        self.mask = mask
        self.source = source
        self.candidates = list(candidates)
        self.picked = picked


def parse_points(text: str, node: str) -> list[tuple[float, float, bool]]:
    """Read clicked points.

    Args:
        text: JSON such as ``[{"x": 0.5, "y": 0.4, "keep": true}]``, or empty.
        node: The node's name, for messages.

    Returns:
        ``(x, y, keep)`` with x and y held to 0 to 1, at most :data:`MAX_POINTS`.

    Raises:
        ValueError: The text is not a list of points.
    """
    if not str(text or "").strip():
        return []
    example = 'such as [{"x": 0.5, "y": 0.4, "keep": true}]'
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{node}: points is not JSON {example}. Click the subject on the node to set them.") from error
    if not isinstance(data, list):
        raise ValueError(f"{node}: points must be a list {example}.")
    points = []
    for entry in data[:MAX_POINTS]:
        if not isinstance(entry, dict) or "x" not in entry or "y" not in entry:
            raise ValueError(f"{node}: every point needs x and y, {example}.")
        try:
            x, y = float(entry["x"]), float(entry["y"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"{node}: point x and y must be numbers from 0 to 1.") from error
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"{node}: point x and y must be numbers from 0 to 1.")
        points.append((min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0), bool(entry.get("keep", True))))
    return points


def _pointed(model, picture: torch.Tensor, points, node: str) -> torch.Tensor:
    """The thing under keep points and away from exclude points, at the picture's size."""
    import comfy.model_management as management

    sam = edit_regions.sam_network(model, node)
    management.load_model_gpu(model)
    device, dtype = management.get_torch_device(), model.model.get_dtype()
    side = edit_regions.DETECT_SIDE
    height, width = picture.shape[:2]
    frame = F.interpolate(picture[None, ..., :3].movedim(-1, 1).float(), size=(side, side), mode="bilinear",
                          align_corners=False).to(device=device, dtype=dtype)
    prompts = {
        "point_coords": torch.tensor([[[x * side, y * side] for x, y, _ in points]], dtype=dtype, device=device),
        "point_labels": torch.tensor([[1 if keep else 0 for _, _, keep in points]], dtype=torch.int32, device=device),
    }
    with torch.no_grad():
        logits = sam.forward_segment(frame, point_inputs=prompts)
        for _ in range(edit_regions.REFINE_PASSES - 1):
            logits = sam.forward_segment(frame, mask_inputs=logits[:, :1])
    return (F.interpolate(logits[:, :1].float(), size=(height, width), mode="bilinear", align_corners=False)[0, 0].cpu() > 0).float()


def pick_subject(model, clip, picture: torch.Tensor, subject: str, points, pick: int, mask, node: str) -> Pick:
    """Choose the subject on the key frame.

    Args:
        model: SAM 3 as a MODEL.
        clip: SAM 3's text encoder, or None.
        picture: The key frame, ``(height, width, channels)``.
        subject: What to look for when no points or mask decide.
        points: From :func:`parse_points`.
        pick: Which text find to use, 0 being the largest.
        mask: ``(height, width)`` mask that decides on its own, or None.
        node: The node's name, for messages.

    Returns:
        A :class:`Pick`. Text finds are listed whenever a CLIP is wired.

    Raises:
        ValueError: Nothing can say what the subject is, or the text finds nothing.
    """
    height, width = picture.shape[:2]
    candidates, finds = [], []
    if clip is not None and str(subject or "").strip():
        finds = edit_regions.find(model, clip, picture, (str(subject).strip(),), FIND_THRESHOLD, node)
        finds = sorted(finds, key=lambda found: found.area, reverse=True)
        candidates = [{"index": index, "box": [round(v, 4) for v in found.box], "score": round(found.score, 3)}
                      for index, found in enumerate(finds)]
    if mask is not None:
        held = mask.float()
        if held.shape != (height, width):
            held = F.interpolate(held[None, None], size=(height, width), mode="bilinear", align_corners=False)[0, 0]
        return Pick(held.clamp(0.0, 1.0).cpu(), "mask", candidates)
    if points:
        if not any(keep for _, _, keep in points):
            raise ValueError(f"{node}: every point excludes; click the subject once without shift to keep it.")
        return Pick(_pointed(model, picture, points, node), "points", candidates)
    if clip is None:
        raise ValueError(
            f"{node} has nothing to pick the subject with: click it on the node's panel, wire a mask, or wire "
            "SAM 3's CLIP from the same Load Checkpoint so `subject` can be searched for."
        )
    if not finds:
        raise ValueError(
            f"{node} found no '{subject}' on frame shown as the key frame. Name it another way, move key_frame to "
            "a frame where it is clear, or click it on the panel."
        )
    index = min(max(int(pick), 0), len(finds) - 1)
    return Pick(edit_regions.segment_mask(model, picture, finds[index], node), "text", candidates, index)


def _follow(model, frames: torch.Tensor, key: torch.Tensor, node: str) -> torch.Tensor:
    """Masks for every frame of ``frames``, starting from ``key`` on frame 0."""
    import comfy.model_management as management
    import comfy.utils
    from comfy.ldm.sam3.tracker import unpack_masks

    count, height, width = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
    if count == 1:
        return key[None].clone()
    sam = edit_regions.sam_network(model, node)
    management.load_model_gpu(model)
    device, dtype = management.get_torch_device(), model.model.get_dtype()
    with torch.no_grad():
        result = sam.forward_video(
            images=frames[..., :3].movedim(-1, 1), initial_masks=key[None, None].to(device=device, dtype=dtype),
            pbar=comfy.utils.ProgressBar(count), text_prompts=None, new_det_thresh=0.5, max_objects=1,
            detect_interval=1, target_device=device, target_dtype=dtype,
        )
    packed = result.get("packed_masks")
    if packed is None or packed.shape[1] == 0:
        logger.warning("%s lost the subject straight away; only its key frame carries a matte", node)
        empty = torch.zeros((count, height, width))
        empty[0] = key
        return empty
    masks = unpack_masks(packed[:, 0]).float()[:, None]
    return F.interpolate(masks, size=(height, width), mode="bilinear", align_corners=False)[:, 0].cpu()


def track(model, frames: torch.Tensor, key_frame: int, key: torch.Tensor, mode: str, node: str) -> torch.Tensor:
    """Follow the subject from its key frame through the batch.

    Args:
        model: SAM 3 as a MODEL.
        frames: ``(batch, height, width, channels)``.
        key_frame: Index of the frame the subject was picked on.
        key: Its ``(height, width)`` mask there.
        mode: One of :data:`TRACK_MODES`.
        node: The node's name, for messages.

    Returns:
        ``(batch, height, width)`` mattes; frames the mode does not reach are empty.
    """
    if mode not in TRACK_MODES:
        raise ValueError(f"{node}: track must be one of {', '.join(TRACK_MODES)}.")
    count, height, width = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
    mattes = torch.zeros((count, height, width))
    if mode in ("both ways", "forward"):
        mattes[key_frame:] = _follow(model, frames[key_frame:], key, node)
    if mode in ("both ways", "backward") and key_frame > 0:
        back = _follow(model, frames[:key_frame + 1].flip(0), key, node).flip(0)
        mattes[:key_frame] = back[:key_frame]
    mattes[key_frame] = key
    return mattes


def finish(mattes: torch.Tensor, frames: torch.Tensor, grow: float, feather: float) -> torch.Tensor:
    """Snap each frame's matte to that frame's own edges, then grow and soften it.

    Args:
        mattes: ``(batch, height, width)``.
        frames: The frames they belong to.
        grow: Share of the short side, in hundredths, to grow by; negative shrinks.
        feather: Share of the short side, in hundredths, the edge softens over.

    Returns:
        ``(batch, height, width)`` mattes in 0 to 1.
    """
    count, height, width = mattes.shape
    short = min(height, width)
    radius = max(2, round(EDGE_RADIUS * short))
    snapped = torch.empty_like(mattes)
    for index in range(count):
        guide = frames[index:index + 1, :, :, :3].float().mean(dim=-1, keepdim=True).clamp(0.0, 1.0)
        snapped[index] = filter_with_guide(mattes[index][None, ..., None], guide, radius, 1e-3)[0, ..., 0].clamp(0.0, 1.0)
    reach = round(abs(float(grow)) / 100.0 * short)
    if reach >= 1:
        planes = snapped[:, None]
        if grow > 0:
            planes = F.max_pool2d(planes, 2 * reach + 1, 1, reach)
        else:
            planes = -F.max_pool2d(-planes, 2 * reach + 1, 1, reach)
        snapped = planes[:, 0]
    sigma = float(feather) / 100.0 * short
    if sigma >= 0.5:
        snapped = soft_blur(snapped[:, None], sigma)[:, 0]
    return snapped.clamp(0.0, 1.0)

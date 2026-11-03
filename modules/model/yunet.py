"""YuNet face detection, in torch.

The network emits a classification score, an objectness score, a bounding box and five
landmarks per anchor at each stride in :data:`STRIDES`, which :func:`detect` decodes into
pixel boxes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import log
from ..data import paths
from . import managed_module
from .yunet_net import YuNet

__all__ = [
    "DEFAULT_SIZE",
    "STRIDES",
    "WEIGHTS",
    "Detector",
    "detect",
    "load",
]

logger = log.get_logger("model.yunet")

#: Config group whose node reached here, named in an availability error.
FEATURE = "features.yunet"

#: The weights, which ship with the pack. Nothing is downloaded and nothing is installed.
WEIGHTS = "yunet.safetensors"

#: Feature-map strides the network predicts at, smallest first. Fixed by the architecture.
STRIDES = (8, 16, 32)

#: Input size used when the model accepts any, chosen to match the shape the published
#: fixed-size variant was exported at.
DEFAULT_SIZE = 640


@dataclass(frozen=True)
class Detector:
    """A loaded detector and the frame size it reads.

    Attributes:
        network: The torch module, in eval mode on ComfyUI's compute device.
        width: Frame width the network reads.
        height: Frame height the network reads.
        name: What the detector is called, for a log line.
    """

    network: object
    width: int
    height: int
    name: str


def load(name: str = WEIGHTS) -> Detector:
    """The face detector, built once and kept for the process.

    Args:
        name: The weights file, which ships with the pack.

    Returns:
        The loaded :class:`Detector`.

    Raises:
        FileNotFoundError: The weights are not beside the pack's other data.
    """
    backend = managed_module(("yunet", name), _build)
    return Detector(backend, DEFAULT_SIZE, DEFAULT_SIZE, name)


def _build() -> YuNet:
    """Read the bundled weights into a freshly built network."""
    from safetensors.torch import load_file

    path = paths.data_directory() / "models" / WEIGHTS
    if not path.is_file():
        raise FileNotFoundError(
            f"The YuNet weights are missing from the pack at {path}. Reinstall the pack: "
            f"the file ships with it and is not downloaded."
        )
    network = YuNet()
    network.load_state_dict(load_file(str(path)))
    network.eval()
    logger.debug("loaded YuNet from %s", path)
    return network


def _letterbox(image: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float, int, int]:
    """Fit an image into a fixed frame without changing its proportions.

    Args:
        image: ``RGB`` pixels shaped ``(height, width, 3)``.
        width: Frame width.
        height: Frame height.

    Returns:
        ``(framed, scale, left, top)``: the padded image, the factor the source was scaled
        by, and where it sits in the frame. The three numbers are what turn a detection
        back into source coordinates.
    """
    from PIL import Image

    source_h, source_w = image.shape[:2]
    scale = min(width / source_w, height / source_h)
    new_w, new_h = max(1, round(source_w * scale)), max(1, round(source_h * scale))
    resized = np.asarray(Image.fromarray(image).resize((new_w, new_h), Image.BILINEAR))

    framed = np.zeros((height, width, 3), dtype=image.dtype)
    left, top = (width - new_w) // 2, (height - new_h) // 2
    framed[top:top + new_h, left:left + new_w] = resized
    return framed, scale, left, top


def _decode(outputs: dict, width: int, height: int, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Turn the network's per-stride tensors into boxes and scores.

    Args:
        outputs: Output name to array, as the network answers them.
        width: Input width the tensors were produced at.
        height: Input height.
        threshold: Least confidence a detection must reach.

    Returns:
        ``(boxes, scores)``, boxes as ``(x, y, w, h)`` rows in input pixels.
    """
    boxes, scores = [], []
    for stride in STRIDES:
        cls = outputs[f"cls_{stride}"].reshape(-1)
        obj = outputs[f"obj_{stride}"].reshape(-1)
        bbox = outputs[f"bbox_{stride}"].reshape(-1, 4)

        columns, rows = width // stride, height // stride
        grid = np.stack(np.meshgrid(np.arange(columns), np.arange(rows)), axis=-1).reshape(-1, 2)
        # The two heads are trained jointly and neither is a probability on its own; their
        # geometric mean is the confidence the reference implementation reports.
        confidence = np.sqrt(np.clip(cls, 0.0, 1.0) * np.clip(obj, 0.0, 1.0))

        keep = confidence >= threshold
        if not keep.any():
            continue
        anchors, offsets = grid[keep], bbox[keep]
        centre_x = (anchors[:, 0] + offsets[:, 0]) * stride
        centre_y = (anchors[:, 1] + offsets[:, 1]) * stride
        box_w = np.exp(offsets[:, 2]) * stride
        box_h = np.exp(offsets[:, 3]) * stride
        boxes.append(np.stack([centre_x - box_w / 2, centre_y - box_h / 2, box_w, box_h], axis=1))
        scores.append(confidence[keep])

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.concatenate(boxes), np.concatenate(scores)


def _suppress(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    """Greedy non-maximum suppression.

    Args:
        boxes: ``(x, y, w, h)`` rows.
        scores: One confidence per box.
        threshold: Overlap above which the weaker of two boxes is dropped.

    Returns:
        Indices to keep, strongest first.
    """
    order, kept = scores.argsort()[::-1], []
    while order.size:
        best = order[0]
        kept.append(int(best))
        if order.size == 1:
            break
        rest = order[1:]
        x1 = np.maximum(boxes[best, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[best, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[best, 0] + boxes[best, 2], boxes[rest, 0] + boxes[rest, 2])
        y2 = np.minimum(boxes[best, 1] + boxes[best, 3], boxes[rest, 1] + boxes[rest, 3])
        overlap = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        union = boxes[best, 2] * boxes[best, 3] + boxes[rest, 2] * boxes[rest, 3] - overlap
        order = rest[overlap / np.maximum(union, 1e-9) <= threshold]
    return kept


def detect(
    detector: Detector,
    image: np.ndarray,
    score_threshold: float = 0.6,
    nms_threshold: float = 0.3,
) -> np.ndarray:
    """Find faces in one image.

    Args:
        detector: A loaded model from :func:`load`.
        image: ``RGB`` pixels shaped ``(height, width, 3)``, 8-bit.
        score_threshold: Least confidence a detection must reach, 0.0 to 1.0. Lower finds
            more faces and more false positives.
        nms_threshold: Overlap above which two detections are treated as one face.

    Returns:
        One row per face as ``(x, y, width, height, score)`` in the source image's own
        pixel coordinates, strongest first. Empty when nothing was found.
    """
    import torch

    framed, scale, left, top = _letterbox(image, detector.width, detector.height)
    # The network reads its channels in BGR order.
    blob = np.ascontiguousarray(framed[..., ::-1].astype(np.float32).transpose(2, 0, 1)[None])

    backend = detector.network
    device = backend.load()
    with torch.no_grad():
        answered = backend.model(torch.from_numpy(blob).to(device))
    outputs = {name: value.detach().cpu().float().numpy() for name, value in answered.items()}

    boxes, scores = _decode(outputs, detector.width, detector.height, score_threshold)
    if not len(boxes):
        return np.zeros((0, 5), dtype=np.float32)

    kept = _suppress(boxes, scores, nms_threshold)
    boxes, scores = boxes[kept], scores[kept]

    # Back out of the letterbox: undo the padding, then the scale.
    boxes[:, 0] = (boxes[:, 0] - left) / scale
    boxes[:, 1] = (boxes[:, 1] - top) / scale
    boxes[:, 2] = boxes[:, 2] / scale
    boxes[:, 3] = boxes[:, 3] / scale
    return np.column_stack([boxes, scores]).astype(np.float32)

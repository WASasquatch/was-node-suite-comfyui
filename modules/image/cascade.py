"""Haar and LBP classifier cascades evaluated in torch.

Reads the OpenCV cascade XML bundled with the pack. Images are 8-bit and rectangles are
``(x, y, width, height)`` in pixels of the image passed in.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F

from .. import log
from ..data import paths

__all__ = [
    "Cascade",
    "Stage",
    "detect",
    "detect_first",
    "grayscale",
    "group_rectangles",
    "load",
]

logger = log.get_logger("image.cascade")

#: Ratio in size between one pyramid level and the next.
SCALE_FACTOR = 1.1

#: Fraction of their size two detections may differ by and still be one object.
GROUP_EPS = 0.2

#: Longest side, in pixels, of the largest pyramid level searched.
MAX_DETECT_SIDE = 1280

#: Most windows evaluated in one pass.
WINDOW_BLOCK = 1 << 22

#: Feature readings one pass may hold, which sets how many windows it covers.
GATHER_BUDGET = 1 << 23

#: Fewest windows evaluated in one pass.
MIN_BLOCK = 1 << 14

#: Layout of the packed cascade files. A file written by another number is ignored.
CACHE_VERSION = 2

#: Rectangles compared against every other in one pass while grouping.
PAIR_BLOCK = 1 << 12

#: Rectangles one Haar feature may hold.
MAX_RECTS = 3

#: Categories one LBP node chooses between.
LBP_CATEGORIES = 256

#: Q14 fixed-point luminance weights for red, green and blue.
GRAY_WEIGHTS = (4899, 9617, 1868)

#: LBP cells compared against the centre one, as ``(row, column, bit)``.
LBP_NEIGHBOURS = (
    (0, 0, 128),
    (0, 1, 64),
    (0, 2, 32),
    (1, 2, 16),
    (2, 2, 8),
    (2, 1, 4),
    (2, 0, 2),
    (1, 0, 1),
)


@dataclass(frozen=True)
class Stage:
    """One stage of a cascade.

    Attributes:
        threshold: Total leaf value a window must reach to pass.
        nodes: Per node, ``(left, right, feature, split)``, where ``split`` is a threshold
            on Haar and a tuple of subset words on LBP. A branch above zero is the next
            node; anything else is the leaf at ``leaf_offset - branch``.
        leaves: Leaf values of every weak classifier, concatenated.
        roots: First node of each weak classifier.
        leaf_offsets: First leaf of each weak classifier.
    """

    threshold: float
    nodes: tuple
    leaves: tuple
    roots: tuple
    leaf_offsets: tuple


@dataclass(frozen=True)
class Cascade:
    """A classifier cascade read from OpenCV XML.

    Attributes:
        name: File the cascade was read from.
        kind: ``"haar"`` or ``"lbp"``.
        width: Detection window width in pixels.
        height: Detection window height in pixels.
        rects: Per feature, its rectangles as ``(x, y, width, height, weight)``.
        tilted: Per feature, whether its rectangles are turned 45 degrees.
        stages: The stages, in file order.
        children: Per stage, the stages tried when it passes. Empty is a detection.
    """

    name: str
    kind: str
    width: int
    height: int
    rects: tuple
    tilted: tuple
    stages: tuple
    children: tuple

    @property
    def has_tilted(self) -> bool:
        """Whether any feature is turned 45 degrees."""
        return any(self.tilted)


def grayscale(image) -> torch.Tensor:
    """Convert 8-bit RGB pixels to luminance.

    Args:
        image: ``(height, width, 3)`` of 8-bit values, as a numpy array or a tensor.

    Returns:
        A ``(height, width)`` uint8 tensor.

    Raises:
        ValueError: ``image`` is not three-channel pixels.
    """
    pixels = torch.as_tensor(image)
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        raise ValueError(f"expected (height, width, 3) pixels, got {tuple(pixels.shape)}")
    channels = pixels[:, :, :3].to(torch.int32)
    weights = torch.tensor(GRAY_WEIGHTS, dtype=torch.int32, device=channels.device)
    return (((channels * weights).sum(2) + (1 << 13)) >> 14).to(torch.uint8)


@lru_cache(maxsize=None)
def load(name: str) -> Cascade:
    """Read one of the classifier cascades bundled with the pack.

    Args:
        name: File name of a cascade in the pack's ``cascades`` directory.

    Returns:
        The parsed cascade, held after the first read.

    Raises:
        ValueError: ``name`` is not bundled with the pack, or the file holds a cascade
            this cannot evaluate.
        FileNotFoundError: The file is missing from the installation.
    """
    path = paths.cascade_file(name)
    if not path.is_file():
        raise FileNotFoundError(
            f"the classifier cascade {name} is missing from {path.parent}. The cascade "
            f"files ship inside WAS Node Suite, so reinstalling the pack restores them."
        )
    return _parse(path, name)


def detect(
    image,
    name: str,
    scale_factor: float = SCALE_FACTOR,
    min_neighbors: int = 5,
    min_size: int = 0,
    max_size: int = 0,
    device=None,
) -> list[tuple[int, int, int, int]]:
    """Find everything one cascade reports.

    Args:
        image: ``(height, width, 3)`` of 8-bit RGB pixels, or a ``(height, width)``
            8-bit array that is already luminance.
        name: File name of a bundled cascade.
        scale_factor: Ratio between pyramid levels, above 1.0.
        min_neighbors: Overlapping detections a rectangle needs to be kept.
        min_size: Shortest side, in pixels, an object may have. 0 for no limit.
        max_size: Longest side, in pixels, an object may have. 0 for no limit.
        device: Where to compute, or ``None`` for ComfyUI's device.

    Returns:
        ``(x, y, width, height)`` rectangles, in the order they were found.
    """
    return detect_first(
        image, (name,), scale_factor, min_neighbors, min_size, max_size, device
    )[1]


@torch.no_grad()
def detect_first(
    image,
    names,
    scale_factor: float = SCALE_FACTOR,
    min_neighbors: int = 5,
    min_size: int = 0,
    max_size: int = 0,
    device=None,
) -> tuple[str | None, list[tuple[int, int, int, int]]]:
    """Try cascades in turn, answering with the first one that finds anything.

    Args:
        image: ``(height, width, 3)`` of 8-bit RGB pixels, or a ``(height, width)``
            8-bit array that is already luminance.
        names: File names of bundled cascades, in the order they are tried.
        scale_factor: Ratio between pyramid levels, above 1.0.
        min_neighbors: Overlapping detections a rectangle needs to be kept.
        min_size: Shortest side, in pixels, an object may have. 0 for no limit.
        max_size: Longest side, in pixels, an object may have. 0 for no limit.
        device: Where to compute, or ``None`` for ComfyUI's device.

    Returns:
        ``(name, rectangles)`` for the first cascade that reported anything, or
        ``(None, [])`` when none of them did.

    Raises:
        ValueError: ``names`` is empty, or ``scale_factor`` is 1.0 or below.
    """
    if scale_factor <= 1.0:
        raise ValueError(f"scale_factor has to be above 1.0, got {scale_factor}")
    names = tuple(names)
    if not names:
        raise ValueError("no cascade was named")
    for name in names:
        paths.cascade_file(name)

    gray = _luminance(image)
    device = _device() if device is None else torch.device(device)
    settings = (scale_factor, min_neighbors, min_size, max_size)
    try:
        return _sweep(names, gray, device, *settings)
    except torch.cuda.OutOfMemoryError:
        if device.type == "cpu":
            raise
        logger.warning(
            "%s ran out of memory while searching a %dx%d image for a face, so it was "
            "searched on the CPU instead. The same faces are found, more slowly. A "
            "smaller image, or freeing VRAM before this node runs, avoids it.",
            device,
            gray.shape[1],
            gray.shape[0],
        )
        torch.cuda.empty_cache()
        return _sweep(names, gray, torch.device("cpu"), *settings)


def _sweep(
    names,
    gray: torch.Tensor,
    device,
    scale_factor: float,
    min_neighbors: int,
    min_size: int,
    max_size: int,
):
    """Share one pyramid between the cascades, until one of them reports something."""
    pyramid, window = None, None
    for name in names:
        packed = _packed(name, str(device))
        # A cascade with a smaller window reaches scales the pyramid does not hold yet.
        smallest = (
            (packed.width, packed.height)
            if window is None
            else (min(window[0], packed.width), min(window[1], packed.height))
        )
        if smallest != window:
            window = smallest
            pyramid = _Pyramid(
                gray, device, scale_factor, window[0], window[1], min_size
            )
            logger.debug(
                "searching %dx%d on %s over %d scale(s)",
                gray.shape[1],
                gray.shape[0],
                device,
                len(pyramid.factors),
            )
        found = group_rectangles(_run(packed, pyramid, min_size, max_size), min_neighbors)
        if found:
            return name, found
    return None, []


def group_rectangles(
    rects, min_neighbors: int, eps: float = GROUP_EPS
) -> list[tuple[int, int, int, int]]:
    """Merge overlapping detections and drop those too few of them agree on.

    Args:
        rects: ``(x, y, width, height)`` rectangles.
        min_neighbors: Overlapping detections a cluster needs to survive. 0 returns every
            rectangle unmerged.
        eps: Fraction of their size two rectangles may differ by and still merge.

    Returns:
        One averaged rectangle per surviving cluster.
    """
    rects = [tuple(int(value) for value in rect) for rect in rects]
    if min_neighbors <= 0 or not rects:
        return rects

    clusters: dict[int, list[int]] = {}
    for index, label in enumerate(_cluster(rects, eps)):
        clusters.setdefault(label, []).append(index)

    merged, weights = [], []
    for members in clusters.values():
        share = 1.0 / len(members)
        merged.append(
            tuple(_round(sum(rects[i][axis] for i in members) * share) for axis in range(4))
        )
        weights.append(len(members))

    kept = []
    for index, rect in enumerate(merged):
        weight = weights[index]
        if weight <= min_neighbors:
            continue
        covered = any(
            _inside(rect, merged[other], eps)
            and (weights[other] > max(3, weight) or weight < 3)
            for other in range(len(merged))
            if other != index and weights[other] > min_neighbors
        )
        if not covered:
            kept.append(rect)
    return kept


def _luminance(image) -> torch.Tensor:
    """Luminance for whichever of the two accepted shapes the caller passed."""
    pixels = torch.as_tensor(image)
    return pixels.to(torch.uint8) if pixels.ndim == 2 else grayscale(pixels)


def _device():
    """Where a detection runs when the caller names no device."""
    from ..model import compute_device

    device = compute_device()
    return device if device.type == "cuda" else torch.device("cpu")


def _round(value: float) -> int:
    """Round half to even, the way OpenCV rounds."""
    return int(round(value))


def _numbers(text: str) -> list[float]:
    """Whitespace-separated numbers from one XML element."""
    return [float(token) for token in (text or "").split()]


def _parse(path: Path, name: str) -> Cascade:
    """Read either of the two cascade layouts OpenCV writes.

    Args:
        path: File to read.
        name: File name, carried onto the cascade and into error messages.

    Returns:
        The parsed cascade.

    Raises:
        ValueError: The file is not a cascade, or holds features this cannot evaluate.
    """
    root = ET.parse(path).getroot()  # noqa: S314
    node = root.find("cascade")
    if node is not None:
        return _parse_current(node, name)
    for child in root:
        if child.get("type_id") == "opencv-haar-classifier":
            return _parse_legacy(child, name)
    raise ValueError(f"{name} is not an OpenCV classifier cascade")


def _parse_current(node, name: str) -> Cascade:
    """Read the ``opencv-cascade-classifier`` layout, Haar or LBP."""
    kind = (node.findtext("featureType") or "").strip().upper()
    if kind not in ("HAAR", "LBP"):
        raise ValueError(f"{name} holds {kind or 'unnamed'} features, which this cannot read")
    width, height = int(node.findtext("width")), int(node.findtext("height"))

    rects, tilted = [], []
    for feature in node.find("features"):
        if kind == "LBP":
            values = _numbers(feature.findtext("rect"))
            rects.append(((int(values[0]), int(values[1]), int(values[2]), int(values[3]), 1.0),))
            tilted.append(False)
            continue
        rows = [_numbers(rect.text) for rect in feature.find("rects")]
        if len(rows) > MAX_RECTS:
            raise ValueError(f"{name} holds a feature of {len(rows)} rectangles")
        rects.append(
            tuple(
                (int(row[0]), int(row[1]), int(row[2]), int(row[3]), float(row[4]))
                for row in rows
            )
        )
        tilted.append(bool(int(feature.findtext("tilted") or 0)))

    words = 0
    if kind == "LBP":
        words = (int(node.find("featureParams").findtext("maxCatCount")) + 31) // 32
    stages = tuple(_parse_stage(stage, words) for stage in node.find("stages"))
    children = tuple(
        (index + 1,) if index + 1 < len(stages) else () for index in range(len(stages))
    )
    return Cascade(
        name, kind.lower(), width, height, tuple(rects), tuple(tilted), stages, children
    )


def _parse_stage(node, words: int) -> Stage:
    """One stage of the current layout. ``words`` above zero reads LBP subsets."""
    nodes, leaves, roots, leaf_offsets = [], [], [], []
    step = 3 + words if words else 4
    for weak in node.find("weakClassifiers"):
        internal = weak.findtext("internalNodes").split()
        roots.append(len(nodes))
        leaf_offsets.append(len(leaves))
        for start in range(0, len(internal), step):
            entry = internal[start : start + step]
            split = tuple(int(word) for word in entry[3:]) if words else float(entry[3])
            nodes.append((int(entry[0]), int(entry[1]), int(entry[2]), split))
        leaves.extend(_numbers(weak.findtext("leafValues")))
    return Stage(
        float(node.findtext("stageThreshold")),
        tuple(nodes),
        tuple(leaves),
        tuple(roots),
        tuple(leaf_offsets),
    )


def _parse_legacy(node, name: str) -> Cascade:
    """Read the ``opencv-haar-classifier`` layout, whose stages are walked in file order."""
    size = _numbers(node.findtext("size"))
    width, height = int(size[0]), int(size[1])
    rects, tilted, seen = [], [], {}
    stages = []

    for stage_node in node.find("stages"):
        nodes, leaves, roots, leaf_offsets = [], [], [], []
        for tree in stage_node.find("trees"):
            roots.append(len(nodes))
            leaf_offsets.append(len(leaves))
            for entry in tree:
                feature = entry.find("feature")
                rows = tuple(
                    (int(row[0]), int(row[1]), int(row[2]), int(row[3]), float(row[4]))
                    for row in (_numbers(rect.text) for rect in feature.find("rects"))
                )
                key = (rows, bool(int(feature.findtext("tilted") or 0)))
                if key not in seen:
                    seen[key] = len(rects)
                    rects.append(key[0])
                    tilted.append(key[1])
                left = _legacy_branch(entry, "left", leaves, leaf_offsets[-1])
                right = _legacy_branch(entry, "right", leaves, leaf_offsets[-1])
                nodes.append((left, right, seen[key], float(entry.findtext("threshold"))))
        stages.append(
            Stage(
                float(stage_node.findtext("stage_threshold")),
                tuple(nodes),
                tuple(leaves),
                tuple(roots),
                tuple(leaf_offsets),
            )
        )

    children = tuple(
        (index + 1,) if index + 1 < len(stages) else () for index in range(len(stages))
    )
    return Cascade(
        name, "haar", width, height, tuple(rects), tuple(tilted), tuple(stages), children
    )


def _legacy_branch(entry, side: str, leaves: list, leaf_offset: int) -> int:
    """Branch index for one side of a legacy node, appending its leaf when it has one."""
    child = entry.findtext(f"{side}_node")
    if child is not None:
        return int(child)
    branch = -(len(leaves) - leaf_offset)
    leaves.append(float(entry.findtext(f"{side}_val")))
    return branch


def _corners(x: int, y: int, w: int, h: int, turned: bool) -> tuple:
    """The four ``(column, row)`` points a rectangle's sum is read from."""
    if turned:
        return (x, y, x - h, y + h, x + w, y + w, x + w - h, y + w + h)
    return (x + w, y + h, x + w, y, x, y + h, x, y)


def _depth(stage: Stage) -> int:
    """Most nodes any one weak classifier of a stage holds."""
    bounds = list(stage.roots) + [len(stage.nodes)]
    return max(1, max(bounds[i + 1] - bounds[i] for i in range(len(stage.roots))))


def _subset_table(subsets, device) -> torch.Tensor:
    """``(nodes, 256)`` truth table of each LBP node's category subset."""
    words = torch.tensor(
        [[word & 0xFFFFFFFF for word in subset] for subset in subsets], dtype=torch.int64
    )
    categories = torch.arange(LBP_CATEGORIES, dtype=torch.int64)
    return (((words[:, categories >> 5] >> (categories & 31)) & 1) > 0).to(device)


@lru_cache(maxsize=None)
def _packed(name: str, device: str) -> "_Packed":
    """A cascade's tensors on one device, built once per pair."""
    cached = _read_cache(name, device)
    if cached is not None:
        return cached
    return _Packed(load(name), torch.device(device))


def _cache_file(name: str) -> Path:
    """Where the packed form of one cascade is kept, beside its XML."""
    return paths.cascade_file(name).with_suffix(".safetensors")


def cache_payload(packed: "_Packed") -> tuple[dict, dict]:
    """``(tensors, metadata)`` for one packed cascade, ready to write."""
    tensors = {
        "corners": packed.corners.cpu(),
        "weights": packed.weights.cpu(),
        "first": packed.first.cpu(),
        "turned": packed.turned.cpu(),
    }
    keys = [key for key, value in packed.stages[0].items() if torch.is_tensor(value)]
    for key in keys:
        tensors[f"stage.{key}"] = torch.cat(
            [stage[key].cpu() for stage in packed.stages], 0
        )
    stages = []
    for stage in packed.stages:
        plain = {key: value for key, value in stage.items() if not torch.is_tensor(value)}
        plain["lengths"] = {key: int(stage[key].shape[0]) for key in keys}
        stages.append(plain)
    meta = {
        "version": CACHE_VERSION,
        "name": packed.name,
        "kind": packed.kind,
        "width": packed.width,
        "height": packed.height,
        "has_tilted": packed.has_tilted,
        "children": [list(child) for child in packed.children],
        "rect_counts": list(packed.rect_counts),
        "stages": stages,
    }
    return tensors, {"cascade": json.dumps(meta)}


def _read_cache(name: str, device: str) -> "_Packed | None":
    """The packed cascade read straight from its file, or None to parse the XML."""
    try:
        from safetensors import safe_open
    except ImportError:
        return None
    path = _cache_file(name)
    if not path.is_file():
        return None
    try:
        with safe_open(str(path), framework="pt", device=device) as handle:
            meta = json.loads((handle.metadata() or {}).get("cascade", "{}"))
            if meta.get("version") != CACHE_VERSION:
                return None
            tensors = {key: handle.get_tensor(key) for key in handle.keys()}
        return _Packed.from_cache(tensors, meta, torch.device(device))
    except Exception:
        logger.warning(
            "the packed cascade %s could not be read, so its XML was parsed instead. "
            "Rebuild the packed files to stop this.",
            path.name,
        )
        return None


class _Packed:
    """One cascade's features and stages as tensors on one device."""

    @classmethod
    def from_cache(cls, tensors: dict, meta: dict, device) -> "_Packed":
        """Rebuild from a packed file, without reading the XML."""
        self = cls.__new__(cls)
        self.device = device
        self.name = meta["name"]
        self.kind = meta["kind"]
        self.width = meta["width"]
        self.height = meta["height"]
        self.has_tilted = meta["has_tilted"]
        self.children = tuple(tuple(child) for child in meta["children"])
        self.rect_counts = list(meta["rect_counts"])
        self._offsets = {}
        self._lattice = {}
        self.corners = tensors["corners"]
        self.weights = tensors["weights"]
        self.first = tensors["first"]
        self.turned = tensors["turned"]
        self.stages = []
        cursors = {}
        for plain in meta["stages"]:
            stage = {k: v for k, v in plain.items() if k != "lengths"}
            for key, length in plain["lengths"].items():
                start = cursors.get(key, 0)
                stage[key] = tensors[f"stage.{key}"][start : start + length]
                cursors[key] = start + length
            self.stages.append(stage)
        return self

    def __init__(self, cascade: Cascade, device):
        self.device = device
        self.name = cascade.name
        self.kind = cascade.kind
        self.width = cascade.width
        self.height = cascade.height
        self.children = cascade.children
        self.has_tilted = cascade.has_tilted
        self._offsets: dict = {}
        self._lattice: dict = {}
        count = len(cascade.rects)
        corners = torch.zeros(count, MAX_RECTS, 8, dtype=torch.int64)
        weights = torch.zeros(count, MAX_RECTS, dtype=torch.float32)
        first = torch.zeros(count, 4, dtype=torch.int64)
        for index, rows in enumerate(cascade.rects):
            first[index] = torch.tensor(rows[0][:4], dtype=torch.int64)
            for slot, (x, y, w, h, weight) in enumerate(rows):
                corners[index, slot] = torch.tensor(
                    _corners(x, y, w, h, cascade.tilted[index]), dtype=torch.int64
                )
                weights[index, slot] = weight
        self.corners = corners.to(device)
        self.weights = weights.to(device)
        self.first = first.to(device)
        self.turned = torch.tensor(cascade.tilted, dtype=torch.int64).to(device)
        self.rect_counts = [len(rows) for rows in cascade.rects]
        self.stages = [self._pack(stage) for stage in cascade.stages]

    def _pack(self, stage: Stage) -> dict:
        """Node arrays, leaves and the feature list of one stage."""
        device = self.device
        features = [node[2] for node in stage.nodes]
        # Features of two rectangles come first, so the third gather covers a slice.
        order = sorted(set(features), key=lambda index: (self.rect_counts[index] > 2, index))
        row_of = {feature: row for row, feature in enumerate(order)}
        packed = {
            "threshold": float(stage.threshold),
            "left": _ints([node[0] for node in stage.nodes], device),
            "right": _ints([node[1] for node in stage.nodes], device),
            "leaves": torch.tensor(stage.leaves, dtype=torch.float32, device=device),
            "roots": _ints(stage.roots, device),
            "leaf_offsets": _ints(stage.leaf_offsets, device),
            "features": _ints(order, device),
            "rows": _ints([row_of[feature] for feature in features], device),
            "split": sum(1 for feature in order if self.rect_counts[feature] <= 2),
            "depth": _depth(stage),
        }
        if self.kind == "lbp":
            packed["lut"] = _subset_table([node[3] for node in stage.nodes], device)
        else:
            packed["cut"] = torch.tensor(
                [node[3] for node in stage.nodes], dtype=torch.float32, device=device
            )
        return packed

    def offsets(self, stride: int, turned_base: int) -> list[torch.Tensor]:
        """Buffer offsets of every feature rectangle's four corners.

        Args:
            stride: Elements per row of the integral buffer.
            turned_base: Where the turned integral starts in the same buffer.

        Returns:
            Four ``(features, MAX_RECTS)`` tensors, read as ``a - b - c + d``.
        """
        key = (stride, turned_base)
        if key not in self._offsets:
            flat = self.corners[..., 1::2] * stride + self.corners[..., 0::2]
            if turned_base:
                flat = flat + (self.turned * turned_base)[:, None, None]
            self._offsets[key] = [flat[..., corner] for corner in range(4)]
        return self._offsets[key]

    def lattice(self, stride: int) -> torch.Tensor:
        """``(features, 16)`` buffer offsets of each LBP feature's point lattice."""
        if stride not in self._lattice:
            x, y, w, h = (self.first[:, axis] for axis in range(4))
            columns = torch.stack([x + step * w for step in range(4)], 1)
            rows = torch.stack([y + step * h for step in range(4)], 1)
            self._lattice[stride] = (
                rows[:, :, None] * stride + columns[:, None, :]
            ).reshape(-1, 16)
        return self._lattice[stride]


def _ints(values, device) -> torch.Tensor:
    """A 1-D index tensor."""
    return torch.tensor(list(values), dtype=torch.int64, device=device)


def _factors(
    height: int, width: int, win_w: int, win_h: int, scale_factor: float, min_size: int = 0
):
    """Scale of each pyramid level and the level's size, largest image first.

    Args:
        height: Source height in pixels.
        width: Source width in pixels.
        win_w: Narrowest detection window any cascade will use.
        win_h: Shortest detection window any cascade will use.
        scale_factor: Ratio between one level and the next.
        min_size: Shortest side, in pixels, an object may have. 0 for no limit.

    Returns:
        ``(factors, sizes)``, where a size is ``(width, height)``. Levels longer than
        :data:`MAX_DETECT_SIDE` on a side are left out, down to the last one, except
        where ``min_size`` still needs them.
    """
    factors, sizes, factor = [], [], 1.0
    while True:
        level = (_round(width / factor), _round(height / factor))
        if level[0] - win_w + 1 <= 0 or level[1] - win_h + 1 <= 0:
            break
        factors.append(factor)
        sizes.append(level)
        factor *= scale_factor

    start = 0
    while start < len(factors) - 1 and max(sizes[start]) > MAX_DETECT_SIDE:
        window = min(_round(win_w * factors[start]), _round(win_h * factors[start]))
        if min_size and window >= min_size:
            break
        start += 1
    if start:
        logger.debug(
            "skipped %d level(s) longer than %d pixels; smallest window is now %d pixels",
            start,
            MAX_DETECT_SIDE,
            min(_round(win_w * factors[start]), _round(win_h * factors[start])),
        )
    return factors[start:], sizes[start:]


def _integral(level: torch.Tensor) -> torch.Tensor:
    """Running sum down then across, as ``(height, width)``."""
    return torch.cumsum(torch.cumsum(level, 0, dtype=torch.int32), 1, dtype=torch.int32)


def _turned_integral(level: torch.Tensor) -> torch.Tensor:
    """Running sum over the 45 degree triangle above each point, as ``(h + 1, w + 1)``."""
    height, width = level.shape
    device = level.device
    prefix = torch.zeros(height, width + 1, dtype=torch.int32, device=device)
    prefix[:, 1:] = torch.cumsum(level, 1, dtype=torch.int32)

    rows = torch.arange(height, device=device)
    diagonals = torch.arange(width + height + 1, device=device)
    picked = rows[None, :].expand(width + height + 1, height)
    right = (diagonals[:, None] - rows[None, :] - 1).clamp(0, width)
    left = (diagonals[:, None] - height + rows[None, :]).clamp(0, width)

    down = torch.zeros(width + height + 1, height + 1, dtype=torch.int32, device=device)
    up = torch.zeros(width + height + 1, height + 1, dtype=torch.int32, device=device)
    down[:, 1:] = torch.cumsum(prefix[picked, right], 1, dtype=torch.int32)
    up[:, 1:] = torch.cumsum(prefix[picked, left], 1, dtype=torch.int32)

    ys = torch.arange(height + 1, device=device)[:, None].expand(height + 1, width + 1)
    xs = torch.arange(width + 1, device=device)[None, :].expand(height + 1, width + 1)
    return down[xs + ys, ys] - up[xs - ys + height, ys]


class _Pyramid:
    """Integral images of one picture at every scale, packed into one buffer."""

    def __init__(
        self, gray, device, scale_factor: float, win_w: int, win_h: int, min_size: int = 0
    ):
        self.device = device
        self.factors, self.sizes = _factors(
            gray.shape[0], gray.shape[1], win_w, win_h, scale_factor, min_size
        )
        self.row_offsets, self.rows, self.stride = [], 0, 0
        self.sum = self.sqsum = self._combined = None
        if not self.factors:
            return

        self.stride = max(size[0] for size in self.sizes) + 1
        for _, level_height in self.sizes:
            self.row_offsets.append(self.rows)
            self.rows += level_height + 1
        self.source = gray.to(device=device, dtype=torch.float32)[None, None]

        sums = torch.zeros(self.rows, self.stride, dtype=torch.int32, device=device)
        squares = torch.zeros(self.rows, self.stride, dtype=torch.int32, device=device)
        for index, (level_width, level_height) in enumerate(self.sizes):
            level = self._level(index).to(torch.int32)
            top = self.row_offsets[index]
            sums[top + 1 : top + level_height + 1, 1 : level_width + 1] = _integral(level)
            squares[top + 1 : top + level_height + 1, 1 : level_width + 1] = _integral(
                level * level
            )
        self.sum = sums.reshape(-1)
        self.sqsum = squares.reshape(-1)

    def _level(self, index: int) -> torch.Tensor:
        """The picture resized to one level, as rounded 8-bit values."""
        level_width, level_height = self.sizes[index]
        if self.factors[index] == 1.0:
            return self.source[0, 0]
        resized = F.interpolate(
            self.source, size=(level_height, level_width), mode="bilinear", align_corners=False
        )
        return resized[0, 0].round().clamp(0, 255)

    def _turned(self) -> torch.Tensor:
        """The 45 degree integral of every level, laid out like the upright one."""
        buffer = torch.zeros(self.rows, self.stride, dtype=torch.int32, device=self.device)
        for index, (level_width, level_height) in enumerate(self.sizes):
            top = self.row_offsets[index]
            buffer[top : top + level_height + 1, : level_width + 1] = _turned_integral(
                self._level(index).to(torch.int32)
            )
        return buffer.reshape(-1)

    def buffer(self, turned: bool):
        """``(values, base offset of the turned half)`` for one cascade's gathers."""
        if not turned:
            return self.sum, 0
        if self._combined is None:
            self._combined = torch.cat([self.sum, self._turned()])
            self.sum = self._combined[: self.rows * self.stride]
        return self._combined, self.rows * self.stride


class _Layout:
    """Where every window of every level sits in one flat run."""

    def __init__(self, pyramid: _Pyramid, levels, win_w: int, win_h: int, device):
        steps, columns, tops, widths, heights, scales = [], [], [], [], [], []
        cumulative = [0]
        for index in levels:
            factor = pyramid.factors[index]
            level_width, level_height = pyramid.sizes[index]
            step = 1 if factor > 2.0 else 2
            across = (level_width - win_w) // step + 1
            down = (level_height - win_h) // step + 1
            steps.append(step)
            columns.append(across)
            tops.append(pyramid.row_offsets[index])
            widths.append(_round(win_w * factor))
            heights.append(_round(win_h * factor))
            scales.append(factor)
            cumulative.append(cumulative[-1] + across * down)
        self.total = cumulative[-1]
        self.cumulative = _ints(cumulative, device)
        self.steps = _ints(steps, device)
        self.columns = _ints(columns, device)
        self.tops = _ints(tops, device)
        self.widths = _ints(widths, device)
        self.heights = _ints(heights, device)
        self.scales = torch.tensor(scales, dtype=torch.float64, device=device)


def _levels(pyramid: _Pyramid, win_w: int, win_h: int, min_size: int, max_size: int):
    """Levels of the pyramid one cascade searches, in order."""
    chosen = []
    for index, factor in enumerate(pyramid.factors):
        level_width, level_height = pyramid.sizes[index]
        if level_width - win_w + 1 <= 0 or level_height - win_h + 1 <= 0:
            break
        window = (_round(win_w * factor), _round(win_h * factor))
        if max_size and max(window) > max_size:
            break
        if min_size and min(window) < min_size:
            continue
        chosen.append(index)
    return chosen


def _run(packed: _Packed, pyramid: _Pyramid, min_size: int, max_size: int) -> list[tuple]:
    """Every window one cascade accepts, before grouping.

    Args:
        packed: The cascade's tensors.
        pyramid: Integral images of the picture.
        min_size: Shortest side, in pixels, an object may have. 0 for no limit.
        max_size: Longest side, in pixels, an object may have. 0 for no limit.

    Returns:
        ``(x, y, width, height)`` rectangles, in the order the windows were searched.
    """
    if pyramid.sum is None:
        return []
    levels = _levels(pyramid, packed.width, packed.height, min_size, max_size)
    if not levels:
        return []

    layout = _Layout(pyramid, levels, packed.width, packed.height, pyramid.device)
    values, turned_base = pyramid.buffer(packed.has_tilted)
    offsets = packed.offsets(pyramid.stride, turned_base)
    lattice = packed.lattice(pyramid.stride) if packed.kind == "lbp" else None

    found = []
    for start in range(0, layout.total, WINDOW_BLOCK):
        found += _search(
            packed, pyramid, layout, values, offsets, lattice, start, WINDOW_BLOCK
        )
    return found


def _search(
    packed: _Packed,
    pyramid: _Pyramid,
    layout: _Layout,
    values: torch.Tensor,
    offsets,
    lattice,
    start: int,
    block: int,
) -> list[tuple]:
    """Run the whole cascade over one block of windows."""
    device = pyramid.device
    stop = min(start + block, layout.total)
    window = torch.arange(start, stop, dtype=torch.int64, device=device)
    slot = torch.searchsorted(layout.cumulative, window, right=True) - 1
    local = window - layout.cumulative[slot]
    step = layout.steps[slot]
    x = (local % layout.columns[slot]) * step
    y = (local // layout.columns[slot]) * step
    base = (layout.tops[slot] + y) * pyramid.stride + x

    if packed.kind == "lbp":
        alive = torch.arange(base.numel(), dtype=torch.int64, device=device)
        norm = None
    else:
        spread, norm = _variance(pyramid, base, packed)
        alive = spread.nonzero(as_tuple=True)[0]

    accepted, pending = [], [(0, alive)]
    while pending:
        index, alive = pending.pop()
        alive = _stage(packed, index, values, base, norm, alive, offsets, lattice)
        if alive.numel() == 0:
            continue
        children = packed.children[index]
        if not children:
            accepted.append(alive)
            continue
        for child in reversed(children):
            pending.append((child, alive))

    if not accepted:
        return []
    alive = accepted[0] if len(accepted) == 1 else torch.unique(torch.cat(accepted))
    scale = layout.scales[slot[alive]]
    rects = torch.stack(
        [
            (x[alive].to(torch.float64) * scale).round().to(torch.int64),
            (y[alive].to(torch.float64) * scale).round().to(torch.int64),
            layout.widths[slot[alive]],
            layout.heights[slot[alive]],
        ],
        1,
    )
    return [tuple(rect) for rect in rects.cpu().tolist()]


def _variance(pyramid: _Pyramid, base: torch.Tensor, packed: "_Packed"):
    """Which windows have any contrast at all, and one over each one's spread."""
    stride = pyramid.stride
    width, height = packed.width, packed.height
    area = (width - 2) * (height - 2)
    corners = (
        (height - 1) * stride + width - 1,
        stride + width - 1,
        (height - 1) * stride + 1,
        stride + 1,
    )
    total = (
        pyramid.sum[base + corners[0]]
        - pyramid.sum[base + corners[1]]
        - pyramid.sum[base + corners[2]]
        + pyramid.sum[base + corners[3]]
    ).to(torch.int64)
    squares = (
        pyramid.sqsum[base + corners[0]]
        - pyramid.sqsum[base + corners[1]]
        - pyramid.sqsum[base + corners[2]]
        + pyramid.sqsum[base + corners[3]]
    ).to(torch.int64)
    spread = area * squares - total * total
    inside = spread > 0
    norm = torch.zeros(base.shape, dtype=torch.float32, device=base.device)
    norm[inside] = (1.0 / spread[inside].to(torch.float64).sqrt()).to(torch.float32)
    return inside, norm


def _stage(
    packed: _Packed,
    index: int,
    values: torch.Tensor,
    base: torch.Tensor,
    norm,
    alive: torch.Tensor,
    offsets,
    lattice,
) -> torch.Tensor:
    """Which of the surviving windows pass one more stage."""
    stage = packed.stages[index]
    chunk = max(MIN_BLOCK, GATHER_BUDGET // max(stage["features"].numel(), 1))
    if alive.numel() <= chunk:
        return _stage_pass(packed, stage, values, base, norm, alive, offsets, lattice)
    return torch.cat(
        [
            _stage_pass(
                packed,
                stage,
                values,
                base,
                norm,
                alive[start : start + chunk],
                offsets,
                lattice,
            )
            for start in range(0, alive.numel(), chunk)
        ]
    )


def _stage_pass(
    packed: _Packed,
    stage: dict,
    values: torch.Tensor,
    base: torch.Tensor,
    norm,
    alive: torch.Tensor,
    offsets,
    lattice,
) -> torch.Tensor:
    """Which windows of one chunk of the survivors pass the stage."""
    windows = base[alive]
    categorical = packed.kind == "lbp"
    if categorical:
        read = _codes(values, windows, lattice[stage["features"]])
    else:
        read = _features(values, windows, stage, offsets, packed.weights)
        read *= norm[alive][None, :]
    branch = _walk(stage, read, categorical)
    score = stage["leaves"][stage["leaf_offsets"][:, None] - branch].sum(0)
    return alive[score >= stage["threshold"]]


def _features(
    values: torch.Tensor, windows: torch.Tensor, stage: dict, offsets, weights
) -> torch.Tensor:
    """Weighted rectangle sums of one stage's features, as ``(features, windows)``."""
    features = stage["features"]
    split = stage["split"]
    picked = [corner[features] for corner in offsets]
    weight = weights[features]
    total = None
    for slot in range(MAX_RECTS):
        low = 0 if slot < 2 else split
        if low >= features.numel():
            continue
        term = _sums(values, windows, [corner[low:, slot : slot + 1] for corner in picked])
        term = term.to(torch.float32) * weight[low:, slot : slot + 1]
        if total is None:
            total = term
        elif slot < 2:
            total += term
        else:
            total[low:] += term
    return total


def _sums(values: torch.Tensor, windows: torch.Tensor, corners) -> torch.Tensor:
    """One rectangle's sum at every window, from four corners of the integral."""
    total = values[windows + corners[0]]
    total -= values[windows + corners[1]]
    total -= values[windows + corners[2]]
    total += values[windows + corners[3]]
    return total


def _codes(values: torch.Tensor, windows: torch.Tensor, lattice) -> torch.Tensor:
    """LBP category of each feature at each window, as ``(features, windows)``."""

    def cell(row: int, column: int) -> torch.Tensor:
        top = row * 4 + column
        bottom = top + 4
        return _sums(
            values,
            windows,
            (
                lattice[:, bottom + 1 : bottom + 2],
                lattice[:, bottom : bottom + 1],
                lattice[:, top + 1 : top + 2],
                lattice[:, top : top + 1],
            ),
        )

    centre = cell(1, 1)
    code = torch.zeros_like(centre)
    for row, column, bit in LBP_NEIGHBOURS:
        code += (cell(row, column) >= centre).to(code.dtype) * bit
    return code.to(torch.int64)


def _walk(stage: dict, read: torch.Tensor, categorical: bool) -> torch.Tensor:
    """Leaf each weak classifier reaches, as ``(classifiers, windows)``."""
    count = stage["roots"].numel()
    if stage["depth"] == 1:
        nodes = torch.arange(count, dtype=torch.int64, device=read.device)[:, None]
        return _branch(stage, read[stage["rows"]], nodes, categorical)

    branch = torch.zeros(count, read.shape[1], dtype=torch.int64, device=read.device)
    live = torch.ones_like(branch, dtype=torch.bool)
    for _ in range(stage["depth"]):
        nodes = stage["roots"][:, None] + branch.clamp(min=0)
        picked = torch.gather(read, 0, stage["rows"][nodes])
        branch = torch.where(live, _branch(stage, picked, nodes, categorical), branch)
        live = branch > 0
    return branch


def _branch(stage: dict, picked: torch.Tensor, nodes: torch.Tensor, categorical: bool):
    """Where each node sends the value it read."""
    if categorical:
        return torch.where(
            stage["lut"][nodes, picked], stage["left"][nodes], stage["right"][nodes]
        )
    return torch.where(
        picked >= stage["cut"][nodes], stage["right"][nodes], stage["left"][nodes]
    )


def _cluster(rects, eps: float) -> list[int]:
    """Which cluster each rectangle belongs to, numbered in first-appearance order."""
    parent = list(range(len(rects)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for first, second in _neighbours(rects, eps):
        left, right = root(first), root(second)
        if left != right:
            parent[right] = left

    labels, numbering = [], {}
    for index in range(len(rects)):
        labels.append(numbering.setdefault(root(index), len(numbering)))
    return labels


def _neighbours(rects, eps: float):
    """Index pairs of rectangles close enough in place and size to be one object."""
    boxes = torch.tensor(rects, dtype=torch.float64)
    x, y, w, h = (boxes[:, axis] for axis in range(4))
    right, bottom = x + w, y + h
    pairs = []
    for start in range(0, len(rects), PAIR_BLOCK):
        rows = slice(start, min(start + PAIR_BLOCK, len(rects)))
        margin = (
            eps
            * (torch.minimum(w[rows, None], w[None, :]) + torch.minimum(h[rows, None], h[None, :]))
            * 0.5
        )
        close = (x[rows, None] - x[None, :]).abs() <= margin
        close &= (y[rows, None] - y[None, :]).abs() <= margin
        close &= (right[rows, None] - right[None, :]).abs() <= margin
        close &= (bottom[rows, None] - bottom[None, :]).abs() <= margin
        first, second = torch.nonzero(close, as_tuple=True)
        pairs += zip((first + start).tolist(), second.tolist())
    return pairs


def _inside(rect, other, eps: float) -> bool:
    """Whether one rectangle sits inside another, allowing ``eps`` of slack."""
    slack_x, slack_y = _round(other[2] * eps), _round(other[3] * eps)
    return (
        rect[0] >= other[0] - slack_x
        and rect[1] >= other[1] - slack_y
        and rect[0] + rect[2] <= other[0] + other[2] + slack_x
        and rect[1] + rect[3] <= other[1] + other[3] + slack_y
    )

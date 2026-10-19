"""Skin masks: the bare skin of every person in a picture, eyes, brows and mouths cut out.

Frames are ``(height, width, channels)`` on a 0 to 1 scale and masks ``(height, width)``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional

from . import guided, morphology

__all__ = ["AREAS", "EYES", "MOUTHS", "Found", "find_by_colour", "find_by_parts"]

#: Which skin the mask keeps.
AREAS = ("all skin", "face", "body")

#: What is cut out around each eye.
EYES = ("remove with brows", "remove", "keep")

#: What is cut out of each mouth.
MOUTHS = ("remove inside", "remove lips", "keep")

#: Clothing-label classes, grouped: skin, clothing that may be bare skin, clothing, background.
GROUPS = {
    "skin": ("Face", "Left-arm", "Right-arm", "Left-leg", "Right-leg"),
    "loose": ("Upper-clothes", "Dress"),
    "cloth": (
        "Hat", "Hair", "Sunglasses", "Skirt", "Pants", "Belt", "Left-shoe", "Right-shoe", "Bag",
        "Scarf",
    ),
    "background": ("Background",),
}

#: Body-part classes, grouped: bare skin, lips, teeth with tongue, garments that may be bare
#: skin, what is never skin, and the background. Every class falls in one group.
PART_GROUPS = {
    "skin": (
        "Face_Neck", "Left_Foot", "Left_Hand", "Left_Lower_Arm", "Left_Lower_Leg",
        "Left_Upper_Arm", "Left_Upper_Leg", "Right_Foot", "Right_Hand", "Right_Lower_Arm",
        "Right_Lower_Leg", "Right_Upper_Arm", "Right_Upper_Leg", "Torso",
    ),
    "lips": ("Lower_Lip", "Upper_Lip"),
    "inside": ("Lower_Teeth", "Upper_Teeth", "Tongue"),
    "loose": ("Apparel", "Upper_Clothing", "Lower_Clothing"),
    "cloth": ("Eyeglass", "Hair", "Left_Shoe", "Left_Sock", "Right_Shoe", "Right_Sock"),
    "background": ("Background",),
}

#: People a body-part network reads in one batch.
PART_BATCH = 4

#: Longest side the whole picture is labelled at, and each person's crop.
FRAME_SIDE = 1024
CROP_SIDE = 768

#: A person whose padded box spans more than this share of the picture is not cropped.
CROP_SHARE = 0.6

#: Padding around a person's box, as a share of its longer side.
CROP_PADDING = 0.15
COLOUR_PADDING = 0.05

#: Weight a crop keeps at the far edge of its padding, where its reads are of a neighbour.
NEIGHBOUR_WEIGHT = 0.05

#: Weight a person's crop carries against the whole-picture labels, and its feathered rim.
CROP_WEIGHT = 4.0
CROP_RIM = 0.1

#: Normalisation the clothing-label model was trained with.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

#: Label confidence a pixel needs to be sampled as sure skin, sure clothing or sure background.
SURE_SKIN = 0.9
SURE_CLOTH = 0.9
SURE_BACKGROUND = 0.97
LOOSE_SAMPLE = 0.6

#: Sure-skin pixels a colour model needs, and samples it clusters at most.
MIN_SAMPLES = 150
MAX_SAMPLES = 30000

#: Skin colour components, clothing and background components, and clustering rounds.
SKIN_CLUSTERS = 2
OTHER_CLUSTERS = 6
ROUNDS = 10

#: Robust spread a sure-skin sample may sit at, squared, before it is dropped as an outlier.
OUTLIER = 9.0

#: Covariance inflation of the skin components.
INFLATE = 1.2

#: Squared distance past which loose-clothing colour is sampled as clothing.
FAR = 25.0

#: Squared distance past which a colour is not skin at all, and the softness of that cut.
CLOSED_AT = 25.0
CLOSED_SOFTNESS = 3.0

#: Weight of log lightness beside the two chroma-over-lightness features.
LIGHTNESS_WEIGHT = 0.1

#: Lightness below which colour is unreliable, and the span it fades in over.
DARK = 0.06
DARK_SPAN = 0.12

#: Chroma spread below which a picture is treated as colourless.
COLOURLESS = 0.01

#: Detector confidence a person needs to be counted.
PERSON_THRESHOLD = 0.4

#: Share of a box that must lie inside another for that other to count as holding it.
GROUP_SHARE = 0.8

#: Landmark confidence a feature needs to be cut out.
LANDMARK_THRESHOLD = 0.3

#: Enclosed gaps filled as skin: largest share of the skin, colour step across the border for a
#: seamless gap, and for a highlight.
GAP_SHARE = 0.03
SEAMLESS_STEP = 0.03
HIGHLIGHT_STEP = 0.1

#: The 133-point whole-body layout: where the 68 face points start, and the features in it.
FACE = 23
JAW = tuple(range(FACE, FACE + 17))
BROWS = (tuple(range(FACE + 17, FACE + 22)), tuple(range(FACE + 22, FACE + 27)))
EYE_RINGS = (tuple(range(FACE + 36, FACE + 42)), tuple(range(FACE + 42, FACE + 48)))
LIPS = tuple(range(FACE + 48, FACE + 60))
MOUTH = tuple(range(FACE + 60, FACE + 68))

#: Eye ring scale across and down, the margin around it and the brow half-width, as shares
#: of the eye's width.
EYE_SCALE = (1.1, 1.3)
EYE_MARGIN = 0.06
BROW_WIDTH = 0.10

#: Forehead height above the brows, as a share of the brow-to-chin height.
FOREHEAD = 0.6


@dataclass(frozen=True)
class Found:
    """One frame's skin mask and what was found.

    Attributes:
        mask: ``(height, width)`` float tensor on a 0 to 1 scale.
        people: People the detector found.
        faces: Faces whose features were cut out.
    """

    mask: torch.Tensor
    people: int
    faces: int


def find_by_parts(
    frame: torch.Tensor,
    network,
    pair,
    areas: str = AREAS[0],
    eyes: str = EYES[0],
    mouth: str = MOUTHS[0],
    tolerance: float = 1.0,
    threshold: float = PERSON_THRESHOLD,
) -> Found:
    """Find the bare skin in one frame from a body-part network and each person's skin colour.

    Args:
        frame: ``(height, width, channels)`` on a 0 to 1 scale.
        network: The body-part ``Backend`` from :func:`modules.model.sapiens2.load`.
        pair: The whole-body pose ``Pair`` from :func:`modules.model.pose.load`.
        areas: One of :data:`AREAS`.
        eyes: One of :data:`EYES`.
        mouth: One of :data:`MOUTHS`.
        tolerance: Above 1 more colours count as skin, below 1 fewer.
        threshold: Detector confidence a person needs.

    Returns:
        A :class:`Found`.
    """
    picture = frame[..., :3].float().clamp(0.0, 1.0)
    boxes, points, scores = _people(pair, picture, threshold)
    # Loaded after the detector, so making room for it cannot move this network off the GPU.
    device = network.load()
    rgb = picture.to(device)
    boxes = boxes.to(device) if boxes is not None else torch.zeros((0, 4), device=device)
    skin, lips, inside, loose, cloth, background = _part_labels(network.model, rgb, boxes)
    # The mouth's own classes count as skin or as never skin, as the mouth option asks.
    kept_lips = mouth != "remove lips"
    kept_inside = mouth == "keep"
    skin = skin + lips * kept_lips + inside * kept_inside
    cloth = cloth + lips * (not kept_lips) + inside * (not kept_inside)
    labels = (skin, loose, cloth, background)
    decided, band = _decide(rgb, labels, boxes, tolerance, edges_by_colour=False)
    return _finish(decided, rgb, boxes, points, scores, areas, eyes, "keep", band)


def find_by_colour(
    frame: torch.Tensor,
    parts,
    pair,
    areas: str = AREAS[0],
    eyes: str = EYES[0],
    mouth: str = MOUTHS[0],
    tolerance: float = 1.0,
    threshold: float = PERSON_THRESHOLD,
) -> Found:
    """Find the bare skin in one frame from clothing labels and each person's skin colour.

    Args:
        frame: ``(height, width, channels)`` on a 0 to 1 scale.
        parts: The clothing-label ``Backend`` from :func:`modules.model.body_parts.load`.
        pair: The whole-body pose ``Pair`` from :func:`modules.model.pose.load`.
        areas: One of :data:`AREAS`.
        eyes: One of :data:`EYES`.
        mouth: One of :data:`MOUTHS`.
        tolerance: Above 1 more colours count as skin, below 1 fewer.
        threshold: Detector confidence a person needs.

    Returns:
        A :class:`Found`.
    """
    picture = frame[..., :3].float().clamp(0.0, 1.0)
    boxes, points, scores = _people(pair, picture, threshold)
    # Loaded after the detector, so making room for it cannot move this network off the GPU.
    device = parts.load()
    rgb = picture.to(device)
    boxes = boxes.to(device) if boxes is not None else torch.zeros((0, 4), device=device)
    labels = _labels(parts.model, rgb, boxes)
    decided, band = _decide(rgb, labels, boxes, tolerance)
    return _finish(decided, rgb, boxes, points, scores, areas, eyes, mouth, band)


def _decide(rgb, labels, boxes, tolerance: float, edges_by_colour: bool = True):
    """Skin from the labels, garments read against each person's skin colour.

    Args:
        rgb: ``(h, w, 3)`` on a 0 to 1 scale.
        labels: ``(skin, loose, cloth, background)`` chances, each ``(h, w)``.
        boxes: ``(count, 4)`` people as ``x, y, width, height``.
        tolerance: Above 1 more colours count as skin, below 1 fewer.
        edges_by_colour: Decide a band along every edge by each pixel's own colour. Off, the
            labels' own edges are kept.

    Returns:
        ``(decided, band)``: the ``(h, w)`` skin before the cut-outs, and the edge band's width.
    """
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    device = rgb.device
    skin, loose, cloth, background = labels

    features, light, colourless = _features(rgb)
    if colourless:
        lenient = strict = torch.full_like(light, 0.5)
    else:
        lenient, strict = _colour(features, labels, boxes, float(tolerance))
        reliable = ((light - DARK) / DARK_SPAN).clamp(0.0, 1.0)
        lenient = 0.5 + (lenient - 0.5) * reliable
        strict = 0.5 + (strict - 0.5) * reliable

    short = min(height, width)
    band = max(2, round(0.6 * max(height, width) / 256))
    regional = _guide(lenient, rgb, max(4, short // 80), 1e-2)
    soft = (skin + loose * regional).clamp(0.0, 1.0)
    if len(boxes):
        soft = soft * _covered(boxes, height, width, COLOUR_PADDING, device)
    if not edges_by_colour:
        return soft, band

    hard = (soft > 0.5).float()
    # Specks go and pinholes close before the edge is decided.
    hard = _grow(_shrink(hard, band // 2), band // 2)
    hard = _shrink(_grow(hard, band), band)
    hard = _fill_gaps(hard, rgb, labels, max(2, band // 2))

    # Inside the band around the edge each pixel's own colour decides; elsewhere the region does.
    inner, outer = _shrink(hard, band), _grow(hard, band)
    in_band = (outer - inner) > 0.5
    edge = soft if colourless else strict * (2.0 * soft).clamp(0.0, 1.0)
    return torch.where(in_band, edge, hard), band


def _finish(decided, rgb, boxes, points, scores, areas, eyes, mouth, band) -> Found:
    """Cut out the eyes and mouths, keep the chosen areas, and settle the edge on the picture."""
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    device = rgb.device
    faces = 0
    if points is not None:
        points = points.to(device=device, dtype=torch.float32)
        scores = scores.to(device=device, dtype=torch.float32)
        removed, faces = _cut_outs(points, scores, height, width, eyes, mouth, device)
        decided = decided * (1.0 - removed)
        if areas != AREAS[0]:
            face = _faces(points, scores, height, width, device)
            decided = decided * (face if areas == "face" else 1.0 - face)
    elif areas == "face":
        decided = torch.zeros_like(decided)
    mask = _guide(decided, rgb, max(1, band // 2), 1e-4)
    return Found(mask=mask.clamp(0.0, 1.0), people=int(len(boxes)), faces=faces)


def _part_labels(model, rgb, boxes):
    """Every group's chance over the frame, each person read at the network's own size.

    Returns:
        One ``(h, w)`` plane per :data:`PART_GROUPS` group, in its order. With nobody found,
        the whole frame is read; outside every person, the background is certain.
    """
    from ..model import sapiens2

    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    read_tall, read_wide = sapiens2.READ_SIZE
    planes = rgb.permute(2, 0, 1).unsqueeze(0)
    people = [(box, _padded(box, height, width, CROP_PADDING)) for box in boxes]
    people = [(box, one) for box, one in people if one[1] - one[0] >= 16 and one[3] - one[2] >= 16]
    if not people:
        return tuple(_part_groups(model, [planes], whole=True)[0])
    total = torch.zeros((len(PART_GROUPS), height, width), device=rgb.device)
    weight = torch.zeros((1, height, width), device=rgb.device)
    for start in range(0, len(people), PART_BATCH):
        chunk = people[start:start + PART_BATCH]
        crops = [planes[..., top:bottom, left:right] for _, (top, bottom, left, right) in chunk]
        for (box, (top, bottom, left, right)), groups in zip(chunk, _part_groups(model, crops)):
            # A crop is trusted on its own person; a neighbour at its edge is read badly.
            closeness = min(read_tall / (bottom - top), read_wide / (right - left))
            own = _ownership(box, (top, bottom, left, right), rgb.device)
            share = _rim(bottom - top, right - left, rgb.device).clamp_min(1e-3) * own * closeness
            total[:, top:bottom, left:right] += groups * share
            weight[:, top:bottom, left:right] += share
    outside = torch.zeros_like(total)
    outside[list(PART_GROUPS).index("background")] = 1.0
    return tuple(torch.where(weight > 0, total / weight.clamp_min(1e-6), outside))


def _ownership(box, window, device):
    """A weight over a crop window: 1 inside the person's own box, falling outside it.

    Args:
        box: The person's ``x, y, width, height``.
        window: The crop's ``(top, bottom, left, right)``.
        device: Where to build the plane.

    Returns:
        A ``(h, w)`` plane on the window, :data:`NEIGHBOUR_WEIGHT` at its far edge.
    """
    top, bottom, left, right = window
    x, y, w, h = (float(value) for value in box)
    reach = max(1.0, CROP_PADDING * max(w, h))
    rows = torch.arange(top, bottom, device=device, dtype=torch.float32) + 0.5
    columns = torch.arange(left, right, device=device, dtype=torch.float32) + 0.5
    over_y = torch.maximum(y - rows, rows - (y + h)).clamp_min(0.0) / reach
    over_x = torch.maximum(x - columns, columns - (x + w)).clamp_min(0.0) / reach
    beyond = torch.maximum(over_y[:, None], over_x[None, :])
    return (1.0 - beyond).clamp(min=NEIGHBOUR_WEIGHT, max=1.0)


def _rim(tall: int, wide: int, device):
    """A weight that is 1 inside a window and falls to 0 over its outer tenth."""
    rim_y = torch.linspace(0.0, 1.0, tall, device=device)
    rim_x = torch.linspace(0.0, 1.0, wide, device=device)
    return (torch.minimum(rim_y, 1.0 - rim_y).clamp(max=CROP_RIM) / CROP_RIM)[:, None] * (
        torch.minimum(rim_x, 1.0 - rim_x).clamp(max=CROP_RIM) / CROP_RIM
    )[None, :]


def _part_groups(model, pictures, whole: bool = False):
    """Skin, lip and mouth chances for each picture, at that picture's own size.

    Args:
        model: The body-part network.
        pictures: ``(1, 3, h, w)`` pictures on a 0 to 1 scale.
        whole: Read one picture at its own shape with its long side at the network's height,
            rather than letterboxed into the network's frame.

    Returns:
        A list of ``(groups, h, w)`` tensors, one plane per :data:`PART_GROUPS` group.
    """
    from ..model import sapiens2

    tall, wide = sapiens2.READ_SIZE
    step = sapiens2.PATCH
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    mean = torch.tensor(sapiens2.MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(sapiens2.STD, device=device).view(1, 3, 1, 1)
    index = {name: number for number, name in enumerate(sapiens2.CLASSES)}
    reads, sizes = [], []
    for picture in pictures:
        height, width = int(picture.shape[-2]), int(picture.shape[-1])
        if whole:
            scale = tall / max(height, width)
            size = (max(step, round(height * scale / step) * step),
                    max(step, round(width * scale / step) * step))
            frame = size
        else:
            scale = min(tall / height, wide / width)
            size = (max(1, round(height * scale)), max(1, round(width * scale)))
            frame = (tall, wide)
        read = functional.interpolate(
            picture.to(device), size=size, mode="bilinear", antialias=True, align_corners=False
        )
        # The letterbox around a crop is filled with the training mean, which normalises to 0.
        canvas = torch.zeros((1, 3, *frame), device=device)
        canvas[..., : size[0], : size[1]] = (read - mean) / std
        reads.append(canvas)
        sizes.append((size, (height, width)))
    with torch.no_grad():
        chances = model(torch.cat(reads).to(dtype)).float().softmax(1)
    answers = []
    for one, (size, original) in zip(chances, sizes):
        grouped = torch.stack([
            one[[index[name] for name in names]].sum(0) for names in PART_GROUPS.values()
        ])[:, : size[0], : size[1]]
        answers.append(functional.interpolate(
            grouped[None], size=original, mode="bilinear", align_corners=False
        )[0])
    return answers


def _people(pair, rgb, threshold):
    """Every person's box, joints and joint confidence, or ``(None, None, None)``."""
    import numpy as np

    from . import preprocess_models

    detector, poser = pair.detector, pair.poser
    device = detector.load()
    poser.load()
    picture = (rgb.cpu().numpy() * 255.0 + 0.5).astype(np.uint8)
    height, width = picture.shape[:2]
    wanted = preprocess_models._subject_labels(detector.model.config, pair.layout)
    boxes = preprocess_models._subjects(
        detector, picture, height, width, float(threshold), wanted, device
    )
    if boxes is None:
        return None, None, None
    boxes = _singles(boxes.float())
    points, scores = preprocess_models._joints(poser, picture, boxes, pair.experts, device)
    return boxes, points, scores


def _singles(boxes):
    """Drop every box that holds two or more of the others, which frames a group, not a person.

    Args:
        boxes: ``(count, 4)`` as ``x, y, width, height``.

    Returns:
        The boxes that frame one person each.
    """
    if len(boxes) < 3:
        return boxes
    left, top = boxes[:, 0], boxes[:, 1]
    right, bottom = left + boxes[:, 2], top + boxes[:, 3]
    overlap_w = (torch.minimum(right[:, None], right[None]) - torch.maximum(left[:, None], left[None]))
    overlap_h = (torch.minimum(bottom[:, None], bottom[None]) - torch.maximum(top[:, None], top[None]))
    overlap = overlap_w.clamp_min(0) * overlap_h.clamp_min(0)
    area = (boxes[:, 2] * boxes[:, 3]).clamp_min(1e-6)
    # Row i, column j: the share of box j that lies inside box i.
    inside = overlap / area[None]
    inside.fill_diagonal_(0.0)
    held = (inside >= GROUP_SHARE).sum(1)
    return boxes[held < 2]


def _groups(model, planes, side: int):
    """Class groups for one ``(1, 3, h, w)`` picture, ``(4, h, w)`` at the picture's size."""
    height, width = int(planes.shape[-2]), int(planes.shape[-1])
    scale = side / max(height, width)
    size = (
        max(64, round(height * scale / 32) * 32),
        max(64, round(width * scale / 32) * 32),
    )
    read = functional.interpolate(
        planes, size=size, mode="bilinear", antialias=True, align_corners=False
    )
    mean = torch.tensor(MEAN, device=planes.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=planes.device).view(1, 3, 1, 1)
    with torch.no_grad():
        chances = model(pixel_values=(read - mean) / std).logits.float().softmax(1)
    index = {name: int(number) for number, name in model.config.id2label.items()}
    grouped = torch.stack(
        [chances[:, [index[name] for name in names]].sum(1) for names in GROUPS.values()], 1
    )
    return functional.interpolate(
        grouped, size=(height, width), mode="bilinear", align_corners=False
    )[0]


def _padded(box, height: int, width: int, share: float) -> tuple[int, int, int, int]:
    """A box grown by a share of its longer side, as ``(top, bottom, left, right)``."""
    x, y, w, h = (float(value) for value in box)
    pad = share * max(w, h)
    return (
        max(0, int(y - pad)),
        min(height, int(math.ceil(y + h + pad))),
        max(0, int(x - pad)),
        min(width, int(math.ceil(x + w + pad))),
    )


def _labels(model, rgb, boxes):
    """Class groups over the whole frame, sharpened by a closer read of every small person."""
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    planes = rgb.permute(2, 0, 1).unsqueeze(0)
    total = _groups(model, planes, FRAME_SIDE)
    weight = torch.ones((1, height, width), device=rgb.device)
    for box in boxes:
        top, bottom, left, right = _padded(box, height, width, CROP_PADDING)
        tall, wide = bottom - top, right - left
        if max(tall, wide) > CROP_SHARE * max(height, width) or min(tall, wide) < 16:
            continue
        crop = _groups(model, planes[..., top:bottom, left:right], CROP_SIDE)
        rim_y = torch.linspace(0.0, 1.0, tall, device=rgb.device)
        rim_x = torch.linspace(0.0, 1.0, wide, device=rgb.device)
        rim = (torch.minimum(rim_y, 1.0 - rim_y).clamp(max=CROP_RIM) / CROP_RIM)[:, None] * (
            torch.minimum(rim_x, 1.0 - rim_x).clamp(max=CROP_RIM) / CROP_RIM
        )[None, :]
        share = CROP_WEIGHT * rim[None]
        total[:, top:bottom, left:right] += crop * share
        weight[:, top:bottom, left:right] += share
    return total / weight


def _oklab(rgb):
    """Oklab of sRGB values, channels last."""
    linear = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    to_lms = torch.tensor(
        [
            [0.4122214708, 0.5363325363, 0.0514459929],
            [0.2119034982, 0.6806995451, 0.1073969566],
            [0.0883024619, 0.2817188376, 0.6299787005],
        ],
        device=rgb.device,
    )
    to_lab = torch.tensor(
        [
            [0.2104542553, 0.7936177850, -0.0040720468],
            [1.9779984951, -2.4285922050, 0.4505937099],
            [0.0259040371, 0.7827717662, -0.8086757660],
        ],
        device=rgb.device,
    )
    return ((linear @ to_lms.T).clamp_min(0.0) ** (1.0 / 3.0)) @ to_lab.T


def _features(rgb):
    """Colour features that shading leaves alone, the lightness, and whether colour is absent.

    Returns:
        ``(features, lightness, colourless)``: ``(h, w, 3)`` log lightness weighted down and
        the two chroma axes over lightness, ``(h, w)`` lightness, and a bool.
    """
    lab = _oklab(rgb)
    light = lab[..., 0].clamp_min(0.05)
    features = torch.stack(
        [LIGHTNESS_WEIGHT * light.log(), lab[..., 1] / light, lab[..., 2] / light], -1
    )
    colourless = float(lab[..., 1:].std()) < COLOURLESS
    return features, lab[..., 0], colourless


def _gaussian(samples, inflate: float = 1.0):
    """Mean, inverse covariance and log determinant of a sample."""
    centre = samples.mean(0)
    spread = samples - centre
    count = max(1, int(samples.shape[0]) - 1)
    covariance = spread.T @ spread / count * inflate
    covariance = covariance + torch.eye(3, device=samples.device) * 1e-5
    return centre, torch.linalg.inv(covariance), torch.logdet(covariance)


def _log_likelihood(values, model):
    """Gaussian log likelihood of each value, up to a constant."""
    centre, inverse, log_det = model
    offset = values - centre
    return -0.5 * ((offset @ inverse) * offset).sum(-1) - 0.5 * log_det


def _distance(values, model):
    """Squared Mahalanobis distance of each value."""
    centre, inverse, _ = model
    offset = values - centre
    return ((offset @ inverse) * offset).sum(-1)


def _clusters(samples, count: int):
    """K-means groups of a sample, dropping any group of fewer than 40."""
    generator = torch.Generator().manual_seed(0)
    if int(samples.shape[0]) > MAX_SAMPLES:
        picked = torch.randperm(int(samples.shape[0]), generator=generator)[:MAX_SAMPLES]
        samples = samples[picked.to(samples.device)]
    count = min(count, int(samples.shape[0]))
    order = torch.randperm(int(samples.shape[0]), generator=generator)[:count]
    centres = samples[order.to(samples.device)]
    nearest = torch.zeros(int(samples.shape[0]), dtype=torch.long, device=samples.device)
    for _ in range(ROUNDS):
        nearest = torch.cdist(samples, centres).argmin(1)
        centres = torch.stack([
            samples[nearest == index].mean(0) if bool((nearest == index).any()) else centres[index]
            for index in range(count)
        ])
    return [samples[nearest == index] for index in range(count) if int((nearest == index).sum()) >= 40]


def _posterior(features, labels, rows, columns, tolerance: float):
    """Lenient and strict skin chances over one window, from that window's own sure labels.

    Returns:
        A ``(2, h, w)`` tensor, or None where the window holds too little sure skin.
    """
    skin, loose, cloth, background = (plane[rows, columns] for plane in labels)
    values = features[rows, columns]
    skin_samples = values[skin > SURE_SKIN]
    if int(skin_samples.shape[0]) < MIN_SAMPLES:
        return None
    centre = skin_samples.median(0).values
    spread = (skin_samples - centre).abs().median(0).values.clamp_min(1e-3) * 1.4826
    skin_samples = skin_samples[(((skin_samples - centre) / spread) ** 2).sum(-1) < OUTLIER]
    if int(skin_samples.shape[0]) < MIN_SAMPLES:
        return None
    skin_models = [_gaussian(group, INFLATE) for group in _clusters(skin_samples, SKIN_CLUSTERS)]

    other_samples = values[(cloth > SURE_CLOTH) | (background > SURE_BACKGROUND)]
    loose_samples = values[loose > LOOSE_SAMPLE]
    if int(loose_samples.shape[0]):
        nearest = torch.stack([_distance(loose_samples, model) for model in skin_models]).min(0).values
        other_samples = torch.cat([other_samples, loose_samples[nearest > FAR]])
    other_models = []
    if int(other_samples.shape[0]) >= MIN_SAMPLES:
        other_models = [_gaussian(group) for group in _clusters(other_samples, OTHER_CLUSTERS)]

    shape = values.shape[:2]
    flat = values.reshape(-1, 3)
    bias = 2.0 * math.log(max(tolerance, 1e-3))
    skin_score = torch.stack([_log_likelihood(flat, model) for model in skin_models]).logsumexp(0)
    skin_score = skin_score - math.log(len(skin_models))
    nearest = torch.stack([_distance(flat, model) for model in skin_models]).min(0).values
    closed = torch.sigmoid((CLOSED_AT * tolerance * tolerance - nearest) / CLOSED_SOFTNESS)
    if not other_models:
        return torch.stack([torch.ones_like(closed), closed]).view(2, *shape)
    other_score = torch.stack([_log_likelihood(flat, model) for model in other_models]).logsumexp(0)
    other_score = other_score - math.log(len(other_models))
    lenient = torch.sigmoid(skin_score - other_score + bias)
    return torch.stack([lenient, lenient * closed]).view(2, *shape)


def _colour(features, labels, boxes, tolerance: float):
    """Lenient and strict skin chances over the frame, each person read from their own colour."""
    height, width = int(features.shape[0]), int(features.shape[1])
    whole = _posterior(features, labels, slice(0, height), slice(0, width), tolerance)
    chances = whole if whole is not None else torch.full(
        (2, height, width), 0.5, device=features.device
    )
    for box in boxes:
        top, bottom, left, right = _padded(box, height, width, COLOUR_PADDING)
        local = _posterior(
            features, labels, slice(top, bottom), slice(left, right), tolerance
        )
        if local is not None:
            chances[:, top:bottom, left:right] = local
    return chances[0], chances[1]


def _covered(boxes, height: int, width: int, share: float, device):
    """A plane set inside every padded box."""
    covered = torch.zeros((height, width), device=device)
    for box in boxes:
        top, bottom, left, right = _padded(box, height, width, share)
        covered[top:bottom, left:right] = 1.0
    return covered


def _grow(plane, radius: int):
    """Dilate a ``(h, w)`` plane by a square of the given radius."""
    if radius <= 0:
        return plane
    return morphology.dilate(plane[None, None], 2 * radius + 1)[0, 0]


def _shrink(plane, radius: int):
    """Erode a ``(h, w)`` plane by a square of the given radius."""
    if radius <= 0:
        return plane
    return morphology.erode(plane[None, None], 2 * radius + 1)[0, 0]


def _guide(plane, rgb, radius: int, epsilon: float):
    """Guided-filter a ``(h, w)`` plane against the frame."""
    smoothed = guided.filter_with_guide(
        plane[None, ..., None], rgb[None], radius=radius, epsilon=epsilon
    )
    return smoothed[0, ..., 0].clamp(0.0, 1.0)


def _fill_gaps(hard, rgb, labels, ring: int):
    """Fill enclosed gaps the labels call body that continue the skin or are a highlight on it."""
    from ..mask import regions

    skin, loose, _, background = labels
    total = float(hard.sum())
    if total < 1.0:
        return hard
    height, width = int(hard.shape[0]), int(hard.shape[1])
    roots = regions._label(hard < 0.5)
    edges = torch.cat([roots[0], roots[-1], roots[:, 0], roots[:, -1]])
    open_air = torch.unique(edges[edges >= 0])

    flat = roots.reshape(-1)
    kept = flat >= 0
    found, inverse = torch.unique(flat[kept], return_inverse=True)
    counts = torch.bincount(inverse, minlength=len(found)).float()

    def mean(plane):
        summed = torch.zeros(len(found), device=plane.device)
        summed.index_add_(0, inverse, plane.reshape(-1)[kept].float())
        return summed / counts

    candidate = (
        ~torch.isin(found, open_air)
        & (counts <= GAP_SHARE * total)
        & (mean(skin + loose) >= 0.6)
        & (mean(background) <= 0.3)
    )
    if not bool(candidate.any()):
        return hard

    positions = torch.arange(height * width, device=roots.device)[kept]
    rows, columns = positions // width, positions % width
    span = [torch.full((len(found),), fill, device=roots.device, dtype=torch.long)
            for fill in (height, -1, width, -1)]
    span[0].scatter_reduce_(0, inverse, rows, reduce="amin")
    span[1].scatter_reduce_(0, inverse, rows, reduce="amax")
    span[2].scatter_reduce_(0, inverse, columns, reduce="amin")
    span[3].scatter_reduce_(0, inverse, columns, reduce="amax")

    lab = _oklab(rgb)
    chroma = lab[..., 1:].norm(dim=-1)
    filled = []
    for index in torch.nonzero(candidate).flatten().tolist():
        reach = ring + 1
        top = max(0, int(span[0][index]) - reach)
        bottom = min(height, int(span[1][index]) + reach + 1)
        left = max(0, int(span[2][index]) - reach)
        right = min(width, int(span[3][index]) + reach + 1)
        gap = roots[top:bottom, left:right] == found[index]
        plane = gap.float()
        inner = (plane - _shrink(plane, ring)) > 0.5
        outer = (_grow(plane, ring) - plane) > 0.5
        if not bool(inner.any()) or not bool(outer.any()):
            filled.append(int(found[index]))
            continue
        window = lab[top:bottom, left:right]
        step = float((window[inner].mean(0) - window[outer].mean(0)).norm())
        brighter = float(window[..., 0][gap].mean()) >= float(window[..., 0][outer].mean())
        tint = chroma[top:bottom, left:right]
        paler = float(tint[gap].mean()) <= float(tint[outer].mean())
        if step < SEAMLESS_STEP or (brighter and paler and step < HIGHLIGHT_STEP):
            filled.append(int(found[index]))
    if not filled:
        return hard
    chosen = torch.tensor(filled, device=roots.device)
    return torch.where(torch.isin(roots, chosen), torch.ones_like(hard), hard)


def _window(points, pad: float, height: int, width: int):
    """The pixel window around some points, as ``(top, bottom, left, right)``, or None."""
    left = max(0, int(math.floor(float(points[:, 0].min()) - pad)))
    right = min(width, int(math.ceil(float(points[:, 0].max()) + pad)) + 1)
    top = max(0, int(math.floor(float(points[:, 1].min()) - pad)))
    bottom = min(height, int(math.ceil(float(points[:, 1].max()) + pad)) + 1)
    if right <= left or bottom <= top:
        return None
    return top, bottom, left, right


def _grid(window, device):
    """Pixel centre coordinates over a window, as ``(y, x)`` planes."""
    top, bottom, left, right = window
    return torch.meshgrid(
        torch.arange(top, bottom, device=device, dtype=torch.float32) + 0.5,
        torch.arange(left, right, device=device, dtype=torch.float32) + 0.5,
        indexing="ij",
    )


def _polygon(points, margin: int, height: int, width: int, device):
    """A filled polygon, grown by a margin, on a full-size plane."""
    plane = torch.zeros((height, width), device=device)
    window = _window(points, margin + 1, height, width)
    if window is None:
        return plane
    ys, xs = _grid(window, device)
    inside = torch.zeros_like(xs, dtype=torch.bool)
    for index in range(len(points)):
        x0, y0 = points[index - 1]
        x1, y1 = points[index]
        crosses = (y0 > ys) != (y1 > ys)
        at = (x1 - x0) * (ys - y0) / (y1 - y0 + 1e-9) + x0
        inside ^= crosses & (xs < at)
    filled = _grow(inside.float(), margin)
    top, bottom, left, right = window
    plane[top:bottom, left:right] = filled
    return plane


def _stroke(points, half_width: float, height: int, width: int, device):
    """A polyline drawn at a half-width, on a full-size plane."""
    plane = torch.zeros((height, width), device=device)
    window = _window(points, half_width + 1, height, width)
    if window is None:
        return plane
    ys, xs = _grid(window, device)
    nearest = torch.full_like(xs, float("inf"))
    for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
        dx, dy = x1 - x0, y1 - y0
        along = (((xs - x0) * dx + (ys - y0) * dy) / (dx * dx + dy * dy + 1e-9)).clamp(0.0, 1.0)
        nearest = torch.minimum(nearest, torch.hypot(xs - (x0 + along * dx), ys - (y0 + along * dy)))
    top, bottom, left, right = window
    plane[top:bottom, left:right] = (nearest <= half_width).float()
    return plane


def _sure(scores, indices) -> bool:
    """Whether a feature's landmarks are confident enough to use."""
    return float(scores[list(indices)].mean()) >= LANDMARK_THRESHOLD


def _cut_outs(points, scores, height: int, width: int, eyes: str, mouth: str, device):
    """The eyes, brows and mouths to cut, and how many faces had any cut.

    Returns:
        ``(plane, faces)``: a ``(h, w)`` plane set where skin is removed, and a count.
    """
    removed = torch.zeros((height, width), device=device)
    faces = 0
    for person, sure in zip(points, scores):
        cut = False
        if eyes != "keep":
            for side, ring in enumerate(EYE_RINGS):
                if not _sure(sure, ring):
                    continue
                outline = person[list(ring)]
                span = float(outline[:, 0].max() - outline[:, 0].min())
                if span < 2.0:
                    continue
                centre = outline.mean(0)
                scale = torch.tensor(EYE_SCALE, device=device)
                widened = centre + (outline - centre) * scale
                margin = max(1, round(span * EYE_MARGIN))
                removed = torch.maximum(removed, _polygon(widened, margin, height, width, device))
                cut = True
                brow = BROWS[side]
                if eyes == "remove with brows" and _sure(sure, brow):
                    line = person[list(brow)]
                    removed = torch.maximum(
                        removed, _stroke(line, span * BROW_WIDTH, height, width, device)
                    )
        if mouth != "keep" and _sure(sure, LIPS):
            ring = person[list(MOUTH if mouth == "remove inside" else LIPS)]
            removed = torch.maximum(removed, _polygon(ring, 1, height, width, device))
            cut = True
        faces += int(cut)
    return removed, faces


def _faces(points, scores, height: int, width: int, device):
    """Every confident face, from the jaw up to the forehead, feathered at its edge."""
    plane = torch.zeros((height, width), device=device)
    for person, sure in zip(points, scores):
        brows = tuple(BROWS[0]) + tuple(BROWS[1])
        if not _sure(sure, JAW) or not _sure(sure, brows):
            continue
        jaw = person[list(JAW)]
        brow_line = person[list(brows)]
        chin = float(jaw[:, 1].max())
        lift = FOREHEAD * max(1.0, chin - float(brow_line[:, 1].mean()))
        forehead = brow_line.flip(0) - torch.tensor([0.0, lift], device=device)
        outline = torch.cat([jaw, forehead])
        plane = torch.maximum(plane, _polygon(outline, 0, height, width, device))
    feather = max(1, round(min(height, width) / 300))
    return functional.avg_pool2d(
        plane[None, None], 2 * feather + 1, stride=1, padding=feather, count_include_pad=False
    )[0, 0]


#: Colour and strength of the tint drawn over the skin on a node's panel.
TINT = (0.15, 1.0, 0.35)
TINT_STRENGTH = 0.55


def publish(image, masks, people: int, faces: int, facts: dict) -> str:
    """Hand a skin node's panel its first picture with the skin tinted, and its counts.

    Args:
        image: The ``IMAGE`` batch the masks were found in.
        masks: The ``MASK`` batch found.
        people: Most people found in one picture.
        faces: Most faces cut in one picture.
        facts: The settings to list, name to value.

    Returns:
        The summary line.
    """
    from ..interface import preview, run_result

    coverage = float(masks.mean()) * 100.0
    noun = "person" if people == 1 else "people"
    summary = (
        f"skin on {people} {noun}, {coverage:.1f}% of the picture"
        if people else f"no person found; {coverage:.1f}% of the picture is skin"
    )
    first = image[:1, ..., :3].float()
    tint = torch.tensor(TINT, dtype=first.dtype).view(1, 1, 1, 3)
    weight = masks[:1, ..., None].to(first.dtype) * TINT_STRENGTH
    preview.publish_output(first * (1.0 - weight) + tint * weight)
    run_result.publish(
        summary=summary,
        counts={"people": people, "faces": faces, "skin %": round(coverage, 1)},
        facts=facts,
    )
    return summary

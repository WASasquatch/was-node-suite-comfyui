"""Finding where a clip cuts from one shot to the next, from its measured motion.

A scene is a run of frames ``start`` to ``stop``, ``stop`` excluded.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

__all__ = ["SHEET_COLUMNS", "THUMB_WIDTH", "contact_sheet", "ranges", "starts"]

#: Width, in pixels, of one scene's picture on the contact sheet.
THUMB_WIDTH = 192

#: Most pictures in one row of the contact sheet.
SHEET_COLUMNS = 6

#: Gap between pictures on the contact sheet, in pixels.
SHEET_GAP = 4


def starts(cuts, min_frames: int = 1) -> list[int]:
    """The first frame of every scene.

    Args:
        cuts: One flag per pair of neighbouring frames, True where the pair is a cut.
        min_frames: Fewest frames a scene may hold; a cut sooner than this after the last one
            is not taken.

    Returns:
        Frame indices, starting with 0, in order.
    """
    found = [0]
    for index, cut in enumerate(cuts):
        if cut and index + 1 - found[-1] >= max(1, int(min_frames)):
            found.append(index + 1)
    return found


def ranges(first_frames, count: int) -> list[tuple[int, int]]:
    """Each scene as ``(start, stop)``.

    Args:
        first_frames: What :func:`starts` returned.
        count: Frames in the clip.

    Returns:
        One pair per scene, ``stop`` excluded.
    """
    bounds = list(first_frames) + [int(count)]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def contact_sheet(frames, first_frames):
    """The first frame of every scene, side by side, numbered.

    Args:
        frames: ``(frames, height, width, channels)`` in ``[0, 1]``.
        first_frames: What :func:`starts` returned.

    Returns:
        ``(1, height, width, 3)`` in ``[0, 1]``.
    """
    from PIL import Image, ImageDraw

    height, width = (int(v) for v in frames.shape[1:3])
    thumb_h = max(1, int(round(THUMB_WIDTH * height / max(width, 1))))
    columns = min(SHEET_COLUMNS, len(first_frames))
    rows = int(math.ceil(len(first_frames) / columns))
    sheet_w = columns * THUMB_WIDTH + (columns + 1) * SHEET_GAP
    sheet_h = rows * thumb_h + (rows + 1) * SHEET_GAP
    sheet = Image.new("RGB", (sheet_w, sheet_h), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)
    for number, start in enumerate(first_frames):
        picture = frames[start, ..., :3].float().permute(2, 0, 1).unsqueeze(0)
        picture = F.interpolate(picture, size=(thumb_h, THUMB_WIDTH), mode="bilinear",
                                align_corners=False, antialias=True)
        pixels = (picture[0].permute(1, 2, 0).clamp(0, 1) * 255 + 0.5).byte().cpu().numpy()
        row, column = divmod(number, columns)
        x = SHEET_GAP + column * (THUMB_WIDTH + SHEET_GAP)
        y = SHEET_GAP + row * (thumb_h + SHEET_GAP)
        sheet.paste(Image.fromarray(pixels), (x, y))
        label = f"{number + 1}  f{start}"
        draw.rectangle((x, y, x + 8 + 7 * len(label), y + 16), fill=(0, 0, 0))
        draw.text((x + 4, y + 2), label, fill=(255, 255, 255))
    data = torch.frombuffer(bytearray(sheet.tobytes()), dtype=torch.uint8)
    return (data.view(sheet_h, sheet_w, 3).float() / 255.0).unsqueeze(0)

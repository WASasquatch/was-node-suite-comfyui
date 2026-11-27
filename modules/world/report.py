"""What the point cloud nodes publish for their panels: a strip of views and their figures."""

from __future__ import annotations

__all__ = ["strip", "publish"]

from collections.abc import Sequence

import torch

from ..interface import preview, run_result

#: Height of each picture in the strip drawn on a node, in pixels.
STRIP_HEIGHT = 256


def strip(pictures: Sequence[torch.Tensor], gap: int = 6) -> torch.Tensor:
    """Pictures ``(H, W, 3)`` side by side at one height, as a ``(1, H, W, 3)`` image."""
    tiles = []
    for picture in pictures:
        height, width = int(picture.shape[0]), int(picture.shape[1])
        side = max(1, round(width * STRIP_HEIGHT / max(1, height)))
        tile = torch.nn.functional.interpolate(picture.float().cpu().permute(2, 0, 1)[None], size=(STRIP_HEIGHT, side),
                                               mode="bilinear", align_corners=False, antialias=True)[0]
        tiles.append(tile)
        tiles.append(torch.full((3, STRIP_HEIGHT, gap), 0.12))
    if not tiles:
        return torch.zeros(1, STRIP_HEIGHT, STRIP_HEIGHT, 3)
    return torch.cat(tiles[:-1], dim=2).permute(1, 2, 0)[None].clamp(0, 1)


def publish(pictures: Sequence[torch.Tensor], summary: str, counts: dict, facts: dict | None = None,
            status: str = run_result.OK) -> None:
    """Publish a strip of views and a run's figures for the node's panel."""
    if pictures:
        preview.publish_output(strip(pictures))
    run_result.publish(status=status, summary=summary, counts=counts, facts=dict(facts or {}))

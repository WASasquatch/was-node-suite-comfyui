"""Making an image tile against itself, and repeating the tile into a grid.

Images are ``(batch, height, width, channels)`` on a 0 to 1 scale. An answer is smaller
than its source by the blended fraction on each axis.
"""

from __future__ import annotations

import math

import torch

from . import scratch

__all__ = ["LARGEST_BLEND", "make_seamless", "seamless_size"]

#: Largest fraction of a side the blend can span. Above a half the stripe taken off one edge
#: reaches past the middle and there is nothing left to keep.
LARGEST_BLEND = 0.5


def _blended(planes: torch.Tensor, fraction: float, axis: int) -> torch.Tensor:
    """Fade the far edge over the near edge along one axis and drop the far stripe.

    Args:
        planes: ``(batch, height, width, channels)`` tensor.
        fraction: Share of the axis the fade spans, 0 to :data:`LARGEST_BLEND`.
        axis: 1 for height, 2 for width.

    Returns:
        A tensor shorter along ``axis`` by the stripe, the near edge carrying the fade.
    """
    length = int(planes.shape[axis])
    stripe = int(math.floor(length * fraction))
    if stripe < 1 or stripe * 2 > length:
        return planes

    far = planes.narrow(axis, length - stripe, stripe)
    near = planes.narrow(axis, 0, stripe)
    # Opaque where the far edge lands and clear where the near edge carries on alone.
    ramp = torch.linspace(1.0, 0.0, stripe, device=planes.device, dtype=planes.dtype)
    shape = [1] * planes.dim()
    shape[axis] = stripe
    weight = ramp.view(shape)

    kept = planes.narrow(axis, 0, length - stripe).clone()
    kept.narrow(axis, 0, stripe).copy_(far * weight + near * (1.0 - weight))
    return kept


def seamless_size(height: int, width: int, blending: float) -> tuple[int, int]:
    """The size :func:`make_seamless` answers for one frame.

    Args:
        height: Source height in pixels.
        width: Source width in pixels.
        blending: Share of each side the fade spans.

    Returns:
        A ``(height, width)`` pair.
    """
    fraction = min(max(float(blending), 0.0), LARGEST_BLEND)
    down = int(math.floor(height * fraction))
    right = int(math.floor(width * fraction))
    if right < 1 or right * 2 > width:
        right = 0
    if down < 1 or down * 2 > height:
        down = 0
    return height - down, width - right


def make_seamless(
    images: torch.Tensor,
    blending: float = 0.4,
    tiled: bool = False,
    tiles: int = 2,
    node: str = "",
    advice: str = "",
) -> torch.Tensor:
    """Make every frame tile against itself, optionally answering a grid of the tiles.

    Args:
        images: ``(batch, height, width, channels)`` tensor on a 0 to 1 scale.
        blending: Share of each side the fade spans, held to 0 to :data:`LARGEST_BLEND`.
            0 answers the source unchanged.
        tiled: Answer a ``tiles`` by ``tiles`` grid rather than the single tile.
        tiles: Grid size along each axis, read only when ``tiled`` is set.
        node: Display name of the calling node, for a refusal to hold the answer.
        advice: Closing sentence of that refusal, naming what to change on the node.

    Returns:
        The tiles, or the grids of them, as ``(batch, height, width, channels)``.

    Raises:
        MemoryError: Neither free memory nor a scratch drive can hold the answer.
    """
    fraction = min(max(float(blending), 0.0), LARGEST_BLEND)
    batch, height, width, channels = (int(side) for side in images.shape)
    tile_height, tile_width = seamless_size(height, width, fraction)
    if not tiled and (tile_height, tile_width) == (height, width):
        return images.contiguous()

    count = max(int(tiles), 1) if tiled else 1
    shape = (batch, tile_height * count, tile_width * count, channels)
    if images.device.type == "cpu":
        answer = scratch.allocate(shape, dtype=images.dtype, node=node, advice=advice)
    else:
        answer = torch.empty(shape, dtype=images.dtype, device=images.device)
    for index in range(batch):
        tile = _blended(_blended(images[index:index + 1], fraction, 2), fraction, 1)[0]
        for row in range(count):
            for column in range(count):
                answer[
                    index,
                    row * tile_height:(row + 1) * tile_height,
                    column * tile_width:(column + 1) * tile_width,
                ].copy_(tile)
    return answer

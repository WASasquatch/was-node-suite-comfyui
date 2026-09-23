"""Running a model over an image one tile at a time.

:func:`tiled_upscale` overlaps the tiles, fades the overlap, and moves one tile at a time to
the device. A target equal to the source does not magnify.
"""

from __future__ import annotations

import torch

__all__ = ["tiled_upscale"]


#: Masks already built, keyed by shape, fade widths, dtype and device.
_MASKS: dict = {}

#: Masks held before the oldest is dropped.
_MASK_CACHE = 16


def _ramp(length: int, taper: int, device, dtype) -> torch.Tensor:
    """A 1-D weight running up from near zero at each faded end.

    Args:
        length: Axis length in pixels.
        taper: Pixels faded at each end, capped at half the length.
        device: Where the ramp is built.
        dtype: Working dtype.

    Returns:
        A ``length`` long tensor, 1.0 across the middle.
    """
    weights = torch.ones(length, device=device, dtype=dtype)
    taper = min(int(taper), length // 2)
    if taper > 0:
        rising = torch.arange(1, taper + 1, device=device, dtype=dtype) / float(taper)
        weights[:taper] = rising
        weights[length - taper:] = rising.flip(0)
    return weights


# A tile is upscaled without knowing what its neighbours contain, so the model's guesses
# disagree along the join. The cross-fade hides that seam.
def _feather_mask(tile: torch.Tensor, rows: int, columns: int) -> torch.Tensor:
    """The cross-fade mask for one upscaled tile.

    Args:
        tile: The upscaled tile, used for its size, device and dtype.
        rows: Rows faded at the top and bottom, capped at half the tile's height.
        columns: Columns faded at the left and right, capped at half its width.

    Returns:
        A ``(1, 1, height, width)`` mask that is 1.0 in the middle and falls linearly to
        near zero at each faded edge.
    """
    height, width = tile.shape[2], tile.shape[3]
    key = (height, width, int(rows), int(columns), tile.dtype, str(tile.device))
    held = _MASKS.get(key)
    if held is not None:
        return held
    mask = (_ramp(height, rows, tile.device, tile.dtype)[:, None]
            * _ramp(width, columns, tile.device, tile.dtype)[None, :])
    mask = mask.reshape(1, 1, height, width)
    if len(_MASKS) >= _MASK_CACHE:
        _MASKS.pop(next(iter(_MASKS)))
    _MASKS[key] = mask
    return mask



#: Video memory left free beyond the accumulators, for the tiles themselves.
_HEADROOM = 1024 * 1024 * 1024


def _fits(device, height: int, width: int, channels: int, element_size: int) -> bool:
    """Whether the accumulators for a whole picture fit on the compute device.

    Args:
        device: The compute device.
        height: Target height in pixels.
        width: Target width in pixels.
        channels: Colour channels the model answers with.
        element_size: Bytes per value.

    Returns:
        True where they fit with :data:`_HEADROOM` to spare.
    """
    if str(device) == "cpu":
        return False
    try:
        from comfy import model_management

        free = model_management.get_free_memory(device)
    except Exception:
        return False
    wanted = height * width * (channels + 1) * element_size
    return free > wanted + _HEADROOM


def _starts(length: int, tile: int, step: int) -> list[int]:
    """Where each tile begins along one axis, the last one flush with the far edge.

    Args:
        length: Axis length in pixels.
        tile: Tile edge in pixels.
        step: Distance between the starts of neighbouring tiles.

    Returns:
        Ascending start positions. Every tile is a full ``tile`` wide unless the axis is
        shorter than one.
    """
    if length <= tile:
        return [0]
    found, position = [], 0
    while position + tile < length:
        found.append(position)
        position += max(1, step)
    found.append(length - tile)
    return sorted(set(found))


@torch.inference_mode()
def tiled_upscale(
    samples,
    function,
    tile_size=512,
    overlap=32,
    output_device="cpu",
    pbar=None,
    feather=0,
    target_height=None,
    target_width=None,
    resample_method="lanczos",
    device=None,
):
    """Upscale a batch tile by tile and cross-fade the overlaps.

    Args:
        samples: ``(batch, channels, height, width)`` tensor of the source images.
        function: Callable run on each tile, normally the upscale model itself.
        tile_size: Tile edge in *input* pixels. Larger tiles are faster and need more of
            the compute device's memory. Every tile is this wide, the last on each axis
            sitting flush with the far edge, unless the image is smaller than one tile.
        overlap: How far neighbouring tiles overlap, in input pixels. Clamped below the
            tile size, since a tile cannot overlap itself entirely.
        output_device: Device the accumulators and the result live on.
        pbar: Optional progress bar; ``update(1)`` is called once per tile.
        feather: Cross-fade width in *output* pixels. 0 or less derives it from the
            overlap, scaled by the magnification actually being applied.
        target_height: Final height in pixels. Required.
        target_width: Final width in pixels. Required.
        resample_method: Kernel used where the model's own output size does not match the
            share of the target the tile covers.
        device: Compute device tiles are moved to. Defaults to ComfyUI's torch device.

    Returns:
        A ``(batch, channels, target_height, target_width)`` tensor on ``output_device``.

    Raises:
        ValueError: ``samples`` is not four-dimensional, the target size is missing, or
            ``tile_size`` is not positive.
    """
    import comfy.utils
    from comfy import model_management

    if samples.ndim != 4:
        raise ValueError(
            f"tiled_upscale() takes a (batch, channels, height, width) tensor and was given "
            f"{tuple(samples.shape)}"
        )
    if target_height is None or target_width is None:
        raise ValueError("tiled_upscale() needs both target_height and target_width")

    tile_size = int(tile_size)
    if tile_size <= 0:
        raise ValueError(f"tile_size must be a positive number of pixels, not {tile_size}")

    overlap = max(0, int(overlap))
    if overlap >= tile_size:
        overlap = tile_size - 1 if tile_size > 1 else 0
    tile_step = tile_size - overlap if tile_size > overlap else tile_size

    if device is None:
        device = model_management.get_torch_device()

    samples = samples.to(output_device)
    batch_size, channels, in_height, in_width = samples.shape

    scale_y = float(target_height) / float(in_height)
    scale_x = float(target_width) / float(in_width)

    # Accumulators sit on the compute device where they fit, on the output device otherwise.
    gather = device if _fits(device, target_height, target_width, channels,
                             samples.element_size()) else output_device
    blended = None

    for index in range(batch_size):
        source = samples[index:index + 1]
        accumulator = None
        weights = None

        for y in _starts(in_height, tile_size, tile_step):
            for x in _starts(in_width, tile_size, tile_step):
                y_end = min(y + tile_size, in_height)
                x_end = min(x + tile_size, in_width)

                tile_source = source[:, :, y:y_end, x:x_end].to(device, non_blocking=False)
                tile_native = function(tile_source)

                if blended is None:
                    blended = torch.zeros(
                        (batch_size, tile_native.shape[1], target_height, target_width),
                        device=output_device,
                        dtype=tile_native.dtype,
                    )
                if accumulator is None:
                    accumulator = torch.zeros(
                        (1, tile_native.shape[1], target_height, target_width),
                        device=gather,
                        dtype=tile_native.dtype,
                    )
                    weights = torch.zeros(
                        (1, 1, target_height, target_width),
                        device=gather,
                        dtype=tile_native.dtype,
                    )

                out_y = int(round(y * target_height / in_height))
                out_x = int(round(x * target_width / in_width))
                tile_height = max(1, int(round(y_end * target_height / in_height)) - out_y)
                tile_width = max(1, int(round(x_end * target_width / in_width)) - out_x)

                if tile_native.shape[2] != tile_height or tile_native.shape[3] != tile_width:
                    tile_scaled = comfy.utils.common_upscale(
                        tile_native, tile_width, tile_height, resample_method, "disabled"
                    )
                else:
                    tile_scaled = tile_native
                tile = tile_scaled.to(gather)

                if feather is None or feather <= 0:
                    rows = int(round(overlap * scale_y))
                    columns = int(round(overlap * scale_x))
                else:
                    rows = columns = int(feather)
                mask = _feather_mask(tile, rows, columns)

                accumulator[:, :, out_y:out_y + tile.shape[2], out_x:out_x + tile.shape[3]] += (
                    tile * mask
                )
                weights[:, :, out_y:out_y + tile.shape[2], out_x:out_x + tile.shape[3]] += mask

                if pbar is not None:
                    pbar.update(1)

                del tile_scaled, tile_native, tile_source

        # A pixel no tile reached keeps a zero weight; dividing by one there leaves it black
        # rather than turning it into a division by zero.
        safe = torch.where(weights == 0.0, torch.ones_like(weights), weights)
        blended[index:index + 1] = (accumulator / safe).to(output_device)

        del accumulator, weights, safe

    return blended.to(output_device)

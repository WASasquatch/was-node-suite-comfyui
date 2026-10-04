"""Running a model over an image one tile at a time.

:func:`tiled_upscale` overlaps the tiles, fades the overlap, and moves one tile at a time to
the device. A target equal to the source does not magnify.
"""

from __future__ import annotations

import torch

from . import scratch

__all__ = ["PRECISIONS", "pick_dtype", "release", "tiled_upscale"]

#: What an upscale model may run in, `auto` following what the model declares it supports.
PRECISIONS = ("auto", "32 bit float", "16 bit float", "bfloat16")


#: Masks already built, keyed by shape, fade widths, dtype and device.
_MASKS: dict = {}

#: Masks held before the oldest is dropped.
_MASK_CACHE = 16

#: Lanczos axis matrices already built, keyed by source length, target length and device.
_AXES: dict = {}

#: Axis matrices held before the oldest is dropped.
_AXIS_CACHE = 32


def pick_dtype(precision: str, upscale_model, device) -> torch.dtype:
    """What an upscale model runs in.

    Args:
        precision: An entry from :data:`PRECISIONS`.
        upscale_model: The loaded upscale model, read for the dtypes it declares.
        device: The device the model runs on.

    Returns:
        A ``torch.dtype``.
    """
    from comfy import model_management

    named = {
        "32 bit float": torch.float32,
        "16 bit float": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if precision in named:
        return named[precision]
    if not model_management.should_use_fp16(device):
        return torch.float32
    if getattr(upscale_model, "supports_half", False):
        return torch.float16
    if getattr(upscale_model, "supports_bfloat16", False):
        return torch.bfloat16
    return torch.float32


def release(upscale_model) -> None:
    """Return an upscale model to float32 on the CPU.

    Args:
        upscale_model: The loaded upscale model.
    """
    upscale_model.model.to(torch.float32)
    upscale_model.to("cpu")


def _axis_weights(size_in: int, size_out: int, device) -> torch.Tensor:
    """The ``(size_out, size_in)`` lanczos matrix for one axis, as float32 on ``device``."""
    from .resample import matrix

    key = (int(size_in), int(size_out), str(device))
    held = _AXES.get(key)
    if held is None:
        held = matrix(int(size_in), int(size_out), "lanczos").to(device, torch.float32)
        if len(_AXES) >= _AXIS_CACHE:
            _AXES.pop(next(iter(_AXES)))
        _AXES[key] = held
    return held


def _resample(tile: torch.Tensor, width: int, height: int, method: str) -> torch.Tensor:
    """Resize one ``(1, channels, height, width)`` tile on its own device, in float.

    Args:
        tile: The model's answer for one tile.
        width: Target width in pixels.
        height: Target height in pixels.
        method: ``nearest-exact``, ``bilinear``, ``area``, ``bicubic`` or ``lanczos``.

    Returns:
        ``(1, channels, height, width)`` on the tile's device and dtype.
    """
    import torch.nn.functional as F

    if method == "lanczos":
        down = _axis_weights(tile.shape[2], height, tile.device)
        across = _axis_weights(tile.shape[3], width, tile.device)
        return torch.matmul(torch.matmul(down, tile.float()), across.T).to(tile.dtype)
    if method in ("bilinear", "bicubic"):
        return F.interpolate(
            tile, size=(height, width), mode=method, align_corners=False, antialias=True
        )
    return F.interpolate(tile, size=(height, width), mode=method)


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
    out=None,
    limits=None,
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
        out: A ``(batch, target_height, target_width, channels)`` tensor each finished frame
            is written into, channels last. None allocates the result on ``output_device``.
        limits: ``(low, high)`` each finished frame is clamped to, or None to leave it as
            the model answered.

    Returns:
        ``out`` where one is given, otherwise a ``(batch, channels, target_height,
        target_width)`` tensor on ``output_device``.

    Raises:
        ValueError: ``samples`` is not four-dimensional, the target size is missing,
            ``tile_size`` is not positive, or ``out`` does not match the result's shape.
    """
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
        # The frame moves to the device whole; its tiles are cut there.
        source = samples[index:index + 1].to(device)
        accumulator = None
        weights = None

        for y in _starts(in_height, tile_size, tile_step):
            for x in _starts(in_width, tile_size, tile_step):
                y_end = min(y + tile_size, in_height)
                x_end = min(x + tile_size, in_width)

                tile_source = source[:, :, y:y_end, x:x_end]
                tile_native = function(tile_source)

                if out is not None and blended is None:
                    wanted = (batch_size, target_height, target_width, tile_native.shape[1])
                    if tuple(out.shape) != wanted:
                        raise ValueError(
                            f"tiled_upscale() was given an out tensor shaped "
                            f"{tuple(out.shape)} for a result shaped {wanted}"
                        )
                    blended = out
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
                    tile_scaled = _resample(tile_native, tile_width, tile_height, resample_method)
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
        frame = accumulator.div_(safe)
        if limits is not None:
            frame.clamp_(limits[0], limits[1])
        if out is not None:
            # Reordered to channels last on the frame's own device before the copy out.
            out[index].copy_(frame[0].permute(1, 2, 0).contiguous())
            scratch.trim(out[index])
        else:
            blended[index:index + 1] = frame.to(output_device)

        del accumulator, weights, safe, frame, source

    return out if out is not None else blended.to(output_device)

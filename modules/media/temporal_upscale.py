"""Upscaling a clip with a model, its added detail held steady along the clip's motion.

Frames are ``(frames, height, width, channels)`` in ``[0, 1]``. The finished clip is encoded
one frame at a time; residuals are kept in half precision.
"""

from __future__ import annotations

import os
from contextlib import ExitStack
from dataclasses import dataclass
from fractions import Fraction

import torch

from .. import log
from ..image import motion as motion_field
from ..image import optical_flow, scratch, temporal_consistency, tiled_upscale
from . import clip as clips
from . import retime, temp_video, video

__all__ = [
    "CUT_SIDE", "MINIMUM_BLOCK", "MINIMUM_TILE", "RATE_MATCH", "ROOM_SHARE", "Result", "output_rate",
    "render",
]

logger = log.get_logger("media.temporal_upscale")

#: Long side, in pixels, cuts are looked for at before new frames are drawn.
CUT_SIDE = 384

#: Smallest tile the out-of-memory retry falls back to before giving up.
MINIMUM_TILE = 64

#: Fewest frames steadied together when the clip does not fit whole.
MINIMUM_BLOCK = 8

#: Share of the roomiest place, memory or scratch drive, the steadying keeps its frames in.
ROOM_SHARE = 0.5

#: Share two frame rates may differ by and still be read as the same rate.
RATE_MATCH = 1e-3

#: Colour spaces the finished clip cannot be written in.
HDR_SPACES = ("HDR", "HDR PQ")


@dataclass
class Result:
    """What one run wrote.

    Attributes:
        frames: Frames in the finished clip.
        rate: Its frames per second.
        size: Its ``(width, height)``.
        drawn: In-between frames drawn by the interpolation network.
        held: In-between frames repeating the nearer source frame.
        tile: Tile edge the upscale finished with, in source pixels.
        precision: What the upscale model ran in.
        before: The plain upscale written for playback, or ``None``.
        after: The finished clip written smaller for playback, or ``None`` where the
            finished clip itself plays back.
    """

    frames: int
    rate: Fraction
    size: tuple[int, int]
    drawn: int
    held: int
    tile: int
    precision: str
    before: str | None = None
    after: str | None = None


def output_rate(source_rate, wanted: float) -> Fraction:
    """The frame rate a run writes at.

    Args:
        source_rate: The source's frames per second.
        wanted: Frames per second asked for; 0 or a rate within :data:`RATE_MATCH` of the
            source's keeps the source's.

    Returns:
        The rate as a ``Fraction``.
    """
    source_rate = Fraction(source_rate)
    if wanted is None or float(wanted) <= 0.0:
        return source_rate
    rate = Fraction(float(wanted)).limit_denominator(1001)
    if abs(float(rate) / float(source_rate) - 1.0) <= RATE_MATCH:
        return source_rate
    return rate


def _even(length: float) -> int:
    """``length`` rounded to an even number of pixels, at least 2."""
    return max(2, 2 * int(round(length / 2.0)))


def _codes(frame, bit_depth: int):
    """A ``(1, 3, height, width)`` frame as a ``(height, width, 3)`` uint8 or uint16 numpy array."""
    import numpy as np

    frame = frame[0].clamp(0.0, 1.0).permute(1, 2, 0)
    if bit_depth >= 10:
        return (frame * 65535.0).round().to(torch.int32).contiguous().cpu().numpy().astype(np.uint16)
    return (frame * 255.0).round().to(torch.uint8).contiguous().cpu().numpy()


def render(
    clip,
    upscale_model,
    target: str,
    before: str | None = None,
    after: str | None = None,
    factor: float = 4.0,
    strength: float = 0.8,
    rate: float = 0.0,
    tile_size: int = 512,
    overlap: int = 32,
    precision: str = "auto",
    crf: float = 16.0,
    motion_model=None,
    ema_vfi_model=None,
    new_frames: str = "auto",
    device=None,
    node: str = "",
) -> Result:
    """Upscale a clip frame by frame, hold the added detail steady, and encode it.

    Args:
        clip: A :class:`~.clip.Clip`, its frames float or uint8 codes.
        upscale_model: A loaded upscale model.
        target: The mp4 file the finished clip is written to.
        before: An mp4 file the plain upscale is written to for playback, or ``None``.
        after: An mp4 file the finished clip is written to for playback where it is larger
            than :data:`~.temp_video.PREVIEW_SIDE`, or ``None``.
        factor: Final size relative to the source, rounded to even sides.
        strength: How much of each frame's detail comes from its neighbours, 0 to 1.
        rate: Frames per second of the finished clip; 0 keeps the source's.
        tile_size: Tile edge in source pixels.
        overlap: How far neighbouring tiles overlap, in source pixels.
        precision: An entry from :data:`~modules.image.tiled_upscale.PRECISIONS`.
        crf: x264 quality of the finished clip; lower is better and larger.
        motion_model: A flow network for the motion, or ``None`` for the texture flow.
        ema_vfi_model: An EMA-VFI model for frames between two source frames, or ``None``
            to repeat the nearer one.
        new_frames: How a frame between two source frames is made, one of
            :data:`~.retime.MODES`; ``interpolate`` and ``auto`` fall back to ``hold`` without
            ``ema_vfi_model``.
        device: Where the work runs.
        node: Display name of the calling node, for the log and the refusal.

    Returns:
        The :class:`Result`.

    Raises:
        ValueError: The clip is HDR, the frame rate would make too many frames, or the
            EMA-VFI checkpoint cannot land where the new frames fall.
        MemoryError: Neither free memory nor a scratch drive can hold the residuals.
    """
    from comfy import model_management

    if clip.color_space in HDR_SPACES:
        raise ValueError(
            f"{node} writes sRGB video and this clip is {clip.color_space}. Load an sRGB "
            f"version of the clip."
        )
    device = torch.device(device) if device is not None else model_management.get_torch_device()
    frames = clip.frames
    count = int(frames.shape[0])
    rate_out = output_rate(clip.rate, rate)
    strength = max(0.0, min(1.0, float(strength)))

    where = None
    if rate_out != clip.rate:
        where = retime.rate_positions(count, clip.rate, rate_out)
        if ema_vfi_model is not None:
            retime.check_network(ema_vfi_model.name, where)
    total = len(where) if where is not None else count
    steady = strength > 0.0 and total > 1
    steps = max(count - 1, 0) + total if where is not None else 0
    steps += (total - 1 if steady else 0) + 2 * total
    step = clips.progress(steps)

    in_height, in_width = int(frames.shape[1]), int(frames.shape[2])
    height, width = _even(in_height * float(factor)), _even(in_width * float(factor))
    size = (height, width)
    advice = (
        "Lowering upscale_factor or frame_rate, or upscaling the clip in shorter parts, "
        "also fits it."
    )
    residuals = behind = None
    block = None
    if steady:
        per_frame = height * width * 3 * 2
        budget = int(scratch.room() * ROOM_SHARE)
        if 2 * total * per_frame > budget:
            block = max(MINIMUM_BLOCK, (budget // per_frame - temporal_consistency.LOOKAHEAD) // 2)
        keep = total if block is None else min(total, block + temporal_consistency.LOOKAHEAD)
        hold = total if block is None else min(total, block)
        residuals = scratch.FrameStore(
            keep, (height, width, 3), torch.float16, node, advice, alongside=hold * per_frame
        )
        behind = scratch.FrameStore(
            hold, (height, width, 3), torch.float16, node, advice,
            alongside=keep * per_frame if residuals.handle is None else 0,
        )
        if block is not None:
            logger.info(
                "steadying %d frame(s) in blocks of %d, each looking %d frame(s) ahead",
                total, block, temporal_consistency.LOOKAHEAD,
            )

    try:
        drawn = held = 0
        if where is not None:
            mode = str(new_frames)
            if mode in ("interpolate", "auto") and ema_vfi_model is None:
                mode = "hold"
            cuts, pairs = [], []
            if count > 1:
                measured = motion_field.measure(frames, CUT_SIDE, device, step, network=motion_model)
                cuts = measured.cuts()
                pairs = retime.drawable(measured) if mode == "auto" else [not cut for cut in cuts]
                del measured
            between = [
                int(at) for at in where
                if retime.EXACT < at - int(at) < 1.0 - retime.EXACT and int(at) < count - 1
            ]
            if mode in ("interpolate", "auto"):
                drawn = sum(1 for low in between if pairs[low])
            held = len(between) - drawn
            net = None
            if mode in ("interpolate", "auto") and drawn:
                ema_vfi_model.backend.load()
                net = ema_vfi_model.backend.model
            drawn_on = next(net.parameters()).device if net is not None else device
            frames = retime.frames_at(frames, where, mode, cuts, net, drawn_on, step, node, drawn=pairs)

        motion = None
        if steady:
            motion = motion_field.measure(
                frames, motion_field.MOTION_SIDE, device, step, network=motion_model
            )

        required = model_management.module_size(upscale_model.model)
        required += tile_size * tile_size * 3 * 4 * max(float(getattr(upscale_model, "scale", 4)), 1.0) * 384
        required += height * width * 4 * 4 * 6
        model_management.free_memory(required, device)
        upscale_model.to(device)
        working = tiled_upscale.pick_dtype(precision, upscale_model, device)
        if working is not torch.float32:
            upscale_model.model.to(working)
        tile = int(tile_size)

        def model(piece):
            return upscale_model(piece.to(working).contiguous(memory_format=torch.channels_last)).float()

        def upscaled(index):
            nonlocal tile
            low = optical_flow.unit(frames[index:index + 1, ..., :3], device).permute(0, 3, 1, 2)
            while True:
                try:
                    return tiled_upscale.tiled_upscale(
                        low, model, tile, overlap, output_device=device, target_height=height,
                        target_width=width, resample_method="lanczos", device=device,
                        limits=(0.0, 1.0),
                    )
                except model_management.OOM_EXCEPTION:
                    tile //= 2
                    if tile < MINIMUM_TILE:
                        raise
                    logger.warning("the upscale ran out of memory; retrying with %d pixel tiles", tile)
                    model_management.soft_empty_cache()

        def base(index):
            return temporal_consistency.base(frames, index, size, 3, device)

        made = set()
        last = {}

        def residual(index):
            if index in made:
                plain = residuals.read(index, device).permute(2, 0, 1).unsqueeze(0)
            else:
                plain = upscaled(index) - base(index)
                made.add(index)
                if steady:
                    residuals.write(index, plain[0].permute(1, 2, 0).to(torch.float16).contiguous())
            last.clear()
            last[index] = plain
            return plain

        bit_depth = 10 if int(clip.bit_depth) >= 10 else 8
        sound = clips.slice_audio(clip.audio, 0, total, rate_out)
        shown = temp_video.preview_size(width, height)
        before = before if steady else None
        after = after if max(width, height) > temp_video.PREVIEW_SIDE else None
        written = [path for path in (target, before, after) if path]

        def opened(stack, path):
            if not path:
                return None
            return stack.enter_context(video.Encoder(
                path, "h264", shown[0], shown[1], rate_out, options=temp_video.PREVIEW_OPTIONS,
                color_space="sRGB",
            ))

        try:
            with ExitStack() as stack:
                finished = stack.enter_context(video.Encoder(
                    target, "h264", width, height, rate_out, audio=sound,
                    options={"crf": f"{float(crf):g}"}, color_space="sRGB", bit_depth=bit_depth,
                ))
                plain_side = opened(stack, before)
                small_side = opened(stack, after)
                for index, steadied in temporal_consistency.passes(
                    total, residual, motion, strength, size, device, behind, step, block
                ):
                    under = base(index)
                    done = under + steadied
                    finished.write_rgb(_codes(done, bit_depth))
                    plain = last.pop(index)
                    if residuals is not None:
                        residuals.release(index)
                    if plain_side is not None:
                        plain_side.write_rgb(temp_video.shrink(under + plain, shown))
                    if small_side is not None:
                        small_side.write_rgb(temp_video.shrink(done, shown))
        except BaseException:
            for path in written:
                try:
                    os.remove(path)
                except OSError:
                    pass
            raise
        finally:
            tiled_upscale.release(upscale_model)
    finally:
        for store in (residuals, behind):
            if store is not None:
                store.close()
        model_management.soft_empty_cache()

    logger.info(
        "upscaled %d frame(s) to %dx%d at %g fps: %d drawn between, %d held, strength %g",
        total, width, height, float(rate_out), drawn, held, strength,
    )
    return Result(
        frames=total,
        rate=rate_out,
        size=(width, height),
        drawn=drawn,
        held=held,
        tile=tile,
        precision=str(working).replace("torch.", ""),
        before=before,
        after=after,
    )

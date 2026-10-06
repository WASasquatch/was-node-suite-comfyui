"""Sampling a MiniMax H3 latent in overlapping tiles across time, height and width.

Each model call runs once per tile, predictions blended where tiles overlap; a tile keeps its
positions in the whole clip.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .. import log

__all__ = [
    "AUTO_OVERLAP",
    "AUTO_TOKENS",
    "AUTO_WINDOW_OVERLAP",
    "TILES_WRAPPER",
    "Tiling",
    "auto_tiling",
    "cut_report",
    "manual_tiling",
    "planned",
    "plot",
    "scene_spans",
    "scene_tiling",
    "tiled_model",
    "tiling_settings",
]

logger = log.get_logger("model.h3_tiles")

#: Key the tiled model's wrapper is registered under.
TILES_WRAPPER = "was_h3_tiles"

#: Latent rows in one H3 chunk; a time window starts on a chunk.
CHUNK_ROWS = 5

#: Video frames each latent row covers, by its place in a chunk.
ROW_FRAMES = (1, 4, 4, 4, 4)

#: Positions one video frame advances along the time axis.
FRAME_TIME = 5.0 / 3.0

#: Pixels per latent pixel in the H3 video latent.
LATENT_SCALE = 16

#: Most tokens one tile holds when tiles are planned automatically.
AUTO_TOKENS = 48000

#: Latent pixels neighbouring tiles share when planned automatically.
AUTO_OVERLAP = 12

#: Latent rows neighbouring time windows share when planned automatically.
AUTO_WINDOW_OVERLAP = 5

#: Share of a plan's work each tile across the frame beyond the first adds when planned automatically.
SPATIAL_COST = 0.5

#: Smallest blend weight at a tile's edge.
EDGE_WEIGHT = 1e-3


@dataclass(frozen=True)
class Tiling:
    """Where the tiles of an H3 latent sit, in latent rows and latent pixels.

    Attributes:
        rows: ``(start, stop)`` latent rows of each time window.
        heights: ``(start, stop)`` latent pixels of each band down the frame.
        widths: ``(start, stop)`` latent pixels of each band across the frame.
    """

    rows: tuple
    heights: tuple
    widths: tuple

    @property
    def count(self) -> int:
        """Tiles in all."""
        return len(self.rows) * len(self.heights) * len(self.widths)

    def tokens(self, extra: int = 0) -> int:
        """Tokens in the largest tile.

        Args:
            extra: Text and audio tokens every tile adds.

        Returns:
            The count.
        """
        rows = max(stop - start for start, stop in self.rows)
        height = max(stop - start for start, stop in self.heights)
        width = max(stop - start for start, stop in self.widths)
        return rows * ((height + 1) // 2) * ((width + 1) // 2) + int(extra)



def _spans(length: int, pieces: int, overlap: int, align: int) -> tuple:
    """``(start, stop)`` ranges of one size covering ``length`` in ``pieces``, one stride apart.

    Args:
        length: Extent to cover.
        pieces: Ranges wanted.
        overlap: Least extent neighbouring ranges share; every pair shares the same.
        align: Starts fall on multiples of this.

    Returns:
        The ranges.
    """
    length, overlap = int(length), max(1, int(overlap))
    pieces = min(max(1, int(pieces)), max(1, (length - overlap) // max(1, int(align))))
    if pieces == 1 or length <= align:
        return ((0, length),)
    stride = (length - overlap) // pieces // align * align
    size = length - (pieces - 1) * stride
    return tuple((index * stride, index * stride + size) for index in range(pieces))


def _pieces(length: int, tile: int, overlap: int) -> int:
    """Tiles of ``tile`` sharing ``overlap`` needed to cover ``length``."""
    if tile >= length:
        return 1
    return max(1, math.ceil((length - overlap) / max(1, tile - overlap)))


def _audio_tokens(rows: int) -> int:
    """Audio tokens over as many video frames as ``rows`` latent rows cover."""
    frames = sum(ROW_FRAMES[k % CHUNK_ROWS] for k in range(int(rows)))
    return 2 * math.ceil(frames * FRAME_TIME)


def auto_tiling(rows: int, height: int, width: int, text: int = 0, max_tokens: int = AUTO_TOKENS,
                overlap: int = AUTO_OVERLAP, window_overlap: int = AUTO_WINDOW_OVERLAP) -> Tiling:
    """The tiling with the least work, tiles across the frame weighted by ``SPATIAL_COST``, under ``max_tokens``.

    Args:
        rows: Latent rows of the clip.
        height: Latent height, even.
        width: Latent width, even.
        text: Text tokens every tile carries.
        max_tokens: Most tokens one tile may hold.
        overlap: Latent pixels neighbouring tiles share.
        window_overlap: Latent rows neighbouring time windows share.

    Returns:
        The tiling; the smallest tiles on offer when none stay under ``max_tokens``.
    """
    best, smallest = None, None
    for time_pieces in range(1, max(1, rows // CHUNK_ROWS) + 1):
        row_spans = _spans(rows, time_pieces, window_overlap, CHUNK_ROWS)
        window = max(stop - start for start, stop in row_spans)
        extra = int(text) + _audio_tokens(window)
        for down in range(1, max(1, height // (2 * overlap + 2)) + 1):
            heights = _spans(height, down, overlap, 2)
            for across in range(1, max(1, width // (2 * overlap + 2)) + 1):
                widths = _spans(width, across, overlap, 2)
                tiling = Tiling(row_spans, heights, widths)
                tokens = tiling.tokens(extra)
                spatial = len(heights) * len(widths)
                work = tokens * tiling.count * (1.0 + SPATIAL_COST * (spatial - 1))
                if smallest is None or tokens < smallest[0]:
                    smallest = (tokens, tiling)
                if tokens <= max_tokens and (best is None or (work, tiling.count) < best[0]):
                    best = ((work, tiling.count), tiling)
    if best is not None:
        return best[1]
    logger.warning("H3 tiles: no tiling keeps a tile under %d tokens; the smallest holds %d",
                   max_tokens, smallest[0])
    return smallest[1]


def manual_tiling(rows: int, height: int, width: int, tile_height: int, tile_width: int,
                  window_rows: int, overlap: int, window_overlap: int) -> Tiling:
    """The tiling given tile sizes in latent units.

    Args:
        rows: Latent rows of the clip.
        height: Latent height, even.
        width: Latent width, even.
        tile_height: Latent pixels a tile spans down the frame.
        tile_width: Latent pixels a tile spans across the frame.
        window_rows: Latent rows a time window spans, ``0`` for the whole clip.
        overlap: Latent pixels neighbouring tiles share.
        window_overlap: Latent rows neighbouring time windows share.

    Returns:
        The tiling.
    """
    window_rows = rows if window_rows <= 0 else window_rows
    return Tiling(
        _spans(rows, _pieces(rows, window_rows, window_overlap), window_overlap, CHUNK_ROWS),
        _spans(height, _pieces(height, tile_height, overlap), overlap, 2),
        _spans(width, _pieces(width, tile_width, overlap), overlap, 2),
    )


def scene_spans(starts, rows: int) -> tuple:
    """``(start, stop)`` latent rows of each scene of a clip.

    Args:
        starts: Latent rows each scene opens on, or ``None`` for one scene.
        rows: Latent rows of the clip.

    Returns:
        Spans in order covering every row, one where ``starts`` names no cut inside the clip.
    """
    rows = int(rows)
    kept = sorted({int(start) for start in starts or ()
                   if 0 < int(start) < rows and int(start) % CHUNK_ROWS == 0})
    edges = [0, *kept, rows]
    return tuple((edges[index], edges[index + 1]) for index in range(len(edges) - 1))


def _fitted_spans(length: int, window: int, overlap: int) -> tuple:
    """The fewest equal ``(start, stop)`` ranges covering ``length`` with none longer than ``window``."""
    most = max(1, (int(length) - max(1, int(overlap))) // CHUNK_ROWS)
    for pieces in range(1, most + 1):
        spans = _spans(length, pieces, overlap, CHUNK_ROWS)
        if max(stop - start for start, stop in spans) <= window:
            break
    return spans


def scene_tiling(tiling: Tiling, settings: tuple, starts, rows: int) -> Tiling:
    """A tiling whose time windows each lie inside one scene, none longer than its longest.

    Args:
        tiling: A tiling of the whole clip.
        settings: The settings it was planned from, as :func:`tiling_settings` gives them.
        starts: Latent rows each scene opens on, or ``None`` for one scene.
        rows: Latent rows of the clip.

    Returns:
        The tiling, unchanged where it holds one time window or the clip one scene.
    """
    scenes = scene_spans(starts, rows)
    if len(tiling.rows) < 2 or len(scenes) < 2:
        return tiling
    overlap = settings[5] if settings[0] == "manual" else AUTO_WINDOW_OVERLAP
    window = max(stop - start for start, stop in tiling.rows)
    windows = []
    for start, stop in scenes:
        windows.extend((start + a, start + b) for a, b in _fitted_spans(stop - start, window, overlap))
    return Tiling(tuple(windows), tiling.heights, tiling.widths)


def planned(settings: tuple, rows: int, height: int, width: int, text: int = 0,
            starts=None) -> Tiling:
    """The tiling a run plans for a latent of this size.

    Args:
        settings: ``("auto", max_tokens)`` or ``("manual", tile_height, tile_width, window_rows,
            overlap, window_overlap)`` in latent units.
        rows: Latent rows of the clip.
        height: Latent height, even.
        width: Latent width, even.
        text: Text tokens every tile carries.
        starts: Latent rows each scene opens on, or ``None`` for one scene.

    Returns:
        The tiling.
    """
    if settings[0] == "manual":
        _, tile_h, tile_w, window_rows, overlap, window_overlap = settings
        tiling = manual_tiling(rows, height, width, tile_h, tile_w, window_rows, overlap,
                               window_overlap)
    else:
        tiling = auto_tiling(rows, height, width, text, max_tokens=settings[1])
    return scene_tiling(tiling, settings, starts, rows)


def _smooth(weight: torch.Tensor) -> torch.Tensor:
    """Raised-cosine form of a linear ramp in ``[0, 1]``; two ramps summing to 1 still do."""
    return 0.5 - 0.5 * torch.cos(math.pi * weight.clamp(0.0, 1.0))


def _ramp(spans: tuple, index: int) -> torch.Tensor:
    """Blend weights along one axis for one tile: raised-cosine ramps across the overlaps it shares."""
    start, stop = spans[index]
    size = stop - start
    position = torch.arange(size, dtype=torch.float32)
    weight = torch.ones(size, dtype=torch.float32)
    if index > 0:
        left = spans[index - 1][1] - start
        if left > 0:
            weight = torch.minimum(weight, (position + 1) / (left + 1))
    if index < len(spans) - 1:
        right = stop - spans[index + 1][0]
        if right > 0:
            weight = torch.minimum(weight, (size - position) / (right + 1))
    return _smooth(weight).clamp(min=EDGE_WEIGHT)


def _row_times(rows: int) -> torch.Tensor:
    """Start time of every latent row, then the end of the last, on the H3 time axis."""
    spans = [FRAME_TIME * ROW_FRAMES[k % CHUNK_ROWS] for k in range(int(rows))]
    return torch.tensor([0.0] + spans, dtype=torch.float64).cumsum(0)


def _time_weight(times: torch.Tensor, window: tuple, previous: tuple | None,
                 following: tuple | None) -> torch.Tensor:
    """Blend weights at ``times`` for a window spanning ``window``, ramped across shared time."""
    start, stop = window
    weight = torch.ones_like(times, dtype=torch.float64)
    if previous is not None and previous[1] > start:
        weight = torch.minimum(weight, (times - start) / (previous[1] - start))
    if following is not None and stop > following[0]:
        weight = torch.minimum(weight, (stop - times) / (stop - following[0]))
    return _smooth(weight).clamp(min=EDGE_WEIGHT, max=1.0).to(torch.float32)


def _pad_plane(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """``tensor`` with its last two axes grown to ``height`` by ``width``, edges repeated."""
    pad_h, pad_w = max(0, height - tensor.shape[-2]), max(0, width - tensor.shape[-1])
    if not (pad_h or pad_w):
        return tensor
    if tensor.ndim == 5:
        return torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h, 0, 0), mode="replicate")
    return torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")


def _resize_plane(latent: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """A ``[B, C, T, H, W]`` latent resized across its frames to ``height`` by ``width``."""
    if tuple(latent.shape[-2:]) == (height, width):
        return latent
    batch, channels, frames = latent.shape[:3]
    grow = height * width > latent.shape[-2] * latent.shape[-1]
    flat = latent.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, *latent.shape[-2:])
    resized = torch.nn.functional.interpolate(
        flat.float(), size=(height, width), mode="bicubic" if grow else "bilinear",
        align_corners=False, antialias=not grow,
    ).to(latent.dtype)
    return resized.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)


def _segments(layout, kind: str) -> list:
    """``(start, stop)`` of each segment of ``kind`` in a packed layout, in order."""
    return [(start, stop) for start, stop, name in layout.segments if name == kind]


def _crop_keyframes(keyframes, frames: tuple, heights: tuple, widths: tuple, size: tuple,
                    padded: tuple):
    """The keyframes inside a window, their latents resized to the video's latent and cut to the tile.

    Args:
        keyframes: The payload's keyframes, or None.
        frames: ``(first, stop)`` video frames of the window.
        heights: ``(start, stop)`` latent pixels of the tile down the frame.
        widths: ``(start, stop)`` latent pixels of the tile across the frame.
        size: ``(height, width)`` of the video latent.
        padded: ``(height, width)`` of the video latent padded to whole patches.

    Returns:
        The kept keyframes, each a copy.
    """
    kept = []
    for keyframe in keyframes or ():
        index = int(keyframe.get("resolved_frame_index", 0))
        if not frames[0] <= index < frames[1]:
            continue
        entry = dict(keyframe)
        latent = keyframe.get("latent")
        if latent is not None and latent.ndim == 5:
            latent = _pad_plane(_resize_plane(latent, *size), *padded)
            entry["latent"] = latent[..., heights[0]:heights[1], widths[0]:widths[1]]
        kept.append(entry)
    return kept


def _tile_layout(layout_class, full, text: int, rows: tuple, heights: tuple, widths: tuple,
                 audio: tuple, keyframes, refs, grid: tuple):
    """A packed layout for one tile, every position taken from the whole clip's layout."""
    tile = layout_class(text, rows[1] - rows[0], heights[1] - heights[0], widths[1] - widths[0],
                        audio[1] - audio[0], keyframes=keyframes or None, refs=refs)
    position = tile.position_ids.clone()
    full_position = full.position_ids
    for kind in ("text", "ref_img", "ref_audio"):
        for (a, b), (c, d) in zip(_segments(tile, kind), _segments(full, kind)):
            if b - a == d - c:
                position[a:b] = full_position[c:d]
    frames, latent_h, latent_w = grid
    (video_start, video_stop), = _segments(full, "video")
    video = full_position[video_start:video_stop].reshape(frames, latent_h // 2, latent_w // 2, 3)
    video = video[rows[0]:rows[1], heights[0] // 2:heights[1] // 2, widths[0] // 2:widths[1] // 2]
    (a, b), = _segments(tile, "video")
    position[a:b] = video.reshape(-1, 3)
    plane = video[0, :, :, 1:].reshape(-1, 2)
    for a, b in _segments(tile, "cond"):
        count = (b - a) // plane.shape[0]
        position[a:b, 1:] = plane.repeat(count, 1)
    (audio_start, audio_stop), = _segments(full, "audio")
    full_audio = full_position[audio_start:audio_stop]
    length = full_audio.shape[0] // 2
    low, high = float(full_audio[0, 2]), float(full_audio[length, 2])
    (a, b), = _segments(tile, "audio")
    position[a:b] = torch.cat([full_audio[audio[0]:audio[1]],
                               full_audio[length + audio[0]:length + audio[1]]])
    for a, b in _segments(tile, "cond_audio"):
        half = (b - a) // 2
        position[a:a + half, 2] = low
        position[a + half:b, 2] = high
    tile.position_ids = position
    return tile


def _crop_mask(mask, rows: tuple, heights: tuple, widths: tuple, padded: tuple):
    """A video denoise mask cut to one tile, padded first as the model pads the video."""
    if not isinstance(mask, torch.Tensor) or mask.ndim < 5:
        return mask
    if mask.shape[-2] > 1 and mask.shape[-1] > 1:
        mask = _pad_plane(mask, *padded)
    if mask.shape[2] > 1:
        mask = mask[:, :, rows[0]:rows[1]]
    if mask.shape[3] > 1:
        mask = mask[:, :, :, heights[0]:heights[1]]
    if mask.shape[4] > 1:
        mask = mask[..., widths[0]:widths[1]]
    return mask


def _longest_prompt(guider) -> int:
    """Most text tokens among a guider's conditionings, ``0`` when none can be read."""
    longest = 0
    for conds in (getattr(guider, "conds", None) or {}).values():
        for cond in conds or ():
            cross = cond.get("cross_attn") if isinstance(cond, dict) else None
            if isinstance(cross, torch.Tensor) and cross.ndim >= 2:
                longest = max(longest, int(cross.shape[1]))
    return longest


class _Tiled:
    """A diffusion model wrapper that runs each H3 call over overlapping tiles and blends them.

    Attributes:
        settings: ``("auto", max_tokens)`` or ``("manual", tile_height, tile_width, window_rows,
            overlap, window_overlap)`` in latent units.
        text: Most text tokens among the current run's conditionings.
        plans: Tilings planned in the current run, by latent shape.
        reported: Plans already logged.
        starts: Latent rows each scene of the clip opens on.
    """

    def __init__(self, settings: tuple, starts=None):
        self.settings = settings
        self.text = 0
        self.plans = {}
        self.reported = set()
        self.starts = tuple(int(start) for start in starts or ())

    def sample(self, executor, *args, **kwargs):
        """Runs a sampling run with its tilings planned afresh from the run's longest prompt."""
        self.plans, self.text = {}, _longest_prompt(getattr(executor, "class_obj", None))
        return executor(*args, **kwargs)

    def tiling(self, rows: int, height: int, width: int, text: int) -> Tiling:
        """The tiling for a latent of this size."""
        return planned(self.settings, rows, height, width, text, self.starts)

    def plan(self, frames: int, height: int, width: int, text: int) -> Tiling:
        """The run's tiling for a latent of this size, planned once and shared by every call."""
        key = (frames, height, width)
        plan = self.plans.get(key)
        if plan is None:
            text = max(int(text), self.text)
            plan = self.plans[key] = self.tiling(frames, height, width, text)
            if (key, plan) not in self.reported:
                self.reported.add((key, plan))
                logger.info("H3 tiles: %d tiles over %dx%dx%d latent (%d window(s), %dx%d across), "
                            "largest %d tokens", plan.count, frames, height, width,
                            len(plan.rows), len(plan.heights), len(plan.widths),
                            plan.tokens(text + _audio_tokens(max(b - a for a, b in plan.rows))))
                note = cut_report(self.starts, frames, len(plan.rows))
                if note:
                    logger.info("H3 tiles: %s", note)
        return plan

    def __call__(self, executor, *args, **kwargs):
        from comfy.ldm.minimax.model import PackedLayout

        args = list(args)
        x, timestep, context = args[0], args[1], args[2]
        streams = list(getattr(x, "tensors", None) or (x if isinstance(x, (list, tuple)) else ()))
        if len(streams) < 2 or streams[0].ndim != 5:
            return executor(*args, **kwargs)
        video, audio = streams[0], streams[1]
        frames, height, width = video.shape[2], video.shape[3], video.shape[4]
        padded = _pad_plane(video, height + height % 2, width + width % 2)
        latent_h, latent_w = padded.shape[3], padded.shape[4]
        text = int(context.shape[1])
        plan = self.plan(frames, latent_h, latent_w, text)
        if plan.count == 1:
            if padded is video:
                return executor(*args, **kwargs)
            out = executor([padded, audio] + streams[2:], timestep, context, *args[3:], **kwargs)
            return [out[0][..., :frames, :height, :width], out[1]]

        payload = dict(kwargs.get("minimax_payload") or {})
        keyframes, refs = payload.get("keyframes"), payload.get("refs")
        length = audio.shape[-1]
        full = payload.get("layout")
        if full is None or getattr(full, "signature", None) != (text, frames, latent_h, latent_w, length):
            full = PackedLayout(text, frames, latent_h, latent_w, length, keyframes=keyframes, refs=refs)
        times = _row_times(frames)
        frame_starts = [0]
        for k in range(frames):
            frame_starts.append(frame_starts[-1] + ROW_FRAMES[k % CHUNK_ROWS])
        windows = [(float(times[a]), float(times[b])) for a, b in plan.rows]
        row_centres = (times[:-1] + times[1:]) / 2
        audio_centres = torch.arange(length, dtype=torch.float64) + 0.5
        mask = kwargs.get("denoise_mask")
        audio_mask = kwargs.get("audio_denoise_mask")
        ref_video = [r["latent"] for r in refs or () if "latent" in r]
        ref_audio = [r["audio_latent"] for r in refs or () if r.get("audio_latent") is not None]

        video_sum = video_weight = audio_sum = audio_weight = None
        for t_index, rows in enumerate(plan.rows):
            previous = windows[t_index - 1] if t_index > 0 else None
            following = windows[t_index + 1] if t_index < len(windows) - 1 else None
            row_weight = _time_weight(row_centres[rows[0]:rows[1]], windows[t_index], previous, following)
            first = max(0, int(math.floor(windows[t_index][0])))
            last = min(length, int(math.ceil(windows[t_index][1])))
            if t_index == len(plan.rows) - 1:
                last = length
            if t_index == 0:
                first = 0
            first = min(first, length - 1)
            last = max(last, first + 1)
            sound_weight = _time_weight(audio_centres[first:last], windows[t_index], previous, following)
            frame_range = (frame_starts[rows[0]], frame_starts[rows[1]])
            for h_index, heights in enumerate(plan.heights):
                for w_index, widths in enumerate(plan.widths):
                    kept = _crop_keyframes(keyframes, frame_range, heights, widths, (height, width),
                                           (latent_h, latent_w))
                    tile_payload = dict(payload)
                    tile_payload["layout"] = _tile_layout(
                        PackedLayout, full, text, rows, heights, widths, (first, last), kept, refs,
                        (frames, latent_h, latent_w))
                    if keyframes is not None:
                        tile_payload["keyframes"] = kept
                    if "cond_video_latents" in payload:
                        tile_payload["cond_video_latents"] = (
                            [k["latent"] for k in kept if k.get("latent") is not None] + ref_video)
                    if "cond_audio_latents" in payload:
                        tile_payload["cond_audio_latents"] = (
                            [k["audio_latent"] for k in kept if k.get("audio_latent") is not None]
                            + ref_audio)
                    tile_kwargs = dict(kwargs)
                    tile_kwargs["minimax_payload"] = tile_payload
                    if mask is not None:
                        tile_kwargs["denoise_mask"] = _crop_mask(mask, rows, heights, widths,
                                                                 (latent_h, latent_w))
                    if isinstance(audio_mask, torch.Tensor) and audio_mask.shape[-1] == length:
                        tile_kwargs["audio_denoise_mask"] = audio_mask[..., first:last]
                    tile_x = [padded[:, :, rows[0]:rows[1], heights[0]:heights[1], widths[0]:widths[1]],
                              audio[..., first:last]] + streams[2:]
                    out = executor(tile_x, timestep, context, *args[3:], **tile_kwargs)
                    weight = (row_weight[:, None, None]
                              * _ramp(plan.heights, h_index)[None, :, None]
                              * _ramp(plan.widths, w_index)[None, None, :]).to(out[0].device)
                    if video_sum is None:
                        video_sum = torch.zeros(padded.shape, dtype=torch.float32, device=out[0].device)
                        video_weight = torch.zeros(padded.shape[2:], dtype=torch.float32,
                                                   device=out[0].device)
                        audio_sum = torch.zeros(audio.shape, dtype=torch.float32, device=out[1].device)
                        audio_weight = torch.zeros(length, dtype=torch.float32, device=out[1].device)
                    region = (slice(None), slice(None), slice(rows[0], rows[1]),
                              slice(heights[0], heights[1]), slice(widths[0], widths[1]))
                    video_sum[region] += out[0].float() * weight
                    video_weight[region[2:]] += weight
                    sound = sound_weight.to(out[1].device)
                    audio_sum[..., first:last] += out[1].float() * sound
                    audio_weight[first:last] += sound
                    dtypes = (out[0].dtype, out[1].dtype)
                    del out
        video_out = (video_sum / video_weight)[..., :frames, :height, :width]
        audio_out = audio_sum / audio_weight
        return [video_out.to(dtypes[0]), audio_out.to(dtypes[1])]


def _rows(frames: int) -> int:
    """Latent rows covering ``frames`` video frames."""
    return max(1, math.ceil(int(frames) * CHUNK_ROWS / 17))


def tiling_settings(choice: dict) -> tuple:
    """Settings for :func:`tiled_model` from the tiling widget's value.

    Args:
        choice: The widget's value, its option under ``tiling`` and that option's inputs.

    Returns:
        ``("auto", max_tokens)`` or ``("manual", tile_height, tile_width, window_rows, overlap,
        window_overlap)`` in latent units.
    """
    def latent(pixels):
        return max(2, -(-int(pixels) // (2 * LATENT_SCALE)) * 2)

    if choice.get("tiling") == "manual":
        window = int(choice.get("window_frames", 0))
        return ("manual", latent(choice.get("tile_height", 576)), latent(choice.get("tile_width", 1024)),
                _rows(window) if window > 0 else 0, latent(choice.get("overlap", 128)),
                _rows(choice.get("window_overlap", 17)))
    return ("auto", int(choice.get("max_tokens", AUTO_TOKENS)))


def tiled_model(model, settings: tuple, starts=None):
    """A low VRAM clone of an H3 model whose every call runs over overlapping tiles, blended per step.

    Args:
        model: A MiniMax H3 model patcher.
        settings: ``("auto", max_tokens)`` or ``("manual", tile_height, tile_width, window_rows,
            overlap, window_overlap)`` in latent units.
        starts: Latent rows each scene of the clip opens on, or ``None`` for one scene.

    Returns:
        The patched clone.
    """
    import comfy.patcher_extension

    from . import h3_low_vram

    kind = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    outer = comfy.patcher_extension.WrappersMP.OUTER_SAMPLE
    patched = model.clone() if h3_low_vram.is_low_vram(model) else h3_low_vram.low_vram_model(model)
    tiles = _Tiled(tuple(settings), starts)
    for wrapper_kind in (kind, outer):
        patched.wrappers.get(wrapper_kind, {}).pop(TILES_WRAPPER, None)
    patched.add_wrapper_with_key(kind, TILES_WRAPPER, tiles)
    patched.add_wrapper_with_key(outer, TILES_WRAPPER, tiles.sample)
    wrappers = patched.wrappers[kind]
    patched.wrappers[kind] = {TILES_WRAPPER: wrappers.pop(TILES_WRAPPER), **wrappers}
    return patched


def plot(tiling: Tiling, rows: int, height: int, width: int, size: int = 960,
         starts=None) -> torch.Tensor:
    """A picture of a tiling: the tiles over one frame, and the time windows along the clip.

    Args:
        tiling: The tiling.
        rows: Latent rows of the clip.
        height: Latent height.
        width: Latent width.
        size: Picture width in pixels.
        starts: Latent rows each scene opens on, or ``None`` for one scene; cuts are marked.

    Returns:
        An IMAGE tensor ``[1, H, W, 3]``.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    frame_w = size - 40
    frame_h = max(60, int(frame_w * height / max(1, width)))
    picture = Image.new("RGB", (size, frame_h + 110), (18, 20, 26))
    draw = ImageDraw.Draw(picture)
    colours = [(214, 170, 72), (90, 170, 230), (230, 110, 90), (120, 200, 130)]
    top, left = 30, 20
    for h_index, (h0, h1) in enumerate(tiling.heights):
        for w_index, (w0, w1) in enumerate(tiling.widths):
            colour = colours[(h_index + w_index) % len(colours)]
            box = [left + w0 / width * frame_w, top + h0 / height * frame_h,
                   left + w1 / width * frame_w - 1, top + h1 / height * frame_h - 1]
            draw.rectangle(box, outline=colour, width=3)
    bar = top + frame_h + 30
    for index, (r0, r1) in enumerate(tiling.rows):
        colour = colours[index % len(colours)]
        lane = bar + (index % 2) * 18
        draw.rectangle([left + r0 / rows * frame_w, lane, left + r1 / rows * frame_w - 1, lane + 12],
                       fill=colour)
    for cut, _ in scene_spans(starts, rows)[1:]:
        x = left + cut / rows * frame_w
        draw.line([(x, bar - 6), (x, bar + 32)], fill=(236, 236, 242), width=2)
    note = cut_report(starts, rows, len(tiling.rows))
    if note:
        draw.text((left, bar + 34), note, fill=(150, 154, 165))
    draw.text((left, 8), f"{tiling.count} tiles: {len(tiling.rows)} time window(s), "
              f"{len(tiling.heights)} x {len(tiling.widths)} across the frame, "
              f"{width * LATENT_SCALE}x{height * LATENT_SCALE} px", fill=(230, 230, 235))
    return torch.from_numpy(np.asarray(picture, dtype=np.float32) / 255.0)[None]


def cut_report(starts, rows: int, windows: int) -> str:
    """One line saying how a tiling treats the clip's cuts.

    Args:
        starts: Latent rows each scene opens on, or ``None`` for one scene.
        rows: Latent rows of the clip.
        windows: Time windows the tiling holds.

    Returns:
        The line, empty for a clip of one scene.
    """
    cuts = len(scene_spans(starts, rows)) - 1
    if not cuts:
        return ""
    count = f"{cuts} {'cut' if cuts == 1 else 'cuts'}"
    if windows > 1:
        return f"{count} kept out of the window blend, every time window inside one scene"
    return f"{count} inside the one time window, sampled whole"

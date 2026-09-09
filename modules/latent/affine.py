"""The affine ``y = x * ((1 - m) + m * scale) + bias * m`` over a latent.

``m`` comes from :mod:`modules.latent.affine_patterns`. A latent is 3D, 4D, 5D or a
NestedTensor of them.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from ..util.easing import EASING_NAMES, ease
from . import affine_patterns as patterns
from .filters import gaussian_blur_depthwise

__all__ = [
    "BIAS_FIELDS",
    "CONTENT_GATES",
    "DEFAULTS",
    "SCHEDULE_DEFAULTS",
    "STREAM_MODES",
    "TEMPORAL_MODES",
    "apply_affine",
    "apply_plane",
    "content_pattern_fits",
    "mask_like",
    "resolve",
    "step_schedule",
    "stream_indices",
    "streams_of",
]

#: Every pattern parameter and mask shaping value, with the value used when none is given.
DEFAULTS = {
    "mask_strength": 1.0,
    "threshold": 0.0,
    "invert_mask": False,
    "mask_blur": 0.0,
    "mask_sharpen": 0.0,
    "sharpen_radius": 0.8,
    "sharpen_threshold": 0.0,
    "clamp": False,
    "clamp_min": -10.0,
    "clamp_max": 10.0,
    "frame_seed_stride": 9973,
    "drift_speed": 0.35,
    "drift_angle_deg": 0.0,
    "drift_renew": 0.0,
    "perlin_scale": 64.0,
    "perlin_octaves": 3,
    "perlin_persistence": 0.5,
    "perlin_lacunarity": 2.0,
    "checker_size": 8,
    "bayer_size": 8,
    "velvet_taps_per_kpx": 10,
    "green_center_frac": 0.35,
    "green_bandwidth_frac": 0.15,
    "black_bins_per_kpx": 512,
    "hatch_freq_cyc_px": 0.45,
    "hatch_angle1_deg": 0.0,
    "hatch_angle2_deg": 90.0,
    "hatch_square": False,
    "hatch_phase_jitter": 0.0,
    "hatch_supersample": 1,
    "highpass_cutoff_frac": 0.7,
    "highpass_order": 2,
    "ring_center_frac": 0.9,
    "ring_bandwidth_frac": 0.05,
    "poisson_radius_px": 8.0,
    "poisson_softness": 6.0,
    "worley_points_per_kpx": 2.0,
    "worley_metric": "L2",
    "worley_edge_sharpness": 1.0,
    "tile_line_tile_size": 32,
    "tile_line_freq_cyc_px": 0.4,
    "tile_line_jitter": 0.25,
    "dot_cell_size": 12,
    "dot_jitter_px": 1.5,
    "dot_fill_ratio": 0.3,
    "content_window": 7,
    "content_gate": "off",
    "bias_field": "constant",
    "solid_alpha": 1.0,
}

#: How the schedule reads when an affine node is given none.
SCHEDULE_DEFAULTS = {
    "start": 0.2,
    "end": 0.8,
    "bias": 0.5,
    "exponent": 1.0,
    "start_offset": 0.0,
    "end_offset": 0.0,
    "curve": "ease_in_out_sine",
}

#: Which streams of a packed latent an affine reaches.
STREAM_MODES = ["video", "audio", "both"]

#: How a video latent's mask varies from frame to frame.
TEMPORAL_MODES = ["static", "per_frame", "drift"]

#: What a generated pattern may be gated by, as the option list offers it.
CONTENT_GATES = ["off", *patterns.CONTENT_PATTERNS]

#: What the bias adds where the mask is white.
BIAS_FIELDS = ["constant", "gaussian"]


def resolve(*layers) -> dict:
    """Merge option dictionaries over :data:`DEFAULTS`, later layers winning.

    Args:
        *layers: Mappings of option name to value. Anything that is not a mapping is
            skipped, and a key outside :data:`DEFAULTS` is carried through untouched.

    Returns:
        A full option dictionary holding every key in :data:`DEFAULTS`.
    """
    out = dict(DEFAULTS)
    for layer in layers:
        if isinstance(layer, dict):
            out.update(layer)
    return out


def step_schedule(steps: int, schedule: dict | None) -> list[float]:
    """Build the per-step strength curve an affine follows through a sampler run.

    Args:
        steps: How many steps the curve spans.
        schedule: ``start``, ``end``, ``bias``, ``exponent``, ``start_offset``,
            ``end_offset`` and ``curve``. Missing keys take :data:`SCHEDULE_DEFAULTS`.

    Returns:
        One value per step, each 0.0 to 1.0.
    """
    values = dict(SCHEDULE_DEFAULTS)
    if isinstance(schedule, dict):
        values.update(schedule)

    if steps <= 0:
        return []

    start = min(max(float(values["start"]), 0.0), 1.0)
    end = min(max(float(values["end"]), 0.0), 1.0)
    if start > end:
        start, end = end, start
    bias = float(values["bias"])
    exponent = max(0.0, float(values["exponent"]))
    start_offset = float(values["start_offset"])
    end_offset = float(values["end_offset"])
    curve = str(values["curve"])
    if curve not in EASING_NAMES:
        curve = SCHEDULE_DEFAULTS["curve"]

    last = max(steps - 1, 0)
    mid = start + bias * (end - start)
    start_i, mid_i, end_i = (int(round(point * last)) for point in (start, mid, end))

    out = [0.0] * steps

    # Each end of the window is read one place inside it.
    if mid_i >= start_i:
        count = mid_i - start_i + 1
        for n in range(count):
            t = (n + 1) / count
            value = ease(curve, t) ** exponent
            out[start_i + n] = value * (1.0 - start_offset) + start_offset

    if end_i >= mid_i:
        count = end_i - mid_i + 1
        for n in range(count):
            t = (count - n) / count
            value = ease(curve, t) ** exponent
            out[mid_i + n] = value * (1.0 - end_offset) + end_offset

    for i in range(min(start_i, steps)):
        out[i] = start_offset
    for i in range(end_i + 1, steps):
        out[i] = end_offset
    return out


def streams_of(samples) -> list[torch.Tensor]:
    """Split a latent into its streams.

    Args:
        samples: A tensor, or a NestedTensor packing several.

    Returns:
        One tensor per stream. A plain tensor answers a list of one.
    """
    if getattr(samples, "is_nested", False):
        return list(samples.unbind())
    return [samples]


def stream_indices(mode: str, count: int) -> list[int]:
    """Which streams of a latent an affine setting reaches.

    Args:
        mode: ``"video"`` for stream 0, ``"audio"`` for the rest, ``"both"`` for all.
        count: How many streams the latent has.

    Returns:
        Stream indices, lowest first. Empty where the mode selects none, which is
        ``"audio"`` against a latent that carries only one stream.
    """
    name = str(mode or "video").lower()
    if count <= 1:
        return [0] if name in ("video", "both") else []
    if name == "video":
        return [0]
    if name == "audio":
        return list(range(1, count))
    return list(range(count))


def mask_like(samples) -> torch.Tensor | None:
    """An all-zero MASK shaped the way an affine over this latent would answer one.

    Args:
        samples: A tensor, or a NestedTensor whose first stream is measured.

    Returns:
        A ``[N, H, W]`` mask, or None where the latent has no shape a mask can follow.
    """
    first = streams_of(samples)[0]
    if not torch.is_tensor(first):
        return None
    if first.ndim == 5:
        b, _c, t, h, w = first.shape
        return torch.zeros((b * t, h, w), dtype=first.dtype, device=first.device)
    if first.ndim == 4:
        b, _c, h, w = first.shape
        return torch.zeros((b, h, w), dtype=first.dtype, device=first.device)
    if first.ndim == 3:
        b, _c, length = first.shape
        return torch.zeros((b, 1, length), dtype=first.dtype, device=first.device)
    return None


def _shape_mask(mask: torch.Tensor, params: dict) -> torch.Tensor:
    """Sharpen, threshold, invert, blur and scale a raw field into the mask.

    Args:
        mask: A ``[N, 1, H, W]`` field.
        params: Resolved options.

    Returns:
        The shaped mask, the same shape, held inside 0.0 to 2.0.
    """
    out = patterns.unsharp(
        mask,
        float(params["sharpen_radius"]),
        float(params["mask_sharpen"]),
        float(params["sharpen_threshold"]),
    )
    threshold = float(params["threshold"])
    if threshold > 0.0:
        out = (out >= threshold).to(out.dtype)
    if bool(params["invert_mask"]):
        out = 1.0 - out
    blur = float(params["mask_blur"])
    if blur > 0.0:
        out = gaussian_blur_depthwise(out, blur).clamp(0.0, 1.0)
    return (out * float(params["mask_strength"])).clamp(0.0, 2.0)


def _external_plane(
    external: torch.Tensor, height: int, width: int, device, dtype, params: dict
) -> torch.Tensor:
    """Resize a supplied MASK onto the latent grid and shape it.

    Args:
        external: A ``[N, H, W]`` mask, or a ``[N, H, W, C]`` image read as one.
        height: Latent height.
        width: Latent width.
        device: Device to answer on.
        dtype: Dtype to answer in.
        params: Resolved options.

    Returns:
        A ``[N, 1, height, width]`` shaped mask.

    Raises:
        ValueError: The tensor is neither a mask nor an image.
    """
    if external.ndim == 4:
        planes = external.mean(dim=-1, keepdim=True).permute(0, 3, 1, 2)
    elif external.ndim == 3:
        planes = external.unsqueeze(1)
    else:
        raise ValueError(
            "external_mask must be a MASK shaped [N, H, W]; "
            f"a tensor with {external.ndim} dimensions was given"
        )
    planes = planes.to(device=device, dtype=torch.float32)
    planes = F.interpolate(planes, size=(height, width), mode="bilinear", align_corners=False)
    return _shape_mask(planes.to(dtype), params)


def _fit_batch(mask: torch.Tensor, batch: int) -> torch.Tensor:
    """Repeat or trim a mask's leading dimension to match a batch.

    Args:
        mask: A mask whose first dimension is its batch.
        batch: How many entries are wanted.

    Returns:
        The mask with exactly ``batch`` entries.
    """
    if mask.shape[0] == batch:
        return mask
    if mask.shape[0] == 1:
        return mask.repeat(batch, *([1] * (mask.ndim - 1)))
    return mask[:1].repeat(batch, *([1] * (mask.ndim - 1)))


def _generated_plane(
    height: int, width: int, pattern: str, params: dict, seed: int, device, dtype
) -> torch.Tensor:
    """One procedural field, shaped and moved onto the latent.

    Args:
        height: Latent height.
        width: Latent width.
        pattern: A generated pattern name.
        params: Resolved options.
        seed: Seeds the field.
        device: Device to answer on.
        dtype: Dtype to answer in.

    Returns:
        A ``[1, 1, height, width]`` shaped mask.
    """
    raw = patterns.field(pattern, height, width, seed, params).view(1, 1, height, width)
    return _shape_mask(raw.to(device=device, dtype=dtype), params)


def _slide(plane: torch.Tensor, down: float, right: float) -> torch.Tensor:
    """Move a field by a fractional number of samples, wrapping at the edges.

    Args:
        plane: A field whose last two dimensions are height and width.
        down: Samples to move it down by.
        right: Samples to move it right by.

    Returns:
        The moved field, the same shape.
    """
    whole_y, whole_x = int(math.floor(down)), int(math.floor(right))
    part_y, part_x = float(down - whole_y), float(right - whole_x)
    corner = [
        torch.roll(plane, shifts=(whole_y + step_y, whole_x + step_x), dims=(-2, -1))
        for step_y in (0, 1)
        for step_x in (0, 1)
    ]
    top = corner[0] * (1.0 - part_x) + corner[1] * part_x
    bottom = corner[2] * (1.0 - part_x) + corner[3] * part_x
    return top * (1.0 - part_y) + bottom * part_y


def _drifting_planes(
    height: int, width: int, pattern: str, params: dict, seed: int, frames: int
) -> torch.Tensor:
    """A field per frame, slid across the frame and partly renewed as the clip runs.

    Args:
        height: Latent height.
        width: Latent width.
        pattern: A generated pattern name.
        params: Resolved options.
        seed: Seeds the first field.
        frames: How many frames to build.

    Returns:
        A ``[frames, 1, height, width]`` stack of raw fields, scaled to 0.0 to 1.0.
    """
    speed = float(params["drift_speed"])
    angle = math.radians(float(params["drift_angle_deg"]))
    renew = min(max(float(params["drift_renew"]), 0.0), 1.0)
    kept, fresh = math.sqrt(1.0 - renew), math.sqrt(renew)
    stride = int(params["frame_seed_stride"])
    step_down, step_right = speed * math.sin(angle), speed * math.cos(angle)

    carried = patterns.field(pattern, height, width, seed, params)
    out = []
    for index in range(frames):
        if index and renew > 0.0:
            arriving = patterns.field(pattern, height, width, seed + index * stride, params)
            carried = (0.5 + kept * (carried - 0.5) + fresh * (arriving - 0.5)).clamp(0.0, 1.0)
        moved = _slide(carried, step_down * index, step_right * index)
        out.append(moved.view(1, 1, height, width))
    return torch.cat(out, dim=0)


def _gate_name(params: dict) -> str | None:
    """The content pattern a generated field is gated by.

    Args:
        params: Resolved options.

    Returns:
        A name from :data:`modules.latent.affine_patterns.CONTENT_PATTERNS`, or None where
        no gate is asked for.
    """
    name = str(params.get("content_gate", "off"))
    return name if name in patterns.CONTENT_PATTERNS else None


def _content_gate(source: torch.Tensor, name: str, params: dict) -> torch.Tensor:
    """A reading of the picture, for multiplying a generated field by.

    Args:
        source: A ``[N, C, H, W]`` latent or clean estimate.
        name: A content pattern name.
        params: Resolved options.

    Returns:
        An ``[N, 1, H, W]`` field, scaled 0.0 to 1.0.
    """
    return patterns.content_field(source, name, int(params["content_window"]))


def _mask_for_plane(
    x: torch.Tensor,
    pattern: str,
    params: dict,
    seed: int,
    external: torch.Tensor | None,
    content: torch.Tensor | None = None,
) -> torch.Tensor:
    """The mask a 4D latent plane is transformed through.

    Args:
        x: A ``[B, C, H, W]`` latent.
        pattern: The pattern name.
        params: Resolved options.
        seed: Seeds the field.
        external: A supplied mask, used alone for ``external_mask`` and as a gate otherwise.
        content: What a content-aware pattern reads instead of ``x``, the same shape.

    Returns:
        A ``[B, 1, H, W]`` mask.
    """
    b, _c, h, w = x.shape
    device, dtype = x.device, x.dtype

    if pattern == "external_mask":
        if external is None:
            raise ValueError(
                "pattern is 'external_mask' but no external_mask was connected. "
                "Wire a MASK into external_mask, or pick a generated pattern."
            )
        return _fit_batch(_external_plane(external, h, w, device, dtype, params), b)

    if pattern in patterns.CONTENT_PATTERNS:
        source = content if content is not None else x
        mask = patterns.content_field(source, pattern, int(params["content_window"]))
        mask = _shape_mask(mask, params)
    else:
        mask = _generated_plane(h, w, pattern, params, seed, device, dtype).repeat(b, 1, 1, 1)
        gate = _gate_name(params)
        if gate is not None and content_pattern_fits(x):
            source = content if content is not None else x
            mask = mask * _content_gate(source, gate, params).to(device=device, dtype=dtype)

    if external is not None:
        mask = mask * _fit_batch(_external_plane(external, h, w, device, dtype, params), b)
    return mask


def _mask_for_frames(
    x: torch.Tensor,
    pattern: str,
    params: dict,
    seed: int,
    external: torch.Tensor | None,
    temporal_mode: str,
    content: torch.Tensor | None = None,
) -> torch.Tensor:
    """The mask a 5D latent is transformed through.

    Args:
        x: A ``[B, C, T, H, W]`` latent.
        pattern: The pattern name.
        params: Resolved options.
        seed: Seeds the field.
        external: A supplied mask. One entry covers every frame, T entries one each.
        temporal_mode: ``"static"`` for one field across the clip, ``"per_frame"`` to
            rebuild it for each frame, ``"drift"`` to slide one field across the frame and
            renew part of it as the clip runs.
        content: What a content-aware pattern reads instead of ``x``, the same shape.

    Returns:
        A ``[B, 1, T, H, W]`` mask.
    """
    b, c, t, h, w = x.shape
    device, dtype = x.device, x.dtype
    stride = int(params["frame_seed_stride"])

    def external_frames() -> torch.Tensor:
        supplied = external if external.shape[0] in (1, t) else external[:1]
        frames = [
            _external_plane(supplied[0 if supplied.shape[0] == 1 else i : i + 1], h, w, device, dtype, params)
            for i in range(t)
        ]
        return torch.stack(frames, dim=2).repeat(b, 1, 1, 1, 1)

    if pattern == "external_mask":
        if external is None:
            raise ValueError(
                "pattern is 'external_mask' but no external_mask was connected. "
                "Wire a MASK into external_mask, or pick a generated pattern."
            )
        return external_frames()

    if pattern in patterns.CONTENT_PATTERNS:
        source = content if content is not None else x
        flat = source.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        per_frame = patterns.content_field(flat, pattern, int(params["content_window"]))
        per_frame = _shape_mask(per_frame, params)
        mask = per_frame.view(b, t, 1, h, w).permute(0, 2, 1, 3, 4).contiguous()
    elif temporal_mode == "per_frame":
        frames = [
            _generated_plane(h, w, pattern, params, seed + i * stride, device, dtype)
            for i in range(t)
        ]
        mask = torch.stack(frames, dim=2).repeat(b, 1, 1, 1, 1)
    elif temporal_mode == "drift":
        raw = _drifting_planes(h, w, pattern, params, seed, t)
        shaped = _shape_mask(raw.to(device=device, dtype=dtype), params)
        mask = shaped.unsqueeze(0).transpose(1, 2).repeat(b, 1, 1, 1, 1)
    else:
        plane = _generated_plane(h, w, pattern, params, seed, device, dtype)
        mask = plane.unsqueeze(2).repeat(b, 1, t, 1, 1)

    gate = _gate_name(params)
    if gate is not None and pattern not in patterns.CONTENT_PATTERNS and content_pattern_fits(x):
        source = content if content is not None else x
        flat = source.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        field = _content_gate(flat, gate, params).to(device=device, dtype=dtype)
        mask = mask * field.view(b, t, 1, h, w).permute(0, 2, 1, 3, 4).contiguous()

    if external is not None:
        mask = mask * external_frames()
    return mask


def _noise_field(like: torch.Tensor, seed: int) -> torch.Tensor:
    """A standard normal draw the shape of a latent, one value per element.

    Args:
        like: The latent it is added to.
        seed: Seeds the draw, so one seed answers one field.

    Returns:
        A tensor on the latent's device and dtype, mean 0.0 and variance 1.0.
    """
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed) & 0xFFFFFFFF)
    field = torch.randn(tuple(like.shape), generator=rng, dtype=torch.float32, device="cpu")
    return field.to(device=like.device, dtype=like.dtype)


def _operand(value, like: torch.Tensor, flat: bool):
    """A scale or bias ready to broadcast against a latent.

    Args:
        value: A number, or a tensor holding one entry per channel.
        like: The latent it is applied to.
        flat: Whether a 3D latent was widened to 4D, which the tensor form follows.

    Returns:
        A float, or a tensor on the latent's device and dtype.
    """
    if not torch.is_tensor(value):
        return float(value)
    out = value.to(device=like.device, dtype=like.dtype)
    return out.unsqueeze(-2) if flat and out.ndim == like.ndim - 1 else out


def apply_plane(
    x: torch.Tensor,
    scale,
    bias,
    pattern: str,
    temporal_mode: str,
    seed: int,
    external_mask: torch.Tensor | None = None,
    options: dict | None = None,
    content: torch.Tensor | None = None,
    bias_dc=0.0,
    signal: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the affine over one latent tensor.

    Args:
        x: A ``[B, C, L]``, ``[B, C, H, W]`` or ``[B, C, T, H, W]`` latent.
        scale: What the latent is multiplied by where the mask is 1.0. A tensor holding one
            entry per channel is broadcast across the rest.
        bias: What is added where the mask is 1.0, in the same two forms. On a
            ``bias_field`` of ``"gaussian"`` it scales a standard normal draw instead
            of adding a flat offset.
        pattern: A name from :data:`modules.latent.affine_patterns.PATTERNS`.
        temporal_mode: ``"static"``, ``"per_frame"`` or ``"drift"``, read for a 5D
            latent only.
        seed: Seeds the mask and the gaussian bias field.
        external_mask: A supplied MASK, used alone for ``external_mask`` and as a gate
            otherwise.
        options: Pattern parameters and mask shaping, over :data:`DEFAULTS`.
        content: What a content-aware pattern reads instead of the latent, the same shape.
            A sampler passes the model's clean estimate here, since a latent part way
            through a run is mostly noise and has no detail to find.
        bias_dc: A further offset added where the mask is 1.0, in the same two forms as
            ``bias``. A ``bias_field`` of ``"gaussian"`` leaves this one flat.
        signal: The part of the latent that is picture rather than noise, the same shape.
            Given one, the multiplier is applied to it alone and the latent's noise is left
            as it was. Without one the multiplier is applied to the whole latent.

    Returns:
        ``(latent, mask)``. The mask is ``[B, H, W]`` for a 4D latent, ``[B * T, H, W]``
        for a 5D one and ``[B, 1, L]`` for a 3D one.

    Raises:
        ValueError: The latent has a shape the affine cannot mask.
    """
    params = resolve(options)
    flat = x.ndim == 3
    work = x.unsqueeze(-2) if flat else x
    gain = _operand(scale, work, flat)
    offset = _operand(bias, work, flat)
    if str(params.get("bias_field", "constant")) == "gaussian":
        offset = offset * _noise_field(work, seed)
    offset = offset + _operand(bias_dc, work, flat)
    read = content
    if read is not None and flat:
        read = read.unsqueeze(-2)
    if read is not None and read.shape != work.shape:
        read = None
    picture = signal
    if picture is not None and flat:
        picture = picture.unsqueeze(-2)
    if picture is not None and picture.shape != work.shape:
        picture = None

    def combine(latent, mask):
        """The affine over one masked plane, on the picture alone where one was given."""
        if picture is None:
            return latent * ((1.0 - mask) + mask * gain) + offset * mask
        return latent + (gain - 1.0) * mask * picture + offset * mask

    if work.ndim == 4:
        mask = _mask_for_plane(work, pattern, params, seed, external_mask, read)
        out = combine(work, mask)
        mask_out = mask.squeeze(1)
    elif work.ndim == 5:
        mask = _mask_for_frames(
            work, pattern, params, seed, external_mask, temporal_mode, read
        )
        out = combine(work, mask)
        b, _c, t, h, w = work.shape
        mask_out = mask.squeeze(1).contiguous().reshape(b * t, h, w)
    else:
        raise ValueError(
            "a latent the affine can mask is [B, C, L], [B, C, H, W] or [B, C, T, H, W]; "
            f"a tensor with {x.ndim} dimensions was given"
        )

    if bool(params["clamp"]):
        out = out.clamp(float(params["clamp_min"]), float(params["clamp_max"]))
    if flat:
        out = out.squeeze(-2)
    return out, mask_out.clamp(0.0, 1.0).to(x.dtype)


def apply_affine(
    samples,
    scale: float,
    bias: float,
    pattern: str,
    temporal_mode: str,
    seed: int,
    streams: str = "video",
    external_mask: torch.Tensor | None = None,
    options: dict | None = None,
) -> tuple[object, torch.Tensor | None]:
    """Run the affine over every selected stream of a latent.

    Args:
        samples: A tensor, or a NestedTensor packing several streams.
        scale: What the latent is multiplied by where the mask is 1.0.
        bias: What is added where the mask is 1.0.
        pattern: A name from :data:`modules.latent.affine_patterns.PATTERNS`.
        temporal_mode: ``"static"``, ``"per_frame"`` or ``"drift"``.
        seed: Seeds the mask. Each stream past the first adds its index.
        streams: ``"video"``, ``"audio"`` or ``"both"``.
        external_mask: A supplied MASK.
        options: Pattern parameters and mask shaping, over :data:`DEFAULTS`.

    Returns:
        ``(latent, mask)`` in the layout the latent arrived in. The mask belongs to the
        first stream transformed, and is None where no stream was.
    """
    import comfy.nested_tensor

    parts = streams_of(samples)
    wanted = stream_indices(streams, len(parts))
    params = resolve(options)

    out = list(parts)
    mask_out = None
    for index in wanted:
        if index >= len(parts):
            continue
        source = parts[index]
        if not torch.is_tensor(source) or source.ndim not in (3, 4, 5):
            continue
        chosen = pattern
        if chosen in patterns.CONTENT_PATTERNS and not content_pattern_fits(source):
            chosen = "solid"
        out[index], mask = apply_plane(
            source, scale, bias, chosen, temporal_mode, seed + index, external_mask, params
        )
        if mask_out is None:
            mask_out = mask

    if getattr(samples, "is_nested", False):
        return comfy.nested_tensor.NestedTensor(out), mask_out
    return out[0], mask_out


def content_pattern_fits(x: torch.Tensor) -> bool:
    """Report whether a stream has enough spatial extent for a content-aware pattern.

    Args:
        x: A latent stream.

    Returns:
        True where a local neighbourhood can be pooled over.
    """
    if x.ndim == 3:
        return False
    return min(int(x.shape[-2]), int(x.shape[-1])) >= patterns.MIN_CONTENT_EXTENT

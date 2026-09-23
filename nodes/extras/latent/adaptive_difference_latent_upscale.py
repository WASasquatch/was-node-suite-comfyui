"""Latent upscale that mixes a blocky enlargement with a smooth one, per position."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from comfy_api.latest import io

from ....modules.compat.sockets import require_input

REQUIRES = "extras"

SMOOTH_MODES = ["bilinear", "bicubic", "area"]
GATE_MODES = ["none", "weight", "weight_sqrt"]
PREVIEW_MODES = ["weight", "damp", "both"]


def resize_tensor(x: torch.Tensor, size_hw: tuple[int, int], mode: str) -> torch.Tensor:
    """Resize a ``[B, C, H, W]`` tensor with one of the interpolation modes.

    Args:
        x: The tensor to resize.
        size_hw: Target ``(height, width)``.
        mode: ``"bilinear"``, ``"bicubic"``, ``"area"``, ``"nearest-exact"``, or any other
            mode ``torch.nn.functional.interpolate`` accepts.

    Returns:
        The resized tensor. Corners are not aligned on the two interpolating modes, so the
        result keeps the same framing as the input rather than stretching half a sample
        outwards.
    """
    if mode == "bilinear":
        return F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
    if mode == "bicubic":
        return F.interpolate(x, size=size_hw, mode="bicubic", align_corners=False)
    if mode == "area":
        return F.interpolate(x, size=size_hw, mode="area")
    if mode == "nearest-exact":
        return F.interpolate(x, size=size_hw, mode="nearest-exact")
    return F.interpolate(x, size=size_hw, mode=mode)


def sigmoid_weight(d: torch.Tensor, threshold: float, softness: float) -> torch.Tensor:
    """Turn a measurement into a 0-1 weight with a soft cut-off.

    Args:
        d: The measured values.
        threshold: Where the weight passes 0.5.
        softness: How wide the transition is. Small values approach a hard step; the value
            is floored at 1e-8 so a softness of zero cannot divide by zero.

    Returns:
        Weights between 0.0 and 1.0, same shape as ``d``.
    """
    t = float(threshold)
    s = max(1e-8, float(softness))
    return torch.sigmoid((d - t) / s)


def apply_temporal_ema(weight_bt1hw: torch.Tensor, b: int, t: int, ema: float) -> torch.Tensor:
    """Smooth a per-frame map along time so it cannot flicker.

    Args:
        weight_bt1hw: A ``[B*T, 1, H, W]`` map in the order
            :func:`modules.latent.shape.flatten_5d_to_4d` produces.
        b: Batch size.
        t: Frame count.
        ema: How much of the running result each frame keeps, 0.0 to just under 1.0. 0.0
            disables the smoothing; 0.9 leaves a long trail.

    Returns:
        The smoothed map in the same layout. A single-frame clip is returned untouched.
    """
    if ema <= 0.0 or t <= 1:
        return weight_bt1hw

    ema = float(ema)
    bt, c, h, w = weight_bt1hw.shape
    wgt = weight_bt1hw.reshape(b, t, c, h, w).contiguous()

    for bi in range(b):
        prev = wgt[bi, 0]
        for ti in range(1, t):
            cur = wgt[bi, ti]
            prev = prev * ema + cur * (1.0 - ema)
            wgt[bi, ti] = prev

    return wgt.reshape(bt, c, h, w).contiguous()


def normalize_map(x: torch.Tensor) -> torch.Tensor:
    """Scale each map so its largest value is 1.0.

    Args:
        x: A ``[B, C, H, W]`` tensor of non-negative values.

    Returns:
        The tensor divided by its own per-map maximum, which makes a threshold mean the
        same thing whatever the absolute size of the values. A map that is entirely zero
        stays zero.
    """
    mx = x.amax(dim=(2, 3), keepdim=True)
    return x / (mx + 1e-8)


@dataclass
class AdaptiveBlendConfig:
    """Every setting the upscale reads, with the widget defaults as its own."""

    scale: float = 2.0
    smooth_mode: str = "bilinear"
    diff_blur_sigma: float = 0.6
    threshold: float = 0.12
    softness: float = 0.05
    weight_power: float = 1.0
    weight_blur_sigma: float = 0.0
    temporal_ema: float = 0.0

    enable_directional_damping: bool = True
    damping_strength: float = 0.35
    damping_gate_mode: str = "weight_sqrt"
    damping_grad_blur_sigma: float = 0.0
    damping_threshold: float = 0.25
    damping_softness: float = 0.08
    damping_power: float = 1.0
    damping_mask_blur_sigma: float = 0.6
    damping_highpass_sigma: float = 1.0
    damping_temporal_ema: float = 0.25

    preview_mode: str = "both"
    output_mask_pixel_scale: int = 8


def compute_damp_mask(
    latent_btchw_fp32: torch.Tensor,
    weight_bt1hw_fp32: torch.Tensor,
    gate_mode: str,
    grad_blur_sigma: float,
    damp_threshold: float,
    damp_softness: float,
    damp_power: float,
    damp_mask_blur_sigma: float,
) -> torch.Tensor:
    """Find where the latent has strong boundaries, as a 0-1 mask.

    Args:
        latent_btchw_fp32: The blended latent, ``[B*T, C, H, W]`` in float32.
        weight_bt1hw_fp32: The blend weight map, ``[B*T, 1, H, W]`` in float32.
        gate_mode: ``"none"``, ``"weight"`` or ``"weight_sqrt"``.
        grad_blur_sigma: Blur applied to the energy before the gradient is taken.
        damp_threshold: Gradient level, relative to the strongest one present, where the
            mask passes 0.5.
        damp_softness: Width of that transition.
        damp_power: Exponent applied to the finished mask.
        damp_mask_blur_sigma: Blur applied to the finished mask.

    Returns:
        A ``[B*T, 1, H, W]`` mask between 0.0 and 1.0.
    """
    from ....modules.latent.filters import clamp01, gaussian_blur_depthwise, sobel_grad_mag

    energy = latent_btchw_fp32.abs().mean(dim=1, keepdim=True)

    if grad_blur_sigma > 0.0:
        energy = gaussian_blur_depthwise(energy, float(grad_blur_sigma))

    grad = sobel_grad_mag(energy)
    grad = normalize_map(grad)

    mask = sigmoid_weight(grad, float(damp_threshold), float(damp_softness))
    mask = clamp01(mask)

    if gate_mode == "weight":
        mask = mask * clamp01(weight_bt1hw_fp32)
    elif gate_mode == "weight_sqrt":
        mask = mask * torch.sqrt(clamp01(weight_bt1hw_fp32) + 1e-8)
    elif gate_mode == "none":
        pass
    else:
        pass

    if damp_power != 1.0:
        mask = clamp01(mask).pow(float(damp_power))

    if damp_mask_blur_sigma > 0.0:
        mask = gaussian_blur_depthwise(mask, float(damp_mask_blur_sigma))
        mask = clamp01(mask)

    return mask


def apply_highpass_damping(
    latent_btchw_fp32: torch.Tensor,
    damp_mask_bt1hw_fp32: torch.Tensor,
    strength: float,
    highpass_sigma: float,
) -> torch.Tensor:
    """Take the fine detail back out of the latent where the mask says to.

    Args:
        latent_btchw_fp32: The latent to treat, ``[B*T, C, H, W]`` in float32.
        damp_mask_bt1hw_fp32: Where to treat it, ``[B*T, 1, H, W]``.
        strength: How much of the fine detail is removed where the mask is 1.0. 0.0 or less
            returns the latent unchanged; 1.0 removes all of it.
        highpass_sigma: Blur radius that separates broad from fine. Larger values count more
            of the image as fine detail.

    Returns:
        The treated latent, same shape and dtype.
    """
    from ....modules.latent.filters import clamp01, gaussian_blur_depthwise

    s = float(strength)
    if s <= 0.0:
        return latent_btchw_fp32

    low = (
        gaussian_blur_depthwise(latent_btchw_fp32, float(highpass_sigma))
        if highpass_sigma > 0.0
        else latent_btchw_fp32
    )
    high = latent_btchw_fp32 - low

    m = clamp01(damp_mask_bt1hw_fp32)
    return low + high * (1.0 - s * m)


class AdaptiveDifferenceLatentUpscale(io.ComfyNode):
    """Upscale a latent by blending a nearest and a smooth enlargement per position."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASNSAdaptiveDifferenceLatentUpscale",
            display_name="WAS Adaptive Difference Latent Upscale (Damped)",
            search_aliases=[
                "WASNSAdaptiveDifferenceLatentUpscale",
                "WAS Adaptive Difference Latent Upscale (Damped)",
                "adaptive latent upscale",
                "latent upscale",
                "damped upscale",
            ],
            category="WAS Suite/Latent/Transform",
            description=(
                "Enlarge a latent twice, once by copying the nearest block, once by "
                "interpolating, and take the smooth version only where the two disagree. "
                "Flat areas keep the crispness of the blocky enlargement while edges and "
                "texture get the smooth one, which is what stops a plain latent upscale "
                "either going soft everywhere or ringing along every boundary. Works on "
                "video latents as well as single images, and reports the maps it used so "
                "the settings can be seen rather than guessed at."
            ),
            inputs=[
                io.Latent.Input(
                    "latent",
                    tooltip=(
                        "The latent to enlarge. A video latent with a time axis is handled "
                        "frame by frame."
                    ),
                ),
                io.Float.Input(
                    "scale",
                    default=2.0,
                    min=1.0,
                    max=8.0,
                    step=0.05,
                    tooltip=(
                        "How much larger the result is. 2.0 doubles both sides, 1.5 adds "
                        "half again, 1.0 leaves the size alone and only applies the "
                        "damping. Sizes are rounded to whole latent blocks."
                    ),
                ),
                io.Combo.Input(
                    "smooth_mode",
                    options=SMOOTH_MODES,
                    default="bilinear",
                    tooltip=(
                        "How the smooth half of the blend is enlarged. `bilinear` is the "
                        "safe default; `bicubic` is sharper and can overshoot slightly at "
                        "a hard edge; `area` averages the source region and is the softest "
                        "of the three."
                    ),
                ),
                io.Float.Input(
                    "diff_blur_sigma",
                    default=0.6,
                    min=0.0,
                    max=8.0,
                    step=0.05,
                    tooltip=(
                        "How far the disagreement between the two enlargements is spread "
                        "before it decides anything, in latent blocks. A little blur keeps "
                        "the blend from switching on and off block by block; 0.0 uses the "
                        "raw per-block difference and gives the busiest map."
                    ),
                ),
                io.Float.Input(
                    "threshold",
                    default=0.12,
                    min=0.0,
                    max=1.0,
                    step=0.005,
                    tooltip=(
                        "How much disagreement counts as detail. Below this the blocky "
                        "enlargement is kept, above it the smooth one takes over. Lower "
                        "values smooth more of the picture; 0.12 leaves flat areas crisp "
                        "and treats edges."
                    ),
                ),
                io.Float.Input(
                    "softness",
                    default=0.05,
                    min=0.0005,
                    max=1.0,
                    step=0.001,
                    tooltip=(
                        "How gradual the changeover at the threshold is. Small values such "
                        "as 0.005 give a hard switch that can be seen as a rim; 0.05 fades "
                        "between the two enlargements over a comfortable range."
                    ),
                ),
                io.Float.Input(
                    "weight_power",
                    default=1.0,
                    min=0.1,
                    max=6.0,
                    step=0.05,
                    tooltip=(
                        "Bends the blend map after it is built. Above 1.0 pulls it towards "
                        "the blocky enlargement everywhere but the strongest edges; below "
                        "1.0 spreads the smooth enlargement into weaker detail. 1.0 leaves "
                        "the map as measured."
                    ),
                ),
                io.Float.Input(
                    "weight_blur_sigma",
                    default=0.0,
                    min=0.0,
                    max=8.0,
                    step=0.05,
                    tooltip=(
                        "Softens the finished blend map, in latent blocks. Raise it when "
                        "the treated areas have visible outlines of their own; 0.0 leaves "
                        "the map alone."
                    ),
                ),
                io.Float.Input(
                    "temporal_ema",
                    default=0.0,
                    min=0.0,
                    max=0.99,
                    step=0.01,
                    tooltip=(
                        "How much each frame of a video latent carries over from the frames "
                        "before it, which stops the blend map flickering. 0.0 treats every "
                        "frame on its own; 0.5 is a light smoothing; 0.9 is heavy and can "
                        "smear the map behind fast motion. Ignored on a single image."
                    ),
                ),
                io.Boolean.Input(
                    "enable_directional_damping",
                    default=True,
                    tooltip=(
                        "Whether the second pass runs, which takes fine detail back out "
                        "along strong boundaries. It is what removes the halo that an "
                        "upscale leaves around hard edges. Turn it off to see the blend on "
                        "its own, or when the source is already soft."
                    ),
                ),
                io.Float.Input(
                    "damping_strength",
                    default=0.35,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How much fine detail is removed where the damping mask is fully "
                        "on. 0.0 removes none and turns the pass off; 0.35 takes the edge "
                        "off a halo; 1.0 flattens the detail there completely."
                    ),
                ),
                io.Combo.Input(
                    "damping_gate_mode",
                    options=GATE_MODES,
                    default="weight_sqrt",
                    tooltip=(
                        "Where the damping is allowed to act. `none` lets it act on every "
                        "boundary it finds. `weight` confines it to the areas the blend "
                        "already treated. `weight_sqrt` is in between, allowing some "
                        "damping in areas the blend touched only lightly."
                    ),
                ),
                io.Float.Input(
                    "damping_grad_blur_sigma",
                    default=0.0,
                    min=0.0,
                    max=8.0,
                    step=0.05,
                    tooltip=(
                        "Blur applied before boundaries are looked for, in latent blocks. "
                        "Raise it so that texture is not mistaken for an edge; 0.0 finds "
                        "the finest boundaries."
                    ),
                ),
                io.Float.Input(
                    "damping_threshold",
                    default=0.25,
                    min=0.0,
                    max=1.0,
                    step=0.005,
                    tooltip=(
                        "How strong a boundary has to be to be damped, measured against the "
                        "strongest one in the picture. 0.25 catches the clear outlines; "
                        "lower values reach into texture as well."
                    ),
                ),
                io.Float.Input(
                    "damping_softness",
                    default=0.08,
                    min=0.0005,
                    max=1.0,
                    step=0.001,
                    tooltip=(
                        "How gradually the damping fades in around that threshold. Small "
                        "values give a hard-edged mask; 0.08 fades over a comfortable range."
                    ),
                ),
                io.Float.Input(
                    "damping_power",
                    default=1.0,
                    min=0.1,
                    max=6.0,
                    step=0.05,
                    tooltip=(
                        "Bends the damping mask. Above 1.0 confines the damping to the very "
                        "strongest boundaries; below 1.0 spreads it over more of the "
                        "picture. 1.0 leaves the mask as measured."
                    ),
                ),
                io.Float.Input(
                    "damping_mask_blur_sigma",
                    default=0.6,
                    min=0.0,
                    max=8.0,
                    step=0.05,
                    tooltip=(
                        "Softens the damping mask before it is used, in latent blocks. A "
                        "little blur keeps the damped strip from having a visible border of "
                        "its own; 0.0 uses the mask as found."
                    ),
                ),
                io.Float.Input(
                    "damping_highpass_sigma",
                    default=1.0,
                    min=0.0,
                    max=8.0,
                    step=0.05,
                    tooltip=(
                        "Which detail counts as fine enough to be removed, in latent "
                        "blocks. 1.0 takes out ringing while leaving the shapes; larger "
                        "values reach into broader structure and start to blur. 0.0 removes "
                        "the whole signal under the mask instead."
                    ),
                ),
                io.Float.Input(
                    "damping_temporal_ema",
                    default=0.25,
                    min=0.0,
                    max=0.99,
                    step=0.01,
                    tooltip=(
                        "How much of the damping mask each frame of a video latent carries "
                        "over from the frames before it, so damped areas do not shimmer. "
                        "0.0 treats every frame on its own. Ignored on a single image."
                    ),
                ),
                io.Combo.Input(
                    "preview_mode",
                    options=PREVIEW_MODES,
                    default="both",
                    tooltip=(
                        "Which map leaves the node. `weight` shows where the smooth "
                        "enlargement was used, `damp` shows where fine detail was removed, "
                        "`both` puts the two side by side in the preview and sends the "
                        "damping map to the mask output."
                    ),
                ),
                io.Int.Input(
                    "output_mask_pixel_scale",
                    default=8,
                    min=1,
                    max=16,
                    step=1,
                    tooltip=(
                        "How many preview pixels each latent block becomes. 8 matches the "
                        "size the latent decodes to on most VAEs, so the preview lines up "
                        "with the finished picture; 1 gives the small raw map. This affects "
                        "the preview image only, never the mask output."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output(
                    display_name="latent",
                    tooltip="The enlarged latent, ready for a sampler or a decode.",
                ),
                io.Mask.Output(
                    display_name="mask",
                    tooltip=(
                        "The map the node worked from, at latent resolution: the damping "
                        "map on `damp` and `both`, the blend map on `weight`. Useful for "
                        "driving another node from the same areas this one treated."
                    ),
                ),
                io.Image.Output(
                    display_name="mask_preview",
                    tooltip=(
                        "The same map as a viewable image, enlarged by "
                        "output_mask_pixel_scale. On `both` the blend map is on the left "
                        "and the damping map on the right."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        latent,
        scale,
        smooth_mode,
        diff_blur_sigma,
        threshold,
        softness,
        weight_power,
        weight_blur_sigma,
        temporal_ema,
        enable_directional_damping,
        damping_strength,
        damping_gate_mode,
        damping_grad_blur_sigma,
        damping_threshold,
        damping_softness,
        damping_power,
        damping_mask_blur_sigma,
        damping_highpass_sigma,
        damping_temporal_ema,
        preview_mode,
        output_mask_pixel_scale,
    ) -> io.NodeOutput:
        """Upscale the latent and blend the difference back under a damping mask.

        Raises:
            ValueError: Nothing is connected to the latent input, or the latent holds no
                samples.
        """
        from ....modules.latent.filters import clamp01, gaussian_blur_depthwise, mask_preview
        from ....modules.latent.shape import flatten_5d_to_4d, is_latent_5d, unflatten_4d_to_5d

        require_input(
            latent,
            "WAS Adaptive Difference Latent Upscale (Damped)",
            "latent",
            "latent",
            "latent source such as Empty Latent Image or VAE Encode",
            "LATENT",
        )
        if "samples" not in latent:
            raise ValueError("LATENT input must be a dict containing key 'samples'.")

        samples: torch.Tensor = latent["samples"]
        orig_dtype = samples.dtype

        cfg = AdaptiveBlendConfig(
            scale=float(scale),
            smooth_mode=str(smooth_mode),
            diff_blur_sigma=float(diff_blur_sigma),
            threshold=float(threshold),
            softness=float(softness),
            weight_power=float(weight_power),
            weight_blur_sigma=float(weight_blur_sigma),
            temporal_ema=float(temporal_ema),
            enable_directional_damping=bool(enable_directional_damping),
            damping_strength=float(damping_strength),
            damping_gate_mode=str(damping_gate_mode),
            damping_grad_blur_sigma=float(damping_grad_blur_sigma),
            damping_threshold=float(damping_threshold),
            damping_softness=float(damping_softness),
            damping_power=float(damping_power),
            damping_mask_blur_sigma=float(damping_mask_blur_sigma),
            damping_highpass_sigma=float(damping_highpass_sigma),
            damping_temporal_ema=float(damping_temporal_ema),
            preview_mode=str(preview_mode),
            output_mask_pixel_scale=int(output_mask_pixel_scale),
        )

        was_5d = is_latent_5d(samples)
        if was_5d:
            samples_4d, b, t = flatten_5d_to_4d(samples)
            h, w = samples.shape[-2], samples.shape[-1]
        else:
            b = samples.shape[0]
            t = 1
            samples_4d = samples
            h, w = samples.shape[-2], samples.shape[-1]

        target_h = int(round(h * cfg.scale))
        target_w = int(round(w * cfg.scale))
        if target_h < 1 or target_w < 1:
            raise ValueError("Invalid target size computed from scale.")

        base = resize_tensor(samples_4d, (target_h, target_w), mode="nearest-exact")
        smooth = resize_tensor(samples_4d, (target_h, target_w), mode=cfg.smooth_mode)

        # The blend and every map built from it run in float32 whatever the latent's own
        # precision: a sigmoid and a normalised gradient on float16 quantise into steps
        # that show up as banding in the mask.
        base_fp = base.float()
        smooth_fp = smooth.float()

        diff = (base_fp - smooth_fp).abs().mean(dim=1, keepdim=True)
        if cfg.diff_blur_sigma > 0.0:
            diff = gaussian_blur_depthwise(diff, cfg.diff_blur_sigma)

        wgt = sigmoid_weight(diff, cfg.threshold, cfg.softness)
        wgt = clamp01(wgt)

        if cfg.weight_power != 1.0:
            wgt = clamp01(wgt).pow(cfg.weight_power)

        if cfg.weight_blur_sigma > 0.0:
            wgt = gaussian_blur_depthwise(wgt, cfg.weight_blur_sigma)
            wgt = clamp01(wgt)

        if was_5d and cfg.temporal_ema > 0.0:
            wgt = apply_temporal_ema(wgt, b=b, t=t, ema=cfg.temporal_ema)

        out_fp = base_fp * (1.0 - wgt) + smooth_fp * wgt

        damp_mask = torch.zeros_like(wgt)
        if cfg.enable_directional_damping and cfg.damping_strength > 0.0:
            damp_mask = compute_damp_mask(
                latent_btchw_fp32=out_fp,
                weight_bt1hw_fp32=wgt,
                gate_mode=cfg.damping_gate_mode,
                grad_blur_sigma=cfg.damping_grad_blur_sigma,
                damp_threshold=cfg.damping_threshold,
                damp_softness=cfg.damping_softness,
                damp_power=cfg.damping_power,
                damp_mask_blur_sigma=cfg.damping_mask_blur_sigma,
            )

            if was_5d and cfg.damping_temporal_ema > 0.0:
                damp_mask = apply_temporal_ema(damp_mask, b=b, t=t, ema=cfg.damping_temporal_ema)

            out_fp = apply_highpass_damping(
                latent_btchw_fp32=out_fp,
                damp_mask_bt1hw_fp32=damp_mask,
                strength=cfg.damping_strength,
                highpass_sigma=cfg.damping_highpass_sigma,
            )

        out = out_fp.to(dtype=orig_dtype)

        if was_5d:
            out = unflatten_4d_to_5d(out, b=b, t=t)

        out_latent = dict(latent)
        out_latent["samples"] = out

        mask_for_output = damp_mask if cfg.preview_mode in ("damp", "both") else wgt
        mask_out = mask_for_output[:, 0, :, :].contiguous()

        ps = max(1, int(cfg.output_mask_pixel_scale))

        if cfg.preview_mode == "weight":
            prev_img = mask_preview(wgt, ps)
        elif cfg.preview_mode == "damp":
            prev_img = mask_preview(damp_mask, ps)
        else:
            a = mask_preview(wgt, ps)
            bimg = mask_preview(damp_mask, ps)
            prev_img = torch.cat([a, bimg], dim=2)

        return io.NodeOutput(out_latent, mask_out, prev_img)

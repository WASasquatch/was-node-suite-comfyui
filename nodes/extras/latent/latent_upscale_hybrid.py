"""Latent upscale that decides where to be smooth by looking at the decoded picture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from comfy_api.latest import io

from ....modules.compat.sockets import require_input
from ....modules.image.convolve import (
    canny,
    dilate,
    ellipse_kernel,
    gaussian_blur,
    luminance,
)

REQUIRES = "extras"

MASK_RESOLUTIONS = ["image", "latent"]

#: Channel counts a latent is known to have. Used to tell a ``[B, C, T, H, W]`` video
#: latent from a ``[B, T, C, H, W]`` one, which are otherwise the same five numbers in a
#: different order and cannot be told apart by shape alone.
LATENT_CHANNELS = (4, 8, 16)


@dataclass
class EdgeBlendConfig:
    """Every setting the upscale reads, with the widget defaults as its own."""

    scale: float = 2.0

    pre_blur_sigma_px: float = 1.0
    canny_threshold1: int = 25
    canny_threshold2: int = 155
    canny_l2gradient: bool = True

    dilate_radius_px: int = 8
    feather_sigma_px: float = 6.0

    mask_min: float = 0.0
    mask_max: float = 1.0

    nearest_exact: bool = True
    align_corners: bool | None = False

    output_mask_resolution: str = "image"

    video_decode_horizontal_tiles: int = 2
    video_decode_vertical_tiles: int = 2
    video_decode_overlap_latent: int = 4
    video_decode_last_frame_fix: bool = False
    video_decode_enable_cudnn: bool = True


def is_probable_latent_channels(v: int) -> bool:
    """Report whether a dimension is a plausible latent channel count.

    Args:
        v: The size of one dimension.

    Returns:
        ``True`` when it is one of :data:`LATENT_CHANNELS`.
    """
    return int(v) in LATENT_CHANNELS


def normalize_latent_to_bchw(x: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any]]:
    """Flatten a latent of any supported layout to 4D ``[B, C, H, W]``.

    Args:
        x: The tensor from a LATENT's ``samples`` key.

    Returns:
        ``(x4, meta)``, the latent as ``[B', C, H, W]`` where ``B'`` is ``B`` for an image
        and ``B*T`` for a video, and the record :func:`restore_latent_from_bchw` needs to
        put it back.

    Raises:
        TypeError: The value is not a tensor.
        ValueError: The tensor is neither 4D nor 5D.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("latent_samples must be a torch.Tensor")

    if x.dim() == 4:
        b, c, h, w = x.shape
        return x, {"layout": "4d_bchw", "B": int(b), "C": int(c), "H": int(h), "W": int(w)}

    if x.dim() != 5:
        raise ValueError(f"Expected latent 4D or 5D, got {tuple(x.shape)}")

    b = int(x.shape[0])

    if is_probable_latent_channels(int(x.shape[1])):
        c = int(x.shape[1])
        t = int(x.shape[2])
        h = int(x.shape[3])
        w = int(x.shape[4])
        x4 = x.permute(0, 2, 1, 3, 4).contiguous().reshape(b * t, c, h, w)
        return x4, {"layout": "5d_bcthw", "B": b, "C": c, "T": t, "H": h, "W": w}

    if is_probable_latent_channels(int(x.shape[2])):
        t = int(x.shape[1])
        c = int(x.shape[2])
        h = int(x.shape[3])
        w = int(x.shape[4])
        x4 = x.contiguous().reshape(b * t, c, h, w)
        return x4, {"layout": "5d_btchw", "B": b, "C": c, "T": t, "H": h, "W": w}

    c = int(x.shape[1])
    t = int(x.shape[2])
    h = int(x.shape[3])
    w = int(x.shape[4])
    x4 = x.permute(0, 2, 1, 3, 4).contiguous().reshape(b * t, c, h, w)
    return x4, {"layout": "5d_bcthw", "B": b, "C": c, "T": t, "H": h, "W": w}


def restore_latent_from_bchw(x4: torch.Tensor, meta: dict[str, Any]) -> torch.Tensor:
    """Undo :func:`normalize_latent_to_bchw`.

    Args:
        x4: The ``[B', C, H, W]`` latent. Its height and width may differ from the ones that
            went in, which is what makes this usable after a resize.
        meta: The record the flattening returned.

    Returns:
        The latent back in the layout it arrived in.

    Raises:
        ValueError: The tensor is not 4D, its batch or channel count does not match the
            record, or the record names a layout this cannot rebuild.
    """
    layout = meta["layout"]

    if layout == "4d_bchw":
        return x4

    if x4.dim() != 4:
        raise ValueError(f"Expected 4D [B',C,H,W], got {tuple(x4.shape)}")

    b = int(meta["B"])
    c = int(meta["C"])
    t = int(meta["T"])
    h = int(x4.shape[-2])
    w = int(x4.shape[-1])

    if int(x4.shape[0]) != b * t:
        raise ValueError(f"Expected batch {b * t}, got {int(x4.shape[0])}")
    if int(x4.shape[1]) != c:
        raise ValueError(f"Expected channels {c}, got {int(x4.shape[1])}")

    if layout == "5d_bcthw":
        return x4.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()

    if layout == "5d_btchw":
        return x4.reshape(b, t, c, h, w).contiguous()

    raise ValueError(f"Unknown latent layout: {layout}")


def upscale_latent_nearest_exact(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Enlarge a latent by copying the nearest block.

    Args:
        x: A ``[B, C, H, W]`` latent.
        size: Target ``(height, width)``.

    Returns:
        The enlarged latent, with every value in it one that was already there. Falls back
        to plain nearest-neighbour on a torch build without the exact variant.
    """
    try:
        return F.interpolate(x, size=size, mode="nearest-exact")
    except Exception:
        return F.interpolate(x, size=size, mode="nearest")


def upscale_latent_bilinear(
    x: torch.Tensor, size: tuple[int, int], align_corners: bool | None
) -> torch.Tensor:
    """Enlarge a latent by interpolating between blocks.

    Args:
        x: A ``[B, C, H, W]`` latent.
        size: Target ``(height, width)``.
        align_corners: Whether the outermost values are pinned to the edges of the result.

    Returns:
        The enlarged latent.
    """
    return F.interpolate(x, size=size, mode="bilinear", align_corners=align_corners)


def make_elliptical_kernel(radius_px: int, device=None):
    """Build the round brush the edge mask is grown with.

    Args:
        radius_px: Radius in pixels.
        device: Where the kernel is built.

    Returns:
        An elliptical structuring element ``2 * radius + 1`` across, or ``None`` when the
        radius is 0 or less and there is nothing to grow.
    """
    r = int(radius_px)
    if r <= 0:
        return None
    return ellipse_kernel(r, device=device)


def latent_mask_to_comfy_mask(mask_b1hw: torch.Tensor) -> torch.Tensor:
    """Drop the channel axis so a mask can leave the node.

    Args:
        mask_b1hw: A ``[B, 1, H, W]`` mask.

    Returns:
        The ``[B, H, W]`` mask a MASK socket carries, clamped to 0.0-1.0.

    Raises:
        ValueError: The tensor is not 4D with exactly one channel.
    """
    if mask_b1hw.dim() != 4 or int(mask_b1hw.shape[1]) != 1:
        raise ValueError(f"Expected [B,1,H,W], got {tuple(mask_b1hw.shape)}")
    return torch.clamp(mask_b1hw[:, 0, :, :], 0.0, 1.0)


def get_vae_scale_factors(vae) -> tuple[int, int, int]:
    """How much larger than its latent a decode comes out.

    Args:
        vae: The VAE that will decode.

    Returns:
        ``(time_scale, width_scale, height_scale)``, each at least 1. A VAE that reports
        nothing usable is treated as 1 in every direction.
    """
    df = getattr(vae, "downscale_index_formula", None)
    if df:
        try:
            t, w, h = df
            return max(1, int(t)), max(1, int(w)), max(1, int(h))
        except Exception:
            pass

    spatial = 1
    temporal = 1

    scd = getattr(vae, "spacial_compression_decode", None)
    if callable(scd):
        try:
            v = scd()
            spatial = 1 if v is None else int(v)
        except Exception:
            spatial = 1

    tcd = getattr(vae, "temporal_compression_decode", None)
    if callable(tcd):
        try:
            v = tcd()
            temporal = 1 if v is None else int(v)
        except Exception:
            temporal = 1

    return max(1, int(temporal)), max(1, int(spatial)), max(1, int(spatial))


def decode_image_latent_regular(vae, latent_bchw: torch.Tensor) -> torch.Tensor:
    """Decode a 4D image latent to pixels.

    Args:
        vae: The VAE to decode with.
        latent_bchw: A ``[B, C, H, W]`` latent.

    Returns:
        A ``[B, H, W, C]`` image tensor. A decode that comes back with a single-frame time
        axis has it removed.

    Raises:
        ValueError: The latent is not 4D, the decode did not return a tensor, the result is
            not an image, or it has fewer than three colour channels.
    """
    if latent_bchw.dim() != 4:
        raise ValueError(f"Expected 4D [B,C,H,W], got {tuple(latent_bchw.shape)}")
    images = vae.decode(latent_bchw)
    if not isinstance(images, torch.Tensor):
        raise ValueError("vae.decode did not return a torch.Tensor")
    if images.dim() == 5 and int(images.shape[1]) == 1:
        images = images[:, 0, :, :, :]
    if images.dim() != 4:
        raise ValueError(f"Expected decoded [B,H,W,C], got {tuple(images.shape)}")
    if int(images.shape[-1]) < 3:
        raise ValueError(f"Decoded channels must be >=3, got {int(images.shape[-1])}")
    return images


def decode_video_latent_lazy_tiled(
    vae,
    latent_bcthw: torch.Tensor,
    horizontal_tiles: int,
    vertical_tiles: int,
    overlap_latent: int,
    last_frame_fix: bool,
    enable_cudnn: bool,
) -> torch.Tensor:
    """Decode a 5D video latent a tile at a time.

    Args:
        vae: The VAE to decode with.
        latent_bcthw: A ``[B, C, T, H, W]`` latent.
        horizontal_tiles: How many columns to split into, at least 1.
        vertical_tiles: How many rows to split into, at least 1.
        overlap_latent: How far neighbouring tiles overlap, in latent units.
        last_frame_fix: Whether the final frame is duplicated before decoding and the extra
            output frames dropped afterwards, which covers a VAE that mishandles the end of
            a clip.
        enable_cudnn: Whether cuDNN is left on for the decode. Turning it off is slower and
            avoids the workspace allocations that make some cards run out of memory here.

    Returns:
        A ``[B, T_out, H_px, W_px, C]`` image tensor.

    Raises:
        ValueError: The latent is not 5D, or a tile decode did not return a tensor.
        RuntimeError: A decoded tile has an unusable shape, batch or channel count.
    """
    if latent_bcthw.dim() != 5:
        raise ValueError(f"Expected 5D [B,C,T,H,W], got {tuple(latent_bcthw.shape)}")

    with torch.backends.cudnn.flags(enabled=bool(enable_cudnn)):
        samples = latent_bcthw
        b, _, t, h, w = samples.shape

        time_sf, w_sf, h_sf = get_vae_scale_factors(vae)

        if last_frame_fix and t > 0:
            last_frame = samples[:, :, -1:, :, :]
            samples = torch.cat([samples, last_frame], dim=2)
            t = int(samples.shape[2])

        t_out = 1 + (t - 1) * int(time_sf)
        out_h = int(h) * int(h_sf)
        out_w = int(w) * int(w_sf)

        horizontal_tiles = max(1, int(horizontal_tiles))
        vertical_tiles = max(1, int(vertical_tiles))
        overlap_latent = max(0, int(overlap_latent))

        base_tile_h = (int(h) + (vertical_tiles - 1) * overlap_latent) // vertical_tiles
        base_tile_w = (int(w) + (horizontal_tiles - 1) * overlap_latent) // horizontal_tiles

        output = None
        weights = None

        for vv in range(vertical_tiles):
            for hh in range(horizontal_tiles):
                w_start = hh * (base_tile_w - overlap_latent)
                h_start = vv * (base_tile_h - overlap_latent)
                w_end = min(w_start + base_tile_w, int(w)) if hh < horizontal_tiles - 1 else int(w)
                h_end = min(h_start + base_tile_h, int(h)) if vv < vertical_tiles - 1 else int(h)

                tile = samples[:, :, :, h_start:h_end, w_start:w_end]
                decoded_tile = vae.decode(tile)
                if not isinstance(decoded_tile, torch.Tensor):
                    raise ValueError("vae.decode did not return a torch.Tensor for video tile")

                if decoded_tile.dim() == 4:
                    decoded_tile = decoded_tile.unsqueeze(1)
                elif decoded_tile.dim() != 5:
                    raise RuntimeError(f"Unexpected decoded tile shape: {tuple(decoded_tile.shape)}")

                if int(decoded_tile.shape[0]) != int(b):
                    raise RuntimeError("Decoded tile batch mismatch")

                c_out = int(decoded_tile.shape[-1])
                if c_out < 3:
                    raise RuntimeError("Decoded tile channels must be >=3")

                if output is None:
                    output = torch.zeros(
                        (b, t_out, out_h, out_w, c_out),
                        device=decoded_tile.device,
                        dtype=decoded_tile.dtype,
                    )
                    weights = torch.zeros(
                        (b, t_out, out_h, out_w, 1),
                        device=decoded_tile.device,
                        dtype=decoded_tile.dtype,
                    )

                out_h_start = int(h_start) * int(h_sf)
                out_h_end = int(h_end) * int(h_sf)
                out_w_start = int(w_start) * int(w_sf)
                out_w_end = int(w_end) * int(w_sf)

                expected_h = out_h_end - out_h_start
                expected_w = out_w_end - out_w_start

                dec_h = int(decoded_tile.shape[2])
                dec_w = int(decoded_tile.shape[3])

                if dec_h != expected_h or dec_w != expected_w:
                    mh = min(dec_h, expected_h)
                    mw = min(dec_w, expected_w)
                    decoded_tile = decoded_tile[:, :, :mh, :mw, :]
                    expected_h = mh
                    expected_w = mw
                    out_h_end = out_h_start + mh
                    out_w_end = out_w_start + mw

                tile_weights = torch.ones(
                    (b, t_out, expected_h, expected_w, 1),
                    device=decoded_tile.device,
                    dtype=decoded_tile.dtype,
                )

                overlap_out_h = min(int(overlap_latent) * int(h_sf), expected_h)
                overlap_out_w = min(int(overlap_latent) * int(w_sf), expected_w)

                if hh > 0 and overlap_out_w > 0:
                    hb = torch.linspace(0, 1, overlap_out_w, device=decoded_tile.device, dtype=decoded_tile.dtype)
                    tile_weights[:, :, :, :overlap_out_w, :] *= hb.view(1, 1, 1, -1, 1)
                if hh < horizontal_tiles - 1 and overlap_out_w > 0:
                    hb = torch.linspace(1, 0, overlap_out_w, device=decoded_tile.device, dtype=decoded_tile.dtype)
                    tile_weights[:, :, :, -overlap_out_w:, :] *= hb.view(1, 1, 1, -1, 1)

                if vv > 0 and overlap_out_h > 0:
                    vb = torch.linspace(0, 1, overlap_out_h, device=decoded_tile.device, dtype=decoded_tile.dtype)
                    tile_weights[:, :, :overlap_out_h, :, :] *= vb.view(1, 1, -1, 1, 1)
                if vv < vertical_tiles - 1 and overlap_out_h > 0:
                    vb = torch.linspace(1, 0, overlap_out_h, device=decoded_tile.device, dtype=decoded_tile.dtype)
                    tile_weights[:, :, -overlap_out_h:, :, :] *= vb.view(1, 1, -1, 1, 1)

                t_dec = int(decoded_tile.shape[1])
                if t_dec == t_out:
                    decoded_for_add = decoded_tile
                elif t_dec == 1:
                    decoded_for_add = decoded_tile.repeat(1, t_out, 1, 1, 1)
                else:
                    if t_out % t_dec == 0:
                        factor = t_out // t_dec
                        decoded_for_add = decoded_tile.repeat(1, factor, 1, 1, 1)
                    else:
                        if t_dec > t_out:
                            decoded_for_add = decoded_tile[:, :t_out, :, :, :]
                        else:
                            reps = (t_out + t_dec - 1) // t_dec
                            decoded_for_add = decoded_tile.repeat(1, reps, 1, 1, 1)[:, :t_out, :, :, :]

                output[:, :, out_h_start:out_h_end, out_w_start:out_w_end, :] += decoded_for_add * tile_weights
                weights[:, :, out_h_start:out_h_end, out_w_start:out_w_end, :] += tile_weights

        output = output / (weights + 1e-8)

        if bool(last_frame_fix) and int(time_sf) > 0:
            output = output[:, :-int(time_sf), :, :, :]

        return output.contiguous()


def decode_for_edge_detection(
    vae,
    latent_bchw_or_flat: torch.Tensor,
    meta: dict[str, Any],
    cfg: EdgeBlendConfig,
) -> tuple[torch.Tensor, tuple[int, int], int]:
    """Decode a latent to the pixels the edge mask is found in.

    Args:
        vae: The VAE to decode with.
        latent_bchw_or_flat: The flattened ``[B', C, H, W]`` latent.
        meta: The record :func:`normalize_latent_to_bchw` returned.
        cfg: The settings, read here for the video tiling.

    Returns:
        ``(images_bhwc, (himg, wimg), frames_out)``, the decoded pixels, the size of one
        frame, and how many frames each clip decoded to, which is 1 for an image.

    Raises:
        ValueError: The flattened latent does not match the record, or the decode came back
            with the wrong batch or too few channels.
    """
    layout = meta.get("layout", "4d_bchw")

    if layout == "4d_bchw":
        images = decode_image_latent_regular(vae, latent_bchw_or_flat)
        _, himg, wimg, _ = images.shape
        return images, (int(himg), int(wimg)), 1

    b = int(meta["B"])
    c = int(meta["C"])
    t = int(meta["T"])
    h = int(meta["H"])
    w = int(meta["W"])

    if latent_bchw_or_flat.dim() != 4:
        raise ValueError(f"Expected flattened [B*T,C,H,W], got {tuple(latent_bchw_or_flat.shape)}")
    if int(latent_bchw_or_flat.shape[0]) != b * t or int(latent_bchw_or_flat.shape[1]) != c:
        raise ValueError(
            f"Video flatten mismatch: expected {(b * t, c, h, w)}, "
            f"got {tuple(latent_bchw_or_flat.shape)}"
        )

    latent_bcthw = latent_bchw_or_flat.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()

    images_bt = decode_video_latent_lazy_tiled(
        vae=vae,
        latent_bcthw=latent_bcthw,
        horizontal_tiles=cfg.video_decode_horizontal_tiles,
        vertical_tiles=cfg.video_decode_vertical_tiles,
        overlap_latent=cfg.video_decode_overlap_latent,
        last_frame_fix=cfg.video_decode_last_frame_fix,
        enable_cudnn=cfg.video_decode_enable_cudnn,
    )

    b2, t_out, himg, wimg, ch = images_bt.shape
    if int(b2) != b:
        raise ValueError(f"Decoded batch mismatch: got {int(b2)} expected {b}")
    if int(ch) < 3:
        raise ValueError("Decoded channels must be >=3")

    images_bhwc = images_bt.reshape(b * t_out, int(himg), int(wimg), int(ch)).contiguous()
    return images_bhwc, (int(himg), int(wimg)), int(t_out)


def build_edge_masks(
    vae,
    latent_samples_bchw_or_flat: torch.Tensor,
    meta: dict[str, Any],
    target_latent_size: tuple[int, int],
    cfg: EdgeBlendConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find the edges in the decoded picture and return them as two masks.

    Args:
        vae: The VAE to decode with.
        latent_samples_bchw_or_flat: The flattened ``[B', C, H, W]`` latent.
        meta: The record :func:`normalize_latent_to_bchw` returned.
        target_latent_size: The ``(height, width)`` the latent is being enlarged to.
        cfg: The edge-finding settings.

    Returns:
        ``(mask_img, mask_lat)``, ``[B', 1, H_px, W_px]`` and ``[B', 1, Ht, Wt]``, both
        float32 on the latent's device.

    Raises:
        ValueError: The decode did not produce a usable image.
    """
    images, (himg, wimg), _ = decode_for_edge_detection(
        vae=vae,
        latent_bchw_or_flat=latent_samples_bchw_or_flat,
        meta=meta,
        cfg=cfg,
    )

    if images.dim() != 4 or int(images.shape[-1]) < 3:
        raise ValueError(f"Decoded images must be [B',H,W,C>=3], got {tuple(images.shape)}")

    if meta["layout"] == "4d_bchw":
        _, _, h_lat, w_lat = latent_samples_bchw_or_flat.shape
    else:
        h_lat = int(meta["H"])
        w_lat = int(meta["W"])

    scale_y = float(himg) / float(h_lat)
    scale_x = float(wimg) / float(w_lat)

    ht, wt = target_latent_size
    target_himg = max(1, int(round(ht * scale_y)))
    target_wimg = max(1, int(round(wt * scale_x)))

    pixels = images[..., :3].permute(0, 3, 1, 2).float().clamp(0.0, 1.0).mul_(255.0).round_()
    gray = luminance(pixels)

    pre_sigma = float(cfg.pre_blur_sigma_px)
    if pre_sigma > 0.0:
        gray = gaussian_blur(gray, sigma=pre_sigma)

    edges = canny(
        gray,
        int(cfg.canny_threshold1),
        int(cfg.canny_threshold2),
        l2=bool(cfg.canny_l2gradient),
    ).float()

    dilate_kernel = make_elliptical_kernel(cfg.dilate_radius_px, device=edges.device)
    if dilate_kernel is not None:
        edges = dilate(edges, dilate_kernel)

    feather_sigma = float(cfg.feather_sigma_px)
    if feather_sigma > 0.0:
        edges = gaussian_blur(edges, sigma=feather_sigma)

    mask_f = edges.clamp(0.0, 1.0)

    if (target_himg != himg) or (target_wimg != wimg):
        mask_img = F.interpolate(
            mask_f, size=(target_himg, target_wimg), mode="bilinear", align_corners=False
        )
    else:
        mask_img = mask_f

    if (target_himg != ht) or (target_wimg != wt):
        mask_lat = F.interpolate(mask_img, size=(ht, wt), mode="area")
    else:
        mask_lat = mask_img

    device = latent_samples_bchw_or_flat.device
    low, high = float(cfg.mask_min), float(cfg.mask_max)
    return (
        mask_img.clamp(low, high).to(device=device, dtype=torch.float32),
        mask_lat.clamp(low, high).to(device=device, dtype=torch.float32),
    )


def run_hybrid_upscale_with_edge_mask(
    latent_samples: torch.Tensor,
    cfg: EdgeBlendConfig,
    vae,
    donor_latent: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Enlarge a latent, taking the smooth enlargement only along edges.

    Args:
        latent_samples: The latent to enlarge, 4D or 5D.
        cfg: The settings.
        vae: The VAE used to decode for edge finding.
        donor_latent: A second latent to take the smooth half from instead of interpolating
            the first, or ``None``.

    Returns:
        ``(up_latent, mask_img, mask_lat)``, the enlarged latent in the layout it arrived
        in, and the edge mask at picture and at latent resolution.

    Raises:
        ValueError: The donor does not match the latent, or the scale produces no size.
    """
    latent_bchw, meta = normalize_latent_to_bchw(latent_samples)

    donor_bchw = None
    if donor_latent is not None:
        donor_bchw, donor_meta = normalize_latent_to_bchw(donor_latent)
        for k in ("layout", "B", "C"):
            if donor_meta.get(k) != meta.get(k):
                raise ValueError(
                    f"donor_latent mismatch on {k}: latent={meta.get(k)} donor={donor_meta.get(k)}"
                )
        if meta["layout"] != "4d_bchw" and int(donor_meta.get("T", -1)) != int(meta.get("T", -1)):
            raise ValueError(
                f"donor_latent mismatch on T: latent={meta.get('T')} donor={donor_meta.get('T')}"
            )

    _, _, h, w = latent_bchw.shape
    ht = int(round(h * float(cfg.scale)))
    wt = int(round(w * float(cfg.scale)))
    if ht <= 0 or wt <= 0:
        raise ValueError("Invalid target size computed from scale.")
    target_latent_size = (ht, wt)

    if cfg.nearest_exact:
        base = upscale_latent_nearest_exact(latent_bchw, target_latent_size)
    else:
        base = F.interpolate(latent_bchw, size=target_latent_size, mode="nearest")

    if donor_bchw is not None:
        donor = F.interpolate(
            donor_bchw, size=target_latent_size, mode="bilinear", align_corners=cfg.align_corners
        )
    else:
        donor = upscale_latent_bilinear(latent_bchw, target_latent_size, cfg.align_corners)

    mask_img, mask_lat = build_edge_masks(
        vae=vae,
        latent_samples_bchw_or_flat=latent_bchw,
        meta=meta,
        target_latent_size=target_latent_size,
        cfg=cfg,
    )

    if meta["layout"] != "4d_bchw":
        b = int(meta["B"])
        t = int(meta["T"])
        b_flat = int(latent_bchw.shape[0])
        b_prime = int(mask_lat.shape[0])

        # A video VAE decodes one latent frame into several picture frames, so there are
        # more masks than latent frames. Take the mask of the first picture frame each
        # latent frame produced.
        if b_prime != b_flat:
            time_sf, _, _ = get_vae_scale_factors(vae)
            time_sf = max(1, int(time_sf))

            t_out = b_prime // b
            m = mask_lat.reshape(b, t_out, 1, ht, wt)

            idx = torch.arange(0, t * time_sf, step=time_sf, device=m.device)
            idx = torch.clamp(idx, 0, t_out - 1)
            m_sel = torch.index_select(m, dim=1, index=idx)

            mask_lat = m_sel.reshape(b_flat, 1, ht, wt)

    mask_lat = mask_lat.to(dtype=base.dtype)
    out_bchw = base * (1.0 - mask_lat) + donor * mask_lat

    out_latent = restore_latent_from_bchw(out_bchw, meta)
    return out_latent, mask_img, mask_lat


class LatentUpscaleHybrid(io.ComfyNode):
    """Blend a blocky and a smooth latent enlargement using a decoded edge mask."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASNSLatentUpscaleHybrid",
            display_name="Latent Hybrid Upscale",
            search_aliases=[
                "WASNSLatentUpscaleHybrid",
                "Latent Hybrid Upscale",
                "hybrid latent upscale",
                "edge mask upscale",
                "canny latent",
            ],
            category="WAS Suite/Latent/Transform",
            description=(
                "Enlarge a latent and decide where to be smooth by looking at the picture "
                "it decodes to. Edges found in that picture are grown and feathered into a "
                "mask; where the mask is white the enlargement is interpolated, and "
                "everywhere else it keeps the crisp block-copied version. Flat areas "
                "therefore stay sharp while outlines avoid the stair-stepping that a plain "
                "enlargement leaves. Handles video latents, with tiled decoding to keep "
                "VRAM in check."
            ),
            inputs=[
                io.Latent.Input(
                    "latent",
                    tooltip=(
                        "The latent to enlarge. A video latent with a time axis is handled "
                        "frame by frame."
                    ),
                ),
                io.Vae.Input(
                    "vae",
                    tooltip=(
                        "The VAE used to decode the latent so its edges can be found. It "
                        "must be the one that matches the latent, or the edges will be "
                        "found in the wrong places. It only reads the latent; the result is "
                        "still built in latent space."
                    ),
                ),
                io.Float.Input(
                    "scale",
                    default=2.0,
                    min=1.0,
                    max=8.0,
                    step=0.01,
                    tooltip=(
                        "How much larger the result is. 2.0 doubles both sides, 1.5 adds "
                        "half again. Sizes are rounded to whole latent blocks."
                    ),
                ),
                io.Float.Input(
                    "pre_blur_sigma_px",
                    default=1.0,
                    min=0.0,
                    max=20.0,
                    step=0.01,
                    tooltip=(
                        "Blur applied to the decoded picture before edges are looked for, "
                        "in pixels. It keeps film grain and fine texture from registering "
                        "as edges; 0.0 finds every last one, 2.0 or more keeps only the "
                        "major outlines."
                    ),
                ),
                io.Int.Input(
                    "canny_threshold1",
                    default=25,
                    min=0,
                    max=1000,
                    step=1,
                    tooltip=(
                        "The lower of the two edge-detection levels, on a 0-255 scale. A "
                        "faint edge is kept only when it joins a strong one, and this is "
                        "how faint it may be. Lower values trace more of an outline; raise "
                        "it if speckles appear in flat areas."
                    ),
                ),
                io.Int.Input(
                    "canny_threshold2",
                    default=155,
                    min=0,
                    max=1000,
                    step=1,
                    tooltip=(
                        "The upper of the two edge-detection levels, on a 0-255 scale. "
                        "Anything this strong starts an edge on its own. Raise it to keep "
                        "only bold outlines, lower it to catch soft ones."
                    ),
                ),
                io.Boolean.Input(
                    "canny_l2gradient",
                    default=True,
                    tooltip=(
                        "How edge strength is measured. On, the true length of the gradient "
                        "is used, which is slightly slower and more accurate on diagonals. "
                        "Off, a cheaper approximation is used that reads diagonal edges as "
                        "stronger than they are."
                    ),
                ),
                io.Int.Input(
                    "dilate_radius_px",
                    default=8,
                    min=0,
                    max=64,
                    step=1,
                    tooltip=(
                        "How far the found edges are grown, in pixels. Edges are hairline "
                        "by themselves, so growing them is what gives the smooth "
                        "enlargement a band to work in: 8 covers a typical outline, 0 "
                        "leaves the raw one-pixel lines."
                    ),
                ),
                io.Float.Input(
                    "feather_sigma_px",
                    default=6.0,
                    min=0.0,
                    max=50.0,
                    step=0.01,
                    tooltip=(
                        "How far the grown edge fades out, in pixels. Without it the band "
                        "would have a visible border of its own; 6.0 gives a soft "
                        "changeover, 0.0 leaves a hard-edged band."
                    ),
                ),
                io.Float.Input(
                    "mask_min",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Floor under the finished mask. Raise it above 0.0 to let a little "
                        "of the smooth enlargement into areas with no edges at all, which "
                        "takes the hard blockiness off the whole picture."
                    ),
                ),
                io.Float.Input(
                    "mask_max",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Ceiling over the finished mask. Lower it below 1.0 to keep some of "
                        "the crisp enlargement even on the strongest edges, which is the "
                        "way back when outlines come out too soft."
                    ),
                ),
                io.Boolean.Input(
                    "use_nearest_exact",
                    default=True,
                    tooltip=(
                        "How the crisp half of the blend is enlarged. On, each output block "
                        "takes the value of the source block whose centre is nearest, which "
                        "keeps the picture from drifting half a block sideways. Off uses "
                        "the older nearest-neighbour rule."
                    ),
                ),
                io.Combo.Input(
                    "output_mask_resolution",
                    options=MASK_RESOLUTIONS,
                    default="image",
                    tooltip=(
                        "Which size the mask output comes out at. `image` gives it at the "
                        "size the enlarged latent decodes to, ready to view or reuse "
                        "against the finished picture. `latent` gives the small version "
                        "that actually drove the blend."
                    ),
                ),
                io.Int.Input(
                    "video_decode_horizontal_tiles",
                    default=2,
                    min=1,
                    max=8,
                    tooltip=(
                        "How many columns a video latent is split into for the decode that "
                        "finds edges. More tiles means less VRAM and more time. Ignored on "
                        "an image latent."
                    ),
                ),
                io.Int.Input(
                    "video_decode_vertical_tiles",
                    default=2,
                    min=1,
                    max=8,
                    tooltip=(
                        "How many rows a video latent is split into for that decode. 2 rows "
                        "and 2 columns is four tiles, each a quarter of the frame. Ignored "
                        "on an image latent."
                    ),
                ),
                io.Int.Input(
                    "video_decode_overlap_latent",
                    default=4,
                    min=0,
                    max=32,
                    tooltip=(
                        "How far neighbouring tiles overlap, in latent units. The overlap "
                        "is cross-faded, so raise it if seams show along the tile "
                        "boundaries; 0 turns the fade off entirely."
                    ),
                ),
                io.Boolean.Input(
                    "video_decode_last_frame_fix",
                    default=False,
                    tooltip=(
                        "Whether the final frame is duplicated before decoding and the "
                        "extra output dropped afterwards. Turn it on when the last frames "
                        "of a clip decode to something corrupt, which some video VAEs do."
                    ),
                ),
                io.Boolean.Input(
                    "video_decode_enable_cudnn",
                    default=True,
                    tooltip=(
                        "Whether cuDNN is left on for the video decode. Turning it off is "
                        "slower and avoids the large workspace allocations that make some "
                        "cards run out of memory part way through a clip."
                    ),
                ),
                io.Latent.Input(
                    "donor_latent",
                    optional=True,
                    tooltip=(
                        "Where the smooth half of the blend comes from. Leave it "
                        "unconnected and the node interpolates the input latent. Connect a "
                        "second latent of the same batch and channel shape, a version "
                        "sampled at a higher resolution, say, and its detail is what gets "
                        "laid into the edges."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output(
                    display_name="latent",
                    tooltip="The enlarged latent, ready for a sampler or a decode.",
                ),
                io.Mask.Output(
                    display_name="edge_mask",
                    tooltip=(
                        "Where the smooth enlargement was used: white along the edges the "
                        "node found, black elsewhere. Watch it while setting the two Canny "
                        "levels, or reuse it to treat the same areas downstream."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        latent,
        vae,
        scale,
        pre_blur_sigma_px,
        canny_threshold1,
        canny_threshold2,
        canny_l2gradient,
        dilate_radius_px,
        feather_sigma_px,
        mask_min,
        mask_max,
        use_nearest_exact,
        output_mask_resolution,
        video_decode_horizontal_tiles,
        video_decode_vertical_tiles,
        video_decode_overlap_latent,
        video_decode_last_frame_fix,
        video_decode_enable_cudnn,
        donor_latent=None,
    ) -> io.NodeOutput:
        """Upscale the latent along its edges.

        Raises:
            ValueError: Nothing is connected to the latent input.
        """
        require_input(
            latent,
            "Latent Hybrid Upscale",
            "latent",
            "latent",
            "latent source such as Empty Latent Image or VAE Encode",
            "LATENT",
        )

        latent_samples = latent["samples"]
        donor_samples = donor_latent["samples"] if donor_latent is not None else None

        cfg = EdgeBlendConfig(
            scale=float(scale),
            pre_blur_sigma_px=float(pre_blur_sigma_px),
            canny_threshold1=int(canny_threshold1),
            canny_threshold2=int(canny_threshold2),
            canny_l2gradient=bool(canny_l2gradient),
            dilate_radius_px=int(dilate_radius_px),
            feather_sigma_px=float(feather_sigma_px),
            mask_min=float(mask_min),
            mask_max=float(mask_max),
            nearest_exact=bool(use_nearest_exact),
            output_mask_resolution=str(output_mask_resolution).strip().lower(),
            video_decode_horizontal_tiles=int(video_decode_horizontal_tiles),
            video_decode_vertical_tiles=int(video_decode_vertical_tiles),
            video_decode_overlap_latent=int(video_decode_overlap_latent),
            video_decode_last_frame_fix=bool(video_decode_last_frame_fix),
            video_decode_enable_cudnn=bool(video_decode_enable_cudnn),
        )

        up_latent, mask_img, mask_lat = run_hybrid_upscale_with_edge_mask(
            latent_samples=latent_samples,
            cfg=cfg,
            vae=vae,
            donor_latent=donor_samples,
        )

        out = dict(latent)
        out["samples"] = up_latent

        if cfg.output_mask_resolution == "latent":
            edge_mask = latent_mask_to_comfy_mask(mask_lat)
        else:
            edge_mask = latent_mask_to_comfy_mask(mask_img)

        return io.NodeOutput(out, edge_mask)

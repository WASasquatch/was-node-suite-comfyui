"""Add micro-detail to a latent with a contrast-limited band-pass."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ....modules.compat.sockets import require_input

REQUIRES = "extras"

NODE = "Latent Detail Boost"

PREVIEW_MODES = ["edge_mask", "detail_mask"]


class LatentContrastLimitedDetailBoost(io.ComfyNode):
    """Enhance fine latent detail without the halos an unsharp mask leaves."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASNSLatentContrastLimitedDetailBoost",
            display_name=NODE,
            search_aliases=[
                "WASNSLatentContrastLimitedDetailBoost",
                NODE,
                "latent sharpen",
                "detail boost",
                "band pass",
                "h3 detail",
            ],
            category="WAS Suite/Latent",
            description=(
                "Bring out fine texture in an image, video or MiniMax H3 latent before a "
                "second sampling pass or a decode, lifting smooth areas as much as busy ones "
                "without the dark outlines of a plain sharpen. An H3 latent's sound passes "
                "through untouched."
            ),
            inputs=[
                io.Latent.Input(
                    "latent",
                    tooltip=(
                        "The latent to enhance: image, video, or a packed audio and video "
                        "latent such as MiniMax H3. Video is treated frame by frame; the "
                        "audio half comes back unchanged."
                    ),
                ),
                io.Float.Input(
                    "sigma_small",
                    default=0.6,
                    min=0.0,
                    max=8.0,
                    step=0.05,
                    tooltip=(
                        "The fine end of the detail that is boosted, in latent blocks. "
                        "Together with sigma_large it picks which size of feature is "
                        "affected: 0.6 keeps the treatment on the smallest structures. Set "
                        "to 0.0 to reach the very finest detail, including noise."
                    ),
                ),
                io.Float.Input(
                    "sigma_large",
                    default=1.4,
                    min=0.0,
                    max=16.0,
                    step=0.05,
                    tooltip=(
                        "The coarse end of the detail that is boosted, in latent blocks. "
                        "Widening the gap between it and sigma_small treats larger features "
                        "as well; 1.4 against 0.6 gives a narrow band that reads as "
                        "texture. The two are swapped if entered the wrong way round."
                    ),
                ),
                io.Float.Input(
                    "gain",
                    default=0.35,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    tooltip=(
                        "How much of the isolated detail is added back. 0.0 leaves the "
                        "latent alone, 0.35 is a gentle lift, and above about 1.0 the "
                        "texture starts to dominate the picture."
                    ),
                ),
                io.Float.Input(
                    "limit",
                    default=1.25,
                    min=0.1,
                    max=8.0,
                    step=0.05,
                    tooltip=(
                        "How hard the detail is squashed before it is added, which is what "
                        "stops a strong edge ringing. High values such as 4.0 flatten the "
                        "strongest detail to a uniform level; 1.25 keeps most of the "
                        "variation; 0.1 barely limits at all."
                    ),
                ),
                io.Float.Input(
                    "rms_sigma",
                    default=1.2,
                    min=0.0,
                    max=16.0,
                    step=0.05,
                    tooltip=(
                        "Area the local contrast is measured over, in latent blocks: 0.0 = the "
                        "whole frame, sharpening everything equally; 1.2 = a smooth sky gains "
                        "as much texture as a busy tree."
                    ),
                ),
                io.Float.Input(
                    "rms_floor",
                    default=0.06,
                    min=0.0,
                    max=1.0,
                    step=0.005,
                    tooltip=(
                        "A floor under that local measurement, which keeps genuinely flat "
                        "areas from being amplified into noise. Raise it towards 0.2 if a "
                        "clear sky or a plain wall comes out grainy; lower it towards 0.0 "
                        "to treat flat areas as hard as everything else."
                    ),
                ),
                io.Float.Input(
                    "edge_protect",
                    default=0.45,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How much the enhancement is held back on strong boundaries, which "
                        "is what prevents dark outlines around objects. 0.0 turns the "
                        "protection off and skips finding edges at all; 1.0 leaves "
                        "boundaries completely untouched; 0.45 halves the effect there."
                    ),
                ),
                io.Float.Input(
                    "edge_sigma",
                    default=0.8,
                    min=0.0,
                    max=8.0,
                    step=0.05,
                    tooltip=(
                        "Blur applied before boundaries are looked for, in latent blocks. "
                        "Raise it so that fine texture is not counted as an edge and "
                        "protected from the very treatment it wants; 0.0 finds the finest "
                        "boundaries."
                    ),
                ),
                io.Float.Input(
                    "edge_threshold",
                    default=0.25,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How strong a boundary has to be to be protected, measured against "
                        "the strongest one in the picture. 0.25 covers the clear outlines; "
                        "lower values protect more and enhance less."
                    ),
                ),
                io.Float.Input(
                    "edge_softness",
                    default=0.10,
                    min=0.0005,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How gradually the protection fades in around that threshold. Small "
                        "values give a hard-edged protected strip that can be seen; 0.10 "
                        "fades over a comfortable range."
                    ),
                ),
                io.Int.Input(
                    "preview_mask_scale",
                    default=8,
                    min=1,
                    max=16,
                    step=1,
                    tooltip=(
                        "Preview pixels per latent block. 8 = the decoded size on most image "
                        "VAEs; 16 = the decoded size of MiniMax H3; 1 = the raw map. Changes "
                        "the preview image only, never the mask output."
                    ),
                ),
                io.Combo.Input(
                    "preview_mode",
                    options=PREVIEW_MODES,
                    default="detail_mask",
                    tooltip=(
                        "Which map leaves the node. `detail_mask` shows where detail was "
                        "added and how much, which is what to watch while setting gain. "
                        "`edge_mask` shows the boundaries that were protected, which is "
                        "what to watch while setting edge_threshold."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output(
                    display_name="latent",
                    tooltip=(
                        "The enhanced latent, ready for a sampler or a decode. A packed "
                        "audio and video latent stays packed."
                    ),
                ),
                io.Mask.Output(
                    display_name="mask",
                    tooltip=(
                        "The chosen map at latent resolution, one per image or frame, for "
                        "driving another node from the same areas this one treated."
                    ),
                ),
                io.Image.Output(
                    display_name="mask_preview",
                    tooltip=(
                        "The same map as a viewable image, enlarged by preview_mask_scale."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        latent,
        sigma_small,
        sigma_large,
        gain,
        limit,
        rms_sigma,
        rms_floor,
        edge_protect,
        edge_sigma,
        edge_threshold,
        edge_softness,
        preview_mask_scale,
        preview_mode,
    ) -> io.NodeOutput:
        """Raise the latent's local detail.

        Raises:
            ValueError: Nothing is connected to the latent input, the latent holds no
                samples, or its picture stream is not an image or video tensor.
        """
        from ....modules.latent.filters import clamp01, gaussian_blur_depthwise, mask_preview, sobel_grad_mag
        from ....modules.latent.shape import flatten_5d_to_4d, is_latent_5d, unflatten_4d_to_5d

        require_input(
            latent,
            NODE,
            "latent",
            "latent",
            "latent source such as Empty Latent Image or VAE Encode",
            "LATENT",
        )
        samples = latent.get("samples") if isinstance(latent, dict) else None
        if samples is None:
            raise ValueError(
                f"{NODE} was given a latent with nothing under 'samples'. Wire a sampler, a "
                f"VAE Encode or an Empty Latent Image into its latent input."
            )

        # A packed latent is boosted on its first stream, the picture; the rest pass through.
        packed = getattr(samples, "is_nested", False)
        streams = list(samples.unbind()) if packed else [samples]
        picture = streams[0] if streams else None
        if not torch.is_tensor(picture) or picture.ndim not in (4, 5):
            shape = tuple(picture.shape) if torch.is_tensor(picture) else None
            raise ValueError(
                f"{NODE} needs a picture latent shaped [batch, channels, height, width] or "
                f"[batch, channels, frames, height, width], and this one is shaped {shape}. "
                f"An audio-only latent has no picture to boost; wire an image or video latent."
            )
        if picture.numel() == 0:
            raise ValueError(
                f"{NODE} was given an empty latent shaped {tuple(picture.shape)}. Check the "
                f"batch size, frame count and size of the node that made it."
            )
        orig_dtype = picture.dtype

        was_5d = is_latent_5d(picture)
        if was_5d:
            x, b, t = flatten_5d_to_4d(picture)
        else:
            x = picture
            b, t = x.shape[0], 1

        # Work in float32 whatever the latent's own precision.
        x_fp = x.float()

        s_small = float(sigma_small)
        s_large = float(sigma_large)
        if s_large < s_small:
            s_small, s_large = s_large, s_small

        # The difference of two blurs keeps one band of detail.
        low_small = gaussian_blur_depthwise(x_fp, s_small) if s_small > 0.0 else x_fp
        low_large = gaussian_blur_depthwise(x_fp, s_large) if s_large > 0.0 else x_fp
        dog = low_small - low_large

        # The band divided by its local RMS contrast.
        if float(rms_sigma) > 0.0:
            rms = gaussian_blur_depthwise(dog * dog, float(rms_sigma))
            rms = torch.sqrt(torch.clamp(rms, min=0.0) + 1e-8)
        else:
            rms = torch.sqrt(torch.mean(dog * dog, dim=(2, 3), keepdim=True) + 1e-8)

        rms = rms + float(rms_floor)
        dog_n = dog / rms

        # tanh squashes the extremes and leaves the middle nearly linear.
        lim = float(limit)
        dog_l = torch.tanh(dog_n * lim) / max(1e-6, lim)

        detail_mag = dog_l.abs().mean(dim=1, keepdim=True)
        detail_mag = detail_mag / (detail_mag.amax(dim=(2, 3), keepdim=True) + 1e-8)
        detail_mag = clamp01(detail_mag)

        if float(edge_protect) > 0.0:
            energy = x_fp.abs().mean(dim=1, keepdim=True)
            if float(edge_sigma) > 0.0:
                energy = gaussian_blur_depthwise(energy, float(edge_sigma))
            gmag = sobel_grad_mag(energy)
            gmag = gmag / (gmag.amax(dim=(2, 3), keepdim=True) + 1e-8)

            t0 = float(edge_threshold)
            s0 = max(1e-6, float(edge_softness))
            edge = torch.sigmoid((gmag - t0) / s0)
            edge = clamp01(edge)

            protect = float(edge_protect)
            edge_gate = 1.0 - protect * edge
            edge_gate = torch.clamp(edge_gate, 0.0, 1.0)
        else:
            edge = torch.zeros_like(detail_mag)
            edge_gate = 1.0

        out = x_fp + float(gain) * dog_l * edge_gate
        out = out.to(dtype=orig_dtype)

        if was_5d:
            out = unflatten_4d_to_5d(out, b=b, t=t)
        if packed:
            import comfy.nested_tensor

            out = comfy.nested_tensor.NestedTensor([out, *streams[1:]])

        out_latent = dict(latent)
        out_latent["samples"] = out

        if preview_mode == "edge_mask":
            mask_bt = edge[:, 0]
            prev_img = mask_preview(edge, preview_mask_scale)
        else:
            mask_bt = detail_mag[:, 0]
            prev_img = mask_preview(detail_mag, preview_mask_scale)

        return io.NodeOutput(out_latent, mask_bt.contiguous(), prev_img)

"""Add micro-detail to a latent with a contrast-limited band-pass."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ....modules.compat.sockets import require_input

REQUIRES = "extras"

PREVIEW_MODES = ["edge_mask", "detail_mask"]


class LatentContrastLimitedDetailBoost(io.ComfyNode):
    """Enhance fine latent detail without the halos an unsharp mask leaves."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASNSLatentContrastLimitedDetailBoost",
            display_name="WAS Latent Detail Boost",
            search_aliases=[
                "WASNSLatentContrastLimitedDetailBoost",
                "WAS Latent Detail Boost",
                "latent sharpen",
                "detail boost",
                "band pass",
            ],
            category="WAS Suite/Latent",
            description=(
                "Bring out fine detail in a latent by isolating one band of detail, "
                "levelling it against the local amount of contrast and adding it back. The "
                "added detail is normalised and limited before it lands: a busy area and a "
                "smooth one gain the same amount, and the dark outlines and embossed look "
                "that come from sharpening a latent directly do not appear. Handles video "
                "latents as well as single images."
            ),
            inputs=[
                io.Latent.Input(
                    "latent",
                    tooltip=(
                        "The latent to enhance. A video latent with a time axis is handled "
                        "frame by frame."
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
                        "How large an area the local amount of contrast is measured over, "
                        "in latent blocks. It is what lets a smooth sky gain as much "
                        "texture as a busy tree instead of being left behind. 0.0 measures "
                        "over the whole frame instead, which restores the plain behaviour "
                        "of sharpening everything by the same amount."
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
                        "How many preview pixels each latent block becomes. 8 matches the "
                        "size the latent decodes to on most VAEs, so the preview lines up "
                        "with the finished picture; 1 gives the small raw map. This affects "
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
                    tooltip="The enhanced latent, ready for a sampler or a decode.",
                ),
                io.Mask.Output(
                    display_name="mask",
                    tooltip=(
                        "The chosen map at latent resolution, useful for driving another "
                        "node from the same areas this one treated."
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
            ValueError: Nothing is connected to the latent input, or the latent holds no
                samples.
        """
        from ....modules.latent.filters import clamp01, gaussian_blur_depthwise, mask_preview, sobel_grad_mag
        from ....modules.latent.shape import flatten_5d_to_4d, is_latent_5d, unflatten_4d_to_5d

        require_input(
            latent,
            "WAS Latent Detail Boost",
            "latent",
            "latent",
            "latent source such as Empty Latent Image or VAE Encode",
            "LATENT",
        )
        if "samples" not in latent:
            raise ValueError("LATENT input must be a dict containing key 'samples'.")

        samples: torch.Tensor = latent["samples"]
        orig_dtype = samples.dtype

        was_5d = is_latent_5d(samples)
        if was_5d:
            x, b, t = flatten_5d_to_4d(samples)
        else:
            x = samples
            b, t = x.shape[0], 1

        # Everything below runs in float32 whatever the latent's own precision: the
        # normalisation divides by a local measurement, and float16 loses the small
        # differences that decides.
        x_fp = x.float()

        s_small = float(sigma_small)
        s_large = float(sigma_large)
        if s_large < s_small:
            s_small, s_large = s_large, s_small

        # The difference of two blurs keeps one band of detail and drops everything
        # coarser, so the enhancement cannot lift broad brightness with it.
        low_small = gaussian_blur_depthwise(x_fp, s_small) if s_small > 0.0 else x_fp
        low_large = gaussian_blur_depthwise(x_fp, s_large) if s_large > 0.0 else x_fp
        dog = low_small - low_large

        # Dividing by the local amount of contrast is what makes this contrast-limited: a
        # quiet area and a busy one are lifted by the same amount instead of in proportion
        # to what is already there, which is where dark halos come from.
        if float(rms_sigma) > 0.0:
            rms = gaussian_blur_depthwise(dog * dog, float(rms_sigma))
            rms = torch.sqrt(torch.clamp(rms, min=0.0) + 1e-8)
        else:
            rms = torch.sqrt(torch.mean(dog * dog, dim=(2, 3), keepdim=True) + 1e-8)

        rms = rms + float(rms_floor)
        dog_n = dog / rms

        # tanh squashes the extremes and leaves the middle nearly linear, so the strongest
        # detail cannot overshoot into a ring.
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

        out_latent = dict(latent)
        out_latent["samples"] = out

        if preview_mode == "edge_mask":
            mask_bt = edge[:, 0]
            prev_img = mask_preview(edge, preview_mask_scale)
        else:
            mask_bt = detail_mag[:, 0]
            prev_img = mask_preview(detail_mag, preview_mask_scale)

        return io.NodeOutput(out_latent, mask_bt.contiguous(), prev_img)

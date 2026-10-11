"""Resize a latent so its longest side decodes to a chosen number of pixels."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat import limits
from ....modules.compat.sockets import require_input
from ....modules.interface import size_report

REQUIRES = "extras"

UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
SCALE_MODES = ["always", "downscale_only", "upscale_only"]


def resolve_spatial_compression(vae, fallback: int) -> int:
    """How many pixels one latent unit becomes along the height and width.

    Args:
        vae: A VAE to read the ratio from, or ``None`` to use ``fallback``. A VAE that does
            not report one, or reports something unusable, falls back as well.
        fallback: The ratio to use when the VAE cannot supply one. Values below 1 are
            raised to 1.

    Returns:
        The pixels-per-latent-unit ratio, at least 1.
    """
    if vae is not None:
        try:
            ratio = vae.spacial_compression_decode()
            if callable(ratio):
                ratio = ratio(1)
            ratio = int(round(float(ratio)))
            if ratio >= 1:
                return ratio
        except Exception:
            pass
    return max(1, int(fallback))


class LatentScaleToMaxDimension(io.ComfyNode):
    """Scale a latent to a pixel-space size cap, keeping its aspect ratio."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLatentScaleToMaxDimension",
            display_name="Scale Latent to Max Dimension",
            search_aliases=[
                "WASLatentScaleToMaxDimension",
                "Scale Latent to Max Dimension (WAS)",
                "latent max dimension",
                "latent resize",
                "longest side",
            ],
            category="WAS Suite/Latent/Transform",
            description=(
                "Resize a latent so the picture it decodes to has its longest side at a chosen "
                "number of pixels, aspect ratio kept, with no decode or re-encode. The new "
                "pixel width and height come out alongside the latent."
            ),
            inputs=[
                io.Latent.Input(
                    "samples",
                    tooltip=(
                        "The latent to resize. A video latent with a time axis is resized "
                        "on its height and width only, and keeps every frame."
                    ),
                ),
                io.Combo.Input(
                    "upscale_method",
                    options=UPSCALE_METHODS,
                    default="bislerp",
                    tooltip=(
                        "`nearest-exact` = copies the closest value, blocky; `area` = averages, "
                        "suits shrinking; `bilinear`, `bicubic` = ordinary smooth; `bislerp` = "
                        "follows the latent's shape, safest for a latent."
                    ),
                ),
                io.Int.Input(
                    "largest_size",
                    default=2048,
                    min=8,
                    max=limits.max_resolution(),
                    step=8,
                    tooltip=(
                        "How long the longest side should be, in pixels of the decoded "
                        "picture rather than in latent units. 2048 on a 3:2 latent gives "
                        "2048x1360. The target is rounded down to whole latent units, so "
                        "the result never comes out larger than asked for."
                    ),
                ),
                io.Combo.Input(
                    "scale_mode",
                    options=SCALE_MODES,
                    default="always",
                    tooltip=(
                        "`always` = hits the target from either side; `downscale_only` = "
                        "largest_size is a ceiling, smaller latents left alone; `upscale_only` "
                        "= brings small latents up, large ones left alone."
                    ),
                ),
                io.Int.Input(
                    "spatial_compression",
                    default=8,
                    min=1,
                    max=64,
                    step=1,
                    tooltip=(
                        "Pixels per latent unit on the decoding VAE: 8 = SD, SDXL, Flux, Wan "
                        "2.1; 16 = Wan 2.2 TI2V; 32 = Hunyuan Image. A wrong value scales the "
                        "result by the ratio. Ignored when vae is connected."
                    ),
                ),
                io.Vae.Input(
                    "vae",
                    optional=True,
                    tooltip=(
                        "The VAE this latent will be decoded with. Connect it and the "
                        "compression ratio is read straight off it, which removes the need "
                        "to know the right spatial_compression for the model in use."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output(
                    display_name="samples",
                    tooltip=(
                        "The resized latent. It is passed through untouched when scale_mode "
                        "rules the resize out, or when it is already the right size."
                    ),
                ),
                io.Int.Output(
                    display_name="width",
                    tooltip=(
                        "Width the latent now decodes to, in pixels. Feed it to anything "
                        "that has to be built at the same size, such as an empty image or a "
                        "second resize."
                    ),
                ),
                io.Int.Output(
                    display_name="height",
                    tooltip="Height the latent now decodes to, in pixels.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls, samples, upscale_method, largest_size, scale_mode, spatial_compression, vae=None
    ) -> io.NodeOutput:
        """Resize the latent to fit the longest side asked for.

        Raises:
            ValueError: Nothing is connected to the samples input, or the latent holds no
                samples.
        """
        import comfy.utils

        require_input(
            samples,
            "Scale Latent to Max Dimension (WAS)",
            "samples",
            "latent",
            "latent source such as Empty Latent Image or VAE Encode",
            "LATENT",
        )
        if "samples" not in samples:
            raise ValueError("LATENT input must be a dict containing key 'samples'.")

        latent = samples["samples"]
        lh, lw = int(latent.shape[-2]), int(latent.shape[-1])

        factor = resolve_spatial_compression(vae, spatial_compression)

        # Floor so the decoded image never overshoots largest_size, since only whole
        # latent units are addressable.
        max_latent = max(1, int(largest_size) // factor)
        longest = max(lh, lw)

        if scale_mode == "downscale_only" and longest <= max_latent:
            new_lw, new_lh = lw, lh
        elif scale_mode == "upscale_only" and longest >= max_latent:
            new_lw, new_lh = lw, lh
        elif lh > lw:
            new_lh = max_latent
            new_lw = max(1, min(max_latent, round(lw / lh * max_latent)))
        elif lw > lh:
            new_lw = max_latent
            new_lh = max(1, min(max_latent, round(lh / lw * max_latent)))
        else:
            new_lw = new_lh = max_latent

        out = samples.copy()
        if (new_lw, new_lh) != (lw, lh):
            out["samples"] = comfy.utils.common_upscale(
                latent, new_lw, new_lh, upscale_method, "disabled"
            )

        size_report.publish(
            (lw, lh),
            (new_lw, new_lh),
            action="scaled",
            unit=size_report.LATENT,
            facts={
                "decodes to": f"{new_lw * factor}x{new_lh * factor}",
                "longest": f"{max(new_lw, new_lh) * factor} of {int(largest_size)} asked",
            },
        )
        return io.NodeOutput(out, new_lw * factor, new_lh * factor)

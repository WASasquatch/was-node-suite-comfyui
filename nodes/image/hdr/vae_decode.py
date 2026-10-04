"""Decode a latent keeping the values above white and below black that the VAE produced."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.image import vae as vae_range
from ....modules.interface import image_report

logger = log.get_logger("nodes.image.hdr")

#: What one stop of exposure multiplies the light by.
STOP = 2.0

#: Widest exposure adjustment offered, in stops either way.
STOPS = 16.0

#: Highest ceiling offered, and the value that means no ceiling at all.
CEILING = 1024.0
UNCAPPED = 0.0

#: Widget option -> what happens to values below black.
NEGATIVES = ("hold at black", "keep")


class HDRVAEDecode(io.ComfyNode):
    """Turn a latent into an image without folding its highlights into white."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASHDRVAEDecode",
            display_name="HDR VAE Decode",
            search_aliases=[
                "WASHDRVAEDecode", "HDR VAE Decode", "vae decode", "decode", "hdr decode",
                "latent to image", "unclamped decode", "high dynamic range",
            ],
            category="WAS Suite/Image/HDR",
            description=(
                "Decode a latent to an image and keep the values the decoder put above white "
                "and below black, which the standard VAE Decode holds at 1.0 and 0.0. A "
                "bright sky, a specular hit or a light source comes out with its shape "
                "intact instead of a flat white shelf, ready for EXR Save, DNG Save, a tone "
                "map, or an exposure pull with Linear Light. Exposure, ceiling and negatives "
                "set what reaches the output, and the panel reports the peak and how much of "
                "the picture is above white."
            ),
            inputs=[
                io.Latent.Input("samples", tooltip="The latent to decode."),
                io.Vae.Input(
                    "vae",
                    tooltip=(
                        "The VAE that decodes it, the same one the standard VAE Decode "
                        "would take."
                    ),
                ),
                io.Float.Input(
                    "exposure",
                    default=0.0,
                    min=-STOPS,
                    max=STOPS,
                    step=0.1,
                    tooltip=(
                        "Stops applied to the decoded values. 0.0 leaves them alone, -1.0 "
                        "halves them, -2.0 brings a highlight that reached 4.0 back under "
                        "white. Applied before the ceiling."
                    ),
                ),
                io.Float.Input(
                    "ceiling",
                    default=UNCAPPED,
                    min=UNCAPPED,
                    max=CEILING,
                    step=0.5,
                    tooltip=(
                        "Highest value kept. 0.0 keeps everything the decoder produced, 4.0 "
                        "holds anything brighter at 4.0, and 1.0 gives the same picture the "
                        "standard VAE Decode does."
                    ),
                ),
                io.Combo.Input(
                    "negatives",
                    options=list(NEGATIVES),
                    tooltip=(
                        "'hold at black' = values below 0.0 come out at 0.0, which is what "
                        "every save and blend expects; 'keep' = they come through, for an "
                        "EXR that records the ringing around a hard edge."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip=(
                        "The decoded images. Values above 1.0 read as white in a preview and "
                        "survive into EXR Save, DNG Save and Linear Light."
                    ),
                ),
                io.Float.Output(
                    display_name="peak",
                    tooltip=(
                        "The highest value in the batch, 1.0 or under when the decode stayed "
                        "in range. Feed it to Linear Light through To Number to pull a "
                        "highlight back by a measured amount."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls, samples, vae, exposure=0.0, ceiling=UNCAPPED, negatives=NEGATIVES[0]
    ) -> io.NodeOutput:
        """Decode the latent, apply the levels, and answer the images and their peak.

        Raises:
            ValueError: ``negatives`` names neither treatment.
        """
        if negatives not in NEGATIVES:
            raise ValueError(
                f"HDR VAE Decode negatives must be one of {', '.join(NEGATIVES)}, "
                f"not {negatives!r}"
            )

        with vae_range.unclamped(vae) as lifted:
            images = vae.decode(samples["samples"])
        if images.ndim == 5:
            images = images.reshape(-1, *images.shape[-3:])
        decoded = float(images.amax())

        if float(exposure) != 0.0:
            images.mul_(STOP ** float(exposure))
        if float(ceiling) > UNCAPPED:
            images.clamp_(max=float(ceiling))
        if negatives == NEGATIVES[0]:
            images.clamp_(min=0.0)

        peak = float(images.amax())
        bright = sum(int((frame[..., :3] > 1.0).count_nonzero()) for frame in images)
        above = bright / max(1, images[..., :3].numel()) * 100.0
        logger.info(
            "decoded %d frame(s), peak %.4g, %.2f%% above white%s",
            int(images.shape[0]), peak, above, "" if lifted else ", nothing was held",
        )
        image_report.publish(
            images,
            facts={
                "peak": f"{peak:.4g}" + ("" if lifted else ", the VAE held nothing"),
                "above white": f"{above:.2f}% of samples",
                "levels": (
                    f"{exposure:+.2f} stop(s), "
                    + ("no ceiling" if float(ceiling) <= UNCAPPED else f"ceiling {ceiling:.4g}")
                ),
            },
            summary=(
                f"decoded to {decoded:.4g} at most, "
                + (f"answered {peak:.4g}" if peak != decoded else "answered as decoded")
            ),
        )
        return io.NodeOutput(images, peak)

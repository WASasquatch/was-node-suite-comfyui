"""Render scene-linear light for a display through the ACES 2.0 Output Transform."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.image import aces_output

logger = log.get_logger("nodes.image.color_management")

DEFAULT_SOURCE = "Linear Rec.709 (sRGB)"
DEFAULT_PRESET = "SDR 100 nit sRGB"


class ImageACESOutput(io.ComfyNode):
    """The ACES 2.0 Output Transform for one reference display."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageACESOutput",
            display_name="Image ACES Output",
            search_aliases=[
                "WASImageACESOutput", "Image ACES Output", "ACES 2.0", "output transform", "ODT",
                "display transform", "view transform", "tone map", "HDR10", "PQ", "HLG",
                "Rec.2100", "Display P3",
            ],
            category="WAS Suite/Image/Color Management",
            description=(
                "Render scene-linear light for a screen through the Academy's ACES 2.0 Output "
                "Transform: its tone scale, chroma compression and gamut mapping. Pick the "
                "display, SDR or HDR, and the node hands on that display's code values, ready to "
                "save as a picture or video in the same color space."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "Scene-linear light, not picture codes: 0.18 = mid grey, 1.0 = diffuse "
                        "white, highlights above. Wire EXR Load, HDR VAE Decode or Linear Light; "
                        "log footage goes through Image Color Space to Linear first."
                    ),
                ),
                io.Combo.Input(
                    "source",
                    options=list(aces_output.SOURCES),
                    default=DEFAULT_SOURCE,
                    tooltip=(
                        "Gamut of the scene-linear input. Linear Rec.709 (sRGB) = Linear Light, "
                        "HDR VAE Decode and most EXRs; ACEScg = AP1 renders and comps; "
                        "ACES2065-1 = AP0 archive files; Linear P3-D65 and Linear Rec.2020 for "
                        "those gamuts."
                    ),
                ),
                io.Combo.Input(
                    "preset",
                    options=list(aces_output.PRESETS),
                    default=DEFAULT_PRESET,
                    tooltip=(
                        "The display and its encoding, from the ACES 2.0 reference outputs. "
                        "sRGB = web, PNG and most monitors; Rec.709 BT.1886 = broadcast video; "
                        "Display P3 = Apple screens; PQ = HDR10; HLG = broadcast HDR; nit = the "
                        "peak highlights roll off to."
                    ),
                ),
                io.Combo.Input(
                    "white",
                    options=list(aces_output.WHITES),
                    default=aces_output.WHITES[0],
                    tooltip=(
                        "D65 = ACES white shows as the display's own white; D60 sim = ACES "
                        "white held at D60, a touch warmer, scaled to fit the display, for "
                        "matching a D60 cinema master."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip=(
                        "Display code values: gamma codes for SDR, PQ or HLG codes for HDR, up "
                        "to 2.70 for Display P3 extended. Save tagged as the preset's space in "
                        "Save Video (Advanced) or as an image; PQ and HLG look flat in an SDR "
                        "preview."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, images, source=DEFAULT_SOURCE, preset=DEFAULT_PRESET,
                white=aces_output.WHITES[0]) -> io.NodeOutput:
        """Render the batch for the chosen display.

        Raises:
            ValueError: A menu value this node does not offer, an empty batch, or two channels.
        """
        if images.shape[0] == 0:
            raise ValueError(
                "Image ACES Output was given an empty image batch, so there is nothing to "
                "render. Check the node feeding its images input."
            )
        if images.shape[-1] == 2:
            raise ValueError(
                "Image ACES Output needs RGB or RGBA images, or a single grey channel; these "
                "have two channels."
            )
        codes = aces_output.render(images, source, preset, white)
        logger.info("%s from %s, white %s, %d frame(s)", preset, source, white, int(images.shape[0]))
        return io.NodeOutput(codes)

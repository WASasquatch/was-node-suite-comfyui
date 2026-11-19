"""Convert a batch between camera log, scene-linear, ACES and display colour spaces."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import deps, log
from ....modules.image import color_spaces

logger = log.get_logger("nodes.image.color_management")

#: Brightest reference white offered for a PQ or HLG end, in cd/m2.
WHITE_MAX = 10000.0


class ImageColorSpace(io.ComfyNode):
    """Move pictures from one gamut and transfer curve to another."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        gamuts = list(color_spaces.GAMUTS)
        curves = list(color_spaces.TRANSFERS)
        return io.Schema(
            node_id="WASImageColorSpace",
            display_name="Image Color Space",
            search_aliases=[
                "WASImageColorSpace", "Image Color Space", "color space transform", "CST",
                "log to linear", "S-Log3", "LogC", "V-Log", "ACES", "ACEScct", "ACEScg",
                "Rec.2020", "PQ", "HLG", "HDR10", "gamut", "colour management",
            ],
            category="WAS Suite/Image/Color Management",
            description=(
                "Convert pictures between camera log, scene-linear, ACES and display color "
                "spaces, like a grading tool's color space transform. Decode camera footage to "
                "linear light, move between gamuts, or encode Rec.2100 PQ or HLG for HDR video. "
                "Nothing is clamped on the way."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "Pictures in the source space: codes on the source curve, or light when "
                        "it is Linear. Values above 1 and below 0 are read as they are."
                    ),
                ),
                io.Combo.Input(
                    "source_color_space",
                    options=gamuts,
                    default="Rec.709 / sRGB",
                    tooltip=(
                        "Gamut the pictures were recorded in. Rec.709 / sRGB = ordinary pictures "
                        "and sampler output; a camera's own gamut for its log footage, such as "
                        "Sony S-Gamut3.Cine or ARRI Wide Gamut 4. Apple Log is Rec.2020."
                    ),
                ),
                io.Combo.Input(
                    "source_gamma",
                    options=curves,
                    default="sRGB",
                    tooltip=(
                        "Curve the pictures carry. Linear = light, 1.0 white and 0.18 mid grey; "
                        "sRGB = PNG, JPEG and sampler output; a camera log = that camera's "
                        "codes; Rec.2100 PQ or HLG = HDR video codes."
                    ),
                ),
                io.Combo.Input(
                    "target_color_space",
                    options=gamuts,
                    default="Rec.2020",
                    tooltip=(
                        "Gamut to convert into. Rec.2020 = HDR10 and HLG delivery; Rec.709 / "
                        "sRGB = SDR screens and ordinary nodes; ACEScg (AP1) = compositing and "
                        "grading in light. Colors outside it come out below 0 or above 1."
                    ),
                ),
                io.Combo.Input(
                    "target_gamma",
                    options=curves,
                    default="Rec.2100 PQ (ST 2084)",
                    tooltip=(
                        "Curve to encode with. Linear = light for EXR Save and light-linear "
                        "blending; Rec.2100 PQ or HLG = HDR codes for Save Video (Advanced) in "
                        "the same mode; sRGB or Gamma 2.4 (BT.1886) = SDR display codes."
                    ),
                ),
                io.Combo.Input(
                    "chromatic_adaptation",
                    options=list(color_spaces.ADAPTATIONS),
                    default="Bradford",
                    optional=True,
                    advanced=True,
                    tooltip=(
                        "How white is carried between different white points, read only when "
                        "they differ: ACES spaces and Blackmagic Wide Gamut Gen 5. Bradford = "
                        "the usual choice; CAT02 and CAT16 = CIECAM02 and CAM16; None = XYZ "
                        "passed across unadapted."
                    ),
                ),
                io.Combo.Input(
                    "source_levels",
                    options=list(color_spaces.LEVELS),
                    default="full",
                    optional=True,
                    advanced=True,
                    tooltip=(
                        "full = codes are the curve's own values, as an EXR, a DPX or a render "
                        "holds them; legal = a camera clip decoded from video levels, 10-bit "
                        "code 64 to 940 read as 0 to 1. S-Log3 18% grey reads 0.406 legal, "
                        "0.411 full."
                    ),
                ),
                io.Combo.Input(
                    "target_levels",
                    options=list(color_spaces.LEVELS),
                    default="full",
                    optional=True,
                    advanced=True,
                    tooltip=(
                        "full = write the curve's own values; legal = write 10-bit code values "
                        "as video levels, code 64 landing on 0 and 940 on 1, the way a camera "
                        "clip decodes."
                    ),
                ),
                io.Float.Input(
                    "hdr_reference_white",
                    default=color_spaces.DEFAULT_WHITE_NITS,
                    min=1.0,
                    max=WHITE_MAX,
                    step=1.0,
                    optional=True,
                    advanced=True,
                    tooltip=(
                        "cd/m2 that linear 1.0 stands for at a PQ or HLG end. 203 = BT.2408 HDR "
                        "reference white, where 75% HLG lands; 100 = SDR reference white. HLG "
                        "is read through its 1000 cd/m2 reference display."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip=(
                        "The converted pictures: light when the target is Linear, otherwise "
                        "codes on the target curve. Values above 1 and below 0 are kept; log "
                        "and PQ codes look flat or dark in an ordinary preview."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls, images, source_color_space="Rec.709 / sRGB", source_gamma="sRGB",
        target_color_space="Rec.2020", target_gamma="Rec.2100 PQ (ST 2084)",
        chromatic_adaptation="Bradford", source_levels="full", target_levels="full",
        hdr_reference_white=color_spaces.DEFAULT_WHITE_NITS,
    ) -> io.NodeOutput:
        """Convert the batch and answer it.

        Raises:
            ValueError: A menu value is not one this node offers, the batch is empty, or the
                images are not RGB.
        """
        choices = (
            ("source_color_space", source_color_space, color_spaces.GAMUTS),
            ("source_gamma", source_gamma, color_spaces.TRANSFERS),
            ("target_color_space", target_color_space, color_spaces.GAMUTS),
            ("target_gamma", target_gamma, color_spaces.TRANSFERS),
            ("chromatic_adaptation", chromatic_adaptation, color_spaces.ADAPTATIONS),
            ("source_levels", source_levels, color_spaces.LEVELS),
            ("target_levels", target_levels, color_spaces.LEVELS),
        )
        for name, value, table in choices:
            if value not in table:
                raise ValueError(
                    f"Image Color Space {name} must be one of {', '.join(table)}, not {value!r}."
                )
        if images.shape[0] == 0:
            raise ValueError(
                "Image Color Space was given an empty image batch, so there is nothing to "
                "convert. Check the node feeding its images input."
            )
        white = float(hdr_reference_white)
        if not 1.0 <= white <= WHITE_MAX:
            raise ValueError(
                f"Image Color Space hdr_reference_white must be 1 to {WHITE_MAX:g} cd/m2, "
                f"not {white:g}."
            )

        answer = color_spaces.convert(
            images, source_color_space, source_gamma, target_color_space, target_gamma,
            chromatic_adaptation, source_levels, target_levels, white,
            device=deps.float64_device(),
        )
        logger.info(
            "%s %s to %s %s, %d frame(s)", source_color_space, source_gamma,
            target_color_space, target_gamma, int(images.shape[0]),
        )
        return io.NodeOutput(answer)

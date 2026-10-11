"""Bring linear light above white down to a range a display can show."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ....modules import log
from ....modules.image import raw, tonemap

logger = log.get_logger("nodes.image.hdr")

#: What the node answers, in menu order.
OUTPUTS = ("picture codes", "linear light")

#: Widest exposure adjustment offered, in stops either way.
STOPS = 16.0


class ImageToneMap(io.ComfyNode):
    """Map linear light into 0 to 1 through a tone curve."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageToneMap",
            display_name="Image Tone Map",
            search_aliases=[
                "WASImageToneMap",
                "Image Tone Map",
                "tonemap",
                "reinhard",
                "aces",
                "filmic",
                "hdr to sdr",
                "highlight rolloff",
            ],
            category="WAS Suite/Image/HDR",
            description=(
                "Roll highlights above white down into a range a screen can show, instead of "
                "clipping them flat. Use it before a preview or an 8-bit save of linear light, "
                "such as HDR Reconstruct, HDR VAE Decode or EXR Load hand on."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The frames to map, as linear light. Wire in HDR Reconstruct, HDR VAE "
                        "Decode, EXR Load, or Linear Light on `sRGB to linear`."
                    ),
                ),
                io.Combo.Input(
                    "operator",
                    options=list(tonemap.OPERATORS),
                    tooltip=(
                        "The curve the highlights roll down. `aces` and `hable` are filmic "
                        "and darken the midtones a little; `reinhard` is gentle and never "
                        "quite reaches white; `drago` holds detail furthest into a very "
                        "bright highlight; `clip` cuts at 1.0."
                    ),
                ),
                io.Float.Input(
                    "exposure",
                    default=0.0,
                    min=-STOPS,
                    max=STOPS,
                    step=0.1,
                    tooltip=(
                        "Stops applied before the curve. 0.0 = as it arrived, -2.0 = a "
                        "quarter as bright, which pulls a blown highlight back into the "
                        "shoulder, +1.0 = twice as bright."
                    ),
                ),
                io.Float.Input(
                    "white_point",
                    default=4.0,
                    min=1.0,
                    max=1000.0,
                    step=0.1,
                    tooltip=(
                        "The level that comes out as white. 4.0 = four times diffuse white "
                        "reaches 1.0; 40.0 keeps a rebuilt sun inside the curve. Read by "
                        "`reinhard extended` and `drago` only."
                    ),
                ),
                io.Combo.Input(
                    "applied_to",
                    options=list(tonemap.APPLIED_TO),
                    tooltip=(
                        "`each channel` maps red, green and blue on their own, which washes a "
                        "saturated highlight towards white the way film does; `brightness` "
                        "maps brightness and scales the three together, keeping the hue."
                    ),
                ),
                io.Combo.Input(
                    "output",
                    options=list(OUTPUTS),
                    tooltip=(
                        "`picture codes` applies the sRGB curve, which is what a preview, a "
                        "PNG and a JPEG expect; `linear light` leaves the result linear, for "
                        "more grading or an EXR."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="The mapped frames, in 0 to 1.",
                ),
                io.Float.Output(
                    display_name="peak",
                    tooltip=(
                        "The largest value that went into the curve, after exposure. 40.07 = "
                        "the frame held a highlight 40 times white; 1.0 = nothing was above "
                        "white and the curve only darkened the picture."
                    ),
                ),
                io.Float.Output(
                    display_name="recovered",
                    tooltip=(
                        "The share of pixels that were above 1.0 and came back inside it. "
                        "0.0 = nothing was clipping; 0.03 = 3% of the frame was rolled down "
                        "rather than cut to white."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls, images, operator=tonemap.OPERATORS[0], exposure=0.0, white_point=4.0,
        applied_to=tonemap.APPLIED_TO[0], output=OUTPUTS[0],
    ) -> io.NodeOutput:
        light = images.to(dtype=torch.float32).clamp(min=0.0)
        if exposure:
            light = light * (tonemap.STOP ** float(exposure))

        peak = float(light.max()) if light.numel() else 0.0
        above = float((light > 1.0).to(dtype=torch.float32).mean()) if light.numel() else 0.0

        toned = tonemap.mapped(light, operator, float(white_point), applied_to)
        if output == OUTPUTS[0]:
            toned = raw.encode(toned).clamp(0.0, 1.0)

        logger.info(
            "Image Tone Map ran %s on a peak of %.3f, %.2f%% of the frame above white",
            operator, peak, above * 100.0,
        )
        return io.NodeOutput(toned.to(dtype=images.dtype), peak, above)

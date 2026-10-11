"""Move a batch between the sRGB curve a file is stored with and the light it stands for."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ....modules import log
from ....modules.image import raw

logger = log.get_logger("nodes.image.hdr")

#: Widget option -> whether the transfer is undone or applied.
DIRECTIONS = ("sRGB to linear", "linear to sRGB")

#: What one stop of exposure multiplies the light by.
STOP = 2.0

#: Widest exposure adjustment offered, in stops either way.
STOPS = 16.0


class LinearLight(io.ComfyNode):
    """Undo or apply the sRGB transfer function, with an exposure adjustment in stops."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLinearLight",
            display_name="Linear Light",
            search_aliases=[
                "WASLinearLight", "Linear Light", "gamma", "srgb to linear",
                "linear to srgb", "transfer function", "degamma", "exposure",
            ],
            category="WAS Suite/Image/HDR",
            description=(
                "Convert between the sRGB curve a picture is stored with and the linear light "
                "it stands for, adjusting exposure in stops on the way. Use it to feed nodes "
                "that work in light, such as EXR Save, and to bring light back for viewing."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The images to convert. Everything above 1.0 survives the trip to "
                        "linear and is brought back into range by exposure on the way out."
                    ),
                ),
                io.Combo.Input(
                    "direction",
                    options=list(DIRECTIONS),
                    tooltip=(
                        "'sRGB to linear' = a PNG, JPEG or sampler picture on its way into EXR "
                        "Save, DNG Save or a light-linear blend; 'linear to sRGB' = light on "
                        "its way to a preview, a save or any node expecting ordinary picture "
                        "codes."
                    ),
                ),
                io.Float.Input(
                    "exposure",
                    default=0.0,
                    min=-STOPS,
                    max=STOPS,
                    step=0.1,
                    tooltip=(
                        "Stops of exposure applied to the light. 0.0 leaves it alone, -1.0 "
                        "halves it, +1.0 doubles it. -2.0 brings a highlight that reached "
                        "4.0 back under white."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip=(
                        "The converted images. Going to linear, the values are light and a "
                        "preview reads them too dark; coming back, they are ordinary picture "
                        "codes from 0 to 1."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, images, direction="sRGB to linear", exposure=0.0) -> io.NodeOutput:
        """Convert the batch and answer it.

        Raises:
            ValueError: ``direction`` names neither transfer.
        """
        if direction not in DIRECTIONS:
            raise ValueError(
                f"Linear Light direction must be one of {', '.join(DIRECTIONS)}, "
                f"not {direction!r}"
            )

        colour = images[..., :3]
        alpha = images[..., 3:]
        linear = raw.linearise(colour) if direction == DIRECTIONS[0] else colour
        gained = linear * (STOP ** float(exposure))
        answered = gained if direction == DIRECTIONS[0] else raw.encode(gained)
        logger.info(
            "%s at %+.2f stop(s), %d frame(s), peak %.4g",
            direction, float(exposure), int(images.shape[0]), float(answered.amax()),
        )
        return io.NodeOutput(torch.cat([answered, alpha], dim=-1) if alpha.shape[-1] else answered)

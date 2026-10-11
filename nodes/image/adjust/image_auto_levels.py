"""Stretch a frame so its darkest pixels reach black and its lightest reach white."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.image import dynamic, grade

logger = log.get_logger("nodes.image.adjust")


class ImageAutoLevels(io.ComfyNode):
    """Find a batch's black and white points and stretch it between them."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageAutoLevels",
            display_name="Image Auto Levels",
            search_aliases=[
                "WASImageAutoLevels",
                "Image Auto Levels",
                "auto contrast",
                "auto colour",
                "auto color",
                "normalise",
                "histogram stretch",
            ],
            category="WAS Suite/Image/Adjustment",
            description=(
                "Stretch a frame's tones so its darkest pixels reach black and its lightest "
                "reach white, giving a flat render, hazy plate or washed-out scan its contrast "
                "back with nothing set by hand. `per channel` also pulls out a colour cast. "
                "Hands on the stretched frames and the black and white points it found."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The frames to stretch. The whole batch is measured together, so a "
                        "sequence keeps a steady exposure rather than flickering frame to "
                        "frame."
                    ),
                ),
                io.Combo.Input(
                    "method",
                    options=list(grade.METHODS),
                    tooltip=(
                        "`per channel` stretches red, green and blue separately, which adds "
                        "contrast and neutralises a cast at once; `on brightness` stretches "
                        "all three by the same amount, which keeps the colour as it was."
                    ),
                ),
                io.Float.Input(
                    "clip_low",
                    default=0.001,
                    min=0.0,
                    max=0.2,
                    step=0.0005,
                    tooltip=(
                        "Share of the darkest pixels allowed to go fully black. 0.0 = the "
                        "single darkest pixel sets the point, so one stuck pixel ruins it; "
                        "0.001 = a thousandth, which ignores those; 0.02 = a deeper crush."
                    ),
                ),
                io.Float.Input(
                    "clip_high",
                    default=0.001,
                    min=0.0,
                    max=0.2,
                    step=0.0005,
                    tooltip=(
                        "Share of the lightest pixels allowed to go fully white. 0.001 = a "
                        "thousandth, which ignores a specular hit; 0.02 = brighter, with more "
                        "of the highlight blown."
                    ),
                ),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How far towards the stretched result the frame moves. 1.0 = the full "
                        "stretch, 0.5 = halfway, 0.0 = the frame untouched."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="The stretched frames.",
                ),
                io.Float.Output(
                    display_name="black_point",
                    tooltip=(
                        "The brightness the stretch read as black, on a 0 to 1 scale. 0.0 = "
                        "the frame already reached black and nothing was gained below."
                    ),
                ),
                io.Float.Output(
                    display_name="white_point",
                    tooltip=(
                        "The brightness the stretch read as white, on a 0 to 1 scale. 1.0 = "
                        "the frame already reached white; 0.6 = it was two fifths short."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls, images, method=grade.METHODS[0], clip_low=0.001, clip_high=0.001, strength=1.0
    ) -> io.NodeOutput:
        folded = dynamic.fold(images)
        stretched, (black, white) = grade.auto_levels(
            folded.images, method, float(clip_low), float(clip_high), float(strength)
        )
        logger.info(
            "Image Auto Levels stretched %d frame(s) from %.4f to %.4f",
            int(images.shape[0]), black, white,
        )
        return io.NodeOutput(dynamic.unfold(stretched, folded), black, white)

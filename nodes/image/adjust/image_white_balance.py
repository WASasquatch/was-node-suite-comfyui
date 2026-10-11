"""Taking a colour cast out of an image, or steadily out of a sequence."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.image import dynamic, white_balance

#: Frames either side the estimate may be averaged over. The estimate is three numbers per
#: frame, so the window costs nothing worth counting; this bounds what a sequence can usefully
#: be averaged across.
MAX_RADIUS = 250


class ImageWhiteBalance(io.ComfyNode):
    """Estimate the colour of the light in each frame and divide it back out."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageWhiteBalance",
            display_name="Image White Balance",
            search_aliases=[
                "WASImageWhiteBalance", "Image White Balance",
                "white balance",
                "colour cast",
                "color cast",
                "auto white balance",
                "neutralise colour",
                "grey world",
                "colour temperature",
            ],
            category="WAS Suite/Image/Adjustment",
            description=(
                "Remove a colour cast by working out what colour the light was and dividing it "
                "back out, leaving the brightness alone. Four ways of guessing the light are "
                "offered, since each is fooled by a different scene. For footage, raise "
                "temporal_radius so the balance stays put instead of shifting shot to shot."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The image or sequence to balance; IMAGE. A batch is treated as frames "
                        "in order when temporal_radius is above 0."
                    ),
                ),
                io.Combo.Input(
                    "estimator",
                    list(white_balance.ESTIMATORS),
                    tooltip=(
                        "How the light's colour is guessed. 'grey world' = the scene averages "
                        "grey; 'white patch' = the brightest point is white; 'shades of grey' = "
                        "between those two; 'grey edge' = the edges average grey, which one "
                        "large block of colour barely moves."
                    ),
                ),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How much of the cast to remove; FLOAT, 0 to 1. Below 1 keeps some of "
                        "the original warmth, for a look that is meant to be there. 0 leaves "
                        "the image alone."
                    ),
                ),
                io.Int.Input(
                    "temporal_radius",
                    default=0,
                    min=0,
                    max=MAX_RADIUS,
                    tooltip=(
                        "Frames either side the guess is averaged over; INT. 0 balances every "
                        "frame on its own, which wanders when the scene's contents change. "
                        "Raise it for footage so the balance holds still."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="The balanced image; IMAGE, the same size and length as it went in.",
                ),
            ],
        )

    @classmethod
    def execute(cls, images, estimator="grey world", strength=1.0,
                temporal_radius=0) -> io.NodeOutput:
        """Balance the image or sequence.

        Raises:
            ValueError: The input is not a batch of images, or the estimator does not exist.
        """
        if getattr(images, "ndim", 0) != 4 or int(images.shape[0]) < 1:
            raise ValueError(
                "Image White Balance needs an image to work on. Connect an image or a batch "
                "of frames."
            )
        folded = dynamic.fold(images)
        return io.NodeOutput(dynamic.unfold(
            white_balance.balance(
                folded.images,
                estimator=str(estimator),
                strength=float(strength),
                radius=int(temporal_radius),
            ),
            folded,
        ))

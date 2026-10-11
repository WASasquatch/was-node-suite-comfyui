"""Average a frame along a path: a straight smear, a zoom out of the centre, or a spin."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.image import dynamic, optics

logger = log.get_logger("nodes.image.filter")


class ImageDirectionalBlur(io.ComfyNode):
    """Blur a batch along a direction, a radius or an arc."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageDirectionalBlur",
            display_name="Image Directional Blur",
            search_aliases=[
                "WASImageDirectionalBlur",
                "Image Directional Blur",
                "motion blur",
                "zoom blur",
                "radial blur",
                "spin blur",
                "speed lines",
            ],
            category="WAS Suite/Image/Filter",
            description=(
                "Smear the frame along a path instead of spreading it evenly. `linear` is "
                "the streak a moving camera or subject leaves; `zoom` rushes out of a point, "
                "for speed towards the viewer; `spin` sweeps round one, for rotation."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The frames to smear. Each one gets the same path and comes back at "
                        "the size it went in at."
                    ),
                ),
                io.Combo.Input(
                    "blur",
                    options=list(optics.BLURS),
                    tooltip=(
                        "Which path the samples are taken along. `linear` runs in one "
                        "direction; `zoom` runs out from the centre; `spin` runs round it."
                    ),
                ),
                io.Float.Input(
                    "length",
                    default=0.05,
                    min=0.0,
                    max=1.0,
                    step=0.005,
                    tooltip=(
                        "How far the smear travels. On `linear` and `zoom` it is a share of "
                        "the frame: 0.05 = 5%, a hand-held wobble; 0.3 = a long streak. On "
                        "`spin` it is turns: 0.02 = about 7 degrees."
                    ),
                ),
                io.Float.Input(
                    "angle",
                    default=0.0,
                    min=-360.0,
                    max=360.0,
                    step=1.0,
                    tooltip=(
                        "Direction of a `linear` smear. 0 = to the right, 90 = downwards, 45 "
                        "= down and to the right. Ignored by `zoom` and `spin`."
                    ),
                ),
                io.Int.Input(
                    "taps",
                    default=16,
                    min=2,
                    max=128,
                    step=1,
                    tooltip=(
                        "How many samples are averaged along the path. 8 = fast and visibly "
                        "stepped on a long smear; 16 = smooth for most lengths; 64 = smooth "
                        "for the longest, and four times the work."
                    ),
                ),
                io.Float.Input(
                    "centre_x",
                    default=0.5,
                    min=-1.0,
                    max=2.0,
                    step=0.01,
                    tooltip=(
                        "Where `zoom` and `spin` turn about, across the frame. 0.5 = the "
                        "middle, 0.0 = the left edge. Ignored by `linear`."
                    ),
                ),
                io.Float.Input(
                    "centre_y",
                    default=0.5,
                    min=-1.0,
                    max=2.0,
                    step=0.01,
                    tooltip=(
                        "Where `zoom` and `spin` turn about, down the frame. 0.5 = the "
                        "middle, 0.0 = the top edge. Ignored by `linear`."
                    ),
                ),
                io.Combo.Input(
                    "edge",
                    options=list(optics.EDGES),
                    tooltip=(
                        "What the samples read past the edge of the frame. `hold the edge` "
                        "smears the outermost pixel out; `mirror` folds the frame back; "
                        "`empty` darkens the border as the smear runs off it."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="The smeared frames.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls, images, blur=optics.BLURS[0], length=0.05, angle=0.0, taps=16,
        centre_x=0.5, centre_y=0.5, edge=optics.EDGES[0],
    ) -> io.NodeOutput:
        folded = dynamic.fold(images)
        smeared = optics.smeared(
            folded.images, blur, float(length), float(angle), int(taps),
            float(centre_x), float(centre_y), edge,
        )
        logger.info(
            "Image Directional Blur ran %s over %d frame(s) with %d tap(s)",
            blur, int(images.shape[0]), int(taps),
        )
        return io.NodeOutput(dynamic.unfold(smeared, folded))

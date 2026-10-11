"""Bend straight lines the way a lens does, or take an existing bend back out."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.image import dynamic, optics

logger = log.get_logger("nodes.image.transform")


class ImageLensDistortion(io.ComfyNode):
    """Apply or correct radial lens distortion across a batch."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageLensDistortion",
            display_name="Image Lens Distortion",
            search_aliases=[
                "WASImageLensDistortion",
                "Image Lens Distortion",
                "barrel",
                "pincushion",
                "fisheye",
                "lens correction",
                "defish",
            ],
            category="WAS Suite/Image/Transform",
            description=(
                "Bow the frame outwards or pinch it inwards and split the colour channels "
                "apart across the radius, as a real lens does. Adds a wide-lens barrel to a "
                "render, or takes one out of footage that has it."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The frames to bend. Each one gets the same distortion and comes back "
                        "at the size it went in at."
                    ),
                ),
                io.Float.Input(
                    "k1",
                    default=-0.15,
                    min=-1.0,
                    max=1.0,
                    step=0.005,
                    tooltip=(
                        "The main radial term. 0.0 = straight, -0.15 = a gentle barrel, -0.5 "
                        "= a strong fisheye bow, 0.15 = a pincushion that takes a barrel back "
                        "out."
                    ),
                ),
                io.Float.Input(
                    "k2",
                    default=0.0,
                    min=-1.0,
                    max=1.0,
                    step=0.005,
                    tooltip=(
                        "A second radial term acting furthest from the centre. 0.0 = none, "
                        "0.05 = pulls the very corners back after a strong k1, which is how a "
                        "real lens profile is written."
                    ),
                ),
                io.Float.Input(
                    "scale",
                    default=1.0,
                    min=0.25,
                    max=4.0,
                    step=0.01,
                    tooltip=(
                        "Zoom applied with the bend. 1.0 = none, so a barrel leaves empty "
                        "corners; 1.2 = zoomed in far enough to fill them; 0.8 = pulled back "
                        "to keep everything the frame held."
                    ),
                ),
                io.Float.Input(
                    "dispersion",
                    default=0.0,
                    min=-0.1,
                    max=0.1,
                    step=0.001,
                    tooltip=(
                        "How far red and blue are scaled apart, growing towards the corners. "
                        "0.0 = none, 0.004 = a faint colour fringe at the edges, 0.02 = an "
                        "obvious one."
                    ),
                ),
                io.Combo.Input(
                    "edge",
                    options=list(optics.EDGES),
                    tooltip=(
                        "What fills the space the bend opens up. `hold the edge` smears the "
                        "outermost pixel out; `mirror` folds the frame back on itself; "
                        "`empty` leaves it black."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="The frames with the distortion applied.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls, images, k1=-0.15, k2=0.0, scale=1.0, dispersion=0.0, edge=optics.EDGES[0]
    ) -> io.NodeOutput:
        folded = dynamic.fold(images)
        bent = optics.distorted(
            folded.images, float(k1), float(k2), float(scale), float(dispersion), edge
        )
        logger.info(
            "Image Lens Distortion bent %d frame(s) with k1 %.3f, k2 %.3f",
            int(images.shape[0]), k1, k2,
        )
        return io.NodeOutput(dynamic.unfold(bent, folded))

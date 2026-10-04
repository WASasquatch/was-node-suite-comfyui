"""Pin the four corners of a frame to four new places."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ....modules import log
from ....modules.compat import limits
from ....modules.image import dynamic, optics

logger = log.get_logger("nodes.image.transform")


class ImagePerspective(io.ComfyNode):
    """Warp a batch of images through a four-corner mapping."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImagePerspective",
            display_name="Image Perspective",
            search_aliases=[
                "WASImagePerspective",
                "Image Perspective",
                "corner pin",
                "keystone",
                "four corner",
                "quad warp",
                "homography",
            ],
            category="WAS Suite/Image/Transform",
            description=(
                "Drag each corner of the frame somewhere else and let the picture follow. "
                "That maps a flat render onto a surface seen at an angle: a poster onto a "
                "wall, a screen into a photograph, a label onto a box. It also takes a "
                "keystone back out of a plate shot off-axis. Image Displacement Warp bends "
                "the picture locally; this is the one straight-line mapping a camera makes."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The frames to warp. Each one gets the same mapping and comes back at "
                        "the output size."
                    ),
                ),
                io.Int.Input(
                    "top_left_x",
                    default=0,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Where the frame's top left corner lands, across the output, in "
                        "pixels. 0 = the output's own top left, 120 = 120px in from it."
                    ),
                ),
                io.Int.Input(
                    "top_left_y",
                    default=0,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Where the frame's top left corner lands, down the output, in pixels. "
                        "0 = flush with the top, 80 = 80px down."
                    ),
                ),
                io.Int.Input(
                    "top_right_x",
                    default=0,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Where the frame's top right corner lands, measured in from the right "
                        "edge. 0 = flush with it, 120 = 120px in."
                    ),
                ),
                io.Int.Input(
                    "top_right_y",
                    default=0,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Where the frame's top right corner lands, down the output, in "
                        "pixels. 0 = flush with the top, 80 = 80px down."
                    ),
                ),
                io.Int.Input(
                    "bottom_right_x",
                    default=0,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Where the frame's bottom right corner lands, measured in from the "
                        "right edge. 0 = flush with it, 120 = 120px in."
                    ),
                ),
                io.Int.Input(
                    "bottom_right_y",
                    default=0,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Where the frame's bottom right corner lands, measured up from the "
                        "bottom edge. 0 = flush with it, 80 = 80px up."
                    ),
                ),
                io.Int.Input(
                    "bottom_left_x",
                    default=0,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Where the frame's bottom left corner lands, across the output, in "
                        "pixels. 0 = flush with the left, 120 = 120px in."
                    ),
                ),
                io.Int.Input(
                    "bottom_left_y",
                    default=0,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Where the frame's bottom left corner lands, measured up from the "
                        "bottom edge. 0 = flush with it, 80 = 80px up."
                    ),
                ),
                io.Int.Input(
                    "width",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Width of the answer in pixels. 0 = the same width the frames came "
                        "in at; 1920 makes room for corners pushed outside them."
                    ),
                ),
                io.Int.Input(
                    "height",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Height of the answer in pixels. 0 = the same height the frames came "
                        "in at; 1080 makes room for corners pushed outside them."
                    ),
                ),
                io.Combo.Input(
                    "edge",
                    options=list(optics.EDGES),
                    tooltip=(
                        "What fills the space outside the warped picture. `empty` leaves it "
                        "black, which is what a composite wants; `hold the edge` smears the "
                        "outermost pixel out; `mirror` folds the frame back."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="The warped frames, at the output size.",
                ),
                io.Mask.Output(
                    display_name="mask",
                    tooltip=(
                        "White where the warped picture landed, black around it, for "
                        "compositing it onto a plate."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls, images, top_left_x=0, top_left_y=0, top_right_x=0, top_right_y=0,
        bottom_right_x=0, bottom_right_y=0, bottom_left_x=0, bottom_left_y=0,
        width=0, height=0, edge=optics.EDGES[2],
    ) -> io.NodeOutput:
        """Warp every frame and build the coverage mask.

        Raises:
            ValueError: The four corners lie on one line.
            MemoryError: Neither free memory nor a scratch drive can hold the warped frames.
        """
        from ....modules.image import scratch

        source_h, source_w = int(images.shape[1]), int(images.shape[2])
        out_w = int(width) or source_w
        out_h = int(height) or source_h
        corners = (
            (int(top_left_x), int(top_left_y)),
            (out_w - 1 - int(top_right_x), int(top_right_y)),
            (out_w - 1 - int(bottom_right_x), out_h - 1 - int(bottom_right_y)),
            (int(bottom_left_x), out_h - 1 - int(bottom_left_y)),
        )

        warped = None
        if images.device.type == "cpu":
            warped = scratch.allocate(
                (int(images.shape[0]), out_h, out_w, int(images.shape[3])),
                dtype=images.dtype,
                node="Image Perspective",
                advice="A smaller width and height or fewer frames also fit it.",
            )
        peak = dynamic.peak(images) if dynamic.carries(images) else None
        warped = optics.perspective(images, corners, out_w, out_h, edge, out=warped, peak=peak)

        solid = torch.ones(
            (1, source_h, source_w, 1), dtype=torch.float32, device=images.device
        )
        cover = optics.perspective(solid, corners, out_w, out_h, optics.EDGES[2])

        logger.info(
            "Image Perspective warped %d frame(s) from %dx%d to %dx%d",
            int(images.shape[0]), source_w, source_h, out_w, out_h,
        )
        return io.NodeOutput(warped, cover[..., 0].clamp(0.0, 1.0))

"""Refine a rough mask into a soft matte, and unmix the colour behind it."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ....modules import log
from ....modules.image import dynamic, matting
from ....modules.interface import mask_report

logger = log.get_logger("nodes.image.process")

#: Levels of a trimap, as a widget reads them on a 0 to 255 scale.
LEVELS = 255.0


class ImageMatte(io.ComfyNode):
    """Solve a soft alpha matte from a frame and a rough mask."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageMatte",
            display_name="Image Matte",
            search_aliases=[
                "WASImageMatte",
                "Image Matte",
                "alpha matting",
                "closed form matting",
                "trimap",
                "refine mask",
                "hair mask",
                "unmix",
            ],
            category="WAS Suite/Image/Process",
            description=(
                "Turn a rough mask into a matte that holds hair, fur, smoke and motion blur. "
                "The band between the certain foreground and the certain background is "
                "solved against the picture's own colours, so a hard cut-out from a "
                "segmenter or a threshold comes back with a soft, correct edge. The "
                "foreground output is the subject's colour with the background unmixed out "
                "of it, which is what stops a green or a white fringe following a cut-out "
                "onto a new plate."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The frames the matte is solved against. Each one is run on its own "
                        "and comes back at the size it went in at."
                    ),
                ),
                io.Mask.Input(
                    "masks",
                    tooltip=(
                        "The rough mask, white where the subject is. A hard cut-out from "
                        "Image Remove Background, CLIPSeg Masking or SAM is what this is for. "
                        "One mask is used for every frame; a batch is paired frame by frame."
                    ),
                ),
                io.Int.Input(
                    "certain_foreground",
                    default=240,
                    min=1,
                    max=255,
                    step=1,
                    tooltip=(
                        "Mask level above which a pixel is certainly the subject, on a 0 to "
                        "255 scale. 240 trusts only what the mask calls solid; 128 trusts "
                        "more of it and leaves a narrower band to solve."
                    ),
                ),
                io.Int.Input(
                    "certain_background",
                    default=10,
                    min=0,
                    max=254,
                    step=1,
                    tooltip=(
                        "Mask level below which a pixel is certainly not the subject, on a 0 "
                        "to 255 scale. 10 trusts only what the mask calls empty; 64 trusts "
                        "more of it."
                    ),
                ),
                io.Int.Input(
                    "band",
                    default=10,
                    min=0,
                    max=128,
                    step=1,
                    tooltip=(
                        "Pixels pulled back off both certain regions, which is what the "
                        "matte is solved across. 0 = solve nothing and answer the mask; 10 = "
                        "a 10px band, enough for a soft edge; 40 = enough for flyaway hair."
                    ),
                ),
                io.Boolean.Input(
                    "unmix_foreground",
                    default=True,
                    tooltip=(
                        "`on` also estimates the subject's own colour with the background "
                        "taken out of it, which removes a fringe. `off` answers the frame "
                        "unchanged on that output and is several times faster."
                    ),
                ),
            ],
            outputs=[
                io.Mask.Output(
                    display_name="mattes",
                    tooltip=(
                        "The solved matte, white where the subject is, soft across the band."
                    ),
                ),
                io.Image.Output(
                    display_name="foreground",
                    tooltip=(
                        "The subject's colour with the background unmixed out of it. Composite "
                        "this rather than the original frame, or the old background follows "
                        "the edge onto the new one. A frame carrying light above white comes "
                        "back on the scale it arrived on."
                    ),
                ),
                io.Image.Output(
                    display_name="cutout",
                    tooltip=(
                        "The foreground with the matte as its fourth channel, ready for Add "
                        "Layer, Join Image with Alpha or a PNG save."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls, images, masks, certain_foreground=240, certain_background=10, band=10,
        unmix_foreground=True,
    ) -> io.NodeOutput:
        from ....modules.image import scratch

        if certain_background >= certain_foreground:
            raise ValueError(
                f"Image Matte needs certain_background ({certain_background}) below "
                f"certain_foreground ({certain_foreground}), or there is no band left to "
                f"solve. Lower certain_background, or raise certain_foreground."
            )

        # A frame above white is solved inside 0 to 1 and its colour put back on its scale.
        folded = dynamic.fold(images if images.ndim == 4 else images.unsqueeze(0))
        frames = folded.images
        planes = cls.planes(masks)
        if int(planes.shape[0]) == 0:
            raise ValueError(
                "Image Matte was handed an empty mask batch. Wire in a mask holding at least "
                "one plane."
            )

        count, height, width = (int(side) for side in frames.shape[:3])
        if count == 0:
            raise ValueError(
                "Image Matte was handed an empty image batch. Wire in at least one image."
            )
        advice = "Passing fewer or smaller frames also fits it."
        solved = scratch.allocate((count, height, width), node="Image Matte", advice=advice)
        fronts = scratch.allocate((count, height, width, 3), node="Image Matte", advice=advice)
        cutouts = scratch.allocate((count, height, width, 4), node="Image Matte", advice=advice)
        for index in range(count):
            frame = frames[index, :, :, :3].to(dtype=torch.float32).clamp(0.0, 1.0)
            rough = cls.fitted(planes, index, int(frame.shape[0]), int(frame.shape[1]))
            marked = matting.trimap(
                rough,
                float(certain_foreground) / LEVELS,
                float(certain_background) / LEVELS,
                int(band),
            )
            matte = matting.alpha(frame, marked)
            front = matting.foreground(frame, matte) if unmix_foreground else frame
            colour = dynamic.unfold(front, folded)
            solved[index] = matte
            fronts[index] = colour
            cutouts[index, :, :, :3] = colour
            cutouts[index, :, :, 3] = matte

        mask_report.publish(planes, solved, source="masks")
        logger.info("Image Matte solved %d frame(s) over a %dpx band", count, int(band))
        return io.NodeOutput(solved, fronts, cutouts)

    @staticmethod
    def planes(masks):
        """A mask of any layout as ``(frames, height, width)``.

        Args:
            masks: A ``MASK`` of two, three or more axes.

        Returns:
            A three-axis float tensor. Axes past the last three are folded into the frames.
        """
        found = masks.to(dtype=torch.float32)
        if found.ndim == 2:
            return found.unsqueeze(0)
        if found.ndim == 3:
            return found
        return found.reshape(-1, int(found.shape[-2]), int(found.shape[-1]))

    @staticmethod
    def fitted(planes, index: int, height: int, width: int):
        """One mask plane at the frame's own size.

        Args:
            planes: The mask batch, ``(frames, height, width)``.
            index: Which frame of the image batch, counting 0.
            height: The frame's height in pixels.
            width: The frame's width in pixels.

        Returns:
            A ``(height, width)`` float tensor in 0 to 1.
        """
        slot = index if index < int(planes.shape[0]) else int(planes.shape[0]) - 1
        plane = planes[slot].to(dtype=torch.float32)
        if tuple(plane.shape) == (height, width):
            return plane.clamp(0.0, 1.0)
        stretched = torch.nn.functional.interpolate(
            plane.unsqueeze(0).unsqueeze(0), size=(height, width),
            mode="bilinear", align_corners=False,
        )
        return stretched[0, 0].clamp(0.0, 1.0)

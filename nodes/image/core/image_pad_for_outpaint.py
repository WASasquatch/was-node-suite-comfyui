"""Put a picture on a larger grey canvas and answer the mask that fills the new room."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ....modules.compat import limits
from ....modules.interface import size_report
from ....modules.log import get_logger

logger = get_logger("nodes.image.core")

#: Level the new canvas is filled with, which is mid grey in picture codes.
CANVAS_LEVEL = 0.5


def _feathered(height, width, left, top, right, bottom, feathering):
    """The soft edge written into the mask over the frame that was kept.

    Args:
        height: Frame height in pixels.
        width: Frame width in pixels.
        left: Pixels of canvas added on the left.
        top: Pixels of canvas added above.
        right: Pixels of canvas added on the right.
        bottom: Pixels of canvas added below.
        feathering: Pixels the fade runs over at each seam that has canvas beside it.

    Returns:
        A ``(height, width)`` float32 plane on the CPU, 0 through the middle and rising to
        just under 1 at a seam. All zeros where the fade would meet itself across the frame.
    """
    if feathering <= 0 or feathering * 2 >= height or feathering * 2 >= width:
        return torch.zeros((height, width), dtype=torch.float32)
    rows = torch.arange(height, dtype=torch.float64).unsqueeze(1)
    columns = torch.arange(width, dtype=torch.float64).unsqueeze(0)
    down = torch.full_like(rows, float(height))
    across = torch.full_like(columns, float(width))
    near = torch.minimum(
        torch.minimum(rows if top else down, (height - rows) if bottom else down),
        torch.minimum(columns if left else across, (width - columns) if right else across),
    )
    rise = (feathering - near) / feathering
    return torch.where(near < feathering, rise * rise, torch.zeros_like(rise)).to(torch.float32)


class ImagePadForOutpaint(io.ComfyNode):
    """Place a picture on a bigger canvas and mark the new room for a sampler."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImagePadForOutpaint",
            display_name="Image Pad for Outpaint",
            search_aliases=[
                "WASImagePadForOutpaint",
                "Image Pad for Outpaint",
                "ImagePadForOutpaint",
                "outpaint",
                "extend canvas",
                "expand image",
                "border",
            ],
            category="WAS Suite/Image/Process",
            description=(
                "Set a picture on a larger canvas of mid grey and answer the mask covering "
                "everything that was added, ready for an outpainting pass. The band on the "
                "node draws the frame that went in inside the canvas that came out, at one "
                "scale, with both sizes, the margins and the feather beside them, so four "
                "numbers typed into empty boxes are read off the node rather than queued to "
                "find out."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The picture to set on the larger canvas. A batch is padded frame by "
                        "frame by the same margins and comes back the same length."
                    ),
                ),
                io.Int.Input(
                    "left",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=8,
                    tooltip=(
                        "Pixels of new canvas added on the left; INT. 0 adds none, 256 adds a "
                        "quarter of a 1024 wide frame. Multiples of 8 keep the padded size on "
                        "a latent step."
                    ),
                ),
                io.Int.Input(
                    "top",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=8,
                    tooltip=(
                        "Pixels of new canvas added above; INT. 0 adds none, 256 adds a "
                        "quarter of a 1024 tall frame. Multiples of 8 keep the padded size on "
                        "a latent step."
                    ),
                ),
                io.Int.Input(
                    "right",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=8,
                    tooltip=(
                        "Pixels of new canvas added on the right; INT. 0 adds none, 256 adds "
                        "a quarter of a 1024 wide frame. Multiples of 8 keep the padded size "
                        "on a latent step."
                    ),
                ),
                io.Int.Input(
                    "bottom",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=8,
                    tooltip=(
                        "Pixels of new canvas added below; INT. 0 adds none, 256 adds a "
                        "quarter of a 1024 tall frame. Multiples of 8 keep the padded size on "
                        "a latent step."
                    ),
                ),
                io.Int.Input(
                    "feathering",
                    default=40,
                    min=0,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Pixels the mask fades over on the inside of each seam; INT. 0 = a "
                        "hard edge, 40 = a 40px falloff that lets a sampler blend the new "
                        "canvas into the frame, 128 = a wide blend that repaints more of the "
                        "original. Ignored where twice this reaches across the frame."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip=(
                        "The frame sitting on the larger canvas, every added margin filled "
                        "with mid grey at 0.5 for a sampler to paint over."
                    ),
                ),
                io.Mask.Output(
                    tooltip=(
                        "White over the added canvas and black over the frame, fading in over "
                        "feathering pixels at each seam. One plane whatever the batch length, "
                        "since every frame is padded alike. Wire it into Set Latent Noise Mask "
                        "or an inpaint conditioning node."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, image, left=0, top=0, right=0, bottom=0, feathering=40) -> io.NodeOutput:
        """Set every frame on the canvas and build the mask.

        Raises:
            MemoryError: Neither free memory nor a scratch drive can hold the canvas.
        """
        from ....modules.image import scratch

        frames, height, width, channels = (int(size) for size in image.shape)
        shape = (frames, height + top + bottom, width + left + right, channels)
        if image.device.type == "cpu":
            canvas = scratch.allocate(
                shape,
                node="Image Pad for Outpaint",
                advice="Smaller margins or fewer frames also fit it.",
            )
        else:
            canvas = torch.empty(shape, dtype=torch.float32, device=image.device)
        canvas[:, :top].fill_(CANVAS_LEVEL)
        canvas[:, top + height:].fill_(CANVAS_LEVEL)
        band = canvas[:, top:top + height]
        band[:, :, :left].fill_(CANVAS_LEVEL)
        band[:, :, left + width:].fill_(CANVAS_LEVEL)
        band[:, :, left:left + width].copy_(image)

        mask = torch.ones(
            (height + top + bottom, width + left + right),
            dtype=torch.float32,
            device=image.device,
        )
        kept = _feathered(height, width, left, top, right, bottom, feathering)
        mask[top:top + height, left:left + width] = kept.to(mask.device)

        cls.report(image, canvas, left, top, right, bottom, feathering, width, height)
        return io.NodeOutput(canvas, mask.unsqueeze(0))

    @classmethod
    def report(cls, image, canvas, left, top, right, bottom, feathering, width, height) -> None:
        """Draw both frames on the node and log a feather the frame is too small for.

        Args:
            image: The picture that went in.
            canvas: The larger canvas that came out.
            left: Pixels of canvas added on the left.
            top: Pixels of canvas added above.
            right: Pixels of canvas added on the right.
            bottom: Pixels of canvas added below.
            feathering: Pixels the fade was asked to run over.
            width: Frame width in pixels.
            height: Frame height in pixels.
        """
        faded = feathering > 0 and feathering * 2 < height and feathering * 2 < width
        if feathering > 0 and not faded:
            logger.warning(
                "Image Pad for Outpaint was given a feathering of %d, which needs a frame "
                "wider and taller than %d, and this one is %dx%d, so the mask has a hard "
                "edge at every seam. Set feathering under %d to soften it.",
                feathering, feathering * 2, width, height, min(width, height) // 2,
            )
        if feathering <= 0:
            fade = "off, hard seams"
        elif faded:
            fade = f"{feathering}px"
        else:
            fade = f"{feathering}px, too wide for the frame"
        size_report.publish(
            image,
            canvas,
            action="padded",
            facts={
                "margins": f"{left} left, {top} top, {right} right, {bottom} bottom",
                "feather": fade,
            },
        )

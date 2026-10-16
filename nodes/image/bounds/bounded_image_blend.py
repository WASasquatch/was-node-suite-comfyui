"""Blend a source image back into a bounded region of a target."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.sockets import require_input
from ....modules.compat.types import IMAGE_BOUNDS
from ....modules.convert.tensors import pil2tensor, tensor2pil
from ....modules.interface import size_report


def checked_window(node, row, width, height, feathering=0):
    """Refuse a bounds window that holds no pixels or is larger than the pack builds.

    Args:
        node: The node's display name, opening the message.
        row: The ``(rmin, rmax, cmin, cmax)`` bounds row the window came from.
        width: Window width in pixels.
        height: Window height in pixels.
        feathering: Fade width in pixels taken in from each edge, 0 for none.

    Raises:
        ValueError: The window is empty, or narrower than twice the feathering.
        CanvasTooLarge: The window is past 30000 pixels a side or 16384x16384 in all.
    """
    from ....modules.image import sizing

    if width < 1 or height < 1:
        raise ValueError(
            f"{node} was given the bounds row {tuple(row)}, which encloses no pixels: rmax "
            f"must be at least rmin and cmax at least cmin. Connect the bounds the region "
            f"was cropped with."
        )
    if min(width, height) < 2 * feathering:
        raise ValueError(
            f"{node} cannot fade a {width}x{height} window by {feathering} pixels from each "
            f"edge. Lower feathering to {min(width, height) // 2} or less."
        )
    sizing.check_canvas(
        width, height, f"{node}'s bounds window",
        "Narrow target_bounds, such as with Inset Image Bounds, before blending.",
    )


class BoundedImageBlend(io.ComfyNode):
    """Scale a source into a target's bounds and blend it in through a feathered mask."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Bounded Image Blend",
            display_name="Bounded Image Blend",
            search_aliases=["Bounded Image Blend", "paste region", "blend bounds"],
            category="WAS Suite/Image/Bound",
            description=(
                "Put a source image back into the region of a target image its bounds "
                "describe, stretched to fit and faded in at the edges. This is the return "
                "half of Bounded Image Crop: crop a region, work on it, then blend it home "
                "without a visible seam."
            ),
            inputs=[
                io.Image.Input(
                    "target",
                    tooltip=(
                        "The image being pasted into. The result is this image's size. One "
                        "target per source image pairs them up; any other count blends every "
                        "source into the first target."
                    ),
                ),
                IMAGE_BOUNDS.Input(
                    "target_bounds",
                    tooltip=(
                        "Where in the target the source lands, normally the same bounds the "
                        "region was cropped with. One row per source image pairs them up; any "
                        "other count uses the first row for all of them."
                    ),
                ),
                io.Image.Input(
                    "source",
                    tooltip=(
                        "The image pasted in, stretched to the size of the bounds whatever "
                        "its own size. The batch length here sets how many results come out."
                    ),
                ),
                io.Float.Input(
                    "blend_factor",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    tooltip=(
                        "How opaque the pasted region is. 1.0 replaces the target inside the "
                        "bounds, 0.0 leaves the target untouched, 0.5 mixes the two evenly."
                    ),
                ),
                io.Int.Input(
                    "feathering",
                    default=16,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip=(
                        "Width of the fade around the pasted region, in pixels. 0 = a hard "
                        "rectangle; 16 = a soft join. Must stay under half the bounds' width "
                        "and height, or the node raises an error."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip=(
                        "One image per source, each being the target with that source blended "
                        "into the bounded region."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, target, target_bounds, source, blend_factor, feathering) -> io.NodeOutput:
        """Blend each source image into its window of the target.

        Raises:
            ValueError: Nothing is connected to the target_bounds input, source holds no
                images, or a bounds window is empty or narrower than twice the feathering.
            CanvasTooLarge: A bounds window is past 30000 pixels a side or 16384x16384 in
                all.
            MemoryError: Neither free memory nor a scratch drive can hold the result.
        """
        from PIL import Image, ImageFilter, ImageOps

        from ....modules.image import scratch

        require_input(
            target_bounds,
            "Bounded Image Blend",
            "target_bounds",
            "bounds",
            "Image Bounds, Inset Image Bounds or Image Crop by Mask",
            "IMAGE_BOUNDS",
        )

        target = target.unsqueeze(0) if target.dim() == 3 else target
        source = source.unsqueeze(0) if source.dim() == 3 else source

        # A count that does not match the source batch means one target, or one bounds, for
        # every source image, so it is read once and reused.
        tgt_len = 1 if len(target) != len(source) else len(source)
        bounds_len = 1 if len(target_bounds) != len(source) else len(source)

        blended = None
        # The first window, kept for the readout.
        first_window = None
        for idx in range(len(source)):
            src = tensor2pil(source[idx])
            if (tgt_len == 1 and idx == 0) or tgt_len > 1:
                tgt = tensor2pil(target[idx])

            if (bounds_len == 1 and idx == 0) or bounds_len > 1:
                rmin, rmax, cmin, cmax = target_bounds[idx]

                height, width = (rmax - rmin + 1, cmax - cmin + 1)
                checked_window(
                    "Bounded Image Blend", (rmin, rmax, cmin, cmax), width, height, feathering
                )
                first_window = first_window or (width, height)

                if feathering > 0:
                    inner_mask = Image.new(
                        'L', (width - (2 * feathering), height - (2 * feathering)), 255
                    )
                    inner_mask = ImageOps.expand(inner_mask, border=feathering, fill=0)
                    inner_mask = inner_mask.filter(ImageFilter.GaussianBlur(radius=feathering))
                else:
                    inner_mask = Image.new('L', (width, height), 255)

                inner_mask = inner_mask.point(lambda p: p * blend_factor)

                tgt_mask = Image.new('L', tgt.size, 0)
                tgt_mask.paste(inner_mask, (cmin, rmin))

            src_resized = src.resize((width, height), Image.Resampling.LANCZOS)

            src_positioned = Image.new(tgt.mode, tgt.size)
            src_positioned.paste(src_resized, (cmin, rmin))

            frame = pil2tensor(Image.composite(src_positioned, tgt, tgt_mask))
            if blended is None:
                blended = scratch.allocate(
                    (len(source),) + tuple(frame.shape[1:]),
                    node="Bounded Image Blend",
                    advice="Passing fewer source images or a smaller target also fits it.",
                )
            blended[idx] = frame[0]

        if blended is None:
            raise ValueError(
                "Bounded Image Blend was given no source images. Connect an image or a batch "
                "of them to source."
            )

        # The canvas keeps the target's size, so the pair worth reporting is the source
        # against the window the bounds row cut for it.
        if first_window is not None:
            size_report.publish(
                source,
                first_window,
                action="blended",
                resampled=True,
                facts={"canvas": size_report.spell(target)},
            )

        return io.NodeOutput(blended)

"""Blend a source image into a bounded region of a target through a mask."""

from __future__ import annotations

import numpy as np
from comfy_api.latest import io

from ....modules.compat.sockets import require_input
from ....modules.compat.types import IMAGE_BOUNDS
from ....modules.convert.tensors import pil2tensor, tensor2pil
from ....modules.interface import size_report


class BoundedImageBlendWithMask(io.ComfyNode):
    """Scale a source into a target's bounds and blend it in through a supplied mask."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Bounded Image Blend with Mask",
            display_name="Bounded Image Blend with Mask",
            search_aliases=[
                "Bounded Image Blend with Mask",
                "paste region",
                "masked blend",
            ],
            category="WAS Suite/Image/Bound",
            description=(
                "Put a source image back into the region of a target image its bounds "
                "describe, letting a mask decide the shape of the join instead of a plain "
                "rectangle. Pair it with Bounded Image Crop with Mask, which produces both "
                "the crop and its bounds."
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
                io.Mask.Input(
                    "target_mask",
                    tooltip=(
                        "Where the source is allowed to show: white lets it through, black "
                        "keeps the target. It may be the size of the whole target or of the "
                        "bounds alone, in which case it is positioned at the bounds. One mask "
                        "per source image pairs them up; any other count uses the first for "
                        "all of them."
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
                        "How opaque the pasted region is where the mask allows it. 1.0 "
                        "replaces the target, 0.0 leaves the target untouched, 0.5 mixes the "
                        "two evenly."
                    ),
                ),
                io.Int.Input(
                    "feathering",
                    default=16,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip=(
                        "How far the mask's own edges are blurred, in pixels, which softens "
                        "the join. 0 uses the mask as it is, hard edges and all."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip=(
                        "One image per source, each being the target with that source blended "
                        "into the bounded region wherever the mask allows."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, target, target_mask, target_bounds, source, blend_factor,
                feathering) -> io.NodeOutput:
        """Blend each source image into its window of the target through the mask.

        Raises:
            ValueError: Nothing is connected to the target_bounds input, or source holds no
                images.
            MemoryError: Neither free memory nor a scratch drive can hold the result.
        """
        from PIL import Image, ImageFilter

        from ....modules.image import scratch

        require_input(
            target_bounds,
            "Bounded Image Blend with Mask",
            "target_bounds",
            "bounds",
            "Image Bounds, Inset Image Bounds or Image Crop by Mask",
            "IMAGE_BOUNDS",
        )

        target = target.unsqueeze(0) if target.dim() == 3 else target
        source = source.unsqueeze(0) if source.dim() == 3 else source
        target_mask = target_mask.unsqueeze(0) if target_mask.dim() == 2 else target_mask

        # A count that does not match the source batch means one target, one mask, or one
        # bounds for every source image, so it is read once and reused.
        tgt_mask_len = 1 if len(target_mask) != len(source) else len(source)
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
                first_window = first_window or (width, height)

            if (tgt_mask_len == 1 and idx == 0) or tgt_mask_len > 1:
                np_array = np.clip((target_mask[idx].cpu().numpy().squeeze() * 255.0), 0, 255)
                mask_plane = Image.fromarray((np_array).astype(np.uint8), mode='L')

            # The placed mask is built once for one mask and one bounds, and per frame otherwise.
            if (tgt_mask_len == 1 and bounds_len == 1 and idx == 0) or \
                    (tgt_mask_len > 1 or bounds_len > 1):
                tgt_mask = mask_plane

                # A mask the size of the target is already in place; anything else is taken
                # to be the size of the bounds and is positioned there.
                if tgt_mask.size != tgt.size:
                    mask_extended_canvas = Image.new('L', tgt.size, 0)

                    mask_extended_canvas.paste(tgt_mask, (cmin, rmin))

                    tgt_mask = mask_extended_canvas

                if feathering > 0:
                    tgt_mask = tgt_mask.filter(ImageFilter.GaussianBlur(radius=feathering))

                tgt_mask = tgt_mask.point(lambda p: p * blend_factor)

            src_resized = src.resize((width, height), Image.Resampling.LANCZOS)

            src_positioned = Image.new(tgt.mode, tgt.size)
            src_positioned.paste(src_resized, (cmin, rmin))

            frame = pil2tensor(Image.composite(src_positioned, tgt, tgt_mask))
            if blended is None:
                blended = scratch.allocate(
                    (len(source),) + tuple(frame.shape[1:]),
                    node="Bounded Image Blend with Mask",
                    advice="Passing fewer source images or a smaller target also fits it.",
                )
            blended[idx] = frame[0]

        if blended is None:
            raise ValueError(
                "Bounded Image Blend with Mask was given no source images. Connect an image or "
                "a batch of them to source."
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

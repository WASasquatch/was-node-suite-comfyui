"""Take one mask out of a batch."""

from __future__ import annotations

from comfy_api.latest import io

from . import mask_planes
from ...modules.compat import limits
from ...modules.interface import mask_report
from ...modules.log import get_logger

logger = get_logger("nodes.mask")


class MaskBatchToMask(io.ComfyNode):
    """Select a single mask from a batch by index."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Mask Batch to Mask",
            display_name="Mask Batch to Mask",
            search_aliases=["Mask Batch to Mask", "batch index", "select mask"],
            category="WAS Suite/Image/Masking",
            description="Pick one mask out of a batch. An index past the end of the batch "
            "falls back to the last mask and says so in the console.",
            inputs=[
                io.Mask.Input(
                    "masks",
                    tooltip=(
                        "The batch of masks to take one mask out of. A single unbatched mask "
                        "counts as a batch of one, so index 0 returns it as it is."
                    ),
                ),
                io.Int.Input(
                    "index",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Which mask to take, counting from zero: 0 = the first; 1 = the second. "
                        "An index past the end returns the last mask and prints the index and "
                        "batch length to the console instead of failing."
                    ),
                ),
            ],
            outputs=[
                io.Mask.Output(
                    tooltip="The one mask selected from the batch, no longer batched.",
                )
            ],
        )

    @classmethod
    def execute(cls, masks, index) -> io.NodeOutput:
        planes = mask_planes(masks)
        count = len(planes)
        if index < count:
            mask_report.publish(masks, planes[index], source="masks")
            return io.NodeOutput(planes[index])

        logger.error(
            "index is %s and this batch holds %s mask(s), numbered 0 to %s, so mask "
            "%s is returned instead",
            index, count, count - 1, count - 1,
        )
        mask_report.publish(masks, planes[-1], source="masks")
        return io.NodeOutput(planes[-1])

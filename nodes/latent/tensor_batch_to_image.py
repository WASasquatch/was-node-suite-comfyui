"""Pick one image out of a batched IMAGE tensor."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat import limits
from ...modules.interface import batch_report

logger = log.get_logger("latent.tensor_batch_to_image")


class TensorBatchToImage(io.ComfyNode):
    """Select a single image from a batch by index."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Tensor Batch to Image",
            display_name="Tensor Batch to Image",
            search_aliases=[
                "Tensor Batch to Image",
                "batch index",
                "select image",
                "image from batch",
            ],
            category="WAS Suite/Image",
            description=(
                "Return one image from a batched IMAGE tensor. An index beyond the batch "
                "returns the last image and says so in the console."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "A batch of images, such as the several images one sampler run "
                        "produces, to pick a single one out of."
                    ),
                ),
                io.Int.Input(
                    "index",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Which image to take, counting from 0: 0 = the first; 2 = the third. "
                        "A number past the end returns the last image and logs the index and "
                        "batch length to the console."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip="The single selected image, on its own as a batch of one.",
                ),
            ],
        )

    @classmethod
    def execute(cls, images, index) -> io.NodeOutput:
        count = len(images)
        if index < count:
            chosen = images[index].unsqueeze(0)
            cls.report(chosen, index, count, f"frame {index} of {count}")
            return io.NodeOutput(chosen)
        logger.error(
            "index is %s and this batch holds %s image(s), numbered 0 to %s, "
            "so image %s is returned instead",
            index, count, count - 1, count - 1,
        )
        chosen = images[-1].unsqueeze(0)
        cls.report(
            chosen, count - 1, count,
            f"frame {index} asked of {count}, frame {count - 1} returned",
            warn=True,
        )
        return io.NodeOutput(chosen)

    @staticmethod
    def report(chosen, index, count, detail, warn=False) -> None:
        """File which frame was taken, for the strip on the node.

        Args:
            chosen: The frame the node answers, as a batch of one.
            index: Which frame of the batch that is, counting from 0.
            count: How many frames there were to pick from.
            detail: How the pick reads in a sentence, for the summary line.
            warn: True where the index was held to the last frame.
        """
        size, mode = batch_report.describe_images(chosen)
        batch_report.publish_sample(
            1, count, "index", size, detail=detail, facts={"mode": mode, "frame": str(index)},
            warn=warn,
        )

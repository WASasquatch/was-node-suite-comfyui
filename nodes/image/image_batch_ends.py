"""Take the ends off an image batch."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat import limits


class ImageBatchEnds(io.ComfyNode):
    """Split an image batch into its end frames and the batch without them."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageBatchEnds",
            display_name="Image Batch Ends",
            search_aliases=[
                "WASImageBatchEnds",
                "Image Batch Ends",
                "last frame",
                "first frame",
                "drop first frame",
                "extend video",
                "trim batch",
            ],
            category="WAS Suite/Image",
            description=(
                "Take the ends off an image batch: the first and last frames, the first and "
                "last `count` frames, the batch without its first or last frame, and the frame "
                "count. Use `last` to seed a clip extension and `without_first` on what comes "
                "back, so the join has no repeated frame."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The batch to take the ends off. Frames come back in the order they "
                        "arrived, and a batch with no frames in it is refused."
                    ),
                ),
                io.Int.Input(
                    "count",
                    default=1,
                    min=1,
                    max=limits.max_resolution(),
                    tooltip=(
                        "How many frames first_n and last_n give back. 1 = one frame each; 8 = "
                        "the opening eight and the closing eight. More than the batch holds "
                        "gives the whole batch. first, last, without_first and without_last "
                        "ignore it."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="first",
                    tooltip=(
                        "Frame 0 on its own, as a batch of one. A batch of one answers that "
                        "frame here and on last."
                    ),
                ),
                io.Image.Output(
                    display_name="last",
                    tooltip=(
                        "The closing frame on its own, as a batch of one. Feed it to an image "
                        "to video model to carry on from where the clip stopped."
                    ),
                ),
                io.Image.Output(
                    display_name="first_n",
                    tooltip=(
                        "The opening count frames, in order. With 10 frames and count = 3 that "
                        "is frames 0, 1 and 2; a count above 10 gives all 10."
                    ),
                ),
                io.Image.Output(
                    display_name="last_n",
                    tooltip=(
                        "The closing count frames, in order. With 10 frames and count = 3 that "
                        "is frames 7, 8 and 9; a count above 10 gives all 10."
                    ),
                ),
                io.Image.Output(
                    display_name="without_first",
                    tooltip=(
                        "Every frame but frame 0, so 10 frames give 9. A continuation repeats "
                        "the frame it was seeded with, and this drops it before the join. A "
                        "batch of one has nothing to drop and answers its single frame."
                    ),
                ),
                io.Image.Output(
                    display_name="without_last",
                    tooltip=(
                        "Every frame but the closing one, so 10 frames give 9. It joins a clip "
                        "to a continuation from the other side. A batch of one has nothing to "
                        "drop and answers its single frame."
                    ),
                ),
                io.Int.Output(
                    display_name="batch_size",
                    tooltip=(
                        "How many frames arrived, so they are numbered 0 to batch_size - 1. "
                        "Feed it to a node that needs the length of the clip told to it."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, images, count=1) -> io.NodeOutput:
        """Answer each end of the batch, and the batch without each end.

        Args:
            images: The batch to take the ends off.
            count: How many frames first_n and last_n hold.

        Returns:
            The first frame, the last frame, the first and last ``count`` frames, the batch
            without its first frame, the batch without its last frame, and the batch size.

        Raises:
            ValueError: The batch holds no frames.
        """
        total = int(images.shape[0])
        if total == 0:
            raise ValueError(
                "Image Batch Ends was given a batch with no frames in it, so it has no first "
                "or last frame to answer. Check the node feeding images: a loader that matched "
                "no files, or a sampler that kept no frames, produces an empty batch."
            )

        wanted = max(1, min(int(count), total))
        # A batch of one has nothing left after either end is dropped, so it answers itself.
        without_first = images[1:] if total > 1 else images
        without_last = images[:-1] if total > 1 else images

        return io.NodeOutput(
            images[:1],
            images[total - 1:],
            images[:wanted],
            images[total - wanted:],
            without_first,
            without_last,
            total,
        )

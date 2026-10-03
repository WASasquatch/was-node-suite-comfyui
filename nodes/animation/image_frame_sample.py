"""Keep a smaller set of frames from an image batch."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat import limits
from ...modules.interface import batch_report
from ...modules.media import sampling

logger = log.get_logger("nodes.animation")


class ImageFrameSample(io.ComfyNode):
    """Sample frames from an image batch by one of several strategies."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageFrameSample",
            display_name="Image Frame Sample",
            search_aliases=[
                "WASImageFrameSample",
                "Image Frame Sample",
                "sample frames",
                "every nth frame",
                "frame select",
            ],
            category="WAS Suite/Animation",
            description=(
                "Keep a smaller set of frames from an image batch: evenly spaced, the first, "
                "middle or last few, a random pick, or every nth frame. start and end narrow "
                "the range it picks from."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The frames to sample, in order.",
                ),
                io.Int.Input(
                    "start",
                    default=0,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    optional=True,
                    tooltip=(
                        "First frame to consider, counting from 0. Negative counts back from "
                        "the end, so -30 starts thirty frames before it. The strategy picks "
                        "from this range rather than from the whole sequence."
                    ),
                ),
                io.Int.Input(
                    "end",
                    default=-1,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    optional=True,
                    tooltip=(
                        "Last frame to consider, inclusive. -1 is the final frame, which is "
                        "the whole sequence together with a start of 0. An end before the "
                        "start is ignored and the whole sequence is used."
                    ),
                ),
                io.Int.Input(
                    "num_frames",
                    default=16,
                    min=1,
                    max=limits.max_resolution(),
                    tooltip=(
                        "How many frames to keep, eg 16. Capped at what the batch holds, and "
                        "every_nth stops here too."
                    ),
                ),
                io.Combo.Input(
                    "strategy",
                    options=list(sampling.STRATEGIES),
                    default="uniform",
                    tooltip=(
                        "uniform = evenly spaced; head = first; center = middle; tail = last; "
                        "random = a seeded pick; every_nth = every nth frame. head, center and "
                        "tail are the consecutive ones anything temporal needs."
                    ),
                ),
                io.Int.Input(
                    "nth",
                    default=1,
                    min=1,
                    max=limits.max_resolution(),
                    tooltip=(
                        "Step between the frames the strategy may choose from. 1 uses every "
                        "frame; 2 thins to every other one first, so `head` takes the opening "
                        "of the clip on alternate frames. It applies to every strategy."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=io.ControlAfterGenerate.fixed,
                    tooltip=(
                        "Seed for random, so a re-run keeps the same frames. Ignored by the "
                        "other strategies. Any whole number; `0` is as good a seed as any."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="The frames that were kept, in order.",
                ),
            ],
        )

    @classmethod
    def execute(cls, images, num_frames, strategy, nth, seed, start=0, end=-1) -> io.NodeOutput:
        total = int(images.shape[0])
        first, stop = sampling.slice_bounds(total, start, end)
        window = images[first:stop]
        indices = sampling.frame_indices(int(window.shape[0]), num_frames, strategy, nth, seed)
        kept = window[indices]
        logger.debug(
            "kept %d of %d frame(s) by %s, from frames %d to %d",
            len(indices), total, strategy, first, stop - 1,
        )

        size, mode = batch_report.describe_images(kept)
        batch_report.publish_sample(
            len(indices), total, strategy, size,
            detail=sampling.describe(strategy, nth)
            + ("" if (first, stop) == (0, total) else f", of frames {first} to {stop - 1}"),
            facts={
                "mode": mode,
                "memory": batch_report.readable_bytes(batch_report.memory_of(kept)),
            },
        )
        return io.NodeOutput(kept)

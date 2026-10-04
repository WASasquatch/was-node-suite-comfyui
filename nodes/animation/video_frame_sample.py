"""Keep a smaller set of frames from a video."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat import limits
from ...modules.interface import batch_report
from ...modules.media import sampling

logger = log.get_logger("nodes.animation")


class VideoFrameSample(io.ComfyNode):
    """Sample frames from a video by one of several strategies."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASVideoFrameSample",
            display_name="Video Frame Sample (Advanced)",
            search_aliases=[
                "WASVideoFrameSample",
                "Video Frame Sample",
                "sample frames",
                "every nth frame",
                "trim video",
            ],
            category="WAS Suite/Animation",
            description=(
                "Keep a smaller set of frames from a video: evenly spaced, the first, middle "
                "or last few, a random pick, or every nth frame."
            ),
            inputs=[
                io.Video.Input(
                    "video",
                    tooltip="The video to sample.",
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
                        "from this range rather than from the whole clip."
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
                        "the whole clip together with a start of 0. An end before the start "
                        "is ignored and the whole clip is used."
                    ),
                ),
                io.Int.Input(
                    "num_frames",
                    default=16,
                    min=1,
                    max=limits.max_resolution(),
                    tooltip=(
                        "How many frames to keep, eg 16. Capped at what the video holds, and "
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
                        "tail are consecutive, and decode nothing."
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
                io.Video.Output(
                    display_name="video",
                    tooltip="The frames that were kept, as a video at the source's rate.",
                ),
            ],
        )

    @classmethod
    def execute(cls, video, num_frames, strategy, nth, seed, start=0, end=-1) -> io.NodeOutput:
        total = int(video.get_frame_count())
        rate = float(video.get_frame_rate())
        width, height = video.get_dimensions()
        first, stop = sampling.slice_bounds(total, start, end)
        span = stop - first

        if strategy in sampling.CONTIGUOUS:
            offset, taken = sampling.frame_span(span, num_frames, strategy)
            offset += first
            logger.debug("trimming %d frame(s) from %d at %d", taken, total, offset)
            # Trimmed rather than decoded: the frames are consecutive, so the source can
            # answer them lazily and a long clip costs no more than a short one.
            answer = video.as_trimmed(offset / rate, taken / rate, strict_duration=False)
            cls.report(taken, total, strategy, nth, width, height, rate, decoded=False)
            return io.NodeOutput(answer)

        picked = sampling.frame_indices(span, num_frames, strategy, nth, seed)
        indices = [first + index for index in picked]
        logger.debug("decoding %d of %d frame(s) by %s", len(indices), total, strategy)
        answer = sampling.decode_frames(video, indices, "Video Frame Sample (Advanced)")
        cls.report(len(indices), total, strategy, nth, width, height, rate, decoded=True)
        return io.NodeOutput(answer)

    @staticmethod
    def report(kept, total, strategy, nth, width, height, rate, decoded) -> None:
        """Publish what was kept, for the strip on the node.

        Args:
            kept: How many frames the node answered.
            total: How many the video held.
            strategy: Which strategy chose them.
            nth: The step 'every nth' counts by, which the other strategies ignore.
            width: Frame width.
            height: Frame height.
            rate: Frames per second.
            decoded: Whether frames had to be decoded to answer.
        """
        batch_report.publish_sample(
            kept, total, strategy, f"{int(width)}x{int(height)}",
            detail=sampling.describe(strategy, nth),
            facts={
                "rate": f"{rate:.6g} fps",
                "read": "decoded" if decoded else "trimmed, nothing decoded",
            },
        )

"""Cutting an image batch in two at a frame number.

The cut is the first frame of the tail, and it runs from 1 to one less than the batch size,
so both sides always hold at least one frame.
"""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat import limits
from ...modules.compat.types import NUMBER
from ...modules.logic.switch_index import OUT_OF_RANGE

#: The name the messages below report the node by.
NODE = "Image Batch Split"


def resolve_cut(at, count: int, out_of_range: str) -> int:
    """Turn a requested cut point into the frame number the tail starts at.

    Args:
        at: The requested cut, counting from 0. Negative counts back from the end.
        count: How many frames the batch holds.
        out_of_range: One of :data:`modules.logic.switch_index.OUT_OF_RANGE`.

    Returns:
        A cut from 1 to ``count - 1``.

    Raises:
        ValueError: The batch holds fewer than two frames, or the cut is outside that range
            and ``out_of_range`` is ``error``.
    """
    if count < 2:
        raise ValueError(
            f"{NODE} was given {count} frame(s) and needs at least 2 to cut in two. Feed it a "
            f"batch rather than a single image, or join more frames on with one of the Image "
            f"Batch nodes first."
        )
    wanted = int(at)
    cut = wanted + count if wanted < 0 else wanted
    if 1 <= cut < count:
        return cut
    if out_of_range == "clamp":
        return 1 if cut < 1 else count - 1
    if out_of_range == "wrap":
        # Wraps over the cuts 1 to count-1 rather than over 0 to count-1.
        return (cut - 1) % (count - 1) + 1
    raise ValueError(
        f"{NODE} was asked to cut at {wanted} and the batch holds {count} frame(s), so the cut "
        f"has to be 1 to {count - 1}, or -1 to -{count - 1} counting from the end. Change at, "
        f"or set out_of_range to clamp or wrap."
    )


class ImageBatchSplit(io.ComfyNode):
    """Cut an image batch into the frames before a cut point and the frames from it onward."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageBatchSplit",
            display_name="Image Batch Split",
            search_aliases=[
                "WASImageBatchSplit",
                "Image Batch Split",
                "split batch",
                "cut batch",
                "divide batch",
                "drop first frame",
                "drop last frame",
                "head and tail",
            ],
            category="WAS Suite/Image",
            description=(
                "Cut an image batch in two at a frame number: frames before the cut come out "
                "on head, the cut frame and everything after on tail. A cut at -1 holds back "
                "the last frame, for dropping an overlap before stitching on a continuation."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The batch to cut. It needs at least 2 frames, and neither side is "
                        "altered: head and tail joined back together are the batch that "
                        "came in."
                    ),
                ),
                io.MultiType.Input(
                    io.Int.Input(
                        "at",
                        default=1,
                        min=-limits.max_resolution(),
                        max=limits.max_resolution(),
                        step=1,
                    ),
                    [io.Int, NUMBER, io.Float],
                    tooltip=(
                        "Where to cut, counting frames from 0; that frame starts the tail. With "
                        "8 frames: 3 = head 0 to 2, tail 3 to 7; -1 = the last frame alone on "
                        "tail. A decimal is truncated: 2.7 = 2."
                    ),
                ),
                io.Combo.Input(
                    "out_of_range",
                    options=list(OUT_OF_RANGE),
                    default="clamp",
                    tooltip=(
                        "A cut outside 1..count-1, where both sides keep a frame. With 4 "
                        "frames and at 5: `wrap` = 2, `clamp` = 3, `error` stops the prompt. "
                        "An at of 0 is outside as well: `wrap` = 3, `clamp` = 1."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="head",
                    tooltip=(
                        "Frames before the cut. With 8 frames and at 3 that is frames 0, 1 "
                        "and 2. Never empty, so whatever reads it always has a frame."
                    ),
                ),
                io.Image.Output(
                    display_name="tail",
                    tooltip=(
                        "The cut frame and everything after it. With 8 frames and at 3 that "
                        "is frames 3 to 7, so a cut at -1 answers the final frame on its own."
                    ),
                ),
                io.Int.Output(
                    display_name="head_count",
                    tooltip=(
                        "How many frames head holds, which is the cut point after clamp or "
                        "wrap has moved it. Always 1 or more."
                    ),
                ),
                io.Int.Output(
                    display_name="tail_count",
                    tooltip=(
                        "How many frames tail holds, which is the batch size minus "
                        "head_count. Always 1 or more."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, images, at=1, out_of_range="clamp") -> io.NodeOutput:
        """Cut the batch in two at the requested frame.

        Args:
            images: The batch to cut.
            at: Where to cut, counting frames from 0.
            out_of_range: What a cut outside the batch does.

        Returns:
            The frames before the cut, the frames from the cut onward, and how many frames
            each side holds.

        Raises:
            ValueError: The batch holds fewer than two frames, or the cut is refused.
        """
        count = int(images.shape[0])
        cut = resolve_cut(at, count, out_of_range)
        return io.NodeOutput(images[:cut], images[cut:], cut, count - cut)

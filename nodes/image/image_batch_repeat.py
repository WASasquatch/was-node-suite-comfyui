"""Repeat an image batch into a longer one, by a count or up to an exact length."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ...modules.compat import limits

#: Repeat the frames a set number of times.
MODE_TIMES = "times"

#: Repeat the frames until the batch holds exactly the length asked for.
MODE_LENGTH = "to length"

#: The ways a repeat is measured, in the order they are offered.
MODES: tuple[str, ...] = (MODE_TIMES, MODE_LENGTH)


def repeat_indices(count: int, mode: str, times: int, length: int, each_frame: bool) -> list[int]:
    """Which source frame each output frame is taken from.

    Args:
        count: How many frames the batch holds, which is one or more.
        mode: One of :data:`MODES`.
        times: How many times the frames are used, for :data:`MODE_TIMES`.
        length: How many frames come out, for :data:`MODE_LENGTH`.
        each_frame: Repeat each frame in place rather than the whole run.

    Returns:
        A position from 0 to ``count - 1`` for every output frame, in order.

    Raises:
        ValueError: ``mode`` is unknown, or the count it reads is below 1.
    """
    if mode == MODE_LENGTH:
        if length < 1:
            raise ValueError(
                f"Image Batch Repeat was asked for a length of {length}, and a batch cannot "
                "be empty. Set length to 1 or more."
            )
        if each_frame:
            # The share each frame holds is the length over the count, in whole frames.
            return [position * count // length for position in range(length)]
        return [position % count for position in range(length)]
    if mode == MODE_TIMES:
        if times < 1:
            raise ValueError(
                f"Image Batch Repeat was asked to repeat {times} times. Set times to 1 or "
                "more, where 1 passes the batch through unchanged."
            )
        if each_frame:
            return [position for position in range(count) for _ in range(times)]
        return [position % count for position in range(count * times)]
    raise ValueError(
        f"Image Batch Repeat does not know the mode {mode!r}. Set mode to "
        f"{MODE_TIMES!r} or {MODE_LENGTH!r}."
    )


class ImageBatchRepeat(io.ComfyNode):
    """Lengthen an image batch by repeating the whole run or each frame in place."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageBatchRepeat",
            display_name="Image Batch Repeat",
            search_aliases=[
                "WASImageBatchRepeat",
                "Image Batch Repeat",
                "repeat frames",
                "loop batch",
                "extend clip",
                "hold frame",
                "still frames",
            ],
            category="WAS Suite/Image",
            description=(
                "Repeat an image batch into a longer one: a set number of times, or up to an "
                "exact frame count. 'to length' cuts the last repeat short so the answer is "
                "exactly the length asked for, which is how a short clip is extended to the "
                "81 frames a video model wants. Turn each_frame on and every frame is "
                "repeated where it stands instead, so a single image becomes a still run and "
                "a clip is slowed down evenly."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The frames to repeat, in order. A single image is a batch of one.",
                ),
                io.Combo.Input(
                    "mode",
                    options=list(MODES),
                    default=MODE_TIMES,
                    tooltip=(
                        "`times` = use the frames as many times as asked; `to length` = "
                        "repeat until there are exactly length frames, cutting the last "
                        "repeat short. `times` ignores length, `to length` ignores times."
                    ),
                ),
                io.Int.Input(
                    "times",
                    default=2,
                    min=1,
                    max=limits.max_resolution(),
                    tooltip=(
                        "How many times the frames are used, so 1 = unchanged and 3 = three "
                        "copies. 4 frames at times 3 answer 12 frames. Ignored while mode is "
                        "`to length`."
                    ),
                ),
                io.Int.Input(
                    "length",
                    default=16,
                    min=1,
                    max=limits.max_resolution(),
                    tooltip=(
                        "Exactly how many frames come out, eg 81 for a model that wants 81. "
                        "4 frames to length 10 answer A B C D A B C D A B. A length under "
                        "the batch size trims it instead. Ignored while mode is `times`."
                    ),
                ),
                io.Boolean.Input(
                    "each_frame",
                    default=False,
                    tooltip=(
                        "Repeat each frame in place rather than the whole run. A B C at "
                        "times 2: false = A B C A B C; true = A A B B C C. On `to length` "
                        "each frame is held an even share: 3 frames to length 7 = A A A B B "
                        "C C."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip=(
                        "The repeated frames, in order, at the same size and channel count "
                        "as the input."
                    ),
                ),
                io.Int.Output(
                    display_name="batch_size",
                    tooltip=(
                        "How many frames came out, which is exactly length in `to length` "
                        "mode. Feed it to anything that needs the run's frame count told to "
                        "it."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, images, mode, times, length, each_frame) -> io.NodeOutput:
        """Build the longer batch and say how many frames it holds.

        Args:
            images: The frames to repeat.
            mode: One of :data:`MODES`.
            times: How many times the frames are used.
            length: How many frames come out.
            each_frame: Repeat each frame in place rather than the whole run.

        Returns:
            The repeated frames and their count.

        Raises:
            ValueError: The batch is empty, the mode is unknown, or a count is below 1.
            MemoryError: Neither free memory nor any scratch drive has room for the batch.
        """
        count = int(images.shape[0])
        if count == 0:
            raise ValueError(
                "Image Batch Repeat was given an empty batch and has nothing to repeat. "
                "Connect an images input carrying at least one frame."
            )
        from ...modules.image import scratch

        order = repeat_indices(count, str(mode), int(times), int(length), bool(each_frame))
        index = torch.tensor(order, dtype=torch.long, device=images.device)
        shape = (len(order),) + tuple(images.shape[1:])
        if images.device.type == "cpu":
            repeated = scratch.allocate(
                shape,
                images.dtype,
                node="Image Batch Repeat",
                advice="Lowering times or length, or passing fewer frames, also fits it.",
            )
        else:
            repeated = images.new_empty(shape)
        torch.index_select(images, 0, index, out=repeated)
        return io.NodeOutput(repeated, int(repeated.shape[0]))

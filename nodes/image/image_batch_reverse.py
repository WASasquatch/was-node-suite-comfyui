"""Play a batch of frames backwards, or forwards and back again."""

from __future__ import annotations

import torch
from comfy_api.latest import io


def replay(images, mode: str):
    """Reorder a batch of frames into the playback the mode names.

    Args:
        images: An ``IMAGE`` batch, ``(frames, height, width, channels)``.
        mode: ``reverse``, ``ping-pong`` or ``ping-pong trimmed``.

    Returns:
        The frames in their new order, as a new batch.

    Raises:
        ValueError: The batch holds no frames, or the mode is not one of the three.
        MemoryError: Neither free memory nor any scratch drive has room for the batch.
    """
    count = int(images.shape[0])
    if count == 0:
        raise ValueError(
            "Image Batch Reverse was handed a batch with no frames in it. Connect the images "
            "input to something that produces at least one frame, such as Load Image, a "
            "sampler, or Load Video."
        )
    backwards = list(range(count - 1, -1, -1))
    if mode == "reverse":
        order = backwards
    elif mode == "ping-pong":
        order = list(range(count)) + backwards
    elif mode == "ping-pong trimmed":
        # The tail starts one frame in and stops one frame short.
        order = list(range(count)) + backwards[1:-1]
    else:
        raise ValueError(
            f"Image Batch Reverse does not know the mode {mode!r}. Set mode to 'reverse', "
            "'ping-pong' or 'ping-pong trimmed'."
        )

    from ...modules.image import scratch

    shape = (len(order),) + tuple(images.shape[1:])
    if images.device.type == "cpu":
        played = scratch.allocate(
            shape,
            images.dtype,
            node="Image Batch Reverse",
            advice="Passing fewer frames, or choosing reverse over ping-pong, also fits it.",
        )
    else:
        played = images.new_empty(shape)
    index = torch.tensor(order, dtype=torch.long, device=images.device)
    return torch.index_select(images, 0, index, out=played)


class ImageBatchReverse(io.ComfyNode):
    """Reverse a batch of frames, or append the reverse to make it ping-pong."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageBatchReverse",
            display_name="Image Batch Reverse",
            search_aliases=[
                "WASImageBatchReverse",
                "Image Batch Reverse",
                "reverse frames",
                "ping pong",
                "boomerang",
                "loop clip",
                "play backwards",
            ],
            category="WAS Suite/Image",
            description=(
                "Play a batch of frames backwards, or append the reverse so a short clip "
                "runs out and back and loops on itself. 'ping-pong' shows the first and "
                "last frames twice at the joins, which reads as a pause; 'ping-pong "
                "trimmed' leaves them out, so the loop runs at an even pace. The frame "
                "count comes out beside the images, ready for a video save."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The frames to reorder, in the order they arrived. Feed it a video "
                        "load, a sampler's frames, or any batch. A one-frame batch comes "
                        "back as it went in, except on `ping-pong`, which shows that frame "
                        "twice."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=["reverse", "ping-pong", "ping-pong trimmed"],
                    default="reverse",
                    tooltip=(
                        "How the frames are laid out. Given 8 frames: `reverse` = 8 frames, "
                        "played 8 to 1; `ping-pong` = 16, 1 to 8 then 8 to 1, holding on "
                        "frames 8 and 1 for two; `ping-pong trimmed` = 14, 1 to 8 then 7 to "
                        "2, which is the one that loops without a stutter."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip=(
                        "The frames in their new order, same size and count of channels as "
                        "they went in. Wire it to Save Video or an encoder."
                    ),
                ),
                io.Int.Output(
                    display_name="batch_size",
                    tooltip=(
                        "How many frames came out: 8 in gives 8 on `reverse`, 16 on "
                        "`ping-pong` and 14 on `ping-pong trimmed`. Feed it to whatever "
                        "downstream needs the length told to it."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, images, mode="reverse") -> io.NodeOutput:
        """Answer the reordered frames and how many there are.

        Args:
            images: The frames to reorder.
            mode: Which playback to lay out.

        Returns:
            The reordered batch, and its frame count.

        Raises:
            ValueError: The batch is empty, or the mode is unknown.
            MemoryError: Neither free memory nor any scratch drive has room for the batch.
        """
        played = replay(images, mode)
        return io.NodeOutput(played, int(played.shape[0]))

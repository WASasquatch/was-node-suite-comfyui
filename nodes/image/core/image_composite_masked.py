"""Paste one picture onto another at a pixel position, through an optional mask."""

from __future__ import annotations

import math

import torch
from comfy_api.latest import io

from ....modules.compat import limits


def _matched_channels(destination, source):
    """Bring a source's channel count to the destination's.

    Args:
        destination: ``(batch, height, width, channels)`` picture being pasted onto.
        source: ``(batch, height, width, channels)`` picture being pasted.

    Returns:
        ``(destination, source)``, the source trimmed to the destination's channels or given
        opaque channels where it had fewer.

    Raises:
        ValueError: The source carries fewer than three channels and the destination carries
            more than the source.
    """
    wanted, carried = int(destination.shape[-1]), int(source.shape[-1])
    if wanted < carried:
        return destination, source[..., :wanted]
    if wanted == carried:
        return destination, source
    if carried < 3:
        raise ValueError(
            f"Image Composite Masked was given a source of {carried} channel(s) and a "
            f"destination of {wanted}, so there is no colour in the source to paste. Wire "
            "the source through Images to RGB first, or feed a picture with red, green and "
            "blue channels."
        )
    source = torch.nn.functional.pad(source, (0, wanted - carried))
    source[..., carried:] = 1.0
    return destination, source


def _to_batch(tensor, batch):
    """Bring a tensor's batch axis to a given length.

    Args:
        tensor: Any tensor whose first axis is the batch.
        batch: How many entries the first axis is to hold.

    Returns:
        The tensor trimmed to ``batch`` entries, or repeated from the start until it holds
        that many.
    """
    if tensor.shape[0] > batch:
        return tensor.narrow(0, 0, batch)
    if tensor.shape[0] < batch:
        times = math.ceil(batch / tensor.shape[0])
        return tensor.repeat([times] + [1] * (tensor.ndim - 1)).narrow(0, 0, batch)
    return tensor


def _composite(destination, source, x, y, mask, resize_source):
    """Write the source into the destination in place, in channels-first layout.

    Args:
        destination: ``(batch, channels, height, width)`` picture, written into.
        source: ``(batch, channels, height, width)`` picture being placed.
        x: Column the source's left edge lands on.
        y: Row the source's top edge lands on.
        mask: A ``MASK`` tensor deciding where the source shows, or None for all of it.
        resize_source: Bring the source to the destination's whole frame first.

    Returns:
        The destination tensor that was passed in, now carrying the composite.
    """
    source = source.to(destination.device)
    if resize_source:
        source = torch.nn.functional.interpolate(
            source, size=(destination.shape[-2], destination.shape[-1]), mode="bilinear"
        )
    source = _to_batch(source, destination.shape[0])

    x = max(-source.shape[-1], min(x, destination.shape[-1]))
    y = max(-source.shape[-2], min(y, destination.shape[-2]))
    left, top = x, y
    right, bottom = left + source.shape[-1], top + source.shape[-2]

    if mask is None:
        mask = torch.ones_like(source)
    else:
        mask = mask.to(destination.device, copy=True)
        mask = torch.nn.functional.interpolate(
            mask.reshape((-1, 1, mask.shape[-2], mask.shape[-1])),
            size=(source.shape[-2], source.shape[-1]),
            mode="bilinear",
        )
        mask = _to_batch(mask, source.shape[0])

    # Only the part of the source standing over the destination is written.
    visible_width = destination.shape[-1] - left + min(0, x)
    visible_height = destination.shape[-2] - top + min(0, y)
    mask = mask[:, :, :visible_height, :visible_width]
    if mask.ndim < source.ndim:
        mask = mask.unsqueeze(1)

    inverse = torch.ones_like(mask) - mask
    covered = mask * source[..., :visible_height, :visible_width]
    kept = inverse * destination[..., top:bottom, left:right]
    destination[..., top:bottom, left:right] = covered + kept
    return destination


class ImageCompositeMasked(io.ComfyNode):
    """Lay one picture over another at a position, weighted by a mask."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageCompositeMasked",
            display_name="Image Composite Masked",
            search_aliases=[
                "WASImageCompositeMasked",
                "Image Composite Masked",
                "ImageCompositeMasked",
                "composite",
                "overlay",
                "paste image",
                "layer",
            ],
            category="WAS Suite/Image/Process",
            description=(
                "Lay one picture over another at a pixel position, showing it only where a "
                "mask allows. The band on the node draws the destination beside the result "
                "with the source that went between them, and measures the difference, so a "
                "source dropped off the edge or a mask the wrong way round shows on the node "
                "rather than after a preview is wired up."
            ),
            inputs=[
                io.Image.Input(
                    "destination",
                    tooltip=(
                        "The picture the source is laid over. Its size and batch length are "
                        "what come out, and every frame is composited alike."
                    ),
                ),
                io.Image.Input(
                    "source",
                    tooltip=(
                        "The picture being laid on top. It is repeated or trimmed to the "
                        "destination's batch length, and whatever hangs off the right or "
                        "bottom edge is cut."
                    ),
                ),
                io.Int.Input(
                    "x",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Pixels from the left of the destination to the source's left edge; "
                        "INT. 0 sits it in the corner, 256 moves it 256 across."
                    ),
                ),
                io.Int.Input(
                    "y",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Pixels from the top of the destination to the source's top edge; "
                        "INT. 0 sits it in the corner, 256 moves it 256 down."
                    ),
                ),
                io.Boolean.Input(
                    "resize_source",
                    default=False,
                    tooltip=(
                        "Whether the source is stretched to the destination's whole frame "
                        "first. `false` places it at its own size at x,y; `true` fills the "
                        "frame edge to edge, and x and y then only push it off."
                    ),
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip=(
                        "Where the source shows through: white shows all of it, black none, "
                        "0.5 mixes the two evenly. Scaled to the source's size. Unconnected, "
                        "the whole source is pasted."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip=(
                        "The destination with the source composited into it, at the "
                        "destination's size, batch length and channel count."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, destination, source, x=0, y=0, resize_source=False,
                mask=None) -> io.NodeOutput:
        from ....modules.image import scratch

        sources = int(source.shape[0])
        masks = None if mask is None else mask.reshape((-1, mask.shape[-2], mask.shape[-1]))
        if not sources:
            raise ValueError(
                "Image Composite Masked was given a source holding no frames, so there is "
                "nothing to paste. Wire a source holding at least one picture."
            )
        if masks is not None and not masks.shape[0]:
            raise ValueError(
                "Image Composite Masked was given a mask holding no frames. Wire a mask "
                "holding at least one, or leave mask unconnected to paste the whole source."
            )
        if destination.device.type == "cpu":
            canvas = scratch.allocate(
                destination.shape, destination.dtype, node="Image Composite Masked",
                advice="A shorter or smaller destination also fits it.",
            )
        else:
            canvas = torch.empty_like(destination)
        for index in range(int(destination.shape[0])):
            pick = index % sources
            _, placed = _matched_channels(destination, source[pick:pick + 1])
            held = None
            if masks is not None:
                pick = index % int(masks.shape[0])
                held = masks[pick:pick + 1]
            frame = canvas[index:index + 1]
            frame.copy_(destination[index:index + 1])
            _composite(frame.movedim(-1, 1), placed.movedim(-1, 1), x, y, held, resize_source)
        return io.NodeOutput(canvas)

"""Batch any number of images into one, on a slot list that grows."""

from __future__ import annotations

import functools

import torch
from comfy_api.latest import io

from ...modules.image.batching import (
    IMAGE_SLOTS,
    MAX_SLOTS,
    as_batch,
    check_image_dimensions,
)
from ...modules.interface import batch_report
from ...modules.util.slots import connected_in_order


class ImageBatch(io.ComfyNode):
    """Concatenate the connected image slots into one IMAGE."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Batch",
            display_name="Image Batch",
            search_aliases=[
                "Image Batch", "WASImageBatchAutogrow", "batch images", "combine images",
                "concatenate images", "dynamic batch",
            ],
            category="WAS Suite/Image",
            description=(
                "Join any number of images into one batch that later nodes process in a single "
                "pass. A new empty slot appears below the last one filled, up to 26. Every "
                "image must share a width, a height and a channel count, and a slot holding a "
                "batch contributes all of its frames."
            ),
            inputs=[
                io.Image.Input(
                    "images_a",
                    optional=True,
                    tooltip=(
                        "First image or batch. Every connected slot has to share the same "
                        "width, height and channel count, and at least one slot must be "
                        "connected."
                    ),
                ),
                io.Image.Input(
                    "images_b",
                    optional=True,
                    tooltip="Second image or batch. Leave it disconnected to skip it.",
                ),
                io.Image.Input(
                    "images_c",
                    optional=True,
                    tooltip="Third image or batch. Leave it disconnected to skip it.",
                ),
                io.Image.Input(
                    "images_d",
                    optional=True,
                    tooltip="Fourth image or batch. Leave it disconnected to skip it.",
                ),
                io.Image.Input(
                    "images_e",
                    optional=True,
                    tooltip="Image 5, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "images_f",
                    optional=True,
                    tooltip="Image 6, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "images_g",
                    optional=True,
                    tooltip="Image 7, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "images_h",
                    optional=True,
                    tooltip="Image 8, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "images_i",
                    optional=True,
                    tooltip="Image 9, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "images_j",
                    optional=True,
                    tooltip="Image 10, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "images_k",
                    optional=True,
                    tooltip="Image 11, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "images_l",
                    optional=True,
                    tooltip="Image 12, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "images_m",
                    optional=True,
                    tooltip="Image 13, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "images_n",
                    optional=True,
                    tooltip="Image 14, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "images_o",
                    optional=True,
                    tooltip="Image 15, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "images_p",
                    optional=True,
                    tooltip="Image 16, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "images_q",
                    optional=True,
                    tooltip="Image 17, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "images_r",
                    optional=True,
                    tooltip="Image 18, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "images_s",
                    optional=True,
                    tooltip="Image 19, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "images_t",
                    optional=True,
                    tooltip="Image 20, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "images_u",
                    optional=True,
                    tooltip="Image 21, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "images_v",
                    optional=True,
                    tooltip="Image 22, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "images_w",
                    optional=True,
                    tooltip="Image 23, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "images_x",
                    optional=True,
                    tooltip="Image 24, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "images_y",
                    optional=True,
                    tooltip="Image 25, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Image.Input(
                    "images_z",
                    optional=True,
                    tooltip="Image 26, the last slot. Unconnected is skipped.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="image",
                    tooltip=(
                        "One batch holding every connected input end to end, in slot order. A "
                        "slot holding a batch contributes all of its frames."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip=(
                        "How many frames the batch holds, which is the total across the slots "
                        "rather than the number of slots."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, **images) -> io.NodeOutput:
        """Concatenate every connected slot into one batch.

        Raises:
            ValueError: No slot holds an image, a slot holds something that is not one, or
                two frame shapes disagree.
            MemoryError: Neither free memory nor any scratch drive has room for the batch.
        """
        names = connected_in_order(images, IMAGE_SLOTS)
        tensors = [as_batch(images[name], name) for name in names]

        if not tensors:
            raise ValueError(
                f"Image Batch has no images connected. Connect at least one of the {MAX_SLOTS} "
                f"slots."
            )

        size, mode = batch_report.describe_images(tensors[0])
        try:
            check_image_dimensions(tensors, names)
        except ValueError as refused:
            # The refusal reaches the node's own panel before it is raised.
            batch_report.publish(
                frames=sum(int(tensor.shape[0]) for tensor in tensors),
                slots=len(tensors),
                size=size,
                mode=mode,
                memory=sum(batch_report.memory_of(tensor) for tensor in tensors),
                refused=str(refused),
            )
            raise
        if all(tensor.device.type == "cpu" for tensor in tensors):
            from ...modules.image import scratch

            batched = scratch.join(
                list(tensors),
                node="Image Batch",
                advice="Connecting fewer or smaller images also fits it.",
                dtype=functools.reduce(torch.promote_types, (t.dtype for t in tensors)),
            )
        else:
            batched = torch.cat(tensors, dim=0)
        batch_report.publish(
            frames=int(batched.shape[0]),
            slots=len(tensors),
            size=size,
            mode=mode,
            memory=batch_report.memory_of(batched),
        )
        return io.NodeOutput(batched, int(batched.shape[0]))

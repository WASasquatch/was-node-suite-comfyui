"""Batch any number of masks into one, on a slot list that grows."""

from __future__ import annotations

import functools

import torch
from comfy_api.latest import io

from ...modules.interface import batch_report
from ...modules.util.slots import connected_in_order
from . import mask_planes

#: Slot ids in the order the sockets are drawn and the masks are joined.
SLOT_NAMES = (
    "masks_a", "masks_b", "masks_c", "masks_d", "masks_e", "masks_f", "masks_g",
    "masks_h", "masks_i", "masks_j", "masks_k", "masks_l", "masks_m", "masks_n",
    "masks_o", "masks_p", "masks_q", "masks_r", "masks_s", "masks_t", "masks_u",
    "masks_v", "masks_w", "masks_x", "masks_y", "masks_z",
)

#: How many slots :data:`SLOT_NAMES` names.
MAX_SLOTS = 26


class MaskBatch(io.ComfyNode):
    """Join masks from a slot list that grows a socket as each one is filled."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Mask Batch",
            display_name="Mask Batch",
            search_aliases=[
                "Mask Batch", "batch masks", "stack masks", "combine masks", "dynamic batch",
            ],
            category="WAS Suite/Image/Masking",
            description=(
                "Join any number of masks into one batch, so a single downstream node processes "
                "all of them in one run. A new empty slot appears below the last one filled, up "
                "to 26. Every mask must be the same size, and a slot holding a batch contributes "
                "all of its frames."
            ),
            inputs=[
                io.Mask.Input(
                    "masks_a",
                    optional=True,
                    tooltip=(
                        "First mask or batch. Every connected slot has to be the same size, "
                        "and at least one slot must be connected."
                    ),
                ),
                io.Mask.Input(
                    "masks_b",
                    optional=True,
                    tooltip="Mask 2, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "masks_c",
                    optional=True,
                    tooltip="Mask 3, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "masks_d",
                    optional=True,
                    tooltip="Mask 4, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "masks_e",
                    optional=True,
                    tooltip="Mask 5, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "masks_f",
                    optional=True,
                    tooltip="Mask 6, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "masks_g",
                    optional=True,
                    tooltip="Mask 7, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "masks_h",
                    optional=True,
                    tooltip="Mask 8, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "masks_i",
                    optional=True,
                    tooltip="Mask 9, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "masks_j",
                    optional=True,
                    tooltip="Mask 10, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "masks_k",
                    optional=True,
                    tooltip="Mask 11, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "masks_l",
                    optional=True,
                    tooltip="Mask 12, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "masks_m",
                    optional=True,
                    tooltip="Mask 13, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "masks_n",
                    optional=True,
                    tooltip="Mask 14, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "masks_o",
                    optional=True,
                    tooltip="Mask 15, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "masks_p",
                    optional=True,
                    tooltip="Mask 16, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "masks_q",
                    optional=True,
                    tooltip="Mask 17, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "masks_r",
                    optional=True,
                    tooltip="Mask 18, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "masks_s",
                    optional=True,
                    tooltip="Mask 19, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "masks_t",
                    optional=True,
                    tooltip="Mask 20, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "masks_u",
                    optional=True,
                    tooltip="Mask 21, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "masks_v",
                    optional=True,
                    tooltip="Mask 22, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "masks_w",
                    optional=True,
                    tooltip="Mask 23, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "masks_x",
                    optional=True,
                    tooltip="Mask 24, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "masks_y",
                    optional=True,
                    tooltip="Mask 25, joined on after the one before it. Unconnected is skipped.",
                ),
                io.Mask.Input(
                    "masks_z",
                    optional=True,
                    tooltip="Mask 26, the last slot. Unconnected is skipped.",
                ),
            ],
            outputs=[
                io.Mask.Output(
                    display_name="masks",
                    tooltip=(
                        "Every connected mask as one batch, in slot order, so a node "
                        "downstream processes all of them in one run."
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
    def execute(cls, **masks) -> io.NodeOutput:
        """Join every connected slot into one batch.

        Raises:
            ValueError: No slot holds a mask, or two masks are different sizes.
            MemoryError: Neither free memory nor any scratch drive has room for the batch.
        """
        names = connected_in_order(masks, SLOT_NAMES)
        # Every slot split into one plane per frame.
        per_slot = [(name, mask_planes(masks[name])) for name in names]
        planes = [plane for _, found in per_slot for plane in found]
        if not planes:
            raise ValueError(
                f"Mask Batch has no masks connected. Connect at least one of the {MAX_SLOTS} "
                f"slots."
            )

        size, mode = batch_report.describe_masks(planes[0])
        try:
            cls.check_mask_dimensions(per_slot)
        except ValueError as refused:
            # The refusal reaches the node's own panel before it is raised.
            batch_report.publish(
                frames=len(planes),
                slots=len(per_slot),
                size=size,
                mode=mode,
                memory=sum(batch_report.memory_of(plane) for plane in planes),
                refused=str(refused),
            )
            raise

        if all(plane.device.type == "cpu" for plane in planes):
            from ...modules.image import scratch

            batched = scratch.stack(
                planes,
                node="Mask Batch",
                advice="Connecting fewer or smaller masks also fits it.",
                dtype=functools.reduce(torch.promote_types, (p.dtype for p in planes)),
            )
        else:
            batched = torch.stack(planes, dim=0)
        batch_report.publish(
            frames=int(batched.shape[0]),
            slots=len(per_slot),
            size=size,
            mode=mode,
            memory=batch_report.memory_of(batched),
        )
        return io.NodeOutput(batched, int(batched.shape[0]))

    @staticmethod
    def check_mask_dimensions(per_slot) -> None:
        """Reject masks that cannot be stacked into one batch.

        Args:
            per_slot: ``(slot name, planes)`` pairs, in slot order.

        Raises:
            ValueError: Two slots carry different frame sizes.
        """
        sizes = {}
        for name, planes in per_slot:
            for plane in planes:
                sizes.setdefault(tuple(plane.shape), []).append(name)
        if len(sizes) > 1:
            listed = ", ".join(
                f"{'/'.join(sorted(set(names)))} is {shape[1]}x{shape[0]}"
                for shape, names in sizes.items()
            )
            raise ValueError(
                f"Mask Batch stacks its masks into one batch, so every connected mask must be "
                f"the same size. These do not match: {listed}. Resize the odd one, or send the "
                f"batches on separately."
            )

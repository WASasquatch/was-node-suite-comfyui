"""Move a mask's edge out or in by whole pixels, keeping its grey levels."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ....modules.compat import limits
from ....modules.interface import mask_report


def _stepped(planes, inward, tapered_corners):
    """One pass of grey dilation or erosion under a 3 by 3 footprint.

    Args:
        planes: ``(batch, 1, height, width)`` float tensor.
        inward: Take the smallest value in each neighbourhood rather than the largest.
        tapered_corners: Leave the four corners of the footprint out.

    Returns:
        A new tensor the same shape, the edge moved by one pixel.
    """
    padded = torch.nn.functional.pad(planes, (1, 1, 1, 1), mode="replicate")
    if inward:
        padded = -padded
    if tapered_corners:
        rows, columns = int(planes.shape[-2]), int(planes.shape[-1])
        middle = padded[..., 1:1 + rows, 1:1 + columns]
        across = torch.maximum(
            padded[..., 1:1 + rows, 0:columns], padded[..., 1:1 + rows, 2:2 + columns]
        )
        down = torch.maximum(
            padded[..., 0:rows, 1:1 + columns], padded[..., 2:2 + rows, 1:1 + columns]
        )
        result = torch.maximum(middle, torch.maximum(across, down))
    else:
        result = torch.nn.functional.max_pool2d(padded, kernel_size=3, stride=1)
    return -result if inward else result


class MaskGrow(io.ComfyNode):
    """Grow or shrink every mask in a batch by a number of pixels."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASMaskGrow",
            display_name="Mask Grow",
            search_aliases=[
                "WASMaskGrow",
                "Mask Grow",
                "GrowMask",
                "expand mask",
                "shrink mask",
                "dilate",
                "erode",
            ],
            category="WAS Suite/Image/Masking",
            description=(
                "Move a mask's edge out or in by a number of pixels, keeping every grey "
                "level it had, so a soft edge stays soft. The node reports the coverage "
                "before and after, so an expand that swallowed the frame or a shrink that "
                "erased the mask shows at once."
            ),
            inputs=[
                io.Mask.Input(
                    "mask",
                    tooltip=(
                        "The mask to grow or shrink. A batch is handled one plane at a time, "
                        "all by the same amount."
                    ),
                ),
                io.Int.Input(
                    "expand",
                    default=0,
                    min=-limits.max_resolution(),
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Pixels the edge moves by; INT. 0 = no change; 8 = grows 8px every way, "
                        "closing gaps up to 16 across; -8 = pulls in 8px, erasing anything "
                        "thinner than 16. A few hundred takes a while."
                    ),
                ),
                io.Boolean.Input(
                    "tapered_corners",
                    default=True,
                    tooltip=(
                        "Whether the four corners of the 3 by 3 step are left out. `true` "
                        "rounds the shape off, `false` squares it and reaches a pixel further "
                        "on the diagonals."
                    ),
                ),
            ],
            outputs=[
                io.Mask.Output(
                    tooltip=(
                        "The mask with its edge moved, at the same size and batch length as "
                        "the one that went in."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, mask, expand=0, tapered_corners=True) -> io.NodeOutput:
        planes = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).unsqueeze(1).clone()
        for _ in range(abs(expand)):
            planes = _stepped(planes, expand < 0, tapered_corners)
        moved = planes.squeeze(1)
        mask_report.publish(mask, moved)
        return io.NodeOutput(moved)

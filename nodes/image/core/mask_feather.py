"""Fade a mask to black over a run of pixels at each of its four edges."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ....modules.compat import limits
from ....modules.interface import mask_report


def _ramp(span, like):
    """The rising fade one edge is multiplied by.

    Args:
        span: How many pixels the fade runs over.
        like: A tensor whose dtype and device the ramp is built for.

    Returns:
        A one dimensional tensor of ``span`` values, ``1/span`` at the outermost pixel
        rising to ``1.0`` at the innermost.
    """
    steps = torch.arange(1, span + 1, dtype=torch.float64) / span
    return steps.to(dtype=like.dtype, device=like.device)


class MaskFeather(io.ComfyNode):
    """Fade each named edge of every mask in a batch to black."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASMaskFeather",
            display_name="Mask Feather",
            search_aliases=[
                "WASMaskFeather",
                "Mask Feather",
                "FeatherMask",
                "soft edge",
                "fade mask",
                "gradient edge",
            ],
            category="WAS Suite/Image/Masking",
            description=(
                "Fade a mask to black over a run of pixels at each of its four edges, one "
                "distance per edge, so a composite or an inpaint shows no hard seam. The node "
                "reports the coverage before and after, so a fade that ate the whole mask "
                "shows at once."
            ),
            inputs=[
                io.Mask.Input(
                    "mask",
                    tooltip=(
                        "The mask to soften. A batch is handled one plane at a time, all by "
                        "the same distances."
                    ),
                ),
                io.Int.Input(
                    "left",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Pixels the fade runs over at the left edge; INT. 0 leaves it hard, 2 "
                        "is a hairline, 64 is a wide falloff. The outermost column is cut to "
                        "a fraction of its value and the innermost keeps all of it."
                    ),
                ),
                io.Int.Input(
                    "top",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Pixels the fade runs over at the top edge; INT. 0 leaves it hard, 2 "
                        "is a hairline, 64 is a wide falloff."
                    ),
                ),
                io.Int.Input(
                    "right",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Pixels the fade runs over at the right edge; INT. 0 leaves it hard, "
                        "2 is a hairline, 64 is a wide falloff."
                    ),
                ),
                io.Int.Input(
                    "bottom",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=1,
                    tooltip=(
                        "Pixels the fade runs over at the bottom edge; INT. 0 leaves it hard, "
                        "2 is a hairline, 64 is a wide falloff."
                    ),
                ),
            ],
            outputs=[
                io.Mask.Output(
                    tooltip=(
                        "The mask with the named edges faded to black, at the same size and "
                        "batch length as the one that went in."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, mask, left=0, top=0, right=0, bottom=0) -> io.NodeOutput:
        faded = mask.reshape((-1, mask.shape[-2], mask.shape[-1])).clone()
        rows, columns = int(faded.shape[-2]), int(faded.shape[-1])
        left, right = min(left, columns), min(right, columns)
        top, bottom = min(top, rows), min(bottom, rows)

        if left > 0:
            faded[:, :, :left] *= _ramp(left, faded).view(1, 1, left)
        if right > 0:
            faded[:, :, columns - right:] *= _ramp(right, faded).flip(0).view(1, 1, right)
        if top > 0:
            faded[:, :top, :] *= _ramp(top, faded).view(1, top, 1)
        if bottom > 0:
            faded[:, rows - bottom:, :] *= _ramp(bottom, faded).flip(0).view(1, bottom, 1)

        mask_report.publish(mask, faded)
        return io.NodeOutput(faded)

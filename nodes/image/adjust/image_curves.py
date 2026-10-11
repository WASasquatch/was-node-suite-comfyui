"""Per-channel tone curves, drawn on the node."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ....modules.image import curves, dynamic
from ....modules.interface import preview


class ImageCurves(io.ComfyNode):
    """Bend an image's tonal response along a curve drawn per channel."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageCurves",
            display_name="Image Curves",
            search_aliases=[
                'WASImageCurves',
                "Image Curves",
                "curves",
                "tone curve",
                "rgb curves",
                "s-curve",
                "contrast",
                "levels",
            ],
            category="WAS Suite/Image/Adjustment",
            description=(
                "Photoshop-style curves for contrast and colour. Drag control points on the "
                "node to bend the tonal response, on the composite curve or one colour channel "
                "at a time, and hand on the adjusted image."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="The image to adjust. Every image in a batch gets the same curve.",
                ),
                io.String.Input(
                    "curve_points",
                    default="",
                    optional=True,
                    socketless=True,
                    tooltip=(
                        "The control points, written by the interface and saved with the "
                        "workflow, as 'rgb:0,0;255,255|r:...|g:...|b:...' on a 0-255 scale. "
                        "A straight line leaves the channel alone, and empty is every "
                        "channel straight. Clear the field to reset every curve."
                    ),
                ),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    optional=True,
                    tooltip=(
                        "How much of the curved result to keep, mixed against the original. "
                        "1.0 is the full curve, 0.5 is halfway, and 0.0 passes the image "
                        "through untouched."
                    ),
                ),
                io.Curve.Input(
                    "curve",
                    optional=True,
                    tooltip=(
                        "A curve drawn somewhere else, from ComfyUI's Curve Editor. It runs "
                        "over all three channels before curve_points does, so the two "
                        "combine. Left unwired, only curve_points is read."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(tooltip="The image with the curves applied."),
            ],
        )

    @classmethod
    def execute(cls, image, curve_points="", strength=1.0, curve=None) -> io.NodeOutput:
        # The editor draws the levels of this picture behind its grid, so the curve is bent
        # against the image it is bending rather than against an empty plot.
        preview.publish(image)

        points = curves.parse(curve_points)
        if strength <= 0.0 or (curve is None and curves.is_identity(points)):
            return io.NodeOutput(image)

        folded = dynamic.fold(image)
        curved = folded.images
        if curve is not None:
            curved = curves.through(curved, cls.levels(curve))
        if not curves.is_identity(points):
            curved = curves.apply(curved, points)
        if strength < 1.0:
            curved = torch.lerp(folded.images, curved, float(strength))
        return io.NodeOutput(dynamic.unfold(curved, folded))

    @staticmethod
    def levels(curve):
        """One lookup table off a CURVE value.

        Args:
            curve: A ``CURVE`` from ComfyUI's Curve Editor, or the control points behind one.

        Returns:
            ``curves.LUT_SIZE`` output levels, one per input level.
        """
        if hasattr(curve, "to_lut"):
            return curve.to_lut(curves.LUT_SIZE)
        from comfy_api.input import CurveInput

        return CurveInput.from_raw(curve).to_lut(curves.LUT_SIZE)

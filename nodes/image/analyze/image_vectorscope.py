"""A Rec.709 vectorscope of an image batch."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.convert.tensors import image_planes, plane_shape
from ....modules.interface import preview, run_result

logger = log.get_logger("nodes.image.analyze")

#: Slot the first frame's scope is published under.
SCOPE_SLOT = "vectorscope"


def _beyond_gamut(cb, cr):
    """Which Cb, Cr pairs lie outside the hexagon the 100% targets span."""
    import torch

    from ....modules.image import scopes

    corners = [(cb_t, cr_t) for _label, level, cb_t, cr_t in scopes.target_points() if level == 1.0]
    outside = torch.zeros_like(cb, dtype=torch.bool)
    for (x0, y0), (x1, y1) in zip(corners, corners[1:] + corners[:1]):
        side = (x1 - x0) * (cr - y0) - (y1 - y0) * (cb - x0)
        centre = (x1 - x0) * (0.0 - y0) - (y1 - y0) * (0.0 - x0)
        outside |= side * centre < -1e-9
    return outside


def _publish_report(frames: int, pixels: int, beyond: int) -> None:
    """Report what was plotted to the node's own interface. Never raises."""
    try:
        if not run_result.watching():
            return
        share = 100.0 * beyond / pixels if pixels else 0.0
        shown = "one frame" if frames == 1 else f"frame 1 of {frames} shown"
        run_result.publish(
            status=run_result.WARNING if beyond else run_result.OK,
            summary=(
                f"Rec.709 Cb and Cr, {shown}, {share:.2f}% beyond the gamut"
                if beyond else f"Rec.709 Cb and Cr, {shown}, all inside the gamut"
            ),
            counts={"frames": frames, "pixels": pixels, "beyond gamut %": round(share, 2)},
        )
    except Exception as error:
        logger.debug("no vectorscope report was published (%s)", error)


class ImageVectorscope(io.ComfyNode):
    """Plot each picture's colour on a Rec.709 vectorscope."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageVectorscope",
            display_name="Image Vectorscope",
            search_aliases=[
                "WASImageVectorscope", "Image Vectorscope", "vectorscope", "scope", "chroma",
                "skin tone line", "color bars", "hue",
            ],
            category="WAS Suite/Image/Analyze",
            description=(
                "Plot a picture's colour on a Rec.709 vectorscope: hue as the angle, saturation "
                "as the distance from the centre, against 75% and 100% bar targets and the skin "
                "tone line. Shows on the node and hands the scope on as a picture."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The pictures to plot, in picture codes. Each frame of a batch gets its "
                        "own scope; the node shows the first."
                    ),
                ),
                io.Int.Input(
                    "size", default=512, min=256, max=2048, step=1,
                    tooltip=(
                        "Side of the scope in pixels. 512 = the usual; 1024 = a finer trace for a "
                        "large picture; 256 = compact."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="scope",
                    tooltip=(
                        "The vectorscope, size by size, one per frame: +Cb to the right, +Cr up, "
                        "trace brightness the log of how many pixels land there."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, image, size=512) -> io.NodeOutput:
        import torch

        from ....modules.image import scopes

        size = int(size)
        drawn, pixels, beyond = [], 0, 0
        for frame in image_planes(image):
            height, width, channels = plane_shape(frame)
            plane = frame.reshape(height, width, channels)
            if channels < 3:
                plane = plane[..., :1].expand(height, width, 3)
            plane = plane[..., :3]
            _luma, cb, cr = scopes.luma_cbcr(plane)
            pixels += height * width
            beyond += int(_beyond_gamut(cb, cr).sum())
            drawn.append(scopes.draw_vectorscope(scopes.vectorscope_density(plane, size)))
        scope = torch.stack(drawn, dim=0)
        preview.publish_output(scope, slot=SCOPE_SLOT)
        _publish_report(len(drawn), pixels, beyond)
        return io.NodeOutput(scope)

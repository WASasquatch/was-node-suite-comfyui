"""Vivid-light sharpening of an image batch."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.convert.tensors import filtered_planes

REQUIRES = "extras"


class WASVividSharpen(io.ComfyNode):
    """Sharpen through an inverted, blurred copy blended back in vivid light."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASVividSharpen",
            display_name="Vivid Sharpen",
            search_aliases=["WASVividSharpen", "sharpen", "high pass", "clarity", "detail"],
            category="WAS Suite/Image/Filter",
            description=(
                "Sharpen images by blending an inverted, blurred copy back over them in "
                "vivid light. Edges gain local contrast and flat areas are left alone, "
                "which reads as detail rather than as the halo an ordinary sharpen leaves. "
                "Good on renders and upscales that came out soft."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The pictures to sharpen. A batch is handled one frame at a time, "
                        "so a whole video's worth of frames can go through at once."
                    ),
                ),
                io.Float.Input(
                    "radius", default=1.5, min=0.01, max=64.0, step=0.01,
                    tooltip=(
                        "Size in pixels of the detail the sharpening picks out. Around 1.0 "
                        "accents fine texture such as skin and fabric; 5.0 and above accents "
                        "broad shapes and starts to look like added contrast rather than "
                        "added detail."
                    ),
                ),
                io.Float.Input(
                    "strength", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip=(
                        "How much of the sharpened version is mixed back over the original. "
                        "0.0 returns the picture untouched, 1.0 uses the sharpened version "
                        "outright, and 0.3 to 0.6 is the usual range for a subtle pass."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="The sharpened pictures, same size and batch order as the input.",
                ),
            ],
        )

    @classmethod
    def execute(cls, images, radius, strength) -> io.NodeOutput:
        from ....modules.image.sharpen import vivid_sharpen

        return io.NodeOutput(filtered_planes(
            images, lambda plane: vivid_sharpen(plane, radius=radius, strength=strength)
        ))

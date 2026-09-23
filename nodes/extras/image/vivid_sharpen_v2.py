"""Vivid-light sharpening with every stage of the blend stack exposed."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.image import dynamic

REQUIRES = "extras"


class WASVividSharpenV2(io.ComfyNode):
    """Sharpen a whole batch at once, with the high-pass layer under manual control."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASVividSharpenV2",
            display_name="Vivid Sharpen (V2)",
            search_aliases=["WASVividSharpenV2", "sharpen", "high pass", "clarity", "detail"],
            category="WAS Suite/Image/Filter",
            description=(
                "Sharpen images by blending an inverted, blurred copy back over them in "
                "vivid light, with each stage of the stack adjustable: two blur radii, a "
                "brightness and contrast trim on the high-pass layer, and separate "
                "opacities for the two blends. Runs on the whole batch at once on the GPU, "
                "so it suits long video sequences."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The pictures to sharpen. The whole batch is processed in one pass, "
                        "so a long sequence costs little more than a single frame."
                    ),
                ),
                io.Float.Input(
                    "radius_highpass", default=5.0, min=0.01, max=64.0, step=0.01,
                    tooltip=(
                        "Size in pixels of the blur that builds the high-pass layer. This "
                        "sets which detail is accented: 1.0 to 2.0 for fine texture, 5.0 and "
                        "above for broad shapes and general punch."
                    ),
                ),
                io.Float.Input(
                    "radius_blur", default=2.5, min=0.01, max=64.0, step=0.01,
                    tooltip=(
                        "Size in pixels of a second blur applied to the high-pass layer. "
                        "Raising it softens the accent and suppresses the halo that appears "
                        "along hard edges; lowering it keeps the result crisp."
                    ),
                ),
                io.Combo.Input(
                    "blur_mode", options=["gaussian", "box"], default="gaussian",
                    tooltip=(
                        "Shape of both blurs. `gaussian` falls off smoothly and is the "
                        "natural-looking choice; `box` weights every pixel in the window "
                        "equally, which is faster and gives a harder, more graphic accent."
                    ),
                ),
                io.Float.Input(
                    "hp_brightness", default=1.0, min=0.5, max=2.0, step=0.01,
                    tooltip=(
                        "Brightness of the high-pass layer before it is blended. Above 1.0 "
                        "pushes the result lighter overall, below 1.0 darker. Use it to "
                        "correct the slight lift or drop sharpening leaves behind; 1.0 "
                        "changes nothing."
                    ),
                ),
                io.Float.Input(
                    "hp_contrast", default=1.0, min=0.5, max=2.0, step=0.01,
                    tooltip=(
                        "Contrast of the high-pass layer before it is blended. Above 1.0 "
                        "makes the accent bite harder, below 1.0 softens it. This is the "
                        "control to reach for when the sharpening is right but too strong."
                    ),
                ),
                io.Float.Input(
                    "vivid_opacity", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip=(
                        "How much of the vivid-light blend is kept. This is the stage that "
                        "creates the edge accent, so 0.0 disables the sharpening and leaves "
                        "only the overlay pass."
                    ),
                ),
                io.Float.Input(
                    "overlay_opacity", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip=(
                        "How much of the overlay pass is kept. Overlay restores the contrast "
                        "the vivid-light stage flattens, so lowering it gives a flatter, "
                        "more filmic result."
                    ),
                ),
                io.Float.Input(
                    "strength", default=1.0, min=0.0, max=3.0, step=0.01,
                    tooltip=(
                        "How much of the finished result is mixed back over the original. "
                        "0.0 returns the picture untouched, 1.0 uses the result outright, "
                        "and values above 1.0 push past it for an exaggerated accent."
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
    def execute(
        cls,
        images,
        radius_highpass,
        radius_blur,
        blur_mode,
        hp_brightness,
        hp_contrast,
        vivid_opacity,
        overlay_opacity,
        strength,
    ) -> io.NodeOutput:
        from ....modules.image.sharpen import sharpen_batch

        folded = dynamic.fold(images)
        images = folded.images

        return io.NodeOutput(dynamic.unfold(
            sharpen_batch(
                images,
                radius_highpass=radius_highpass,
                radius_blur=radius_blur,
                blur_mode=blur_mode,
                hp_brightness=hp_brightness,
                hp_contrast=hp_contrast,
                vivid_opacity=vivid_opacity,
                overlay_opacity=overlay_opacity,
                strength=strength,
            ),
            folded,
        ))

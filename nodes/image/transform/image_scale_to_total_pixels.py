"""Scale a batch of images to a total pixel count, keeping their shape."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.sockets import require_input
from ....modules.image import resolution, scratch, sizing
from ....modules.interface import size_report

NODE_NAME = "Scale Image to Total Pixels (Advanced)"

UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]
CROP_METHODS = ["disabled", "center"]


class ImageScaleToTotalPixelsAdvanced(io.ComfyNode):
    """Scale images to an area in megapixels and report the size they came to."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageScaleToTotalPixels",
            display_name=NODE_NAME,
            search_aliases=[
                "WASImageScaleToTotalPixels",
                "Scale Image to Total Pixels (Advanced)",
                "Scale Image to Total Pixels",
                "ImageScaleToTotalPixels",
                "megapixels",
                "total pixels",
                "resize to megapixels",
            ],
            category="WAS Suite/Image/Transform",
            description=(
                "Scale pictures to a total pixel count with their shape kept and both sides on a "
                "whole step. The panel draws the source frame against the scaled one, and "
                "dragging its corner sets megapixels. The width, height, megapixels and pixel "
                "ratio it came to are handed on to size the rest of a graph."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="The pictures to scale. Every picture in a batch comes out at the same size.",
                ),
                io.Combo.Input(
                    "upscale_method",
                    options=UPSCALE_METHODS,
                    default="lanczos",
                    tooltip=(
                        "`nearest-exact` = blocky, hard pixels kept; `bilinear` = soft; `area` = "
                        "averages, suits shrinking; `bicubic` = sharper; `lanczos` = sharpest, "
                        "rounded to 8 bits a channel."
                    ),
                ),
                io.Float.Input(
                    "megapixels",
                    default=1.0,
                    min=0.01,
                    max=64.0,
                    step=0.01,
                    tooltip=(
                        "The area to scale to, counted as realworld_mp says. With it off, 1.0 is "
                        "1024 x 1024 pixels, and 4.0 turns 1024 x 1024 into 2048 x 2048. Dragging "
                        "a corner of the scaled frame on the panel sets it."
                    ),
                ),
                io.Int.Input(
                    "resolution_steps",
                    default=8,
                    min=1,
                    max=256,
                    tooltip=(
                        "Step both sides land on. 1 = the exact area; 8 = the least a latent "
                        "takes; 64 = what most model families were trained on. Snapping moves a "
                        "side by up to half a step, so the area drifts slightly."
                    ),
                ),
                io.Combo.Input(
                    "scale_mode",
                    options=list(resolution.SCALE_MODES),
                    default="always",
                    tooltip=(
                        "`always` = scales to the area from either side; `downscale_only` = "
                        "megapixels is a ceiling, smaller pictures pass through untouched; "
                        "`upscale_only` = megapixels is a floor, larger pictures pass through "
                        "untouched."
                    ),
                ),
                io.Combo.Input(
                    "crop",
                    options=CROP_METHODS,
                    default="disabled",
                    tooltip=(
                        "`disabled` = the picture is stretched onto the stepped size, a sliver "
                        "off its shape; `center` = the source is trimmed to the stepped shape "
                        "first, so nothing is stretched."
                    ),
                ),
                io.Boolean.Input(
                    "realworld_mp",
                    default=False,
                    tooltip=(
                        "Off: a megapixel is 1024 x 1024, as ComfyUI counts it. On: 1,000,000 "
                        "pixels. Applies to megapixels, the megapixels output and the panel. 4.0 "
                        "on a square picture is 2048 x 2048 off, 2000 x 2000 on."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="image",
                    tooltip=(
                        "The scaled pictures, or the input untouched where scale_mode rules the "
                        "scale out or the size already matches."
                    ),
                ),
                io.Int.Output(
                    display_name="width",
                    tooltip=(
                        "Width of the scaled pictures in pixels. Feed it to anything built at the "
                        "same size, such as an empty latent or a mask."
                    ),
                ),
                io.Int.Output(
                    display_name="height",
                    tooltip="Height of the scaled pictures in pixels.",
                ),
                io.Float.Output(
                    display_name="megapixels",
                    tooltip=(
                        "Area of the scaled pictures in megapixels, counted as realworld_mp says. "
                        "It sits within a step of the megapixels asked for."
                    ),
                ),
                io.Float.Output(
                    display_name="mp_ratio",
                    tooltip=(
                        "Pixels out divided by pixels in. 4.0 = four times the pixels, each side "
                        "about twice as long; 1.0 = unchanged; 0.25 = a quarter."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls, image, upscale_method, megapixels, resolution_steps, scale_mode="always",
        crop="disabled", realworld_mp=False,
    ) -> io.NodeOutput:
        """Scale every picture in the batch to the area asked for.

        Raises:
            ValueError: Nothing is connected to the image input.
            CanvasTooLarge: The scaled size is past 30000 pixels a side.
            MemoryError: Neither free memory nor a scratch drive can hold the result.
        """
        import comfy.model_management
        import comfy.utils

        require_input(image, NODE_NAME, "image", "picture", "Load Image or VAE Decode", "IMAGE")
        frames, height, width = int(image.shape[0]), int(image.shape[1]), int(image.shape[2])
        unit = resolution.megapixel_unit(bool(realworld_mp))
        new_width, new_height, scaled = resolution.scale_to_total(
            width, height, megapixels, resolution_steps, unit, scale_mode,
        )

        if (new_width, new_height) == (width, height):
            result = image
        else:
            sizing.check_canvas(new_width, new_height, f"{NODE_NAME}'s result", "Lower megapixels.")
            result = scratch.allocate(
                (frames, new_height, new_width, int(image.shape[3])), image.dtype,
                node=NODE_NAME, advice="Lower megapixels, or split the batch.",
            )
            progress = comfy.utils.ProgressBar(frames) if frames > 1 else None
            for index in range(frames):
                comfy.model_management.throw_exception_if_processing_interrupted()
                frame = comfy.utils.common_upscale(
                    image[index:index + 1].movedim(-1, 1), new_width, new_height, upscale_method, crop,
                )
                result[index:index + 1].copy_(frame.movedim(1, -1))
                scratch.trim(result[index])
                if progress is not None:
                    progress.update(1)

        size_report.publish(
            (width, height),
            (new_width, new_height),
            action="scaled" if scaled else "left as is",
            facts={"target": f"{float(megapixels):.2f} MP"},
        )
        return io.NodeOutput(
            result,
            new_width,
            new_height,
            new_width * new_height / unit,
            (new_width * new_height) / (width * height),
        )

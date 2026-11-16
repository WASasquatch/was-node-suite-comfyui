"""Taking baked lighting out of a picture at its full size, using an albedo estimate of any size."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ....modules.image import delight


class ImageDelight(io.ComfyNode):
    """Divide a picture's shading out, keeping all of its own detail."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageDelight",
            display_name="Image Delight",
            search_aliases=[
                "WASImageDelight", "Image Delight", "delight", "delighting", "remove lighting",
                "remove shadows", "remove highlights", "albedo", "intrinsic", "flat texture",
                "unlit texture",
            ],
            category="WAS Suite/Image/Adjustment",
            description=(
                "Take the lighting out of full-size pictures by dividing out the shading an albedo estimate "
                "reveals, smoothed so the pictures' own fine detail stays. Highlights and shadows go, turning "
                "enhanced views of a model into flat colour for Mesh Project Views."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="The pictures to delight, at full size, such as 4096 pixel enhanced views. A whole "
                            "batch is processed.",
                ),
                io.Image.Input(
                    "albedo",
                    tooltip="The same pictures' albedo from Power Preprocessor, any size of the same shape, "
                            "such as 1024. One per picture, or one for all.",
                ),
                io.Float.Input(
                    "detail", default=16.0, min=0.0, max=delight.MAX_DETAIL, step=1.0,
                    tooltip="Radius the shading is smoothed over, in pixels of `image`. 0 = the albedo sets "
                            "everything coarser than its own pixels; 16 = suits 4096 views; 64 = keeps more of "
                            "the picture's own texture, and soft fold shading with it.",
                ),
                io.Float.Input(
                    "strength", default=1.0, min=0.0, max=1.0, step=0.05,
                    tooltip="How much of the shading is divided out. 1.0 = all; 0.5 = half; 0.0 = the picture "
                            "unchanged.",
                ),
                io.Combo.Input(
                    "colour", options=list(delight.COLOURS), default="albedo",
                    tooltip="`albedo` = colour comes from the albedo, removing tinted light too; `image` = "
                            "only brightness is divided out, keeping the picture's own colours.",
                ),
                io.Mask.Input(
                    "mask", optional=True,
                    tooltip="Where the subject is: 1 = subject, 0 = background left unchanged, such as Mesh "
                            "Render Views' masks. Shading is smoothed inside it only. Unwired = whole picture.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    "albedo",
                    tooltip="The pictures with their lighting divided out, at the size of `image`, for Mesh "
                            "Project Views or any colour map.",
                ),
                io.Image.Output(
                    "shading",
                    tooltip="The shading divided out, raised to 1/2.2 for showing: white = lit as bright as "
                            "the albedo, darker = shadow. Highlights clip at white.",
                ),
            ],
        )

    @classmethod
    def execute(cls, image, albedo, detail=16.0, strength=1.0, colour="albedo", mask=None) -> io.NodeOutput:
        """Delight every picture of the batch.

        Raises:
            ValueError: A picture is missing, or the albedo or mask count matches neither one nor the
                picture count.
        """
        if getattr(image, "ndim", 0) != 4 or int(image.shape[0]) < 1:
            raise ValueError("Image Delight needs a picture on image; wire the pictures to delight")
        if getattr(albedo, "ndim", 0) != 4 or int(albedo.shape[0]) < 1:
            raise ValueError("Image Delight needs an albedo; wire Power Preprocessor's albedo of the same "
                             "pictures into albedo")
        count = int(image.shape[0])
        if int(albedo.shape[0]) not in (1, count):
            raise ValueError(f"{int(albedo.shape[0])} albedo picture(s) arrived for {count} picture(s); wire one "
                             "albedo per picture, or one for all")
        masks = None
        if mask is not None:
            masks = mask if mask.ndim == 3 else mask.reshape(-1, mask.shape[-2], mask.shape[-1])
            if int(masks.shape[0]) not in (1, count):
                raise ValueError(f"{int(masks.shape[0])} mask(s) arrived for {count} picture(s); wire one mask "
                                 "per picture, or one for all")
        if colour not in delight.COLOURS:
            colour = "albedo"
        delit, shading = [], []
        for index in range(count):
            plain, shade = delight.delight(
                image[index].cpu(), albedo[min(index, int(albedo.shape[0]) - 1)].cpu(), float(detail),
                float(strength), colour,
                None if masks is None else masks[min(index, int(masks.shape[0]) - 1)].cpu())
            delit.append(plain)
            shading.append(shade)
        return io.NodeOutput(torch.stack(delit).to(image.dtype), torch.stack(shading).to(image.dtype))

"""Fake depth of field driven by a depth map."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.convert.tensors import image_planes, stack_images, tensor2pil
from ....modules.image import dynamic


def portrait_blur(img, mask, radius: int, samples: int, mode: str = 'mock'):
    """Composite a blurred copy of an image back over it through a depth mask.

    Args:
        img: Source PIL image.
        mask: Depth map. Resized to the image and read as greyscale.
        radius: Blur strength. Its meaning depends on ``mode``.
        samples: How many times the composite is repeated. Each repeat pulls the grey
            parts of the mask further towards the blur.
        mode: ``'mock'``, ``'gaussian'`` or ``'box'``.

    Returns:
        A PIL image the same size as the source, or ``None`` when ``mode`` is none of the
        three.
    """
    from PIL import Image, ImageFilter

    from ....modules.image.basic import medianFilter

    mask = mask.resize(img.size).convert('L')

    bimg = None
    if mode == 'mock':
        bimg = medianFilter(img, radius, (radius * 1500), 75)
    elif mode == 'gaussian':
        bimg = img.filter(ImageFilter.GaussianBlur(radius=radius))
    elif mode == 'box':
        bimg = img.filter(ImageFilter.BoxBlur(radius))
    else:
        return None

    rimg = img
    for _ in range(samples):
        rimg = Image.composite(rimg, bimg, mask)

    return rimg.convert('RGB')


class ImageFDOFFilter(io.ComfyNode):
    """Blur an image everywhere its depth map is dark, leaving the bright areas sharp."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image fDOF Filter",
            display_name="Image fDOF Filter",
            search_aliases=[
                "Image fDOF Filter",
                "depth of field",
                "bokeh",
                "defocus",
                "portrait blur",
            ],
            category="WAS Suite/Image/Filter",
            description=(
                "Fake a shallow depth of field: keep the image sharp where a depth map is "
                "bright and blur it where the map is dark, so a subject stays crisp against "
                "a soft background."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="The image to defocus. A batch is handled one image at a time.",
                ),
                io.Image.Input(
                    "depth",
                    tooltip=(
                        "The depth map deciding what stays sharp: white = full detail; black = "
                        "full blur; greys blend. Resized to the image. A batch shorter than the "
                        "image batch repeats its last entry."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=["mock", "gaussian", "box"],
                    tooltip=(
                        "How the out-of-focus areas are blurred. `mock` smooths within areas but "
                        "not across their edges, which keeps outlines clean and is the slowest; "
                        "`gaussian` is an ordinary soft blur; `box` is a square blur that is "
                        "faster and slightly harder-edged."
                    ),
                ),
                io.Int.Input(
                    "radius",
                    default=8,
                    min=1,
                    max=128,
                    step=1,
                    tooltip=(
                        "How strong the blur is, in pixels. 8 is a mild defocus, 40 makes the "
                        "background unreadable. In `mock` mode this also widens how far colours "
                        "are allowed to mix, so it gets slow quickly."
                    ),
                ),
                io.Int.Input(
                    "samples",
                    default=1,
                    min=1,
                    max=3,
                    step=1,
                    tooltip=(
                        "How many times the sharp image is composited back over the blurred one. "
                        "1 gives a smooth blend across the grey parts of the depth map; 2 and 3 "
                        "pull those grey parts towards the blur, so only the brightest areas "
                        "stay fully sharp."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(tooltip="The image with the depth-driven blur applied."),
            ],
        )

    @classmethod
    def execute(cls, image, depth, mode, radius, samples) -> io.NodeOutput:
        folded = dynamic.fold(image)
        maps = dynamic.fold(depth).images

        def blurred(index, img):
            return portrait_blur(
                img, tensor2pil(maps[index if index < len(maps) else -1]), radius, samples, mode
            )

        planes = image_planes(folded.images)
        return io.NodeOutput(dynamic.unfold(
            stack_images([blurred(i, tensor2pil(p)) for i, p in enumerate(planes)]), folded
        ))

"""Ambient occlusion cast by tracing rays across a height map."""

from __future__ import annotations

import torch
from comfy_api.latest import io
from torch.nn import functional

from ....modules import log
from ....modules.image import dynamic, occlusion

logger = log.get_logger("nodes.image.filter")

#: Gaussian radius in pixels the specular mask's edge is softened over.
SPECULAR_BLUR = 2.5


def create_ambient_occlusion(rgb_image: torch.Tensor, height_image: torch.Tensor,
                             strength: float = 1.0, radius: float = 30.0,
                             height_scale: float = 64.0, rays: int = 16, steps: int = 16,
                             bias: float = 0.15, ao_blur: float = 2.5,
                             spec_threshold: int = 200,
                             enable_specular_masking: bool = False, levels: int = 0):
    """Shade an image with the ambient occlusion its height map casts.

    Args:
        rgb_image: ``(height, width, channels)`` float tensor, 0 to 1 for picture codes and
            unbounded above for linear light.
        height_image: Height map, bright standing high. Resized to the source when the two
            differ.
        strength: Multiplier on the occlusion before it is taken off the light. 0 leaves
            the image unshaded.
        radius: How far across the surface each ray travels, in pixels.
        height_scale: Pixels the full 0 to 1 height range stands above the base plane.
        rays: Directions traced, spaced evenly around the circle.
        steps: Samples taken along each ray.
        bias: Slope, as a rise over a run, taken off every horizon before it counts.
        ao_blur: Gaussian radius in pixels the shading is softened by.
        spec_threshold: Brightness above which a source pixel counts as specular, on the
            0 to 255 scale. Light above 1.0 reads above 255 and always counts.
        enable_specular_masking: Hold the specular area at full light.
        levels: Steps one unit of the answer is divided into, from
            :data:`~modules.image.occlusion.PRECISIONS`. 0 keeps every value.

    Returns:
        ``(composited, visibility, specular_mask)`` as float32 ``(height, width, 3)``
        tensors. Visibility is bright where the sky reaches and dark in the crevices, and
        the composite keeps whatever range the source arrived with.
    """
    from ....modules.model import compute_device

    device = compute_device()
    rgb = _picture(rgb_image).to(device)
    rows, columns, _ = rgb.shape
    field = _field(height_image, rows, columns).to(device)

    shaded = occlusion.horizon_occlusion(field, radius, rays, steps, height_scale, bias)
    visibility = (1.0 - shaded * strength).clamp(0.0, 1.0)
    visibility = occlusion.blurred_field(visibility, ao_blur).clamp(0.0, 1.0)

    specular = occlusion.blurred_field(
        (_grey(rgb) > spec_threshold / 255.0).to(torch.float32), SPECULAR_BLUR
    ).clamp(0.0, 1.0)
    if enable_specular_masking:
        visibility = visibility + (1.0 - visibility) * specular

    composited = rgb * visibility.unsqueeze(-1)
    return (occlusion.quantised(composited, levels),
            occlusion.quantised(_spread(visibility), levels),
            occlusion.quantised(_spread(specular), levels))


def _grey(rgb: torch.Tensor) -> torch.Tensor:
    """Flatten a float colour picture to one plane.

    Args:
        rgb: ``(height, width, 3)`` float tensor.

    Returns:
        A ``(height, width)`` float tensor on the same scale as the input.
    """
    red, green, blue = occlusion.GREY_WEIGHTS
    return (rgb[:, :, 0] * red + rgb[:, :, 1] * green + rgb[:, :, 2] * blue) / 65536.0


def _spread(plane: torch.Tensor) -> torch.Tensor:
    """One plane repeated across three channels.

    Args:
        plane: ``(height, width)`` float tensor.

    Returns:
        A ``(height, width, 3)`` float tensor.
    """
    return plane.unsqueeze(-1).expand(-1, -1, 3)


def _picture(plane: torch.Tensor) -> torch.Tensor:
    """One image plane as a float32 colour picture inside 0 to 1.

    Args:
        plane: An image tensor holding one frame, with or without a batch axis and with
            any channel count.

    Returns:
        A ``(height, width, 3)`` float32 tensor. A plane with fewer than three channels is
        repeated across them and a fourth channel is dropped.
    """
    if plane.ndim == 4:
        plane = plane[0]
    if plane.ndim == 2:
        plane = plane.unsqueeze(-1)
    plane = plane.to(torch.float32)
    if plane.shape[2] < 3:
        plane = plane[:, :, :1].expand(-1, -1, 3)
    return plane[:, :, :3].contiguous()


def _field(plane: torch.Tensor, rows: int, columns: int) -> torch.Tensor:
    """A height map as one float32 plane on the 0 to 1 scale, sized to the source.

    Args:
        plane: Height map image tensor, with or without a batch axis.
        rows: Row count to match.
        columns: Column count to match.

    Returns:
        A ``(rows, columns)`` float32 tensor.
    """
    picture = _picture(plane)
    if picture.shape[0] != rows or picture.shape[1] != columns:
        picture = functional.interpolate(
            picture.permute(2, 0, 1).unsqueeze(0), size=(rows, columns),
            mode="bilinear", align_corners=False,
        )[0].permute(1, 2, 0)
    red, green, blue = occlusion.GREY_WEIGHTS
    return (picture[:, :, 0] * red + picture[:, :, 1] * green
            + picture[:, :, 2] * blue) / 65536.0


def _output(picture: torch.Tensor) -> torch.Tensor:
    """One frame as the image tensor an output socket carries.

    Args:
        picture: ``(height, width, 3)`` float tensor.

    Returns:
        A float32 tensor with a leading batch axis of one, on the CPU. Nothing is clamped,
        so linear light above 1.0 reaches the socket intact.
    """
    return picture.to(torch.float32).unsqueeze(0).cpu()


class ImageAmbientOcclusion(io.ComfyNode):
    """Darken the crevices of an image using the height map that goes with it."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image SSAO (Ambient Occlusion)",
            display_name="Image SSAO (Ambient Occlusion)",
            search_aliases=[
                "Image SSAO (Ambient Occlusion)",
                "ambient occlusion",
                "ssao",
                "contact shadow",
                "height map shading",
                "bump shading",
            ],
            category="WAS Suite/Image/Filter",
            description=(
                "Shade an image as though its height map were real relief standing off the "
                "page. Rays are traced out from every pixel in a full circle, and a pixel is "
                "darkened by how much of the sky the surrounding ridges block. Sinks and the "
                "insides of corners go dark, open ground stays bright."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The image to shade. A batch is handled one image at a time. Linear "
                        "light is carried through unclipped, so shading a plate on its way to "
                        "EXR Save keeps every highlight above 1.0."
                    ),
                ),
                io.Image.Input(
                    "height_maps",
                    tooltip=(
                        "The matching height map, where bright stands high and dark lies low. "
                        "It is resized to the image, so it need not match its size. MiDaS Depth "
                        "Approximation produces a suitable one."
                    ),
                ),
                io.Float.Input(
                    "strength",
                    min=0.0,
                    max=5.0,
                    default=1.0,
                    step=0.01,
                    tooltip=(
                        "How dark the shading goes. 1.0 is the measured amount, 0.5 is half as "
                        "deep, 2.0 exaggerates it. 0.0 leaves the image exactly as it arrived."
                    ),
                ),
                io.Float.Input(
                    "radius",
                    min=0.01,
                    max=1024,
                    default=30,
                    step=0.01,
                    tooltip=(
                        "How far each ray travels from its pixel, in pixels. 4 catches only the "
                        "tight creases right at an edge; 30 gathers broad soft shading; 200 "
                        "lets a distant ridge shade a whole valley. Cost does not grow with "
                        "this, only with ray_count and step_count."
                    ),
                ),
                io.Float.Input(
                    "height_scale",
                    min=0.0,
                    max=1024,
                    default=64,
                    step=0.1,
                    tooltip=(
                        "How far the map is extruded, in pixels, from black to white. This "
                        "against radius is what sets the depth of the shading: 64 with a radius "
                        "of 30 gives steep relief and heavy occlusion, 16 gives a gentle "
                        "emboss. 0 flattens the map and the shading disappears."
                    ),
                ),
                io.Int.Input(
                    "ray_count",
                    min=4,
                    max=64,
                    default=16,
                    step=1,
                    tooltip=(
                        "How many directions are traced around the circle. 8 is fast and can "
                        "band on smooth gradients, 16 is clean for most images, 32 and above "
                        "for large radii where the banding shows. Cost is directly this times "
                        "step_count."
                    ),
                ),
                io.Int.Input(
                    "step_count",
                    min=1,
                    max=64,
                    default=16,
                    step=1,
                    tooltip=(
                        "How many samples are taken along each ray. Too few for the radius and "
                        "a narrow ridge is stepped straight over, so raise this when a large "
                        "radius starts missing thin occluders. 16 suits a radius up to about "
                        "64; use 32 beyond that."
                    ),
                ),
                io.Float.Input(
                    "angle_bias",
                    min=0.0,
                    max=1.0,
                    default=0.15,
                    step=0.005,
                    tooltip=(
                        "How steep a ridge has to be before it shades at all, as a rise over a "
                        "run. Most height maps arrive with only 256 levels, and every one of "
                        "those steps is a tiny cliff that shades as concentric rings across "
                        "ground that should be flat. 0.15 clears that at the default relief; "
                        "raise it towards 0.3 if rings survive a larger height_scale, and drop "
                        "it to 0 for a height map that came in as smooth floating point."
                    ),
                ),
                io.Float.Input(
                    "ao_blur",
                    min=0.0,
                    max=1024,
                    default=2.5,
                    step=0.01,
                    tooltip=(
                        "How much the shading is softened before it is applied, in pixels. 2.5 "
                        "smooths away the sampling noise; 20 turns the shading into a broad "
                        "gradient. 0 applies it exactly as traced."
                    ),
                ),
                io.Int.Input(
                    "specular_threshold",
                    min=0,
                    max=255,
                    default=200,
                    step=1,
                    tooltip=(
                        "How bright a pixel has to be, on a 0-255 scale, to count as a "
                        "highlight that should not be shaded. 200 protects only genuine "
                        "highlights; 25 protects everything that is not nearly black and leaves "
                        "the shading doing nothing. Only read when enable_specular_masking is "
                        "on, but it always decides the third output."
                    ),
                ),
                io.Boolean.Input(
                    "enable_specular_masking",
                    default=False,
                    tooltip=(
                        "Keep the bright areas picked out by specular_threshold free of "
                        "shading. On protects highlights and light sources from being darkened; "
                        "off shades the whole image from its relief alone."
                    ),
                ),
                io.Combo.Input(
                    "precision",
                    options=list(occlusion.PRECISIONS),
                    default="32 bit float",
                    tooltip=(
                        "How finely the three outputs are stepped, measured on the 0 to 1 "
                        "scale. '32 bit float' keeps every value and is what EXR Save and DNG "
                        "Save want; '16 bit' rounds to steps of 1/65535, still smooth enough "
                        "for a graded plate; '8 bit' rounds to steps of 1/255, which bands a "
                        "soft gradient and only matches what a PNG can hold anyway. Nothing is "
                        "clipped at any setting: linear light above 1.0 keeps its value and "
                        "lands on the same ladder of steps, so a highlight at 4.0 has four "
                        "times as many steps under it as one at 1.0."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="composited_images",
                    tooltip=(
                        "The source image with the shading multiplied into it, on the scale it "
                        "arrived on. Light above 1.0 is dimmed rather than clipped."
                    ),
                ),
                io.Image.Output(
                    display_name="ssao_images",
                    tooltip=(
                        "The shading on its own, as a greyscale image: white where the sky "
                        "reaches, dark in the crevices."
                    ),
                ),
                io.Image.Output(
                    display_name="specular_mask_images",
                    tooltip=(
                        "The area treated as highlight, white where it was protected from "
                        "shading. Produced whether or not the masking was enabled."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, images, height_maps, strength, radius, height_scale, ray_count,
                step_count, angle_bias, ao_blur, specular_threshold,
                enable_specular_masking, precision) -> io.NodeOutput:
        folded = dynamic.fold(images)
        maps = dynamic.fold(height_maps).images
        composited = []
        occlusions = []
        speculars = []
        for i, image in enumerate(folded.images):
            logger.info("Processing SSAO image %d/%d ...", i + 1, len(folded.images))
            composited_image, occlusion_image, specular_mask = create_ambient_occlusion(
                image,
                maps[i if i < len(maps) else -1],
                strength=strength,
                radius=radius,
                height_scale=height_scale,
                rays=ray_count,
                steps=step_count,
                bias=angle_bias,
                ao_blur=ao_blur,
                spec_threshold=specular_threshold,
                enable_specular_masking=enable_specular_masking,
                levels=occlusion.PRECISIONS[precision],
            )
            composited.append(_output(composited_image))
            occlusions.append(_output(occlusion_image))
            speculars.append(_output(specular_mask))

        return io.NodeOutput(
            dynamic.unfold(torch.cat(composited, dim=0), folded),
            torch.cat(occlusions, dim=0),
            torch.cat(speculars, dim=0),
        )

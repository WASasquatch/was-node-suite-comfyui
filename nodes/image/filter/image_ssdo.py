"""Screen-space direct occlusion from an image and its depth map."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ....modules import log
from ....modules.image import dynamic, occlusion
from ....modules.mask.regions import dominant_plane

logger = log.get_logger("nodes.image.filter")


def find_light_source(rgb_codes: torch.Tensor, threshold: int) -> torch.Tensor:
    """Locate the brightest connected region of an image and return it as a mask.

    Args:
        rgb_codes: ``(height, width, 3)`` int64 picture codes.
        threshold: Grey level above which an inverted pixel joins the region, 0-255.

    Returns:
        A ``(height, width)`` int64 tensor, 1 over the winning region and 0 elsewhere.
    """
    return dominant_plane(occlusion.grey_codes(rgb_codes), threshold) // 255


def create_direct_occlusion(rgb_image: torch.Tensor, depth_image: torch.Tensor,
                            strength: float = 1.0, radius: float = 10,
                            threshold: int = 200, colored: bool = True):
    """Light an image from its brightest region using the matching depth map.

    Args:
        rgb_image: ``(height, width, channels)`` float tensor scaled to ``[0, 1]``.
        depth_image: Depth map, the same size as the source. Only its first channel is
            read.
        strength: Upper cutoff percentage for the contrast stretch. 0 leaves the stretch to
            the darkest and brightest pixels; larger values clip more of the bright end to
            white.
        radius: Neighbourhood half-width in pixels the occlusion is measured over.
        threshold: Grey level deciding which region counts as the light source, 0-255.
        colored: Carry the source colours into the lighting pass. ``False`` keeps the pass
            neutral grey.

    Returns:
        ``(composited, occlusion, occlusion_mask, light_mask)`` as int64 picture codes. The
        composite and the occlusion carry a channel axis of 3; the other two have none, and
        the light mask is inverted, so the light source is black on white.
    """
    rgb_codes = _codes(_picture(rgb_image))
    depth_codes = _codes(_picture(depth_image))
    height, width, _ = rgb_codes.shape

    occlusion_codes = occlusion.calculate_direct_occlusion_factor(
        _normalized(rgb_codes), _normalized(depth_codes), height, width, radius
    ).to(torch.int64)
    rgb_codes = rgb_codes.to(occlusion_codes.device)
    light_mask = find_light_source(rgb_codes, threshold)

    occlusion_codes = occlusion.blurred_codes(occlusion_codes, 0.5)
    occlusion_codes = occlusion.smoothed_codes(occlusion_codes)

    if colored:
        lit = occlusion.darkened_codes(rgb_codes, occlusion_codes)
    else:
        lit = (255 - occlusion_codes).unsqueeze(-1).expand(-1, -1, 3).contiguous()
    lit = occlusion.stretched_codes(lit, (0, strength))

    return (
        occlusion.screened_codes(rgb_codes, lit),
        lit,
        occlusion_codes,
        255 - light_mask * 255,
    )


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


def _codes(picture: torch.Tensor) -> torch.Tensor:
    """A float picture as int64 picture codes.

    Args:
        picture: Float tensor scaled to ``[0, 1]``.

    Returns:
        An int64 tensor of the same shape holding 0 to 255.
    """
    return (picture * 255.0).clamp(0, 255).to(torch.int64)


def _normalized(codes: torch.Tensor) -> torch.Tensor:
    """Picture codes back on the 0 to 1 scale the occlusion kernel measures.

    Args:
        codes: Int64 picture codes.

    Returns:
        A float32 tensor of the same shape.
    """
    return codes.to(torch.float32) / 255.0


def _output(codes: torch.Tensor) -> torch.Tensor:
    """Picture codes as the image tensor an output socket carries.

    Args:
        codes: Int64 picture codes for one frame.

    Returns:
        A float32 tensor scaled to ``[0, 1]`` with a leading batch axis of one.
    """
    return (codes.to(torch.float32) / 255.0).unsqueeze(0).cpu()


class ImageDirectOcclusion(io.ComfyNode):
    """Relight an image from its own brightest area, casting shadows out of its depth map."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image SSDO (Direct Occlusion)",
            display_name="Image SSDO (Direct Occlusion)",
            search_aliases=[
                "Image SSDO (Direct Occlusion)",
                "direct occlusion",
                "ssdo",
                "relight",
                "cast shadow",
            ],
            category="WAS Suite/Image/Filter",
            description=(
                "Relight an image from the brightest thing in it, using a depth map to work "
                "out what stands in front of what. Brightens the lit side of every edge and "
                "leaves the shadowed side alone."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The image to relight. A batch is handled one image at a time.",
                ),
                io.Image.Input(
                    "depth_images",
                    tooltip=(
                        "The matching depth map, where bright means near and dark means far. "
                        "Unlike the ambient-occlusion node this one does not resize it, so it "
                        "has to be the same size as the image."
                    ),
                ),
                io.Float.Input(
                    "strength",
                    min=0.0,
                    max=5.0,
                    default=1.0,
                    step=0.01,
                    tooltip=(
                        "How much of the bright end of the lighting is pushed to full white, as "
                        "a percentage of the pixels. 0.0 keeps the whole range, 1.0 clips the "
                        "brightest one percent, 5.0 clips the brightest five and gives a harder "
                        "light."
                    ),
                ),
                io.Float.Input(
                    "radius",
                    min=0.01,
                    max=1024,
                    default=30,
                    step=0.01,
                    tooltip=(
                        "How far around each pixel the depth is compared, in pixels. Small "
                        "values such as 4 give a tight rim of light along edges; 30 spreads it "
                        "broadly. Cost grows with the square of this, so large values are very "
                        "slow."
                    ),
                ),
                io.Int.Input(
                    "specular_threshold",
                    min=0,
                    max=255,
                    default=128,
                    step=1,
                    tooltip=(
                        "How dark a pixel has to be to join the area the light source is looked "
                        "for in. It is read against the inverted image, so 128 takes everything "
                        "darker than mid grey and a higher value such as 200 narrows the search "
                        "to the darkest pixels only. It decides the fourth output alone, not the "
                        "lighting."
                    ),
                ),
                io.Boolean.Input(
                    "colored_occlusion",
                    default=True,
                    tooltip=(
                        "On carries the source colours into the lighting pass, tinting the "
                        "relight with the scene; off keeps the pass neutral grey, so the "
                        "shadows shade without shifting hue."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="composited_images",
                    tooltip="The source image with the lighting screened over it.",
                ),
                io.Image.Output(
                    display_name="ssdo_images",
                    tooltip=(
                        "The lighting on its own: the source colours where the light reaches and "
                        "black where it does not, after the contrast stretch."
                    ),
                ),
                io.Image.Output(
                    display_name="ssdo_image_masks",
                    tooltip=(
                        "The raw occlusion field as a greyscale image, before any colour is "
                        "applied: white where something stands in front."
                    ),
                ),
                io.Image.Output(
                    display_name="light_source_image_masks",
                    tooltip=(
                        "Where the light source was found, drawn black on a white background."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, images, depth_images, strength, radius, specular_threshold,
                colored_occlusion) -> io.NodeOutput:
        from ....modules.image import scratch

        folded = dynamic.fold(images)
        maps = dynamic.fold(depth_images).images
        count = len(folded.images)
        batches = None
        for i, image in enumerate(folded.images):
            logger.info("Processing SSDO image %d/%d ...", i + 1, count)
            composited_image, occlusion_image, occlusion_mask, light_source = (
                create_direct_occlusion(
                    image,
                    maps[i if i < len(maps) else -1],
                    strength=strength,
                    radius=radius,
                    threshold=specular_threshold,
                    colored=colored_occlusion,
                )
            )
            frames = (
                _output(composited_image),
                _output(occlusion_image),
                _output(occlusion_mask),
                _output(light_source),
            )
            if batches is None:
                batches = tuple(
                    scratch.allocate(
                        (count,) + tuple(frame.shape[1:]),
                        node="Image SSDO (Direct Occlusion)",
                        advice="Passing fewer or smaller frames also fits it.",
                    )
                    for frame in frames
                )
            batches[0][i] = dynamic.unfold(frames[0][0], folded)
            for batch, frame in zip(batches[1:], frames[1:]):
                batch[i] = frame[0]

        if batches is None:
            raise ValueError(
                "Image SSDO (Direct Occlusion) was handed an empty image batch. Wire in at "
                "least one image."
            )
        return io.NodeOutput(*batches)

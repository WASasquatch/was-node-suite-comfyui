"""Show a batch as it really is, whatever its numbers mean, and mark what will not fit."""

from __future__ import annotations

import torch
from comfy_api.latest import io, ui

from ....modules import log
from ....modules.compat.types import WAS_COLOUR_PROFILE
from ....modules.image import colour_profile, raw
from ....modules.interface import image_report

logger = log.get_logger("nodes.image.hdr")

#: Widget option -> what the numbers on the wire mean.
ENCODINGS = ("sRGB", "linear light")

#: What one stop of exposure multiplies the light by, and the widest adjustment offered.
STOP = 2.0
STOPS = 16.0

#: Colours painted over a sample that will not fit, over white and under black.
OVER = (1.0, 0.0, 0.0)
UNDER = (0.0, 0.4, 1.0)

#: Level a sample counts as clipped above, and below.
CEILING = 1.0
FLOOR = 0.0


def shown(images: torch.Tensor, encoding: str, exposure: float) -> torch.Tensor:
    """The batch as a display shows it.

    Args:
        images: ``(batch, height, width, channels)``.
        encoding: One of :data:`ENCODINGS`.
        exposure: Stops applied to the light.

    Returns:
        A ``(batch, height, width, 3)`` tensor on a 0 to 1 scale, sRGB encoded. Light that
        does not fit is held at white, which is what a display does with it.
    """
    colour = images[..., :3].float()
    linear = raw.linearise(colour) if encoding == ENCODINGS[0] else colour
    return raw.encode(linear * (STOP ** float(exposure))).clamp(0.0, 1.0)


def marked(picture: torch.Tensor, light: torch.Tensor) -> torch.Tensor:
    """The picture with the samples that do not fit painted over.

    Args:
        picture: ``(batch, height, width, 3)`` on a 0 to 1 scale.
        light: The light the picture was made from, on the same axes.

    Returns:
        A tensor of the same shape as ``picture``.
    """
    over = (light > CEILING).any(dim=-1, keepdim=True)
    under = (light < FLOOR).any(dim=-1, keepdim=True)
    painted = torch.where(over, picture.new_tensor(OVER), picture)
    return torch.where(under, picture.new_tensor(UNDER), painted)


def interpreted(picture: torch.Tensor, profile) -> torch.Tensor:
    """The picture read as colour in one profile's space rather than in sRGB.

    Args:
        picture: ``(batch, height, width, 3)`` on a 0 to 1 scale.
        profile: A :class:`~modules.image.colour_profile.Carried`.

    Returns:
        A tensor of the same shape, sRGB encoded. The picture comes back unchanged where the
        profile cannot be applied.
    """
    from ....modules.convert.tensors import pil2tensor, tensor2pil

    frames = []
    for frame in picture:
        opened = tensor2pil(frame.unsqueeze(0))
        opened.info["icc_profile"] = profile.data
        frames.append(pil2tensor(colour_profile.to_srgb(opened, profile.name)))
    return torch.cat(frames, dim=0)[..., :3]


def _publish_report(images, picture, encoding, exposure, above, profile) -> None:
    """Report what the batch holds and how far the view is from it.

    Never raises, and never changes what the node returns.

    Args:
        images: The batch as it arrived.
        picture: The batch as it is drawn.
        encoding: What the numbers were read as.
        exposure: Stops applied to the view.
        above: Share of the frame over white, from 0 to 100.
        profile: The colour profile that was applied, or None.
    """
    try:
        moved = image_report.drift(images[0], picture[0])
        facts = {
            "read as": encoding,
            "exposure": f"{exposure:+.2f} stop(s)" if exposure else "as it is",
            "over white": f"{above:.2f}% of the frame",
            "frames": str(int(images.shape[0])),
        }
        if profile is not None:
            facts["profile"] = profile.name
        image_report.publish(
            images,
            facts=facts,
            moved=moved or None,
            summary=(
                f"drawn {moved['worst']:.1f} codes from the numbers"
                if moved and moved.get("moved")
                else "drawn exactly as the numbers stand"
            ),
        )
    except Exception as error:
        logger.debug("no preview report was published (%s)", error)


class ImagePreview(io.ComfyNode):
    """Show a batch as a display would, reading its numbers the way they were meant."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImagePreview",
            display_name="Image Preview",
            search_aliases=[
                "WASImagePreview", "Image Preview", "accurate preview", "linear preview",
                "hdr preview", "exposure preview", "clipping",
            ],
            category="WAS Suite/Image/HDR",
            description=(
                "Show a batch the way it will really look, and hand it on unchanged. Set "
                "`linear light` for frames from HDR Reconstruct or EXR Load, dial exposure to "
                "bring a highlight into view, and mark every sample that will not fit."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The images to look at. They are passed on untouched, so this can "
                        "sit anywhere in a chain."
                    ),
                ),
                io.Combo.Input(
                    "encoding",
                    options=list(ENCODINGS),
                    tooltip=(
                        "'sRGB' = ordinary picture codes, what a sampler and a PNG carry; "
                        "'linear light' = what HDR Reconstruct, EXR Load and Linear Light "
                        "answer with. Reading light as sRGB is what makes a preview look "
                        "too dark and too contrasty."
                    ),
                ),
                io.Float.Input(
                    "exposure",
                    default=0.0,
                    min=-STOPS,
                    max=STOPS,
                    step=0.1,
                    tooltip=(
                        "Stops applied before the picture is drawn. 0.0 shows it as it is, "
                        "-2.0 brings a highlight that reached 4.0 into view, +1.0 opens up "
                        "a shadow. The images handed on are not changed by it."
                    ),
                ),
                WAS_COLOUR_PROFILE.Input(
                    "profile",
                    optional=True,
                    tooltip=(
                        "A colour profile from Image Load. Changes the picture only where the "
                        "loader kept the file's own space, so the view matches the file; a "
                        "profile from a converted load is already sRGB and draws the same."
                    ),
                ),
                io.Boolean.Input(
                    "mark_clipping",
                    default=False,
                    tooltip=(
                        "'true' paints every sample over white red and every sample under "
                        "black blue, so the exposure can be dialled until nothing is "
                        "marked; 'false' draws the picture alone."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="The images exactly as they arrived, so this can sit in a chain.",
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls, images, encoding="sRGB", exposure=0.0, profile=None, mark_clipping=False
    ) -> io.NodeOutput:
        """Draw the batch and hand it on.

        Raises:
            ValueError: ``encoding`` names nothing known.
        """
        if encoding not in ENCODINGS:
            raise ValueError(
                f"Image Preview encoding must be one of {', '.join(ENCODINGS)}, "
                f"not {encoding!r}"
            )

        colour = images[..., :3].float()
        light = (raw.linearise(colour) if encoding == ENCODINGS[0] else colour) * (
            STOP ** float(exposure)
        )
        picture = shown(images, encoding, exposure)
        if profile is not None and not profile.converted:
            picture = interpreted(picture, profile)

        above = float((light.amax(dim=-1) > CEILING).to(torch.float32).mean()) * 100.0
        # Measured before the marks go on, so the figure is the distance between the view
        # and the numbers rather than the area the marks cover.
        _publish_report(images, picture, encoding, float(exposure), above, profile)
        if mark_clipping:
            picture = marked(picture, light)
        logger.info(
            "previewing %d frame(s) as %s at %+.2f stop(s), %.2f%% of the frame over white",
            int(images.shape[0]), encoding, float(exposure), above,
        )
        return io.NodeOutput(images, ui=ui.PreviewImage(picture, cls=cls))

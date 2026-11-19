"""ARRI exposure false colour for an image batch."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log
from ....modules.convert.tensors import image_planes, plane_shape
from ....modules.interface import preview, run_result

logger = log.get_logger("nodes.image.analyze")

#: Slot the first frame's false colour is published under.
FALSE_COLOR_SLOT = "false_color"

#: Name the share of pixels in no band is reported under.
REST = "rest, in grey"


def _hex(colour) -> str:
    """A band colour as ``#rrggbb``."""
    return "#" + "".join(f"{round(channel * 255):02x}" for channel in colour)


def _publish_report(counts, total: int, frames: int) -> None:
    """Report each band's share of the batch to the node's own interface. Never raises.

    Args:
        counts: Pixels per band, in band order, then the grey rest.
        total: Pixels in the batch.
        frames: Frames in the batch.
    """
    try:
        if not run_result.watching():
            return
        from ....modules.image import scopes

        bands = scopes.FALSE_COLOUR_BANDS
        shares = [100.0 * count / total if total else 0.0 for count in counts]
        tallies = [
            {"name": name, "value": round(share, 2)}
            for (name, _low, _high, _colour), share in zip(bands, shares)
        ]
        tallies.append({"name": REST, "value": round(shares[-1], 2)})
        clipped = shares[0] + shares[len(bands) - 1]
        where = "" if frames == 1 else f" over {frames} frames"
        run_result.publish(
            status=run_result.OK,
            summary=(
                f"{shares[len(bands) - 1]:.2f}% clipped white, {shares[0]:.2f}% clipped "
                f"black{where}"
                if clipped else f"nothing clipped{where}"
            ),
            tallies=tallies,
            facts={
                "colours": " ".join(_hex(colour) for _name, _low, _high, colour in bands),
                "levels": " ".join(f"{low:g}-{high:g}" for _name, low, high, _colour in bands),
            },
        )
    except Exception as error:
        logger.debug("no false colour report was published (%s)", error)


class ImageFalseColor(io.ComfyNode):
    """Paint each picture in ARRI exposure false colour."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASImageFalseColor",
            display_name="Image False Color",
            search_aliases=[
                "WASImageFalseColor", "Image False Color", "false colour", "false color",
                "exposure", "arri", "clipping", "zebra",
            ],
            category="WAS Suite/Image/Analyze",
            description=(
                "Paint a picture in ARRI exposure false colour: purple and red where it clips, "
                "blue and yellow just inside, green and pink at ARRI's levels for 18% grey and "
                "one stop over, grey elsewhere. Shows on the node with a legend and each band's "
                "share of the frame."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "The pictures to check, in picture codes. Bands read Rec.709 luma: "
                        "0 = 0%, 1 = 100%, so 0.40 is green and 0.995 is clipped."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip=(
                        "The false colour pictures, the input's size. Purple under 2.5%, blue "
                        "2.5 to 4, green 38 to 42, pink 52 to 56, yellow 97 to 99, red 99 and up."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, image) -> io.NodeOutput:
        import torch

        from ....modules.image import scopes

        bands = len(scopes.FALSE_COLOUR_BANDS)
        counts = torch.zeros(bands + 1, dtype=torch.int64)
        painted, total = [], 0
        for frame in image_planes(image):
            height, width, channels = plane_shape(frame)
            plane = frame.reshape(height, width, channels)
            if channels < 3:
                plane = plane[..., :1].expand(height, width, 3)
            picture, band = scopes.false_colour(plane[..., :3])
            painted.append(picture)
            total += height * width
            found = torch.bincount((band.flatten() + 1).cpu(), minlength=bands + 1)
            counts[:bands] += found[1:]
            counts[bands] += found[0]
        result = torch.stack(painted, dim=0)
        preview.publish_output(result, slot=FALSE_COLOR_SLOT)
        _publish_report(counts.tolist(), total, len(painted))
        return io.NodeOutput(result)

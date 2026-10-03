"""Write a batch of linear light out as OpenEXR files."""

from __future__ import annotations

import os

import torch
from comfy_api.latest import io

from ....modules import log
from ....modules.image import exr
from ....modules.interface import file_report
from ....modules.io import naming
from ....modules.util import sandbox

logger = log.get_logger("nodes.image.hdr")

#: Stand-in name used to resolve the output directory when the prefix widget is empty.
PLACEHOLDER_PREFIX = "_"

#: What separates the name from the number, and the digits the number is padded to.
DELIMITER = "_"
PADDING = 4

#: Extension every file this node writes carries.
EXTENSION = "exr"


def _with_alpha(image: torch.Tensor, alpha, index: int) -> torch.Tensor:
    """One frame carrying the coverage that belongs to it.

    Args:
        image: ``(height, width, channels)`` frame of linear light.
        alpha: ``(frames, height, width)`` coverage, ``(height, width)``, or None.
        index: Which frame of the batch this is. A shorter run of masks holds its last.

    Returns:
        The frame with four channels where coverage was given, and unchanged where it was
        not.

    Raises:
        ValueError: The coverage is a different size from the frame.
    """
    if alpha is None:
        return image
    masks = alpha
    while masks.ndim > 3 and masks.shape[-1] == 1:
        masks = masks.squeeze(-1)
    if masks.ndim == 2:
        masks = masks.unsqueeze(0)
    if masks.ndim != 3 or masks.shape[0] == 0:
        return image
    coverage = masks[min(index, int(masks.shape[0]) - 1)]
    if tuple(coverage.shape[-2:]) != tuple(image.shape[:2]):
        raise ValueError(
            f"alpha is {int(coverage.shape[-1])}x{int(coverage.shape[-2])} and the image is "
            f"{int(image.shape[1])}x{int(image.shape[0])}. Resize the mask to the image, or "
            f"leave alpha unconnected"
        )
    return torch.cat([image[..., :3], coverage.reshape(*image.shape[:2], 1)], dim=-1)


class EXRSave(io.ComfyNode):
    """Write every image in the batch to a permitted output directory as an OpenEXR file."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASEXRSave",
            display_name="EXR Save",
            search_aliases=[
                "WASEXRSave", "EXR Save", "OpenEXR", "save exr", "hdr save", "linear save"
            ],
            category="WAS Suite/Image/HDR",
            description=(
                "Write every image in the batch as an OpenEXR file of linear light, one "
                "file per frame, numbered in sequence, at 16 or 32 bits per channel. "
                "Values above one are kept, so recovered highlights reach a grading or "
                "compositing program at their real brightness. A mask on alpha is written "
                "as a fourth channel. Files land under ComfyUI's output directory, and the "
                "prefix may name a subfolder."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The images to write, read as linear light. Every image in the "
                        "batch gets its own file, each with the next number in the "
                        "sequence."
                    ),
                ),
                io.String.Input(
                    "filename_prefix",
                    default="ComfyUI_hdr",
                    multiline=False,
                    tooltip=(
                        "Name and folder under the output directory, before the number. "
                        "`ComfyUI_hdr` gives `ComfyUI_hdr_0001.exr`; `plates/hdr` puts it "
                        "in that subfolder. Tokens expand, so `[time(%Y-%m-%d)]/shot` "
                        "dates the folder."
                    ),
                ),
                io.Combo.Input(
                    "depth",
                    options=list(exr.DEPTHS),
                    tooltip=(
                        "'16 bit half' = half the file size, values to 65504 and about "
                        "three decimal digits; '32 bit float' = the exact values on the "
                        "wire. Half for delivery, float for a plate that gets graded "
                        "further."
                    ),
                ),
                io.Combo.Input(
                    "compression",
                    options=list(exr.PACKINGS),
                    default="zip",
                    tooltip=(
                        "'zip' = smaller files, opened by every compositing program; "
                        "'none' = the pixels stored as they are, which every reader takes "
                        "and which stays the frame size on disk. Both keep the exact "
                        "values."
                    ),
                ),
                io.Mask.Input(
                    "alpha",
                    optional=True,
                    tooltip=(
                        "Coverage to write as a fourth channel, white opaque and black "
                        "clear. Left unconnected, a four channel image writes its own alpha "
                        "and a three channel one writes colour only. One mask covers a whole "
                        "batch; a batch of masks is matched frame by frame."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="files",
                    tooltip=(
                        "Full path of every file written this run, one per line, in batch "
                        "order. A file that could not be written is left out of the list."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        images,
        filename_prefix="ComfyUI_hdr",
        depth="16 bit half",
        compression="zip",
        alpha=None,
    ) -> io.NodeOutput:
        """Write one EXR per image and report what landed in the output directory.

        Raises:
            PathNotAllowed: The prefix resolved outside the output directory.
            ValueError: ``depth`` or ``compression`` names nothing known, an image is not
                three channel, or the alpha is a different size from the image.
        """
        import folder_paths

        # get_save_image_path splits its first argument into a directory and a name, so a
        # cleared prefix widget is resolved with a stand-in that is dropped again below.
        named = filename_prefix or PLACEHOLDER_PREFIX
        full_output_folder, resolved, _, _, _ = sandbox.save_image_path(
            named, folder_paths.get_output_directory(), images[0].shape[1], images[0].shape[0]
        )
        destination = sandbox.resolve_write(full_output_folder)
        os.makedirs(destination, exist_ok=True)
        prefix = resolved if filename_prefix else ""

        names = naming.next_names(
            str(destination), prefix, DELIMITER, PADDING, EXTENSION, len(images)
        )

        written = []
        written_alpha = False
        for index, (image, name) in enumerate(zip(images, names)):
            target = str(sandbox.resolve_write_file(destination, name))
            frame = _with_alpha(image, alpha, index)
            written_alpha = int(frame.shape[-1]) >= 4
            try:
                exr.write(target, frame, depth, compression)
            except OSError as error:
                logger.error("unable to save file to: %s\n%s", target, error)
                continue
            logger.info("EXR file saved to: %s", target)
            written.append(target)

        file_report.publish(
            written,
            intended=len(images),
            kind=EXTENSION,
            folder=str(destination),
            facts={
                "depth": depth,
                "compression": compression,
                "channels": "RGBA" if written_alpha else "RGB",
            },
        )
        return io.NodeOutput("\n".join(written))

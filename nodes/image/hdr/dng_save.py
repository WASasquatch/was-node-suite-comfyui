"""Write a batch of linear light out as DNG files a raw developer opens."""

from __future__ import annotations

import os

import torch
from comfy_api.latest import io

from ....modules import log
from ....modules.image import dng, raw
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
EXTENSION = "dng"

#: Widget option -> samples each pixel is written with.
LAYOUTS = {"demosaiced": 3, "colour filter array": 1}

#: Levels a 16-bit reading spans.
FULL_SCALE = 65535


def sensor_plane(image: torch.Tensor, profile, white_point: float, samples: int):
    """One frame as the 16-bit readings a DNG holds.

    Args:
        image: ``(height, width, channels)`` of linear light, unbounded above.
        profile: A :class:`~modules.image.raw.Profile`.
        white_point: Linear value written as the top of the 16-bit range.
        samples: 3 to keep every colour at every pixel, 1 to keep the sensor's own tile.

    Returns:
        ``(height, width, body)`` for :func:`modules.image.dng.write`.
    """
    frame = image[..., :3].unsqueeze(0).float()
    camera = raw.apply_matrix(frame, raw.camera_from_srgb(profile, frame.device, frame.dtype))
    planes = raw.invert_gains(camera, profile)
    if samples == 1:
        planes = raw.mosaic(planes, profile.cfa).unsqueeze(-1)

    scaled = (planes / max(white_point, 1e-6)).clamp(0.0, 1.0) * FULL_SCALE
    readings = scaled.round().clamp(0, FULL_SCALE).to(torch.int32).cpu().numpy()
    height, width = int(readings.shape[1]), int(readings.shape[2])
    return height, width, readings.astype("<u2").tobytes()


class DNGSave(io.ComfyNode):
    """Write every image in the batch as a DNG holding 16-bit linear readings."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASDNGSave",
            display_name="DNG Save",
            search_aliases=[
                "WASDNGSave", "DNG Save", "save dng", "raw save", "save raw", "lightroom",
                "camera raw", "digital negative",
            ],
            category="WAS Suite/Image/HDR",
            description=(
                "Write every image in the batch as a DNG, the raw file Lightroom, Camera "
                "Raw, darktable and RawTherapee open with their raw controls. Readings are "
                "16 bit and linear, so a recovered highlight arrives with room to pull back "
                "and a rebuilt gradient arrives without its 8-bit steps. The images are read "
                "as linear light, which is what HDR Reconstruct, EXR Load and Linear Light "
                "answer with."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The images to write, read as linear light. Every image in the "
                        "batch gets its own file, each with the next number in the "
                        "sequence. Put Linear Light in front of a picture that came "
                        "straight out of a PNG or a JPEG."
                    ),
                ),
                io.String.Input(
                    "filename_prefix",
                    default="ComfyUI_raw",
                    multiline=False,
                    tooltip=(
                        "Name and folder under the output directory, before the number. "
                        "`ComfyUI_raw` gives `ComfyUI_raw_0001.dng`; `plates/shot` puts it "
                        "in that subfolder. Tokens expand, so `[time(%Y-%m-%d)]/shot` "
                        "dates the folder."
                    ),
                ),
                io.Combo.Input(
                    "profile",
                    options=list(raw.PROFILES),
                    tooltip=(
                        "'sRGB primaries' = the colours come out of a developer exactly as "
                        "they went in; 'generic camera' = a camera's own primaries and "
                        "white balance, which a developer corrects for and which looks "
                        "closer to a photograph out of the box."
                    ),
                ),
                io.Combo.Input(
                    "layout",
                    options=list(LAYOUTS),
                    tooltip=(
                        "'demosaiced' = red, green and blue kept at every pixel, so nothing "
                        "is thrown away; 'colour filter array' = one colour per pixel in a "
                        "Bayer tile, which is what a sensor holds and which a developer "
                        "demosaics itself."
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
                io.Float.Output(
                    display_name="white_point",
                    tooltip=(
                        "The linear value written as the top of the 16-bit range. 1.0 = the "
                        "images fitted as they were; 2.99 = a recovered highlight reached "
                        "that far and the whole range was fitted under it."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        images,
        filename_prefix="ComfyUI_raw",
        profile="sRGB primaries",
        layout="demosaiced",
    ) -> io.NodeOutput:
        """Write one DNG per image and report what landed in the output directory.

        Raises:
            PathNotAllowed: The prefix resolved outside the output directory.
            ValueError: ``profile`` or ``layout`` names nothing known.
        """
        import folder_paths

        sensor = raw.PROFILES.get(profile)
        if sensor is None:
            raise ValueError(
                f"DNG profile must be one of {', '.join(raw.PROFILES)}, not {profile!r}"
            )
        samples = LAYOUTS.get(layout)
        if samples is None:
            raise ValueError(
                f"DNG layout must be one of {', '.join(LAYOUTS)}, not {layout!r}"
            )

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

        # One white point for the whole batch, so a sequence keeps its brightness from frame
        # to frame.
        white_point = max(1.0, float(images[..., :3].amax()))

        written = []
        for image, name in zip(images, names):
            target = str(sandbox.resolve_write_file(destination, name))
            plane = sensor_plane(image, sensor, white_point, samples)
            try:
                dng.write(target, plane, sensor, sensor.name, samples)
            except OSError as error:
                logger.error("unable to save file to: %s\n%s", target, error)
                continue
            logger.info("DNG file saved to: %s", target)
            written.append(target)

        file_report.publish(
            written,
            intended=len(images),
            kind=EXTENSION,
            folder=str(destination),
            facts={
                "profile": sensor.name,
                "layout": layout,
                "white point": f"{white_point:.4g}",
            },
        )
        return io.NodeOutput("\n".join(written), white_point)

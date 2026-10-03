"""Read an OpenEXR file back as linear light, with its coverage beside it."""

from __future__ import annotations

import os

import torch
from comfy_api.latest import io

from ....modules import log
from ....modules.image import exr
from ....modules.interface import run_result
from ....modules.io import picker
from ....modules.util import sandbox

logger = log.get_logger("nodes.image.hdr")

#: Extensions the file menu offers.
EXTENSIONS = (".exr",)

#: Level a sample is counted as carrying headroom above.
WHITE = 1.0


def exr_labels() -> list[str]:
    """Every EXR under ComfyUI's own directories, as the labels a widget stores.

    Returns:
        ``<relative path> [input]``, ``[output]`` or ``[temp]`` per file. Empty outside
        ComfyUI and where no root can be read.
    """
    try:
        return picker.labels(EXTENSIONS)
    except Exception as error:
        logger.debug("the file listing could not be read: %s", error)
        return []


def target(file: str) -> str:
    """The file the menu names, resolved inside a permitted read root.

    Args:
        file: A label the menu offered, such as ``plate_0001.exr [output]``.

    Returns:
        The absolute path to read.

    Raises:
        PathNotAllowed: The file resolved outside every permitted read root.
        ValueError: Nothing was chosen, or the entry names no file that is there.
    """
    chosen = (file or "").strip()
    if not chosen:
        raise ValueError(
            "no EXR was chosen. Pick one from the file list. A folder added under "
            "paths.allow_read in config.yaml appears there under its own name"
        )
    found = picker.resolve(chosen, EXTENSIONS)
    if found is None:
        raise ValueError(
            f"`{chosen}` names no .exr that is there any more. Pick another from the file "
            f"list, or add its folder to paths.allow_read in config.yaml"
        )
    return found


def _publish_report(reading, peak: float, above: float, name: str) -> None:
    """Report what came off disk to the node's own interface.

    Never raises, and never changes what the node returns.

    Args:
        reading: The :class:`modules.image.exr.Reading` that was read.
        peak: The largest value in it.
        above: Share of its pixels over 1.0, from 0 to 1.
        name: The file's own name, without the folders leading to it.
    """
    try:
        if not run_result.watching():
            return
        height, width = (int(size) for size in reading.pixels.shape[:2])
        share = above * 100.0
        run_result.publish(
            status=run_result.OK,
            summary=f"{name}, peak {peak:.2f}, {share:.2f}% of the frame above 1.0",
            counts={"peak": round(peak, 2), "above one %": round(share, 2)},
            facts={
                "size": f"{width} x {height}",
                "channels": "".join(reading.channels) or "none",
                "compression": reading.compression,
                "depth": reading.depth,
                "file": name,
            },
        )
    except Exception as error:
        logger.debug("no EXR reading report was published (%s)", error)


class EXRLoad(io.ComfyNode):
    """Read one OpenEXR file as linear light, with its coverage as a mask."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASEXRLoad",
            display_name="EXR Load",
            search_aliases=[
                "WASEXRLoad", "EXR Load", "OpenEXR", "open exr", "load exr", "hdr load",
                "linear load", "read exr",
            ],
            category="WAS Suite/Image/HDR",
            description=(
                "Read an OpenEXR file as linear light, with everything above one kept, so a "
                "plate written by this pack or by a compositing program comes back into the "
                "graph at its real brightness. A fourth channel arrives as the alpha "
                "output. Uncompressed, ZIP, ZIPS and RLE files are read; a PIZ, DWA, B44 or "
                "PXR24 file is named in the message and refused."
            ),
            inputs=[
                io.Combo.Input(
                    "file",
                    options=exr_labels(),
                    tooltip=(
                        "Which EXR to read. Each entry carries the folder it sits in: "
                        "`plate_0001.exr [output]`, `shot.exr [input]`, "
                        "`scratch.exr [temp]`. EXR Save writes into output, so a file it "
                        "wrote is listed here."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="image",
                    tooltip=(
                        "The file's colour as linear light, as a batch of one. Values above "
                        "one are kept, so a preview clips them back to white."
                    ),
                ),
                io.Mask.Output(
                    display_name="alpha",
                    tooltip=(
                        "The file's coverage, white where the pixel is opaque and black "
                        "where it is clear. A file with no alpha channel gives a white mask "
                        "at the frame's size."
                    ),
                ),
                io.Float.Output(
                    display_name="peak",
                    tooltip=(
                        "The largest value anywhere in the image. 40.07 = a highlight forty "
                        "times white; 1.0 = the file holds nothing above white. Divide by "
                        "it to scale the frame back into range."
                    ),
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, file=""):
        """The file's modification time, so a re-run reads it again once it has changed."""
        try:
            return os.path.getmtime(target(file))
        except Exception:
            return float("NaN")

    @classmethod
    def validate_inputs(cls, file):
        """Whether the chosen file is still in one of ComfyUI's own folders."""
        if not (file or "").strip():
            return (
                "no EXR was chosen. Pick one from the file list"
            )
        if sandbox.names_another_host(file):
            return f"`{file}` names another machine. Pick an EXR from the file list"
        if sandbox.annotated_path(file) is None:
            return (
                f"`{file}` names no .exr that is there any more. Pick another from the "
                f"file list"
            )
        return True

    @classmethod
    def execute(cls, file="") -> io.NodeOutput:
        """Read the file and answer its colour, its coverage and its peak.

        Raises:
            PathNotAllowed: ``path`` resolved outside every permitted read root.
            ValueError: Neither widget names a readable EXR, or the file is not one this
                reader unpacks.
        """
        chosen = target(file)
        reading = exr.read(chosen)
        image = reading.pixels.unsqueeze(0)
        coverage = (
            torch.ones(1, *reading.pixels.shape[:2], dtype=torch.float32)
            if reading.alpha is None
            else reading.alpha.unsqueeze(0)
        )

        peak = float(image.amax())
        # A pixel counts once however many of its channels carry the headroom.
        above = float((image.amax(dim=-1) > WHITE).to(torch.float32).mean())
        name = os.path.basename(chosen)
        logger.info(
            "read %s at %dx%d, %s, %s, channels %s, peak %.4g",
            name, int(image.shape[2]), int(image.shape[1]), reading.depth,
            reading.compression, "".join(reading.channels), peak,
        )
        _publish_report(reading, peak, above, name)
        return io.NodeOutput(image, coverage, peak)

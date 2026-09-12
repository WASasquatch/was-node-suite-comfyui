"""Write a layer stack out as a Photoshop document or a layered TIFF."""

from __future__ import annotations

import os

import torch
from comfy_api.latest import io, ui

from ....modules import log
from ....modules.image import layer_ops, psd
from ....modules.interface import file_report
from ....modules.io import naming, rooted
from ....modules.util import sandbox

REQUIRES = "photoshop"

logger = log.get_logger("nodes.image.layers")

NODE_NAME = "Layers Save"

#: What separates the name from the number, and the digits the number is padded to.
DELIMITER = "_"
PADDING = 4

#: What a layer with no name of its own is called in the file.
UNNAMED = "layer"


class LayersSave(io.ComfyNode):
    """Write a ``LAYERS`` document as a layered PSD or TIFF."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLayersSave",
            display_name=NODE_NAME,
            search_aliases=[
                "WASLayersSave",
                "Layers Save",
                "save psd",
                "photoshop",
                "psd export",
                "layered tiff",
                "export layers",
            ],
            category="WAS Suite/Image/Layers",
            description=(
                "Write a layer stack to a file an image editor opens with its layers "
                "intact. Every layer keeps its name, its place on the canvas, its "
                "opacity, its blend mode and whether it was hidden, so the composite can "
                "be taken apart and reworked in Photoshop, Affinity Photo, GIMP or Krita. "
                "A stack whose layers carry a batch is written as one file per frame."
            ),
            inputs=[
                io.Layers.Input(
                    "layers",
                    tooltip="The stack to write, one file per frame it carries; LAYERS.",
                ),
                io.Combo.Input(
                    "file_format",
                    options=list(psd.FORMATS),
                    default=psd.FORMATS[0],
                    tooltip=(
                        "'psd' = a Photoshop document, which every editor opens; 'tiff' = "
                        "a layered TIFF, whose layers Photoshop and Affinity Photo read "
                        "and whose flattened picture opens anywhere at all."
                    ),
                ),
                io.Combo.Input(
                    "root",
                    options=rooted.options(),
                    tooltip=(
                        "Which folder the files land in: ComfyUI's own 'output' or 'temp', "
                        "or any folder added under paths.allow_write in config.yaml, "
                        "listed by its own name."
                    ),
                ),
                io.String.Input(
                    "filename_prefix",
                    default="ComfyUI_layers",
                    multiline=False,
                    tooltip=(
                        "Name and folder below the root, before the number. "
                        "`ComfyUI_layers` gives `ComfyUI_layers_0001.psd`; `plates/shot` "
                        "puts it in that subfolder. Tokens expand, so "
                        "`[time(%Y-%m-%d)]/shot` dates the folder."
                    ),
                ),
                io.Combo.Input(
                    "depth",
                    options=list(psd.DEPTHS),
                    default=psd.DEPTHS[0],
                    tooltip=(
                        "'8 bit' = 256 levels a channel, the smallest file and what every "
                        "editor expects; '16 bit' = 65536 levels, which keeps a graded "
                        "plate off a banded gradient; '32 bit float' = the exact values on "
                        "the wire, keeping anything above white for further grading."
                    ),
                ),
                io.Combo.Input(
                    "compression",
                    options=list(psd.PACKINGS),
                    default=psd.PACKINGS[0],
                    tooltip=(
                        "'rle' = the packing an editor writes itself, the most widely "
                        "read; 'zip' = smaller layers, read by Photoshop 6 and later; "
                        "'none' = stored as they are, largest and fastest. All three are "
                        "lossless, and the flattened picture is packed the same either way."
                    ),
                ),
                io.Image.Input(
                    "composite",
                    optional=True,
                    tooltip=(
                        "A flattened picture to store as the file's preview instead of the "
                        "one this node composites; IMAGE. Wire the graded result so an "
                        "editor shows that until it redraws the layers itself. It has to "
                        "be the size of the canvas."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="files",
                    tooltip=(
                        "Full path of every file written this run, one per line, in frame "
                        "order."
                    ),
                ),
                io.Image.Output(
                    display_name="composite",
                    tooltip=(
                        "The flattened picture stored in each file, as a batch in frame "
                        "order; IMAGE."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        layers,
        file_format=psd.FORMATS[0],
        root=rooted.DEFAULT,
        filename_prefix="ComfyUI_layers",
        depth=psd.DEPTHS[0],
        compression=psd.PACKINGS[0],
        composite=None,
    ) -> io.NodeOutput:
        """Write one file per frame and report what landed in the folder.

        Args:
            layers: A ``LAYERS`` document.
            file_format: One of :data:`modules.image.psd.FORMATS`.
            root: A name from :func:`modules.io.rooted.options`.
            filename_prefix: Folder and name below the root, before the number.
            depth: One of :data:`modules.image.psd.DEPTHS`.
            compression: One of :data:`modules.image.psd.PACKINGS`.
            composite: A flattened picture to store instead of the composited stack.

        Returns:
            The paths written, one per line, and the flattened pictures as a batch.

        Raises:
            ValueError: The stack holds no layer, or the canvas is empty, too large or a
                different size from the composite wired in.
            OSError: The folder could not be created, or a file could not be written.
            PathNotAllowed: The root and prefix resolved outside every permitted write
                root.
        """
        width, height = layer_ops.size_of(layers)
        total = layer_ops.longest(layers)
        if not total:
            raise ValueError(
                f"{NODE_NAME} was handed a stack with no layers in it, so there is nothing "
                f"to write. Wire in a stack built by Layers from Image Batch, or by any of "
                f"the Layer nodes."
            )

        below, _, leaf = (filename_prefix or "").replace("\\", "/").rpartition("/")
        directory = rooted.destination(root, below)
        os.makedirs(directory, exist_ok=True)
        extension = psd.EXTENSIONS[file_format]
        names = naming.next_names(
            str(directory), leaf, DELIMITER, PADDING, extension, total
        )

        written, flattened = [], []
        for frame in range(total):
            document = layer_ops.framed(layers, frame)
            plates = cls.plates(document)
            if not plates:
                continue
            picture, coverage = cls.flattened(document, width, height, composite, frame)
            target = sandbox.resolve_write_file(directory, names[frame])
            psd.write(
                target, plates, (width, height), picture, coverage,
                depth, compression, file_format,
            )
            logger.info("%s wrote %d layer(s) to %s", NODE_NAME, len(plates), target)
            written.append(str(target))
            flattened.append(picture)

        if not written:
            raise ValueError(
                f"{NODE_NAME} found no layer with a picture in it, so no file was written. "
                f"Check that the layers wired in are not all empty."
            )

        batch = torch.stack(flattened, dim=0)
        file_report.publish(
            written,
            intended=total,
            kind=extension,
            folder=str(directory),
            facts={
                "canvas": f"{width}x{height}",
                "layers": str(len(layer_ops.entries(layers))),
                "stored as": f"{depth}, {compression}",
            },
        )
        return io.NodeOutput(
            "\n".join(written), batch, ui=ui.PreviewImage(batch, cls=cls)
        )

    @classmethod
    def plates(cls, document) -> list:
        """Every layer of one frame as a plate the writer takes.

        Args:
            document: A ``LAYERS`` document holding one picture per layer.

        Returns:
            The plates, lowest in the stack first. A layer covering nothing is left out.
        """
        entries = layer_ops.entries(document)
        found = []
        for position, frame in enumerate(layer_ops.drawn(document)):
            if not int(frame.image.shape[0]) or not int(frame.image.shape[1]):
                continue
            entry = entries[position] if position < len(entries) else {}
            found.append(
                psd.Plate(
                    name=frame.name or f"{UNNAMED} {position + 1}",
                    x=int(frame.x),
                    y=int(frame.y),
                    image=frame.image,
                    alpha=frame.coverage,
                    opacity=float(entry.get("opacity", 1.0) or 0.0),
                    blend_mode=str(entry.get("blend_mode") or "normal"),
                    visible=bool(frame.visible),
                )
            )
        return found

    @classmethod
    def flattened(cls, document, width, height, composite, frame):
        """The picture stored as the file's own flattened copy, and its coverage.

        Args:
            document: A ``LAYERS`` document holding one picture per layer.
            width: Canvas width in pixels.
            height: Canvas height in pixels.
            composite: The batch wired in, or None to composite the stack.
            frame: Which frame of that batch to read. A shorter batch holds its last.

        Returns:
            ``(image, coverage)``. The image is ``(height, width, 3)`` and the coverage
            ``(height, width)``, or None where nothing is cut away.

        Raises:
            ValueError: The picture wired in is a different size from the canvas.
        """
        if composite is not None and int(composite.shape[0]):
            picked = composite[min(frame, int(composite.shape[0]) - 1)]
            if int(picked.shape[0]) != height or int(picked.shape[1]) != width:
                raise ValueError(
                    f"composite is {int(picked.shape[1])}x{int(picked.shape[0])} and the "
                    f"canvas is {width}x{height}. Resize it to the canvas, set the canvas "
                    f"to it with Layers Canvas, or leave composite unconnected"
                )
            cover = picked[..., 3] if int(picked.shape[2]) >= 4 else None
            return picked[..., :3], cover
        return layer_ops.composited(layer_ops.entries(document), width, height)

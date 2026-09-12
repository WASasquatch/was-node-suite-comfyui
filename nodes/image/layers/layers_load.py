"""Read a Photoshop document or a layered TIFF back into a layer stack."""

from __future__ import annotations

import os

from comfy_api.latest import io, ui

from ....modules import log
from ....modules.image import layer_ops, psd
from ....modules.util import file_listing, sandbox

REQUIRES = "photoshop"

logger = log.get_logger("nodes.image.layers")

NODE_NAME = "Layers Load"

#: The files this node offers in its menu.
ALLOWED_EXT = (".psd", ".psb", ".tif", ".tiff")

#: Which folders the menu walks.
TAGS = (*file_listing.TAGS, file_listing.CONFIGURED)

#: What a layer with no name in the file is called in the stack.
UNNAMED = "layer"


def document_labels() -> list[str]:
    """Every layered document under the folders a menu offers, as widget values.

    Returns:
        The labels, each carrying the folder it sits in. Empty outside ComfyUI and where
        no root can be read.
    """
    try:
        return file_listing.labels(ALLOWED_EXT, tags=TAGS)
    except Exception as error:
        logger.debug("the file listing could not be read: %s", error)
        return []


class LayersLoad(io.ComfyNode):
    """Read a layered PSD or TIFF into a ``LAYERS`` document."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLayersLoad",
            display_name=NODE_NAME,
            search_aliases=[
                "WASLayersLoad",
                "Layers Load",
                "load psd",
                "photoshop",
                "psd import",
                "layered tiff",
                "import layers",
            ],
            category="WAS Suite/Image/Layers",
            description=(
                "Read a Photoshop document or a layered TIFF into a layer stack, so work "
                "done in an image editor carries on in a graph. Every layer arrives with "
                "its name, its place on the canvas, its opacity, its blend mode and "
                "whether it was hidden. Layers inside a group arrive as ordinary layers, "
                "and adjustment, text and shape layers arrive as the pixels the file "
                "stored for them."
            ),
            inputs=[
                io.Combo.Input(
                    "file",
                    options=document_labels(),
                    tooltip=(
                        "Which document to read. Each entry carries the folder it sits "
                        "in, as `art.psd [input]` or `plate.tif [output]`. A folder added "
                        "under paths.allow_read in config.yaml appears under its own name."
                    ),
                ),
            ],
            outputs=[
                io.Layers.Output(
                    display_name="layers",
                    tooltip="The stack the file held, lowest layer first; LAYERS.",
                ),
                io.Image.Output(
                    display_name="composite",
                    tooltip=(
                        "The flattened picture the file stored, or the stack composited "
                        "here where it stored none; IMAGE."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many layers the stack holds; INT.",
                ),
                io.String.Output(
                    display_name="names",
                    tooltip="What each layer is called, one per line, lowest first.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, file):
        """When the file was last written, so an edited document is read again."""
        try:
            return os.path.getmtime(sandbox.resolve_read(cls.located(file)))
        except Exception:
            return float("NaN")

    @classmethod
    def located(cls, file) -> str:
        """The path one menu label names.

        Args:
            file: A label the menu offered.

        Returns:
            The absolute path.

        Raises:
            ValueError: The label is empty or names nothing in the readable folders.
        """
        label = (file or "").strip()
        if not label:
            raise ValueError(
                f"{NODE_NAME} has no file picked. Choose one from the menu, or put a .psd "
                f"or a layered .tif in ComfyUI's input folder and reload the page."
            )
        found = file_listing.resolve(label, ALLOWED_EXT, tags=TAGS)
        if not found:
            raise ValueError(
                f"{NODE_NAME} cannot find `{label}`. It may have been renamed, moved or "
                f"deleted since the menu was built; reload the page and pick it again."
            )
        return found

    @classmethod
    def execute(cls, file) -> io.NodeOutput:
        """Read the file and answer the stack, its composite and its names.

        Args:
            file: A label the menu offered.

        Returns:
            The stack, the flattened picture, the layer count and the names.

        Raises:
            ValueError: No file is picked, the label names nothing readable, or the file
                is stored in a way this does not read.
            OSError: The file could not be read.
            PathNotAllowed: The file resolved outside every permitted read root.
        """
        path = sandbox.resolve_read(cls.located(file))
        canvas, plates, stored = psd.read(path)
        width, height = int(canvas[0]), int(canvas[1])

        stack = []
        for position, plate in enumerate(plates):
            entry = {
                "image": plate.image.unsqueeze(0),
                "type": "raster",
                "x": int(plate.x),
                "y": int(plate.y),
                "w": int(plate.image.shape[1]),
                "h": int(plate.image.shape[0]),
                "name": plate.name or f"{UNNAMED} {position + 1}",
                "opacity": float(plate.opacity),
                "blend_mode": plate.blend_mode,
                "visible": bool(plate.visible),
                "flip_h": False,
                "flip_v": False,
                "rotation": 0.0,
            }
            if float(plate.alpha.min()) < 1.0:
                entry["mask"] = (1.0 - plate.alpha).unsqueeze(0)
            stack.append(entry)

        document = layer_ops.rebuilt({}, stack)
        document["canvas"] = (width, height)

        if stored is None:
            stored, _cover = layer_ops.composited(stack, width, height)
        composite = stored.unsqueeze(0)
        names = [str(entry["name"]) for entry in stack]

        line = f"{len(stack)} layer(s) read from {os.path.basename(str(path))}"
        layer_ops.report(
            NODE_NAME, line, document,
            counts={"hidden": sum(1 for entry in stack if not entry["visible"])},
            facts={
                "file": os.path.basename(str(path)),
                "blend modes": ", ".join(
                    sorted({str(entry["blend_mode"]) for entry in stack})
                ) or "none",
            },
        )
        logger.info("%s %s", NODE_NAME, line)
        return io.NodeOutput(
            document, composite, len(stack), "\n".join(names),
            ui=ui.PreviewImage(composite, cls=cls),
        )

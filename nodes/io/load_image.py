"""Load an image from a path or a URL."""

from __future__ import annotations

import os

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import WAS_COLOUR_PROFILE
from ...modules.constants import ALLOWED_EXT
from ...modules.image import colour_profile
from ...modules.interface import image_report, run_result
from ...modules.state import history
from ...modules.util import file_listing, sandbox
from ...modules.util.hashing import get_sha256

logger = log.get_logger("nodes.io")

#: Config key of the group that permits this node to reach the network. The node itself is
#: default tier and normally given a path; the URL branch is not.
FEATURE = "features.network"

#: Widget value that leaves a tagged file in its own colour space.
KEEP_PROFILE = colour_profile.KEEP

#: What the file listing puts after a name in ComfyUI's input folder, taken off again so the
#: menu holds what the upload button answers with.
INPUT_TAG = f" [{file_listing.INPUT}]"


def _publish_report(
    answered, decoded, kind, tagged, colour_space, icc_mode, recorded
) -> None:
    """Report what came off disk and what changed on the way to the tensor.

    Never raises, and never changes what the node returns.

    Args:
        answered: The ``(1, height, width, channels)`` tensor the node answers with.
        decoded: The file as it decoded, before any colour conversion, or None.
        kind: The format the file was stored in, as PIL named it.
        tagged: The profile the file carried, or None.
        colour_space: The space widget's value.
        icc_mode: The mode widget's value.
        recorded: The path or address the picture came from.
    """
    try:
        moved = image_report.drift(decoded, answered[0]) if decoded is not None else {}
        if moved:
            # A picture that moved with no conversion behind it is worth the warning colour.
            moved["unexplained"] = moved["moved"] and tagged is None
        kept = colour_space == KEEP_PROFILE
        facts = {
            "file": f"{os.path.basename(recorded) or 'none'}, {kind or 'no format'}",
            "profile": tagged.name if tagged is not None else "none",
            "read as": "unchanged" if kept else f"{colour_space}, {icc_mode}",
        }
        summary = (
            f"{tagged.name}, kept as it is" if tagged is not None and kept
            else f"{tagged.name}, {icc_mode}ed to {colour_space}" if tagged is not None
            else ""
        )
        image_report.publish(answered, facts=facts, moved=moved or None, summary=summary)
    except Exception as error:
        logger.debug("no load report was published (%s)", error)


def decode(opened, kind, recorded, RGBA, filename_text_extension, colour_space, name=None,
           icc_mode=colour_profile.CONVERT):
    """Turn one decoded picture into the four outputs both loaders answer with.

    Args:
        opened: The PIL image, or None to substitute a black 512x512 one.
        kind: The format it was stored in, as PIL named it.
        recorded: The path or address it came from, for the history and the report.
        name: What to call it, or None to take the last part of ``recorded``.
        RGBA: Keep transparency in the image itself.
        filename_text_extension: Keep the extension on the name.
        colour_space: A value of :func:`modules.image.colour_profile.spaces`.
        icc_mode: :data:`~modules.image.colour_profile.CONVERT` or ``ASSIGN``.

    Returns:
        The image, the mask, the name and the profile, as a node output.
    """
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    import node_helpers

    tagged = None
    # Kept only while a browser is watching, since it is a second copy of the picture and
    # nothing but the readout reads it.
    decoded = None
    if opened is None:
        opened = Image.new(mode="RGB", size=(512, 512), color=(0, 0, 0))
    else:
        opened = node_helpers.pillow(ImageOps.exif_transpose, opened)
        if run_result.watching():
            decoded = torch.from_numpy(
                np.array(opened.convert("RGB")).astype(np.float32) / 255.0
            )
        opened, tagged = colour_profile.interpret(
            opened, colour_space, icc_mode, os.path.basename(str(recorded))
        )
        history.update_history_images(recorded)

    picture = opened if RGBA else opened.convert("RGB")
    picture = np.array(picture).astype(np.float32) / 255.0
    picture = torch.from_numpy(picture)[None,]

    if "A" in opened.getbands():
        mask = np.array(opened.getchannel("A")).astype(np.float32) / 255.0
        mask = 1.0 - torch.from_numpy(mask)
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

    called = name or os.path.basename(str(recorded).split(" [")[0])
    filename = called if filename_text_extension else os.path.splitext(called)[0]

    _publish_report(
        picture, decoded, kind, tagged, colour_space, icc_mode, str(recorded)
    )
    return io.NodeOutput(picture, mask, filename, tagged)


def chosen_path(label: str):
    """The file one menu entry names, resolved inside a permitted read root.

    Args:
        label: The widget's value, such as ``cat.png [input]``.

    Returns:
        The absolute path as a string, or None when the entry names nothing that is there.

    Raises:
        PathNotAllowed: The file resolved outside every permitted read root.
    """
    chosen = (label or "").strip()
    if not chosen:
        return None
    found = None
    # A bare name is an input file, which is what the upload button leaves behind.
    try:
        import folder_paths

        if folder_paths.exists_annotated_filepath(chosen):
            found = folder_paths.get_annotated_filepath(chosen)
    except Exception as error:
        logger.debug("`%s` could not be resolved through folder_paths: %s", chosen, error)
    if found is None:
        found = file_listing.resolve(
            chosen, ALLOWED_EXT, tags=file_listing.ROOTS
        )
    if found is None:
        return None
    return str(sandbox.resolve_read(found))


def image_labels() -> list[str]:
    """Every picture under the folders a menu offers, as the labels a widget stores.

    Returns:
        A bare ``<relative path>`` for a file in the input folder, and
        ``<relative path> [output]``, ``[temp]`` or a configured folder's own name for the
        rest. Empty outside ComfyUI and where no root can be read.
    """
    try:
        found = file_listing.labels(
            ALLOWED_EXT, tags=file_listing.ROOTS
        )
    except Exception as error:
        logger.debug("the file listing could not be read: %s", error)
        return []
    # The upload button answers a bare name, the way ComfyUI's own loader stores an input
    # file, so an input entry carries no tag and an uploaded picture is a value in this list.
    return [
        label[: -len(INPUT_TAG)] if label.endswith(INPUT_TAG) else label for label in found
    ]


class ImageLoad(io.ComfyNode):
    """Load one image from a filesystem path or an ``http``/``https`` URL."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Load",
            display_name="Image Load",
            search_aliases=["Image Load", "load image from path", "open image"],
            category="WAS Suite/IO",
            description=(
                "Load an image chosen from a menu of every picture in ComfyUI's input, "
                "output and temp folders, and any folder listed under paths.allow_read in "
                "config.yaml. Upload one with the button and it is selected. A file tagged "
                "with a colour profile is converted to sRGB as it is read, or kept in "
                "its own space, and either way the profile comes out on its own socket. Anything "
                "that cannot be read gives a black 512x512 image so the rest of the "
                "workflow still runs."
            ),
            inputs=[
                io.Combo.Input(
                    "image",
                    options=image_labels(),
                    upload=io.UploadType.image,
                    tooltip=(
                        "Which picture to read. A file in ComfyUI's input folder is listed "
                        "by name; anything else carries the folder it sits in, as "
                        "`render.png [output]` or `scratch.png [temp]`. A folder added under "
                        "paths.allow_read appears under its own name. The button below "
                        "uploads one into input and selects it."
                    ),
                ),
                io.Boolean.Input(
                    "RGBA",
                    default=False,
                    tooltip=(
                        "`off` discards any transparency and hands on a plain colour image, "
                        "which is what samplers and most nodes expect; `on` keeps "
                        "the transparency channel in the image itself. The mask output is "
                        "produced either way."
                    ),
                ),
                io.Boolean.Input(
                    "filename_text_extension",
                    default=True,
                    optional=True,
                    tooltip=(
                        "Whether the filename_text output keeps the extension. On = 'cat.png', "
                        "off = 'cat'. Handy when the name is being reused "
                        "as a caption or as a save prefix."
                    ),
                ),
                io.Combo.Input(
                    "colour_space",
                    options=colour_profile.spaces(),
                    default="sRGB",
                    optional=True,
                    tooltip=(
                        "Which colour space the picture comes out in. \"the file's own\" "
                        "leaves a tagged file exactly as it was written, for post work that "
                        "stays there. 'sRGB' is what a sampler, a filter and a LUT expect. "
                        "The rest, such as 'Adobe RGB (1998)' and 'Display P3', are for a "
                        "photograph that goes back out in its own space."
                    ),
                ),
                io.Combo.Input(
                    "icc_mode",
                    options=list(colour_profile.MODES),
                    optional=True,
                    tooltip=(
                        "What to do with the space above. 'convert' changes the numbers so "
                        "the colour stays put, which is what a photograph wants. 'assign' "
                        "leaves the numbers alone and says they were in that space all "
                        "along, which is how an untagged file that is really Display P3 is "
                        "put right. Ignored for \"the file's own\"."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="image",
                    tooltip="The loaded image, as a batch of one.",
                ),
                io.Mask.Output(
                    display_name="mask",
                    tooltip=(
                        "The image's transparency as a mask, with the transparent parts "
                        "white and the opaque parts black. An image with no transparency "
                        "gives an empty 64x64 mask."
                    ),
                ),
                io.String.Output(
                    display_name="filename_text",
                    tooltip=(
                        "The file's own name, without the folders leading to it, for reuse "
                        "as a caption or a save prefix."
                    ),
                ),
                WAS_COLOUR_PROFILE.Output(
                    display_name="profile",
                    tooltip=(
                        "The colour profile the file was tagged with, such as Adobe RGB "
                        "(1998). Wire it into Image Save to write the result back in that "
                        "space rather than in sRGB. Empty for a file carrying no profile, "
                        "which is most of them."
                    ),
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(
        cls, image="", RGBA=False, filename_text_extension=True, colour_space="sRGB",
        icc_mode=colour_profile.CONVERT,
    ):
        """The file's digest, so a re-run reads it again once it has changed on disk."""
        found = chosen_path(image)
        if found is None:
            return float("NaN")
        return get_sha256(found)

    @classmethod
    def execute(
        cls, image="", RGBA=False, filename_text_extension=True, colour_space="sRGB",
        icc_mode=colour_profile.CONVERT,
    ) -> io.NodeOutput:
        """Load the chosen picture, or substitute a black one for anything unreadable.

        Raises:
            PathNotAllowed: The chosen file resolved outside every permitted read root.
        """
        from PIL import Image

        import node_helpers

        opened, kind = None, ""
        found = chosen_path(image)
        if found is None:
            logger.error(
                "`%s` names no picture under ComfyUI's input, output or temp folders, or "
                "under a folder listed in paths.allow_read. Pick another from the menu, or "
                "upload one with the button.", image,
            )
        else:
            try:
                opened = node_helpers.pillow(Image.open, found)
                kind = (opened.format or "").lower()
            except OSError:
                logger.error("the image `%s` could not be opened", found)
        return decode(
            opened, kind, str(found or image), RGBA, filename_text_extension, colour_space,
            icc_mode=icc_mode,
        )

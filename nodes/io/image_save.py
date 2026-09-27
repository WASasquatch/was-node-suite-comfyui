"""Save a batch of images with WAS's filename scheme and format options."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

from comfy_api.latest import io, ui

from ...modules import log
from ...modules.compat.types import WAS_COLOUR_PROFILE
from ...modules.image import colour_profile, depths, exr, png
from ...modules.interface import file_report
from ...modules.io import naming, rooted
from ...modules.state import history
from ...modules.util import sandbox

logger = log.get_logger("nodes.io")

#: Extensions whose writer carries an ICC profile. A profile is dropped for anything else,
#: since the writer would refuse the keyword and lose the file with it.
PROFILE_FORMATS = frozenset({"png", "jpg", "jpeg", "webp", "tiff"})

#: History key holding every path this node has written.
HISTORY_KEY = "Output_Images"

#: The most files encoded at once.
MAX_WRITERS = 8

#: Stand-in name used to resolve the output directory when the prefix widget is empty.
PLACEHOLDER_PREFIX = "_"


def subfolder_of(directory: str, root: str) -> str | None:
    """``directory`` relative to ``root``.

    Args:
        directory: A directory path.
        root: The directory the result is relative to.

    Returns:
        The relative directory, ``""`` when it is ``root`` itself, or None when it is not
        inside ``root`` at all, in which case nothing in it can be previewed: a preview is
        addressed as a name and a subfolder of one of ComfyUI's own directories.
    """
    import folder_paths

    if not folder_paths.is_within_directory(root, directory):
        return None
    relative = os.path.relpath(os.path.abspath(directory), root)
    return "" if relative == "." else relative


class ImageSave(io.ComfyNode):
    """Write every image in the batch to a permitted output directory under a numbered name."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Save",
            display_name="Image Save",
            search_aliases=["Image Save", "save image", "write image"],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            category="WAS Suite/IO",
            description=(
                "Save images with a token-expanded path, a numbered filename and a choice "
                "of format. root chooses which folder they land in, and filename_prefix "
                "straight into ComfyUI's output directory; a full path writes to that "
                "folder instead, as long as it lands inside ComfyUI's output or temp "
                "folder, the pack's own folder, or a folder listed under paths.allow_write "
                "in config.yaml. Anywhere else is refused, the input folder included."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip=(
                        "The images to write. Every image in the batch gets its own file, "
                        "each with the next number in the sequence."
                    ),
                ),
                io.Combo.Input(
                    "root",
                    options=rooted.options(),
                    tooltip=(
                        "Which folder the files land in: ComfyUI's own 'output' or 'temp', "
                        "or any folder added under paths.allow_write in config.yaml, listed "
                        "by its own name. filename_prefix names the part below it."
                    ),
                ),
                io.String.Input(
                    "filename_prefix",
                    default="ComfyUI",
                    tooltip=(
                        "The name part of each file, before the number. Tokens are expanded "
                        "here too, so '[time(%H-%M)]' or a custom token can go in the name "
                        "rather than the folder. Cleared, the file is just the delimiter and "
                        "the number."
                    ),
                ),
                io.String.Input(
                    "filename_delimiter",
                    default="_",
                    tooltip=(
                        "What sits between the name and the number: 'ComfyUI_00001.png' with "
                        "the default, 'ComfyUI-00001.png' with '-'."
                    ),
                ),
                io.Int.Input(
                    "filename_number_padding",
                    default=4,
                    min=1,
                    max=9,
                    step=1,
                    tooltip=(
                        "How many digits the number is padded to with leading zeros. 4 gives "
                        "'_0001', 1 gives '_1'. The count keeps rising past the padding, so "
                        "too few digits sorts the files oddly rather than failing."
                    ),
                ),
                io.Boolean.Input(
                    "filename_number_start",
                    default=False,
                    tooltip=(
                        "Where the number goes. Off = last, 'ComfyUI_0001.png'; on = first, "
                        "'0001_ComfyUI.png', which sorts the files by "
                        "number rather than by name."
                    ),
                ),
                io.Combo.Input(
                    "extension",
                    options=["png", "jpg", "jpeg", "gif", "tiff", "webp", "bmp", "exr"],
                    tooltip=(
                        "The file format. `png` is lossless, carries the workflow and takes "
                        "16 bits a channel; `exr` holds unclipped linear light at 16 or 32 "
                        "bit; `jpg` and `jpeg` are small and lossy; `webp` is small and can "
                        "be either; `tiff` and `bmp` are large and lossless; `gif` is "
                        "limited to 256 colours."
                    ),
                ),
                io.Int.Input(
                    "dpi",
                    default=300,
                    min=1,
                    max=2400,
                    step=1,
                    tooltip=(
                        "Print resolution recorded in the file, in dots per inch. It does "
                        "not resize anything, it only tells a printer or a layout program "
                        "how large to place the image. Written for png, jpg and jpeg only."
                    ),
                ),
                io.Int.Input(
                    "quality",
                    default=100,
                    min=1,
                    max=100,
                    step=1,
                    tooltip=(
                        "How much detail is kept when the format throws some away: 100 is "
                        "the best the format offers, 80 is a common balance, 1 is heavily "
                        "degraded. Applies to jpg, jpeg, webp and tiff; png, bmp and gif "
                        "ignore it."
                    ),
                ),
                io.Boolean.Input(
                    "optimize_image",
                    default=True,
                    tooltip=(
                        "Whether to spend extra time packing the file smaller without "
                        "changing how it looks. Every format except webp and bmp uses it."
                    ),
                ),
                io.Boolean.Input(
                    "lossless_webp",
                    default=False,
                    tooltip=(
                        "For the webp format only. `on` keeps every pixel exactly and makes a "
                        "much larger file; `off` compresses it at the quality set "
                        "above."
                    ),
                ),
                io.Boolean.Input(
                    "overwrite_mode",
                    default=False,
                    tooltip=(
                        "`off` numbers every file, so nothing is ever replaced. `on` drops "
                        "the number and writes the prefix alone, overwriting the same file "
                        "on every run, which suits a fixed path an external tool watches."
                    ),
                ),
                io.Boolean.Input(
                    "show_history",
                    default=False,
                    tooltip=(
                        "`off` previews the images this run wrote; `on` previews the "
                        "files this node has written before instead, newest first, up to the "
                        "limit in the pack's config."
                    ),
                ),
                io.Boolean.Input(
                    "show_history_by_prefix",
                    default=True,
                    tooltip=(
                        "Narrows the history preview to files in the same folder whose names "
                        "start with the same prefix. Only has an effect when show_history is on."
                    ),
                ),
                io.Boolean.Input(
                    "embed_workflow",
                    default=True,
                    tooltip=(
                        "Whether to store the workflow inside the file, so dragging the "
                        "image back into ComfyUI rebuilds the graph. Only png and webp can "
                        "carry it, and nothing is stored if ComfyUI was started with "
                        "--disable-metadata."
                    ),
                ),
                io.Boolean.Input(
                    "show_previews",
                    default=True,
                    tooltip=(
                        "Whether the saved images appear on the node. `off` still writes "
                        "the files and keeps the node small, which suits a long batch."
                    ),
                ),
                io.Combo.Input(
                    "bit_depth",
                    options=list(depths.OPTIONS),
                    tooltip=(
                        "How finely each channel is stored. `8-bit` is 256 levels and is "
                        "what every format takes; `16-bit` is 65536 and needs `png`, which "
                        "keeps a gradient banding-free through further grading; `32-bit "
                        "float` needs `exr` and stores light above 1.0 rather than clipping "
                        "it. A depth the chosen format cannot hold fails the prompt."
                    ),
                ),
                WAS_COLOUR_PROFILE.Input(
                    "profile",
                    optional=True,
                    tooltip=(
                        "A colour profile from Image Load. The images are converted from "
                        "sRGB into that space and the file is written carrying it, so a "
                        "photograph goes out tagged the way it came in. Left unconnected, "
                        "the file is written in sRGB and carries no profile."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip=(
                        "The same images that came in, unchanged, so the node can sit in the "
                        "middle of a chain instead of ending it."
                    ),
                ),
                io.String.Output(
                    display_name="files",
                    tooltip=(
                        "Full path of every file written this run, in batch order. A file "
                        "that could not be written is left out of the list."
                    ),
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        images,
        root=rooted.DEFAULT,
        filename_prefix="ComfyUI",
        filename_delimiter="_",
        filename_number_padding=4,
        filename_number_start=False,
        extension="png",
        dpi=300,
        quality=100,
        optimize_image=True,
        lossless_webp=False,
        overwrite_mode=False,
        show_history=False,
        show_history_by_prefix=True,
        embed_workflow=True,
        show_previews=True,
        bit_depth=depths.DEFAULT,
        profile=None,
    ) -> io.NodeOutput:
        """Write the batch and preview what landed in the output directory.

        Raises:
            PathNotAllowed: The chosen root is not a folder this pack may write to.
            ValueError: The chosen format cannot hold the chosen bit depth.
        """
        import folder_paths

        # Only png and webp carry the workflow, and ComfyUI's --disable-metadata drops it
        # from both.
        metadata_source = cls if embed_workflow else None


        output_dir = folder_paths.get_output_directory()
        # The prefix is a path below the root, so its folders are resolved here, where a
        # refusal can name the root that was chosen. get_save_image_path holds the name to
        # that folder afterwards, which catches anything this missed.
        wanted = (filename_prefix or "").replace("\\", "/")
        below, _, leaf = wanted.rpartition("/")
        base = str(rooted.destination(root, below))
        named = leaf or PLACEHOLDER_PREFIX

        full_output_folder, resolved, _, _, _ = folder_paths.get_save_image_path(
            named, base, images[0].shape[1], images[0].shape[0]
        )
        destination = sandbox.resolve_write(full_output_folder)
        prefix = resolved if filename_prefix else ""

        # A preview is addressed relative to ComfyUI's output directory, so files written
        # anywhere else are saved without one.
        subfolder = subfolder_of(str(destination), output_dir)

        if extension not in depths.FORMATS:
            logger.error(
                "the extension `%s` is not valid. The valid formats are: %s",
                extension, ", ".join(sorted(depths.FORMATS)),
            )
            extension = "png"

        refused = depths.refusal(extension, bit_depth)
        if refused:
            raise ValueError(refused)

        if profile is not None and extension not in PROFILE_FORMATS:
            logger.warning(
                "a %s file carries no colour profile, so the %s profile is not written with "
                "it. Save as PNG, JPEG, WebP or TIFF to keep it.",
                extension, getattr(profile, "name", "given"),
            )

        names = naming.next_names(
            str(destination), prefix, filename_delimiter, filename_number_padding,
            extension, len(images),
            overwrite=overwrite_mode,
            number_first=filename_number_start,
        )

        text = cls.text_chunks(metadata_source)

        def encode(image, file):
            output_file = str(sandbox.resolve_write_file(destination, file))
            try:
                if extension == "exr":
                    exr.write(output_file, image, depths.EXR_DEPTHS[bit_depth], "zip")
                elif extension == "png":
                    frame, icc = colour_profile.from_srgb_array(
                        image.detach().cpu().numpy(), profile, deep=bit_depth != "8-bit"
                    )
                    png.write(
                        output_file,
                        frame,
                        depth=depths.BITS[bit_depth],
                        dpi=dpi,
                        icc=icc or None,
                        text=text,
                        optimize=optimize_image,
                    )
                else:
                    img = ui.ImageSaveHelper._convert_tensor_to_pil(image)
                    img, icc = colour_profile.from_srgb(img, profile)
                    tag = {"icc_profile": icc} if icc and extension in PROFILE_FORMATS else {}

                    if extension in ("jpg", "jpeg"):
                        img.save(output_file, quality=quality, optimize=optimize_image, dpi=(dpi, dpi), **tag)
                    elif extension == "webp":
                        exif_data = ui.ImageSaveHelper._create_webp_metadata(img, metadata_source)
                        img.save(output_file, quality=quality, lossless=lossless_webp, exif=exif_data, **tag)
                    elif extension == "bmp":
                        img.save(output_file)
                    elif extension == "tiff":
                        img.save(output_file, quality=quality, optimize=optimize_image, **tag)
                    else:
                        img.save(output_file, optimize=optimize_image, **tag)
                return output_file
            except OSError as error:
                logger.error("unable to save file to: %s\n%s", output_file, error)
            except Exception as error:
                logger.error("unable to save file due to the following error:\n%s", error)
            return None

        # Encoders release the interpreter lock while compressing, so frames encode in parallel.
        workers = max(1, min(len(images), os.cpu_count() or 1, MAX_WRITERS))
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                written = list(pool.map(encode, images, names))
        else:
            written = [encode(image, file) for image, file in zip(images, names)]

        results = []
        output_files = []
        for output_file, file in zip(written, names):
            if output_file is None:
                continue
            logger.info("image file saved to: %s", output_file)
            output_files.append(output_file)

            # No browser draws an EXR, so one is reported in the panel and nowhere else.
            if (
                extension != "exr"
                and not show_history
                and show_previews
                and subfolder is not None
            ):
                results.append(ui.SavedResult(file, subfolder, io.FolderType.output))

            history.update_history_output_images(output_file)

        if show_history and show_previews:
            results += cls.history_previews(
                output_dir, subfolder, prefix, show_history_by_prefix
            )

        file_report.publish(
            output_files,
            intended=len(images),
            kind=extension,
            folder=str(destination),
            facts={"overwrite": overwrite_mode, "depth": bit_depth},
        )

        if not show_previews:
            results = []
        return io.NodeOutput(images, output_files, ui=ui.SavedImages(results))

    @classmethod
    def text_chunks(cls, source) -> dict:
        """The prompt and workflow a PNG carries, as keyword to JSON text.

        Args:
            source: The node class carrying the hidden prompt and extra pnginfo, or None
                when no workflow is stored.

        Returns:
            One entry per stored document, empty when nothing is to be stored.
        """
        import json

        from comfy.cli_args import args

        if args.disable_metadata or source is None or not source.hidden:
            return {}
        text = {}
        if source.hidden.prompt:
            text["prompt"] = json.dumps(source.hidden.prompt)
        for key, value in (source.hidden.extra_pnginfo or {}).items():
            text[key] = json.dumps(value)
        return text

    @classmethod
    def history_previews(cls, output_dir, subfolder, filename_prefix, by_prefix) -> list:
        """Previews for the files this node has written before, newest first.

        Args:
            output_dir: ComfyUI's output directory.
            subfolder: Subfolder of ``output_dir`` this run wrote to, or None when it wrote
                outside the output directory, where nothing this run wrote can match.
            filename_prefix: File-name part of the token-expanded prefix this run wrote
                with, which is what the saved names start with.
            by_prefix: Restrict the list to files in the same subfolder whose name starts
                with the same prefix.

        Returns:
            ``ui.SavedResult`` entries, capped at ``history.display_limit()``.
        """
        database = history.open_history_db()
        if not (database.catExists("History") and database.keyExists("History", HISTORY_KEY)):
            return []
        history_paths = database.get("History", HISTORY_KEY)
        if not history_paths:
            return []

        filtered = []
        for image_path in history_paths:
            image_subdir = subfolder_of(os.path.dirname(image_path), output_dir)
            if image_subdir is None or not os.path.exists(image_path):
                continue
            if by_prefix and image_subdir != subfolder:
                continue
            if by_prefix and not os.path.basename(image_path).startswith(filename_prefix):
                continue
            filtered.append((image_path, image_subdir))

        limit = history.display_limit()
        filtered = filtered[-limit:]
        filtered.reverse()
        return [
            ui.SavedResult(os.path.basename(path), where, io.FolderType.output)
            for path, where in filtered
        ]

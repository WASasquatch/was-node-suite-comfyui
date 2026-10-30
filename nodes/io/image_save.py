"""Save a batch of images with WAS's filename scheme and format options."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

from comfy_api.latest import io, ui

from ...modules import log
from ...modules.compat.types import LIST, WAS_COLOUR_PROFILE
from ...modules.image import colour_profile, depths, exr, png
from ...modules.interface import file_report
from ...modules.io import naming, rooted
from ...modules.state import history
from ...modules.util import sandbox

logger = log.get_logger("nodes.io")

#: Extension of the caption file written beside each image.
CAPTION_EXTENSION = ".txt"

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
    if sandbox.names_another_host(directory) or not sandbox.contains(
        Path(os.path.realpath(root)), Path(os.path.realpath(directory))
    ):
        return None
    relative = os.path.relpath(os.path.abspath(directory), root)
    return "" if relative == "." else relative


def preview_place(directory: str):
    """Where ComfyUI serves a file in ``directory`` from, as ``(subfolder, folder type)``.

    Args:
        directory: The folder a file was saved in.

    Returns:
        The subfolder below ComfyUI's output, input or temp folder with that folder's type,
        or None when the folder is inside none of them.
    """
    import folder_paths

    for kind, getter in (
        (io.FolderType.output, folder_paths.get_output_directory),
        (io.FolderType.input, folder_paths.get_input_directory),
        (io.FolderType.temp, folder_paths.get_temp_directory),
    ):
        found = subfolder_of(directory, getter())
        if found is not None:
            return found, kind
    return None


def _items(value) -> list:
    """One input's values as the node receives them: a list as it is, anything else as one."""
    return value if isinstance(value, list) else [value]


def _at(values: list, index: int):
    """The value one pass reads, the last repeating past the end of a shorter list."""
    return values[min(index, len(values) - 1)]


def spread(value, total: int, name: str, lines: bool = False) -> list[str] | None:
    """One entry per image out of a list output, a list on one wire, or one value for all.

    Args:
        value: The input as the node received it.
        total: How many images arrived.
        name: The input's name, for the error.
        lines: Read a string holding several lines as one entry per line.

    Returns:
        ``total`` strings, or None where nothing arrived.

    Raises:
        ValueError: More than one entry arrived and not one per image.
    """
    found = []
    for item in _items(value):
        if item is None:
            continue
        for entry in item if isinstance(item, (list, tuple)) else [item]:
            text = "" if entry is None else str(entry)
            parts = [part for part in text.splitlines() if part.strip()] if lines else []
            found.extend(parts if len(parts) > 1 else [text])
    if not found:
        return None
    if len(found) == 1:
        return found * total
    if len(found) != total:
        raise ValueError(
            f"{len(found)} {name} arrived for {total} image(s). Wire one per image, in the "
            f"images' order, or a single one for all of them"
        )
    return found


def write_caption(directory, name: str, text: str) -> str | None:
    """Write one caption file, its surrounding blank space trimmed.

    Args:
        directory: The resolved folder the image went to.
        name: The caption's file name.
        text: The caption.

    Returns:
        The written path, or None where the write failed, which the log names.

    Raises:
        PathNotAllowed: The name leaves the folder.
    """
    path = sandbox.resolve_save_file(directory, name)
    try:
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text.strip())
    except OSError as error:
        logger.error("unable to save the caption to: %s\n%s", path, error)
        return None
    return str(path)


class _Plan(NamedTuple):
    """One pass's batch: its file names, where they go, and how they are encoded."""

    images: object
    names: list
    texts: list | None
    destination: object
    subfolder: str | None
    place: tuple | None
    prefix: str
    extension: str
    options: dict


class ImageSave(io.ComfyNode):
    """Write every image in the batch to a permitted output directory under a numbered name."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Save",
            display_name="Image Save",
            search_aliases=[
                "Image Save", "save image", "write image", "caption", "save captions",
                "dataset", "training data", "filenames",
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            category="WAS Suite/IO",
            is_input_list=True,
            description=(
                "Save every image in a batch as a numbered file in png, jpg, webp, tiff, bmp, "
                "gif or exr. root picks the folder and filename_prefix names the subfolders "
                "and file below it, with tokens expanded. Wired filenames and captions name "
                "each image and write its caption beside it in a .txt, ready for a training "
                "dataset."
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
                    options=rooted.save_options(),
                    tooltip=(
                        "`output`, `temp` or `input` = ComfyUI's own folders, `input` being "
                        "where loaders read uploads and datasets from; any folder under "
                        "paths.allow_write or paths.allow_read in config.yaml is listed by its "
                        "own name. filename_prefix names the path below it."
                    ),
                ),
                io.String.Input(
                    "filename_prefix",
                    default="ComfyUI",
                    tooltip=(
                        "Folders and name below root: `training/laracroft/lara` saves "
                        "training/laracroft/lara_0001.png; a trailing `/`, as "
                        "`training/laracroft/`, picks the folder alone. Tokens are expanded, as "
                        "`[time(%H-%M)]`. Wired filenames replace the name and keep the folders."
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
                        "`png` = lossless, up to 16-bit, carries the workflow; `exr` = "
                        "unclipped linear light at 16 or 32 bit; `jpg`/`jpeg` = small, lossy; "
                        "`webp` = small, lossy or lossless; `tiff`/`bmp` = large, lossless; "
                        "`gif` = 256 colours."
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
                        "`on` = smaller files for a little more time, png at deflate level 6; "
                        "`off` = the fastest writes, png at level 1 and a few percent larger. "
                        "The pixels are the same either way. webp and bmp ignore it."
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
                        "`8-bit` = 256 levels, every format but `exr`; `16-bit` = 65536 "
                        "levels, `png` or `exr`, keeps gradients banding-free through grading; "
                        "`32-bit float` = `exr` only, keeps light above 1.0. A depth the "
                        "format cannot hold fails the prompt."
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
                io.MultiType.Input(
                    "filenames",
                    [LIST, io.String],
                    optional=True,
                    tooltip=(
                        "One name per image, as filename_list from Load Image Batch. The "
                        "folders in filename_prefix still apply: `training/laracroft/` and "
                        "`cat` save training/laracroft/cat.png. A name already there gets a "
                        "number unless overwrite_mode is on. Several lines are several names."
                    ),
                ),
                io.MultiType.Input(
                    "captions",
                    [LIST, io.String],
                    optional=True,
                    tooltip=(
                        "One caption per image, written as UTF-8 text under the image's own "
                        "name: cat.png gets cat.txt. A single caption goes beside every "
                        "image. Surrounding blank lines and spaces are trimmed."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    is_output_list=True,
                    tooltip=(
                        "The same images that came in, unchanged, so the node can sit in the "
                        "middle of a chain instead of ending it."
                    ),
                ),
                io.String.Output(
                    display_name="files",
                    is_output_list=True,
                    tooltip=(
                        "Full path of every image written this run, in batch order. A file "
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
        filenames=None,
        captions=None,
    ) -> io.NodeOutput:
        """Plan one pass per item of the longest input list, then encode every image of the run at once.

        Raises:
            PathNotAllowed: The chosen root, or a wired name, is not a place this pack may
                write to.
            ValueError: No image arrived, the chosen format cannot hold the chosen bit depth,
                or the filenames or captions are neither one nor one per image.
        """
        import folder_paths

        settings = {
            "images": images, "root": root, "filename_prefix": filename_prefix,
            "filename_delimiter": filename_delimiter,
            "filename_number_padding": filename_number_padding,
            "filename_number_start": filename_number_start, "extension": extension,
            "dpi": dpi, "quality": quality, "optimize_image": optimize_image,
            "lossless_webp": lossless_webp, "overwrite_mode": overwrite_mode,
            "show_history": show_history, "show_history_by_prefix": show_history_by_prefix,
            "embed_workflow": embed_workflow, "show_previews": show_previews,
            "bit_depth": bit_depth, "profile": profile,
        }
        lists = {key: _items(value) for key, value in settings.items()}
        batches = [batch for batch in lists["images"] if batch is not None]
        if not batches:
            raise ValueError("no images arrived. Wire the images to save into images")
        lists["images"] = batches
        passes = max(len(values) for values in lists.values())
        each = [
            {key: _at(values, index) for key, values in lists.items()}
            for index in range(passes)
        ]
        total = sum(int(chosen["images"].shape[0]) for chosen in each)
        stems = spread(filenames, total, "filenames", lines=True)
        texts = spread(captions, total, "captions")

        claimed: set = set()
        plans, start = [], 0
        for chosen in each:
            count = int(chosen["images"].shape[0])
            plans.append(cls.plan_batch(
                **chosen,
                stems=stems[start:start + count] if stems else None,
                texts=texts[start:start + count] if texts else None,
                claimed=claimed,
            ))
            start += count

        written = cls.encode_all(plans)

        per_pass: list[list[str]] = [[] for _ in plans]
        captioned, previews = [], []
        for (number, position), output_file in written:
            if output_file is None:
                continue
            plan = plans[number]
            name = plan.names[position]
            logger.info("image file saved to: %s", output_file)
            per_pass[number].append(output_file)
            if plan.texts is not None:
                caption = write_caption(
                    plan.destination, os.path.splitext(name)[0] + CAPTION_EXTENSION,
                    plan.texts[position],
                )
                if caption is not None:
                    captioned.append(caption)
            # No browser draws an EXR, so one is reported in the panel and nowhere else.
            if (
                plan.extension != "exr"
                and not plan.options["show_history"]
                and plan.options["show_previews"]
                and plan.place is not None
            ):
                previews.append(ui.SavedResult(name, *plan.place))

        files = [path for paths in per_pass for path in paths]
        if files:
            history.update_history_output_images(files)

        first = each[0]
        if first["show_history"] and first["show_previews"]:
            previews += cls.history_previews(
                folder_paths.get_output_directory(), plans[0].subfolder, plans[0].prefix,
                first["show_history_by_prefix"],
            )

        facts = {"overwrite": first["overwrite_mode"], "depth": first["bit_depth"]}
        if texts is not None:
            facts["captions"] = len(captioned)
        file_report.publish(
            files,
            intended=total,
            kind=plans[0].extension,
            folder=str(plans[0].destination),
            facts=facts,
        )
        return io.NodeOutput(
            [chosen["images"] for chosen in each],
            per_pass,
            ui=ui.SavedImages(previews),
        )

    @classmethod
    def plan_batch(
        cls,
        images,
        root,
        filename_prefix,
        filename_delimiter,
        filename_number_padding,
        filename_number_start,
        extension,
        dpi,
        quality,
        optimize_image,
        lossless_webp,
        overwrite_mode,
        show_history,
        show_history_by_prefix,
        embed_workflow,
        show_previews,
        bit_depth,
        profile,
        stems=None,
        texts=None,
        claimed=None,
    ) -> _Plan:
        """Settle where one batch goes and what each file is called; earlier arguments mirror the node's inputs.

        Args:
            stems: One name per image, or None to number them after the prefix.
            texts: One caption per image, or None to write none.
            claimed: Paths the run has named so far, lower case, shared by every pass.

        Returns:
            The pass, ready to encode.

        Raises:
            PathNotAllowed: The root or a name is not a place this pack may write to.
            ValueError: The chosen format cannot hold the chosen bit depth.
        """
        import folder_paths

        del show_history_by_prefix
        output_dir = folder_paths.get_output_directory()
        # The prefix is a path below the root, so its folders are resolved here, where a
        # refusal can name the root that was chosen. get_save_image_path holds the name to
        # that folder afterwards, which catches anything this missed.
        wanted = (filename_prefix or "").replace("\\", "/")
        below, _, leaf = wanted.rpartition("/")
        base = str(rooted.save_destination(root, below))
        named = leaf or PLACEHOLDER_PREFIX

        full_output_folder, resolved, _, _, _ = sandbox.save_image_path(
            named, base, images[0].shape[1], images[0].shape[0]
        )
        destination = sandbox.resolve_save(full_output_folder)
        prefix = resolved if leaf else ""

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

        beside = (CAPTION_EXTENSION,) if texts is not None else ()
        if stems is not None:
            names = naming.given_names(
                str(destination),
                [naming.stem_of(stem) or prefix or PLACEHOLDER_PREFIX for stem in stems],
                filename_delimiter, filename_number_padding, extension,
                overwrite=overwrite_mode,
                number_first=filename_number_start,
                beside=beside,
                claimed=claimed,
            )
        else:
            names = naming.next_names(
                str(destination), prefix, filename_delimiter, filename_number_padding,
                extension, len(images),
                overwrite=overwrite_mode,
                number_first=filename_number_start,
                beside=beside,
                claimed=claimed,
            )

        # Only png and webp carry the workflow, and ComfyUI's --disable-metadata drops it
        # from both.
        metadata_source = cls if embed_workflow else None
        return _Plan(
            images=images,
            names=names,
            texts=texts,
            destination=destination,
            # The history lists the output directory only; a preview is served from
            # ComfyUI's output, input or temp folder, so a file anywhere else has none.
            subfolder=subfolder_of(str(destination), output_dir),
            place=preview_place(str(destination)),
            prefix=prefix,
            extension=extension,
            options={
                "dpi": dpi, "quality": quality, "optimize_image": optimize_image,
                "lossless_webp": lossless_webp, "bit_depth": bit_depth, "profile": profile,
                "show_history": show_history, "show_previews": show_previews,
                "metadata_source": metadata_source, "text": cls.text_chunks(metadata_source),
            },
        )

    @classmethod
    def encode_all(cls, plans: list[_Plan]) -> list[tuple[tuple[int, int], str | None]]:
        """Encode every image of every pass, several files at once.

        Args:
            plans: The run's passes.

        Returns:
            ``((pass, position), path)`` per image in run order, the path None where the
            write failed.
        """
        # Images bound for one path go through one worker in order, so a repeated name under
        # overwrite_mode is written whole each time rather than by two writers at once.
        groups: dict[str, list[tuple[int, int]]] = {}
        for number, plan in enumerate(plans):
            for position, name in enumerate(plan.names):
                key = os.path.join(str(plan.destination), name).lower()
                groups.setdefault(key, []).append((number, position))

        def run(jobs):
            return [
                (job, cls.encode(plans[job[0]], plans[job[0]].images[job[1]], plans[job[0]].names[job[1]]))
                for job in jobs
            ]

        ordered = list(groups.values())
        # Encoders release the interpreter lock while filtering and compressing.
        workers = max(1, min(len(ordered), os.cpu_count() or 1, MAX_WRITERS))
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                done = [pair for result in pool.map(run, ordered) for pair in result]
        else:
            done = [pair for jobs in ordered for pair in run(jobs)]
        return sorted(done, key=lambda pair: pair[0])

    @classmethod
    def encode(cls, plan: _Plan, image, file: str) -> str | None:
        """Write one image of a planned pass.

        Args:
            plan: The pass it belongs to.
            image: One ``(H, W, C)`` image.
            file: Its file name.

        Returns:
            The written path, or None where the write failed, which the log names.
        """
        options = plan.options
        extension = plan.extension
        profile = options["profile"]
        output_file = str(sandbox.resolve_save_file(plan.destination, file))
        try:
            if extension == "exr":
                exr.write(output_file, image, depths.EXR_DEPTHS[options["bit_depth"]], "zip")
            elif extension == "png":
                frame, icc = colour_profile.from_srgb_array(
                    image.detach().cpu().numpy(), profile, deep=options["bit_depth"] != "8-bit"
                )
                png.write(
                    output_file,
                    frame,
                    depth=depths.BITS[options["bit_depth"]],
                    dpi=options["dpi"],
                    icc=icc or None,
                    text=options["text"],
                    optimize=options["optimize_image"],
                )
            else:
                img = ui.ImageSaveHelper._convert_tensor_to_pil(image)
                img, icc = colour_profile.from_srgb(img, profile)
                tag = {"icc_profile": icc} if icc and extension in PROFILE_FORMATS else {}
                quality, optimize = options["quality"], options["optimize_image"]
                if extension in ("jpg", "jpeg"):
                    img.save(output_file, quality=quality, optimize=optimize,
                             dpi=(options["dpi"], options["dpi"]), **tag)
                elif extension == "webp":
                    exif_data = ui.ImageSaveHelper._create_webp_metadata(img, options["metadata_source"])
                    img.save(output_file, quality=quality, lossless=options["lossless_webp"],
                             exif=exif_data, **tag)
                elif extension == "bmp":
                    img.save(output_file)
                elif extension == "tiff":
                    img.save(output_file, quality=quality, optimize=optimize, **tag)
                else:
                    img.save(output_file, optimize=optimize, **tag)
            return output_file
        except OSError as error:
            logger.error("unable to save file to: %s\n%s", output_file, error)
        except Exception as error:
            logger.error("unable to save file due to the following error:\n%s", error)
        return None

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

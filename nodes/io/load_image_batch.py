"""Load one image at a time out of a directory, by index, in sequence or at random."""

from __future__ import annotations

import glob
import os
import random

from comfy_api.latest import io

from ...modules import log
from ...modules.constants import ALLOWED_EXT, MAX_SEQUENCE_FRAMES
from ...modules.compat.lists import require_values
from ...modules.convert.tensors import pil2tensor
from ...modules.image import colour_profile, deep_png
from ...modules.interface import image_report
from ...modules.io import picker
from ...modules.state import history
from ...modules.state.database import get_settings_db
from ...modules.util import sandbox
from ...modules.util.hashing import get_sha256

logger = log.get_logger("nodes.io")

#: Settings database categories holding one cursor per loader.
COUNTERS = "Batch Counters"
PATHS = "Batch Paths"
PATTERNS = "Batch Patterns"


def scan(directory: str, pattern: str) -> list[str]:
    """Every image in ``directory`` matching ``pattern``, sorted, as absolute paths.

    Args:
        directory: Directory to scan, already resolved inside a permitted read root.
        pattern: Glob pattern, matched under ``directory``.

    Returns:
        Absolute paths whose extension is in ``ALLOWED_EXT``, sorted.

    Raises:
        PathNotAllowed: A match resolved outside every permitted read root.
        ValueError: The pattern holds a ``..`` segment, which would walk out of the
            directory the node was pointed at.
    """
    if ".." in pattern.replace("\\", "/").split("/"):
        raise ValueError(
            f"the pattern `{pattern}` walks out of `{directory}`; patterns match inside "
            f"the directory they are given"
        )
    # The directory is escaped so a ``[`` or ``*`` in a real directory name is not read as
    # part of the glob. The pattern is the user's and is left unescaped.
    found = [
        str(sandbox.resolve_read(name))
        for name in glob.glob(os.path.join(glob.escape(directory), pattern), recursive=True)
        if name.lower().endswith(ALLOWED_EXT)
    ]
    found.sort()
    return found


class BatchImageLoader:
    """The image list for one loader, plus the cursor the incremental mode walks.

    Attributes:
        image_paths: Absolute paths of every matching image, sorted. Each one has been
            resolved inside a permitted read root by :func:`scan`.
        index: Cursor into ``image_paths`` for the incremental mode.
        key: The name the cursor is stored under, which is the node's own.
    """

    def __init__(self, directory_path: str, key: str, pattern: str):
        """Build the list and read the cursor.

        Args:
            directory_path: Directory to scan, already resolved inside a permitted read
                root. Stored under the key, so it is a string rather than a path.
            key: Name the cursor is stored under.
            pattern: Glob pattern, matched under ``directory_path``.

        Raises:
            PathNotAllowed: A file found in the directory resolved outside every permitted
                read root.
        """
        self.database = get_settings_db()
        self.image_paths = scan(directory_path, pattern)
        self.key = key
        stored_directory = self.database.get(PATHS, key)
        stored_pattern = self.database.get(PATTERNS, key)
        if stored_directory != directory_path or stored_pattern != pattern:
            self.index = 0
            self.database.insert(COUNTERS, key, 0)
            self.database.insert(PATHS, key, directory_path)
            self.database.insert(PATTERNS, key, pattern)
        else:
            self.index = self.database.get(COUNTERS, key) or 0

    def read(self, path: str):
        """Open one image, applying its EXIF orientation.

        Args:
            path: An entry of :attr:`image_paths`, already contained by :func:`scan`.
        """
        from PIL import Image, ImageOps

        import node_helpers

        name = os.path.basename(str(path))
        if deep_png.is_deep(path):
            try:
                picture = deep_png.read(path)
                tagged = None
                if picture.header.profile:
                    with Image.open(path) as opened:
                        tagged = colour_profile.carried(opened, converted=False)
                colour_profile.interpret_deep(tagged, "sRGB", colour_profile.CONVERT, name)
                return picture
            except ValueError as error:
                logger.error("%s, so it is read at 8 bits instead", error)
        image = node_helpers.pillow(Image.open, path)
        image = node_helpers.pillow(ImageOps.exif_transpose, image)
        return colour_profile.to_srgb(image, name)

    def image_by_id(self, image_id: int):
        """``(image, file name)`` for one index, or ``(None, None)`` when out of range."""
        if image_id < 0 or image_id >= len(self.image_paths):
            logger.error("invalid image index `%s`", image_id)
            return (None, None)
        return (self.read(self.image_paths[image_id]), os.path.basename(self.image_paths[image_id]))

    def next_image(self):
        """``(image, file name, position)`` at the cursor, then advance and store it.

        Returns ``(None, None, 0)`` when the directory holds no matching image.
        """
        if not self.image_paths:
            return (None, None, 0)
        if self.index >= len(self.image_paths):
            self.index = 0
        image_path = self.image_paths[self.index]
        position = self.index
        self.index += 1
        if self.index == len(self.image_paths):
            self.index = 0
        logger.info("%s index: %s", self.key, self.index)
        self.database.insert(COUNTERS, self.key, self.index)
        return (self.read(image_path), os.path.basename(image_path), position)


def as_tensor(image, rgba: bool):
    """One loaded picture as a ``(1, height, width, channels)`` tensor.

    Args:
        image: A PIL image, or a 16-bit :class:`modules.image.deep_png.Picture`.
        rgba: Keep transparency as a fourth channel.

    Returns:
        The tensor, at the precision the file was stored in.
    """
    if isinstance(image, deep_png.Picture):
        return deep_png.tensors(image, rgba)[0]
    return pil2tensor(image if rgba else image.convert("RGB"))


class LoadImageBatch(io.ComfyNode):
    """Load one image from a directory by index, in sequence, or at random."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Load Image Batch",
            display_name="Load Image Batch",
            search_aliases=["Load Image Batch", "image folder", "batch loader"],
            hidden=[io.Hidden.unique_id],
            category="WAS Suite/IO",
            description=(
                "Load one image from a folder by index, in sequence, or at random, or "
                "read the whole folder at once. Queue a prompt repeatedly on "
                "`incremental_image` to walk a folder image by image; take `all_images` to "
                "get every match in one run, as image_list and filename_list. The folder is "
                "picked as a root and a path below it, so it always lands inside ComfyUI's "
                "input, output or temp folder or one listed under paths.allow_read in "
                "config.yaml. A folder that is not there fails the prompt. A 16-bit PNG loads "
                "at full precision."
            ),
            inputs=[
                io.Combo.Input(
                    "mode",
                    options=["single_image", "incremental_image", "random", "all_images"],
                    tooltip=(
                        "Which image of the folder to load. `single_image` takes the one at "
                        "index; `incremental_image` takes the next one each run and remembers "
                        "where it stopped, wrapping round; `random` picks one using seed; "
                        "`all_images` reads every match into image_list and filename_list at "
                        f"once, up to {MAX_SEQUENCE_FRAMES}."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=18446744073709551615,
                    tooltip=(
                        "Chooses the image in `random` mode; the same seed always picks the "
                        "same one. Ignored by the other two modes."
                    ),
                ),
                io.Int.Input(
                    "index",
                    default=0,
                    min=0,
                    max=150000,
                    step=1,
                    tooltip=(
                        "Which image to load in `single_image` mode, counting from 0 through "
                        "the matching files sorted by path. Past the last file the prompt "
                        "fails. `incremental_image` writes the image it just read back here, "
                        "so `single_image` carries on from there. The panel says the same, "
                        "as `index 5 of 0 to 363`."
                    ),
                ),
                io.Combo.Input(
                    "folder",
                    options=picker.folders(),
                    tooltip=(
                        "Which folder to read. A bare 'input', 'output' or 'temp' is that "
                        "folder itself; 'plates/shot_01 [input]' is that folder below it. "
                        "Any folder added under paths.allow_read in config.yaml is listed "
                        "under its own name, and so are the folders inside it."
                    ),
                ),
                io.String.Input(
                    "pattern",
                    default="*",
                    multiline=False,
                    tooltip=(
                        "Which files in the folder to consider. '*' takes them all, "
                        "'cat_*.png' only those named that way, and '**/*' also descends "
                        "into subfolders. Files whose format this pack cannot read are "
                        "skipped whatever the pattern says."
                    ),
                ),
                io.Boolean.Input(
                    "allow_RGBA_output",
                    default=False,
                    tooltip=(
                        "`off` discards any transparency and hands on a plain colour image, "
                        "which is what samplers and most nodes expect; `on` keeps "
                        "the transparency channel."
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
            ],
            outputs=[
                io.Image.Output(
                    display_name="image",
                    tooltip="The single image this run selected, as a batch of one.",
                ),
                io.String.Output(
                    display_name="filename_text",
                    tooltip=(
                        "That image's own file name, without the folders leading to it, for "
                        "reuse as a caption or a save prefix."
                    ),
                ),
                io.Image.Output(
                    display_name="image_list",
                    is_output_list=True,
                    tooltip=(
                        "One image at a time rather than one batch, so a node wired here "
                        "runs once per image. On `all_images` that is every match in the "
                        "folder; on the other three modes it is the one image this run "
                        "selected."
                    ),
                ),
                io.String.Output(
                    display_name="filename_list",
                    is_output_list=True,
                    tooltip=(
                        "One name per image, in the same order as image_list, with the "
                        "extension dropped so it can be wired straight into Image Save's "
                        "filename_prefix. Image Save adds the extension it writes."
                    ),
                ),
            ],
        )

    @classmethod
    def cursor_key(cls) -> str:
        """The name this node's place in the folder is stored under.

        Returns:
            A name built from the node's graph id, so every loader keeps its own place.
        """
        return f"node {cls.hidden.unique_id}"

    @classmethod
    def fingerprint_inputs(
        cls,
        mode="single_image",
        seed=0,
        index=0,
        folder="",
        pattern="*",
        allow_RGBA_output=False,
        filename_text_extension=True,
    ):
        """The digest of the image ``index`` selects; every other mode is always stale.

        Raises:
            PathNotAllowed: The chosen root, or a file found in it, is not one this
                pack may read.
        """
        if mode not in ("single_image", "all_images"):
            return float("NaN")
        directory = picker.resolve_folder(folder)
        if directory is None or not directory.is_dir():
            return float("NaN")
        paths = scan(str(directory), pattern)
        if mode == "all_images":
            return repr([(name, os.path.getmtime(name)) for name in paths])
        if 0 <= index < len(paths):
            return get_sha256(paths[index])
        return float("NaN")

    @classmethod
    def execute(
        cls,
        mode="single_image",
        seed=0,
        index=0,
        folder="",
        pattern="*",
        allow_RGBA_output=False,
        filename_text_extension=True,
    ) -> io.NodeOutput:
        """Select one image out of the directory.

        Raises:
            PathNotAllowed: The chosen root, or a file found in it, is not one this
                pack may read.
            ValueError: The directory does not exist, or no image answers the mode.
        """
        directory = picker.resolve_folder(folder)
        if directory is None or not directory.is_dir():
            raise ValueError(
                f"`{folder}` names no folder that is there. Pick another from the menu, or "
                f"add its folder to paths.allow_read in config.yaml"
            )

        loader = BatchImageLoader(str(directory), cls.cursor_key(), pattern)
        if mode == "all_images":
            return cls.every_image(loader, folder, allow_RGBA_output, filename_text_extension)
        if mode == "single_image":
            image, filename = loader.image_by_id(index)
            position = index
            if image is None:
                raise ValueError(f"no valid image was found for the index `{index}`")
        elif mode == "incremental_image":
            image, filename, position = loader.next_image()
            if image is None:
                raise ValueError(
                    "no valid image was found for the next index. Were images removed "
                    "from the source directory?"
                )
        else:
            position = int(random.Random(seed).random() * len(loader.image_paths))
            image, filename = loader.image_by_id(position)
            if image is None:
                raise ValueError(f"no valid image was found for the random index `{position}`")

        history.update_history_images(loader.image_paths)

        if not filename_text_extension:
            filename = os.path.splitext(filename)[0]

        answered = as_tensor(image, allow_RGBA_output)
        cls._publish_report(
            answered, position, len(loader.image_paths), filename, mode, folder,
            depth=16 if isinstance(image, deep_png.Picture) else 8,
        )
        stem = os.path.splitext(filename)[0]
        return io.NodeOutput(answered, filename, [answered], [stem])

    @classmethod
    def every_image(cls, loader, folder, allow_RGBA_output, filename_text_extension):
        """Read every match in the folder, as one image and one name each.

        Args:
            loader: The :class:`BatchImageLoader` holding the matching paths.
            folder: The menu label the folder was picked from.
            allow_RGBA_output: Whether transparency is kept.
            filename_text_extension: Whether filename_text keeps its extension.

        Returns:
            The node's five outputs, the first two describing the first image read.

        Raises:
            ValueError: The folder holds no image this pack can read.
        """
        require_values(
            loader.image_paths,
            f"`{folder}` holds no image this pack can read. Point the node at another "
            f"folder, or widen pattern.",
        )
        paths = loader.image_paths[:MAX_SEQUENCE_FRAMES]
        if len(paths) < len(loader.image_paths):
            logger.warning(
                "all_images read the first %d of %d matches in `%s`; the rest are past the "
                "%d ceiling on one load. Narrow pattern, or use Load Image Sequence, which "
                "takes a range",
                len(paths), len(loader.image_paths), folder, MAX_SEQUENCE_FRAMES,
            )

        images, names, depth = [], [], 8
        for path in paths:
            picture = loader.read(path)
            if not images and isinstance(picture, deep_png.Picture):
                depth = 16
            images.append(as_tensor(picture, allow_RGBA_output))
            names.append(os.path.basename(path))

        history.update_history_images(loader.image_paths)
        first = names[0] if filename_text_extension else os.path.splitext(names[0])[0]
        cls._publish_report(images[0], 0, len(paths), first, "all_images", folder, depth=depth)
        logger.info("Load Image Batch read %d image(s) from `%s`", len(images), folder)
        return io.NodeOutput(
            images[0],
            first,
            images,
            [os.path.splitext(name)[0] for name in names],
        )

    @staticmethod
    def _publish_report(images, position, total, filename, mode, folder, depth=8) -> None:
        """Report which image of the folder was read, for the panel on this node.

        Never raises, and never changes what the node returns.

        Args:
            images: The batch the node answered.
            position: Index of the image within the sorted match list.
            total: How many images matched.
            filename: Name of the image that was read.
            mode: Which of the three ways it was chosen.
            folder: The menu label the folder was picked from.
            depth: Bits a sample the image was stored at.
        """
        try:
            # Counted from 0, the way the index widget counts, so the two never disagree.
            last = max(0, total - 1)
            following = position + 1 if position < last else 0
            image_report.publish(
                images,
                facts={
                    "at": f"index {position} of 0 to {last}",
                    "file": filename,
                    "folder": folder or "input",
                    "picked": mode.replace("_", " "),
                    "next": (
                        f"index {following}"
                        if mode == "incremental_image" else "the same one"
                    ),
                },
                summary=f"{filename}, index {position} of 0 to {last}",
                depth=depth,
            )
        except Exception as error:
            logger.debug("the batch reading was not reported (%s)", error)

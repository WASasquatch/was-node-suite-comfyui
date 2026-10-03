"""Lay the images in a directory out as a contact sheet."""

from __future__ import annotations

import math
import os

from comfy_api.latest import io

from ....modules.io import picker
from ....modules import log
from ....modules.constants import ALLOWED_EXT
from ....modules.convert.tensors import pil2tensor
from ....modules.image import colour_profile
from ....modules.util import sandbox

logger = log.get_logger("nodes.image.process")

#: Number of matches above which the sheet's size is logged before anything is decoded.
#: Every match is opened, resized and held in memory at once, so a folder of holiday
#: photographs is a sheet of a few hundred megapixels and the machine goes quiet for a
#: long time. ComfyUI's own input directory, which the frozen default names, is exactly
#: such a folder on an install that has been used.
LARGE_GRID_MATCHES = 64


class CreateGridImage(io.ComfyNode):
    """Arrange the images in a directory into one grid image."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Create Grid Image",
            display_name="Create Grid Image",
            search_aliases=["Create Grid Image", "contact sheet", "montage", "image grid"],
            category="WAS Suite/Image/Process",
            description=(
                "Build one grid image from the pictures in a folder, for reviewing a whole "
                "output directory at a glance."
            ),
            inputs=[
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
                    "pattern_glob",
                    default="*",
                    multiline=False,
                    tooltip=(
                        "Which files in the folder to include. '*' takes them all, 'cat_*.png' "
                        "only those named that way, and '**/*' also descends into subfolders "
                        "when include_subfolders is on. Files in a format this pack cannot "
                        "read are skipped whatever the pattern says."
                    ),
                ),
                io.Boolean.Input(
                    "include_subfolders",
                    default=False,
                    tooltip=(
                        "Whether '**' in the pattern is allowed to descend into subfolders. With "
                        "off a '**' matches only inside the folder itself."
                    ),
                ),
                io.Int.Input(
                    "border_width",
                    default=3,
                    min=0,
                    max=100,
                    step=1,
                    tooltip=(
                        "Thickness in pixels of the gap between cells, of the outline around "
                        "each cell and of the frame around the sheet. 0 removes all three."
                    ),
                ),
                io.Int.Input(
                    "number_of_columns",
                    default=6,
                    min=1,
                    max=24,
                    step=1,
                    tooltip=(
                        "How many images per row. The number of rows follows from how many "
                        "files matched: 12 images in 6 columns give 2 rows."
                    ),
                ),
                io.Int.Input(
                    "max_cell_size",
                    default=256,
                    min=32,
                    max=1280,
                    step=1,
                    tooltip=(
                        "Size of one cell in pixels, on both sides. Each image is scaled to fit "
                        "inside it and the leftover space is filled with black, so cells stay "
                        "square whatever shape the pictures are."
                    ),
                ),
                io.Int.Input(
                    "border_red",
                    default=0,
                    min=0,
                    max=255,
                    step=1,
                    tooltip="Red level of the gaps, outlines and frame, 0 to 255.",
                ),
                io.Int.Input(
                    "border_green",
                    default=0,
                    min=0,
                    max=255,
                    step=1,
                    tooltip="Green level of the gaps, outlines and frame, 0 to 255.",
                ),
                io.Int.Input(
                    "border_blue",
                    default=0,
                    min=0,
                    max=255,
                    step=1,
                    tooltip=(
                        "Blue level of the gaps, outlines and frame, 0 to 255. All three at 0 "
                        "gives black, all three at 255 gives white."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    tooltip=(
                        "One image holding the whole grid, as a batch of one. A black 512x512 "
                        "image when the folder does not exist."
                    ),
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> float:
        """Re-read the folder on every prompt, never serving a cached grid.

        Returns:
            ``NaN``, which never equals the value cached for the last run.
        """
        return float("NaN")

    @classmethod
    def execute(
        cls,
        folder="",
        pattern_glob="*",
        include_subfolders=False,
        border_width=3,
        number_of_columns=6,
        max_cell_size=256,
        border_red=0,
        border_green=0,
        border_blue=0,
    ) -> io.NodeOutput:
        from PIL import Image

        directory = picker.resolve_folder(folder)
        if directory is None or not directory.is_dir():
            logger.error("`%s` names no folder that is there", folder)
            return io.NodeOutput(pil2tensor(Image.new("RGB", (512, 512), (0, 0, 0))))

        image_paths = cls.scan(directory, pattern_glob, include_subfolders)
        if not image_paths:
            raise ValueError(
                f"no images in `{directory}` match the pattern `{pattern_glob}`. A grid of "
                f"nothing has no size, so there is nothing to build."
            )
        cls.report_size(image_paths, directory, pattern_glob, int(number_of_columns),
                        int(max_cell_size), int(border_width))

        grid_image = cls.smart_grid_image(
            image_paths,
            int(number_of_columns),
            (int(max_cell_size), int(max_cell_size)),
            border_width > 0,
            (int(border_red), int(border_green), int(border_blue)),
            int(border_width),
        )

        return io.NodeOutput(pil2tensor(grid_image))

    @staticmethod
    def report_size(matches, directory, pattern, columns: int, cell: int, border: int) -> None:
        """Log how large the sheet will be, before a single file is opened.

        Nothing is logged at or below :data:`LARGE_GRID_MATCHES` matches.

        Args:
            matches: The files the pattern matched.
            directory: Directory they were matched in.
            pattern: The glob that matched them.
            columns: Cells per row.
            cell: Longest side of one cell, in pixels.
            border: Gap between cells, in pixels.
        """
        if len(matches) <= LARGE_GRID_MATCHES:
            return
        rows = math.ceil(len(matches) / max(columns, 1))
        width = max(columns, 1) * cell + (max(columns, 1) - 1) * border
        height = rows * cell + (rows - 1) * border
        logger.warning(
            "`%s` matches %s image(s) in %s, which lay out as roughly %sx%s pixels: about "
            "%.1f MB of image and four times that as the tensor this node returns. Narrow "
            "pattern_glob, lower max_cell_size or raise number_of_columns if that is more "
            "than you meant.",
            pattern, len(matches), directory, width, height, width * height * 3 / 1_000_000,
        )

    @staticmethod
    def scan(directory, pattern: str, recursive: bool) -> list[str]:
        """The images in ``directory`` matching ``pattern``.

        Args:
            directory: Directory to scan, already resolved and inside a permitted root.
            pattern: Glob pattern, matched under ``directory``.
            recursive: Let ``**`` in the pattern cross directory boundaries.

        Returns:
            Paths whose extension is in ``ALLOWED_EXT``, in the order the directory lists
            them.

        Raises:
            PathNotAllowed: The pattern names another machine, carries a drive, starts at a
                root or climbs out with ``..``, or a match resolved outside every permitted
                read root.
        """
        matches = sandbox.glob_read(directory, pattern, recursive=recursive)
        # A symlink inside a permitted folder can point out of one.
        return [
            str(sandbox.resolve_read(name))
            for name in matches
            if name.lower().endswith(ALLOWED_EXT) and os.path.exists(name)
        ]

    @classmethod
    def smart_grid_image(
        cls,
        images,
        cols=6,
        size=(256, 256),
        add_border=False,
        border_color=(0, 0, 0),
        border_width=3,
    ):
        """Lay out image files as a grid of equal cells.

        Args:
            images: Paths of the images to lay out, in the order they appear in the grid.
            cols: Number of columns.
            size: ``(width, height)`` of one cell in pixels.
            add_border: Outline each cell in ``border_color`` and squash it back to the
                cell size.
            border_color: ``(r, g, b)`` of the gaps, outlines and frame.
            border_width: Thickness in pixels of the gaps, outlines and frame.

        Returns:
            The finished grid as a PIL image.

        Raises:
            OSError: One of the files cannot be opened as an image.
            ValueError: ``images`` is empty, which leaves the grid with no rows and so a
                negative height.
        """
        from PIL import Image, ImageOps

        max_width, max_height = size
        row_height = 0
        images_resized = []
        for image in images:
            img = colour_profile.to_srgb(
                Image.open(image), os.path.basename(str(image))
            ).convert("RGB")

            img_w, img_h = img.size
            aspect_ratio = img_w / img_h
            if aspect_ratio > 1:
                thumb_w = min(max_width, img_w - border_width)
                thumb_h = thumb_w / aspect_ratio
            else:
                thumb_h = min(max_height, img_h - border_width)
                thumb_w = thumb_h * aspect_ratio

            pad_w = max_width - int(thumb_w)
            pad_h = max_height - int(thumb_h)
            left = pad_w // 2
            top = pad_h // 2
            padding = (left, top, pad_w - left, pad_h - top)
            img_resized = ImageOps.expand(img.resize((int(thumb_w), int(thumb_h))), padding)

            images_resized.append(img_resized)
            row_height = max(row_height, img_resized.size[1])
        row_height = int(row_height)

        rows = math.ceil(len(images_resized) / cols)

        new_image = Image.new(
            "RGB",
            (
                cols * size[0] + (cols - 1) * border_width,
                rows * row_height + (rows - 1) * border_width,
            ),
            border_color,
        )

        for i, img in enumerate(images_resized):
            if add_border:
                img = ImageOps.expand(img, border=border_width // 2, fill=border_color)
            x = (i % cols) * (size[0] + border_width)
            y = (i // cols) * (row_height + border_width)
            if img.size != (size[0], size[1]):
                img = img.resize((size[0], size[1]))
            new_image.paste(img, (x, y, x + size[0], y + size[1]))

        return ImageOps.expand(new_image, border=border_width, fill=border_color)

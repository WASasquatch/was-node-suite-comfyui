"""Load a numbered image sequence from a directory as one batch."""

from __future__ import annotations

import os

import torch
from comfy_api.latest import io

from ...modules.io import picker
from ...modules import log
from ...modules import constants
from ...modules.compat import limits
from ...modules.compat.lists import require_values
from ...modules.convert.tensors import stack_images
from ...modules.image import colour_profile, sizing
from ...modules.image.draw import parse_color
from ...modules.interface import batch_report
from ...modules.media import sampling
from .load_image_batch import scan

logger = log.get_logger("nodes.io")

#: Most frames one load answers.
MAX_FRAMES = constants.MAX_SEQUENCE_FRAMES

#: Fill for space a frame does not cover, when one cannot be read from the widget.
FALLBACK_PAD = (0, 0, 0, 255)


class LoadImageSequence(io.ComfyNode):
    """Load every image in a directory, in filename order, as one batch."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLoadImageSequence",
            display_name="Load Image Sequence",
            search_aliases=[
                'WASLoadImageSequence',
                "Load Image Sequence",
                "image sequence",
                "frame sequence",
                "load frames",
                "png sequence",
                "load folder",
            ],
            category="WAS Suite/IO",
            description=(
                "Load a numbered sequence from a folder as one batch, in filename order, "
                "with the same range and strategy controls the frame samplers use. It takes "
                "16 frames unless told otherwise, since a folder can hold thousands. Load "
                "Image Batch beside it serves one frame per run; this serves the run of "
                "frames a video pipeline takes, opening only the files it keeps."
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
                    "pattern",
                    default="*",
                    tooltip=(
                        "Which files to take, as a glob. `*` takes every image in the folder, "
                        "`frame_*.png` takes one numbered run out of a folder holding several. "
                        "Matching is inside the folder only."
                    ),
                ),
                io.Int.Input(
                    "num_frames",
                    default=16,
                    min=0,
                    max=MAX_FRAMES,
                    tooltip=(
                        "How many frames to keep, chosen by the strategy below. 16 by "
                        "default, because a folder can hold thousands and a batch is one "
                        f"tensor in memory. 0 takes every frame in the range, up to the "
                        f"{MAX_FRAMES} ceiling."
                    ),
                ),
                io.Combo.Input(
                    "strategy",
                    options=list(sampling.STRATEGIES),
                    default="uniform",
                    tooltip=(
                        "How num_frames are chosen. uniform = evenly spaced; head = first; "
                        "center = middle; tail = last; random = a seeded pick; every_nth = "
                        "every nth. Only the chosen files are opened, so sampling a long "
                        "capture costs the frames you keep rather than all of them."
                    ),
                ),
                io.Int.Input(
                    "nth",
                    default=1,
                    min=1,
                    max=MAX_FRAMES,
                    tooltip=(
                        "Step between the frames the strategy may choose from. 1 uses every "
                        "frame; 2 thins to every other one first, so `head` takes the opening "
                        "of the clip on alternate frames. It applies to every strategy."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    tooltip=(
                        "Seed for random, so a re-run keeps the same frames. Ignored by the "
                        "other strategies. Any whole number; `0` is as good a seed as any."
                    ),
                ),
                io.Int.Input(
                    "start",
                    default=0,
                    min=-MAX_FRAMES,
                    max=MAX_FRAMES,
                    optional=True,
                    tooltip=(
                        "First file to consider, counting from 0 through the matching files "
                        "in filename order. Negative counts back from the end."
                    ),
                ),
                io.Int.Input(
                    "end",
                    default=-1,
                    min=-MAX_FRAMES,
                    max=MAX_FRAMES,
                    optional=True,
                    tooltip=(
                        "Last file to consider, inclusive. -1 is the final file, which is the "
                        "whole sequence together with a start of 0."
                    ),
                ),
                io.Combo.Input(
                    "resize_mode",
                    options=list(sizing.MODES),
                    default=sizing.FIT_AND_PAD,
                    tooltip=(
                        "How each frame meets the size below, so a folder of mixed sizes "
                        "still stacks. `fit and pad` keeps the whole frame and pads the "
                        "rest, `fill and crop` fills the size and trims the overhang, "
                        "`stretch` distorts to fit, `crop or pad` never resamples."
                    ),
                ),
                io.Int.Input(
                    "width",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=8,
                    tooltip=(
                        "Width every frame is brought to. 0 takes the width of the first "
                        "frame kept, which is what loads a sequence at its own size."
                    ),
                ),
                io.Int.Input(
                    "height",
                    default=0,
                    min=0,
                    max=limits.max_resolution(),
                    step=8,
                    tooltip=(
                        "Height every frame is brought to. 0 takes the height of the first "
                        "frame kept."
                    ),
                ),
                io.Int.Input(
                    "max_size",
                    default=1024,
                    min=0,
                    max=limits.max_resolution(),
                    step=8,
                    optional=True,
                    tooltip=(
                        "Longest edge the derived size is held to, keeping the aspect. Only "
                        "read when width and height are 0, which is where a folder of large "
                        "frames would otherwise fill memory. 0 lifts the cap."
                    ),
                ),
                io.Combo.Input(
                    "interpolation",
                    options=list(sizing.FILTER_NAMES),
                    default=sizing.DEFAULT_FILTER,
                    optional=True,
                    tooltip="Resampling filter. `lanczos` is the sharpest for a downscale.",
                ),
                io.Combo.Input(
                    "align",
                    options=list(sizing.ALIGNMENT_NAMES),
                    default=sizing.DEFAULT_ALIGNMENT,
                    optional=True,
                    tooltip=(
                        "Which part of a frame survives a crop, and which side carries the "
                        "wider bar of a pad."
                    ),
                ),
                io.String.Input(
                    "pad_color",
                    default="#000000",
                    optional=True,
                    tooltip="Fill for space a frame does not cover. Any Pillow colour.",
                ),
                io.Combo.Input(
                    "channels",
                    options=list(sizing.CHANNELS),
                    default="RGB",
                    optional=True,
                    tooltip="Channels the batch carries. `RGBA` keeps the pad transparent.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="The sequence as one batch, in filename order.",
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many frames the batch holds once the range and the strategy have been applied.",
                ),
                io.String.Output(
                    display_name="filenames",
                    tooltip="The filename of each loaded frame, in order, one per line.",
                ),
                io.Image.Output(
                    display_name="image_list",
                    is_output_list=True,
                    tooltip=(
                        "The same frames as one image each rather than one batch, so a node "
                        "wired here runs once per frame. Pair it with filename_list to put "
                        "every frame through Image Save under its own name."
                    ),
                ),
                io.String.Output(
                    display_name="filename_list",
                    is_output_list=True,
                    tooltip=(
                        "One name per frame, in the same order as image_list, with the "
                        "extension dropped so it can be wired straight into Image Save's "
                        "filename_prefix. Image Save adds the extension it writes."
                    ),
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, folder="", pattern="*", **unused):
        """A digest of the matching files as they stand, compared against the last run's.

        Args:
            folder: Folder the images are read from.
            pattern: Glob matched inside it.
            **unused: Every other input, which does not change what is on disk.

        Returns:
            The digest, or ``NaN`` where the folder is not there.
        """
        import hashlib

        found = picker.resolve_folder(folder)
        directory = str(found) if found else ""
        if not directory or not os.path.isdir(directory):
            return float("NaN")
        rows = []
        for name in sorted(os.listdir(directory)):
            try:
                stat = os.stat(os.path.join(directory, name))
            except OSError:
                continue
            rows.append(f"{name}|{stat.st_mtime_ns}|{stat.st_size}")
        return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()

    @classmethod
    def execute(
        cls, folder="", pattern="*", num_frames=16, strategy="head", nth=1, seed=0,
        resize_mode=sizing.FIT_AND_PAD, width=0, height=0, max_size=1024,
        interpolation=sizing.DEFAULT_FILTER, align=sizing.DEFAULT_ALIGNMENT,
        pad_color="#000000", channels="RGB", start=0, end=-1,
    ) -> io.NodeOutput:
        """Load the chosen files as one batch, every frame at one size.

        Raises:
            ValueError: The folder holds no matching image, or the batch does not fit in
                memory at the size asked for.
        """
        import os

        from PIL import Image, ImageOps

        found = picker.resolve_folder(folder)
        directory = str(found) if found else ""
        if not directory or not os.path.isdir(directory):
            raise ValueError(
                f"`{folder}` names no folder that is there. Pick another from the menu, or "
                f"add its folder to paths.allow_read in config.yaml"
            )

        found = scan(directory, pattern)
        if not found:
            raise ValueError(f"no image in `{directory}` matches `{pattern}`")

        first, stop = sampling.slice_bounds(len(found), start, end)
        window = found[first:stop]
        if num_frames:
            picked = sampling.frame_indices(len(window), num_frames, strategy, nth, seed)
            chosen = [window[index] for index in picked]
        else:
            chosen = window[:MAX_FRAMES]
        require_values(
            chosen,
            f"start {start} and end {end} leave no frame out of the {len(found)} that "
            f"`{pattern}` matched in `{directory}`. Widen the range, or set end to -1 for "
            f"everything from start onwards.",
        )

        images = [
            colour_profile.to_srgb(
                ImageOps.exif_transpose(Image.open(name)), os.path.basename(str(name))
            ).convert("RGBA")
            for name in chosen
        ]
        target = cls.target_size(images[0].size, width, height, max_size)
        pad = parse_color(pad_color, FALLBACK_PAD)
        try:
            sized = [
                sizing.as_channels(
                    sizing.fit(image, target[0], target[1], resize_mode, interpolation, align, pad),
                    channels,
                )
                for image in images
            ]
            batched = stack_images(sized)
        except (MemoryError, ArithmeticError, RuntimeError) as short:
            need = len(images) * target[0] * target[1] * 4 * 4 / (1024 ** 3)
            raise ValueError(
                f"{len(images)} frame(s) at {target[0]}x{target[1]} need about {need:.1f} GiB "
                f"as one batch and would not fit ({short}). Take fewer frames with "
                f"num_frames, or set a smaller width and height"
            ) from short

        size, mode = batch_report.describe_images(batched)
        batch_report.publish(
            frames=int(batched.shape[0]),
            slots=len(chosen),
            size=size,
            mode=mode,
            memory=batch_report.memory_of(batched),
        )
        return io.NodeOutput(
            batched,
            int(batched.shape[0]),
            "\n".join(os.path.basename(name) for name in chosen),
            [batched[index : index + 1] for index in range(int(batched.shape[0]))],
            [os.path.splitext(os.path.basename(name))[0] for name in chosen],
        )

    @staticmethod
    def target_size(source, width: int, height: int, cap: int) -> tuple[int, int]:
        """The size every frame is brought to.

        Args:
            source: ``(width, height)`` of the first frame kept.
            width: Requested width, 0 to take it from the frame.
            height: Requested height, 0 to take it from the frame.
            cap: Longest edge the derived size is held to, 0 for none. Read only where both
                sides were derived, since an explicit size is what was asked for.

        Returns:
            ``(width, height)``, never below 1 on either side.
        """
        wide, high = int(source[0]), int(source[1])
        if width and height:
            return max(1, int(width)), max(1, int(height))
        if width:
            return max(1, int(width)), max(1, round(high * width / wide))
        if height:
            return max(1, round(wide * height / high)), max(1, int(height))
        if cap and max(wide, high) > cap:
            scale = cap / max(wide, high)
            return max(1, round(wide * scale)), max(1, round(high * scale))
        return max(1, wide), max(1, high)

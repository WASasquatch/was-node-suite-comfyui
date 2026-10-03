"""Cross-fade a folder of images into an animated GIF or APNG."""

from __future__ import annotations

import os

from comfy_api.latest import io

from ...modules.io import rooted
from ...modules import log
from ...modules.constants import ALLOWED_EXT
from ...modules.util import sandbox

logger = log.get_logger("nodes.animation")


#: Bytes of decoded frame the animation may hold before the run is refused. The frames are
#: built as a list and quantised into a second one, so the peak is about twice this.
FRAME_BUDGET = 3 << 30

#: Bytes one pixel of a frame occupies while the animation is assembled.
BYTES_PER_PIXEL = 3


def measure(paths, steps: int, max_size: int) -> tuple[int, int, int, int]:
    """Read the size of every source without decoding it, and cost the animation.

    Args:
        paths: Source image paths, in order.
        steps: Blended frames drawn between each consecutive pair.
        max_size: Longest edge the canvas is held to.

    Returns:
        ``(frames, width, height, bytes)``: how many frames the animation holds, the
        canvas every one of them is padded to, and what they occupy together.
    """
    from PIL import Image

    from ...modules.media.gif import canvas_size

    width = height = 0
    for path in paths:
        # Opening reads the header only, so this costs nothing per file.
        with Image.open(path) as probe:
            width = max(width, probe.size[0])
            height = max(height, probe.size[1])
    width, height = canvas_size(width, height, int(max_size))
    frames = len(paths) * (1 + max(steps, 0)) + 1
    return frames, width, height, frames * width * height * BYTES_PER_PIXEL



def image_paths(directory, pattern: str) -> list:
    """The image files in one directory whose names match a glob pattern.

    Args:
        directory: Directory to read. Its subdirectories are not searched.
        pattern: Glob pattern matched against the names in it, such as ``"*.png"``.

    Returns:
        The matching paths, in path order. A file whose extension is not in
        ``ALLOWED_EXT`` is skipped. Nothing is opened.

    Raises:
        PathNotAllowed: The pattern names another machine, carries a drive, starts at a
            root or climbs out with ``..``, or a match resolved outside every readable root.
    """
    found = []
    for name in sorted(sandbox.glob_read(directory, pattern, recursive=False)):
        if not name.lower().endswith(ALLOWED_EXT):
            continue
        found.append(sandbox.resolve_read(name))
    return found


def load_images(paths) -> list:
    """Open image files as RGB.

    Args:
        paths: Files to open, in the order they are returned.

    Returns:
        One ``RGB`` image per path.
    """
    from PIL import Image

    from ...modules.image import colour_profile

    images = []
    for path in paths:
        with Image.open(path) as opened:
            corrected = colour_profile.to_srgb(opened, os.path.basename(str(path)))
            images.append(corrected.convert("RGB"))
    return images


class CreateMorphImageFromPath(io.ComfyNode):
    """Blend a folder of images into one animation and write it to disk."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Create Morph Image from Path",
            display_name="Create Morph Image from Path",
            search_aliases=[
                "Create Morph Image from Path",
                "morph folder",
                "animated gif",
                "apng",
                "slideshow",
            ],
            category="WAS Suite/Animation",
            description=(
                "Fade through every image in a folder and save the result as an animated GIF "
                "or APNG. A few older viewers show an APNG as a single still frame. `WEBP` "
                "gives full colour and alpha at a fraction of the size, and `WEBP_LOSSLESS` "
                "keeps every pixel exact for a larger file that is still usually smaller than "
                "the .gif. With palette_mode `per_frame` each frame looks its best, but a "
                "still background can shift colour as the palette changes under it, seen as "
                "flicker when the animation plays; `global` holds a still area put at the "
                "cost of some accuracy in any single frame."
            ),
            inputs=[
                io.Int.Input(
                    "transition_frames",
                    default=30,
                    min=2,
                    max=60,
                    step=1,
                    tooltip=(
                        "How many blended frames are drawn between each pair of images. 2 is "
                        "almost a hard cut, 30 is a smooth fade, and every extra frame adds "
                        "to the file size."
                    ),
                ),
                io.Float.Input(
                    "still_image_delay_ms",
                    default=2500.0,
                    min=0.1,
                    max=60000.0,
                    step=0.1,
                    tooltip=(
                        "How long each source image is held before it starts to fade, in "
                        "milliseconds. 2500 holds it for two and a half seconds."
                    ),
                ),
                io.Float.Input(
                    "duration_ms",
                    default=0.1,
                    min=0.1,
                    max=60000.0,
                    step=0.1,
                    tooltip=(
                        "How long each blended frame is shown, in milliseconds. The default "
                        "0.1 asks for the shortest frame the format allows, which most "
                        "players round up to about 10ms; raise it to slow the fade down."
                    ),
                ),
                io.Int.Input(
                    "loops",
                    default=0,
                    min=0,
                    max=100,
                    step=1,
                    tooltip=(
                        "How many times the animation plays before it stops. 0 means play "
                        "forever, which is what most viewers expect from a GIF."
                    ),
                ),
                io.Int.Input(
                    "max_size",
                    default=512,
                    min=128,
                    max=1280,
                    step=1,
                    tooltip=(
                        "Longest edge of the animation, in pixels. The canvas is otherwise as large as the largest source, so 512 holds a 4000x3000 photograph down to 512x384 and cuts the file and the memory with it. A source already smaller than this is left alone."
                    ),
                ),
                io.Combo.Input(
                    "input_root",
                    options=rooted.read_options(),
                    tooltip=(
                        "Which folder the source images are in: ComfyUI's own 'input', "
                        "'output' or 'temp', or any folder added under paths.allow_read in "
                        "config.yaml, listed by its own name. input_folder names the part "
                        "below it."
                    ),
                ),
                io.String.Input(
                    "input_folder",
                    default="",
                    multiline=False,
                    tooltip=(
                        "Folder below the input root holding the frames, such as "
                        "'plates/shot_01'. Empty reads the root itself."
                    ),
                ),
                io.String.Input(
                    "input_pattern",
                    default="*",
                    multiline=False,
                    tooltip=(
                        "Which names in that folder to take. '*' takes every image, "
                        "'frame_*.png' only the PNGs whose name starts with 'frame_'. "
                        "Subfolders are not searched, and files that are not images are "
                        "ignored whatever the pattern says."
                    ),
                ),
                io.Combo.Input(
                    "output_root",
                    options=rooted.options(),
                    tooltip=(
                        "Which folder the animation lands in: ComfyUI's own 'output' or "
                        "'temp', or any folder added under paths.allow_write in config.yaml, "
                        "listed by its own name. output_folder names the part below it."
                    ),
                ),
                io.String.Input(
                    "output_folder",
                    default="",
                    multiline=False,
                    tooltip=(
                        "Folder below the output root the animation is written to, created "
                        "if it is not there. Tokens expand, so '[time(%Y-%m-%d)]' files each "
                        "day's under a dated folder."
                    ),
                ),
                io.String.Input(
                    "filename",
                    default="morph",
                    multiline=False,
                    tooltip=(
                        "Name of the file, without an extension, the format below adds one. "
                        "Tokens are expanded here too, so '[time(%H-%M-%S)]_morph' gives every "
                        "run its own file instead of overwriting the last one."
                    ),
                ),
                io.Combo.Input(
                    "filetype",
                    options=["GIF", "APNG", "WEBP", "WEBP_LOSSLESS"],
                    tooltip=(
                        "Which format to write. `GIF` plays anywhere but bands gradients at "
                        "256 colours; `APNG`, `WEBP` and `WEBP_LOSSLESS` keep full colour and "
                        "alpha."
                    ),
                ),
                io.Combo.Input(
                    "palette_mode",
                    options=["per_frame", "global"],
                    tooltip=(
                        "How the palette is chosen when colours are reduced. `per_frame` fits "
                        "one per frame, `global` one to the whole animation, so a still area "
                        "stays put."
                    ),
                ),
                io.Int.Input(
                    "max_colors",
                    default=256,
                    min=2,
                    max=256,
                    step=1,
                    tooltip=(
                        "How many colours the palette holds. 256 is the most a .gif can "
                        "carry. Lower values shrink the file and give a flatter, posterised "
                        "look, and below about 64 the choice of palette_mode starts to show. "
                        "The full-colour formats ignore this at 256 and honour it below that, "
                        "so it doubles as a posterise control for them."
                    ),
                ),
                io.Boolean.Input(
                    "dither",
                    default=True,
                    tooltip=(
                        "Scatter the rounding error between neighbouring pixels so a "
                        "gradient stays smooth instead of breaking into bands. Turn it off "
                        "for flat, poster-like colour, which also compresses smaller."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="filepath_text",
                    tooltip=(
                        "Full path of the file that was written, extension included. Empty "
                        "when the input folder held no images."
                    ),
                ),
                io.String.Output(
                    display_name="filename_text",
                    tooltip=(
                        "The filename widget as it was typed, with no folder and no "
                        "extension. Empty when nothing was written."
                    ),
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> float:
        """NaN never equals itself, so the folder is re-read on every prompt."""
        return float("NaN")

    @classmethod
    def execute(
        cls,
        transition_frames=30,
        still_image_delay_ms=2500.0,
        duration_ms=0.1,
        loops=0,
        max_size=512,
        input_root=rooted.READ_DEFAULT,
        input_folder="",
        input_pattern="*",
        output_root=rooted.DEFAULT,
        output_folder="",
        filename="morph",
        filetype="GIF",
        palette_mode="per_frame",
        max_colors=256,
        dither=True,
    ) -> io.NodeOutput:
        from ...modules.media import gif

        source = rooted.source(input_root, input_folder)
        if not source.exists():
            logger.error("the folder `%s` does not exist", source)
            return io.NodeOutput("", "")

        paths = image_paths(source, input_pattern)
        if not paths:
            logger.error(
                "`%s` holds no image matching `%s`. The readable formats are: %s",
                source, input_pattern, ", ".join(sorted(ALLOWED_EXT)),
            )
            return io.NodeOutput("", "")

        if filetype not in gif.FORMATS:
            filetype = gif.DEFAULT_FORMAT

        transition_frames = min(max(transition_frames, 2), 60)
        count, width, height, cost = measure(paths, int(transition_frames), int(max_size))
        if cost > FRAME_BUDGET:
            raise ValueError(
                f"`{input_pattern}` matches {len(paths)} image(s) in {source}, which at "
                f"{transition_frames} transition_frames is {count} frames of "
                f"{width}x{height}, about {cost / (1 << 30):.1f} GB of picture held at once "
                f"and roughly twice that while the palette is fitted. That is more memory "
                f"than this refuses to take, so nothing was read.\n"
                f"    Narrow input_pattern so it matches fewer files, lower "
                f"transition_frames or max_size, or point input_folder at a folder holding "
                f"only the images meant for this animation."
            )

        images = load_images(paths)
        duration_ms = min(max(duration_ms, 0.1), 60000.0)
        target = sandbox.resolve_write_file(
            rooted.destination(output_root, output_folder), filename
        )
        os.makedirs(target.parent, exist_ok=True)

        # morph_images takes a directory and a bare name and resolves the two together, so
        # the resolved destination is split back into those two arguments.
        output_file = gif.morph_images(
            images,
            steps=int(transition_frames),
            max_size=int(max_size),
            loop=int(loops),
            still_duration=int(still_image_delay_ms),
            duration=int(duration_ms),
            output_path=str(target.parent),
            filename=target.name,
            filetype=filetype,
            palette_mode=palette_mode,
            max_colors=int(max_colors),
            dither=bool(dither),
        )

        if output_file is None or not os.path.isfile(output_file):
            raise RuntimeError(
                f"no animation was written for {target.name}; the encoder's own reason is "
                f"logged above this line"
            )

        logger.info("morph animation written to %s from %s image(s)", output_file, len(images))
        return io.NodeOutput(output_file, filename)

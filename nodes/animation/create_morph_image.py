"""Cross-fade two images into an animated GIF or APNG."""

from __future__ import annotations

import os

from comfy_api.latest import io

from ...modules.io import rooted
from ...modules import log
from ...modules.convert.tensors import broadcast_image_planes, tensor2pil
from ...modules.interface import file_report
from ...modules.util import sandbox

logger = log.get_logger("nodes.animation")


class CreateMorphImage(io.ComfyNode):
    """Blend two images into an animation and write it to disk."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Create Morph Image",
            display_name="Create Morph Image",
            search_aliases=["Create Morph Image", "morph", "animated gif", "apng", "cross-fade"],
            category="WAS Suite/Animation",
            description=(
                "Fade one image into another and save the result as an animated GIF, APNG or "
                "WEBP. Both images pass through unchanged, with the path of the written file. "
                "A batch on image_a makes one a-to-b pair per frame, all in one animation."
            ),
            inputs=[
                io.Image.Input(
                    "image_a",
                    tooltip=(
                        "The image the animation starts on, and returns to at the end. A "
                        "batch here becomes one a-to-b pair per frame, all in one animation."
                    ),
                ),
                io.Image.Input(
                    "image_b",
                    tooltip=(
                        "The image the animation fades into. It does not have to match "
                        "image_a's size: both are letterboxed onto a canvas big enough for "
                        "the larger of the two. A single image here fades in from every "
                        "frame of a batched image_a."
                    ),
                ),
                io.Int.Input(
                    "transition_frames",
                    default=30,
                    min=2,
                    max=60,
                    step=1,
                    tooltip=(
                        "How many blended frames are drawn between the two images. 2 is "
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
                        "How long each of the two images is held before it starts to fade, "
                        "in milliseconds. 2500 holds it for two and a half seconds."
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
                        "Longest edge of the animation, in pixels: 512 holds a 4000x3000 "
                        "photograph down to 512x384, cutting file size and memory. A source "
                        "already smaller than this is left alone."
                    ),
                ),
                io.Combo.Input(
                    "root",
                    options=rooted.options(),
                    tooltip=(
                        "Which folder the file lands in: ComfyUI's own 'output' or 'temp', "
                        "or any folder added under paths.allow_write in config.yaml, listed "
                        "by its own name. filename names the part below it, so "
                        "'[time(%Y-%m-%d)]/clip' files each day's under a dated folder."
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
                        "`GIF` = plays anywhere, bands gradients at 256 colours; `APNG` = full "
                        "colour and alpha; `WEBP` = full colour and alpha, far smaller; "
                        "`WEBP_LOSSLESS` = every pixel exact."
                    ),
                ),
                io.Combo.Input(
                    "palette_mode",
                    options=["per_frame", "global"],
                    tooltip=(
                        "How the palette is chosen when colours are reduced: `per_frame` = best "
                        "colour in each frame, a still background can flicker; `global` = one "
                        "palette for the whole animation, so a still area stays put."
                    ),
                ),
                io.Int.Input(
                    "max_colors",
                    default=256,
                    min=2,
                    max=256,
                    step=1,
                    tooltip=(
                        "Colours in the palette: 256 = the most a .gif holds; lower = smaller "
                        "file and a flatter, posterised look, with palette_mode showing below "
                        "about 64. The full-colour formats ignore 256 and posterise below it."
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
                io.Image.Output(
                    display_name="image_a_pass",
                    tooltip="image_a exactly as it arrived, so this node can sit mid-chain.",
                ),
                io.Image.Output(
                    display_name="image_b_pass",
                    tooltip="image_b exactly as it arrived.",
                ),
                io.String.Output(
                    display_name="filepath_text",
                    tooltip="Full path of the file that was written, extension included.",
                ),
                io.String.Output(
                    display_name="filename_text",
                    tooltip=(
                        "The filename widget as it was typed, with no folder and no "
                        "extension, for feeding a caption or a log."
                    ),
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> float:
        """NaN never equals itself, so the animation is rewritten on every prompt."""
        return float("NaN")

    @classmethod
    def execute(
        cls,
        image_a,
        image_b,
        transition_frames=30,
        still_image_delay_ms=2500.0,
        duration_ms=0.1,
        loops=0,
        max_size=512,
        root=rooted.DEFAULT,
        filename="morph",
        filetype="GIF",
        palette_mode="per_frame",
        max_colors=256,
        dither=True,
    ) -> io.NodeOutput:
        from ...modules.media import gif

        if filetype not in gif.FORMATS:
            filetype = gif.DEFAULT_FORMAT

        transition_frames = min(max(transition_frames, 2), 60)
        duration_ms = min(max(duration_ms, 0.1), 60000.0)
        below, _, leaf = (filename or "").replace("\\", "/").rpartition("/")
        target = sandbox.resolve_write_file(rooted.destination(root, below), leaf)
        os.makedirs(target.parent, exist_ok=True)

        # The pairs are laid end to end in one animation rather than one file each, so four
        # images against four give a to b to a to b through all four pairs.
        sequence = [
            tensor2pil(plane)
            for pair in broadcast_image_planes(image_a, image_b)
            for plane in pair
        ]

        # morph_images takes a directory and a bare name and resolves the two together, so
        # the resolved destination is split back into those two arguments.
        output_file = gif.morph_images(
            sequence,
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

        logger.info("morph animation written to %s", output_file)
        file_report.publish(
            [str(output_file)],
            kind=filetype,
            facts={"frames": f"{len(sequence)} still(s)", "palette": palette_mode},
        )
        return io.NodeOutput(image_a, image_b, output_file, filename)

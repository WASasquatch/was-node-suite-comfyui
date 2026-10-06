"""Saving a finished MiniMax H3 clip and its extend loop records to a numbered file."""

from __future__ import annotations

import os

from comfy_api.latest import io

from ...modules import log
from ...modules.interface import file_report
from ...modules.io import naming, rooted
from ...modules.latent import h3_clip_file
from ...modules.util import sandbox

logger = log.get_logger("nodes.h3_save_clip")

NODE_NAME = "H3 Save Clip"

#: What separates the name from the number, and the digits the number is padded to.
DELIMITER = "_"
PADDING = 5

#: The file name used where ``name`` ends in a folder.
DEFAULT_LEAF = "clip"

ROOT_HINT = (
    "Which folder the file lands in: 'output' = kept until deleted; 'temp' = cleared when "
    "ComfyUI restarts; or a folder added under paths.allow_write in config.yaml. H3 Load Clip "
    "lists input, output, temp and paths.allow_read."
)

NAME_HINT = (
    "The file's name below root, numbered on each run, as `h3_clips/episode` for "
    "`h3_clips/episode_00001.h3clip`. Tokens expand, so `[time(%Y-%m-%d)]/clip` dates the folder."
)


class H3SaveClip(io.ComfyNode):
    """Write an H3 clip and its scene records to one numbered file."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASH3SaveClip",
            display_name=NODE_NAME,
            search_aliases=[
                "WASH3SaveClip", NODE_NAME,
                "save h3 latent",
                "save clip",
                "resume extend loop",
                "minimax h3 save",
            ],
            category="WAS Suite/Latent/Video",
            description=(
                "Save a finished MiniMax H3 clip to a file: its picture and sound, where each "
                "scene ends, and the frames each cut trimmed, all kept exactly. H3 Load Clip "
                "reads it back whole or as it stood after any scene, so a later scene can be "
                "rendered again without rendering the scenes before it."
            ),
            inputs=[
                io.Latent.Input(
                    "latent",
                    tooltip=(
                        "The clip to keep, from H3 Extend Append or the value While Loop Close "
                        "hands out once the loop finishes."
                    ),
                ),
                io.Combo.Input(
                    "root",
                    options=rooted.options(),
                    default=rooted.OUTPUT,
                    tooltip=ROOT_HINT,
                ),
                io.String.Input("name", default="h3_clips/clip", tooltip=NAME_HINT),
            ],
            outputs=[
                io.Latent.Output(
                    display_name="latent",
                    tooltip="The same clip, unchanged, for a decode such as H3 Decode Video.",
                ),
                io.String.Output(
                    display_name="path",
                    tooltip="Full path of the file written, ending in `h3_clips/clip_00001.h3clip`.",
                ),
                io.String.Output(
                    display_name="report",
                    tooltip="Scenes, frames and seconds saved, and the frames each scene spans.",
                ),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, latent, root=rooted.OUTPUT, name="h3_clips/clip") -> io.NodeOutput:
        """Write the clip and report what it holds.

        Raises:
            ValueError: The latent is not an H3 joint latent, or root is not an offered folder.
            PathNotAllowed: root and name settle outside every permitted write folder.
        """
        facts = h3_clip_file.summary(latent)
        below, _, leaf = (name or "").replace("\\", "/").rpartition("/")
        parent = rooted.destination(root, below)
        os.makedirs(parent, exist_ok=True)
        file_name = naming.next_name(
            str(parent), leaf.strip() or DEFAULT_LEAF, DELIMITER, PADDING, h3_clip_file.EXTENSION,
        )
        target = str(sandbox.resolve_write_file(parent, file_name))
        skipped = h3_clip_file.write(latent, target)

        lines = [
            f"saved {facts['scenes']} scene(s), {facts['frames']} frames "
            f"({facts['seconds']:.2f}s) to {file_name}",
            *h3_clip_file.scene_lines(facts["scene_ends"]),
        ]
        if skipped:
            lines.append(f"left out, as no file can hold them: {', '.join(skipped)}")
            logger.warning("%s left out %s from %s", NODE_NAME, ", ".join(skipped), target)
        logger.info("saved an H3 clip of %d scene(s) to %s", facts["scenes"], target)
        file_report.publish(
            [target],
            intended=1,
            kind=h3_clip_file.EXTENSION.lstrip("."),
            folder=str(parent),
            facts={
                "scenes": facts["scenes"],
                "frames": facts["frames"],
                "seconds": f"{facts['seconds']:.2f}",
            },
        )
        return io.NodeOutput(latent, target, "\n".join(lines))

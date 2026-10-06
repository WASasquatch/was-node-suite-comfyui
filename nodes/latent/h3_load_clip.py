"""Reading a saved MiniMax H3 clip back, whole or as it stood after one scene."""

from __future__ import annotations

import os

from comfy_api.latest import io

from ...modules import log
from ...modules.archive import picks
from ...modules.interface import run_result
from ...modules.latent import h3_clip_file
from ...modules.util import file_listing, sandbox

logger = log.get_logger("nodes.h3_load_clip")

NODE_NAME = "H3 Load Clip"

#: How many saved clips the menu offers.
MAX_OPTIONS = 500

#: The menu's only entry while no saved clip exists.
NO_CLIPS = "no saved clips found"

#: Most scenes ``up_to_scene`` takes.
MAX_SCENES = 999

CLIP_HINT = (
    "Which saved clip to read, as `h3_clips/clip_00001.h3clip [output]`, written by H3 Save "
    "Clip. A clip saved to temp is gone after a restart."
)

UP_TO_HINT = (
    "`0` = the whole clip; `2` = the clip as it stood when scene 2 finished, every later "
    "scene dropped, so the loop renders scene 3 onward again."
)

NEXT_HINT = (
    "Loop index of the first scene still to render, from `0`: `2` after up_to_scene `2`, "
    "the scene count for the whole clip. Wire into For Loop Open's start, or add it to a "
    "While Loop Open's index."
)


def options() -> list[str]:
    """The menu's entries, or ``[NO_CLIPS]`` when there are none."""
    return list(
        file_listing.labels((h3_clip_file.EXTENSION,), file_listing.ROOTS, MAX_OPTIONS)
    ) or [NO_CLIPS]


def _named(label: str) -> str:
    """The file a menu label names, or an empty string where none is chosen or found."""
    text = str(label or "").strip()
    if not text or text == NO_CLIPS:
        return ""
    return file_listing.resolve(text, (h3_clip_file.EXTENSION,), file_listing.ROOTS) or ""


class H3LoadClip(io.ComfyNode):
    """A clip H3 Save Clip wrote, cut back to the end of any of its scenes."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASH3LoadClip",
            display_name=NODE_NAME,
            search_aliases=[
                "WASH3LoadClip", NODE_NAME,
                "load h3 latent",
                "load clip",
                "resume extend loop",
                "minimax h3 load",
            ],
            category="WAS Suite/Latent/Video",
            description=(
                "Read a MiniMax H3 clip H3 Save Clip wrote, whole or as it stood when any "
                "scene finished, with every record the extend loop keeps. Hand it to the loop "
                "as its starting clip and start the loop at next_index, and only the scenes "
                "after it are rendered again: change scene 3's prompt and run from scene 3 "
                "without sampling scenes 1 and 2."
            ),
            inputs=[
                io.Combo.Input("clip_file", options=options(), tooltip=CLIP_HINT),
                io.Int.Input(
                    "up_to_scene",
                    default=0,
                    min=0,
                    max=MAX_SCENES,
                    tooltip=UP_TO_HINT,
                ),
            ],
            outputs=[
                io.Latent.Output(
                    display_name="latent",
                    tooltip=(
                        "The clip with its scene records, for While Loop Open's or For Loop "
                        "Open's value_1, or for a decode."
                    ),
                ),
                io.Int.Output(display_name="next_index", tooltip=NEXT_HINT),
                io.String.Output(
                    display_name="report",
                    tooltip="Scenes, frames and seconds read, and the frames each scene spans.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, clip_file="", up_to_scene=0):
        """Read again when the file on disk changes."""
        return picks.fingerprint(_named(clip_file), int(up_to_scene))

    @classmethod
    def execute(cls, clip_file="", up_to_scene=0) -> io.NodeOutput:
        """Read the clip and cut it back to the scene asked for.

        Raises:
            ValueError: No clip is chosen, it is gone, it is not a saved clip, or
                up_to_scene is past its last scene.
            PathNotAllowed: The file sits outside every folder this pack may read.
        """
        path = _named(clip_file)
        if not path:
            raise ValueError(
                f"{NODE_NAME} has no clip chosen. Save one with H3 Save Clip, then reload the "
                f"page so it is listed."
            )
        source = str(sandbox.resolve_read(path))
        if not os.path.isfile(source):
            raise ValueError(
                f"{NODE_NAME} cannot find {clip_file}; it was deleted or moved. Pick another clip."
            )
        whole = h3_clip_file.read(source)
        saved = h3_clip_file.summary(whole)
        scene = int(up_to_scene)
        if scene > saved["scenes"]:
            raise ValueError(
                f"{NODE_NAME}'s up_to_scene is {scene} and {os.path.basename(source)} holds "
                f"{saved['scenes']} scene(s). Set it from 1 to {saved['scenes']}, or 0 for the "
                f"whole clip."
            )
        latent, next_index = h3_clip_file.restored(whole, scene)
        kept = h3_clip_file.summary(latent)

        if scene in (0, saved["scenes"]):
            opening = f"read all {kept['scenes']} scene(s)"
            after = f"the loop carries on at index {next_index}, after the last scene"
        else:
            opening = f"read scenes 1 to {scene} of {saved['scenes']}"
            after = f"the loop carries on at index {next_index}, scene {next_index + 1}"
        headline = f"{opening}: {kept['frames']} frames ({kept['seconds']:.2f}s); {after}"
        report = "\n".join([headline, *h3_clip_file.scene_lines(kept["scene_ends"])])
        logger.info("%s read %s up to scene %d", NODE_NAME, source, kept["scenes"])
        run_result.publish(
            summary=headline,
            counts={
                "scenes": kept["scenes"],
                "frames": kept["frames"],
                "next index": next_index,
            },
            facts={
                "file": os.path.basename(source),
                "seconds": f"{kept['seconds']:.2f}",
                "saved": f"{saved['scenes']} scene(s), {saved['frames']} frames",
            },
        )
        return io.NodeOutput(latent, next_index, report)

"""Reading a frame cache back as a clip."""

from __future__ import annotations

import os

from comfy_api.latest import io

from ...modules import log
from ...modules.archive import picks
from ...modules.media import frame_cache
from ...modules.util import file_listing, sandbox

logger = log.get_logger("nodes.load_video_cache")

NODE_NAME = "Load Video Cache"

#: How many caches the menu offers.
MAX_OPTIONS = 500

#: The menu's only entry while no cache exists.
NO_CACHES = "no frame caches found"


def options() -> list[str]:
    """The menu's entries, or ``[NO_CACHES]`` when there are none."""
    return list(
        file_listing.labels((frame_cache.EXTENSION,), file_listing.ROOTS, MAX_OPTIONS)
    ) or [NO_CACHES]


def _named(cache: str) -> str:
    """The manifest path a menu label names, or an empty string where none is chosen."""
    label = str(cache or "").strip()
    if not label or label == NO_CACHES:
        return ""
    return file_listing.resolve(label, (frame_cache.EXTENSION,), file_listing.ROOTS) or label


class LoadVideoCache(io.ComfyNode):
    """A frame cache on disk, answered as a clip."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLoadVideoCache",
            display_name=NODE_NAME,
            search_aliases=[
                "WASLoadVideoCache", NODE_NAME,
                "frame cache",
                "load cached frames",
                "video from disk",
            ],
            category="WAS Suite/IO",
            description=(
                "Pick a frame cache written by Video Cache or H3 Decode Video and use it as a "
                "clip again, to save it in another format or carry on with it after a restart, "
                "without rendering it again."
            ),
            inputs=[
                io.Combo.Input(
                    "cache",
                    options=options(),
                    tooltip=(
                        "Which frame cache to read, as `frame_cache/scene_00001/cache.wasframes "
                        "[output]`. A cache in temp is gone after a restart."
                    ),
                ),
                io.Boolean.Input(
                    "delete_after_save",
                    default=False,
                    tooltip=(
                        "`true` = the cache is deleted once a save has written all of it; "
                        "`false` = it is kept for the next save."
                    ),
                ),
            ],
            outputs=[
                io.Video.Output(
                    display_name="video",
                    tooltip="The cached clip, read from disk by Save Video a frame at a time.",
                ),
                io.Int.Output(display_name="frames", tooltip="Frames the cache holds."),
                io.Float.Output(display_name="frame_rate", tooltip="Its frames per second."),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, cache="", delete_after_save=False):
        """Read again when the cache on disk has changed, and on every queue while it is deleted after saving."""
        if delete_after_save:
            return float("NaN")
        return picks.fingerprint(_named(cache))

    @classmethod
    def execute(cls, cache="", delete_after_save=False) -> io.NodeOutput:
        """Open the cache.

        Raises:
            ValueError: No cache is chosen, or the chosen one is no longer there.
            PathNotAllowed: The cache sits outside every folder this pack may read.
        """
        path = _named(cache)
        if not path:
            raise ValueError(
                f"{NODE_NAME} has no cache chosen. Write one with Video Cache or H3 Decode "
                f"Video, then reload the page so it is listed."
            )
        manifest = sandbox.resolve_read(path)
        if not manifest.is_file():
            raise ValueError(
                f"{NODE_NAME} cannot find {cache}; it was deleted or moved. Pick another cache."
            )
        folder = os.path.dirname(str(manifest))
        video = frame_cache.CachedVideo(folder, delete_after_save=bool(delete_after_save))
        rate = video.get_frame_rate()
        logger.info("opened %d cached frame(s) at %s", video.get_frame_count(), folder)
        return io.NodeOutput(video, video.get_frame_count(), float(rate))

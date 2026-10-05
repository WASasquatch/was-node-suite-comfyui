"""Keeping frames on disk as a clip, one batch at a time."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.io import rooted
from ...modules.media import frame_cache

logger = log.get_logger("nodes.video_cache")

NODE_NAME = "Video Cache"

ROOT_HINT = (
    "Which folder the cache lands in: 'temp' = cleared when ComfyUI restarts; 'output' = "
    "kept until deleted; or a folder added under paths.allow_write in config.yaml."
)

NAME_HINT = (
    "The cache's name below root, numbered on each run, as `frame_cache/scene` for "
    "`frame_cache/scene_00001`."
)

DELETE_HINT = (
    "`true` = the cached frames are deleted once a save has written all of them, and the "
    "node runs again on every queue; `false` = they are kept."
)


class VideoCache(io.ComfyNode):
    """Write frames to a frame cache on disk and answer them as a clip."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASVideoCache",
            display_name=NODE_NAME,
            search_aliases=[
                "WASVideoCache", NODE_NAME,
                "frame cache",
                "cache frames",
                "frames to disk",
                "video from disk",
            ],
            category="WAS Suite/IO",
            description=(
                "Keep frames on disk as a clip instead of holding them in memory, for a long "
                "render that would otherwise fill RAM before it is saved. Wired to its own "
                "output through a loop, it adds each iteration's frames to the same cache. "
                "Save Video writes the clip straight from disk, a frame at a time."
            ),
            inputs=[
                io.Image.Input(
                    "images",
                    tooltip="The frames to keep, added after any already in the cache.",
                ),
                io.Float.Input(
                    "frame_rate", default=24.0, min=1.0, max=240.0, step=0.001,
                    tooltip=(
                        "Frames per second of a new cache: 24 = film; 30 = video. A cache "
                        "being added to keeps its own."
                    ),
                ),
                io.Combo.Input(
                    "root",
                    options=rooted.options(),
                    default=rooted.TEMP,
                    tooltip=ROOT_HINT,
                ),
                io.String.Input("name", default="frame_cache/clip", tooltip=NAME_HINT),
                io.Boolean.Input("delete_after_save", default=False, tooltip=DELETE_HINT),
                io.Audio.Input(
                    "audio",
                    optional=True,
                    tooltip="Sound for the frames, added after any already in the cache.",
                ),
                io.Video.Input(
                    "append_to",
                    optional=True,
                    tooltip=(
                        "A clip from Video Cache, H3 Decode Video or Load Video Cache to add "
                        "these frames to, as a loop's carried value. Left empty, a new cache "
                        "is made."
                    ),
                ),
                io.Combo.Input(
                    "depth",
                    options=list(frame_cache.DEPTHS),
                    default=frame_cache.DEPTHS[0],
                    optional=True,
                    tooltip=(
                        "How frames are kept: 'auto' = 8-bit codes, half floats for a batch "
                        "outside 0 to 1; '16 bit' = half floats always, for a 10-bit save."
                    ),
                ),
            ],
            outputs=[
                io.Video.Output(
                    display_name="video",
                    tooltip="The cached clip, read from disk by Save Video a frame at a time.",
                ),
                io.Int.Output(
                    display_name="frames",
                    tooltip="Frames the cache holds now.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, delete_after_save=False, **inputs):
        """Run on every queue while the frames are deleted after saving."""
        return float("NaN") if delete_after_save else ""

    @classmethod
    def execute(
        cls, images, frame_rate=24.0, root=rooted.TEMP, name="frame_cache/clip",
        delete_after_save=False, audio=None, append_to=None, depth="auto",
    ) -> io.NodeOutput:
        """Write the frames, and the sound, to a new cache or after a cached clip.

        Raises:
            ValueError: append_to is not a cached clip, or the frames differ from it in size.
            PathNotAllowed: root and name settle outside every permitted write folder.
        """
        if append_to is not None:
            if not frame_cache.cached(append_to):
                raise ValueError(
                    f"{NODE_NAME}'s append_to takes a clip from Video Cache, H3 Decode Video or "
                    f"Load Video Cache. Wire one of those, or leave append_to empty for a new cache."
                )
            cache = append_to.cache()
        else:
            below, _, leaf = (name or "").replace("\\", "/").rpartition("/")
            parent = rooted.destination(root, below)
            cache = frame_cache.FrameCache.create(
                str(parent), leaf.strip() or "clip", float(frame_rate), depth=str(depth),
            )
        first = cache.append(images)
        cache.add_audio(audio)
        logger.info("cached frames %d to %d in %s", first, cache.frames - 1, cache.folder)
        video = frame_cache.CachedVideo(cache.folder, 0, cache.frames, bool(delete_after_save))
        return io.NodeOutput(video, cache.frames)

"""Decoding a joined MiniMax H3 clip scene by scene into a frame cache on disk."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.io import rooted

logger = log.get_logger("nodes.h3_decode_video")

NODE_NAME = "H3 Decode Video"


class H3DecodeVideo(io.ComfyNode):
    """Decode an H3 clip one scene at a time into a frame cache, and answer it as a clip."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASH3DecodeVideo",
            display_name=NODE_NAME,
            search_aliases=[
                "WASH3DecodeVideo", NODE_NAME,
                "h3 decode",
                "decode long video",
                "decode scenes",
                "minimax decode",
            ],
            category="WAS Suite/Latent/Video",
            description=(
                "Decode a MiniMax H3 clip built by an extend loop one scene at a time, writing "
                "the frames to a cache on disk, so a long multi-scene run never has to fit in "
                "memory as one batch. Each scene that opens on a cut is decoded on its own, so "
                "its first frames carry nothing of the scene before. Save Video writes the "
                "clip straight from disk."
            ),
            inputs=[
                io.Latent.Input(
                    "latent",
                    tooltip="The finished clip, from the loop H3 Extend Append builds.",
                ),
                io.Vae.Input("vae", tooltip="The H3 video VAE."),
                io.Combo.Input(
                    "root",
                    options=rooted.options(),
                    default=rooted.TEMP,
                    tooltip=(
                        "Which folder the cache lands in: 'temp' = cleared when ComfyUI "
                        "restarts; 'output' = kept until deleted, for Load Video Cache."
                    ),
                ),
                io.String.Input(
                    "name",
                    default="frame_cache/h3",
                    tooltip=(
                        "The cache's name below root, numbered on each run, as "
                        "`frame_cache/episode` for `frame_cache/episode_00001`."
                    ),
                ),
                io.Boolean.Input(
                    "delete_after_save",
                    default=False,
                    tooltip=(
                        "`true` = the cached frames are deleted once a save has written all of "
                        "them, and the decode runs again on every queue; `false` = they are kept."
                    ),
                ),
                io.Vae.Input(
                    "audio_vae",
                    optional=True,
                    tooltip="The H3 audio VAE, for the clip's sound. Left empty, the clip is silent.",
                ),
            ],
            outputs=[
                io.Video.Output(
                    display_name="video",
                    tooltip="The decoded clip at 24 fps, read from disk by Save Video a frame at a time.",
                ),
                io.Int.Output(display_name="frames", tooltip="Frames decoded."),
                io.String.Output(
                    display_name="report",
                    tooltip="Each scene's frames and where it opens in the clip.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, delete_after_save=False, **inputs):
        """Run on every queue while the frames are deleted after saving."""
        return float("NaN") if delete_after_save else ""

    @classmethod
    def execute(
        cls, latent, vae, root=rooted.TEMP, name="frame_cache/h3", delete_after_save=False,
        audio_vae=None,
    ) -> io.NodeOutput:
        """Decode every scene into a new cache and lay the sound under it.

        Raises:
            ValueError: The latent is not an H3 video and audio latent.
            PathNotAllowed: root and name settle outside every permitted write folder.
        """
        import comfy.model_management

        from ...modules.latent import h3_decode, h3_extend
        from ...modules.media import clip as clips
        from ...modules.media import frame_cache

        video, audio = h3_extend.split(latent)
        spans = h3_decode.scenes(latent)
        below, _, leaf = (name or "").replace("\\", "/").rpartition("/")
        parent = rooted.destination(root, below)
        cache = frame_cache.FrameCache.create(str(parent), leaf.strip() or "h3", h3_extend.FPS)
        pieces = sum(
            -(-(stop - start) // (h3_decode.PIECE_CLIPS * h3_extend.CLIP_TOKENS)) for start, stop in spans
        )
        step = clips.progress(pieces + (1 if audio_vae is not None else 0))
        device = comfy.model_management.get_torch_device()
        lines = []
        try:
            for number, (start, stop) in enumerate(spans, 1):
                opening = cache.frames
                fresh = [True]

                def emit(frames):
                    cache.append(frames, scene=fresh[0], device=device)
                    fresh[0] = False
                    step(1)

                h3_decode.decode_scene(vae, video, start, stop, emit)
                lines.append(
                    f"scene {number}: frames {opening} to {cache.frames - 1} "
                    f"({cache.frames - opening} frames, tokens {start} to {stop - 1})"
                )
            if audio_vae is not None:
                cache.add_audio(h3_decode.decode_audio(audio_vae, audio))
                step(1)
        except BaseException:
            cache.delete()
            raise
        report = f"{cache.frames} frames in {len(spans)} scene(s)\n" + "\n".join(lines)
        logger.info("decoded %d frame(s) in %d scene(s) into %s", cache.frames, len(spans), cache.folder)
        clip = frame_cache.CachedVideo(cache.folder, 0, cache.frames, bool(delete_after_save))
        return io.NodeOutput(clip, cache.frames, report)

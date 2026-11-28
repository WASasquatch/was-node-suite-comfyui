"""Scopes along a whole clip: measured once, played on the node, and drawn as a video when wired."""

from __future__ import annotations

import io as stdio
import json
import time

from comfy_api.latest import io

from ....modules import log
from ....modules.image import clip_scopes

logger = log.get_logger("nodes.video_scopes")

NODE_NAME = "Video Scopes"

#: Sizes the analysis video is drawn at.
SIZES = {"720p": (1280, 720), "1080p": (1920, 1080), "1440p": (2560, 1440), "4K": (3840, 2160)}

#: Frames drawn and written at once.
DRAW_CHUNK = 16


def _png(array) -> bytes:
    from PIL import Image

    buffer = stdio.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG", optimize=False, compress_level=6)
    return buffer.getvalue()


class VideoScopes(io.ComfyNode):
    """Measure a clip's scopes frame by frame and play them in step with it."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        template = io.MatchType.Template("video_scopes_source", allowed_types=[io.Video, io.Image])
        return io.Schema(
            node_id="WASVideoScopes",
            display_name=NODE_NAME,
            search_aliases=[
                "WASVideoScopes", NODE_NAME,
                "scopes", "video histogram", "waveform", "vectorscope", "rgb parade", "spectrogram",
                "exposure over time", "clip analysis",
            ],
            category="WAS Suite/Image/Analyze",
            is_output_node=True,
            description=(
                "Measure a video or a batch of frames frame by frame and play its histogram, waveform, "
                "RGB parade, vectorscope and sound spectrogram on the node in step with the clip, over "
                "a timeline of exposure, clipping and saturation. Wire the scopes output to keep them "
                "as a video."
            ),
            inputs=[
                io.MatchType.Input(
                    "source",
                    template=template,
                    tooltip=(
                        "The clip: a VIDEO, which keeps its frame rate and sound, or IMAGE frames. "
                        "It comes back out unchanged."
                    ),
                ),
                io.Float.Input(
                    "fps", default=24.0, min=1.0, max=240.0, step=0.001, advanced=True,
                    tooltip="Frames per second given to IMAGE frames: `24` = film. A VIDEO keeps its own.",
                ),
                io.Combo.Input(
                    "video_size", options=list(SIZES), default="1080p", advanced=True,
                    tooltip="Size of the scopes video: `720p` = 1280x720; `1080p` = 1920x1080; `4K` = 3840x2160.",
                ),
            ],
            outputs=[
                io.MatchType.Output(
                    template=template, display_name="source",
                    tooltip="The clip that came in, unchanged, as the same type.",
                ),
                io.Video.Output(
                    display_name="scopes",
                    tooltip=(
                        "The scopes dashboard as a video, one frame per source frame, with the source's "
                        "sound. Drawn only when wired."
                    ),
                ),
                io.String.Output(
                    display_name="report",
                    tooltip="The whole clip's figures as JSON: mean exposure, saturation, and the frames that clip or crush.",
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, source, fps=24.0, video_size="1080p") -> io.NodeOutput:
        """Measure ``source``, keep the panel's data, and draw the video when it is wired.

        Raises:
            ValueError: The clip holds no frames.
            MemoryError: The clip fits neither in memory nor on a scratch drive.
        """
        import comfy.utils
        import numpy as np
        import torch

        from ....modules.interface.compare_files import KEPT_TYPE, keep_bytes, keep_ui
        from ....modules.io import rooted
        from ....modules.lora import stack
        from ....modules.media import clip as clips
        from ....modules.media import frame_cache, temp_video

        clip = clips.open_clip(source, NODE_NAME, compact=True)
        frames = clip.frames
        count = int(frames.shape[0])
        rate = float(clip.rate) if hasattr(source, "get_components") else float(fps)
        wanted = stack.consumed(cls.hidden.prompt, cls.hidden.unique_id, 1)
        bar = comfy.utils.ProgressBar(count * (2 if wanted else 1))
        began = time.perf_counter()

        measured = clip_scopes.measure(
            lambda start, stop: frames[start:stop], count, rate, clip.audio,
            progress=lambda done, total: bar.update_absolute(done),
        )
        document = clip_scopes.data_document(measured)
        report = json.dumps(document["summary"], indent=1)

        from ....modules.interface.channel import executing_workflow_id

        workflow = executing_workflow_id() or ""

        def kept(data: bytes, suffix: str):
            name = keep_bytes(data, suffix)
            return [{"filename": name, "subfolder": "", "type": KEPT_TYPE, "workflow": workflow}] if name else []

        shown = source if hasattr(source, "get_components") else clips.rebuild(
            clip, frames if frames.dtype != torch.uint8 else frames.float() / 255.0, audio=None, rate=rate)
        written = temp_video.to_temp(shown, "was.video_scopes.source")
        ui = {
            "scope_source": [written] if written else [],
            "scope_data": kept(json.dumps(document, separators=(",", ":")).encode("utf-8"), ".json"),
            "scope_spectrum": kept(_png(measured.spectrum), ".png") if measured.spectrum is not None else [],
        }

        video = None
        if wanted:
            from ....modules.image.clip_scopes_draw import Dashboard

            width, height = SIZES.get(video_size, SIZES["1080p"])
            board = Dashboard(measured, width, height)
            cache = frame_cache.FrameCache.create(
                rooted.destination(rooted.TEMP, "frame_cache"), "video_scopes", rate)
            try:
                for start in range(0, count, DRAW_CHUNK):
                    stop = min(count, start + DRAW_CHUNK)
                    chunk = frames[start:stop][..., :3]
                    drawn = [board.frame(start + offset, chunk[offset]) for offset in range(stop - start)]
                    cache.append(torch.from_numpy(np.stack(drawn)).float() / 255.0)
                    bar.update_absolute(count + stop)
            except BaseException:
                cache.delete()
                raise
            sound = clips.slice_audio(clip.audio, 0, count, clip.rate if hasattr(source, "get_components") else rate)
            if sound is not None:
                cache.add_audio(sound)
            video = frame_cache.CachedVideo(cache.folder, 0, cache.frames)

        logger.info("%s measured %d frame(s) in %.1fs%s", NODE_NAME, count, time.perf_counter() - began,
                    " and drew the scopes video" if wanted else "")
        return io.NodeOutput(source, video, report, ui=keep_ui(ui))

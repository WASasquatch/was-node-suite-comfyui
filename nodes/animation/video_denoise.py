"""Removing noise and grain from a clip by averaging each frame with its neighbours along the motion."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import MOTION_MODEL
from ...modules.image import motion as motion_field
from ...modules.image import temporal_restore

logger = log.get_logger("nodes.video_denoise")

NODE_NAME = "Video Denoise"

#: What both sides of the comparison on the node are written under in the temp folder.
PREFIX = "was.denoise"


class VideoDenoise(io.ComfyNode):
    """Motion-compensated temporal noise reduction."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASVideoDenoise",
            display_name=NODE_NAME,
            search_aliases=[
                "WASVideoDenoise", NODE_NAME,
                "temporal noise reduction",
                "noise reduction",
                "degrain",
                "grain removal",
                "denoise video",
            ],
            category="WAS Suite/Animation",
            is_output_node=True,
            description=(
                "Remove noise and grain from a clip: each frame is averaged with the frames "
                "around it, lined up along the motion, wherever they truly match, and a light "
                "pass within the frame cleans what no neighbour covers. Answers the cleaned "
                "frames, the clip, and a picture of what was removed."
            ),
            inputs=[
                io.MultiType.Input(
                    "video",
                    [io.Video, io.Image],
                    tooltip=(
                        "The clip, as a VIDEO or IMAGE frames in picture codes, two frames or "
                        "more. A VIDEO gives the video output its frame rate and audio."
                    ),
                ),
                io.Int.Input(
                    "neighbours",
                    default=2,
                    min=1,
                    max=5,
                    tooltip=(
                        "Frames averaged on each side of every frame: 1 = fastest; 2 = default; "
                        "3 to 5 = cleaner, slower."
                    ),
                ),
                io.Float.Input(
                    "luma_strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Share of the brightness noise removed: 0 = brightness untouched; "
                        "0.8 = leaves a fine grain; 1.0 = all of it."
                    ),
                ),
                io.Float.Input(
                    "chroma_strength",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Share of the colour noise removed: 0 = colour untouched; 1.0 = all "
                        "of it. Coloured speckle in the shadows is colour noise."
                    ),
                ),
                io.Float.Input(
                    "detail",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How closely the frames must match before they are averaged: 0 = "
                        "smoothest; 0.5 = default; 1 = keeps the finest texture and removes "
                        "less. Raise it when the noise output shows outlines."
                    ),
                ),
                io.Float.Input(
                    "noise_level",
                    default=0.0,
                    min=0.0,
                    max=64.0,
                    step=0.1,
                    tooltip=(
                        "Noise per colour channel, in levels of 255: 0 = measured on every "
                        "frame; 6 = treat the noise as 6 levels. The log states what was "
                        "measured; set it lower to remove less."
                    ),
                ),
                io.Int.Input(
                    "motion_resolution",
                    default=motion_field.MOTION_SIDE,
                    min=0,
                    max=4096,
                    tooltip=(
                        "Long side, in pixels, the motion is measured at: 0 = the clip's own "
                        "size; 512 = fastest; 768 = default; 1280 = small, fine motion."
                    ),
                ),
                MOTION_MODEL.Input(
                    "motion_model",
                    optional=True,
                    tooltip=(
                        "A SEA-RAFT or FlowSeek network from Video Motion Model Loader to line "
                        "the frames up with. Empty = the built-in texture flow, slower and less "
                        "exact under heavy grain."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip="The cleaned frames, the same size and count as the clip, in picture codes.",
                ),
                io.Video.Output(
                    display_name="video",
                    tooltip=(
                        "The cleaned frames as a clip, at the source's frame rate and with its "
                        "audio when it is a VIDEO, otherwise at 24 fps with none."
                    ),
                ),
                io.Image.Output(
                    display_name="noise",
                    tooltip=(
                        "What was removed, magnified 4 times about mid grey: even grain = "
                        "only noise went; outlines = detail went too, so raise detail."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, video, neighbours=2, luma_strength=1.0, chroma_strength=1.0, detail=0.5,
                noise_level=0.0, motion_resolution=motion_field.MOTION_SIDE,
                motion_model=None) -> io.NodeOutput:
        """Denoise the clip and write both sides for the comparison on the node.

        Raises:
            ValueError: The clip holds fewer than two frames, or fewer than three channels.
            MemoryError: Neither free memory nor a scratch drive can hold the result.
        """
        import comfy.model_management

        from ...modules.image import video_denoise
        from ...modules.media import clip as clips

        source = clips.open_clip(video, NODE_NAME)
        frames = source.frames
        count = int(frames.shape[0])
        if count < 2:
            raise ValueError(
                f"{NODE_NAME} averages each frame with its neighbours and the clip holds "
                f"{count}. Connect a clip of two frames or more."
            )
        if int(frames.shape[-1]) < 3:
            raise ValueError(
                f"{NODE_NAME} needs colour frames and these hold {int(frames.shape[-1])} channel(s). "
                f"Convert them to RGB first."
            )
        device = comfy.model_management.get_torch_device()
        flows = temporal_restore.Flows(frames, int(motion_resolution), device, motion_model)
        cleaned, removed, report = video_denoise.denoise(
            frames, flows, radius=int(neighbours), luma=float(luma_strength),
            chroma=float(chroma_strength), detail=float(detail), level=float(noise_level),
            device=device, progress=clips.progress(count), node=NODE_NAME,
        )
        levels = sorted(report.levels)
        logger.info(
            "denoised %d frame(s) with %s at %dx%d: noise %.2f levels (median of frames), "
            "%.2f neighbour(s) averaged on average",
            count, flows.engine, flows.size[1], flows.size[0], levels[len(levels) // 2],
            sum(report.weight) / len(report.weight),
        )
        shown, alpha = cleaned, source.alpha
        if int(cleaned.shape[-1]) > 3:
            shown, alpha = cleaned[..., :3], cleaned[..., 3]
        answer = clips.rebuild(source, shown, alpha=alpha)
        ui = clips.compare(clips.rebuild(source, frames[..., :3], audio=None), answer, PREFIX)
        return io.NodeOutput(cleaned, answer, removed, ui=ui)

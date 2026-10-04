"""Upscaling a clip with a model and holding the added detail steady along its motion."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.sockets import require_input
from ...modules.compat.types import EMA_VFI_MODEL, MOTION_MODEL
from ...modules.image import tiled_upscale as tiling
from ...modules.media import retime

logger = log.get_logger("nodes.video_temporal_upscale")

NODE_NAME = "Video Temporal Upscale"

#: What both sides of the comparison on the node are written under in the temp folder.
PREFIX = "was.temporal_upscale"


class VideoTemporalUpscale(io.ComfyNode):
    """Upscale a clip with a model, its detail held steady, at any frame rate."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASVideoTemporalUpscale",
            display_name=NODE_NAME,
            search_aliases=[
                "WASVideoTemporalUpscale", NODE_NAME,
                "upscale video",
                "video upscale",
                "temporal upscale",
                "deflicker upscale",
                "frame rate",
            ],
            category="WAS Suite/Animation",
            is_output_node=True,
            description=(
                "Upscale a clip with an upscale model without the added detail flickering, "
                "as what the model adds to each frame is carried along the clip's own motion. "
                "It can also change the frame rate, with EMA-VFI drawing new frames where they "
                "fall between two and the audio kept in sync. The clip is written to a file as "
                "it is made, so a long clip at 4x never has to fit in memory."
            ),
            inputs=[
                io.MultiType.Input(
                    "source",
                    [io.Video, io.Image],
                    tooltip=(
                        "The clip to upscale, as a VIDEO or IMAGE frames. A VIDEO gives the "
                        "result its frame rate and audio; frames play at 24 fps with none."
                    ),
                ),
                io.UpscaleModel.Input(
                    "upscale_model",
                    tooltip=(
                        "The upscale model, from Load Upscale Model. A 4x model can make a 2x "
                        "result."
                    ),
                ),
                io.Float.Input(
                    "upscale_factor", default=4.0, min=1.0, max=16.0, step=0.1,
                    tooltip=(
                        "Final size against the source: 2.0 = double; 4.0 = four times. Sides "
                        "are rounded to even numbers of pixels."
                    ),
                ),
                io.Float.Input(
                    "strength", default=0.8, min=0.0, max=1.0, step=0.01,
                    tooltip=(
                        "How much of each frame's detail comes from its neighbours: 0 = each "
                        "frame upscaled on its own; 0.8 = default; 1.0 = holds the detail until "
                        "a surface changes, or on a clip too large to hold at once, until the "
                        "next block of frames."
                    ),
                ),
                io.Float.Input(
                    "frame_rate", default=0.0, min=0.0, max=240.0, step=0.001,
                    tooltip=(
                        "Frames per second of the result: 0 = the source's; 30 = 30 fps. The "
                        "length is kept, so the audio stays in sync."
                    ),
                ),
                io.Int.Input(
                    "tile_size", default=512, min=64, max=4096, step=16,
                    tooltip=(
                        "Tile edge in source pixels: 256 and 512 suit most cards. If the card "
                        "runs out, the tile is halved and the frame retried."
                    ),
                ),
                io.Int.Input(
                    "overlap", default=32, min=0, max=1024, step=1,
                    tooltip=(
                        "How far neighbouring tiles overlap, in source pixels: 0 = hard joins; "
                        "32 to 64 hides them on most models."
                    ),
                ),
                io.Float.Input(
                    "crf", default=16.0, min=0.0, max=51.0, step=0.5,
                    tooltip=(
                        "Quality of the written clip, lower is better and larger: 0 = lossless; "
                        "16 = default; 23 = typical; 28 = small."
                    ),
                ),
                MOTION_MODEL.Input(
                    "motion_model",
                    optional=True,
                    tooltip=(
                        "A flow network from Video Motion Model Loader, which the motion is "
                        "measured with. Left empty, the built-in texture flow measures it."
                    ),
                ),
                EMA_VFI_MODEL.Input(
                    "ema_vfi_model",
                    optional=True,
                    tooltip=(
                        "The interpolation network from EMA-VFI Video Model Loader, which draws "
                        "frames between two when frame_rate changes. Left empty, the nearer "
                        "frame repeats. Any rate but double needs an 'ours_t' checkpoint."
                    ),
                ),
                io.Combo.Input(
                    "precision",
                    options=list(tiling.PRECISIONS),
                    default=tiling.PRECISIONS[0],
                    optional=True,
                    tooltip=(
                        "What the upscale model runs in: 'auto' = half precision where the "
                        "model declares it safe; '32 bit float' = every model's safest."
                    ),
                ),
                io.Combo.Input(
                    "new_frames",
                    options=list(retime.MODES),
                    default="auto",
                    optional=True,
                    tooltip=(
                        "How a frame between two source frames is made when frame_rate changes: "
                        "'auto' = EMA-VFI where the motion accounts for the change, the nearer "
                        "frame where it does not, such as a mouth changing shape; 'interpolate' "
                        "= EMA-VFI always; 'blend' = the two mixed; 'hold' = the nearer frame."
                    ),
                ),
            ],
            outputs=[
                io.Video.Output(
                    display_name="video",
                    tooltip=(
                        "The upscaled clip as an h264 file, with the source's audio. Save Video "
                        "on 'auto' keeps it without encoding it again."
                    ),
                ),
                io.Int.Output(
                    display_name="frames",
                    tooltip="How many frames the upscaled clip holds.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        source,
        upscale_model,
        upscale_factor=4.0,
        strength=0.8,
        frame_rate=0.0,
        tile_size=512,
        overlap=32,
        crf=16.0,
        motion_model=None,
        ema_vfi_model=None,
        precision="auto",
        new_frames="auto",
    ) -> io.NodeOutput:
        """Upscale the clip, hold its detail steady, and write both sides for the player.

        Raises:
            ValueError: Nothing is connected to upscale_model, the clip is HDR, the frame
                rate would make too many frames, or the EMA-VFI checkpoint cannot land where
                the new frames fall.
            MemoryError: Neither free memory nor a scratch drive can hold the detail kept
                between the two passes.
        """
        import os

        import folder_paths
        from comfy_api.latest import InputImpl

        from ...modules.media import clip as clips
        from ...modules.media import temp_video, temporal_upscale
        from ...modules.util import sandbox

        require_input(
            upscale_model, NODE_NAME, "upscale_model", "model", "Load Upscale Model", "UPSCALE_MODEL"
        )
        clip = clips.open_clip(source, NODE_NAME, compact=True)
        height, width = (int(side) for side in clip.frames.shape[1:3])
        factor = max(float(upscale_factor), 1.0)
        temp = folder_paths.get_temp_directory()

        def named(side):
            folder, name, counter, subfolder, _ = sandbox.save_image_path(
                f"{PREFIX}.{side}", temp, int(width * factor), int(height * factor)
            )
            file = f"{name}_{counter:05}_.mp4"
            return os.path.join(folder, file), {"filename": file, "subfolder": subfolder, "type": "temp"}

        target, finished = named("result")
        plain, before = named("before")
        small, after = named("after")
        result = temporal_upscale.render(
            clip,
            upscale_model,
            target,
            before=plain,
            after=small,
            factor=factor,
            strength=float(strength),
            rate=float(frame_rate),
            tile_size=int(tile_size),
            overlap=int(overlap),
            precision=str(precision),
            crf=float(crf),
            motion_model=motion_model,
            ema_vfi_model=ema_vfi_model,
            new_frames=str(new_frames),
            node=NODE_NAME,
        )
        if result.before is None:
            shown = source if hasattr(source, "get_components") else clips.rebuild(clip, clip.frames, audio=None)
            before = temp_video.to_temp(shown, f"{PREFIX}.source")
        if result.after is None:
            after = finished
        logger.info(
            "%d frame(s) at %dx%d, %g fps, tile %d px, %s",
            result.frames, result.size[0], result.size[1], float(result.rate), result.tile,
            result.precision,
        )
        ui = {"a_video": [before] if before else [], "b_video": [after]}
        return io.NodeOutput(InputImpl.VideoFromFile(target), result.frames, ui=ui)

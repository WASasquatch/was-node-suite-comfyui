"""Steadying a clip's camera along the motion measured between its frames."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import MOTION
from ...modules.image import camera_motion

logger = log.get_logger("nodes.video_stabilize")

NODE_NAME = "Video Stabilize"

#: What both sides of the comparison on the node are written under in the temp folder.
PREFIX = "was.stabilize"


class VideoStabilize(io.ComfyNode):
    """Smooth or lock a clip's camera movement."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASVideoStabilize",
            display_name=NODE_NAME,
            search_aliases=[
                "WASVideoStabilize", NODE_NAME,
                "stabilization",
                "stabilise",
                "camera shake",
                "jitter",
                "steady",
                "tripod",
            ],
            category="WAS Suite/Animation",
            is_output_node=True,
            description=(
                "Take the shake out of a clip: the camera's own movement is fitted from the "
                "motion between frames and smoothed, or locked as if on a tripod, and every "
                "frame is moved to follow the steady path. Each scene is steadied on its own."
            ),
            inputs=[
                io.Video.Input("video", tooltip="The clip to steady, at least two frames."),
                io.Combo.Input(
                    "mode",
                    options=list(camera_motion.MODES),
                    default="smooth",
                    tooltip=(
                        "'smooth' keeps the camera's intended move and removes the shake; "
                        "'lock' holds the first frame's framing for the whole scene."
                    ),
                ),
                io.Float.Input(
                    "smoothing",
                    default=1.0,
                    min=0.0,
                    max=30.0,
                    step=0.05,
                    tooltip=(
                        "Seconds of camera movement averaged in 'smooth': 0.25 = removes "
                        "jitter only; 1.0 = default; 3.0 = a slow, floating move."
                    ),
                ),
                io.Combo.Input(
                    "motion_model",
                    options=list(camera_motion.MODELS),
                    default="similarity",
                    tooltip=(
                        "'similarity' steadies sliding, rolling and zooming; 'translation' "
                        "only sliding, for a clip whose rotation is meant."
                    ),
                ),
                io.Combo.Input(
                    "borders",
                    options=list(camera_motion.BORDERS),
                    default="zoom",
                    tooltip=(
                        "What fills the edge a moved frame leaves bare: 'zoom' enlarges until "
                        "none shows, up to max_zoom; 'edge' stretches the last pixels; "
                        "'mirror' reflects; 'black' leaves it black."
                    ),
                ),
                io.Float.Input(
                    "max_zoom",
                    default=1.25,
                    min=1.0,
                    max=3.0,
                    step=0.01,
                    tooltip="Most enlargement 'zoom' may use: 1.0 = none; 1.25 = default; 1.5 = heavy shake.",
                ),
                MOTION.Input(
                    "motion",
                    optional=True,
                    tooltip=(
                        "Motion from Video Motion, measured from this same clip. Left empty, it "
                        "is measured here at 768 px on the long side."
                    ),
                ),
                io.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip=(
                        "Subject mask per frame, white on the subject, such as from SAM 3. The "
                        "subject is left out when the camera's motion is read, so a large moving "
                        "subject cannot pull the frame along with it."
                    ),
                ),
            ],
            outputs=[
                io.Video.Output(
                    display_name="video",
                    tooltip="The steadied clip: same size, length, frame rate and audio.",
                ),
                io.Float.Output(
                    display_name="zoom",
                    tooltip="The enlargement applied to hide the edges, 1.0 for none.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        video,
        mode="smooth",
        smoothing=1.0,
        motion_model="similarity",
        borders="zoom",
        max_zoom=1.25,
        motion=None,
        mask=None,
    ) -> io.NodeOutput:
        """Steady the clip and write both sides for the comparison on the node.

        Raises:
            ValueError: The clip holds fewer than two frames, the motion was measured from
                another clip, or the mask holds a different number of frames than the clip.
        """
        import torch

        import comfy.model_management

        from ...modules.media import clip as clips

        source = clips.open_clip(video, NODE_NAME)
        count, height, width = (int(v) for v in source.frames.shape[:3])
        if count < 2:
            raise ValueError(
                f"{NODE_NAME} follows the camera between frames and the clip holds {count}. "
                f"Connect a clip of two frames or more."
            )
        if mask is not None and int(mask.shape[0]) not in (1, count):
            raise ValueError(
                f"The mask batch holds {int(mask.shape[0])} frame(s) and the clip {count}. "
                f"Connect one mask per frame of the clip, or a single one for all of it."
            )
        device = comfy.model_management.get_torch_device()
        step = clips.progress(3 * count - 2)
        motion = clips.motion_for(source, motion, NODE_NAME, device, step)

        cameras, segments = camera_motion.path(motion, str(motion_model), device, step, mask)
        frames_sigma = max(0.0, float(smoothing)) * float(source.rate)
        fixes = camera_motion.corrections(cameras, segments, str(mode), frames_sigma)
        fixes = [camera_motion.to_frame(fix, motion.size, (height, width)) for fix in fixes]
        zoom = 1.0
        if borders == "zoom":
            zoom = camera_motion.zoom_for(fixes, (height, width), float(max_zoom))

        steadied = torch.empty_like(source.frames)
        alpha = torch.empty_like(source.alpha) if source.alpha is not None else None
        for index in range(count):
            frame = source.frames[index:index + 1].to(device=device, dtype=torch.float32)
            planes = frame.permute(0, 3, 1, 2)
            if alpha is not None:
                planes = torch.cat([planes, source.alpha[index:index + 1].to(device).float().unsqueeze(1)], 1)
            moved = camera_motion.resample(planes, fixes[index], zoom, str(borders))
            steadied[index] = moved[0, :3].permute(1, 2, 0).to(steadied.dtype).to(steadied.device)
            if alpha is not None:
                alpha[index] = moved[0, 3].to(alpha.dtype).to(alpha.device)
            step(1)

        logger.info("steadied %s in %d scene(s), zoom %.3f", motion.describe(), len(segments), zoom)
        answer = clips.rebuild(source, steadied, alpha=alpha)
        return io.NodeOutput(answer, float(zoom), ui=clips.compare(video, answer, PREFIX))

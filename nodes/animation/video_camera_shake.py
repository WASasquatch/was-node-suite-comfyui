"""Handheld camera shake added to a clip, with the blur a real shake leaves."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LIST
from ...modules.image import camera_motion

logger = log.get_logger("nodes.video_camera_shake")

NODE_NAME = "Video Camera Shake"

#: What both sides of the comparison on the node are written under in the temp folder.
PREFIX = "was.camera_shake"

#: Moments across the exposure each blurred frame is averaged from.
BLUR_STEPS = 12


class VideoCameraShake(io.ComfyNode):
    """Shake a clip as a handheld camera would, blur included."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASVideoCameraShake",
            display_name=NODE_NAME,
            search_aliases=[
                "WASVideoCameraShake", NODE_NAME,
                "handheld",
                "shaky cam",
                "camera wobble",
                "impact shake",
                "earthquake",
            ],
            category="WAS Suite/Animation",
            is_output_node=True,
            description=(
                "Shake a clip the way a handheld camera does: sway, roll and a little zoom "
                "breathing, woven from smooth wobbles that settle into the same shake for the "
                "same seed. The shutter angle adds the blur a real shake leaves, and a curve "
                "can ramp it up for an impact. Audio and frame rate carry through."
            ),
            inputs=[
                io.Video.Input("video", tooltip="The clip to shake."),
                io.Float.Input(
                    "amplitude",
                    default=0.01,
                    min=0.0,
                    max=0.2,
                    step=0.001,
                    tooltip=(
                        "Sway as a share of the frame's short side: 0.004 = a steady hand; "
                        "0.01 = handheld; 0.03 = running with the camera."
                    ),
                ),
                io.Float.Input(
                    "rotation",
                    default=0.3,
                    min=0.0,
                    max=20.0,
                    step=0.05,
                    tooltip="Roll in degrees: 0 = level; 0.3 = handheld; 2 = violent.",
                ),
                io.Float.Input(
                    "zoom",
                    default=0.002,
                    min=0.0,
                    max=0.2,
                    step=0.001,
                    tooltip="Zoom breathing as a share of the frame: 0 = none; 0.002 = subtle; 0.02 = pumping.",
                ),
                io.Float.Input(
                    "frequency",
                    default=1.5,
                    min=0.05,
                    max=30.0,
                    step=0.05,
                    tooltip=(
                        "Centre of the shake in hertz: 0.5 = slow drift; 1.5 = handheld; "
                        "6 = engine or impact rattle."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=io.ControlAfterGenerate.fixed,
                    tooltip="Which shake; the same seed always gives the same one. Any whole number, as `7`.",
                ),
                io.Float.Input(
                    "shutter_angle",
                    default=180.0,
                    min=0.0,
                    max=360.0,
                    step=1.0,
                    tooltip=(
                        "Blur the shake leaves, as a film shutter in degrees: 0 = none; 180 = "
                        "the film standard; 360 = the whole frame interval."
                    ),
                ),
                io.Combo.Input(
                    "borders",
                    options=list(camera_motion.BORDERS),
                    default="zoom",
                    tooltip=(
                        "What fills the edge the shaken frame pulls away from: 'zoom' enlarges "
                        "until none shows, up to max_zoom; 'edge', 'mirror' or 'black' fill it."
                    ),
                ),
                io.Float.Input(
                    "max_zoom",
                    default=1.2,
                    min=1.0,
                    max=3.0,
                    step=0.01,
                    tooltip="Most enlargement 'zoom' may use: 1.0 = none; 1.2 = default; 1.5 = heavy shake.",
                ),
                LIST.Input(
                    "shake_curve",
                    optional=True,
                    tooltip=(
                        "Strength across the clip, such as the values of Curve to Numbers: "
                        "[0, 0, 3, 0.5, 0] is calm, then an impact that settles. Stretched to the "
                        "clip's length; multiplies amplitude, rotation and zoom."
                    ),
                ),
            ],
            outputs=[
                io.Video.Output(
                    display_name="video",
                    tooltip="The shaken clip: same size, length, frame rate and audio.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        video,
        amplitude=0.01,
        rotation=0.3,
        zoom=0.002,
        frequency=1.5,
        seed=0,
        shutter_angle=180.0,
        borders="zoom",
        max_zoom=1.2,
        shake_curve=None,
    ) -> io.NodeOutput:
        """Shake the clip and write both sides for the comparison on the node.

        Raises:
            ValueError: The clip holds no frames, or the curve holds something other than
                numbers.
        """
        import torch

        import comfy.model_management

        from ...modules.media import clip as clips

        source = clips.open_clip(video, NODE_NAME)
        count, height, width = (int(v) for v in source.frames.shape[:3])
        strength = [1.0] * count
        if shake_curve is not None:
            try:
                strength = clips.stretch([float(value) for value in shake_curve], count)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{NODE_NAME}'s shake_curve holds something other than numbers. Connect the "
                    f"values of Curve to Numbers, or a list of strengths."
                ) from None

        rate = float(source.rate)
        exposure = max(0.0, float(shutter_angle)) / 360.0 / rate
        steps = BLUR_STEPS if exposure > 0 else 1
        offsets = [0.0] if steps == 1 else [exposure * ((k + 0.5) / steps - 0.5) for k in range(steps)]
        times = [index / rate + offset for index in range(count) for offset in offsets]
        per_moment = [strength[index] for index in range(count) for _ in offsets]
        shakes = camera_motion.shake(
            times, float(min(height, width)), float(amplitude), float(rotation), float(zoom),
            float(frequency), int(seed), per_moment,
        )
        enlarge = 1.0
        if borders == "zoom":
            enlarge = camera_motion.zoom_for(shakes, (height, width), float(max_zoom))

        device = comfy.model_management.get_torch_device()
        step = clips.progress(count)
        shaken = torch.empty_like(source.frames)
        alpha = torch.empty_like(source.alpha) if source.alpha is not None else None
        for index in range(count):
            planes = source.frames[index:index + 1].to(device=device, dtype=torch.float32).permute(0, 3, 1, 2)
            if alpha is not None:
                planes = torch.cat([planes, source.alpha[index:index + 1].to(device).float().unsqueeze(1)], 1)
            moments = shakes[index * steps:(index + 1) * steps]
            total = sum(camera_motion.resample(planes, moment, enlarge, str(borders)) for moment in moments)
            frame = total / len(moments)
            shaken[index] = frame[0, :3].permute(1, 2, 0).to(shaken.dtype).to(shaken.device)
            if alpha is not None:
                alpha[index] = frame[0, 3].to(alpha.dtype).to(alpha.device)
            step(1)

        logger.info("shook %d frame(s) at %.3g Hz, zoom %.3f", count, float(frequency), enlarge)
        answer = clips.rebuild(source, shaken, alpha=alpha)
        return io.NodeOutput(answer, ui=clips.compare(video, answer, PREFIX))

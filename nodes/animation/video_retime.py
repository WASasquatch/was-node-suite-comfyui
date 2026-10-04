"""Changing a clip's speed, with new frames made between the ones it has."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import EMA_VFI_MODEL, LIST, MOTION
from ...modules.media import retime

logger = log.get_logger("nodes.video_retime")

NODE_NAME = "Video Retime"

#: What both sides of the comparison on the node are written under in the temp folder.
PREFIX = "was.retime"

#: What happens to the clip's audio.
AUDIO_MODES = ("keep pitch", "drop")

#: Long side, in pixels, cuts are looked for at when no motion is wired.
CUT_SIDE = 384


class VideoRetime(io.ComfyNode):
    """Slow a clip down or speed it up, at one speed or along a curve."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASVideoRetime",
            display_name=NODE_NAME,
            search_aliases=[
                "WASVideoRetime", NODE_NAME,
                "slow motion",
                "speed ramp",
                "time remap",
                "fast forward",
                "twixtor",
            ],
            category="WAS Suite/Animation",
            is_output_node=True,
            description=(
                "Slow a clip down or speed it up, at one speed or along a speed curve, at the "
                "clip's own frame rate. New frames between the ones it has are drawn by "
                "EMA-VFI, mixed, or held, and never across a cut. The audio follows with its "
                "pitch kept. Follow with Video Motion Blur for the streaks a sped-up shot has."
            ),
            inputs=[
                io.Video.Input("video", tooltip="The clip to retime, at least two frames."),
                io.Float.Input(
                    "speed",
                    default=0.5,
                    min=retime.MIN_SPEED,
                    max=retime.MAX_SPEED,
                    step=0.01,
                    tooltip="0.5 = half speed, twice as long; 1 = unchanged; 2 = double speed, half as long.",
                ),
                io.Combo.Input(
                    "new_frames",
                    options=list(retime.MODES),
                    default="interpolate",
                    tooltip=(
                        "How a frame between two is made: 'interpolate' draws it with EMA-VFI, "
                        "which needs ema_vfi_model; 'blend' mixes the two; 'hold' repeats the "
                        "nearer one; 'auto' draws with EMA-VFI only where the motion accounts "
                        "for the change, such as a body moving rather than a mouth changing shape."
                    ),
                ),
                io.Combo.Input(
                    "audio",
                    options=list(AUDIO_MODES),
                    default="keep pitch",
                    tooltip="'keep pitch' stretches the audio along the new timing at its own pitch; 'drop' leaves it out.",
                ),
                EMA_VFI_MODEL.Input(
                    "ema_vfi_model",
                    optional=True,
                    tooltip=(
                        "The interpolation network from EMA-VFI Video Model Loader, for 'interpolate'. "
                        "Any speed but 0.5 needs an 'ours_t' checkpoint."
                    ),
                ),
                LIST.Input(
                    "speed_curve",
                    optional=True,
                    tooltip=(
                        "Speed across the source clip for a ramp, such as the values of Curve to "
                        "Numbers: [1, 0.25, 1] slows to a quarter mid-clip. Stretched over the "
                        "clip; replaces speed."
                    ),
                ),
                MOTION.Input(
                    "motion",
                    optional=True,
                    tooltip=(
                        "Motion from Video Motion, measured from this same clip at any size, whose cuts are "
                        "kept. Left empty, cuts are found here."
                    ),
                ),
            ],
            outputs=[
                io.Video.Output(
                    display_name="video",
                    tooltip="The retimed clip at the source's frame rate and size.",
                ),
                io.Int.Output(display_name="frames", tooltip="How many frames the retimed clip holds."),
            ],
        )

    @classmethod
    def execute(
        cls,
        video,
        speed=0.5,
        new_frames="interpolate",
        audio="keep pitch",
        ema_vfi_model=None,
        speed_curve=None,
        motion=None,
    ) -> io.NodeOutput:
        """Retime the clip and write both sides for the comparison on the node.

        Raises:
            ValueError: The clip holds fewer than two frames, the speed curve holds something
                other than numbers, 'interpolate' has no network or a checkpoint that cannot
                land between frames as the speed needs, or the retime runs too long.
            MemoryError: Neither free memory nor a scratch drive can hold the retimed clip.
        """
        import comfy.model_management

        from ...modules.image import motion as motion_field
        from ...modules.media import clip as clips

        source = clips.open_clip(video, NODE_NAME)
        count = int(source.frames.shape[0])
        if count < 2:
            raise ValueError(f"{NODE_NAME} needs a clip of two frames or more; this one holds {count}.")
        speeds = [float(speed)]
        if speed_curve is not None:
            try:
                speeds = [float(value) for value in speed_curve]
            except (TypeError, ValueError):
                raise ValueError(
                    f"{NODE_NAME}'s speed_curve holds something other than numbers. Connect the "
                    f"values of Curve to Numbers, or a list of speeds."
                ) from None
            if not speeds:
                raise ValueError(f"{NODE_NAME}'s speed_curve is empty. Connect at least one speed.")
        where = retime.positions(count, speeds)

        net = None
        if new_frames in ("interpolate", "auto"):
            if ema_vfi_model is None:
                raise ValueError(
                    f"{NODE_NAME} draws new frames with EMA-VFI on '{new_frames}'. Wire EMA-VFI "
                    f"Video Model Loader into ema_vfi_model, or choose 'blend' or 'hold'."
                )
            retime.check_network(ema_vfi_model.name, where)
            backend = ema_vfi_model.backend
            backend.load()
            net = backend.model

        device = comfy.model_management.get_torch_device()
        step = clips.progress(count - 1 + len(where))
        if motion is None:
            motion = motion_field.measure(source.frames, CUT_SIDE, device, step)
        else:
            motion = clips.motion_for(source, motion, NODE_NAME, device, step)
        cuts = motion.cuts()
        pairs = retime.drawable(motion) if new_frames == "auto" else None
        alpha = None
        if source.alpha is not None:
            alpha = retime.frames_at(source.alpha.unsqueeze(-1), where, "blend", cuts, name=NODE_NAME)[..., 0]
        drawn_on = next(net.parameters()).device if net is not None else device
        frames = retime.frames_at(
            source.frames, where, str(new_frames), cuts, net, drawn_on, step, NODE_NAME, drawn=pairs
        )
        sound = retime.retimed_audio(source.audio, where, float(source.rate)) if audio == "keep pitch" else None

        logger.info("retimed %d frame(s) to %d", count, len(where))
        answer = clips.rebuild(source, frames, alpha=alpha, audio=sound)
        return io.NodeOutput(answer, len(where), ui=clips.compare(video, answer, PREFIX))

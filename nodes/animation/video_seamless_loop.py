"""Cutting the best loop out of a clip and blending its join."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.media import loop

logger = log.get_logger("nodes.video_seamless_loop")

NODE_NAME = "Video Seamless Loop"

#: What both sides of the comparison on the node are written under in the temp folder.
PREFIX = "was.seamless_loop"


def _crossfaded(audio, start: int, stop: int, blend: int, where, rate):
    """The loop's audio, its join crossfaded where the frames were blended."""
    from ...modules.media import clip as clips

    sound = clips.slice_audio(audio, start, stop, rate)
    if sound is None or where is None or blend <= 0:
        return sound
    if where == "end":
        other = clips.slice_audio(audio, start - blend, start, rate)
        offset = sound["waveform"].shape[-1] - other["waveform"].shape[-1]
    else:
        other = clips.slice_audio(audio, stop, stop + blend, rate)
        offset = 0
    import torch

    span = min(other["waveform"].shape[-1], sound["waveform"].shape[-1] - max(offset, 0))
    if span <= 0:
        return sound
    ramp = torch.linspace(0.0, 1.0, span, dtype=sound["waveform"].dtype)
    if where == "start":
        ramp = 1.0 - ramp
    piece = sound["waveform"][..., offset:offset + span]
    sound["waveform"][..., offset:offset + span] = piece * (1.0 - ramp) + other["waveform"][..., :span] * ramp
    return sound


class VideoSeamlessLoop(io.ComfyNode):
    """Find the stretch of a clip that loops back on itself least visibly."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASVideoSeamlessLoop",
            display_name=NODE_NAME,
            search_aliases=[
                "WASVideoSeamlessLoop", NODE_NAME,
                "loop",
                "seamless loop",
                "boomerang",
                "cinemagraph",
            ],
            category="WAS Suite/Animation",
            is_output_node=True,
            description=(
                "Find the stretch of a clip whose end runs back into its start most closely, "
                "picture and movement both, and blend the join along the motion so it plays as "
                "a loop. The report says how close the two ends were: a clip whose ends never "
                "come near each other loops with a visible join."
            ),
            inputs=[
                io.Video.Input("video", tooltip="The clip to loop."),
                io.Float.Input(
                    "min_seconds",
                    default=2.0,
                    min=0.1,
                    max=600.0,
                    step=0.1,
                    tooltip="Shortest loop in seconds: 1 = a quick cycle; 2 = default; 5 = a long take.",
                ),
                io.Float.Input(
                    "max_seconds",
                    default=0.0,
                    min=0.0,
                    max=600.0,
                    step=0.1,
                    tooltip="Longest loop in seconds: 0 = up to the whole clip; 4 = keeps it under four seconds.",
                ),
                io.Int.Input(
                    "blend_frames",
                    default=8,
                    min=0,
                    max=240,
                    tooltip="Frames blended across the join: 0 = cut straight back; 8 = default; 24 = a long, soft return.",
                ),
                io.Float.Input(
                    "max_difference",
                    default=0.0,
                    min=0.0,
                    max=255.0,
                    step=0.1,
                    tooltip=(
                        "Most the two ends may differ, in levels of 255: 0 = take the closest "
                        "match; 3 = clean; 6 = slight. Above 0, the longest loop within it is "
                        "taken, and the run stops if none is."
                    ),
                ),
                io.Combo.Input(
                    "blend_mode",
                    options=list(loop.BLEND_MODES),
                    default="auto",
                    tooltip=(
                        "'auto' = 'motion' for ends under 8 levels apart, 'crossfade' above; "
                        "'motion' carries the frames into each other along their movement where it "
                        "can be followed; 'crossfade' dissolves, ghosting rather than tearing."
                    ),
                ),
            ],
            outputs=[
                io.Video.Output(display_name="video", tooltip="The loop, its join blended, with its audio."),
                io.Int.Output(display_name="start", tooltip="The source frame the loop starts on, counting from 0."),
                io.Int.Output(display_name="length", tooltip="Frames in the loop."),
                io.Float.Output(
                    display_name="join_difference",
                    tooltip="Mean difference between the two ends before blending, in levels of 255: under 3 is clean, over 8 shows.",
                ),
                io.String.Output(display_name="report", tooltip="The loop and how clean its join is, in words."),
            ],
        )

    @classmethod
    def execute(
        cls,
        video,
        min_seconds=2.0,
        max_seconds=0.0,
        blend_frames=8,
        max_difference=0.0,
        blend_mode="auto",
    ) -> io.NodeOutput:
        """Find the loop, blend its join, and write both sides for the comparison on the node.

        Raises:
            ValueError: The clip is too short for a loop of min_seconds, or no loop comes
                within max_difference.
        """
        import comfy.model_management

        from ...modules.media import clip as clips

        source = clips.open_clip(video, NODE_NAME)
        count = int(source.frames.shape[0])
        rate = float(source.rate)
        shortest = max(2, int(round(float(min_seconds) * rate)))
        longest = int(round(float(max_seconds) * rate)) if max_seconds > 0 else 0
        device = comfy.model_management.get_torch_device()
        step = clips.progress(2)
        found = loop.find(source.frames, shortest, longest, int(blend_frames), device, float(max_difference))
        step(1)
        if found[0] is None:
            closest = found[-1]
            if max_difference > 0 and closest < float("inf"):
                raise ValueError(
                    f"{NODE_NAME} found no loop whose ends differ by {float(max_difference):g} "
                    f"levels or less; the closest this clip has is {closest:.1f}. Raise "
                    f"max_difference, or lower min_seconds to allow shorter loops."
                )
            raise ValueError(
                f"{NODE_NAME} found no loop of at least {float(min_seconds):g} s in a "
                f"{count / rate:.2f} s clip with {int(blend_frames)} frames to blend beside it. "
                f"Lower min_seconds or blend_frames."
            )
        start, stop, difference, _ = found
        blended = loop.blend_for(str(blend_mode), difference)
        frames, where = loop.loop_frames(source.frames, start, stop, int(blend_frames), device, blended)
        alpha = None
        if source.alpha is not None:
            alpha = source.alpha[start:stop]
        sound = _crossfaded(source.audio, start, stop, int(blend_frames), where, source.rate)
        step(1)

        length = stop - start
        report = (
            f"frames {start} to {stop - 1} ({length} frames, {length / rate:.2f} s); "
            f"join difference {difference:.1f} levels, {loop.quality(difference)}, "
            f"{blended} blend over {int(blend_frames)} frame(s)"
        )
        logger.info("looped %s", report)
        answer = clips.rebuild(source, frames, alpha=alpha, audio=sound)
        return io.NodeOutput(answer, start, length, float(difference), report,
                             ui=clips.compare(video, answer, PREFIX))

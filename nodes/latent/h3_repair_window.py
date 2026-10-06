"""The window a sampler redraws part of a finished MiniMax H3 clip in."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import H3_PROMPTS
from ...modules.latent import h3_assets, h3_conditioning, h3_decode, h3_extend, h3_repair

START_HINT = (
    "Where the repair starts, as `3.0` seconds. Widened back to the start of its 17 frame "
    "block, so `3.0` opens on frame 68 (2.83 s)."
)

END_HINT = (
    "Where the repair ends, as `4.9` seconds. Widened on to the end of its 17 frame block, so "
    "`4.9` closes after frame 118 (4.96 s)."
)

PICTURE_HINT = (
    "How much of the picture in the span is drawn again: `1.0` = from fresh noise, `0.6` = "
    "renewed from what is there, `0.0` = kept, for a sound-only repair."
)

SOUND_HINT = (
    "How much of the sound in the span is drawn again: `0.0` = kept, so the picture is "
    "redrawn in sync under it; `0.5` = renewed from what is there; `1.0` = from fresh noise."
)

CONTEXT_HINT = (
    "Frames of the clip either side of the span the sampler sees and keeps, as `34`, in "
    "steps of 17, or `0` for none. Widened by up to `34` so the window's sound starts on a "
    "whole audio step. Never reaches past a scene cut."
)

RELEASE_HINT = (
    "Audio latent steps the redrawn sound fades in and out over beyond the span, as `8` for "
    "0.2 seconds at 40 steps a second, or `0` for a hard edge. Read where sound_strength is "
    "above `0`."
)

FEATHER_HINT = (
    "Strength the 17 frame block either side of the span is redrawn at, as `0.0` for hard "
    "edges or `0.35` to soften the seam. Never above picture_strength; needs context_frames "
    "of at least `17`."
)

POSITIVE_HINT = (
    "Prompt for the repair. Wired, it replaces the segment's prompt from prompts and is used "
    "as written."
)

PROMPTS_HINT = (
    "Every segment's prompt from MiniMax H3 Conditioning. Wired, the repair reuses its "
    "segment's prompt, references, keyframes and model."
)

SEGMENT_HINT = (
    "Which segment's prompt the repair reuses, from `1`, or `0` for the segment holding "
    "most of the span. Read where prompts is wired."
)


class H3RepairWindow(io.ComfyNode):
    """Cut a window around a stretch of a finished H3 clip, masked so a sampler redraws the stretch."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASH3RepairWindow",
            display_name="H3 Repair Window",
            search_aliases=[
                "WASH3RepairWindow",
                "H3 Repair Window",
                "minimax h3 repair",
                "redraw part of a clip",
                "fix frames",
                "video inpaint",
            ],
            category="WAS Suite/Latent/Video",
            description=(
                "Open part of a finished MiniMax H3 clip for a sampler to redraw. The stretch "
                "from start_seconds to end_seconds widens to whole 17 frame blocks, and the "
                "window around it carries context_frames of the clip either side, held as they "
                "are. picture_strength and sound_strength set how much of the stretch's picture "
                "and sound is drawn again: `1.0` from fresh noise, `0.5` half way from what is "
                "there, `0.0` kept. Wire MiniMax H3 Conditioning's prompts to reuse a segment's "
                "prompt, references, keyframes and model; a wired positive replaces the prompt. "
                "Sample the window at denoise 1.0, then hand the result and the clip to H3 "
                "Repair Splice. A stretch crossing a scene cut is refused."
            ),
            inputs=[
                io.Latent.Input(
                    "latent",
                    tooltip="The finished H3 clip to repair, from H3 Extend Append or a sampler.",
                ),
                io.Float.Input(
                    "start_seconds", default=0.0, min=0.0, max=3600.0, step=0.01,
                    tooltip=START_HINT,
                ),
                io.Float.Input(
                    "end_seconds", default=1.0, min=0.0, max=3600.0, step=0.01,
                    tooltip=END_HINT,
                ),
                io.Float.Input(
                    "picture_strength", default=1.0, min=0.0, max=1.0, step=0.05,
                    tooltip=PICTURE_HINT,
                ),
                io.Float.Input(
                    "sound_strength", default=0.0, min=0.0, max=1.0, step=0.05,
                    tooltip=SOUND_HINT,
                ),
                io.Int.Input(
                    "context_frames", default=h3_repair.CONTEXT_FRAMES, min=0, max=340, step=17,
                    tooltip=CONTEXT_HINT,
                ),
                io.Int.Input(
                    "audio_release", default=h3_extend.AUDIO_RELEASE, min=0, max=64,
                    tooltip=RELEASE_HINT,
                ),
                io.Float.Input(
                    "feather", default=0.0, min=0.0, max=1.0, step=0.05,
                    tooltip=FEATHER_HINT,
                ),
                io.Conditioning.Input("positive", optional=True, tooltip=POSITIVE_HINT),
                H3_PROMPTS.Input("prompts", optional=True, tooltip=PROMPTS_HINT),
                io.Int.Input(
                    "segment", default=0, min=0, max=h3_conditioning.MAX_ROWS, optional=True,
                    tooltip=SEGMENT_HINT,
                ),
            ],
            outputs=[
                io.Latent.Output(
                    display_name="window",
                    tooltip=(
                        "The window to sample at denoise `1.0`, for the sampler's latent input. "
                        "Its result goes to H3 Repair Splice."
                    ),
                ),
                io.Conditioning.Output(
                    display_name="positive",
                    tooltip="The repair's prompt, its keyframes moved into the window, for the guider.",
                ),
                io.Model.Output(
                    display_name="model",
                    tooltip=(
                        "The model the segment's row chose on MiniMax H3 Conditioning, for the "
                        "guider. Blocked with a message where prompts is not wired."
                    ),
                ),
                io.String.Output(
                    display_name="report",
                    tooltip="Which frames and seconds the window redraws, at what strength, and its context.",
                ),
            ],
        )

    @classmethod
    def segment_of(cls, latent, prompts, segment: int, record: dict) -> int:
        """The bundled segment whose prompt the repair reuses.

        Args:
            latent: The clip.
            prompts: The bundle from MiniMax H3 Conditioning.
            segment: The widget's value, from 1, or 0 for the segment holding the span.
            record: The window's record.

        Returns:
            A segment number, from 0.

        Raises:
            ValueError: The segment is past the bundle, or 0 is asked of a clip that records no
                segments while the bundle holds several.
        """
        if int(segment) > 0:
            if int(segment) > len(prompts):
                raise ValueError(
                    f"segment {int(segment)} was asked for and MiniMax H3 Conditioning holds "
                    f"{len(prompts)}. Set segment between 1 and {len(prompts)}, or 0 for the "
                    f"segment holding the span"
                )
            return int(segment) - 1
        opened, closed = record["picture"]
        found = h3_repair.segment_holding(
            latent, h3_repair.frame_of(opened), h3_repair.frame_of(closed))
        if found is None:
            if len(prompts) == 1:
                return 0
            raise ValueError(
                f"segment is 0 and this clip carries no record of where its segments end, so "
                f"there is no telling which of the {len(prompts)} prompts it was sampled with. "
                f"Set segment to the one the span sits in, from 1"
            )
        return min(found, len(prompts) - 1)

    @classmethod
    def execute(cls, latent, start_seconds, end_seconds, picture_strength, sound_strength,
                context_frames, audio_release, feather, positive=None, prompts=None,
                segment=0) -> io.NodeOutput:
        """Plan the window, cut it from the clip and pick the prompt it is redrawn with.

        Raises:
            ValueError: Neither a prompt nor a bundle arrived, the latent is not one H3 joint
                latent, the stretch is empty or crosses a scene cut, both strengths are 0, or
                the segment is not in the bundle.
        """
        if positive is None and prompts is None:
            raise ValueError(
                "H3 Repair Window has nothing to prompt the repair with. Wire the prompts "
                "output of MiniMax H3 Conditioning into prompts to reuse a segment's prompt, "
                "or a conditioning into positive"
            )
        video, audio = h3_extend.split(latent)
        if video.shape[0] != 1:
            raise ValueError(
                f"H3 Repair Window repairs one clip at a time and this latent holds a batch of "
                f"{video.shape[0]}. Take one clip out of the batch first"
            )
        tails = h3_repair.trimmed(latent)
        record = h3_repair.plan(
            video.shape[2], audio.shape[-1], start_seconds, end_seconds, picture_strength,
            sound_strength, context_frames, audio_release, feather, h3_decode.scenes(latent),
            {row: int(tail[0].shape[2]) for row, tail in tails.items()},
        )
        window = h3_repair.window(video, audio, record, tails.get(record["cut"]))
        frames = h3_repair.frame_of(record["rows"] + record["lead"])

        notes = [h3_repair.describe(record)]
        if record["cut"] is not None:
            cut = h3_repair.frame_of(record["cut"])
            notes.append(
                f"closing on the cut at {cut / h3_extend.FPS:.2f} s"
                + (f" with the {h3_repair.frame_of(record['lead'])} frames it trimmed"
                   if record["lead"] else ", which recorded no trimmed frames")
            )
        if record["offset"]:
            notes.append(
                f"the window's sound sits {abs(record['offset']) * 1000:.1f} ms "
                f"{'ahead of' if record['offset'] > 0 else 'behind'} its picture"
            )

        conditioning = positive
        index, named = None, ""
        if prompts is not None:
            try:
                index = cls.segment_of(latent, prompts, segment, record)
            except ValueError as error:
                if positive is None:
                    raise
                from comfy_execution.graph_utils import ExecutionBlocker

                model = ExecutionBlocker(str(error))
        if index is not None:
            model = h3_conditioning.model_or_blocker(prompts, index, "H3 Repair Window")
            kind = h3_conditioning.model_of(prompts, index)[1]
            named = f"segment {index + 1}" + (f" on {kind}" if kind else "")
        elif prompts is None:
            model = h3_conditioning.model_or_blocker(None, 0, "H3 Repair Window")
        if positive is None:
            conditioning = h3_conditioning.pick(prompts, index)[0]
            if h3_repair.closes_on_cut(latent, prompts, index):
                conditioning = h3_assets.closing_moved(conditioning, h3_extend.CLIP_LEAD)
            origin = h3_repair.segment_origin(latent, prompts, index)
            shift = None if origin is None else origin - h3_repair.frame_of(record["start"])
            conditioning, kept, dropped = h3_repair.keyframes_moved(
                conditioning, shift, frames, record["audio_length"])
            prompt = f"prompt of {named}"
            if kept:
                prompt += f", {kept} keyframe(s) moved into the window"
            if dropped:
                prompt += (f", {dropped} keyframe(s) dropped as the clip records no start for that "
                           f"segment" if origin is None
                           else f", {dropped} keyframe(s) outside the window dropped")
            notes.append(prompt)
        else:
            notes.append(f"wired prompt, model of {named}" if named else "wired prompt")
        return io.NodeOutput(window, conditioning, model, "; ".join(notes))

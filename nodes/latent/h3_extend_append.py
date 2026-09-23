"""Joining a sampled MiniMax H3 continuation onto the clip it continues."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.latent import h3_extend


class ThreeH3ExtendAppend(io.ComfyNode):
    """Append a sampled continuation to the clip it continues."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASH3ExtendAppend",
            display_name="H3 Extend Append",
            search_aliases=[
                "WASH3ExtendAppend",
                "H3 Extend Append",
                "minimax h3 append",
                "join continuation",
                "extend video",
            ],
            category="WAS Suite/Latent/Video",
            description=(
                "Join a sampled segment onto the clip so far. The opening segment becomes "
                "the clip; a continued segment arrives with the clip's own last frames at "
                "its head, so only the frames past them are added and every earlier frame "
                "is carried through untouched. A cut adds the whole new scene after the "
                "clip's last full 17 frame block, trimming the 5 frames past it. Decode "
                "once, after the last segment."
            ),
            inputs=[
                io.Latent.Input(
                    "latent",
                    tooltip="The clip so far, from the loop's carried value.",
                ),
                io.Latent.Input(
                    "sampled",
                    tooltip="What the sampler returned for the segment H3 Extend Window opened.",
                ),
                io.Int.Input(
                    "overlap_frames",
                    default=22,
                    min=0,
                    max=362,
                    step=17,
                    tooltip="The overlap H3 Extend Window reported, `0` for a cut.",
                ),
                io.Int.Input(
                    "segment_index",
                    default=0,
                    min=0,
                    max=64,
                    optional=True,
                    tooltip=(
                        "Which segment this is, from `0`. Wire the same While Loop Open "
                        "index that drives H3 Extend Window."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output(
                    display_name="latent",
                    tooltip="The longer clip, for the next pass or for a decode.",
                ),
                io.Int.Output(
                    display_name="frames",
                    tooltip="Frames the joined clip now holds.",
                ),
                io.String.Output(
                    display_name="report",
                    tooltip="What was carried and what was added.",
                ),
            ],
        )

    @classmethod
    def execute(cls, latent, sampled, overlap_frames, segment_index=0) -> io.NodeOutput:
        """Drop the carried head off the window and append what the pass added.

        Raises:
            ValueError: Either latent is not an H3 joint latent, the clip is shorter than
                the overlap, or the pass added no frames.
        """
        if int(segment_index) <= 0:
            # Nothing rendered yet, so what came back is the clip.
            opened, _ = h3_extend.split(sampled)
            frames = h3_extend.frames_for(opened.shape[2])
            return io.NodeOutput(
                sampled, frames,
                f"opened the clip at {frames} frames ({opened.shape[2]} tokens)",
            )

        if int(overlap_frames) <= 0:
            # A cut: the whole fresh scene follows the clip's last whole clip.
            done_video, _ = h3_extend.split(latent)
            _, kept = h3_extend.cut_point(done_video.shape[2])
            trimmed = h3_extend.frames_for(done_video.shape[2]) - kept
            joined = h3_extend.append(latent, sampled, 0)
            whole, _ = h3_extend.split(joined)
            added, _ = h3_extend.split(sampled)
            frames = h3_extend.frames_for(whole.shape[2])
            return io.NodeOutput(
                joined, frames,
                f"cut in {h3_extend.frames_for(added.shape[2])} fresh frames after frame "
                f"{kept}, trimming the clip's last {trimmed}; {whole.shape[2]} tokens "
                f"and {frames} frames now",
            )

        overlap = h3_extend.snap_overlap(overlap_frames)
        done_video, _ = h3_extend.split(latent)
        new_video, _ = h3_extend.split(sampled)
        overlap_tokens = h3_extend.tokens_for(overlap)

        if new_video.shape[2] <= overlap_tokens:
            raise ValueError(
                f"the sampled window holds {new_video.shape[2]} tokens and the {overlap} "
                f"frame overlap alone is {overlap_tokens}, so the pass added nothing. Raise "
                f"extension_frames on H3 Extend Window, or lower the overlap"
            )
        if done_video.shape[2] < overlap_tokens:
            raise ValueError(
                f"the clip holds {done_video.shape[2]} tokens and the {overlap} frame "
                f"overlap needs {overlap_tokens}. Lower the overlap"
            )

        joined = h3_extend.append(latent, sampled, overlap)
        joined_video, joined_audio = h3_extend.split(joined)
        frames = h3_extend.frames_for(joined_video.shape[2])
        report = (
            f"carried {done_video.shape[2]} tokens untouched, added "
            f"{new_video.shape[2] - overlap_tokens}; {joined_video.shape[2]} tokens and "
            f"{frames} frames now, audio {joined_audio.shape[-1]} latent frames"
        )
        return io.NodeOutput(joined, frames, report)

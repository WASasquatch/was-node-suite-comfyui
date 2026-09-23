"""The window and carried rows for one MiniMax H3 continuation pass."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import H3_PROMPTS
from ...modules.latent import h3_conditioning, h3_extend, h3_refresh

PROMPTS_HINT = (
    "Every pass's prompt from MiniMax H3 Conditioning. Wired in, it supplies this pass's "
    "prompt and its frame count, and extension_frames is not read."
)

PASS_INDEX_HINT = (
    "Which segment this is, from `0`. Wire a While Loop Open's index in to step through "
    "them one per iteration."
)

OVERLAP_HINT = (
    "Frames of the finished clip carried into the next pass, as `5`, `22` or `39`. "
    "Snapped down to the model's 17k+5 grid. Longer gives the new frames more of the "
    "scene to continue from. `reference` reads at least `56`."
)

CONTINUITY_HINT = (
    "How this segment picks up from the one before it. `carry` = one unbroken shot, "
    "soundtrack held; `refresh` = the same, detail softened; `handoff` = a cut opening "
    "on the last frame; `reference` = a cut keeping the cast; `cut` = a new scene from "
    "an empty latent, nothing carried. `refresh`, `handoff` and `reference` need vae."
)

DRIFT_HINT = (
    "How much of the contrast and fine detail the carried frames have gained is taken "
    "back out, as `0.0` to carry them exactly as sampled, `0.5` for half or `1.0` for "
    "all of it. Measured against the clip's opening frames, and it only ever softens."
)

RELEASE_HINT = (
    "Audio latent steps the held soundtrack opens back up over where it meets the new "
    "frames, as `8` for 0.2 seconds at 40 steps a second, `0` for a hard edge or `20` "
    "for half a second. Read by `carry` and `refresh`."
)

REFRESH_HINT = (
    "Fine detail a segment adds, which `refresh` softens the carried frames below so "
    "the pass lands back on the opening's reading. `1.15` suits most scenes, `1.0` "
    "softens to match the opening exactly. Read by `refresh` and `handoff`."
)

EXTENSION_HINT = (
    "New frames this pass adds, as `17` for about 0.7s or `102` for about 4.2s at 24 fps. "
    "Snapped down to a multiple of 17."
)


class ThreeH3ExtendWindow(io.ComfyNode):
    """Build the window a continuation samples into, and carry the tail as conditioning rows."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASH3ExtendWindow",
            display_name="H3 Extend Window",
            search_aliases=[
                "WASH3ExtendWindow",
                "H3 Extend Window",
                "minimax h3 extend",
                "video continuation",
                "extend video",
            ],
            category="WAS Suite/Latent/Video",
            description=(
                "Open one segment of a MiniMax H3 video for a sampler. Segment 1 samples "
                "the empty latent it is handed. Every segment after it picks up from the "
                "clip so far: `carry` copies its last frames and the soundtrack under "
                "them into the window and masks them, so the sampler holds them and "
                "generates only what follows, "
                "`refresh` softens the detail those frames gained before carrying them, "
                "`handoff` starts the next segment on their last frame alone, and "
                "`reference` hands them over as a video reference. `cut`, or an overlap "
                "of `0`, samples a new scene from an empty latent of its own length with "
                "nothing carried. Send the latent to a sampler and its result to H3 "
                "Extend Append."
            ),
            inputs=[
                io.Conditioning.Input(
                    "positive",
                    optional=True,
                    tooltip="Prompt for the new frames. A different prompt per pass moves the scene on.",
                ),
                io.Latent.Input(
                    "latent",
                    tooltip="The finished H3 video and audio latent this pass continues.",
                ),
                io.Combo.Input(
                    "continuity",
                    options=list(h3_extend.CONTINUITY),
                    default=h3_extend.CONTINUITY[0],
                    tooltip=CONTINUITY_HINT,
                ),
                io.Vae.Input(
                    "vae",
                    optional=True,
                    tooltip="The H3 video VAE. Needed by every mode but `carry`, which uses none.",
                ),
                io.Int.Input(
                    "extension_frames",
                    default=102,
                    min=17,
                    max=3600,
                    step=17,
                    tooltip=EXTENSION_HINT,
                ),
                io.Int.Input(
                    "overlap_frames",
                    default=22,
                    min=5,
                    max=362,
                    step=17,
                    tooltip=OVERLAP_HINT,
                ),
                H3_PROMPTS.Input(
                    "prompts",
                    optional=True,
                    tooltip=PROMPTS_HINT,
                ),
                io.Int.Input(
                    "pass_index",
                    default=0,
                    min=0,
                    max=h3_conditioning.MAX_ROWS,
                    optional=True,
                    tooltip=PASS_INDEX_HINT,
                ),
                io.Float.Input(
                    "drift_control",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    optional=True,
                    tooltip=DRIFT_HINT,
                ),
                io.Float.Input(
                    "refresh_gain",
                    default=h3_extend.SEGMENT_GAIN,
                    min=1.0,
                    max=2.0,
                    step=0.05,
                    optional=True,
                    tooltip=REFRESH_HINT,
                ),
                io.Int.Input(
                    "audio_release",
                    default=h3_extend.AUDIO_RELEASE,
                    min=0,
                    max=64,
                    optional=True,
                    tooltip=RELEASE_HINT,
                ),
            ],
            outputs=[
                io.Latent.Output(
                    display_name="window",
                    tooltip="The empty window to sample, for the sampler's latent input.",
                ),
                io.Conditioning.Output(
                    display_name="positive",
                    tooltip="The prompt for this pass, for the guider that samples the window.",
                ),
                io.Int.Output(
                    display_name="overlap_frames",
                    tooltip="The snapped overlap, for the same input on H3 Extend Append.",
                ),
                io.String.Output(
                    display_name="report",
                    tooltip="What the pass will sample and what it snapped to.",
                ),
            ],
        )

    @classmethod
    def execute(cls, latent, continuity, extension_frames, overlap_frames,
                vae=None, positive=None, prompts=None, pass_index=0,
                drift_control=0.0,
                refresh_gain=h3_extend.SEGMENT_GAIN,
                audio_release=h3_extend.AUDIO_RELEASE) -> io.NodeOutput:
        """Attach the tail as per-token rows and answer the window to sample.

        Raises:
            ValueError: Neither a prompt nor a bundle arrived, the latent is not an H3 joint
                latent, or the latent is shorter than the overlap.
        """
        if prompts is not None:
            positive, extension_frames, carried, choice = h3_conditioning.pick(
                prompts, pass_index)
            overlap_frames = carried
            if choice != h3_conditioning.AS_SET:
                continuity = choice
        elif positive is None:
            raise ValueError(
                "H3 Extend Window has nothing to prompt the new frames with. Wire a "
                "conditioning into positive, or the prompts output of MiniMax H3 "
                "Conditioning into prompts"
            )

        named = f"segment {int(pass_index) + 1}"
        if int(pass_index) <= 0 or int(overlap_frames) <= 0 or continuity == "cut":
            # Either nothing has been rendered yet, or this segment cuts somewhere new.
            video, audio = h3_extend.split(latent)
            length = (h3_extend.frames_for(video.shape[2]) if int(pass_index) <= 0
                      else h3_extend.snap_clip(extension_frames))
            fresh = h3_extend.empty_like(video, audio, length)
            how = "opening" if int(pass_index) <= 0 else "cutting to"
            return io.NodeOutput(
                fresh, positive, 0, f"{named}: {how} a fresh {length} frame scene",
            )

        overlap = h3_extend.snap_overlap(overlap_frames)
        extension = h3_extend.snap_extension(extension_frames)
        video, audio = h3_extend.split(latent)
        video_tail, audio_tail = h3_extend.tail(latent, overlap)

        if continuity == "handoff":
            if vae is None:
                raise ValueError(
                    "H3 Extend Window is set to `handoff`, which decodes the finished "
                    "frames and hands the last one to the next segment. Wire the H3 video "
                    "VAE into vae, or set continuity to `carry`"
                )
            import node_helpers

            opening = video.shape[2]
            if prompts is not None:
                opening = h3_extend.tokens_for(h3_conditioning.pick(prompts, 0)[1])
            # The decoded tail reaches back past the frames the cut trims.
            rows = min(video.shape[2], max(video_tail.shape[2],
                                           h3_extend.CLIP_TOKENS + h3_extend.TOKEN_LEAD))
            start = max(rows, min(opening, video.shape[2]))
            anchor = vae.decode(video[:, :, start - rows:start])
            current = vae.decode(video[:, :, -rows:])
            if anchor.ndim == 5:
                anchor = anchor[0]
            if current.ndim == 5:
                current = current[0]
            # Colour comes back to the opening before anything else reads these frames.
            toned, gains = h3_refresh.levelled(current, anchor)
            was = h3_refresh.texture(toned)
            target = h3_refresh.texture(anchor) / max(1.0, float(refresh_gain))
            eased, sigma, covered = h3_refresh.softened(toned, anchor, target)
            # The next segment opens on the last frame the cut keeps, and nothing else.
            _, kept = h3_extend.cut_point(video.shape[2])
            last = max(0, eased.shape[0] - (h3_extend.frames_for(video.shape[2]) - kept) - 1)
            block = {"resolved_frame_index": 0, "latent": vae.encode(eased[last:last + 1])}
            handed = node_helpers.conditioning_set_values(
                positive, {"minimax_keyframes": [block]}, append=True
            )
            length = h3_extend.snap_clip(extension)
            report = (
                f"{named}: handing frame {kept} on, levelled by "
                f"{', '.join(f'{gain:.3f}' for gain in gains)} and blurred {sigma:.2f} "
                f"over {covered * 100:.1f}% of it, texture {was:.4f} towards "
                f"{target:.4f}; sampling {length} fresh frames"
            )
            return io.NodeOutput(
                h3_extend.empty_like(video, audio, length), handed, 0, report,
            )

        if continuity == "reference":
            if vae is None:
                raise ValueError(
                    "H3 Extend Window is set to `reference`, which decodes the finished "
                    "frames and encodes them again. Wire the H3 video VAE into vae, or set "
                    "continuity to `carry`"
                )
            import node_helpers

            # The reference reaches back at least REFERENCE_FRAMES where the clip holds them.
            rows = min(video.shape[2], max(video_tail.shape[2],
                                           h3_extend.tokens_for(h3_extend.REFERENCE_FRAMES)))
            frames = vae.decode(video[:, :, -rows:])
            if frames.ndim == 5:
                frames = frames[0]
            ref_w, ref_h = h3_extend.reference_canvas(frames.shape[2], frames.shape[1])
            block = h3_extend.video_reference(
                vae.encode(h3_conditioning.fitted_batch(frames, ref_w, ref_h))
            )
            # Joins any references the prompt already carries, as from `ref2va`.
            referenced = node_helpers.conditioning_set_values(
                positive, {"minimax_refs": [block]}, append=True
            )
            length = h3_extend.snap_clip(extension)
            return io.NodeOutput(
                h3_extend.empty_like(video, audio, length), referenced, 0,
                f"{named}: referencing {frames.shape[0]} decoded frames at {ref_w}x{ref_h}; "
                f"sampling {length} fresh frames",
            )

        window = h3_extend.window_frames(overlap, extension)
        tokens = h3_extend.tokens_for(window)
        # The soundtrack is held for the whole latent steps that fit inside the overlap.
        wanted_audio = h3_extend.audio_carry(overlap)
        rows = video_tail.shape[2]

        if continuity == "refresh":
            if vae is None:
                raise ValueError(
                    "H3 Extend Window is set to `refresh`, which decodes the carried "
                    "frames, softens them and encodes them again. Wire the H3 video VAE "
                    "into vae, or set continuity to `carry`"
                )
            # The clip's first segment is the reading every later segment is brought back to.
            opening = video.shape[2]
            if prompts is not None:
                opening = h3_extend.tokens_for(h3_conditioning.pick(prompts, 0)[1])
            start = max(rows, min(opening, video.shape[2]))
            anchor = vae.decode(video[:, :, start - rows:start])
            current = vae.decode(video_tail)
            if anchor.ndim == 5:
                anchor = anchor[0]
            if current.ndim == 5:
                current = current[0]
            # Softened below the anchor by the detail a segment adds.
            target = h3_refresh.texture(anchor) / max(1.0, float(refresh_gain))
            was = h3_refresh.texture(current)
            # Only the pixels carrying more detail than the opening are touched.
            eased, sigma, covered = h3_refresh.softened(current, anchor, target)
            held_rows = vae.encode(eased)
            detail = (
                f"refreshed {rows} rows at blur {sigma:.2f} over {covered * 100:.1f}% of "
                f"the picture, texture {was:.4f} towards {target:.4f}"
            )
        else:
            held_rows, scale, gain = h3_extend.settled(video, rows, drift_control)
            detail = f"settled at scale {scale:.4f} and detail gain {gain:.4f}"

        release = max(0, min(int(audio_release), wanted_audio))
        carried, mask, held_audio = h3_extend.masked_window(
            held_rows, audio_tail, tokens, h3_extend.audio_span(window), wanted_audio,
            release,
        )
        carried["noise_mask"] = mask["samples"]

        held = h3_extend.frames_for(video.shape[2])
        lead = h3_extend.audio_lead(overlap)
        seam = ("on the audio grid exactly" if h3_extend.audio_lands_whole(overlap)
                else f"{lead * 1000:.1f}ms short of the seam")
        sound = (f"{held_audio} audio steps held, released over {release}, {seam}"
                 if held_audio else "picture only")
        report = (
            f"{named}: "
            f"holding {overlap} frames ({video_tail.shape[2]} tokens, {sound}) of a {held} "
            f"frame clip at sigma 0 inside a {window} frame "
            f"window ({tokens} tokens); "
            f"generating {extension} new frames for a {held + extension} frame clip; "
            f"{detail}"
        )
        return io.NodeOutput(carried, positive, overlap, report)

"""The window and carried rows for one MiniMax H3 continuation pass."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import H3_PROMPTS
from ...modules.latent import (
    h3_assets, h3_conditioning, h3_decode, h3_extend, h3_preview, h3_references, h3_refresh,
)

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
    "Snapped to the nearest step of the model's 17k+5 grid, so `16` carries `22`. "
    "Longer gives the new frames more of the "
    "scene to continue from. `reference (video)` reads at least `56`; both sound carries "
    "take whole clips of sound, `17` for about 0.7s."
)

CONTINUITY_HINT = (
    "`carry` = one shot; `refresh` = re-noised; "
    "`handoff` = cut on last frame; `reference (video)` = cut, cast kept; "
    "`reference (sample)` = cut, cast from stills; `cut` = new scene; "
    "`carry (audio only)` = cut, sound kept; `carry (audio) + reference (video)` = cut, "
    "sound and cast kept. A prompt row overrides it. References and handoff need vae."
)

DRIFT_HINT = (
    "How much of the contrast and fine detail the carried frames have gained is taken "
    "back out, as `0.0` to carry them exactly as sampled, `0.5` for half or `1.0` for "
    "all of it. Measured against the clip's opening frames, and it only ever softens."
)

RELEASE_HINT = (
    "Audio latent steps the held soundtrack opens back up over where it meets the new "
    "frames, as `8` for 0.2 seconds at 40 steps a second, `0` for a hard edge or `20` "
    "for half a second. Read by `carry`, `refresh` and both sound carries."
)

SOURCE_HINT = (
    "Which segment this pass continues from: `-1` = the one before, `-2` = the one before "
    "that, `2` = segment 2. From an earlier segment the picture carries and the sound "
    "starts fresh; the new frames still join the clip's end. Both sound carries always "
    "read the segment before. A row's own source replaces this where prompts is wired."
)

SAMPLES_HINT = (
    "Stills `reference (sample)` takes from the clip so far, spread evenly from its first "
    "frame to its last, as `4`, or `6` for a long clip with many scenes. Read by "
    "`reference (sample)`."
)

RENEWAL_HINT = (
    "Fresh noise `refresh` puts into the carried frames, as `0.0` to hold them as "
    "`carry` does, `0.35` to loosen them so the scene can evolve, or `1.0` to resample "
    "them with only the carried picture to start from. Read by `refresh`."
)

REFRESH_HINT = (
    "Fine detail a segment adds, which `handoff` softens the frame it opens on below so "
    "the pass lands back on the opening's reading. `1.15` suits most scenes, `1.0` "
    "softens to match the opening exactly. Read by `handoff`."
)

LIVE_PREVIEW_HINT = (
    "`true` = the Prompt Timeline draws this segment as it samples and keeps its last step, "
    "through the preview decoder in models/vae_approx whose name starts `taeh3`, as ComfyUI's "
    "own previews find theirs; `false` = no previews. Needs prompts."
)

REFERENCE_SOUND_HINT = (
    "On, a clip the window references from the scene before carries that scene's sound "
    "under it, so voices and ambience follow the cast across the cut. Read by "
    "`reference (video)` and `carry (audio) + reference (video)`."
)

REENCODE_HINT = (
    "`true` = a segment this window adds references or a handoff frame to has its prompt "
    "encoded again with them, as wired references are; one text encoder pass for that "
    "segment. `false` = they are attached as they are."
)

#: Where `reference (sample)` takes its stills from.
SAMPLE_SPANS = ("source scene", "whole clip")

SAMPLE_SPAN_HINT = (
    "Where `reference (sample)` takes its stills: `source scene` = spread across the scene "
    "this segment continues from, `whole clip` = spread from the first frame to the last."
)

SEED_HINT = (
    "The run's seed, as `42`. The seed output answers this plus the segment number, or the "
    "segment's own seed where its row names one."
)

EXTENSION_HINT = (
    "New frames this pass adds, as `17` for about 0.7s or `102` for about 4.2s at 24 fps. "
    "Snapped to the nearest multiple of 17."
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
                "the empty latent it is handed. Each later segment picks up from the clip "
                "so far: `carry` holds its last frames and their sound and generates what "
                "follows, `refresh` carries them with fresh noise, `handoff` opens on their "
                "last frame, `reference (video)` hands them over as a reference, and "
                "`reference (sample)` references stills from the whole clip. `carry (audio "
                "only)` cuts to new picture under the carried sound, and `carry (audio) + "
                "reference (video)` also references the last frames. `cut`, or an overlap "
                "of `0` under `carry` or `refresh`, starts a fresh scene. Send the latent "
                "to a sampler and its result to H3 Extend Append."
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
                    tooltip="The H3 video VAE. Needed by `handoff` and both references.",
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
                io.Float.Input(
                    "renewal",
                    default=h3_extend.RENEWAL,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    optional=True,
                    tooltip=RENEWAL_HINT,
                ),
                io.Int.Input(
                    "reference_samples",
                    default=h3_extend.REFERENCE_SAMPLES,
                    min=1,
                    max=8,
                    optional=True,
                    tooltip=SAMPLES_HINT,
                ),
                io.Int.Input(
                    "source",
                    default=h3_conditioning.PREVIOUS_SOURCE,
                    min=-h3_conditioning.MAX_ROWS,
                    max=h3_conditioning.MAX_ROWS,
                    optional=True,
                    tooltip=SOURCE_HINT,
                ),
                io.Boolean.Input("live_preview", default=True, optional=True, tooltip=LIVE_PREVIEW_HINT),
                io.Boolean.Input("reference_sound", default=True, optional=True,
                                 tooltip=REFERENCE_SOUND_HINT),
                io.Boolean.Input("reencode_prompt", default=True, optional=True,
                                 tooltip=REENCODE_HINT),
                io.Combo.Input("sample_span", options=list(SAMPLE_SPANS), default=SAMPLE_SPANS[0],
                               optional=True, tooltip=SAMPLE_SPAN_HINT),
                io.Int.Input("seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, optional=True,
                             control_after_generate=False, tooltip=SEED_HINT),
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
                io.Model.Output(
                    display_name="model",
                    tooltip=(
                        "The model this segment's row chose, from the ones wired into "
                        "MiniMax H3 Conditioning, for the guider that samples the window. "
                        "Blocked with a message where none is wired."
                    ),
                ),
                io.Int.Output(
                    display_name="seed",
                    tooltip="This segment's seed, for the noise the sampler draws, as `43`.",
                ),
            ],
        )

    @classmethod
    def stills(cls, latent, vae, count, named, positive, first_row=0, last_row=None):
        """Stills sampled evenly across rows of the clip, encoded as picture references.

        Args:
            latent: The finished H3 joint latent.
            vae: The H3 video VAE.
            count: Stills to sample.
            named: The segment's name, for the error.
            positive: The prompt the stills join, checked for space.
            first_row: The first latent row the stills come from.
            last_row: The row after the last, or ``None`` for the clip's end.

        Returns:
            ``(blocks, items, taken, size)``: the references, their text encoder entries, when
            each was taken as ``8.0s``, and their size.

        Raises:
            ValueError: No vae is wired, or the prompt already carries the most pictures it takes.
        """
        if vae is None:
            raise ValueError(
                "H3 Extend Window is set to `reference (sample)`, which decodes stills "
                "from the finished clip and encodes them again as pictures. Wire the H3 "
                "video VAE into vae, or set continuity to `cut`"
            )
        free = h3_references.room(positive, "image")
        if free < 1:
            raise ValueError(
                f"{named} is set to `reference (sample)` and its prompt already carries "
                f"{h3_references.MOST_IMAGES} pictures, the most a MiniMax H3 prompt takes. "
                f"Take a picture reference off this segment, or set continuity to `cut`"
            )
        video, _ = h3_extend.split(latent)
        end = video.shape[2] if last_row is None else max(1, min(int(last_row), video.shape[2]))
        begin = max(0, min(int(first_row), end - 1))
        blocks, items, taken, size = [], [], [], None
        for offset in h3_extend.sampled_rows(end - begin, min(int(count), free)):
            row = begin + offset
            start = max(0, row - h3_extend.SAMPLE_CONTEXT_ROWS)
            frames = vae.decode(video[:, :, start:row + 1])
            if frames.ndim == 5:
                frames = frames[0]
            still = frames[-1:]
            wide, high = h3_extend.reference_canvas(still.shape[2], still.shape[1])
            fitted = h3_conditioning.fitted_batch(still, wide, high)
            blocks.append(h3_extend.picture_reference(vae.encode(fitted)))
            items.append({"type": "image", "data": fitted})
            taken.append(f"{h3_extend.last_frame_of(row) / h3_extend.FPS:.1f}s")
            size = f"{wide}x{high}"
        return blocks, items, taken, size

    @classmethod
    def sampled(cls, latent, positive, vae, extension_frames, count, named,
                model, prompts=None, pass_index=0, reencode=True, rows=(0, None)) -> io.NodeOutput:
        """A fresh scene referencing stills sampled across part of the clip.

        Args:
            latent: The finished H3 joint latent.
            positive: This segment's prompt.
            vae: The H3 video VAE.
            extension_frames: Frames the new scene runs for.
            count: Stills to sample.
            named: The segment's name in the report.
            model: The segment's model, or the blocker standing in for it.
            prompts: The bundle, or ``None``.
            pass_index: The segment, from 0.
            reencode: Whether to encode the prompt again with the stills.
            rows: ``(first, end)`` latent rows the stills come from, ``end`` ``None`` for the
                clip's end.

        Returns:
            The empty scene, its prompt carrying the stills, overlap 0, the report and the
            model.

        Raises:
            ValueError: No vae is wired.
        """
        video, audio = h3_extend.split(latent)
        blocks, items, taken, size = cls.stills(latent, vae, count, named, positive, *rows)
        referenced, encoded = cls.with_references(prompts, pass_index, positive, blocks, items,
                                                  reencode=reencode)
        length = h3_extend.snap_clip(extension_frames)
        return io.NodeOutput(
            h3_extend.empty_like(video, audio, length), referenced, 0,
            f"{named}: referencing {len(blocks)} stills at {size} sampled at "
            f"{', '.join(taken)}{encoded}; sampling {length} fresh frames",
            model,
        )

    @classmethod
    def handoff_frame(cls, video, vae, prompts, tail_rows, refresh_gain):
        """The clip's last kept frame, levelled to the opening and softened, for a new scene to open on.

        Args:
            video: The finished clip's video latent.
            vae: The H3 video VAE.
            prompts: The bundle, or ``None``.
            tail_rows: Latent rows the overlap covers.
            refresh_gain: Fine detail a segment is expected to add.

        Returns:
            ``(picture, kept, note)``: the ``[1, H, W, C]`` frame, its frame number in the clip,
            and what was done to it, for the report.

        Raises:
            ValueError: No VAE is wired.
        """
        if vae is None:
            raise ValueError(
                "H3 Extend Window is set to `handoff`, which decodes the finished "
                "frames and hands the last one to the next segment. Wire the H3 video "
                "VAE into vae, or set continuity to `carry`"
            )
        opening = video.shape[2]
        if prompts is not None:
            opening = h3_extend.tokens_for(h3_conditioning.pick(prompts, 0)[1])
        # The decoded tail reaches back past the frames the cut trims.
        rows = min(video.shape[2], max(tail_rows, h3_extend.CLIP_TOKENS + h3_extend.TOKEN_LEAD))
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
        note = (
            f"handing frame {kept} on, levelled by {', '.join(f'{gain:.3f}' for gain in gains)} "
            f"and blurred {sigma:.2f} over {covered * 100:.1f}% of it, texture {was:.4f} "
            f"towards {target:.4f}"
        )
        return eased[last:last + 1], kept, note

    @classmethod
    def looped(cls, video, window, vae, positive):
        """A last segment's prompt closing on the video's first frame.

        Args:
            video: The clip so far.
            window: The window the segment samples.
            vae: The H3 video VAE.
            positive: The segment's prompt.

        Returns:
            ``(prompt, note)``, the note for the report.

        Raises:
            ValueError: No VAE is wired.
        """
        import node_helpers

        if vae is None:
            raise ValueError(
                "MiniMax H3 Conditioning has loop on, which decodes the video's first frame "
                "and closes the last segment on it. Wire the H3 video VAE into vae"
            )
        first = vae.decode(video[:, :, :1])
        if first.ndim == 5:
            first = first[0]
        frames = h3_extend.frames_for(h3_extend.split(window)[0].shape[2])
        block = {"resolved_frame_index": frames - 1, "latent": vae.encode(first[:1]),
                 h3_conditioning.CLOSING_KEY: True}
        return (node_helpers.conditioning_set_values(positive, {"minimax_keyframes": [block]},
                                                     append=True),
                f"closing on the video's first frame at window frame {frames - 1} for a loop")

    @classmethod
    def with_moments(cls, prompts, pass_index, latent, vae, reencode=True):
        """The bundle with one segment's moments of the video in its prompt as pictures.

        Args:
            prompts: The bundle, or ``None``.
            pass_index: The segment, from 0.
            latent: The clip so far.
            vae: The H3 video VAE.
            reencode: Whether to encode the prompt again with them.

        Returns:
            ``(prompts, note)``: the bundle, a copy where the segment's prompt changed, and the
            moments for the report, empty where there were none.

        Raises:
            ValueError: A moment asked of this segment is not made yet, or no VAE is wired.
        """
        import types

        import node_helpers

        index = int(pass_index)
        if not prompts or not 0 <= index < len(prompts):
            return prompts, ""
        entry = prompts[index]
        clip_frames = h3_extend.frames_for(h3_extend.split(latent)[0].shape[2]) if index > 0 else 0
        wanted = []
        for frame, name, every in entry.get(h3_assets.MOMENTS_KEY) or []:
            if frame < clip_frames:
                wanted.append((frame, name))
            elif not every:
                raise ValueError(
                    f"{name} is referenced by segment {index + 1}, which starts at frame "
                    f"{clip_frames} of the video, so that frame is not made yet. Set its frame "
                    f"below {clip_frames}, or move it to a later segment"
                )
        if not wanted:
            return prompts, ""
        if vae is None:
            raise ValueError(
                f"segment {index + 1} references frames of the video being made, which are "
                f"decoded and encoded again. Wire the H3 video VAE into vae"
            )
        positive = entry["conditioning"]
        free = h3_references.room(positive, "image")
        blocks, items, taken = [], [], []
        for frame, name in wanted[:free]:
            still = h3_decode.frame_at(vae, latent, frame)
            wide, high = h3_extend.reference_canvas(still.shape[2], still.shape[1])
            fitted = h3_conditioning.fitted_batch(still, wide, high)
            blocks.append(h3_extend.picture_reference(vae.encode(fitted)))
            items.append({"type": "image", "data": fitted})
            taken.append(f"{name} ({frame / h3_extend.FPS:.2f}s)")
        entry = dict(entry)
        encoding = entry.get(h3_conditioning.ENCODING_KEY)
        fresh = (h3_conditioning.reencoded(entry, positive, items, blocks)
                 if reencode and encoding else None)
        if fresh is not None:
            base = encoding.get("references")
            encoding = dict(encoding)
            encoding["references"] = types.SimpleNamespace(
                items=list(getattr(base, "items", None) or []) + items,
                blocks=list(getattr(base, "blocks", None) or []) + blocks)
            entry[h3_conditioning.ENCODING_KEY] = encoding
            entry["conditioning"] = fresh
        else:
            entry["conditioning"] = node_helpers.conditioning_set_values(
                positive, {"minimax_refs": blocks}, append=True)
        prompts = list(prompts)
        prompts[index] = entry
        return prompts, ", ".join(taken)

    @classmethod
    def with_references(cls, prompts, pass_index, positive, blocks, items, pictures=(),
                        reencode=True, text=None):
        """A prompt carrying a pass's own references, encoded again where it can be.

        Args:
            prompts: The bundle, or ``None``.
            pass_index: The segment, from 0.
            positive: The prompt as the pass received it.
            blocks: The ``minimax_refs`` blocks the pass adds.
            items: Their text encoder entries, in the same order.
            pictures: Keyframe pictures the pass adds.
            reencode: Whether to encode the prompt again.
            text: The prompt text to encode in place of the segment's own, or ``None``.

        Returns:
            ``(prompt, note)``, the note naming an encode for the report.
        """
        import node_helpers

        if reencode and prompts and 0 <= int(pass_index) < len(prompts):
            fresh = h3_conditioning.reencoded(prompts[int(pass_index)], positive, items, blocks,
                                              pictures, text)
            if fresh is not None:
                return fresh, "; prompt encoded again with them"
        if blocks:
            # Joins any references the prompt already carries, as from `ref2va`.
            positive = node_helpers.conditioning_set_values(
                positive, {"minimax_refs": list(blocks)}, append=True)
        return positive, ""

    @classmethod
    def last_frames(cls, video, vae, at_least: int, continuity: str, instead: str,
                    positive=None, audio=None):
        """The clip's last frames, decoded and encoded again as a ``minimax_refs`` video.

        Args:
            video: The finished clip's video latent.
            vae: The H3 video VAE, or ``None``.
            at_least: Latent rows the reference reaches back, at least.
            continuity: The continuity asking, for the error.
            instead: The continuity the error offers in its place.
            positive: The prompt the reference joins, checked for space.
            audio: The clip's audio latent, whose last stretch the reference carries, or ``None``.

        Returns:
            ``(block, frames, width, height, items)``: the reference, the frames it holds, its
            size, and its text encoder entries.

        Raises:
            ValueError: No VAE is wired, or the prompt already carries the most clips it takes.
        """
        if h3_references.room(positive, "video") < 1:
            raise ValueError(
                f"H3 Extend Window is set to `{continuity}` and this segment's prompt already "
                f"carries {h3_references.MOST_VIDEOS} clips, the most a MiniMax H3 prompt "
                f"takes. Take a clip reference off this segment, or set continuity to "
                f"`{instead}`"
            )
        if vae is None:
            raise ValueError(
                f"H3 Extend Window is set to `{continuity}`, which decodes the finished frames "
                f"and encodes them again. Wire the H3 video VAE into vae, or set continuity to "
                f"`{instead}`"
            )
        rows = min(video.shape[2], max(int(at_least),
                                       h3_extend.tokens_for(h3_extend.REFERENCE_FRAMES)))
        frames = vae.decode(video[:, :, -rows:])
        if frames.ndim == 5:
            frames = frames[0]
        ref_w, ref_h = h3_extend.reference_canvas(frames.shape[2], frames.shape[1])
        span = h3_extend.audio_span(int(frames.shape[0])) if audio is not None else 0
        sound = audio[..., -span:] if span > 0 and audio.shape[-1] >= span else None
        fitted = h3_conditioning.fitted_batch(frames, ref_w, ref_h)
        block = h3_extend.video_reference(vae.encode(fitted), sound)
        return (block, int(frames.shape[0]), ref_w, ref_h,
                h3_references.clip_items(fitted, sound is not None))

    @classmethod
    def bridged(cls, latent, positive, extension_frames, overlap_frames, audio_release,
                named, model, vae=None, picture="cut", reference_sound=True, prompts=None,
                pass_index=0, reencode=True, refresh_gain=h3_extend.SEGMENT_GAIN,
                samples=h3_extend.REFERENCE_SAMPLES, sample_rows=(0, None)) -> io.NodeOutput:
        """A new scene sampled under the soundtrack of the clip's last whole clips.

        Args:
            latent: The finished H3 joint latent.
            positive: This segment's prompt.
            extension_frames: Frames the new scene adds.
            overlap_frames: Frames of sound to carry, snapped to whole clips.
            audio_release: Audio steps the held sound opens back up over.
            named: The segment's name in the report.
            model: The segment's model, or the blocker standing in for it.
            vae: The H3 video VAE, read by every picture but ``cut``.
            picture: The picture transition under the bridge: ``cut``, ``handoff``,
                ``reference (video)`` or ``reference (sample)``.
            reference_sound: Whether a referenced clip carries the sound under its frames.
            prompts: The bundle, or ``None``.
            pass_index: The segment, from 0.
            reencode: Whether to encode the prompt again with what the picture adds.
            refresh_gain: Fine detail a segment is expected to add, read by ``handoff``.
            samples: Stills ``reference (sample)`` takes.
            sample_rows: ``(first, end)`` rows the stills come from.

        Returns:
            The window, its prompt, the bridge in frames, the report and the model.

        Raises:
            ValueError: The clip holds less sound than the bridge carries, or a picture that
                decodes the clip is asked for with no VAE wired.
        """
        import node_helpers

        video, audio = h3_extend.split(latent)
        frames = h3_extend.bridge_frames(overlap_frames)
        referencing = ""
        # The bridge plays the scene before as a shot of its own, cut away from where the join keeps.
        own = (h3_conditioning.encoded_text(prompts[int(pass_index)])
               if prompts and 0 <= int(pass_index) < len(prompts) else "")
        cut_text = h3_conditioning.cut_in(own, frames / h3_extend.FPS) if own.strip() else None
        cutting = (f"; its shots cut in at {h3_conditioning.shot_time(frames / h3_extend.FPS)}"
                   if cut_text and reencode else "")
        if picture == h3_extend.REFERENCE_VIDEO:
            block, count, ref_w, ref_h, items = cls.last_frames(
                video, vae, h3_extend.tokens_for(frames), h3_extend.AUDIO_REFERENCE,
                h3_extend.AUDIO_CARRY, positive, audio if reference_sound else None,
            )
            positive, encoded = cls.with_references(prompts, pass_index, positive, [block], items,
                                                    reencode=reencode, text=cut_text)
            sounding = " with its sound" if block.get("audio_latent") is not None else ""
            referencing = (f"; referencing its last {count} frames at {ref_w}x{ref_h}"
                           f"{sounding}{encoded}{cutting if encoded else ''}")
        elif picture == h3_extend.REFERENCE_SAMPLE:
            blocks, items, taken, size = cls.stills(latent, vae, samples, named, positive, *sample_rows)
            positive, encoded = cls.with_references(prompts, pass_index, positive, blocks, items,
                                                    reencode=reencode, text=cut_text)
            referencing = (f"; referencing {len(blocks)} stills at {size} sampled at "
                           f"{', '.join(taken)}{encoded}{cutting if encoded else ''}")
        elif picture == "handoff":
            frame, _, note = cls.handoff_frame(video, vae, prompts, h3_extend.tokens_for(frames),
                                               refresh_gain)
            positive, encoded = cls.with_references(prompts, pass_index, positive, [], [], [frame],
                                                    reencode)
            # The new picture opens on the frame, the first the join keeps after the bridge.
            positive = node_helpers.conditioning_set_values(
                positive, {"minimax_keyframes": [{"resolved_frame_index": frames,
                                                  "latent": vae.encode(frame)}]}, append=True)
            referencing = f"; {note}{encoded.replace('with them', 'with it')}"
        extension = h3_extend.snap_extension(extension_frames)
        window = h3_extend.snap_clip(frames + extension)
        _, kept = h3_extend.cut_point(video.shape[2])
        end = min(audio.shape[-1], h3_extend.audio_span(kept))
        span = h3_extend.audio_span(frames)
        if end < span:
            raise ValueError(
                f"{named} carries {frames} frames of sound and the clip so far holds "
                f"{kept}. Lower the overlap, or carry from a longer clip"
            )
        # The window holds the whole span of sound the join drops.
        release = max(0, min(int(audio_release), span))
        carried, mask, held = h3_extend.masked_window(
            video[:, :, :0], audio[..., end - span:end].clone(), h3_extend.tokens_for(window),
            h3_extend.audio_span(window), span, release,
        )
        carried["noise_mask"] = mask["samples"]
        carried[h3_extend.BRIDGE_KEY] = frames
        report = (
            f"{named}: cutting to new picture after frame {kept} under {held} audio steps "
            f"({frames} frames) of its sound, released over {release}{referencing}; sampling "
            f"{window} frames and keeping the {window - frames} after the bridge"
        )
        return io.NodeOutput(carried, positive, frames, report, model)

    @classmethod
    def execute(cls, latent, continuity, extension_frames, overlap_frames,
                vae=None, positive=None, prompts=None, pass_index=0,
                drift_control=0.0,
                refresh_gain=h3_extend.SEGMENT_GAIN,
                audio_release=h3_extend.AUDIO_RELEASE,
                renewal=h3_extend.RENEWAL,
                reference_samples=h3_extend.REFERENCE_SAMPLES,
                source=h3_conditioning.PREVIOUS_SOURCE,
                live_preview=True, reference_sound=True, reencode_prompt=True,
                sample_span=SAMPLE_SPANS[0], seed=0) -> io.NodeOutput:
        """Open the window, place it on the finished clip, and draw each step through the preview decoder."""
        clip_video, clip_audio = h3_extend.split(latent)
        clip_rows = clip_video.shape[2]
        prompts, moments = cls.with_moments(prompts, pass_index, latent, vae, bool(reencode_prompt))
        output = cls.opened(latent, continuity, extension_frames, overlap_frames, vae,
                            positive, prompts, pass_index, drift_control, refresh_gain,
                            audio_release, renewal, reference_samples, source,
                            reference_sound, reencode_prompt, sample_span, bool(moments))
        window, conditioned, head, report, model = output.args
        if moments:
            report = f"{report}; referencing {moments}"
        if prompts and 0 < int(pass_index) < len(prompts) and prompts[int(pass_index)].get(
                h3_conditioning.LOOP_KEY):
            conditioned, note = cls.looped(clip_video, window, vae, conditioned)
            report = f"{report}; {note}"
        window = dict(window)
        window[h3_extend.WINDOW_KEY] = h3_extend.window_place(
            clip_rows, window, int(head), int(pass_index) <= 0, int(clip_audio.shape[-1]))
        preview = h3_preview.decoder() if live_preview and prompts else None
        if preview is not None and 0 <= int(pass_index) < len(prompts):
            owner = prompts[int(pass_index)].get(h3_conditioning.OWNER_KEY)
            model = h3_preview.watched(model, preview, owner, int(pass_index), int(head))
        own = h3_conditioning.seed_of(prompts, int(pass_index), int(seed))
        return io.NodeOutput(window, conditioned, head, report, model, own)

    @classmethod
    def opened(cls, latent, continuity, extension_frames, overlap_frames,
               vae=None, positive=None, prompts=None, pass_index=0,
               drift_control=0.0,
               refresh_gain=h3_extend.SEGMENT_GAIN,
               audio_release=h3_extend.AUDIO_RELEASE,
               renewal=h3_extend.RENEWAL,
               reference_samples=h3_extend.REFERENCE_SAMPLES,
               source=h3_conditioning.PREVIOUS_SOURCE,
               reference_sound=True, reencode_prompt=True,
               sample_span=SAMPLE_SPANS[0], moments=False) -> io.NodeOutput:
        """Attach the tail as per-token rows and answer the window to sample.

        Raises:
            ValueError: Neither a prompt nor a bundle arrived, the latent is not an H3 joint
                latent, or the latent is shorter than the overlap.
        """
        if prompts is not None:
            positive, extension_frames, carried, choice = h3_conditioning.pick(
                prompts, pass_index)
            overlap_frames = carried
            continuity = choice
        elif positive is None:
            raise ValueError(
                "H3 Extend Window has nothing to prompt the new frames with. Wire a "
                "conditioning into positive, or the prompts output of MiniMax H3 "
                "Conditioning into prompts"
            )

        continuity = h3_extend.CONTINUITY_RENAMED.get(continuity, continuity)
        sound = h3_conditioning.sound_of(prompts, pass_index) if prompts is not None else "auto"
        picture, bridging, fresh_sound = h3_extend.resolved(continuity, sound)
        named = f"segment {int(pass_index) + 1}"
        following = int(pass_index) + 1
        if prompts is not None and following < len(prompts) and h3_assets.cuts_into(
                prompts[following].get("continuity", h3_conditioning.DEFAULT_CONTINUITY),
                prompts[following].get("overlap", 0),
                h3_conditioning.source_of(prompts, following), following):
            # A closing frame moves onto the last frame the next segment's cut keeps.
            positive = h3_assets.closing_moved(positive, h3_extend.CLIP_LEAD)
        # A segment the window adds references to is sampled by ref2va where its row says auto.
        referenced = moments or (int(pass_index) > 0 and picture in (
            h3_extend.REFERENCE_VIDEO, h3_extend.REFERENCE_SAMPLE))
        model = h3_conditioning.model_or_blocker(
            prompts, pass_index, "H3 Extend Window", referenced or None)
        kind = h3_conditioning.model_of(prompts, pass_index, referenced or None)[1] if prompts else ""
        if kind:
            named += f" on {kind}"
        picked = int(pass_index) - 1
        if int(pass_index) > 0:
            if prompts is not None:
                source = h3_conditioning.source_of(prompts, pass_index)
            picked = h3_conditioning.resolved_source(source, pass_index)
        sample_rows = (0, None)
        if sample_span == SAMPLE_SPANS[0] and int(pass_index) > 0:
            sample_rows = h3_extend.segment_rows(latent, picked)
        if bridging and int(pass_index) > 0:
            return cls.bridged(latent, positive, extension_frames, overlap_frames,
                               audio_release, named, model, vae=vae, picture=picture,
                               reference_sound=bool(reference_sound), prompts=prompts,
                               pass_index=pass_index, reencode=bool(reencode_prompt),
                               refresh_gain=refresh_gain, samples=reference_samples,
                               sample_rows=sample_rows)
        continuity = picture
        earlier = False
        seen = 0
        if int(pass_index) > 0:
            if picked != int(pass_index) - 1:
                ends = h3_extend.segment_ends(latent) or []
                seen = h3_extend.seen_rows(ends[picked]) if picked < len(ends) else 0
                latent = h3_extend.until_segment(latent, picked)
                named += f" from segment {picked + 1}"
                earlier = True
        if continuity == h3_extend.REFERENCE_SAMPLE and int(pass_index) > 0:
            return cls.sampled(latent, positive, vae, extension_frames, reference_samples,
                               named, model, prompts, pass_index, bool(reencode_prompt),
                               sample_rows)
        handing = continuity in ("handoff", h3_extend.REFERENCE_VIDEO)
        if int(pass_index) <= 0 or continuity == "cut" or (
                int(overlap_frames) <= 0 and not handing):
            # Either nothing has been rendered yet, or this segment cuts somewhere new.
            video, audio = h3_extend.split(latent)
            length = (h3_extend.frames_for(video.shape[2]) if int(pass_index) <= 0
                      else h3_extend.snap_clip(extension_frames))
            fresh = h3_extend.empty_like(video, audio, length)
            how = "opening" if int(pass_index) <= 0 else "cutting to"
            return io.NodeOutput(
                fresh, positive, 0, f"{named}: {how} a fresh {length} frame scene", model,
            )

        overlap = h3_extend.snap_overlap(overlap_frames)
        extension = h3_extend.snap_extension(extension_frames)
        video, audio = h3_extend.split(latent)
        # Rows the overlap covers, which a handoff or a reference decodes at least.
        tail_rows = h3_extend.tokens_for(overlap)

        if continuity == "handoff":
            import node_helpers

            frame, kept, note = cls.handoff_frame(video, vae, prompts, tail_rows, refresh_gain)
            block = {"resolved_frame_index": 0, "latent": vae.encode(frame)}
            handed, encoded = cls.with_references(prompts, pass_index, positive, [], [],
                                                  [frame], bool(reencode_prompt))
            handed = node_helpers.conditioning_set_values(
                handed, {"minimax_keyframes": [block]}, append=True
            )
            length = h3_extend.snap_clip(extension)
            report = (
                f"{named}: {note}{encoded.replace('with them', 'with it')}; "
                f"sampling {length} fresh frames"
            )
            return io.NodeOutput(
                h3_extend.empty_like(video, audio, length), handed, 0, report, model,
            )

        if continuity == h3_extend.REFERENCE_VIDEO:
            import node_helpers

            block, count, ref_w, ref_h, items = cls.last_frames(
                video, vae, tail_rows, h3_extend.REFERENCE_VIDEO, "carry", positive,
                audio if reference_sound else None)
            with_reference, encoded = cls.with_references(
                prompts, pass_index, positive, [block], items, reencode=bool(reencode_prompt))
            length = h3_extend.snap_clip(extension)
            sounding = " with its sound" if block.get("audio_latent") is not None else ""
            return io.NodeOutput(
                h3_extend.empty_like(video, audio, length), with_reference, 0,
                f"{named}: referencing {count} decoded frames at {ref_w}x{ref_h}{sounding}"
                f"{encoded}; sampling {length} fresh frames",
                model,
            )

        video_tail, audio_tail = h3_extend.tail(latent, overlap)
        skip = 0
        if earlier:
            skip = h3_extend.rejoin_skip(seen, video.shape[2], video_tail.shape[2])
        window = h3_extend.window_frames(overlap, extension)
        tokens = h3_extend.tokens_for(window)
        # The soundtrack is held for the whole latent steps that fit inside the overlap; a
        # return to an earlier segment starts its audio fresh.
        wanted_audio = 0 if earlier or fresh_sound else h3_extend.audio_carry(overlap)
        rows = video_tail.shape[2]

        held_rows, scale, gain = h3_extend.settled(video, rows, drift_control)
        detail = f"settled at scale {scale:.4f} and detail gain {gain:.4f}"
        renewed = 0.0
        if continuity == "refresh":
            renewed = max(0.0, min(1.0, float(renewal)))
            detail += (
                f"; renewing the {rows} carried rows at {renewed:.2f} of each step's noise"
            )

        release = max(0, min(int(audio_release), wanted_audio))
        carried, mask, held_audio = h3_extend.masked_window(
            held_rows, audio_tail, tokens, h3_extend.audio_span(window), wanted_audio,
            release, renewed,
        )
        carried["noise_mask"] = mask["samples"]
        if earlier:
            carried[h3_extend.REJOIN_KEY] = int(skip)

        held = h3_extend.frames_for(video.shape[2])
        lead = h3_extend.audio_lead(overlap)
        seam = ("on the audio grid exactly" if h3_extend.audio_lands_whole(overlap)
                else f"{lead * 1000:.1f}ms short of the seam")
        sound = (f"{held_audio} audio steps held, released over {release}, {seam}"
                 if held_audio else "picture only")
        report = (
            f"{named}: "
            f"holding {overlap} frames ({video_tail.shape[2]} tokens, {sound}) of a {held} "
            f"frame clip {'renewed' if renewed else 'at sigma 0'} inside a {window} frame "
            f"window ({tokens} tokens); "
            f"generating {extension} new frames for a {held + extension} frame clip; "
            f"{detail}"
        )
        return io.NodeOutput(carried, positive, overlap, report, model)

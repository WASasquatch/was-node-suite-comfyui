"""Stretching the fast frames of a MiniMax H3 clip into a latent for a refining pass."""

from __future__ import annotations

import time

from comfy_api.latest import io, ui

from ...modules import log
from ...modules.compat.types import H3_DEROPE
from ...modules.interface.progress import progress_bar
from ...modules.latent import h3_derope, h3_extend, h3_preview, h3_references
from ...modules.model import h3_low_vram, release_models

MANUAL = "manual"

#: Progress steps the audio stretch and encode count for.
AUDIO_STEPS = 8

logger = log.get_logger("latent.h3_derope")


def _coverage_options() -> list:
    """The coverage presets, then manual with its own settings."""
    presets = [io.DynamicCombo.Option(key=name, inputs=[]) for name in h3_derope.COVERAGE]
    return presets + [
        io.DynamicCombo.Option(
            key=MANUAL,
            inputs=[
                io.Float.Input(
                    "threshold", default=0.75, min=0.5, max=0.99, step=0.01,
                    tooltip="Rows with motion above this share of their shot are held, as "
                            "`0.75` for the fastest quarter or `0.85` for the fastest sixth.",
                ),
                io.Int.Input(
                    "peak_hold", default=4, min=2, max=8,
                    tooltip="Times the fastest frames are shown, as `4` or `3`.",
                ),
                io.Int.Input(
                    "bridge", default=8, min=0, max=20,
                    tooltip="Latent rows between two held spans that are held too, as `8`, "
                            "or `0` to hold only the fastest rows.",
                ),
                io.Boolean.Input(
                    "ramp", default=True,
                    tooltip="`true` steps each held span down by one hold per row at its edges.",
                ),
                io.Float.Input(
                    "floor", default=h3_derope.FLOOR, min=0.0, max=10.0, step=0.1,
                    tooltip="Motion a row needs to be held, as a multiple of its scene's typical "
                            "motion: `1.6`; `0` holds each shot's fastest share however calm.",
                ),
            ],
        ),
    ]


def _audio_options() -> list:
    """The audio presets, then manual with its strength."""
    presets = [io.DynamicCombo.Option(key=name, inputs=[]) for name in h3_derope.AUDIO_MODES]
    return presets + [
        io.DynamicCombo.Option(
            key=MANUAL,
            inputs=[
                io.Float.Input(
                    "audio_strength", default=0.5, min=0.0, max=1.0, step=0.05,
                    tooltip="Share of the audio the pass re-renders, as `0.5`, `0.0` to keep "
                            "it or `1.0` to replace it.",
                ),
            ],
        ),
    ]


class H3DeRopeStretch(io.ComfyNode):
    """Hold a clip's fast frames and encode it as the start latent for a sampler."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASH3DeRopeStretch",
            display_name="H3 De-RoPE Stretch",
            search_aliases=[
                "WASH3DeRopeStretch",
                "H3 De-RoPE Stretch",
                "de-rope",
                "derope",
                "motion blur",
                "fast motion",
                "minimax h3",
            ],
            category="WAS Suite/Latent/Video",
            description=(
                "Slow down the fast parts of a MiniMax H3 clip, shot by shot, so H3 De-RoPE "
                "Refine can redraw their smeared motion; calm clips pass through untouched. H3 "
                "De-RoPE Recover then restores the clip's timing. Works on one clip or on each "
                "segment inside a Prompt Timeline loop."
            ),
            inputs=[
                io.Image.Input(
                    "images", optional=True,
                    tooltip="The clip's frames at 24 fps. Leave empty to decode them from latent.",
                ),
                io.Vae.Input("vae", tooltip="The H3 video VAE."),
                io.DynamicCombo.Input(
                    "mode", options=_coverage_options(), display_name="mode",
                    tooltip="How much of each shot is held, of rows moving at least 1.6x their "
                            "scene's typical motion: `balanced` = fastest quarter at 4; `wide` = "
                            "fastest 30% at 4; `economy` = fastest 15% at 3; `manual` = every "
                            "setting.",
                ),
                io.Float.Input(
                    "strength", default=0.5, min=0.05, max=1.0, step=0.05,
                    tooltip="Denoise for the sampler, passed out as `denoise`: `0.5` keeps the "
                            "clip's motion and redraws the smear, `0.7` redraws more and "
                            "re-times the action. Set the sampler's steps to the first pass's "
                            "times this, as `4` after an 8-step pass or `13` after 25.",
                ),
                io.DynamicCombo.Input(
                    "audio_mode", options=_audio_options(), display_name="audio_mode",
                    tooltip="How much of the audio the pass re-renders: `follow` 0.5, `loose` "
                            "0.7, `pin` keeps it, `fresh` replaces it. Needs audio and audio_vae.",
                ),
                io.Audio.Input(
                    "audio", optional=True,
                    tooltip="The clip's soundtrack, from Get Video Components. Left empty, a "
                            "joint H3 latent's own audio is used. Without either, held spans "
                            "come back rushed.",
                ),
                io.Vae.Input(
                    "audio_vae", optional=True,
                    tooltip="The H3 audio VAE, for encoding the audio and decoding a latent's.",
                ),
                io.Float.Input(
                    "fps", default=24.0, min=1.0, max=240.0, step=0.001, optional=True,
                    tooltip="Frame rate of the clip, as `24`, for timing its audio.",
                ),
                io.Latent.Input(
                    "latent", optional=True,
                    tooltip="The clip's finished H3 latent, as a sampler's output. Motion, audio "
                            "and scene cuts are read from it; rows its noise mask holds, as a "
                            "segment's carried frames, stay as they are. An unfinished estimate "
                            "comes back fast-forwarded.",
                ),
                io.Model.Input(
                    "model", optional=True,
                    tooltip="The model to refine with. In a loop, wire H3 Extend Window's model "
                            "output, so each scene is refined by the model it chose; for one clip, "
                            "the loader's model. Passed out without the timeline's live previews.",
                ),
                io.Boolean.Input(
                    "low_vram", default=True, optional=True,
                    tooltip="`true` = each model block runs in 8192-token slices with only the "
                            "weights that fit kept on the card; same output, lower memory peak. "
                            "Needs model.",
                ),
                io.Conditioning.Input(
                    "positive", optional=True,
                    tooltip="The prompt the clip was sampled with, from H3 Extend Window or the "
                            "conditioning node. Its pinned frames, as a first or last frame, are "
                            "moved to where those frames land in the stretched clip.",
                ),
            ],
            outputs=[
                io.Latent.Output(
                    display_name="latent",
                    tooltip="The stretched clip and its audio, for the sampler's latent_image. A "
                            "clip with cuts carries where each scene opens, for H3 Decode Video.",
                ),
                H3_DEROPE.Output(
                    display_name="derope",
                    tooltip="What was held, for H3 De-RoPE Recover.",
                ),
                io.Float.Output(
                    display_name="denoise",
                    tooltip="The strength, for the sampler's denoise.",
                ),
                io.String.Output(
                    display_name="report",
                    tooltip="Frames in and out, frames held, the peak hold, clips kept as "
                            "sampled, shot cuts found and each shot's fastest motion.",
                ),
                io.Model.Output(
                    display_name="model",
                    tooltip="The model for the sampler, run in slices when low_vram is on.",
                ),
                io.Conditioning.Output(
                    display_name="positive",
                    tooltip="The prompt with its pinned frames on the stretched clip, for the "
                            "sampler's positive. Empty when positive is not wired.",
                ),
            ],
        )

    @classmethod
    def execute(cls, vae, mode, strength, audio_mode, images=None, audio=None, audio_vae=None,
                fps=24.0, latent=None, model=None, low_vram=True, positive=None) -> io.NodeOutput:
        """Fit, measure, hold and encode the clip.

        Raises:
            ValueError: Neither images nor latent is wired, the latent is not an H3 video
                latent, or audio is wired without audio_vae.
        """
        import torch

        if images is None and latent is None:
            raise ValueError(
                "H3 De-RoPE Stretch has nothing to stretch. Wire the clip's frames into images, "
                "its H3 latent into latent, or both"
            )
        if audio is not None and audio_vae is None:
            raise ValueError(
                "H3 De-RoPE Stretch has audio and no audio_vae to encode it with. Wire the "
                "H3 audio VAE into audio_vae, or unwire the audio"
            )
        chosen = mode["mode"]
        if chosen == MANUAL:
            quantile, peak, bridge = mode["threshold"], mode["peak_hold"], mode["bridge"]
            ramp = bool(mode["ramp"])
            floor = float(mode.get("floor", h3_derope.FLOOR))
        else:
            quantile, peak, bridge, floor = h3_derope.COVERAGE[chosen]
            ramp = True
        heard = audio_mode["audio_mode"]
        audio_strength = (float(audio_mode["audio_strength"]) if heard == MANUAL
                          else h3_derope.AUDIO_MODES[heard])

        clock = [time.perf_counter()]
        spent = {}

        def lap(name):
            now = time.perf_counter()
            spent[name] = now - clock[0]
            clock[0] = now

        release_models()
        given = h3_derope.video_of(latent) if latent is not None else None
        starts = h3_extend.scene_starts(latent) if latent is not None else None
        heard_from = "wired"
        if audio is None and latent is not None and audio_vae is not None:
            audio = h3_derope.latent_audio(audio_vae, latent)
            heard_from = "from the latent"
        count = (int(images.shape[0]) if images is not None
                 else h3_extend.frames_for(given.shape[2]))
        # Weighted by frames passed through the VAE: decode, measure, then about three
        # times as many stretched.
        bar = progress_bar(count * 5 + AUDIO_STEPS)
        decoded = images is None
        if decoded:
            images = h3_derope.decode_scenes(vae, given, starts)
            lap("decode")
        bar.update(count)

        source, heard_audio = int(images.shape[0]), audio
        images, audio = h3_derope.fit(images, audio, fps)
        reused = given is not None and given.shape[2] == h3_extend.tokens_for(images.shape[0])
        source_latent = given if reused else vae.encode(images[..., :3])
        # Scene starts are read only where the frames follow the latent's timeline.
        scenes = starts if decoded or reused else None
        cuts = [start for start, _ in h3_derope.scene_spans(scenes, source_latent.shape[2])[1:]]
        # Rows a carried window holds keep their frames and are pinned through the pass.
        pinned = (min(h3_derope.pinned_rows(latent), source_latent.shape[2] - 1)
                  if reused else 0)
        profile = h3_derope.motion(source_latent, scenes)
        # Shot cuts inside a scene; a scene's own first frame is already a cut.
        opening = {start for start, _ in h3_derope.scene_frames(scenes, images.shape[0])}
        shots = [frame for frame in h3_derope.shot_cuts(images) if frame not in opening]
        bar.update(count)
        lap("measure")
        # A segment of a loop keeps its last clip unheld, so the next one carries it at pace.
        tail = 0
        if reused and h3_extend.WINDOW_KEY in latent:
            tail = source_latent.shape[2] - h3_derope.row_of_frame(
                max(0, int(images.shape[0]) - h3_extend.CLIP_FRAMES))
        holds, row_holds, peaks = h3_derope.plan(profile, images.shape[0], quantile, peak, bridge,
                                                 ramp, scenes, pinned, tail, shots, floor)
        holds = h3_derope.aligned(h3_derope.settled(holds, scenes), scenes)
        video, encoded, chunks = h3_derope.encode_stretched(vae, images, holds, source_latent,
                                                            pinned, scenes)
        bar.update(count * 3)
        lap("encode")

        span = h3_extend.audio_span(sum(holds))
        if audio is not None:
            slowed = h3_derope.stretch_audio(audio, holds, fps)
            sound, _ = h3_references.audio_latent(audio_vae, slowed)
            sound = sound[..., :span]
            if sound.shape[-1] < span:
                sound = torch.nn.functional.pad(sound, (0, span - sound.shape[-1]))
        else:
            sound = torch.zeros([video.shape[0], 32, 2, span], dtype=video.dtype,
                                device=video.device)
        kept_sound, joint = h3_derope.source_sound(latent if reused else None, audio, audio_vae,
                                                   images.shape[0])
        mask = h3_derope.pass_mask(audio_strength if audio is not None else 1.0, video, sound,
                                   holds, pinned)
        bar.update(AUDIO_STEPS)
        release_models()
        lap("audio")
        extras = ({key: value for key, value in latent.items() if key != "samples"}
                  if reused else None)
        latent = h3_extend.join(video, sound)
        if cuts:
            latent = h3_extend.with_scenes(latent, h3_derope.stretched_starts(holds, scenes))
        if mask is not None:
            latent["noise_mask"] = mask
        moved = 0
        if positive is not None:
            positive, moved = h3_derope.moved_keyframes(positive, holds)

        result = h3_derope.Plan(
            holds=tuple(holds), fps=float(fps), audio=heard_audio,
            rows=tuple(float(value) for value in profile), row_holds=tuple(row_holds),
            source=source, cuts=tuple(cuts), sound=kept_sound, joint=joint, extras=extras,
            pinned=pinned, guide=h3_derope.guide_frames(images), shots=tuple(shots),
            video=source_latent.detach().cpu(),
        )
        held = sum(1 for hold in holds[:source] if hold > 1)
        flags = h3_derope.kept_clips(holds)
        unread = len(h3_derope.scene_spans(starts, given.shape[2])) - 1 if given is not None else 0
        fastest = ", ".join(f"{value:.1f}x" for value in peaks)
        report = (
            f"{source} frames -> {result.stretched} ({result.stretched / source:.2f}x); "
            f"{held} held, peak x{max(row_holds)}; {chosen}; "
            f"{sum(flags)} of {len(flags)} clips kept as sampled; "
            f"{len(shots)} shot {'cut' if len(shots) == 1 else 'cuts'} found"
            + (f" at {', '.join(f'{frame / float(fps):.1f}' for frame in shots)} s" if shots else "")
            + f"; fastest row per shot against its scene's typical motion: {fastest}, "
              f"held from {floor:g}x; {encoded} of {chunks} chunks encoded"
            + (f"; {len(cuts)} {'cut' if len(cuts) == 1 else 'cuts'} kept out of the motion "
               f"measure, each scene {'decoded, ' if decoded else ''}held and encoded on its own"
               if cuts else "")
            + (f"; the latent's {unread} {'cut' if unread == 1 else 'cuts'} went unread, as the "
               f"frames wired are not its length" if unread and not cuts else "")
            + (f"; {pinned} carried rows ({h3_derope.first_frame_of_row(pinned)} frames) held "
               f"as they are" if pinned else "")
            + (f"; the last {tail} rows left unheld for the next segment's carry" if tail else "")
            + (f"; {moved} pinned {'frame' if moved == 1 else 'frames'} moved onto the "
               f"stretched clip" if moved else "")
            + (f"; audio {heard_from}, re-rendered at {audio_strength:g}" if audio is not None
               else "; no audio, so the pass invents its own at natural pace and held spans "
                    "can come back rushed")
            + "; " + ", ".join(f"{name} {seconds:.1f}s" for name, seconds in spent.items())
        )
        if audio is None:
            logger.warning(
                "H3 De-RoPE Stretch has no audio to seed the pass with. Wire the clip's audio, "
                "or a joint H3 latent with audio_vae, or held spans come back rushed"
            )
        logger.info("H3 De-RoPE Stretch: %s", report)
        picture = h3_derope.plot(result)
        if model is not None:
            model = h3_preview.unwatched(model)
        if model is not None and low_vram:
            if not h3_low_vram.is_low_vram(model):
                model = h3_low_vram.low_vram_model(model)
        return io.NodeOutput(latent, result, float(strength), report, model, positive,
                             ui=ui.PreviewImage(picture, cls=cls))

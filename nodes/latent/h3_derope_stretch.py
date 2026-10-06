"""Stretching the fast frames of a MiniMax H3 clip into a latent for a refining pass."""

from __future__ import annotations

import time

from comfy_api.latest import io, ui

from ...modules import log
from ...modules.compat.types import H3_DEROPE
from ...modules.interface.progress import progress_bar
from ...modules.latent import h3_derope, h3_extend, h3_references
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
                    tooltip="Rows with motion above this share of the clip are held, as "
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
                "Find where a MiniMax H3 clip moves too fast, show those frames several times "
                "over, and encode the result with its audio as the start latent for any "
                "sampler. The clip comes in as frames, as a latent, or both; for a clip "
                "generated from text, wire the finished output of its sampler. Sample it at "
                "the `denoise` output's strength, decode it, and H3 De-RoPE Recover puts "
                "the frames back on the clip's timing. A clip joined from several scenes keeps "
                "its cuts when its latent is wired: each scene is measured and held on its own, "
                "and the stretched latent records where each one opens. The GPU's models are "
                "released before and after its VAE work, so the samplers either side of it get "
                "the whole card. The strip shows the motion of each latent row, what was held "
                "and where the scenes cut."
            ),
            inputs=[
                io.Image.Input(
                    "images", optional=True,
                    tooltip="The clip's frames at 24 fps. Leave empty to decode them from latent.",
                ),
                io.Vae.Input("vae", tooltip="The H3 video VAE."),
                io.DynamicCombo.Input(
                    "mode", options=_coverage_options(), display_name="mode",
                    tooltip="How much of the clip is held. `balanced` holds the fastest "
                            "quarter at 4, `wide` the fastest 30% at 4, `economy` the fastest "
                            "15% at 3. `manual` shows every setting.",
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
                    "fps", default=24.0, min=1.0, max=120.0, step=0.001, optional=True,
                    tooltip="Frame rate of the clip, as `24`, for timing its audio.",
                ),
                io.Latent.Input(
                    "latent", optional=True,
                    tooltip="The clip's finished H3 latent, such as a sampler's output. Motion, "
                            "audio and scene cuts are read from it instead of encoding the "
                            "frames, and no hold crosses a cut. An early, unfinished estimate "
                            "carries no motion to keep, and comes back fast-forwarded.",
                ),
                io.Model.Input(
                    "model", optional=True,
                    tooltip="The H3 model the sampler uses. Passed out for the sampler's model.",
                ),
                io.Boolean.Input(
                    "low_vram", default=True, optional=True,
                    tooltip="`true` runs each model block over the stretched clip in slices of "
                            "8192 tokens and keeps only the weights that fit beside it on the "
                            "card, streaming the rest, for the same output at a lower memory "
                            "peak. Needs model.",
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
                    tooltip="Frames in and out, frames held, the peak hold and the cuts kept "
                            "out of the motion measure.",
                ),
                io.Model.Output(
                    display_name="model",
                    tooltip="The model for the sampler, run in slices when low_vram is on.",
                ),
            ],
        )

    @classmethod
    def execute(cls, vae, mode, strength, audio_mode, images=None, audio=None, audio_vae=None,
                fps=24.0, latent=None, model=None, low_vram=True) -> io.NodeOutput:
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
        else:
            quantile, peak, bridge = h3_derope.COVERAGE[chosen]
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
        profile = h3_derope.motion(source_latent, scenes)
        bar.update(count)
        lap("measure")
        holds, row_holds = h3_derope.plan(profile, images.shape[0], quantile, peak, bridge, ramp,
                                          scenes)
        holds = h3_derope.aligned(holds, scenes)
        video, encoded, chunks = h3_derope.encode_stretched(vae, images, holds, source_latent)
        bar.update(count * 3)
        lap("encode")

        span = h3_extend.audio_span(sum(holds))
        mask = None
        if audio is not None:
            slowed = h3_derope.stretch_audio(audio, holds, fps)
            sound, _ = h3_references.audio_latent(audio_vae, slowed)
            sound = sound[..., :span]
            if sound.shape[-1] < span:
                sound = torch.nn.functional.pad(sound, (0, span - sound.shape[-1]))
            if audio_strength < 1.0:
                mask = h3_derope.audio_row_strength(audio_strength, video, sound)
        else:
            sound = torch.zeros([video.shape[0], 32, 2, span], dtype=video.dtype,
                                device=video.device)
        bar.update(AUDIO_STEPS)
        release_models()
        lap("audio")
        latent = h3_extend.join(video, sound)
        if cuts:
            latent = h3_extend.with_scenes(latent, h3_derope.stretched_starts(holds, scenes))
        if mask is not None:
            latent["noise_mask"] = mask

        result = h3_derope.Plan(
            holds=tuple(holds), fps=float(fps), audio=heard_audio,
            rows=tuple(float(value) for value in profile), row_holds=tuple(row_holds),
            source=source, cuts=tuple(cuts),
        )
        held = sum(1 for hold in holds[:source] if hold > 1)
        unread = len(h3_derope.scene_spans(starts, given.shape[2])) - 1 if given is not None else 0
        report = (
            f"{source} frames -> {result.stretched} ({result.stretched / source:.2f}x); "
            f"{held} held, peak x{max(holds)}; {chosen}; {encoded} of {chunks} chunks encoded"
            + (f"; {len(cuts)} {'cut' if len(cuts) == 1 else 'cuts'} kept out of the motion "
               f"measure, each scene {'decoded, ' if decoded else ''}held and encoded on its own"
               if cuts else "")
            + (f"; the latent's {unread} {'cut' if unread == 1 else 'cuts'} went unread, as the "
               f"frames wired are not its length" if unread and not cuts else "")
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
        if model is not None and low_vram:
            if not h3_low_vram.is_low_vram(model):
                model = h3_low_vram.low_vram_model(model)
        return io.NodeOutput(latent, result, float(strength), report, model,
                             ui=ui.PreviewImage(picture, cls=cls))

"""Refining a stretched MiniMax H3 clip window by window, each carrying the one before."""

from __future__ import annotations

import comfy.samplers
from comfy_api.latest import io

from ...modules import log
from ...modules.latent import h3_derope, h3_upscale

logger = log.get_logger("latent.h3_derope")


class H3DeRopeRefine(io.ComfyNode):
    """Sample a stretched clip in windows, each holding the clip it shares with the last."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASH3DeRopeRefine",
            display_name="H3 De-RoPE Refine",
            search_aliases=[
                "WASH3DeRopeRefine",
                "H3 De-RoPE Refine",
                "de-rope",
                "derope",
                "minimax h3",
            ],
            category="WAS Suite/Latent/Video",
            description=(
                "Refine the stretched latent from H3 De-RoPE Stretch window by window, each the "
                "length of a normal H3 pass and carrying the clip it shares with the one before, "
                "so a long stretch fits in memory and keeps its detail to the end. Send the "
                "result to H3 De-RoPE Recover."
            ),
            inputs=[
                io.Model.Input(
                    "model", tooltip="The H3 model, from H3 De-RoPE Stretch's model output.",
                ),
                io.Conditioning.Input(
                    "positive", tooltip="The prompt, from H3 De-RoPE Stretch's positive output.",
                ),
                io.Latent.Input(
                    "latent", tooltip="The stretched latent, from H3 De-RoPE Stretch.",
                ),
                io.Float.Input(
                    "denoise", default=0.5, min=0.05, max=1.0, step=0.05,
                    tooltip="Share of the schedule each window runs, as `0.5`. Wire H3 De-RoPE "
                            "Stretch's denoise output here.",
                ),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True,
                    tooltip="Noise seed of the first window, as `20261003`; each later window "
                            "adds its number.",
                ),
                io.Int.Input(
                    "steps", default=4, min=1, max=10000,
                    tooltip="Steps each window runs, as `4` after an 8-step pass or `13` after "
                            "25.",
                ),
                io.Float.Input(
                    "cfg", default=1.0, min=0.0, max=100.0, step=0.1, round=0.01,
                    tooltip="Guidance, `1` for turbo LoRAs and distilled models.",
                ),
                io.Combo.Input(
                    "sampler_name", options=comfy.samplers.KSampler.SAMPLERS,
                    default="res_multistep",
                    tooltip="The solver each step runs, as `res_multistep` or `euler`.",
                ),
                io.Combo.Input(
                    "scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="simple",
                    tooltip="How the noise level falls from step to step, as `simple`.",
                ),
                io.Int.Input(
                    "window_frames", default=362, min=22, max=362, step=17,
                    tooltip="Stretched frames one window holds at most: `362` = H3's longest, "
                            "one pass for most segments; `243` = about 10 seconds; smaller saves "
                            "memory, but a pinned last frame then shapes only the final window.",
                ),
                io.Int.Input(
                    "overlap_frames", default=17, min=0, max=102, step=17,
                    tooltip="Stretched frames each window carries from the one before, held as "
                            "refined: `17` = one clip; `34` = two, a steadier join; `0` = none.",
                ),
                io.Conditioning.Input(
                    "negative", optional=True,
                    tooltip="Negative prompt. Left empty, the positive zeroed, as at cfg `1`.",
                ),
                io.Sampler.Input(
                    "sampler", optional=True,
                    tooltip="A SAMPLER run in place of sampler_name, as an affine-wrapped one.",
                ),
            ],
            outputs=[
                io.Latent.Output(
                    display_name="latent",
                    tooltip="The refined stretched latent, for H3 De-RoPE Recover's latent.",
                ),
                io.String.Output(
                    display_name="report",
                    tooltip="Each window's stretched frames, the frames it held and its time.",
                ),
            ],
        )

    @classmethod
    def execute(cls, model, positive, latent, denoise, seed, steps, cfg, sampler_name,
                scheduler, window_frames, overlap_frames, negative=None,
                sampler=None) -> io.NodeOutput:
        """Plan the windows and sample each one, carrying the clips it shares.

        Raises:
            ValueError: The latent is not a MiniMax H3 joint latent.
        """
        chunk = h3_upscale.snap_chunk(window_frames)
        shared = h3_upscale.snap_overlap(overlap_frames, chunk)
        refined, lines = h3_derope.refine(
            model, positive, latent, steps=steps, cfg=cfg, sampler_name=sampler_name,
            scheduler=scheduler, denoise=denoise, seed=seed, window_frames=chunk,
            overlap_frames=shared, negative=negative, sampler=sampler,
        )
        report = (f"{len(lines)} window(s) of up to {chunk} frames, {shared} carried; "
                  + "; ".join(lines))
        logger.info("H3 De-RoPE Refine: %s", report)
        return io.NodeOutput(refined, report)

"""Sample a run of latents from a schedule of conditionings."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ....modules import log
from ....modules.compat.types import CONDITIONING_SEQ
from ....modules.sampling import sampler_names, scheduler_names
from ....modules.sampling.sequence import (
    alternate_seed,
    LATENT_INTERPOLATION_MODES,
    MAX_SEED,
    SEED_MODES,
    advance_seed,
    blend_conditioning,
    interpolate_latents,
    unsample,
)

REQUIRES = "extras"

logger = log.get_logger("nodes.extras.sampling")


def conditioning_for_loop(conditioning_seq, loop_count: int, last_conditioning):
    """The conditioning a given loop runs on.

    Args:
        conditioning_seq: ``(frame index, [tensor, dict])`` pairs.
        loop_count: The loop about to run, counting from zero.
        last_conditioning: The pair the previous loop used, or ``None`` on the first loop.

    Returns:
        The pair whose frame index is this loop, the previous loop's pair where no entry
        names this loop, or ``None`` when neither exists.
    """
    for idx, conditioning, *_ in conditioning_seq:
        if int(idx) == loop_count:
            return conditioning
    return last_conditioning if last_conditioning else None


class KSamplerSequence(io.ComfyNode):
    """Sample one latent per loop, each carrying on from the last."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASKSamplerSeq",
            display_name="KSampler Sequence",
            search_aliases=[
                "WASKSamplerSeq",
                "KSampler Sequence",
                "prompt travel",
                "animation sampler",
                "conditioning schedule",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Run the sampler once per loop and stack the results into one latent "
                "batch, switching prompt as the frame schedule from CLIP Text Encode "
                "Sequence (Advanced) says to. Each loop starts from the previous loop's "
                "latent at a lower denoise, so the run reads as a moving picture rather "
                "than as unrelated images. Decode the batch and save it as frames."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="The diffusion model every loop in the run samples with.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=MAX_SEED,
                    tooltip=(
                        "The seed the first loop runs on, and the base every later loop's seed "
                        "is worked out from. The same seed replays the whole run; change it "
                        "for a different one. Any whole number; `0` is as good a seed as any."
                    ),
                ),
                io.Combo.Input(
                    "seed_mode_seq",
                    options=SEED_MODES,
                    tooltip=(
                        "How the seed moves from loop to loop. 'increment' and 'decrement' "
                        "step it by one, which keeps consecutive frames close and the run "
                        "smooth; 'random' picks a fresh seed each loop, which makes every "
                        "frame its own image; 'fixed' holds one seed for the whole run, so "
                        "only the prompt and the denoise change anything."
                    ),
                ),
                io.Boolean.Input(
                    "alternate_values",
                    default=True,
                    tooltip=(
                        "Whether every other loop runs on a second seed that drifts away "
                        "from the first instead of on the stepped one. It gives the run a "
                        "slight back-and-forth flicker between two looks, which reads as "
                        "movement in a short sequence. Turn it off for a single steady "
                        "progression."
                    ),
                ),
                io.Int.Input(
                    "steps",
                    default=20,
                    min=1,
                    max=10000,
                    tooltip=(
                        "Sampling steps per loop. Around 20 suits most models; more takes "
                        "proportionally longer, and the whole run is this many steps times "
                        "sequence_loop_count."
                    ),
                ),
                io.Float.Input(
                    "cfg",
                    default=8.0,
                    min=0.0,
                    max=100.0,
                    step=0.5,
                    round=0.01,
                    tooltip=(
                        "How closely each loop is held to its prompt. Around 7-8 suits most "
                        "models; lower is looser and softer, much higher burns contrast and "
                        "makes a sequence flicker."
                    ),
                ),
                io.Combo.Input(
                    "sampler_name",
                    options=sampler_names(),
                    tooltip=(
                        "The sampling algorithm. 'euler' is the plain, predictable choice "
                        "and the steadiest across a sequence; the 'ancestral' and 'sde' "
                        "variants add fresh noise as they go, which adds detail and also "
                        "adds flicker frame to frame. The list is whatever this ComfyUI "
                        "offers."
                    ),
                ),
                io.Combo.Input(
                    "scheduler",
                    options=scheduler_names(),
                    tooltip=(
                        "How the noise level is stepped down within each loop. 'normal' and "
                        "'karras' are the usual choices, karras spending more steps at low "
                        "noise where fine detail is decided. The list is whatever this "
                        "ComfyUI offers."
                    ),
                ),
                io.Int.Input(
                    "sequence_loop_count",
                    default=20,
                    min=1,
                    max=1024,
                    step=1,
                    tooltip=(
                        "How many loops to run, which is how many latents come out. At 20 "
                        "the output is a 20-image batch; the frame indices in the "
                        "conditioning schedule are counted against this same number."
                    ),
                ),
                CONDITIONING_SEQ.Input(
                    "positive_seq",
                    tooltip=(
                        "The positive prompt schedule from CLIP Text Encode Sequence "
                        "(Advanced): pairs of frame index and conditioning. A loop with no "
                        "entry of its own keeps the last one it was given, so a prompt "
                        "stays in force until the next index in the list."
                    ),
                ),
                CONDITIONING_SEQ.Input(
                    "negative_seq",
                    tooltip=(
                        "The negative prompt schedule, read exactly as positive_seq is. It "
                        "needs at least one entry at frame 0, since a loop with nothing to "
                        "fall back on has no negative prompt at all."
                    ),
                ),
                io.Boolean.Input(
                    "use_conditioning_slerp",
                    default=False,
                    tooltip=(
                        "Whether the prompt changes gradually instead of switching over on "
                        "one frame. On, each loop's conditioning is interpolated towards the "
                        "one before it by cond_slerp_strength, which is what turns a list of "
                        "prompts into a blend rather than a cut."
                    ),
                ),
                io.Float.Input(
                    "cond_slerp_strength",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    tooltip=(
                        "How far each loop moves towards the new prompt when "
                        "use_conditioning_slerp is on. 0.0 keeps the previous prompt, 1.0 "
                        "takes the new one whole, 0.5 sits halfway between them. Ignored "
                        "while that switch is off."
                    ),
                ),
                io.Latent.Input(
                    "latent_image",
                    tooltip=(
                        "The latent the first loop starts from, which also sets the size of "
                        "every frame. An empty latent generates from scratch; an encoded "
                        "image starts the sequence on that picture."
                    ),
                ),
                io.Boolean.Input(
                    "use_latent_interpolation",
                    default=False,
                    tooltip=(
                        "Whether each new latent is mixed back towards the previous frame "
                        "before it is kept. It damps down how much can change between two "
                        "frames, which is the main handle on how jumpy the finished sequence "
                        "looks."
                    ),
                ),
                io.Combo.Input(
                    "latent_interpolation_mode",
                    options=LATENT_INTERPOLATION_MODES,
                    tooltip=(
                        "How the previous frame is mixed in. 'Blend' is a straight average; "
                        "'Slerp' travels along the arc between the two latents and holds "
                        "contrast better; 'Cosine Interp' is a blend that eases in and out, "
                        "so each frame is held a little longer. Ignored while "
                        "use_latent_interpolation is off."
                    ),
                ),
                io.Float.Input(
                    "latent_interp_strength",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    tooltip=(
                        "How much of the newly sampled frame survives the mix. 1.0 keeps it "
                        "whole and changes nothing, 0.5 is an even blend with the frame "
                        "before, and low values nearly freeze the sequence. Ignored while "
                        "use_latent_interpolation is off."
                    ),
                ),
                io.Float.Input(
                    "denoise_start",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How much of the first loop is redrawn. 1.0 ignores latent_image's "
                        "content and generates the opening frame from noise; around 0.5 "
                        "keeps its composition and changes the detail."
                    ),
                ),
                io.Float.Input(
                    "denoise_seq",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How much every loop after the first redraws. This is what decides "
                        "whether the run drifts or jumps: 0.5 lets a frame change noticeably, "
                        "0.2 barely moves, and near 1.0 each frame is a fresh image "
                        "holding nothing of the last."
                    ),
                ),
                io.Boolean.Input(
                    "unsample_latents",
                    default=False,
                    tooltip=(
                        "Whether each loop first runs the sampler backwards over the previous "
                        "frame, pushing it back up the noise schedule before resampling. It "
                        "gives the new prompt something to re-resolve rather than a finished "
                        "image to leave alone, at roughly double the time per loop."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output(
                    tooltip=(
                        "Every loop's latent, stacked into one batch in order. Decode it with "
                        "a VAE Decode to get the frames, then save them as an image sequence "
                        "or a video."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        seed,
        seed_mode_seq,
        alternate_values,
        steps,
        cfg,
        sampler_name,
        scheduler,
        sequence_loop_count,
        positive_seq,
        negative_seq,
        use_conditioning_slerp,
        cond_slerp_strength,
        latent_image,
        use_latent_interpolation,
        latent_interpolation_mode,
        latent_interp_strength,
        denoise_start,
        denoise_seq,
        unsample_latents,
    ) -> io.NodeOutput:
        import comfy.utils

        # ComfyUI's root nodes.py. This pack's own nodes/ package is reachable only as a
        # submodule of the pack, so the bare name resolves to ComfyUI's module.
        from nodes import common_ksampler

        results = []
        positive_conditioning = None
        negative_conditioning = None
        previous_seed = current_seed = seed
        seq_seed = seed
        start_at_step = 0
        progress = comfy.utils.ProgressBar(sequence_loop_count)

        for loop_count in range(sequence_loop_count):
            if alternate_values and loop_count % 2 == 0:
                if seed_mode_seq != "fixed":
                    previous_seed, current_seed = alternate_seed(seed, previous_seed, current_seed)
                    seq_seed = current_seed
                else:
                    seq_seed = seed
            else:
                seq_seed = seed if loop_count <= 0 else advance_seed(seq_seed, seed_mode_seq)

            logger.info("Loop count: %s, seed: %s", loop_count, seq_seed)

            last_positive = positive_conditioning[0] if positive_conditioning else None
            last_negative = negative_conditioning[0] if negative_conditioning else None

            positive_conditioning = conditioning_for_loop(positive_seq, loop_count, last_positive)
            negative_conditioning = conditioning_for_loop(negative_seq, loop_count, last_negative)

            if use_conditioning_slerp and (last_positive and last_negative):
                positive_conditioning = blend_conditioning(
                    cond_slerp_strength, last_positive, positive_conditioning
                )
                negative_conditioning = blend_conditioning(
                    cond_slerp_strength, last_negative, negative_conditioning
                )

            positive_conditioning = [positive_conditioning]
            negative_conditioning = [negative_conditioning]

            end_at_step = steps
            if results:
                latent_input = {"samples": results[-1]}
                denoise = denoise_seq
                start_at_step = round((1 - denoise) * steps)
            else:
                latent_input = latent_image
                denoise = denoise_start

            if unsample_latents and loop_count > 0:
                unsampled = unsample(
                    model=model,
                    seed=seq_seed,
                    cfg=cfg,
                    sampler_name=sampler_name,
                    steps=steps,
                    end_at_step=end_at_step,
                    scheduler=scheduler,
                    normalize=False,
                    positive=positive_conditioning,
                    negative=negative_conditioning,
                    latent_image=latent_input,
                )
                sample = common_ksampler(
                    model,
                    seq_seed,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    positive_conditioning,
                    negative_conditioning,
                    unsampled,
                    denoise=denoise,
                    disable_noise=False,
                    start_step=start_at_step,
                    last_step=end_at_step,
                    force_full_denoise=False,
                )[0]["samples"]
            else:
                sample = common_ksampler(
                    model,
                    seq_seed,
                    steps,
                    cfg,
                    sampler_name,
                    scheduler,
                    positive_conditioning,
                    negative_conditioning,
                    latent_input,
                    denoise=denoise,
                )[0]["samples"]

            if use_latent_interpolation and results and loop_count > 0:
                sample = interpolate_latents(
                    latent_interpolation_mode, latent_interp_strength, results[-1], sample
                )

            results.append(sample)
            progress.update(1)

        return io.NodeOutput({"samples": torch.cat(results, dim=0)})

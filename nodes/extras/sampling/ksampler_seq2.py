"""Sample a run of latents from a list of conditionings and a keyframe schedule."""

from __future__ import annotations

import math

import numpy as np
import torch
from comfy_api.latest import io

from ....modules import log
from ....modules.compat.sockets import require_input
from ....modules.sampling import sampler_names, scheduler_names
from ....modules.sampling.sequence import (
    LATENT_INTERPOLATION_MODES,
    MAX_SEED,
    SEED_MODES,
    advance_seed,
    alternate_seed,
    blend_conditioning,
    interpolate_latents,
    unsample,
)

REQUIRES = "extras"

logger = log.get_logger("nodes.extras.sampling")

#: How the seed is keyed to the loop number, in the order the widget offers them.
SEED_KEYING_MODES = ["sine", "modulo"]


def keyed_seed_modulo(loop_count: int, seed, divisor: int):
    """Offset the seed on every divisor-th loop and leave it alone in between.

    Args:
        loop_count: The loop about to run, counting from zero.
        seed: The base seed.
        divisor: How often the offset lands.

    Returns:
        The seed for this loop.
    """
    if loop_count % divisor == 0:
        return (seed + loop_count) % MAX_SEED
    return seed


def keyed_seed_sine(loop_count: int, seed, divisor: int):
    """Swing the seed around the base value on a sine wave.

    Args:
        loop_count: The loop about to run, counting from zero.
        seed: The base seed the wave is centred on.
        divisor: Loops per full cycle of the wave.

    Returns:
        The seed for this loop.
    """
    return 1000 * np.sin(2 * math.pi * loop_count / divisor) + seed


def keyed_denoise(loop_count: int, total: int, start_denoise: float, max_denoise: float):
    """Swing the denoise between two bounds on a sine wave across the whole run.

    Args:
        loop_count: The loop about to run, counting from zero.
        total: Loops in the run, which is one full cycle of the wave.
        start_denoise: One bound of the swing.
        max_denoise: The other bound.

    Returns:
        The denoise for this loop.
    """
    amplitude = (max_denoise - start_denoise) / 2
    mid_point = (max_denoise + start_denoise) / 2
    return amplitude * math.sin((math.pi * 2 * loop_count) / total) + mid_point


def add_noise(samples, noise_strength: float):
    """Add gaussian noise to a latent.

    Args:
        samples: Latent samples.
        noise_strength: Standard deviation of the noise added.

    Returns:
        New latent samples.
    """
    return samples + torch.randn_like(samples) * noise_strength


class KSamplerSequence2(io.ComfyNode):
    """Sample one latent per loop from a list of prompts and a keyframe schedule."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASKSamplerSeq2",
            display_name="KSampler Sequence (v2)",
            search_aliases=[
                "WASKSamplerSeq2",
                "KSampler Sequence (v2)",
                "prompt travel",
                "animation sampler",
                "keyframe sampler",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Run the sampler once per frame and stack the results into one latent "
                "batch, stepping to the next prompt whenever the frame is one of the "
                "keyframes. Built to be driven by CLIP Text Encode Sequence (v2), which "
                "produces the prompt list, the keyframe schedule and the frame count "
                "together. "
                "Noise injection, a swinging denoise and a keyed seed are all here to keep "
                "a long run moving instead of settling on one image."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="The diffusion model every frame in the run samples with.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=MAX_SEED,
                    tooltip=(
                        "The seed the first frame runs on, and the base every later frame's "
                        "seed is worked out from. The same seed replays the whole run. Any "
                        "whole number; `0` is as good a seed as any."
                    ),
                ),
                io.Combo.Input(
                    "seed_mode_seq",
                    options=SEED_MODES,
                    tooltip=(
                        "How the seed moves from frame to frame, applied after any keying. "
                        "'increment' and 'decrement' step it by one, which keeps consecutive "
                        "frames close; 'random' picks a fresh seed each frame, which makes "
                        "every frame its own image; 'fixed' leaves the keyed seed alone."
                    ),
                ),
                io.Boolean.Input(
                    "alternate_values",
                    default=True,
                    tooltip=(
                        "Whether every other loop runs on a second seed that drifts away "
                        "from the first instead of on the scheduled one. It gives the run a "
                        "slight back-and-forth flicker between two looks, which reads as "
                        "movement in a short sequence. Off, seed_keying and seed_mode decide "
                        "every frame's seed."
                    ),
                ),
                io.Int.Input(
                    "steps",
                    default=20,
                    min=1,
                    max=10000,
                    tooltip=(
                        "Sampling steps per frame. Around 20 suits most models, and the whole "
                        "run costs this many steps times the number of frames."
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
                        "How closely each frame is held to its prompt. Around 7-8 suits most "
                        "models; lower is looser and softer, much higher burns contrast and "
                        "makes a sequence flicker."
                    ),
                ),
                io.Combo.Input(
                    "sampler_name",
                    options=sampler_names(),
                    tooltip=(
                        "The sampling algorithm. 'euler' is the plain, predictable choice and "
                        "the steadiest across a sequence; the 'ancestral' and 'sde' variants "
                        "add fresh noise as they go, which adds detail and also adds flicker. "
                        "The list is whatever this ComfyUI offers."
                    ),
                ),
                io.Combo.Input(
                    "scheduler",
                    options=scheduler_names(),
                    tooltip=(
                        "How the noise level is stepped down within each frame. 'normal' and "
                        "'karras' are the usual choices. The list is whatever this ComfyUI "
                        "offers."
                    ),
                ),
                io.Int.Input(
                    "frame_count",
                    default=0,
                    min=0,
                    max=1024,
                    step=1,
                    tooltip=(
                        "How many frames to render. Wire it from CLIP Text Encode Sequence "
                        "(v2)'s frame_count output. At 0, or with no keyframes connected, "
                        "the run is one frame per prompt instead."
                    ),
                ),
                io.Int.Input(
                    "cond_keyframes",
                    default=0,
                    min=0,
                    max=1024,
                    step=1,
                    tooltip=(
                        "The frame numbers at which the run steps to the next prompt. Wire it "
                        "from CLIP Text Encode Sequence (v2)'s cond_keyframes output, which "
                        "builds the whole schedule; a single number here means one "
                        "changeover at that frame."
                    ),
                ),
                io.Conditioning.Input(
                    "positive_seq",
                    tooltip=(
                        "The list of positive prompts to work through, from CLIP Text "
                        "Encode Sequence (v2). One plain conditioning also works and is "
                        "then used for every frame."
                    ),
                ),
                io.Conditioning.Input(
                    "negative_seq",
                    tooltip=(
                        "The list of negative prompts, stepped through on the same keyframes "
                        "as the positive ones. One plain conditioning is used for every frame."
                    ),
                ),
                io.Boolean.Input(
                    "use_conditioning_slerp",
                    default=False,
                    tooltip=(
                        "Whether each frame's conditioning is rebuilt from its embedding and "
                        "pooled output alone. Anything else the prompt carried, an area, a "
                        "mask, a control hint, is dropped when this is on, so leave it off "
                        "unless the prompts are plain text encodes."
                    ),
                ),
                io.Float.Input(
                    "cond_slerp_strength",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    tooltip=(
                        "Interpolation factor for the rebuild above. The two ends of the "
                        "interpolation are the same prompt here, so the value makes no "
                        "difference to the result. Ignored while use_conditioning_slerp is "
                        "off."
                    ),
                ),
                io.Latent.Input(
                    "latent_image",
                    tooltip=(
                        "The latent the first frame starts from, which also sets the size of "
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
                        "contrast better; 'Cosine Interp' eases in and out. Ignored while "
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
                        "How much of the first frame is redrawn. 1.0 ignores latent_image's "
                        "content and generates the opening frame from noise; around 0.5 keeps "
                        "its composition and changes the detail."
                    ),
                ),
                io.Float.Input(
                    "denoise_seq",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How much every frame after the first redraws, and the low end of the "
                        "swing when denoise_sine is on. 0.5 lets a frame change noticeably, "
                        "0.2 barely moves, and near 1.0 each frame is a fresh image."
                    ),
                ),
                io.Boolean.Input(
                    "unsample_latents",
                    default=False,
                    tooltip=(
                        "Whether each frame first runs the sampler backwards over the previous "
                        "one, pushing it back up the noise schedule before resampling. It "
                        "gives the new prompt something to re-resolve rather than a finished "
                        "image to leave alone, at roughly double the time per frame."
                    ),
                ),
                io.Boolean.Input(
                    "inject_noise",
                    default=True,
                    tooltip=(
                        "Whether fresh noise is stirred into each frame before it is sampled. "
                        "It is what stops a long run settling on one image and holding it; "
                        "turn it off for the steadiest possible sequence."
                    ),
                ),
                io.Float.Input(
                    "noise_strength",
                    default=0.1,
                    min=0.001,
                    max=1.0,
                    step=0.001,
                    tooltip=(
                        "How much noise is stirred in. 0.1 keeps the picture and adds "
                        "movement; above about 0.3 the composition starts breaking up frame "
                        "to frame. Ignored while inject_noise is off."
                    ),
                ),
                io.Boolean.Input(
                    "denoise_sine",
                    default=True,
                    tooltip=(
                        "Whether the denoise swings between denoise_seq and denoise_max over "
                        "the length of the run rather than staying put. The run then breathes "
                        ", settling for a stretch, opening up again, which suits a long "
                        "sequence better than one fixed value."
                    ),
                ),
                io.Float.Input(
                    "denoise_max",
                    default=0.9,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    tooltip=(
                        "The far end of the denoise swing. 0.9 lets the picture change a great "
                        "deal at the top of the wave; bring it closer to denoise_seq for a "
                        "flatter run. Ignored while denoise_sine is off."
                    ),
                ),
                io.Boolean.Input(
                    "seed_keying",
                    default=True,
                    tooltip=(
                        "Whether the seed follows a pattern tied to the frame number instead "
                        "of only stepping. A pattern that repeats brings back seeds the run "
                        "has already used, which is how a sequence comes back round to a look "
                        "rather than drifting away from it for good."
                    ),
                ),
                io.Combo.Input(
                    "seed_keying_mode",
                    options=SEED_KEYING_MODES,
                    tooltip=(
                        "Which pattern the seed follows. `sine` swings it smoothly around the "
                        "base seed once every seed_divisor frames; `modulo` leaves it alone "
                        "and jumps it on every seed_divisor-th frame, which also skips the "
                        "unsample pass on those frames. Ignored while seed_keying is off."
                    ),
                ),
                io.Int.Input(
                    "seed_divisor",
                    default=4,
                    min=2,
                    max=1024,
                    step=1,
                    tooltip=(
                        "How many frames one cycle of the seed pattern takes. 4 gives a fast "
                        "flutter, 24 a slow swing across a second of video. Ignored while "
                        "seed_keying is off."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output(
                    tooltip=(
                        "Every frame's latent, stacked into one batch in order. Decode it with "
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
        frame_count,
        cond_keyframes,
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
        inject_noise,
        noise_strength,
        denoise_sine,
        denoise_max,
        seed_keying,
        seed_keying_mode,
        seed_divisor,
    ) -> io.NodeOutput:
        """Sample one latent per frame and stack them into a batch.

        Raises:
            ValueError: Nothing is connected to the model, positive_seq, negative_seq or
                latent_image input.
        """
        import comfy.utils

        # ComfyUI's root nodes.py. This pack's own nodes/ package is reachable only as a
        # submodule of the pack, so the bare name resolves to ComfyUI's module.
        from nodes import common_ksampler

        for value, socket, thing, source, source_output in (
            (model, "model", "model", "checkpoint loader", "MODEL"),
            (
                positive_seq, "positive_seq", "conditioning",
                "CLIP Text Encode Sequence (v2) or a CLIP Text Encode", "conditioning_sequence",
            ),
            (
                negative_seq, "negative_seq", "conditioning",
                "CLIP Text Encode Sequence (v2) or a CLIP Text Encode", "conditioning_sequence",
            ),
            (latent_image, "latent_image", "latent", "Empty Latent Image", "LATENT"),
        ):
            require_input(
                value, "KSampler Sequence (v2)", socket, thing, source, source_output
            )

        if not isinstance(positive_seq, list):
            positive_seq = [positive_seq]
        if not isinstance(negative_seq, list):
            negative_seq = [negative_seq]
        if not isinstance(cond_keyframes, list):
            cond_keyframes = [cond_keyframes]
        cond_keyframes = sorted(cond_keyframes)

        positive_cond_idx = 0
        negative_cond_idx = 0
        results = []
        previous_seed = current_seed = seq_seed = seed
        start_at_step = 0

        sequence_loop_count = (
            max(frame_count, len(positive_seq)) if cond_keyframes else len(positive_seq)
        )

        logger.info("Starting loop sequence with %s frames.", sequence_loop_count)
        logger.info(
            "Using %s positive conditionings and %s negative conditionings",
            len(positive_seq),
            len(negative_seq),
        )
        logger.info(
            "Conditioning keyframe schedule is: %s", ", ".join(map(str, cond_keyframes))
        )
        progress = comfy.utils.ProgressBar(sequence_loop_count)

        for loop_count in range(sequence_loop_count):
            if loop_count in cond_keyframes:
                positive_cond_idx = min(positive_cond_idx + 1, len(positive_seq) - 1)
                negative_cond_idx = min(negative_cond_idx + 1, len(negative_seq) - 1)

            positive_conditioning = positive_seq[positive_cond_idx]
            negative_conditioning = negative_seq[negative_cond_idx]

            if alternate_values and loop_count % 2 == 0:
                if seed_mode_seq != "fixed":
                    previous_seed, current_seed = alternate_seed(
                        seed, previous_seed, current_seed
                    )
                    seq_seed = current_seed
                else:
                    seq_seed = seed
            elif seed_keying:
                keyed = keyed_seed_sine if seed_keying_mode == "sine" else keyed_seed_modulo
                seq_seed = seed if loop_count <= 0 else keyed(loop_count, seed, seed_divisor)
            else:
                seq_seed = seed if loop_count <= 0 else advance_seed(seq_seed, seed_mode_seq)

            logger.info("Loop count: %s, seed: %s", loop_count, seq_seed)

            if use_conditioning_slerp and positive_conditioning and negative_conditioning:
                positive_conditioning = blend_conditioning(
                    cond_slerp_strength, positive_conditioning, positive_conditioning
                )
                negative_conditioning = blend_conditioning(
                    cond_slerp_strength, negative_conditioning, negative_conditioning
                )

            positive_conditioning = [positive_conditioning]
            negative_conditioning = [negative_conditioning]

            end_at_step = steps
            if results:
                latent_input = {"samples": results[-1]}
                denoise = (
                    keyed_denoise(loop_count, sequence_loop_count, denoise_seq, denoise_max)
                    if denoise_sine
                    else denoise_seq
                )
                start_at_step = round((1 - denoise) * steps)
            else:
                latent_input = latent_image
                denoise = denoise_start

            if unsample_latents and loop_count > 0:
                skip_unsample = (
                    seed_keying
                    and seed_keying_mode == "modulo"
                    and loop_count % seed_divisor == 0
                )
                if skip_unsample:
                    unsampled_latent = latent_input
                else:
                    unsampled_latent = unsample(
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
                if inject_noise:
                    logger.info("Injecting noise at %s strength.", noise_strength)
                    unsampled_latent["samples"] = add_noise(
                        unsampled_latent["samples"], noise_strength
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
                    unsampled_latent,
                    denoise=denoise,
                    disable_noise=False,
                    start_step=start_at_step,
                    last_step=end_at_step,
                    force_full_denoise=False,
                )[0]["samples"]
            else:
                if inject_noise and loop_count > 0:
                    logger.info("Injecting noise at %s strength.", noise_strength)
                    latent_input["samples"] = add_noise(latent_input["samples"], noise_strength)
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

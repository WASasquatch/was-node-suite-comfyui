"""Samplers that scale and offset the latent from inside their own denoising loop."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import DICT
from ...modules.interface import preview, run_result
from ...modules.latent import affine
from ...modules.latent import affine_patterns as patterns
from ...modules.sampling.affine import ACTS_ON, AffineSpec, patch_sampler
from ..latent.latent_affine import (
    EXTERNAL_HINT,
    OPTIONS_HINT,
    PATTERN_HINT,
    SEED_HINT,
    STREAMS_HINT,
    TEMPORAL_HINT,
)

#: What every Affine sampler says about its schedule socket.
SCHEDULE_HINT = (
    "The per-step strength curve from an Affine Schedule node. Left unwired the affine "
    "ramps up over the middle of the run, from a fifth of the way in to four fifths."
)

#: What every Affine sampler says about which space its values are read in.
SPACE_HINT = (
    "Which latent max_scale and max_bias are measured against. 'latent' = the same scale "
    "Latent Affine uses, so a value means the same thing in both places; 'model' = the "
    "sampler's own internal latent, which some models hold at a very different magnitude."
)

#: What every Affine sampler says about what the multiplier reaches.
ACTS_ON_HINT = (
    "What max_scale multiplies. 'content' = only the picture the model has resolved, so "
    "the sampler's own noise is left alone. 'latent' = the whole latent, which amplifies "
    "that noise as well and prints it as fixed grain on a model that holds noise for most "
    "of its run."
)

#: What every Affine sampler says about its debug switch.
DEBUG_HINT = (
    "`true` logs every application to the console: the step, its sigma, the strength, the "
    "resolved scale and bias, and which streams were touched. `false` stays quiet."
)

#: What every Affine sampler says about which sampler it can wrap.
SAMPLER_HINT = (
    "The sampler to wrap, from KSamplerSelect or any other SAMPLER source. Every stock "
    "sampler carries the affine except dpm_fast, dpm_adaptive, uni_pc and uni_pc_bh2, "
    "which is reported in the console rather than failing the run."
)


def sampler_names() -> list[str]:
    """The sampler names this ComfyUI offers."""
    import comfy.samplers

    return comfy.samplers.KSampler.SAMPLERS


def scheduler_names() -> list[str]:
    """The scheduler names this ComfyUI offers."""
    import comfy.samplers

    return comfy.samplers.KSampler.SCHEDULERS


#: The settings every Affine sampler shares, in the order they are drawn.
AFFINE_INPUTS = [
    io.Int.Input(
        "affine_interval",
        default=1,
        min=1,
        max=100,
        tooltip=(
            "Apply on every Nth step of the schedule. 1 = every step, 4 = every fourth, "
            "which leaves the sampler more room to settle between applications."
        ),
    ),
    io.Float.Input(
        "max_scale",
        default=1.02,
        min=0.0,
        max=2.0,
        step=0.001,
        tooltip=(
            "What the latent is multiplied by at the peak of the schedule, compounding "
            "over every step it lands on. 1.0 = no change, 1.02 adds visible texture over "
            "a 20 step run, 1.05 is strong, and past 1.08 the picture breaks up. Below 1.0 "
            "softens instead."
        ),
    ),
    io.Float.Input(
        "max_bias",
        default=0.0,
        min=-2.0,
        max=2.0,
        step=0.001,
        tooltip=(
            "What is added at the peak of the schedule, beside max_scale rather than "
            "through it. 0.0 = nothing, 0.005 = a gentle drift, 0.02 shifts the whole "
            "colour. bias_field on Affine Options decides whether that is one flat offset "
            "or a noise field."
        ),
    ),
    io.Combo.Input("pattern", options=patterns.PATTERNS, default="white_noise", tooltip=PATTERN_HINT),
    io.Int.Input("affine_seed", default=0, min=0, max=0x7FFFFFFF, tooltip=SEED_HINT),
    io.Boolean.Input(
        "affine_seed_increment",
        default=False,
        tooltip=(
            "`true` advances the seed on every application, so the grain moves from step "
            "to step. `false` holds one mask for the whole run, which keeps the affine "
            "landing in the same places."
        ),
    ),
    io.Combo.Input(
        "temporal_mode",
        options=affine.TEMPORAL_MODES,
        default="static",
        tooltip=TEMPORAL_HINT,
    ),
]


#: The settings every Affine sampler shares that stay unwired most of the time.
AFFINE_OPTIONAL = [
    DICT.Input("affine_schedule", optional=True, tooltip=SCHEDULE_HINT),
    io.Combo.Input(
        "affine_streams",
        options=affine.STREAM_MODES,
        default="video",
        optional=True,
        tooltip=STREAMS_HINT,
    ),
    io.Combo.Input(
        "affine_space",
        options=["latent", "model"],
        default="latent",
        optional=True,
        tooltip=SPACE_HINT,
    ),
    io.Mask.Input("external_mask", optional=True, tooltip=EXTERNAL_HINT),
    DICT.Input("affine_options", optional=True, tooltip=OPTIONS_HINT),
    io.Boolean.Input("debug", default=False, optional=True, tooltip=DEBUG_HINT),
    io.Combo.Input(
        "affine_acts_on",
        options=ACTS_ON,
        default="content",
        optional=True,
        tooltip=ACTS_ON_HINT,
    ),
]


def _settings(scope: dict) -> dict:
    """An execute frame's local variables as a plain settings dictionary.

    Args:
        scope: The frame's ``locals()``, taken as its first statement.

    Returns:
        The same mapping without ``cls``.
    """
    return {key: value for key, value in scope.items() if key != "cls"}


def build_spec(values: dict, total_steps=None, step_offset=0) -> AffineSpec:
    """Turn a node's widget values into the settings the engine reads.

    Args:
        values: The node's inputs, keyed by widget name.
        total_steps: Steps the schedule spans, where the run covers only part of it.
        step_offset: Where in that schedule the run starts.

    Returns:
        The resolved settings.
    """
    options = values.get("affine_options")
    asked = values.get("pattern")
    pattern = str((options or {}).get("pattern", asked if asked is not None else "white_noise"))
    return AffineSpec(
        pattern_asked=asked,
        schedule=values.get("affine_schedule"),
        interval=values.get("affine_interval", 1),
        max_scale=values.get("max_scale", 1.02),
        max_bias=values.get("max_bias", 0.0),
        pattern=pattern,
        seed=values.get("affine_seed", 0),
        seed_increment=values.get("affine_seed_increment", False),
        temporal_mode=values.get("temporal_mode", "static"),
        streams=values.get("affine_streams", "video"),
        external_mask=values.get("external_mask"),
        options=options,
        space=values.get("affine_space", "latent"),
        acts_on=values.get("affine_acts_on", "content"),
        total_steps=total_steps,
        step_offset=step_offset,
        debug=values.get("debug", False),
    )


def sampler_label(sampler) -> str:
    """The name to show for a SAMPLER whose own name the node was not told.

    Args:
        sampler: A SAMPLER object.

    Returns:
        The sampler's name as ComfyUI's menu spells it, or a stand-in.
    """
    name = getattr(getattr(sampler, "sampler_function", None), "__name__", "")
    for edge in ("sample_", "affine_"):
        if name.startswith(edge):
            name = name[len(edge):]
    if name.endswith("_function"):
        name = name[: -len("_function")]
    return name or "this sampler"


def report_patch(spec: AffineSpec, holder: dict, sampler_name: str) -> None:
    """Publish what a wrapped sampler will do, for the node's own panel.

    Args:
        spec: The settings the wrap carries.
        holder: What :func:`modules.sampling.affine.patch_sampler` filled in.
        sampler_name: The sampler that was wrapped.
    """
    try:
        if not run_result.watching():
            return
        supported = bool(holder.get("supported", True))
        if not supported:
            summary = f"{sampler_name} carries no affine: {holder.get('reason')}"
        elif spec.is_noop:
            summary = "max_scale 1.0, max_bias 0.0: unchanged"
        else:
            summary = f"{sampler_name} wrapped, {spec.pattern} up to scale {spec.max_scale:g}"
        run_result.publish(
            status=run_result.OK if (supported or spec.is_noop) else run_result.WARNING,
            summary=summary,
            counts={
                "max scale": round(spec.max_scale, 4),
                "max bias": round(spec.max_bias, 4),
                "every": spec.interval,
            },
            facts={
                "pattern": spec.pattern,
                "sampler": sampler_name,
                "streams": spec.streams,
                "curve": str(spec.schedule.get("curve", "")),
                "values read in": spec.space,
                "multiplies": spec.acts_on,
            },
        )
    except Exception:
        run_result.publish(status=run_result.WARNING, summary="the affine report could not be built")


def report(spec: AffineSpec, holder: dict, sampler_name: str) -> None:
    """Publish what the affine did during a run, for the node's own panel.

    Args:
        spec: The settings the run used.
        holder: What :func:`modules.sampling.affine.patch_sampler` filled in.
        sampler_name: The sampler that was wrapped.
    """
    try:
        mask = holder.get("mask")
        if mask is not None:
            preview.publish_mask_output(mask)
        if not run_result.watching():
            return
        applied = int(holder.get("applications", 0))
        supported = bool(holder.get("supported", True))
        counts = {
            "steps affected": applied,
            "max scale": round(spec.max_scale, 4),
            "max bias": round(spec.max_bias, 4),
        }
        if mask is not None and mask.numel():
            counts["mask coverage %"] = round(float(mask.float().mean()) * 100.0, 2)
        landed = list(holder.get("landed") or [])
        if landed:
            counts["first sigma"] = round(landed[0][1], 4)
            counts["last sigma"] = round(landed[-1][1], 4)
        picture = (float(holder.get("gain", 1.0)) - 1.0) * 100.0
        reach = float(holder.get("reach", 1.0)) * 100.0
        if applied:
            counts["picture gain %"] = round(picture, 2)
            counts["reach %"] = round(reach, 1)
        inert = spec.inert()
        if supported and spec.is_noop:
            summary = "off: max_scale 1.0, max_bias 0.0"
        elif not supported:
            summary = f"no affine: {sampler_name} cannot be wrapped"
        elif not applied:
            summary = "no step carried the affine"
        else:
            summary = (f"{spec.pattern} on {applied} step(s): picture {picture:+.1f}%, "
                       f"reach {reach:.0f}%")
        facts = {
            "pattern": spec.pattern,
            "sampler": sampler_name,
            "streams": spec.streams,
            "values read in": spec.space,
            "multiplies": spec.acts_on,
        }
        if landed:
            facts["steps"] = ", ".join(str(step) for step, _ in landed)
        for number, sentence in enumerate(inert, start=1):
            facts[f"not read {number}"] = sentence
        settled = supported and (applied or spec.is_noop) and not inert
        run_result.publish(
            status=run_result.OK if settled else run_result.WARNING,
            summary=summary,
            counts=counts,
            facts=facts,
        )
        if inert:
            from ...modules import log

            for sentence in inert:
                log.get_logger("sampling.affine").warning("%s.", sentence)
    except Exception:
        run_result.publish(status=run_result.WARNING, summary="the affine report could not be built")


def warn_unsupported(holder: dict, sampler_name: str) -> None:
    """Say on the console that a sampler could not carry the affine.

    Args:
        holder: What :func:`modules.sampling.affine.patch_sampler` filled in.
        sampler_name: The sampler that was asked for.
    """
    if holder.get("supported", True):
        return
    from ...modules import log

    log.get_logger("sampling.affine").warning(
        "No affine was injected: %s cannot be wrapped, because %s. Pick another sampler, or "
        "apply the affine to the latent with Latent Affine instead.",
        sampler_name,
        holder.get("reason"),
    )


class AffineSampler(io.ComfyNode):
    """Wrap any sampler so it carries an affine through its own loop."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAffineSampler",
            display_name="Affine Sampler",
            search_aliases=[
                "WASAffineSampler",
                "Affine Sampler",
                "Affine Sampler (Inline)",
                "WASAffineSamplerInline",
                "affine sampler patch",
                "inline affine",
                "sampler wrapper",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Wrap a sampler so it scales and offsets the latent from inside its own "
                "denoising loop, then hand it to SamplerCustomAdvanced or anything else "
                "that takes a SAMPLER. Nothing is restarted, so multistep history, noise "
                "sequences and packed audio and video latents all survive the transform."
            ),
            inputs=[
                io.Sampler.Input("sampler", tooltip=SAMPLER_HINT),
                *AFFINE_INPUTS,
                *AFFINE_OPTIONAL,
            ],
            outputs=[
                io.Sampler.Output(
                    display_name="sampler",
                    tooltip="The same sampler, now applying the affine as it runs.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        sampler,
        affine_interval,
        max_scale,
        max_bias,
        pattern,
        affine_seed,
        affine_seed_increment,
        temporal_mode,
        affine_schedule=None,
        affine_streams="video",
        affine_space="latent",
        external_mask=None,
        affine_options=None,
        debug=False,
        affine_acts_on="content",
    ) -> io.NodeOutput:
        spec = build_spec(_settings(locals()))
        patched, holder = patch_sampler(sampler, spec)
        name = sampler_label(sampler)
        warn_unsupported(holder, name)
        report_patch(spec, holder, name)
        return io.NodeOutput(patched)


class KSamplerAffineAdvanced(io.ComfyNode):
    """Sample a latent in one pass with an affine applied as it denoises."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASKSamplerAffineAdvanced",
            display_name="KSampler Affine Advanced",
            search_aliases=[
                "WASKSamplerAffineAdvanced",
                "KSampler Affine Advanced",
                "KSampler Affine Advanced (Inline)",
                "WASAffineKSamplerAdvancedInline",
                "affine ksampler",
                "affine sampling",
                "inline affine",
                "latent scale during sampling",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Denoise a latent while scaling and offsetting it from inside the sampler's "
                "own loop, on a curve that decides how strong the effect is at each step. "
                "Used to push texture and contrast into a generation as it forms. The run "
                "is never stopped and restarted, so it behaves the same on flow-matching, "
                "multistep and packed audio and video models."
            ),
            inputs=[
                io.Model.Input("model", tooltip="The model to denoise with."),
                io.Conditioning.Input("positive", tooltip="What the image should contain."),
                io.Conditioning.Input("negative", tooltip="What it should avoid."),
                io.Latent.Input(
                    "latent_image",
                    tooltip=(
                        "The latent to denoise. Image, video and packed audio and video "
                        "latents are all handled."
                    ),
                ),
                io.Boolean.Input(
                    "add_noise",
                    default=True,
                    tooltip=(
                        "`true` adds fresh noise before the first step, which is what a run "
                        "from an empty latent needs. `false` starts from the latent as it "
                        "is, for the second half of a run another sampler began."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip=(
                        "Seeds the noise that is added. The same seed gives the same run; "
                        "0 is as good a seed as any."
                    ),
                ),
                io.Int.Input(
                    "steps",
                    default=20,
                    min=1,
                    max=10000,
                    tooltip=(
                        "How many denoising steps to take. 20 suits most models, 8 for a "
                        "turbo or lightning one, 40 where fine detail matters."
                    ),
                ),
                io.Float.Input(
                    "cfg",
                    default=4.5,
                    min=0.0,
                    max=100.0,
                    step=0.1,
                    tooltip=(
                        "How hard the prompt is enforced. 1.0 = the model's own idea, 4.5 "
                        "suits most flow-matching models, 7.0 to 8.0 the older ones."
                    ),
                ),
                io.Combo.Input(
                    "sampler_name",
                    options=sampler_names(),
                    default="euler",
                    tooltip=(
                        "The algorithm the steps are taken with. All of them carry the affine "
                        "except dpm_fast, dpm_adaptive, uni_pc and uni_pc_bh2, which sample "
                        "normally and say so in the console."
                    ),
                ),
                io.Combo.Input(
                    "scheduler",
                    options=scheduler_names(),
                    default="normal",
                    tooltip=(
                        "How the noise level falls across the steps. `normal` suits most "
                        "models, `simple` and `beta` are common on flow-matching ones."
                    ),
                ),
                io.Float.Input(
                    "denoise",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How much of the latent is replaced. 1.0 = a fresh generation, 0.5 "
                        "keeps the broad shape of what came in, 0.2 refines it only."
                    ),
                ),
                *AFFINE_INPUTS,
                io.Int.Input(
                    "start_at_step",
                    default=0,
                    min=0,
                    max=10000,
                    optional=True,
                    tooltip=(
                        "First step to run. The affine curve is still read over the whole "
                        "'steps' range, so a slice of a run carries the part of the curve "
                        "that belongs to it."
                    ),
                ),
                io.Int.Input(
                    "end_at_step",
                    default=10000,
                    min=0,
                    max=10000,
                    optional=True,
                    tooltip=(
                        "Step to stop before. 10000 runs to the end; 12 on a 20 step run "
                        "hands the rest to a second sampler."
                    ),
                ),
                io.Boolean.Input(
                    "return_with_leftover_noise",
                    default=False,
                    optional=True,
                    tooltip=(
                        "`true` leaves the latent partly noisy so another sampler can pick "
                        "the run up. `false` finishes the denoise, which is what a final "
                        "pass wants."
                    ),
                ),
                *AFFINE_OPTIONAL,
            ],
            outputs=[
                io.Latent.Output(display_name="latent", tooltip="The denoised latent."),
                io.Mask.Output(
                    display_name="mask",
                    tooltip=(
                        "The mask the last application ran through, at latent resolution. "
                        "All zero where no affine was applied."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        positive,
        negative,
        latent_image,
        add_noise,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        affine_interval,
        max_scale,
        max_bias,
        pattern,
        affine_seed,
        affine_seed_increment,
        temporal_mode,
        start_at_step=0,
        end_at_step=10000,
        return_with_leftover_noise=False,
        affine_schedule=None,
        affine_streams="video",
        affine_space="latent",
        external_mask=None,
        affine_options=None,
        debug=False,
        affine_acts_on="content",
    ) -> io.NodeOutput:
        values = _settings(locals())
        import comfy.model_management
        import comfy.sample
        import comfy.samplers
        import comfy.utils
        import latent_preview

        if not hasattr(model, "get_model_object"):
            raise ValueError(
                "the model socket needs a diffusion model, as a Load Checkpoint or Load "
                f"Diffusion Model node answers one. It was handed {type(model).__name__}, "
                "which carries nothing to sample with."
            )

        latent = dict(latent_image)
        samples = comfy.sample.fix_empty_latent_channels(
            model,
            latent["samples"],
            latent.get("downscale_ratio_spacial", None),
            latent.get("downscale_ratio_temporal", None),
        )
        latent["samples"] = samples

        ksampler = comfy.samplers.KSampler(
            model,
            steps=int(steps),
            device=model.load_device,
            sampler=sampler_name,
            scheduler=scheduler,
            denoise=float(denoise),
            model_options=model.model_options,
        )
        sigmas = ksampler.sigmas.clone()
        first = max(0, int(start_at_step))
        last = int(end_at_step)
        if last <= 0 or last > int(steps):
            last = int(steps)
        if last < (len(sigmas) - 1):
            sigmas = sigmas[: last + 1]
            if not return_with_leftover_noise:
                sigmas[-1] = 0
        if first >= (len(sigmas) - 1):
            out = _without_ratios(latent)
            return io.NodeOutput(out, affine.mask_like(samples))
        sigmas = sigmas[first:]

        if add_noise:
            noise = comfy.sample.prepare_noise(samples, int(seed), latent.get("batch_index", None))
        else:
            noise = comfy.sample.prepare_empty_noise(samples)

        spec = build_spec(values, total_steps=int(steps), step_offset=first)
        patched, holder = patch_sampler(comfy.samplers.sampler_object(sampler_name), spec)
        warn_unsupported(holder, sampler_name)

        callback = latent_preview.prepare_callback(model, int(sigmas.shape[-1]) - 1)
        result = comfy.samplers.sample(
            model,
            noise,
            positive,
            negative,
            float(cfg),
            model.load_device,
            patched,
            sigmas,
            model_options=model.model_options,
            latent_image=samples,
            denoise_mask=latent.get("noise_mask", None),
            callback=callback,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
            seed=int(seed),
        )
        result = result.to(comfy.model_management.intermediate_device())

        out = _without_ratios(latent)
        out["samples"] = result
        report(spec, holder, str(sampler_name))
        mask = holder.get("mask")
        return io.NodeOutput(out, mask if mask is not None else affine.mask_like(result))


class CustomSamplerAffineAdvanced(io.ComfyNode):
    """Sample from a guider and a sigma schedule with an affine applied as it denoises."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASCustomSamplerAffineAdvanced",
            display_name="Custom Sampler Affine Advanced",
            search_aliases=[
                "WASCustomSamplerAffineAdvanced",
                "Custom Sampler Affine Advanced",
                "Custom Sampler Affine Advanced (Inline)",
                "WASAffineCustomAdvancedInline",
                "affine custom sampler",
                "guider affine",
                "inline affine",
                "affine sigmas",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Run a guider over a sigma schedule while scaling and offsetting the latent "
                "from inside the sampler's loop. The custom sampling form, for a graph that "
                "already builds its own noise, guider, sampler and sigmas. It answers the "
                "mask the affine ran through beside both latents."
            ),
            inputs=[
                io.Noise.Input("noise", tooltip="Where the starting noise comes from."),
                io.Guider.Input("guider", tooltip="What steers the denoising."),
                io.Sampler.Input("sampler", tooltip=SAMPLER_HINT),
                io.Sigmas.Input(
                    "sigmas",
                    tooltip=(
                        "The noise levels to step through. The affine curve is read over "
                        "these, so one entry fewer than the length is the step count."
                    ),
                ),
                io.Latent.Input(
                    "latent_image",
                    tooltip=(
                        "The latent to denoise. Image, video and packed audio and video "
                        "latents are all handled."
                    ),
                ),
                *AFFINE_INPUTS,
                *AFFINE_OPTIONAL,
            ],
            outputs=[
                io.Latent.Output(display_name="output", tooltip="The latent the sampler ended on."),
                io.Latent.Output(
                    display_name="denoised_output",
                    tooltip="The model's own estimate of the clean latent at the last step.",
                ),
                io.Mask.Output(
                    display_name="mask",
                    tooltip=(
                        "The mask the last application ran through, at latent resolution. "
                        "All zero where no affine was applied."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        noise,
        guider,
        sampler,
        sigmas,
        latent_image,
        affine_interval,
        max_scale,
        max_bias,
        pattern,
        affine_seed,
        affine_seed_increment,
        temporal_mode,
        affine_schedule=None,
        affine_streams="video",
        affine_space="latent",
        external_mask=None,
        affine_options=None,
        debug=False,
        affine_acts_on="content",
    ) -> io.NodeOutput:
        values = _settings(locals())
        import comfy.model_management
        import comfy.nested_tensor
        import comfy.sample
        import comfy.utils
        import latent_preview

        if not hasattr(guider, "model_patcher") or not hasattr(guider, "sample"):
            raise ValueError(
                "the guider socket needs a guider, as CFGGuider, BasicGuider and "
                f"DualCFGGuider answer one. It was handed {type(guider).__name__}, which "
                "carries nothing to denoise with."
            )

        latent = dict(latent_image)
        samples = comfy.sample.fix_empty_latent_channels(
            guider.model_patcher,
            latent["samples"],
            latent.get("downscale_ratio_spacial", None),
            latent.get("downscale_ratio_temporal", None),
        )
        latent["samples"] = samples

        spec = build_spec(values)
        patched, holder = patch_sampler(sampler, spec)
        name = sampler_label(sampler)
        warn_unsupported(holder, name)

        x0_output = {}
        callback = latent_preview.prepare_callback(
            guider.model_patcher, int(sigmas.shape[-1]) - 1, x0_output
        )
        result = guider.sample(
            noise.generate_noise(latent),
            samples,
            patched,
            sigmas,
            denoise_mask=latent.get("noise_mask", None),
            callback=callback,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
            seed=noise.seed,
        )
        result = result.to(comfy.model_management.intermediate_device())

        out = _without_ratios(latent)
        out["samples"] = result

        if "x0" in x0_output:
            x0 = x0_output["x0"]
            if getattr(result, "is_nested", False) and not getattr(x0, "is_nested", False):
                shapes = [part.shape for part in result.unbind()]
                x0 = comfy.nested_tensor.NestedTensor(comfy.utils.unpack_latents(x0, shapes))
            denoised = _without_ratios(latent)
            denoised["samples"] = guider.model_patcher.model.process_latent_out(x0.cpu())
        else:
            denoised = out

        report(spec, holder, name)
        mask = holder.get("mask")
        return io.NodeOutput(out, denoised, mask if mask is not None else affine.mask_like(result))


def _without_ratios(latent: dict) -> dict:
    """A copy of a latent dictionary with the downscale ratios dropped.

    Args:
        latent: The latent that came in.

    Returns:
        A new dictionary carrying everything else it held.
    """
    return {
        key: value
        for key, value in latent.items()
        if key not in ("downscale_ratio_spacial", "downscale_ratio_temporal")
    }

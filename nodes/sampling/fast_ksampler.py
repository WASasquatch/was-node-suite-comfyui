"""KSampler with a lighter live preview."""

from __future__ import annotations

import comfy.sample
import comfy.samplers
import comfy.utils
from comfy_api.latest import io

from ...modules.sampling import preview


class FastKSampler(io.ComfyNode):
    """Denoise a latent as KSampler does, previewing at display size on a chosen cadence."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASFastKSampler",
            display_name="Fast KSampler",
            search_aliases=[
                "WASFastKSampler",
                "Fast KSampler",
                "KSampler",
                "sampler",
                "txt2img",
                "img2img",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Denoise a latent exactly as core KSampler does, with the live preview made "
                "cheaper: the preview decoder is kept loaded between runs, the preview is "
                "decoded at the size it is shown rather than full size, and preview_every "
                "skips steps. The sampled latent is the same as KSampler's."
            ),
            inputs=[
                io.Model.Input("model", tooltip="The model used for denoising the input latent."),
                io.Int.Input(
                    "seed", default=0, min=0, max=0xFFFFFFFFFFFFFFFF, control_after_generate=True,
                    tooltip="The noise seed. The same seed with the same settings gives the same picture, `0` as good as any.",
                ),
                io.Int.Input(
                    "steps", default=20, min=1, max=10000,
                    tooltip="Denoising steps. `20` is typical, `4` to `8` for turbo and lightning models.",
                ),
                io.Float.Input(
                    "cfg", default=8.0, min=0.0, max=100.0, step=0.1, round=0.01,
                    tooltip="How strongly the prompt steers. `7` to `8` for SD and SDXL, `1` for distilled models.",
                ),
                io.Combo.Input(
                    "sampler_name", options=comfy.samplers.KSampler.SAMPLERS,
                    tooltip="The solver each step runs, such as `euler` or `dpmpp_2m`.",
                ),
                io.Combo.Input(
                    "scheduler", options=comfy.samplers.KSampler.SCHEDULERS,
                    tooltip="How the noise level falls from step to step, such as `normal` or `karras`.",
                ),
                io.Conditioning.Input("positive", tooltip="What the picture should contain."),
                io.Conditioning.Input("negative", tooltip="What the picture should avoid."),
                io.Latent.Input("latent_image", tooltip="The latent to denoise."),
                io.Float.Input(
                    "denoise", default=1.0, min=0.0, max=1.0, step=0.01,
                    tooltip="`1.0` starts from pure noise; `0.5` keeps half of an input image's structure.",
                ),
                io.Int.Input(
                    "preview_every", default=1, min=1, max=100, optional=True,
                    tooltip="Decode the live preview on every this many steps, and always on the last. `1` previews every step.",
                ),
            ],
            outputs=[io.Latent.Output(display_name="LATENT", tooltip="The denoised latent.")],
        )

    @classmethod
    def execute(cls, model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                latent_image, denoise=1.0, preview_every=1) -> io.NodeOutput:
        """Sample the latent.

        Args:
            model: The model.
            seed: Noise seed.
            steps: Denoising steps.
            cfg: Guidance scale.
            sampler_name: Solver name.
            scheduler: Scheduler name.
            positive: Positive conditioning.
            negative: Negative conditioning.
            latent_image: The latent to denoise.
            denoise: Fraction of the schedule run.
            preview_every: Preview cadence in steps.

        Returns:
            The denoised latent.
        """
        latent = latent_image
        samples = comfy.sample.fix_empty_latent_channels(
            model, latent["samples"],
            latent.get("downscale_ratio_spacial", None),
            latent.get("downscale_ratio_temporal", None),
        )
        noise = comfy.sample.prepare_noise(samples, seed, latent.get("batch_index"))
        callback = preview.prepare_callback(model, steps, every=preview_every)
        result = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler, positive, negative, samples,
            denoise=denoise, noise_mask=latent.get("noise_mask"), callback=callback,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed,
        )
        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = result
        return io.NodeOutput(out)

"""Sampling that starts at a reduced resolution and grows partway through."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.sampling import speed, spectral


class SpeedSampler(io.ComfyNode):
    """Build a sampler that spends its early steps on a smaller latent."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASSpeedSampler",
            display_name="SPEED Sampler",
            search_aliases=[
                'WASSpeedSampler',
                "SPEED Sampler",
                "spectral progressive diffusion",
                "progressive resolution sampler",
                "coarse to fine sampling",
                "multi resolution sampler",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Sample the early steps at a reduced resolution and grow the latent partway "
                "through, in a spectral basis so the detail that appears is resolved rather "
                "than interpolated. Needs a flow-matching model, and the spectrum values it "
                "schedules from should be measured with Latent Power Spectrum."
            ),
            inputs=[
                io.Combo.Input(
                    "base_sampler",
                    options=speed.base_sampler_names(),
                    default="euler",
                    tooltip=(
                        "The solver each segment runs. A solver that carries state between "
                        "steps starts that state again at every transition, so the simpler "
                        "ones behave most predictably here."
                    ),
                ),
                io.String.Input(
                    "scales",
                    default="0.5,1.0",
                    tooltip=(
                        "Fractions of the full resolution to sample at, increasing, ending at "
                        "1.0. `0.5,1.0` starts at half size and grows once. A single `1.0` "
                        "disables growth and samples normally throughout."
                    ),
                ),
                io.Combo.Input(
                    "transform",
                    options=list(spectral.TRANSFORMS),
                    default="dct",
                    tooltip=(
                        "The basis the latent grows in. `dct` and `fft` handle any ratio "
                        "between scales; `dwt` is cheaper but only ever doubles, so every step "
                        "in the list has to be exactly twice the one before it."
                    ),
                ),
                io.Float.Input(
                    "delta",
                    default=0.01,
                    min=0.0001,
                    max=0.5,
                    step=0.001,
                    tooltip=(
                        "How much leftover noise counts as burying a frequency. Larger values "
                        "hold the small grid for more steps, saving more but leaving less of "
                        "the schedule to resolve detail in. The default is conservative: it "
                        "grows after about a sixth of the steps."
                    ),
                ),
                io.Float.Input(
                    "amplitude",
                    default=203.615097,
                    min=0.0,
                    max=1000000.0,
                    step=0.001,
                    tooltip=(
                        "The A of the model's power spectrum, eg 203.6 for FLUX.1-dev. Only "
                        "meaningful alongside delta: a measured A runs larger than a published "
                        "one, so take both from the same place."
                    ),
                ),
                io.Float.Input(
                    "beta",
                    default=1.915461,
                    min=0.0,
                    max=10.0,
                    step=0.001,
                    tooltip=(
                        "How fast the model's spectrum falls away with frequency, as "
                        "`1.915461` for FLUX.1-dev. Unlike the amplitude it does not depend on "
                        "latent scale, so measured and published values compare directly."
                    ),
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFF,
                    tooltip=(
                        "Seeds the noise that fills the frequencies each transition adds. "
                        "Changing it varies the fine detail without moving the composition. "
                        "Any whole number; `0` is as good a seed as any."
                    ),
                ),
                io.String.Input(
                    "manual_sigmas",
                    default="",
                    optional=True,
                    tooltip=(
                        "Sigmas to grow at, one per transition, decreasing. Leave this empty "
                        "to have them worked out from the amplitude and beta instead, which is "
                        "the usual way round. Setting them ignores delta, amplitude and beta."
                    ),
                ),
            ],
            outputs=[
                io.Sampler.Output(
                    display_name="SAMPLER",
                    tooltip="Feeds the SAMPLER socket of SamplerCustom.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        base_sampler: str,
        scales: str,
        transform: str,
        delta: float,
        amplitude: float,
        beta: float,
        seed: int,
        manual_sigmas: str = "",
    ) -> io.NodeOutput:
        """Bind the settings into a sampler ComfyUI can call.

        Raises:
            ValueError: The scales or sigmas do not parse, or there are not as many sigmas as
                there are transitions between scales.
        """
        import comfy.samplers

        parsed_scales = speed.parse_scales(scales)
        parsed_sigmas = speed.parse_sigmas(manual_sigmas) if manual_sigmas.strip() else []
        if parsed_sigmas and len(parsed_sigmas) != len(parsed_scales) - 1:
            raise ValueError(
                f"SPEED Sampler was given {len(parsed_scales)} scales, which means "
                f"{len(parsed_scales) - 1} transition(s), but {len(parsed_sigmas)} manual "
                f"sigma(s). Give one sigma per transition, or leave manual_sigmas empty."
            )

        sampler = comfy.samplers.KSAMPLER(
            speed.sample_speed,
            extra_options={
                "transform": transform,
                "base_sampler": base_sampler,
                "scales": parsed_scales,
                "manual_sigmas": parsed_sigmas,
                "delta": float(delta),
                "amplitude": float(amplitude),
                "beta": float(beta),
                "seed": int(seed),
            },
        )
        return io.NodeOutput(sampler)

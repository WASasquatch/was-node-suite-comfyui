"""Colored noise sampling, attached to a model so any stochastic sampler uses it."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.sampling import cns

MODE_HINT = (
    "`auto` = the published settings, the guided set when the run's CFG is above 1 and the "
    "unguided set otherwise, and ignores every widget below; `manual` = the widgets."
)


class CNSModelPatch(io.ComfyNode):
    """Colour the noise a stochastic sampler injects, band by band."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        defaults = cns.UNGUIDED
        return io.Schema(
            node_id="WASCNSModelPatch",
            display_name="CNS Model Patch",
            search_aliases=[
                "WASCNSModelPatch",
                "CNS Model Patch",
                "colored noise sampling",
                "coloured noise",
                "spectral noise",
                "frequency noise",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Make a stochastic sampler put its fresh noise where the image is still "
                "unfinished. Coarse shapes settle early and fine detail late, so white noise "
                "keeps disturbing what is already done; this moves that noise toward the "
                "detail still forming. Works with any model, and with any sampler that adds "
                "noise each step, such as `euler_ancestral`, `dpmpp_2m_sde`, `er_sde`, "
                "RES4LYF's samplers or `hfx_stochastic`; a sampler that adds none runs "
                "unchanged. It learns each model's progress from its own runs, so it gets "
                "closer from the second run of a model at a size."
            ),
            inputs=[
                io.Model.Input("model", tooltip="The model to patch; any model loads."),
                io.Combo.Input(
                    "mode",
                    options=list(cns.MODES),
                    default=cns.MODES[0],
                    tooltip=MODE_HINT,
                ),
                io.Int.Input(
                    "bands",
                    default=cns.DEFAULT_BANDS,
                    min=cns.MIN_BANDS,
                    max=cns.MAX_BANDS,
                    tooltip=(
                        "Frequency rings the noise is split into, as `32`, or `64` for a "
                        "finer split on a large latent. Read by `manual`."
                    ),
                ),
                io.Float.Input(
                    "divider",
                    default=defaults.divider,
                    min=1.0,
                    max=100.0,
                    step=0.01,
                    tooltip=(
                        "How much noise a finished band keeps, as `1.0` for none, `1.73` for "
                        "at least 42% of it, or `25` for nearly all of it. Read by `manual`."
                    ),
                ),
                io.Float.Input(
                    "power",
                    default=defaults.power,
                    min=0.1,
                    max=2.0,
                    step=0.05,
                    tooltip=(
                        "How sharply noise follows each band's progress, as `0.5` for the "
                        "square root, `0.75` or `1.0` for linear. Read by `manual`."
                    ),
                ),
                io.Float.Input(
                    "tilt_start",
                    default=defaults.tilt_start,
                    min=-2.0,
                    max=2.0,
                    step=0.01,
                    tooltip=(
                        "Extra lean toward fine detail at the first step, as `0.15` to add "
                        "a little high-frequency noise, `0.0` for none or `-0.3` to take some "
                        "away. Read by `manual`."
                    ),
                ),
                io.Float.Input(
                    "tilt_end",
                    default=defaults.tilt_end,
                    min=-2.0,
                    max=2.0,
                    step=0.01,
                    tooltip=(
                        "The same lean at the last step, as `-0.5` to quiet fine noise as "
                        "the image settles or `0.0` for none. Read by `manual`."
                    ),
                ),
                io.Float.Input(
                    "sharpness",
                    default=defaults.sharpness,
                    min=-8.0,
                    max=8.0,
                    step=0.05,
                    tooltip=(
                        "How the lean moves from start to end, as `0.0` for evenly, `0.75` "
                        "for a little later, or `4.0` for mostly at the end. Read by `manual`."
                    ),
                ),
                io.Float.Input(
                    "energy",
                    default=defaults.energy,
                    min=0.5,
                    max=1.5,
                    step=0.001,
                    tooltip=(
                        "Total strength of the noise against white, as `0.98`, `1.0` for "
                        "the same or `1.05` for a little more. Read by `manual`."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="model",
                    tooltip="The patched model, for any sampler node.",
                ),
            ],
        )

    @classmethod
    def execute(cls, model, mode, bands, divider, power, tilt_start, tilt_end, sharpness,
                energy) -> io.NodeOutput:
        """Clone the model with the colouring attached."""
        settings = cns.Settings(
            mode=mode if mode in cns.MODES else cns.MODES[0],
            bands=int(bands),
            allocation=cns.Allocation(
                divider=float(divider),
                power=float(power),
                tilt_start=float(tilt_start),
                tilt_end=float(tilt_end),
                sharpness=float(sharpness),
                energy=float(energy),
            ),
        )
        return io.NodeOutput(cns.patched(model, settings))

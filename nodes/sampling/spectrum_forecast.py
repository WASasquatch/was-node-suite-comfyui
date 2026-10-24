"""Spectrum Forecast: some sampling steps forecast instead of running the transformer."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.log import get_logger
from ...modules.model import spectrum

logger = get_logger("nodes.sampling.spectrum_forecast")

NODE_NAME = "Spectrum Forecast"


class SpectrumForecast(io.ComfyNode):
    """Run some sampling steps in full and forecast the rest from them."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASSpectrumForecast",
            display_name=NODE_NAME,
            search_aliases=[
                "WASSpectrumForecast",
                NODE_NAME,
                "spectrum",
                "feature forecast",
                "sampling speedup",
                "step cache",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Make sampling faster by running some steps through the whole transformer and "
                "forecasting the rest from them, so 25 steps cost 13 full passes. MOSS-Image, "
                "FLUX.1, Fibo 2, Qwen-Image, Wan and MiniMax H3 forecast their last block; other "
                "models forecast their output."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="The diffusion model to sample, after any LoRAs and patches.",
                ),
                io.Float.Input(
                    "start_percent", default=0.3, min=0.0, max=1.0, step=0.01,
                    tooltip=(
                        "Sampling percent where forecasting starts. Every step before it runs in "
                        "full: 0.3 is the first 8 of 25 steps, 0.2 the first 5."
                    ),
                ),
                io.Float.Input(
                    "end_percent", default=0.95, min=0.0, max=1.0, step=0.01,
                    tooltip=(
                        "Sampling percent where forecasting ends. Every step after it runs in full: "
                        "0.95 is the last of 25 steps, 1.0 forecasts to the end."
                    ),
                ),
                io.Float.Input(
                    "interval", default=2.0, min=1.0, max=8.0, step=0.25,
                    tooltip=(
                        "Steps from one full run to the next where the window starts, and where it "
                        "ends when tapering: 2 runs every other step."
                    ),
                ),
                io.Float.Input(
                    "growth", default=2.0, min=0.0, max=8.0, step=0.05,
                    tooltip=(
                        "Steps added to the interval after each full run: 2.0 runs 13 of 25 steps "
                        "in full, 1.0 runs 14, 0 keeps the interval fixed and runs 17."
                    ),
                ),
                io.Boolean.Input(
                    "taper", default=True,
                    tooltip=(
                        "On, the interval widens toward the middle of the window and narrows again "
                        "toward end_percent. Off, it widens all the way to end_percent."
                    ),
                ),
                io.Float.Input(
                    "tolerance", default=0.0, min=0.0, max=1.0, step=0.01, optional=True,
                    advanced=True,
                    tooltip=(
                        "How far the curved and the straight-line forecasts of a step may disagree "
                        "before it runs in full instead: 0.1 for 10%. 0 never checks."
                    ),
                ),
                io.Combo.Input(
                    "target", options=list(spectrum.TARGETS), default=spectrum.FEATURES,
                    optional=True, advanced=True,
                    tooltip=(
                        "What is forecast: block output, the last block's output; block residual, its "
                        "change over the step's own input; model output, the whole prediction."
                    ),
                ),
                io.Combo.Input(
                    "coordinate", options=list(spectrum.COORDINATES), default=spectrum.STEP,
                    optional=True, advanced=True,
                    tooltip=(
                        "What the fit runs over: `step` for the step number, `sigma` for the noise "
                        "level, `log-snr` for the log signal-to-noise ratio."
                    ),
                ),
                io.Int.Input(
                    "degree", default=4, min=1, max=8, optional=True, advanced=True,
                    tooltip="Highest Chebyshev degree of the fit: 4; 2 is smoother, 6 follows more closely.",
                ),
                io.Float.Input(
                    "ridge", default=0.1, min=0.0, max=10.0, step=0.01, optional=True, advanced=True,
                    tooltip="Ridge weight of the fit: 0.1; 0 fits the full steps exactly, 1 smooths more.",
                ),
                io.Int.Input(
                    "history", default=8, min=3, max=64, optional=True, advanced=True,
                    tooltip="Most recent full steps the fit is taken over: 8; fewer uses less memory.",
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="model",
                    tooltip="The model, sampling with forecast steps. Connect it to the sampler.",
                ),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model, start_percent=0.3, end_percent=0.95, interval=2.0, growth=2.0,
                taper=True, tolerance=0.0, target=spectrum.FEATURES, coordinate=spectrum.STEP,
                degree=4, ridge=0.1, history=8) -> io.NodeOutput:
        """Patch the model to forecast steps.

        Args:
            model: The model.
            start_percent: Sampling percent where forecasting starts.
            end_percent: Sampling percent where forecasting ends.
            interval: Steps between full runs at each end of the window.
            growth: Steps added to the interval after each full run.
            taper: Whether the interval narrows again toward ``end_percent``.
            tolerance: Disagreement past which a forecast step runs in full; 0 never checks.
            target: What is forecast.
            coordinate: What the fit runs over.
            degree: Highest Chebyshev degree.
            ridge: Ridge weight.
            history: Most full steps the fit uses.

        Returns:
            The patched model.
        """
        settings = spectrum.Settings(
            start_percent=float(start_percent), end_percent=float(end_percent),
            interval=float(interval), growth=float(growth), taper=bool(taper),
            tolerance=float(tolerance), degree=int(degree),
            ridge=float(ridge), history=int(history), target=str(target),
            coordinate=str(coordinate),
        )
        patched, keys = spectrum.apply(model, settings, node_id=cls.hidden.unique_id)
        logger.info(
            "%s: forecasting at the %s, %s",
            NODE_NAME, "last block" if keys else "model output",
            f"{len(keys)} block hooks" if keys else "no block hooks",
        )
        return io.NodeOutput(patched)

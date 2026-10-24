"""Adaptive Detail: more fine detail from the sampling steps where detail forms."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.log import get_logger
from ...modules.model import adaptive_detail

logger = get_logger("nodes.sampling.adaptive_detail")

NODE_NAME = "Adaptive Detail"


class AdaptiveDetail(io.ComfyNode):
    """Push sampling toward fine detail on the steps where detail forms."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        defaults = adaptive_detail.Settings()
        return io.Schema(
            node_id="WASAdaptiveDetail",
            display_name=NODE_NAME,
            search_aliases=[
                "WASAdaptiveDetail",
                NODE_NAME,
                "detail",
                "more detail",
                "detail push",
                "sharper sampling",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Add fine detail and texture to a render by telling the model a slightly lower "
                "noise level on the steps where detail forms, and optionally by giving the steps "
                "with the most fine-detail error a corrective second pass. Works with samplers "
                "that call the model once per step, such as euler."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="The diffusion model to sample, after any LoRAs and patches.",
                ),
                io.Float.Input(
                    "detail", default=defaults.detail, min=0.0, max=0.3, step=0.005,
                    tooltip=(
                        "0.04 = clearly more texture; 0.02 = subtle; 0.06 = strong, small objects "
                        "start to change; 0 = off. How much lower a noise level the model is told."
                    ),
                ),
                io.Float.Input(
                    "start_percent", default=defaults.start_percent, min=0.0, max=1.0, step=0.01,
                    tooltip=(
                        "Sampling percent where the push and the extra passes start: 0.3 leaves "
                        "the first 8 of 25 steps, where the composition settles, alone."
                    ),
                ),
                io.Float.Input(
                    "end_percent", default=defaults.end_percent, min=0.0, max=1.0, step=0.01,
                    tooltip=(
                        "Sampling percent where the push stops: 0.8 leaves the last 5 of 25 steps "
                        "to clean it up; nearer 1.0 leaves grain."
                    ),
                ),
                io.Float.Input(
                    "extra", default=defaults.extra, min=0.0, max=1.0, step=0.05,
                    tooltip=(
                        "Corrective second passes as a share of the steps, on the steps with the "
                        "most fine-detail error: 0.25 adds 6 to 25 steps; 0 adds none."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="model",
                    tooltip="The model, sampling toward more detail. Connect it to the sampler.",
                ),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model, detail=0.04, start_percent=0.3, end_percent=0.8,
                extra=0.0) -> io.NodeOutput:
        """Patch the model to push its sampling toward detail.

        Args:
            model: The model.
            detail: Share by which the noise level the model is told is lowered.
            start_percent: Sampling percent where the push and the extra passes start.
            end_percent: Sampling percent where the push stops.
            extra: Second-order passes as a share of the steps.

        Returns:
            The patched model.
        """
        settings = adaptive_detail.Settings(
            detail=float(detail), start_percent=float(start_percent),
            end_percent=float(end_percent), extra=float(extra),
        )
        patched = adaptive_detail.apply(model, settings, node_id=cls.hidden.unique_id)
        logger.info("%s: detail %.3f, %.0f%% extra passes", NODE_NAME, settings.detail,
                    settings.extra * 100)
        return io.NodeOutput(patched)

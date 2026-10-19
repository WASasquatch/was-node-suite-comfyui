"""A flow model's shift set as a plain factor, optionally scaled to the picture's size."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.sampling import schedules, tiled_diffusion

logger = log.get_logger("nodes.sampling.flow_shift")


class FlowShift(io.ComfyNode):
    """Set a flow model's shift, the time the sampler spends on layout against detail."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASFlowShift",
            display_name="Flow Shift",
            search_aliases=[
                "WASFlowShift",
                "Flow Shift",
                "flux 3 klein",
                "moss image",
                "flow shift",
                "model sampling flux",
                "model sampling sd3",
                "time shift",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Set a flow model's shift, such as FLUX 3 Image Klein's or MOSS-Image's, optionally "
                "grown with the picture's size. Higher spends more steps on layout and lower on "
                "detail, under schedulers that read the model, as simple and shifted_tangent do. "
                "Beside Tiled Diffusion Model Patch it lays out only the whole-frame steps."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="A flow model: FLUX 3 Image Klein, MOSS-Image, FLUX, SD3 and alike; MODEL.",
                ),
                io.Float.Input(
                    "shift",
                    default=0.0,
                    min=0.0,
                    max=1000.0,
                    step=0.01,
                    tooltip="Shift as a factor. 0 = the model's own: 6.93 on Flux 3 Klein, 5 on "
                    "MOSS-Image; 3 = more steps on detail; 12 = more on layout.",
                ),
                io.DynamicCombo.Input(
                    "resolution",
                    options=[
                        io.DynamicCombo.Option(key="fixed", inputs=[]),
                        io.DynamicCombo.Option(key="scaled", inputs=[
                            io.Int.Input("width", default=1024, min=16, max=16384, step=16,
                                         tooltip="Picture width in pixels, as `1088`."),
                            io.Int.Input("height", default=1024, min=16, max=16384, step=16,
                                         tooltip="Picture height in pixels, as `1856`."),
                        ]),
                    ],
                    tooltip="`fixed` = the shift as set; `scaled` = the shift times the square "
                    "root of the area over 1024 x 1024, so MOSS-Image's 5 becomes 7.1 at 1088 x 1856.",
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="model",
                    tooltip="The model sampling at the new shift; MODEL.",
                ),
            ],
        )

    @classmethod
    def execute(cls, model, shift, resolution) -> io.NodeOutput:
        """Patch a clone of ``model`` with the shift.

        Raises:
            ValueError: The model samples with no flow shift.
        """
        choice = resolution or {}
        area = 0
        if choice.get("resolution") == "scaled":
            area = int(choice.get("width", 1024)) * int(choice.get("height", 1024))
        patched, factor = schedules.flow_shift(model, float(shift), area)
        logger.info("Flow Shift: sampling at shift %.3f", factor)
        if patched is not model:
            tiled_diffusion.settle(patched, "Flow Shift")
        return io.NodeOutput(patched)

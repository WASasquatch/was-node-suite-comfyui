"""A two-stage arctangent noise schedule that follows the model's flow shift."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.sampling import schedules


class ShiftedTangentScheduler(io.ComfyNode):
    """Build the shifted tangent schedule for a model, with every curve setting exposed."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASShiftedTangentScheduler",
            display_name="Shifted Tangent Scheduler",
            search_aliases=[
                "WASShiftedTangentScheduler",
                "Shifted Tangent Scheduler",
                "bong_tangent",
                "tangent scheduler",
                "sigmas",
                "flow shift",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Make the noise schedule of RES4LYF's bong_tangent in the model's own time, so the "
                "model's flow shift moves where the steps fall. It is also offered as "
                "shifted_tangent in every KSampler's scheduler menu, at these defaults."
            ),
            inputs=[
                io.Model.Input("model", tooltip="The model the schedule is read through; MODEL."),
                io.Int.Input("steps", default=20, min=1, max=10000,
                             tooltip="Sampling steps, as `20`."),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Share of the schedule run. 1 = from pure noise; 0.5 = "
                               "the second half only, for a picture already made."),
                io.Float.Input("start", default=1.0, min=0.0, max=1.0, step=0.01, advanced=True,
                               tooltip="Share of the model's time the first step starts from. 1 = full noise."),
                io.Float.Input("middle", default=0.5, min=0.0, max=1.0, step=0.01, advanced=True,
                               tooltip="Share of the model's time where the second stage takes over, as `0.5`."),
                io.Float.Input("end", default=0.0, min=0.0, max=1.0, step=0.01, advanced=True,
                               tooltip="Share of the model's time the last step ends at. 0 = clean."),
                io.Float.Input("pivot_1", default=0.6, min=0.0, max=1.0, step=0.01, advanced=True,
                               tooltip="Share of the steps where the first stage turns, as `0.6`."),
                io.Float.Input("pivot_2", default=0.6, min=0.0, max=1.0, step=0.01, advanced=True,
                               tooltip="Share of the steps where the second stage turns, as `0.6`."),
                io.Float.Input("slope_1", default=0.2, min=0.01, max=10.0, step=0.01, advanced=True,
                               tooltip="Steepness of the first stage. 0.2 = gentle; 1 = abrupt."),
                io.Float.Input("slope_2", default=0.2, min=0.01, max=10.0, step=0.01, advanced=True,
                               tooltip="Steepness of the second stage. 0.2 = gentle; 1 = abrupt."),
            ],
            outputs=[
                io.Sigmas.Output(
                    display_name="sigmas",
                    tooltip="The schedule, steps + 1 values ending at 0, for SamplerCustom; SIGMAS.",
                ),
            ],
        )

    @classmethod
    def execute(cls, model, steps, denoise, start=1.0, middle=0.5, end=0.0, pivot_1=0.6,
                pivot_2=0.6, slope_1=0.2, slope_2=0.2) -> io.NodeOutput:
        """Read the tangent curve through ``model``'s sampling curve."""
        import torch

        if denoise <= 0.0:
            return io.NodeOutput(torch.FloatTensor([]))
        total = steps if denoise >= 1.0 else int(steps / denoise)
        sigmas = schedules.tangent(
            model.get_model_object("model_sampling"), total, start=start, middle=middle, end=end,
            pivot_1=pivot_1, pivot_2=pivot_2, slope_1=slope_1, slope_2=slope_2,
        )
        return io.NodeOutput(sigmas[-(steps + 1):])

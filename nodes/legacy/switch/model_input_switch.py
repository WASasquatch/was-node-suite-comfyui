"""Route one of two diffusion models onward."""

from __future__ import annotations

from comfy_api.latest import io

REQUIRES = "switches"


class ModelInputSwitch(io.ComfyNode):
    """Select between two diffusion models with a boolean."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Model Input Switch",
            display_name="Model Input Switch",
            search_aliases=["Model Input Switch", "model switch", "boolean switch"],
            category="WAS Suite/Logic/Switch",
            description=(
                "Deprecated: use Model Switch instead. It takes the type of whatever is "
                "connected and skips the branch it does not select. This node passes one of two "
                "diffusion models on, chosen by a boolean: model_a when the boolean is true, "
                "model_b when it is false."
            ),
            inputs=[
                io.Model.Input(
                    "model_a",
                    lazy=True,
                    tooltip="The diffusion model sent on when boolean is true.",
                ),
                io.Model.Input(
                    "model_b",
                    lazy=True,
                    tooltip="The diffusion model sent on when boolean is false.",
                ),
                io.Boolean.Input(
                    "boolean",
                    default=True,
                    tooltip=(
                        "Which input passes; BOOLEAN. true = model_a, false = model_b. "
                        "Toggle it, or wire it from Logic Boolean."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output(tooltip="Whichever of the two diffusion models was selected."),
            ],
            is_deprecated=True,
        )

    @classmethod
    def check_lazy_status(cls, boolean=True, **values) -> list[str]:
        """The input the boolean selects while it is still to be worked out; the other is never run."""
        chosen = "model_a" if boolean else "model_b"
        return [chosen] if values.get(chosen) is None else []

    @classmethod
    def execute(cls, model_a, model_b, boolean=True) -> io.NodeOutput:
        return io.NodeOutput(model_a if boolean else model_b)

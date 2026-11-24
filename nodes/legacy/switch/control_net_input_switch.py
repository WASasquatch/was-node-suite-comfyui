"""Route one of two ControlNet models onward."""

from __future__ import annotations

from comfy_api.latest import io

REQUIRES = "switches"


class ControlNetModelInputSwitch(io.ComfyNode):
    """Select between two ControlNet models with a boolean."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Control Net Model Input Switch",
            display_name="Control Net Model Input Switch",
            search_aliases=[
                "Control Net Model Input Switch",
                "controlnet switch",
                "boolean switch",
            ],
            category="WAS Suite/Logic/Switch",
            description=(
                "Deprecated: use Model Switch instead. It takes the type of whatever is "
                "connected and skips the branch it does not select. This node passes one of two "
                "ControlNet models on, chosen by a boolean: control_net_a when the boolean is "
                "true, control_net_b when it is false."
            ),
            inputs=[
                io.ControlNet.Input(
                    "control_net_a",
                    lazy=True,
                    tooltip="The ControlNet model sent on when boolean is true.",
                ),
                io.ControlNet.Input(
                    "control_net_b",
                    lazy=True,
                    tooltip="The ControlNet model sent on when boolean is false.",
                ),
                io.Boolean.Input(
                    "boolean",
                    default=True,
                    tooltip=(
                        "Which input passes; BOOLEAN. true = control_net_a, false = "
                        "control_net_b. Toggle it, or wire it from Logic Boolean."
                    ),
                ),
            ],
            outputs=[
                io.ControlNet.Output(
                    tooltip="Whichever of the two ControlNet models was selected.",
                ),
            ],
            is_deprecated=True,
        )

    @classmethod
    def check_lazy_status(cls, boolean=True, **values) -> list[str]:
        """The input the boolean selects while it is still to be worked out; the other is never run."""
        chosen = "control_net_a" if boolean else "control_net_b"
        return [chosen] if values.get(chosen) is None else []

    @classmethod
    def execute(cls, control_net_a, control_net_b, boolean=True) -> io.NodeOutput:
        return io.NodeOutput(control_net_a if boolean else control_net_b)

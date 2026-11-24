"""Route one of two conditionings onward."""

from __future__ import annotations

from comfy_api.latest import io

REQUIRES = "switches"


class ConditioningInputSwitch(io.ComfyNode):
    """Select between two conditionings with a boolean."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Conditioning Input Switch",
            display_name="Conditioning Input Switch",
            search_aliases=[
                "Conditioning Input Switch",
                "conditioning switch",
                "boolean switch",
            ],
            category="WAS Suite/Logic/Switch",
            description=(
                "Deprecated: use Any Input Switch instead. It takes the type of whatever is "
                "connected and skips the branch it does not select. This node passes one of two "
                "conditionings on, chosen by a boolean: conditioning_a when the boolean is true, "
                "conditioning_b when it is false."
            ),
            inputs=[
                io.Conditioning.Input(
                    "conditioning_a",
                    lazy=True,
                    tooltip="The encoded prompt sent on when boolean is true.",
                ),
                io.Conditioning.Input(
                    "conditioning_b",
                    lazy=True,
                    tooltip="The encoded prompt sent on when boolean is false.",
                ),
                io.Boolean.Input(
                    "boolean",
                    default=True,
                    tooltip=(
                        "Which input passes; BOOLEAN. true = conditioning_a, false = "
                        "conditioning_b. Toggle it, or wire it from Logic Boolean."
                    ),
                ),
            ],
            outputs=[
                io.Conditioning.Output(
                    tooltip="Whichever of the two conditionings was selected.",
                ),
            ],
            is_deprecated=True,
        )

    @classmethod
    def check_lazy_status(cls, boolean=True, **values) -> list[str]:
        """The input the boolean selects while it is still to be worked out; the other is never run."""
        chosen = "conditioning_a" if boolean else "conditioning_b"
        return [chosen] if values.get(chosen) is None else []

    @classmethod
    def execute(cls, conditioning_a, conditioning_b, boolean=True) -> io.NodeOutput:
        return io.NodeOutput(conditioning_a if boolean else conditioning_b)

"""Route one of two strings onward."""

from __future__ import annotations

from comfy_api.latest import io

REQUIRES = "switches"


class TextInputSwitch(io.ComfyNode):
    """Select between two strings with a boolean."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Text Input Switch",
            display_name="Text Input Switch",
            search_aliases=["Text Input Switch", "text switch", "string switch"],
            category="WAS Suite/Logic/Switch",
            description=(
                "Deprecated: use Any Input Switch instead. It takes the type of whatever is "
                "connected and skips the branch it does not select. This node passes one of two "
                "pieces of text on, chosen by a boolean: text_a when the boolean is true, text_b "
                "when it is false."
            ),
            inputs=[
                io.String.Input(
                    "text_a",
                    multiline=True,
                    placeholder="Eg: a long prompt",
                    lazy=True,
                    tooltip=(
                        "Sent out when boolean is true; STRING."
                    ),
                ),
                io.String.Input(
                    "text_b",
                    multiline=True,
                    placeholder="Eg: a short prompt",
                    lazy=True,
                    tooltip=(
                        "Sent out when boolean is false; STRING."
                    ),
                ),
                io.Boolean.Input(
                    "boolean",
                    default=True,
                    tooltip=(
                        "Which input passes; BOOLEAN. true = text_a, false = text_b. "
                        "Toggle it, or wire it from Logic Boolean."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(tooltip="Whichever of the two pieces of text was selected."),
            ],
            is_deprecated=True,
        )

    @classmethod
    def check_lazy_status(cls, boolean=True, **values) -> list[str]:
        """The input the boolean selects while it is still to be worked out; the other is never run."""
        chosen = "text_a" if boolean else "text_b"
        return [chosen] if values.get(chosen) is None else []

    @classmethod
    def execute(cls, text_a, text_b, boolean=True) -> io.NodeOutput:
        return io.NodeOutput(text_a if boolean else text_b)

"""A dropdown that takes the options of the dropdown it is connected to."""

from __future__ import annotations

from comfy_api.latest import io

#: What the dropdown holds until its output is connected to another dropdown.
UNCONNECTED = "connect combo to a dropdown"


class ComboPrimitive(io.ComfyNode):
    """Choose from the options of the dropdown its output feeds."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASComboPrimitive",
            display_name="Combo Primitive",
            search_aliases=[
                "WASComboPrimitive",
                "Combo Primitive",
                "combo clone",
                "clone dropdown",
                "dropdown primitive",
                "copy combo",
            ],
            category="WAS Suite/Logic",
            description=(
                "A dropdown that copies the options and the current choice of any dropdown its "
                "output is connected to, so one choice can be set in one place and drive several "
                "nodes. Connect combo to a dropdown's input, then pick from the copied list."
            ),
            inputs=[
                io.Combo.Input(
                    "value",
                    options=[UNCONNECTED],
                    default=UNCONNECTED,
                    tooltip=(
                        "The choice sent on, from the options of the dropdown combo is connected "
                        "to, such as euler, karras or sd_xl_base_1.0.safetensors. With several "
                        "connected, the first one's options are listed."
                    ),
                ),
            ],
            outputs=[
                io.AnyType.Output(
                    display_name="combo",
                    tooltip=(
                        "The choice, on a wire any dropdown takes: connect it to sampler_name, "
                        "ckpt_name or a node's mode."
                    ),
                ),
                io.String.Output(
                    display_name="text",
                    tooltip="The choice as text: euler, karras, sd_xl_base_1.0.safetensors.",
                ),
            ],
        )

    @classmethod
    def validate_inputs(cls, value) -> bool | str:
        """Accept any choice the copied list offered, and refuse the unconnected placeholder.

        Args:
            value: The chosen option.

        Returns:
            True, or the message naming what is missing.
        """
        if not isinstance(value, str) or not value.strip() or value == UNCONNECTED:
            return (
                "Combo Primitive has no choice yet. Connect combo to a node's dropdown, then "
                "pick from the list it copies."
            )
        return True

    @classmethod
    def execute(cls, value: str) -> io.NodeOutput:
        """Send the choice on.

        Args:
            value: The chosen option.

        Returns:
            The choice on a dropdown wire, and as text.
        """
        return io.NodeOutput(value, value)

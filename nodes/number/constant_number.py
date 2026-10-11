"""A constant numeric value, optionally parsed from text."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import NUMBER


class ConstantNumber(io.ComfyNode):
    """Emit a fixed value, coerced to the selected numeric type."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Constant Number",
            display_name="Constant Number",
            search_aliases=["Constant Number", "constant", "value", "literal"],
            category="WAS Suite/Number",
            description=(
                "Emit a constant number. `integer` truncates the widget value, `float` keeps "
                "it, and `bool` emits 1 when it is greater than 0.5 and 0 otherwise. A "
                "number_as_text holding anything is parsed in place of the widget."
            ),
            inputs=[
                io.Combo.Input(
                    "number_type",
                    options=["integer", "float", "bool"],
                    tooltip=(
                        "How the value is read. `integer` cuts off any fraction, so 8.7 "
                        "becomes 8; `float` keeps it as typed; `bool` collapses it to 1 when "
                        "it is above 0.5 and to 0 otherwise."
                    ),
                ),
                io.Float.Input(
                    "number",
                    default=0,
                    min=-18446744073709551615,
                    max=18446744073709551615,
                    step=0.01,
                    tooltip=(
                        "The value to send on, before number_type is applied: 8.7 leaves as 8 "
                        "in `integer` and 8.7 in `float`. Ignored while number_as_text holds "
                        "anything."
                    ),
                ),
                io.String.Input(
                    "number_as_text",
                    optional=True,
                    placeholder="exact digits, eg 9007199254740993",
                    tooltip=(
                        "Digits used instead of the widget. `integer` keeps every digit: "
                        "9007199254740993; `float` also takes -12.5, .5 or 1e6; `bool` takes "
                        "true, 1, yes, on or false, 0, no, off. A sum such as 2^53+1 is "
                        "refused; use Number Expression."
                    ),
                ),
            ],
            outputs=[
                NUMBER.Output(
                    tooltip=(
                        "The value in the chosen type: whole in `integer` and `bool` mode, "
                        "decimal in `float` mode."
                    ),
                ),
                io.Float.Output(tooltip="The same value as a float, so 8 leaves here as 8.0."),
                io.Int.Output(
                    tooltip=(
                        "The same value as a whole number, cut off rather than rounded, so "
                        "8.7 leaves here as 8."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, number_type, number, number_as_text=None) -> io.NodeOutput:
        """Answer the constant in the chosen type.

        Args:
            number_type: How the value is read.
            number: The widget value.
            number_as_text: The number in digits, read in place of the widget.

        Returns:
            The value as a number, a float and a whole number.

        Raises:
            ValueError: The text is not a number.
        """
        from ...modules.logic.compare import to_boolean

        if number_as_text:
            if number_type == "bool":
                number = to_boolean(number_as_text)
            else:
                try:
                    number = (
                        int(number_as_text) if number_type == "integer"
                        else float(number_as_text)
                    )
                except ValueError as unreadable:
                    raise ValueError(
                        f"Constant Number was given {number_as_text!r} as number_as_text, "
                        f"which is not a number. Type the digits alone, such as 8 or -12.5"
                    ) from unreadable

        if number_type == "integer":
            return io.NodeOutput(int(number), float(number), int(number))
        if number_type == "float":
            return io.NodeOutput(float(number), float(number), int(number))
        if number_type == "bool":
            boolean = 1 if float(number) > 0.5 else 0
            return io.NodeOutput(int(boolean), float(boolean), int(boolean))
        return io.NodeOutput(number, float(number), int(number))

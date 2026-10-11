"""Drive another node's dropdown from text."""

from __future__ import annotations

from comfy_api.latest import io


class TextToCombo(io.ComfyNode):
    """Put text on a wire a dropdown accepts."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASTextToCombo",
            display_name="Text to Combo",
            search_aliases=[
                "WASTextToCombo",
                "Text to Combo",
                "string to combo",
                "text to dropdown",
                "combo from text",
                "choose by name",
            ],
            category="WAS Suite/Logic",
            description=(
                "Put text on a wire any dropdown takes, so a checkpoint, LoRA, sampler or "
                "scheduler can come from a loop, a switch or a text node. Convert the target "
                "node's dropdown to an input and connect combo to it."
            ),
            inputs=[
                io.String.Input(
                    "text",
                    default="",
                    tooltip=(
                        "The option to choose, spelled exactly as the dropdown lists it, such "
                        "as sd_xl_base_1.0.safetensors or euler_ancestral."
                    ),
                ),
                io.Boolean.Input(
                    "strip",
                    default=True,
                    optional=True,
                    tooltip=(
                        "Drop spaces and line ends from both ends: ` euler ` becomes `euler`. "
                        "Off sends the text exactly as typed."
                    ),
                ),
            ],
            outputs=[
                io.AnyType.Output(
                    display_name="combo",
                    tooltip=(
                        "The text, on a wire a dropdown accepts. A name the dropdown does not "
                        "list is refused by the node receiving it, not by this one."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, text: str, strip: bool = True) -> io.NodeOutput:
        """Answer the text as a dropdown choice.

        Args:
            text: The option to choose.
            strip: Whether to drop surrounding whitespace first.

        Returns:
            The text, on the wire a dropdown accepts.

        Raises:
            ValueError: The text is empty, which matches no option.
        """
        chosen = text.strip() if strip else text
        if not chosen:
            raise ValueError(
                "Text to Combo was given nothing to choose. Type the option exactly as the "
                "dropdown spells it, or connect text that answers one"
            )
        return io.NodeOutput(chosen)

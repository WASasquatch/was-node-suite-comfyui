"""Route one of two pictures onward, with the socket typed to what it may carry."""

from __future__ import annotations

from comfy_api.latest import io


class TensorImageSwitch(io.ComfyNode):
    """Select between two pictures with a boolean.

    Both inputs are lazy, so the unselected branch is never evaluated.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        template = io.MatchType.Template("tensor_image_switch", [io.Image, io.Mask, io.Latent])
        return io.Schema(
            node_id="WASTensorImageSwitch",
            display_name="Tensor Switch",
            search_aliases=[
                "WASTensorImageSwitch",
                "Tensor Switch",
                "Tensor Image Switch",
                "switch",
                "image switch",
                "mask switch",
                "latent switch",
                "route",
                "branch",
            ],
            category="WAS Suite/Logic/Switch",
            description=(
                "Pass one of two pictures on, chosen by a boolean, where a picture is an image, a mask or a latent. The socket takes those three and refuses anything else, so a wrong wire is caught as it is drawn. The unselected input is not evaluated, so the work behind it is skipped."
            ),
            inputs=[
                io.MatchType.Input(
                    "input_a",
                    template=template,
                    lazy=True,
                    tooltip=(
                        "Passed on when boolean is true. Takes IMAGE, MASK or LATENT. The first connection "
                        "fixes the type; input_b and output then take that type only."
                    ),
                ),
                io.MatchType.Input(
                    "input_b",
                    template=template,
                    lazy=True,
                    tooltip="Passed on when boolean is false. Must match input_a's type.",
                ),
                io.Boolean.Input(
                    "boolean",
                    default=True,
                    tooltip="Selects the input. true = input_a, false = input_b.",
                ),
            ],
            outputs=[
                io.MatchType.Output(
                    template=template,
                    display_name="output",
                    tooltip="The selected input, typed to whatever was connected.",
                ),
            ],
        )

    @classmethod
    def check_lazy_status(cls, boolean=True, input_a=None, input_b=None) -> list[str]:
        """Ask only for the branch that is about to be used.

        Args:
            boolean: Which input the run will select.
            input_a: The true branch, present once it has been evaluated.
            input_b: The false branch, present once it has been evaluated.

        Returns:
            The input still needed, or an empty list once it has arrived.
        """
        if boolean and input_a is None:
            return ["input_a"]
        if not boolean and input_b is None:
            return ["input_b"]
        return []

    @classmethod
    def execute(cls, input_a=None, input_b=None, boolean=True) -> io.NodeOutput:
        return io.NodeOutput(input_a if boolean else input_b)

"""Print whatever is connected to it, and list an object's members."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.log import get_logger

REQUIRES = "extras"

logger = get_logger("nodes.extras.debug")

#: Values printed as they are. Anything else is an object, so its members are listed too.
PLAIN_TYPES = (str, int, float, bool, list, dict, tuple)


class WASDebugInput(io.ComfyNode):
    """Show what is actually travelling down a wire."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASDebugInput",
            display_name="Debug Input",
            search_aliases=["WASDebugInput", "WAS Extras", "print", "inspect", "debug any"],
            category="WAS Suite/Debug",
            description=(
                "Print whatever is connected to it to the console, and for anything that is "
                "not a plain value, list its members as well. Connect it to a wire you want "
                "to understand; it produces no output of its own."
            ),
            inputs=[
                io.AnyType.Input(
                    "input",
                    tooltip=(
                        "Anything at all: an image, a model, a number, a conditioning. Text "
                        "and numbers are printed as they are, and anything else is printed "
                        "along with the names of everything it carries."
                    ),
                ),
            ],
            outputs=[],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, input) -> io.NodeOutput:
        from pprint import pformat

        logger.info("Debug:\n%s", input)
        if isinstance(input, object) and not isinstance(input, PLAIN_TYPES):
            logger.info("Object's directory listing:\n%s", pformat(dir(input), indent=4))
        return io.NodeOutput()

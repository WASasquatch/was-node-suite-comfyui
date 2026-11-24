"""Route one of two images onward."""

from __future__ import annotations

from comfy_api.latest import io

REQUIRES = "switches"


class ImageInputSwitch(io.ComfyNode):
    """Select between two images with a boolean."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Image Input Switch",
            display_name="Image Input Switch",
            search_aliases=["Image Input Switch", "image switch", "boolean switch"],
            category="WAS Suite/Logic/Switch",
            description=(
                "Deprecated: use Tensor Switch instead. It takes the type of whatever is "
                "connected, an image, a mask or a latent, and skips the branch it does not "
                "select. This node passes one of two images on, chosen by a boolean: image_a "
                "when the boolean is true, image_b when it is false."
            ),
            inputs=[
                io.Image.Input(
                    "image_a",
                    lazy=True,
                    tooltip="The image sent on when boolean is true.",
                ),
                io.Image.Input(
                    "image_b",
                    lazy=True,
                    tooltip="The image sent on when boolean is false.",
                ),
                io.Boolean.Input(
                    "boolean",
                    default=True,
                    tooltip=(
                        "Which input passes; BOOLEAN. true = image_a, false = image_b. "
                        "Toggle it, or wire it from Logic Boolean."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(tooltip="Whichever of the two images was selected."),
            ],
            is_deprecated=True,
        )

    @classmethod
    def check_lazy_status(cls, boolean=True, **values) -> list[str]:
        """The input the boolean selects while it is still to be worked out; the other is never run."""
        chosen = "image_a" if boolean else "image_b"
        return [chosen] if values.get(chosen) is None else []

    @classmethod
    def execute(cls, image_a, image_b, boolean=True) -> io.NodeOutput:
        return io.NodeOutput(image_a if boolean else image_b)

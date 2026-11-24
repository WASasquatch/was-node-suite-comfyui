"""Route one of two latents onward."""

from __future__ import annotations

from comfy_api.latest import io

REQUIRES = "switches"


class LatentInputSwitch(io.ComfyNode):
    """Select between two latents with a boolean."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Latent Input Switch",
            display_name="Latent Input Switch",
            search_aliases=["Latent Input Switch", "latent switch", "boolean switch"],
            category="WAS Suite/Logic/Switch",
            description=(
                "Deprecated: use Tensor Switch instead. It takes the type of whatever is "
                "connected, an image, a mask or a latent, and skips the branch it does not "
                "select. This node passes one of two latents on, chosen by a boolean: latent_a "
                "when the boolean is true, latent_b when it is false."
            ),
            inputs=[
                io.Latent.Input(
                    "latent_a",
                    lazy=True,
                    tooltip="The latent sent on when boolean is true.",
                ),
                io.Latent.Input(
                    "latent_b",
                    lazy=True,
                    tooltip="The latent sent on when boolean is false.",
                ),
                io.Boolean.Input(
                    "boolean",
                    default=True,
                    tooltip=(
                        "Which input passes; BOOLEAN. true = latent_a, false = "
                        "latent_b. Toggle it, or wire it from Logic Boolean."
                    ),
                ),
            ],
            outputs=[
                io.Latent.Output(tooltip="Whichever of the two latents was selected."),
            ],
            is_deprecated=True,
        )

    @classmethod
    def check_lazy_status(cls, boolean=True, **values) -> list[str]:
        """The input the boolean selects while it is still to be worked out; the other is never run."""
        chosen = "latent_a" if boolean else "latent_b"
        return [chosen] if values.get(chosen) is None else []

    @classmethod
    def execute(cls, latent_a, latent_b, boolean=True) -> io.NodeOutput:
        return io.NodeOutput(latent_a if boolean else latent_b)

"""Route one of two VAEs onward."""

from __future__ import annotations

from comfy_api.latest import io

REQUIRES = "switches"


class VAEInputSwitch(io.ComfyNode):
    """Select between two VAEs with a boolean."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="VAE Input Switch",
            display_name="VAE Input Switch",
            search_aliases=["VAE Input Switch", "vae switch", "boolean switch"],
            category="WAS Suite/Logic/Switch",
            description=(
                "Deprecated: use Model Switch instead. It takes the type of whatever is "
                "connected and skips the branch it does not select. This node passes one of two "
                "VAEs on, chosen by a boolean: vae_a when the boolean is true, vae_b when it is "
                "false."
            ),
            inputs=[
                io.Vae.Input(
                    "vae_a",
                    lazy=True,
                    tooltip="The VAE sent on when boolean is true.",
                ),
                io.Vae.Input(
                    "vae_b",
                    lazy=True,
                    tooltip="The VAE sent on when boolean is false.",
                ),
                io.Boolean.Input(
                    "boolean",
                    default=True,
                    tooltip=(
                        "Which input passes; BOOLEAN. true = vae_a, false = vae_b. "
                        "Toggle it, or wire it from Logic Boolean."
                    ),
                ),
            ],
            outputs=[
                io.Vae.Output(tooltip="Whichever of the two VAEs was selected."),
            ],
            is_deprecated=True,
        )

    @classmethod
    def check_lazy_status(cls, boolean=True, **values) -> list[str]:
        """The input the boolean selects while it is still to be worked out; the other is never run."""
        chosen = "vae_a" if boolean else "vae_b"
        return [chosen] if values.get(chosen) is None else []

    @classmethod
    def execute(cls, vae_a, vae_b, boolean=True) -> io.NodeOutput:
        return io.NodeOutput(vae_a if boolean else vae_b)

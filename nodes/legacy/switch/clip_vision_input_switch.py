"""Route one of two CLIP Vision models onward."""

from __future__ import annotations

from comfy_api.latest import io

REQUIRES = "switches"


class CLIPVisionInputSwitch(io.ComfyNode):
    """Select between two CLIP Vision models with a boolean."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="CLIP Vision Input Switch",
            display_name="CLIP Vision Input Switch",
            search_aliases=["CLIP Vision Input Switch", "clip vision switch", "boolean switch"],
            category="WAS Suite/Logic/Switch",
            description=(
                "Deprecated: use Model Switch instead. It takes the type of whatever is "
                "connected and skips the branch it does not select. This node passes one of two "
                "CLIP Vision models on, chosen by a boolean: clip_vision_a when the boolean is "
                "true, clip_vision_b when it is false."
            ),
            inputs=[
                io.ClipVision.Input(
                    "clip_vision_a",
                    lazy=True,
                    tooltip="The CLIP Vision model sent on when boolean is true.",
                ),
                io.ClipVision.Input(
                    "clip_vision_b",
                    lazy=True,
                    tooltip="The CLIP Vision model sent on when boolean is false.",
                ),
                io.Boolean.Input(
                    "boolean",
                    default=True,
                    tooltip=(
                        "Which input passes; BOOLEAN. true = clip_vision_a, false = "
                        "clip_vision_b. Toggle it, or wire it from Logic Boolean."
                    ),
                ),
            ],
            outputs=[
                io.ClipVision.Output(
                    tooltip="Whichever of the two CLIP Vision models was selected.",
                ),
            ],
            is_deprecated=True,
        )

    @classmethod
    def check_lazy_status(cls, boolean=True, **values) -> list[str]:
        """The input the boolean selects while it is still to be worked out; the other is never run."""
        chosen = "clip_vision_a" if boolean else "clip_vision_b"
        return [chosen] if values.get(chosen) is None else []

    @classmethod
    def execute(cls, clip_vision_a, clip_vision_b, boolean=True) -> io.NodeOutput:
        if boolean:
            return io.NodeOutput(clip_vision_a)
        return io.NodeOutput(clip_vision_b)

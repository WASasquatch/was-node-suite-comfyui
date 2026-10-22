"""LoRA application, plus the LoRA's name as a string."""

from __future__ import annotations

import os

from comfy_api.latest import io

from ....modules.compat.sockets import require_input

REQUIRES = "loaders"

#: First option of the ``lora_name`` combo, and its default. Selects no LoRA at all.
NONE_OPTION = "None"

#: Menu label and title bar text, and how a ``require_input`` message names the node.
DISPLAY_NAME = "Lora Loader (Advanced)"

MODEL_TOOLTIP = "The diffusion model the LoRA is applied to."

CLIP_TOOLTIP = (
    "The text encoder the LoRA is applied to. Most LoRAs adjust both halves, so this is "
    "wired from the same checkpoint as model."
)

LORA_NAME_TOOLTIP = (
    "The LoRA file in models/loras to apply. 'None' passes the model and CLIP through "
    "unchanged."
)

STRENGTH_MODEL_TOOLTIP = (
    "How strongly the LoRA modifies the diffusion model. 1.0 is the strength it was trained "
    "at, 0.0 leaves the model alone, and a negative value applies it in reverse."
)

STRENGTH_CLIP_TOOLTIP = (
    "How strongly the LoRA modifies the text encoder. 1.0 is the strength it was trained at; "
    "lowering it keeps the LoRA's look while letting the prompt matter more."
)

MODEL_OUT_TOOLTIP = "The model with the LoRA applied."

CLIP_OUT_TOOLTIP = "The text encoder with the LoRA applied."

NAME_OUT_TOOLTIP = (
    "The LoRA's file name without its folder or extension, for captions, file names and log "
    "lines."
)

#: ``(path, state dict)`` of the LoRA loaded last, or None. Held at module scope.
_loaded_lora: tuple[str, dict] | None = None


def lora_names() -> list[str]:
    """``"None"`` followed by the LoRA files this install offers."""
    import folder_paths

    return [NONE_OPTION, *folder_paths.get_filename_list("loras")]


def lora_state_dict(lora_path: str) -> dict:
    """The LoRA's tensors, reusing the last file loaded when it is the same one.

    Args:
        lora_path: Absolute path of the LoRA file.

    Returns:
        The loaded state dict.
    """
    global _loaded_lora

    import comfy.utils

    if _loaded_lora is not None and _loaded_lora[0] == lora_path:
        return _loaded_lora[1]
    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
    _loaded_lora = (lora_path, lora)
    return lora


class LoraLoader(io.ComfyNode):
    """Apply a LoRA to a model and a CLIP, and report the LoRA's name."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Lora Loader",
            display_name=DISPLAY_NAME,
            search_aliases=["Lora Loader", "Load Lora", "lora", "apply lora", "lycoris"],
            category="WAS Suite/Loaders",
            description=(
                "Deprecated: use ComfyUI's Load LoRA instead. Applies a LoRA to a model and "
                "a CLIP at separate strengths, and returns the LoRA's file name as a "
                "string. That name is already on the lora_name widget."
            ),
            inputs=[
                io.Model.Input("model", tooltip=MODEL_TOOLTIP),
                io.Clip.Input("clip", tooltip=CLIP_TOOLTIP),
                io.Combo.Input("lora_name", options=lora_names(), tooltip=LORA_NAME_TOOLTIP),
                io.Float.Input(
                    "strength_model",
                    default=1.0,
                    min=-10.0,
                    max=10.0,
                    step=0.01,
                    tooltip=STRENGTH_MODEL_TOOLTIP,
                ),
                io.Float.Input(
                    "strength_clip",
                    default=1.0,
                    min=-10.0,
                    max=10.0,
                    step=0.01,
                    tooltip=STRENGTH_CLIP_TOOLTIP,
                ),
            ],
            outputs=[
                io.Model.Output(display_name="MODEL", tooltip=MODEL_OUT_TOOLTIP),
                io.Clip.Output(display_name="CLIP", tooltip=CLIP_OUT_TOOLTIP),
                io.String.Output(display_name="NAME_STRING", tooltip=NAME_OUT_TOOLTIP),
            ],
            is_deprecated=True,
        )

    @classmethod
    def execute(cls, model, clip, lora_name, strength_model, strength_clip) -> io.NodeOutput:
        """Apply the LoRA.

        Raises:
            ValueError: Nothing is connected to the model or clip input.
        """
        import folder_paths

        from ....modules.lora import fit

        require_input(model, DISPLAY_NAME, "model", "model", "checkpoint loader", "MODEL")
        require_input(clip, DISPLAY_NAME, "clip", "text encoder", "checkpoint loader", "CLIP")

        name = os.path.splitext(os.path.basename(lora_name))[0]
        if lora_name == NONE_OPTION or (strength_model == 0 and strength_clip == 0):
            return io.NodeOutput(model, clip, name)

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        model_lora, clip_lora = fit.load_lora_for_models(
            model, clip, lora_state_dict(lora_path), strength_model, strength_clip, name
        )
        return io.NodeOutput(model_lora, clip_lora, name)

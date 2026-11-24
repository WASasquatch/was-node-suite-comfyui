"""Put H3-World's LoRA on MiniMax H3 and bind each action sentence to its own video latent."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.interface import run_result
from ...modules.io import picker
from ...modules.latent import h3_world
from ...modules.model import pika_h3
from ..legacy.loaders.lora_loader import NONE_OPTION, lora_names, lora_state_dict

logger = log.get_logger("nodes.h3_world")

NODE_NAME = "H3 World"


def default_lora(names: list[str]) -> str:
    """The first H3-World file in a LoRA menu, else None."""
    found = [name for name in names
             if "h3-world" in name.lower().replace("\\", "/") or "h3_world" in name.lower()]
    return (found or [NONE_OPTION])[0]


class H3World(io.ComfyNode):
    """Load H3-World's LoRA and route action sentences to their latents."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        names = lora_names()
        return io.Schema(
            node_id="WASH3World",
            display_name=NODE_NAME,
            search_aliases=[
                "WASH3World", NODE_NAME, "h3-world", "world model", "keyboard control",
                "action control", "game camera", "minimax h3",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Turn MiniMax H3 into a world model a keyboard drives: loads the H3-World LoRA and "
                "binds each action sentence from H3 World Actions to its own video latent while "
                "sampling. Feed the model to the sampler that takes the Actions conditioning."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="MiniMax H3 FL2VA from Load Diffusion Model, as minimax_h3_fl2va_int8_convrot.",
                ),
                picker.path_input(io.Combo.Input(
                    "lora_name",
                    options=names,
                    default=default_lora(names),
                    tooltip=(
                        "H3-World LoRA in models/loras, as `H3/H3-World/step-10000`. `None` = route "
                        "only, for a model that already carries it."
                    ),
                ), model=True),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=-2.0,
                    max=2.0,
                    step=0.01,
                    tooltip="LoRA scale: `1.0` = as trained; `0` = routing alone, which the base model does not follow.",
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="model",
                    tooltip="The model with the LoRA, routing the sentences of any H3 World Actions conditioning it samples.",
                ),
            ],
        )

    @classmethod
    def execute(cls, model, lora_name=NONE_OPTION, strength=1.0) -> io.NodeOutput:
        """Load the LoRA and add the routing.

        Raises:
            ValueError: The model is not MiniMax H3 video, or the LoRA does not fit it.
            PathNotAllowed: A wired LoRA path lies outside the LoRA folders.
        """
        import comfy.lora
        import comfy.lora_convert
        import comfy.model_base
        import folder_paths

        if not pika_h3.is_minimax_h3(model) or isinstance(model.model, comfy.model_base.MiniMaxH3Image):
            raise ValueError(
                f"{NODE_NAME} takes a MiniMax H3 video model from Load Diffusion Model, and this one is "
                f"{type(getattr(model, 'model', None)).__name__}.")
        lora_name = picker.model_name(lora_name, ("loras",))
        pairs = attached = 0
        if lora_name and lora_name != NONE_OPTION:
            lora = comfy.lora_convert.convert_lora(
                lora_state_dict(folder_paths.get_full_path_or_raise("loras", lora_name)))
            pairs = sum(1 for key in lora if ".lora_A." in key or ".lora_down." in key)
            key_map = comfy.lora.model_lora_keys_unet(model.model, {})
            loaded = comfy.lora.load_lora(lora, key_map)
            model = model.clone()
            attached = len(set(model.add_patches(loaded, float(strength))))
            if attached < pairs:
                raise ValueError(
                    f"{NODE_NAME} attached {attached} of the {pairs} LoRA pairs in {lora_name}. Pick the "
                    f"H3-World LoRA, step-10000.safetensors from DANNY621/H3-World, on an FL2VA model.")
        patched = h3_world.patched(model)
        run_result.publish(
            summary=(f"{lora_name.replace(chr(92), '/').rsplit('/', 1)[-1]} at {float(strength):g}, "
                     f"sentences routed" if attached else "sentences routed, no LoRA loaded"),
            counts={"pairs": attached},
            facts={"routing": "each sentence read by its own latent"},
        )
        logger.info("%s: %d LoRA pairs at %s, routing on", NODE_NAME, attached, strength)
        return io.NodeOutput(patched)

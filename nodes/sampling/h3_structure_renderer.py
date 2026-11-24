"""Turn MiniMax H3 into SolarRenderer: its two LoRAs, and the soundtrack held at silence."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.interface import run_result
from ...modules.io import picker
from ...modules.model import h3_structure, pika_h3
from ..legacy.loaders.lora_loader import NONE_OPTION, lora_names

logger = log.get_logger("nodes.h3_structure_renderer")

NODE_NAME = "H3 Structure Renderer"


def default_lora(names: list[str], word: str) -> str:
    """The first SolarRenderer clip LoRA in a menu whose name holds ``word``, else None."""
    found = [name for name in names
             if "solarrenderer" in name.lower() and word in name.lower() and "causal" not in name.lower()]
    return (found or [NONE_OPTION])[0]


def _short(name: str) -> str:
    """A LoRA name without its folders."""
    return name.replace("\\", "/").rsplit("/", 1)[-1]


class H3StructureRenderer(io.ComfyNode):
    """Load SolarRenderer's LoRAs onto MiniMax H3 and hold its soundtrack at silence."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        names = lora_names()
        return io.Schema(
            node_id="WASH3StructureRenderer",
            display_name=NODE_NAME,
            search_aliases=[
                "WASH3StructureRenderer", NODE_NAME,
                "solarrenderer",
                "solar renderer",
                "structure to video",
                "grey box render",
                "render a block-out",
                "minimax h3",
                "ref2va",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Make MiniMax H3 Ref2VA render a structure video, such as a grey box, depth or skeletons, "
                "into finished footage in 4 steps, with SolarRenderer's control and four-step LoRAs. Sample "
                "H3 Structure Video's outputs with KSampler: lcm, simple, 4 steps, cfg 1."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip=(
                        "MiniMax H3 Ref2VA from Load Diffusion Model, as minimax_h3_ref2va_pruned_int8_convrot; "
                        "bf16 keeps the LoRAs exact."
                    ),
                ),
                io.Vae.Input(
                    "audio_vae",
                    tooltip=(
                        "The H3 audio VAE, as Load VAE set to minimax_h3_audio_vae_fp32. It encodes the "
                        "silence the soundtrack is held at, at the clip's own length."
                    ),
                ),
                picker.path_input(io.Combo.Input(
                    "control_lora",
                    options=names,
                    default=default_lora(names, "control"),
                    tooltip="solarrenderer_control_lora in models/loras: follows the structure video and the first frame.",
                ), model=True),
                picker.path_input(io.Combo.Input(
                    "steps_lora",
                    options=names,
                    default=default_lora(names, "4step"),
                    tooltip="solarrenderer_4step_lora in models/loras: renders in 4 steps at cfg 1.",
                ), model=True),
                io.Float.Input(
                    "control_strength",
                    default=1.0,
                    min=-2.0,
                    max=2.0,
                    step=0.01,
                    advanced=True,
                    tooltip="Control LoRA scale: 1.0 = as trained; lower follows the structure less.",
                ),
                io.Float.Input(
                    "steps_strength",
                    default=1.0,
                    min=-2.0,
                    max=2.0,
                    step=0.01,
                    advanced=True,
                    tooltip="Four-step LoRA scale: 1.0 = as trained.",
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="model",
                    tooltip=(
                        "MiniMax H3 with both LoRAs, its soundtrack held at encoded silence. Into KSampler "
                        "with H3 Structure Video's positive, negative and latent."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, model, audio_vae, control_lora=NONE_OPTION, steps_lora=NONE_OPTION,
                control_strength=1.0, steps_strength=1.0) -> io.NodeOutput:
        """Load both LoRAs and hold the soundtrack.

        Raises:
            ValueError: The model is not MiniMax H3, a LoRA is missing or does not fit.
            PathNotAllowed: A wired LoRA path lies outside the LoRA folders.
        """
        import folder_paths

        if not pika_h3.is_minimax_h3(model):
            raise ValueError(
                f"{NODE_NAME} takes MiniMax H3 Ref2VA from Load Diffusion Model, and this one is "
                f"{type(getattr(model, 'model', None)).__name__}.")
        picked = []
        for label, value, strength in (("steps_lora", steps_lora, steps_strength),
                                       ("control_lora", control_lora, control_strength)):
            name = picker.model_name(value, ("loras",))
            if not name or name == NONE_OPTION:
                raise ValueError(
                    f"{NODE_NAME} needs {label}: download solarrenderer_control_lora and solarrenderer_4step_lora "
                    f"from huggingface.co/Enderfga/SolarRenderer into ComfyUI/models/loras.")
            lora, alpha = h3_structure.read_lora(folder_paths.get_full_path_or_raise("loras", name))
            found = h3_structure.patches(model, lora, alpha, _short(name))
            picked.append((_short(name), found, float(strength)))
        patched, reached = h3_structure.apply(model, picked)
        patched = h3_structure.hold_silence(patched, audio_vae)
        run_result.publish(
            summary=f"{picked[1][0]} at {picked[1][2]:g} and {picked[0][0]} at {picked[0][2]:g}",
            counts={"control weights": reached[1], "four-step weights": reached[0]},
            facts={
                "sound": "held at encoded silence",
                "sample with": "KSampler lcm, simple, 4 steps, cfg 1",
            },
        )
        logger.info("%s: %s (%d weights) and %s (%d weights)", NODE_NAME, picked[1][0], reached[1],
                    picked[0][0], reached[0])
        return io.NodeOutput(patched)

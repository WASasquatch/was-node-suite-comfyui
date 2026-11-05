"""Put a LoRA on MiniMax H3's sound tokens alone."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.interface import run_result
from ...modules.io import picker
from ...modules.model import h3_sound_lora, pika_h3
from ..legacy.loaders.lora_loader import NONE_OPTION, lora_names, lora_state_dict

logger = log.get_logger("nodes.h3_sound_lora")

NODE_NAME = "H3 Sound LoRA"

#: DMAD's audio shift.
DMAD_AUDIO_SHIFT = 2.0


def default_lora(names: list[str]) -> str:
    """The first DMAD file in a LoRA menu, else None."""
    dmad = [name for name in names if "dmad" in name.lower() and "pdmd" not in name.lower()]
    return (dmad or [NONE_OPTION])[0]


def _short(name: str) -> str:
    """A LoRA name without its folders."""
    return name.replace("\\", "/").rsplit("/", 1)[-1]


class H3SoundLoRA(io.ComfyNode):
    """Give MiniMax H3's sound tokens a LoRA of their own."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        names = lora_names()
        return io.Schema(
            node_id="WASH3SoundLoRA",
            display_name=NODE_NAME,
            search_aliases=[
                "WASH3SoundLoRA", NODE_NAME,
                "sound lora",
                "audio lora",
                "dmad",
                "dmad audio",
                "pika dmad",
                "minimax h3",
                "audio rows",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Put a LoRA on MiniMax H3's sound tokens alone, so the picture keeps the model's own "
                "LoRAs, such as Pika H3 Flex, while the sound takes another, such as DMAD. Sets the "
                "audio shift that LoRA was trained at. Chain a second one on add to mix two on the sound."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip=(
                        "MiniMax H3 after every LoRA loader, Pika H3 Flex LoRA included; the picture "
                        "keeps those LoRAs."
                    ),
                ),
                picker.path_input(io.Combo.Input(
                    "lora_name",
                    options=names,
                    default=default_lora(names),
                    tooltip=(
                        "LoRA in models/loras for the sound tokens, as "
                        "`dmad_minimax_h3_4step_full_critic_comfyui`. Its text refiner pairs are not used."
                    ),
                ), model=True),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=-2.0,
                    max=2.0,
                    step=0.01,
                    tooltip="Sound LoRA scale: `1.0` = as trained; `0` = none, the model's LoRAs still dropped on replace.",
                ),
                io.Combo.Input(
                    "mode",
                    options=list(h3_sound_lora.MODES),
                    default=h3_sound_lora.MODES[0],
                    tooltip=(
                        "`replace` = the sound tokens drop the model's other LoRAs and take this one; "
                        "`add` = this one on top of them, as a second sound LoRA in a chain."
                    ),
                ),
                io.Float.Input(
                    "audio_shift",
                    default=DMAD_AUDIO_SHIFT,
                    min=0.5,
                    max=24.0,
                    step=0.1,
                    tooltip=(
                        "Audio flow shift the sound samples at: `2` = DMAD; `3` = Pika H3 Flex and "
                        "Ref2VA. The video shift is kept."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="model",
                    tooltip=(
                        "The model with its sound tokens on this LoRA at this audio shift. H3 Stream "
                        "Sampler steps the sound on that shift's sigmas."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, model, lora_name=NONE_OPTION, strength=1.0, mode=h3_sound_lora.MODES[0],
                audio_shift=DMAD_AUDIO_SHIFT) -> io.NodeOutput:
        """Apply the sound LoRA.

        Raises:
            ValueError: The model is not MiniMax H3, the mode is unknown, or the LoRA does not fit.
            PathNotAllowed: A wired LoRA path lies outside the LoRA folders.
        """
        import folder_paths

        if not pika_h3.is_minimax_h3(model):
            raise ValueError(
                f"{NODE_NAME} takes a MiniMax H3 model from Load Diffusion Model, and this one is "
                f"{type(getattr(model, 'model', None)).__name__}")
        if mode not in h3_sound_lora.MODES:
            raise ValueError(f"{NODE_NAME} has no mode {mode!r}. Choose one of: {', '.join(h3_sound_lora.MODES)}.")
        lora_name = picker.model_name(lora_name, ("loras",))
        if not lora_name or lora_name == NONE_OPTION:
            raise ValueError(f"{NODE_NAME} needs a LoRA for the sound tokens, as a DMAD 4-step file.")
        lora = lora_state_dict(folder_paths.get_full_path_or_raise("loras", lora_name))
        name = _short(lora_name)
        pairs, skipped = h3_sound_lora.pairs_from_lora(model, lora, float(strength), name)
        entry = h3_sound_lora.Entry(name, float(strength), mode, pairs, skipped)
        patched, stack = h3_sound_lora.apply(model, entry, float(audio_shift))
        run_result.publish(
            summary=f"{name} on the sound at {float(strength):g}, {mode}",
            facts={
                "sound": "; ".join(f"{e.name} {e.strength:g} ({e.mode})" for e in stack.entries),
                "picture": "the model's LoRAs" + ("" if stack.drop else ", also on the sound"),
                "shifts": f"video {stack.video_shift:g}, audio {stack.audio_shift:g}",
            },
        )
        logger.info("%s: %s at %s (%s), %d projections, audio shift %s", NODE_NAME, name, strength, mode,
                    len(pairs), audio_shift)
        return io.NodeOutput(patched)

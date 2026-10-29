"""Apply a Pika H3 Flex adapter to MiniMax H3 at the release's shifts."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.interface import run_result
from ...modules.io import picker
from ...modules.model import pika_h3
from ..legacy.loaders.lora_loader import NONE_OPTION, lora_names, lora_state_dict

logger = log.get_logger("nodes.pika_h3_flex_lora")

NODE_NAME = "Pika H3 Flex LoRA"

#: The adapter's modes: computed beside each layer, or added into the weights.
MODES = ("bypass", "merge")


def default_adapter(names: list[str]) -> str:
    """The first Pika H3 Flex fl2va file in a LoRA menu, else any Pika H3 Flex file, else None."""
    pika = [name for name in names if "pika_h3_flex" in name.lower()]
    fl2va = [name for name in pika if "fl2va" in name.lower()]
    return (fl2va or pika or [NONE_OPTION])[0]


class PikaH3FlexLoRA(io.ComfyNode):
    """Turn MiniMax H3 into Pika H3 Flex."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        names = lora_names()
        return io.Schema(
            node_id="WASPikaH3FlexLoRA",
            display_name=NODE_NAME,
            search_aliases=[
                "WASPikaH3FlexLoRA", NODE_NAME,
                "pika",
                "pika h3 flex",
                "h3 flex",
                "minimax h3",
                "fl2va",
                "ref2va",
                "bypass lora",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Turn a MiniMax H3 fl2va or ref2va model into Pika H3 Flex: the matching Pika "
                "adapter, the release's video and audio shifts of 12 and 3, and its fp32 AdaLN. "
                "Sample it on Pika H3 Flex Sampling's sigmas."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip=(
                        "MiniMax H3 from Load Diffusion Model: fl2va for the fl2va adapter, ref2va "
                        "for the ref2va adapter."
                    ),
                ),
                picker.path_input(io.Combo.Input(
                    "lora_name",
                    options=names,
                    default=default_adapter(names),
                    tooltip=(
                        "Adapter in models/loras: `pika_h3_flex_fl2va` for text and frames, "
                        "`pika_h3_flex_ref2va` for references; `None` = shifts and AdaLN only."
                    ),
                ), model=True),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=-2.0,
                    max=2.0,
                    step=0.01,
                    tooltip="Adapter scale: `1.0` = the release; `0` = the base model.",
                ),
                io.Combo.Input(
                    "mode",
                    options=list(MODES),
                    default=MODES[0],
                    tooltip=(
                        "`bypass` = adapter computed beside each layer, base weights untouched, "
                        "exact on int8 and w6a8 models; `merge` = adapter added into the weights."
                    ),
                ),
                io.Boolean.Input(
                    "adaln_fp32",
                    default=True,
                    tooltip=(
                        "`true` = AdaLN SiLU on the fp32 time embedding, as the release; `false` "
                        "= on the embedding in the compute dtype, as core. Pruned models run fp32."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="model",
                    tooltip="Pika H3 Flex at video shift 12 and audio shift 3.",
                ),
            ],
        )

    @classmethod
    def execute(cls, model, lora_name=NONE_OPTION, strength=1.0, mode=MODES[0],
                adaln_fp32=True) -> io.NodeOutput:
        """Apply the adapter.

        Raises:
            ValueError: The model is not MiniMax H3, or the mode is unknown.
            PathNotAllowed: A wired adapter path lies outside the LoRA folders.
        """
        import folder_paths

        if not pika_h3.is_minimax_h3(model):
            raise ValueError(
                f"{NODE_NAME} takes a MiniMax H3 model from Load Diffusion Model, fl2va or ref2va, "
                f"and this one is {type(getattr(model, 'model', None)).__name__}")
        if mode not in MODES:
            raise ValueError(f"{NODE_NAME} has no mode {mode!r}. Choose one of: {', '.join(MODES)}.")
        lora_name = picker.model_name(lora_name, ("loras",))
        lora = None
        if lora_name and lora_name != NONE_OPTION and float(strength) != 0:
            lora = lora_state_dict(folder_paths.get_full_path_or_raise("loras", lora_name))
        patched, curves = pika_h3.apply(model, lora, float(strength), mode, bool(adaln_fp32))
        adapter = "none" if lora is None else lora_name.replace("\\", "/").rsplit("/", 1)[-1]
        run_result.publish(
            summary=f"{adapter} at {float(strength):g}, {mode}" if lora is not None else "no adapter",
            facts={
                "adapter": adapter,
                "shifts": f"video {pika_h3.VIDEO_SHIFT:g}, audio {pika_h3.AUDIO_SHIFT:g}",
                "AdaLN": ("fp32 table (pruned model)" if curves
                          else "fp32" if adaln_fp32 else "compute dtype"),
            },
        )
        logger.info("%s: %s at %s (%s)", NODE_NAME, adapter, strength, mode)
        return io.NodeOutput(patched)

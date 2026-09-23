"""Apply a LoRA with its blocks scaled by position, and save exactly what was applied."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ....modules.compat.sockets import require_input
from ....modules.compat.types import DICT
from ....modules.log import get_logger

REQUIRES = "extras"

logger = get_logger("nodes.extras.lora")

#: Which half of each LoRA pair the block scaling multiplies.
SCALE_TARGETS = ["up_only", "down_only", "both"]

#: Model families whose block naming the scaling knows, plus automatic detection.
BLOCK_PRESETS = ["auto", "wan", "qwen", "flux", "zimg-turbo", "sd", "sdxl", "generic"]

#: Directory under ComfyUI's output directory the reweighted copies are written to.
OUTPUT_SUBDIRECTORY = "loras"


def lora_names() -> list[str]:
    """The LoRA files this install offers."""
    import folder_paths

    return folder_paths.get_filename_list("loras")


class ApplyReweightedLoRA(io.ComfyNode):
    """Scale a LoRA's blocks by where they sit in the model, then apply it."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASNSApplyReweightedLoRA",
            display_name="Apply Reweighted LoRA",
            search_aliases=[
                "WASNSApplyReweightedLoRA",
                "WAS Apply Reweighted LoRA",
                "WAS Extras",
                "lora block weight",
                "reweight lora",
            ],
            category="WAS Suite/LoRA",
            description=(
                "Load a LoRA, scale its blocks by where they sit in the model, front, "
                "middle, back and the very last block, and apply the result to a model and "
                "clip. The reweighted LoRA is also saved under output/loras so a setting "
                "that works can be reused."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    tooltip="The model the reweighted LoRA is applied to.",
                ),
                io.Clip.Input(
                    "clip",
                    tooltip=(
                        "The clip the LoRA's text-encoder half is applied to. A LoRA with no "
                        "text-encoder tensors leaves it untouched."
                    ),
                ),
                io.Combo.Input(
                    "lora_name",
                    options=lora_names(),
                    tooltip=(
                        "The LoRA file to reweight, from your LoRA folder. It is read from "
                        "disk on every run, so the original file is never modified."
                    ),
                ),
                io.Float.Input(
                    "strength_model",
                    default=0.8,
                    min=-5.0,
                    max=5.0,
                    step=0.01,
                    tooltip=(
                        "How strongly the reweighted LoRA is applied to the model, before "
                        "any block scaling. 1.0 is full strength; a negative value pushes "
                        "away from what the LoRA learned."
                    ),
                ),
                io.Float.Input(
                    "strength_clip",
                    default=0.8,
                    min=-5.0,
                    max=5.0,
                    step=0.01,
                    tooltip=(
                        "The same for the clip. Lowering it while leaving strength_model "
                        "alone keeps the LoRA's look without its trigger words dominating "
                        "the prompt."
                    ),
                ),
                io.Float.Input(
                    "global_scale",
                    default=1.0,
                    min=-5.0,
                    max=5.0,
                    step=0.01,
                    tooltip=(
                        "Multiplier applied to every block before the three below. 1.0 "
                        "changes nothing; use it to turn the whole reweighting up or down "
                        "once the balance between the thirds is right."
                    ),
                ),
                io.Float.Input(
                    "front_scale",
                    default=1.0,
                    min=-5.0,
                    max=5.0,
                    step=0.01,
                    tooltip=(
                        "Extra multiplier for the first third of the blocks, which carry "
                        "composition and overall shape. Lower it to keep a LoRA's style "
                        "while letting the prompt decide the layout."
                    ),
                ),
                io.Float.Input(
                    "mid_scale",
                    default=1.0,
                    min=-5.0,
                    max=5.0,
                    step=0.01,
                    tooltip=(
                        "Extra multiplier for the middle third, which carries subject and "
                        "structure. This is the third to lower when a character LoRA is "
                        "overriding the face you asked for."
                    ),
                ),
                io.Float.Input(
                    "back_scale",
                    default=1.0,
                    min=-5.0,
                    max=5.0,
                    step=0.01,
                    tooltip=(
                        "Extra multiplier for the last third, which carries detail, texture "
                        "and surface style. Raise it to keep a LoRA's look while its subject "
                        "influence is turned down."
                    ),
                ),
                io.Float.Input(
                    "last_block_scale",
                    default=1.0,
                    min=-5.0,
                    max=5.0,
                    step=0.01,
                    tooltip=(
                        "A further multiplier for the final block alone, on top of its "
                        "third's. That block sits closest to the output, so small changes "
                        "here show up strongly in fine detail."
                    ),
                ),
                io.Combo.Input(
                    "scale_target",
                    options=SCALE_TARGETS,
                    default="up_only",
                    tooltip=(
                        "Which half of each LoRA pair is scaled. `up_only` is the usual "
                        "choice and scales the result linearly. `both` scales the two halves "
                        "and so squares the effect, which is much stronger for the same "
                        "numbers. `down_only` is there for comparison."
                    ),
                ),
                io.Combo.Input(
                    "block_preset",
                    options=BLOCK_PRESETS,
                    default="auto",
                    tooltip=(
                        "Which model family's block naming is read to find each block's "
                        "number. `auto` works it out from the LoRA's own keys and is right "
                        "almost always; name the family if the stats output reports 0 blocks "
                        "detected."
                    ),
                ),
                io.Boolean.Input(
                    "filter_by_block_range",
                    default=True,
                    tooltip=(
                        "Drop tensors for blocks the connected model does not have. This is "
                        "what lets a LoRA trained on a larger version of a model be applied "
                        "to a smaller one instead of failing."
                    ),
                ),
                io.Boolean.Input(
                    "save_reweighted",
                    default=True,
                    tooltip=(
                        "Write the reweighted LoRA to output/loras. Switch it off while "
                        "hunting for the right numbers, then on for the run worth keeping."
                    ),
                ),
                io.String.Input(
                    "output_filename",
                    default="",
                    tooltip=(
                        "Name for the saved copy. Left empty, a name is built from the "
                        "source file and every scale, such as "
                        "'style.reweighted.up_only.g1.00.f1.0.m1.0.b1.0.L1.0.safetensors', "
                        "so two settings never overwrite each other."
                    ),
                ),
                io.Boolean.Input(
                    "verify_roundtrip",
                    default=True,
                    tooltip=(
                        "Read the saved file back and compare it tensor by tensor with what "
                        "was applied, reporting the answer in the stats output. Costs a "
                        "second read of the file; it is what proves the saved copy behaves "
                        "the same as this run."
                    ),
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="model",
                    tooltip="The model with the reweighted LoRA applied.",
                ),
                io.Clip.Output(
                    display_name="clip",
                    tooltip="The clip with the reweighted LoRA applied.",
                ),
                DICT.Output(
                    display_name="stats",
                    tooltip=(
                        "What the run did: which naming scheme was detected, how many blocks "
                        "were found, how many tensors were scaled, dropped and kept, where "
                        "the copy was saved with its SHA-256, and whether the round-trip "
                        "check passed. Feed it to a debug node to see why a reweighting had "
                        "no effect."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        model,
        clip,
        lora_name,
        strength_model,
        strength_clip,
        global_scale,
        front_scale,
        mid_scale,
        back_scale,
        last_block_scale,
        scale_target,
        block_preset,
        filter_by_block_range,
        save_reweighted,
        output_filename,
        verify_roundtrip,
    ) -> io.NodeOutput:
        """Scale the LoRA's blocks and apply it.

        Raises:
            ValueError: Nothing is connected to the model or clip input.
        """
        from pathlib import Path

        import folder_paths
        from comfy import sd as comfy_sd
        from comfy.utils import load_torch_file
        from safetensors.torch import load_file, save_file

        from ....modules.lora import reweight
        from ....modules.util import sandbox
        from ....modules.util.hashing import get_sha256

        require_input(
            model, "Apply Reweighted LoRA", "model", "model", "checkpoint loader", "MODEL"
        )
        require_input(
            clip, "Apply Reweighted LoRA", "clip", "text encoder", "checkpoint loader", "CLIP"
        )

        source = folder_paths.get_full_path_or_raise("loras", lora_name)
        state = load_torch_file(str(source), safe_load=True)

        preset = (block_preset or "auto").lower().replace("_", "-")
        if preset == "auto":
            preset = reweight.infer_preset(state.keys())

        model_blocks = reweight.detect_total_blocks_from_model(model)
        lora_blocks = reweight.detect_total_blocks_from_lora(state, preset)
        total_blocks = model_blocks if model_blocks > 0 else lora_blocks

        scaled, counts = reweight.reweight_state_dict(
            state,
            total_blocks,
            global_scale,
            front_scale,
            mid_scale,
            back_scale,
            last_block_scale,
            scale_target,
            filter_by_block_range,
            filter_cutoff_blocks=model_blocks,
            preset=preset,
        )

        patched_model, patched_clip = comfy_sd.load_lora_for_models(
            model.clone(), clip.clone(), scaled, strength_model, strength_clip
        )

        saved_path = ""
        saved_digest = ""
        roundtrip_equal = None
        if save_reweighted:
            name = output_filename.strip() or reweight.suggest_filename(
                Path(source).name,
                scale_target,
                global_scale,
                front_scale,
                mid_scale,
                back_scale,
                last_block_scale,
            )
            directory = Path(folder_paths.get_output_directory()) / OUTPUT_SUBDIRECTORY
            target = sandbox.resolve_write_file(directory, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            save_file(
                scaled,
                str(target),
                metadata={
                    "was_reweighted": "1",
                    "base_file": Path(source).name,
                    "scale_target": scale_target,
                    "global_scale": f"{global_scale}",
                    "front_scale": f"{front_scale}",
                    "mid_scale": f"{mid_scale}",
                    "back_scale": f"{back_scale}",
                    "last_block_scale": f"{last_block_scale}",
                    "strength_model": f"{strength_model}",
                    "strength_clip": f"{strength_clip}",
                },
            )
            saved_path = str(target)
            saved_digest = get_sha256(target)
            logger.info("saved the reweighted LoRA to %s", target)

            if verify_roundtrip:
                roundtrip_equal = cls._matches(scaled, load_file(str(target)))

        stats = {
            "source": str(source),
            "block_preset": preset,
            "total_blocks_detected": total_blocks,
            "model_blocks": model_blocks,
            "lora_blocks": lora_blocks,
            "changed": counts["changed"],
            "dropped_by_block_filter": counts["dropped"],
            "kept": counts["kept"],
            "strength_model": strength_model,
            "strength_clip": strength_clip,
            "global_scale": global_scale,
            "front_scale": front_scale,
            "mid_scale": mid_scale,
            "back_scale": back_scale,
            "last_block_scale": last_block_scale,
            "scale_target": scale_target,
            "saved_path": saved_path,
            "saved_sha256": saved_digest,
            "roundtrip_equal": roundtrip_equal,
        }
        return io.NodeOutput(patched_model, patched_clip, stats)

    @classmethod
    def _matches(cls, applied, reloaded) -> bool:
        """Compare what was applied with what came back off disk.

        Args:
            applied: The state dict that was applied to the model.
            reloaded: The same file, read back after saving.

        Returns:
            True when every key, dtype, shape and value is identical.
        """
        if set(applied.keys()) != set(reloaded.keys()):
            return False
        for key, value in applied.items():
            other = reloaded[key]
            if isinstance(value, torch.Tensor) and isinstance(other, torch.Tensor):
                if value.dtype != other.dtype or value.shape != other.shape:
                    return False
                if not torch.equal(value, other):
                    return False
            elif value != other:
                return False
        return True

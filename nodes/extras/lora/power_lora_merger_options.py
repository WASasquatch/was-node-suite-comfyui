"""The advanced settings the Power LoRA Merger runs with."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.types import WAS_LORA_MERGE_OPTIONS

REQUIRES = "extras"

#: Precisions the merged tensors can be written in.
OUTPUT_DTYPES = ["fp16", "fp32", "bf16"]

#: Precisions the merge arithmetic can run in, plus the automatic choice.
COMPUTE_DTYPES = ["auto", "bf16", "fp16", "fp32"]

#: How block-mix combines the modules it has routed.
BLOCK_MIX_METHODS = ["svd", "stack"]

#: Model families block-mix knows the block naming of, plus automatic detection.
BLOCK_MIX_PRESETS = ["auto", "zimg-turbo", "flux", "wan", "qwen", "sd", "sdxl", "generic"]


class PowerLoraMergerOptions(io.ComfyNode):
    """Collect the Power LoRA Merger's advanced settings onto one wire."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASNSPowerLoraMergerOptions",
            display_name="Power LoRA Merger Options",
            search_aliases=[
                "WASNSPowerLoraMergerOptions",
                "WAS Power LoRA Merger Options",
                "WAS Extras",
                "lora merge options",
                "merge rank",
            ],
            category="WAS Suite/LoRA",
            description=(
                "Advanced settings for the Power LoRA Merger: how far the merged LoRA is "
                "compressed, what precision it is written in, which parts of the model take "
                "part, and the controls belonging to the moe and block-mix modes. Connect it "
                "to the merger's options socket."
            ),
            inputs=[
                io.Int.Input(
                    "rank",
                    default=32,
                    min=0,
                    max=4096,
                    step=1,
                    tooltip=(
                        "How much detail the merged LoRA keeps, for the modes that "
                        "recompress it. Higher keeps more of the sources and makes a bigger "
                        "file: 32 suits most merges, 64 to 128 holds a complicated one "
                        "together. Set it to 0 to let each part of the model choose its own "
                        "rank from auto_rank_threshold instead."
                    ),
                ),
                io.Float.Input(
                    "auto_rank_threshold",
                    default=0.99,
                    min=0.5,
                    max=1.0,
                    step=0.0001,
                    tooltip=(
                        "Only read when rank is 0: how much of each part's strength an "
                        "automatic rank has to keep. 0.99 keeps almost all of it and picks a "
                        "generous rank; 0.9 is far more aggressive and gives a much smaller "
                        "file."
                    ),
                ),
                io.Boolean.Input(
                    "preserve_norm",
                    default=False,
                    tooltip=(
                        "svd mode only. Rescales every merged part back to the average "
                        "strength of the LoRAs it came from, which stops a merge of several "
                        "strong LoRAs coming out overcooked at the strength you normally use."
                    ),
                ),
                io.Boolean.Input(
                    "cap_mult_enable",
                    default=False,
                    tooltip=(
                        "svd mode only. Switches on the ceiling set by cap_mult below. Use it "
                        "when one source LoRA is far stronger than the rest and is taking "
                        "over the merge."
                    ),
                ),
                io.Float.Input(
                    "cap_mult",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=0.01,
                    tooltip=(
                        "The ceiling, as a multiple of the average source strength, applied "
                        "when cap_mult_enable is on. 1.0 holds each part to the average, 1.5 "
                        "allows half again. Ignored while cap_mult_enable is off."
                    ),
                ),
                io.Combo.Input(
                    "dtype",
                    options=OUTPUT_DTYPES,
                    default="bf16",
                    tooltip=(
                        "Precision the merged file is written in. `bf16` is half the size of "
                        "fp32 and is what most LoRAs ship as; `fp32` doubles the file for a "
                        "difference nothing downstream is likely to see; `fp16` matches older "
                        "LoRAs but has less range."
                    ),
                ),
                io.Combo.Input(
                    "compute_dtype",
                    options=COMPUTE_DTYPES,
                    default="auto",
                    tooltip=(
                        "Precision the merge arithmetic runs in, which is separate from what "
                        "is saved. `auto` keeps the sources' own precision, moving to bf16 "
                        "only when they disagree. Force `fp32` if a merge comes out with "
                        "artefacts the sources do not have."
                    ),
                ),
                io.Boolean.Input(
                    "cpu",
                    default=False,
                    tooltip=(
                        "Merge on the processor instead of the graphics card. Slower, but it "
                        "uses no video memory, which is what to reach for when a large merge "
                        "runs out of it."
                    ),
                ),
                io.String.Input(
                    "include_patterns",
                    default="",
                    multiline=True,
                    tooltip=(
                        "Comma-separated text to look for in module names: leave it empty to "
                        "merge everything, or name parts to restrict the merge to them. "
                        "'lora_unet' merges only the image side and leaves the text encoder "
                        "alone."
                    ),
                ),
                io.String.Input(
                    "exclude_patterns",
                    default="",
                    multiline=True,
                    tooltip=(
                        "Comma-separated text that keeps a module out of the merge, applied "
                        "after include_patterns. 'lora_te' drops the text-encoder half, which "
                        "is the usual way to keep one LoRA's trigger words out of a merge."
                    ),
                ),
                io.Float.Input(
                    "moe_temperature",
                    default=1.0,
                    min=1e-6,
                    max=100.0,
                    step=0.01,
                    tooltip=(
                        "moe mode only: how decisively each part of the model picks between "
                        "the LoRAs. Low values such as 0.1 make it choose the strongest one "
                        "almost outright; high values such as 5.0 blend them evenly."
                    ),
                ),
                io.Boolean.Input(
                    "moe_hard",
                    default=False,
                    tooltip=(
                        "moe mode only. Take each part of the model from a single LoRA, the "
                        "strongest one there, instead of blending. Gives a sharper split "
                        "between the sources than any temperature can."
                    ),
                ),
                io.Combo.Input(
                    "block_mix_method",
                    options=BLOCK_MIX_METHODS,
                    default="svd",
                    tooltip=(
                        "block-mix mode only: `svd` recompresses the routed result to the "
                        "rank above and keeps the file small; `stack` keeps both sources' "
                        "ranks exactly, which is more faithful and produces a larger file."
                    ),
                ),
                io.Combo.Input(
                    "block_mix_preset",
                    options=BLOCK_MIX_PRESETS,
                    default="auto",
                    tooltip=(
                        "block-mix mode only: which model family's block naming the routing "
                        "reads. `auto` works it out from the LoRA's own keys and is right "
                        "almost always; name the family when a LoRA uses an unusual naming "
                        "scheme and the routing report shows everything as unclassified."
                    ),
                ),
                io.Boolean.Input(
                    "block_mix_weighted",
                    default=False,
                    tooltip=(
                        "block-mix mode only. Blend the two LoRAs in each part of the model "
                        "by the two ratios below, instead of giving each part to one of them "
                        "outright. Use it when a straight route swings too far towards one "
                        "LoRA."
                    ),
                ),
                io.Float.Input(
                    "block_mix_concept_mix",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Only read while block_mix_weighted is on: how much of LoRA A goes "
                        "into the parts carrying subject and composition, with LoRA B making "
                        "up the rest. 1.0 is all A, 0.0 is all B, 0.5 is even."
                    ),
                ),
                io.Float.Input(
                    "block_mix_style_mix",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Only read while block_mix_weighted is on: the same ratio for the "
                        "parts carrying surface style and texture. Setting concept low and "
                        "style high keeps A's subject in B's look."
                    ),
                ),
            ],
            outputs=[
                WAS_LORA_MERGE_OPTIONS.Output(
                    display_name="options",
                    tooltip=(
                        "The settled settings, for the Power LoRA Merger's options socket."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        rank,
        auto_rank_threshold,
        preserve_norm,
        cap_mult_enable,
        cap_mult,
        dtype,
        compute_dtype,
        cpu,
        include_patterns,
        exclude_patterns,
        moe_temperature,
        moe_hard,
        block_mix_method,
        block_mix_preset,
        block_mix_weighted,
        block_mix_concept_mix,
        block_mix_style_mix,
    ) -> io.NodeOutput:
        return io.NodeOutput(
            {
                "rank": int(rank),
                "auto_rank_threshold": float(auto_rank_threshold),
                "preserve_norm": bool(preserve_norm),
                "cap_mult_enable": bool(cap_mult_enable),
                "cap_mult": float(cap_mult),
                "dtype": str(dtype),
                "compute_dtype": str(compute_dtype),
                "cpu": bool(cpu),
                "include_patterns": str(include_patterns),
                "exclude_patterns": str(exclude_patterns),
                "moe_temperature": float(moe_temperature),
                "moe_hard": bool(moe_hard),
                "block_mix_method": str(block_mix_method),
                "block_mix_preset": str(block_mix_preset),
                "block_mix_weighted": bool(block_mix_weighted),
                "block_mix_concept_mix": float(block_mix_concept_mix),
                "block_mix_style_mix": float(block_mix_style_mix),
            }
        )

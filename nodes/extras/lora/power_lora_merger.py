"""Merge any number of LoRA files into one, save it, and optionally apply it."""

from __future__ import annotations

import json

from comfy_api.latest import io

from ....modules.compat.types import WAS_LORA_MERGE_OPTIONS
from ....modules.log import get_logger
from ...legacy.loaders.lora_loader import NONE_OPTION, lora_names

REQUIRES = "extras"

#: Tooltip on every row's switch. One wording for all of them: the row number is already on
#: the widget.
ROW_ON_TIP = (
    "Whether this row takes part. `true` applies it; `false` mutes it and keeps the file "
    "and the strength it was set to, so neither is typed again to bring it back."
)

#: Tooltip on every row's file widget.
ROW_NAME_TIP = (
    "LoRA file this row uses. `None` leaves the row empty, which counts the same as "
    "switching it off."
)

#: Tooltip on every row's strength widget.
ROW_WEIGHT_TIP = (
    "How strongly this row counts. 1.0 is the LoRA as it was trained, below 1 weakens it, "
    "above 1 pushes it past what it was trained for, and a negative value applies it in "
    "reverse. 0 leaves the row out."
)

logger = get_logger("nodes.extras.lora")

#: The merge modes, in the order the widget offers them. Saved workflows store the chosen
#: option by value, so entries are appended and never reordered or removed.
MERGE_MODES = [
    "svd",
    "rebase",
    "add",
    "add-diff",
    "add-orth",
    "diff-export",
    "moe",
    "obfuscate",
    "block-mix",
]

#: Which modules block-mix takes from LoRA A and which from LoRA B.
BLOCK_MIX_RECIPES = [
    "all_a",
    "all_b",
    "concept_a_style_b",
    "concept_b_style_a",
    "attn_a_ffn_b",
    "attn_b_ffn_a",
    "img_a_txt_b",
    "img_b_txt_a",
]

#: Modes that take a fixed number of LoRAs: the mode, the count, and what to say when the
#: rows do not match it.
EXACT_COUNTS = {
    "rebase": (1, "rebase mode recompresses a single LoRA, so it takes exactly one."),
    "block-mix": (2, "block-mix mode routes between two LoRAs, A and B, so it takes exactly two."),
}

#: Modes that measure later LoRAs against the first, so they need at least two.
NEEDS_TWO = ("add-diff", "add-orth", "diff-export")


class PowerLoraMerger(io.ComfyNode):
    """Combine several LoRAs into a single file that behaves like all of them at once."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASNSPowerLoraMerger",
            display_name="Power LoRA Merger",
            search_aliases=[
                "WASNSPowerLoraMerger",
                "WAS Power LoRA Merger",
                "WAS Extras",
                "lora merge",
                "merge loras",
                "combine lora",
            ],
            category="WAS Suite/LoRA",
            description=(
                (
                    (
                        "Merge any number of LoRAs into one new LoRA file, saved into your "
                        "LoRA folder so it loads like any other. Add and remove rows with the "
                        "buttons on the node, and optionally connect a model and clip to get "
                        "the result applied straight away. Beyond `svd` the modes are: `add`, "
                        "stacking them exactly, for the largest file and the closest match to "
                        "loading them in turn; `rebase`, recompressing one LoRA; `add-diff` "
                        "and `add-orth`, starting from the first and adding what the others "
                        "differ by, orthogonalised in the second so they interfere less; "
                        "`diff-export`, saving only the difference between the first two; "
                        "`moe`, picking or blending the strongest source per module; "
                        "`obfuscate`, rewriting a stack's factors without changing what it "
                        "does; and `block-mix`, routing each part of the model to LoRA A or B "
                        "by block_mix_recipe."
                    )
                )
            ),
            inputs=[
                io.String.Input(
                    "output_filename",
                    default="merged_lora.safetensors",
                    tooltip=(
                        "Name to save the merged LoRA under, inside your LoRA folder. "
                        "Sub-folders are allowed, for example 'styles/mixed.safetensors'. "
                        "'.safetensors' is added when it is missing, and the name cannot "
                        "step outside the LoRA folder."
                    ),
                ),
                io.Float.Input(
                    "output_model_strength",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=0.01,
                    tooltip=(
                        "How strongly the merged LoRA is applied to the model output, if a "
                        "model is connected. 1.0 is full strength, 0.5 is half. This does "
                        "not change the saved file, only what comes out of the model socket."
                    ),
                ),
                io.Float.Input(
                    "output_clip_strength",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=0.01,
                    tooltip=(
                        "The same for the clip output: how strongly the merged LoRA is "
                        "applied to the connected clip. Lower it when a LoRA's trigger "
                        "words are overwhelming the rest of the prompt. Ignored where no "
                        "clip is connected, as with a diffusion transformer."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=MERGE_MODES,
                    default="svd",
                    tooltip=(
                        "How the LoRAs are combined. `svd` recompresses the combined result "
                        "back to one rank, which keeps the file small and is the usual "
                        "choice."
                    ),
                ),
                io.Combo.Input(
                    "block_mix_recipe",
                    options=BLOCK_MIX_RECIPES,
                    default="concept_a_style_b",
                    tooltip=(
                        "Only read in `block-mix` mode: which of the two LoRAs each part of "
                        "the model comes from. `all_a` and `all_b` route everything one way."
                    ),
                ),
                io.Model.Input(
                    "model",
                    optional=True,
                    tooltip=(
                        "Optional. A model to apply the merged LoRA to once it is saved, so "
                        "the merge can be tested in the same run. Leave it unconnected to "
                        "only write the file."
                    ),
                ),
                io.Clip.Input(
                    "clip",
                    optional=True,
                    tooltip=(
                        "Optional, and only used when a model is connected too: the clip the "
                        "merged LoRA's text-encoder half is applied to."
                    ),
                ),
                WAS_LORA_MERGE_OPTIONS.Input(
                    "options",
                    optional=True,
                    tooltip=(
                        "Optional settings from a Power LoRA Merger Options node: rank, "
                        "precision, module filters and the per-mode controls. Leave it "
                        "unconnected to merge at rank 32 in bf16."
                    ),
                ),
                io.Boolean.Input(
                    "lora_1_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                io.Combo.Input(
                    "lora_1", options=lora_names(), default=NONE_OPTION,
                    optional=True, socketless=True, tooltip=ROW_NAME_TIP,
                ),
                io.Float.Input(
                    "lora_1_weight", default=1.0, min=-10.0, max=10.0, step=0.01,
                    optional=True, socketless=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_2_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                io.Combo.Input(
                    "lora_2", options=lora_names(), default=NONE_OPTION,
                    optional=True, socketless=True, tooltip=ROW_NAME_TIP,
                ),
                io.Float.Input(
                    "lora_2_weight", default=1.0, min=-10.0, max=10.0, step=0.01,
                    optional=True, socketless=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_3_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                io.Combo.Input(
                    "lora_3", options=lora_names(), default=NONE_OPTION,
                    optional=True, socketless=True, tooltip=ROW_NAME_TIP,
                ),
                io.Float.Input(
                    "lora_3_weight", default=1.0, min=-10.0, max=10.0, step=0.01,
                    optional=True, socketless=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_4_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                io.Combo.Input(
                    "lora_4", options=lora_names(), default=NONE_OPTION,
                    optional=True, socketless=True, tooltip=ROW_NAME_TIP,
                ),
                io.Float.Input(
                    "lora_4_weight", default=1.0, min=-10.0, max=10.0, step=0.01,
                    optional=True, socketless=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_5_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                io.Combo.Input(
                    "lora_5", options=lora_names(), default=NONE_OPTION,
                    optional=True, socketless=True, tooltip=ROW_NAME_TIP,
                ),
                io.Float.Input(
                    "lora_5_weight", default=1.0, min=-10.0, max=10.0, step=0.01,
                    optional=True, socketless=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_6_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                io.Combo.Input(
                    "lora_6", options=lora_names(), default=NONE_OPTION,
                    optional=True, socketless=True, tooltip=ROW_NAME_TIP,
                ),
                io.Float.Input(
                    "lora_6_weight", default=1.0, min=-10.0, max=10.0, step=0.01,
                    optional=True, socketless=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_7_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                io.Combo.Input(
                    "lora_7", options=lora_names(), default=NONE_OPTION,
                    optional=True, socketless=True, tooltip=ROW_NAME_TIP,
                ),
                io.Float.Input(
                    "lora_7_weight", default=1.0, min=-10.0, max=10.0, step=0.01,
                    optional=True, socketless=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_8_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                io.Combo.Input(
                    "lora_8", options=lora_names(), default=NONE_OPTION,
                    optional=True, socketless=True, tooltip=ROW_NAME_TIP,
                ),
                io.Float.Input(
                    "lora_8_weight", default=1.0, min=-10.0, max=10.0, step=0.01,
                    optional=True, socketless=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_9_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                io.Combo.Input(
                    "lora_9", options=lora_names(), default=NONE_OPTION,
                    optional=True, socketless=True, tooltip=ROW_NAME_TIP,
                ),
                io.Float.Input(
                    "lora_9_weight", default=1.0, min=-10.0, max=10.0, step=0.01,
                    optional=True, socketless=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_10_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                io.Combo.Input(
                    "lora_10", options=lora_names(), default=NONE_OPTION,
                    optional=True, socketless=True, tooltip=ROW_NAME_TIP,
                ),
                io.Float.Input(
                    "lora_10_weight", default=1.0, min=-10.0, max=10.0, step=0.01,
                    optional=True, socketless=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_11_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                io.Combo.Input(
                    "lora_11", options=lora_names(), default=NONE_OPTION,
                    optional=True, socketless=True, tooltip=ROW_NAME_TIP,
                ),
                io.Float.Input(
                    "lora_11_weight", default=1.0, min=-10.0, max=10.0, step=0.01,
                    optional=True, socketless=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_12_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                io.Combo.Input(
                    "lora_12", options=lora_names(), default=NONE_OPTION,
                    optional=True, socketless=True, tooltip=ROW_NAME_TIP,
                ),
                io.Float.Input(
                    "lora_12_weight", default=1.0, min=-10.0, max=10.0, step=0.01,
                    optional=True, socketless=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_13_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                io.Combo.Input(
                    "lora_13", options=lora_names(), default=NONE_OPTION,
                    optional=True, socketless=True, tooltip=ROW_NAME_TIP,
                ),
                io.Float.Input(
                    "lora_13_weight", default=1.0, min=-10.0, max=10.0, step=0.01,
                    optional=True, socketless=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_14_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                io.Combo.Input(
                    "lora_14", options=lora_names(), default=NONE_OPTION,
                    optional=True, socketless=True, tooltip=ROW_NAME_TIP,
                ),
                io.Float.Input(
                    "lora_14_weight", default=1.0, min=-10.0, max=10.0, step=0.01,
                    optional=True, socketless=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_15_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                io.Combo.Input(
                    "lora_15", options=lora_names(), default=NONE_OPTION,
                    optional=True, socketless=True, tooltip=ROW_NAME_TIP,
                ),
                io.Float.Input(
                    "lora_15_weight", default=1.0, min=-10.0, max=10.0, step=0.01,
                    optional=True, socketless=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_16_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                io.Combo.Input(
                    "lora_16", options=lora_names(), default=NONE_OPTION,
                    optional=True, socketless=True, tooltip=ROW_NAME_TIP,
                ),
                io.Float.Input(
                    "lora_16_weight", default=1.0, min=-10.0, max=10.0, step=0.01,
                    optional=True, socketless=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_17_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                io.Combo.Input(
                    "lora_17", options=lora_names(), default=NONE_OPTION,
                    optional=True, socketless=True, tooltip=ROW_NAME_TIP,
                ),
                io.Float.Input(
                    "lora_17_weight", default=1.0, min=-10.0, max=10.0, step=0.01,
                    optional=True, socketless=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_18_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                io.Combo.Input(
                    "lora_18", options=lora_names(), default=NONE_OPTION,
                    optional=True, socketless=True, tooltip=ROW_NAME_TIP,
                ),
                io.Float.Input(
                    "lora_18_weight", default=1.0, min=-10.0, max=10.0, step=0.01,
                    optional=True, socketless=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_19_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                io.Combo.Input(
                    "lora_19", options=lora_names(), default=NONE_OPTION,
                    optional=True, socketless=True, tooltip=ROW_NAME_TIP,
                ),
                io.Float.Input(
                    "lora_19_weight", default=1.0, min=-10.0, max=10.0, step=0.01,
                    optional=True, socketless=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_20_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                io.Combo.Input(
                    "lora_20", options=lora_names(), default=NONE_OPTION,
                    optional=True, socketless=True, tooltip=ROW_NAME_TIP,
                ),
                io.Float.Input(
                    "lora_20_weight", default=1.0, min=-10.0, max=10.0, step=0.01,
                    optional=True, socketless=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_21_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                io.Combo.Input(
                    "lora_21", options=lora_names(), default=NONE_OPTION,
                    optional=True, socketless=True, tooltip=ROW_NAME_TIP,
                ),
                io.Float.Input(
                    "lora_21_weight", default=1.0, min=-10.0, max=10.0, step=0.01,
                    optional=True, socketless=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_22_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                io.Combo.Input(
                    "lora_22", options=lora_names(), default=NONE_OPTION,
                    optional=True, socketless=True, tooltip=ROW_NAME_TIP,
                ),
                io.Float.Input(
                    "lora_22_weight", default=1.0, min=-10.0, max=10.0, step=0.01,
                    optional=True, socketless=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_23_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                io.Combo.Input(
                    "lora_23", options=lora_names(), default=NONE_OPTION,
                    optional=True, socketless=True, tooltip=ROW_NAME_TIP,
                ),
                io.Float.Input(
                    "lora_23_weight", default=1.0, min=-10.0, max=10.0, step=0.01,
                    optional=True, socketless=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_24_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                io.Combo.Input(
                    "lora_24", options=lora_names(), default=NONE_OPTION,
                    optional=True, socketless=True, tooltip=ROW_NAME_TIP,
                ),
                io.Float.Input(
                    "lora_24_weight", default=1.0, min=-10.0, max=10.0, step=0.01,
                    optional=True, socketless=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_25_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                io.Combo.Input(
                    "lora_25", options=lora_names(), default=NONE_OPTION,
                    optional=True, socketless=True, tooltip=ROW_NAME_TIP,
                ),
                io.Float.Input(
                    "lora_25_weight", default=1.0, min=-10.0, max=10.0, step=0.01,
                    optional=True, socketless=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_26_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                io.Combo.Input(
                    "lora_26", options=lora_names(), default=NONE_OPTION,
                    optional=True, socketless=True, tooltip=ROW_NAME_TIP,
                ),
                io.Float.Input(
                    "lora_26_weight", default=1.0, min=-10.0, max=10.0, step=0.01,
                    optional=True, socketless=True, tooltip=ROW_WEIGHT_TIP,
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="model",
                    tooltip=(
                        "The connected model with the merged LoRA applied, or nothing when "
                        "no model was connected."
                    ),
                ),
                io.Clip.Output(
                    display_name="clip",
                    tooltip=(
                        "The connected clip with the merged LoRA applied, or nothing when no "
                        "clip was connected. A model on its own is still patched."
                    ),
                ),
                io.String.Output(
                    display_name="lora_path",
                    tooltip=(
                        "The saved file's name relative to your LoRA folder, such as "
                        "'styles/mixed.safetensors'. Feed it to a loader, or to a text node "
                        "to record what a run produced."
                    ),
                ),
            ],
            # Rows are added and removed on the canvas, so they arrive as extra inputs
            # rather than as declared sockets.
            accept_all_inputs=True,
        )

    @classmethod
    def execute(
        cls,
        output_filename,
        output_model_strength,
        output_clip_strength,
        mode,
        block_mix_recipe,
        model=None,
        clip=None,
        options=None,
        **rows,
    ) -> io.NodeOutput:
        import folder_paths
        from safetensors.torch import save_file

        from ....modules.lora import merge_loras_z, output, rows as lora_rows
        from ....modules.lora.options import MergeSettings
        from ....modules.lora.progress import MergeProgress

        selected = lora_rows.rows_from_inputs(rows)
        if not selected:
            raise ValueError(
                "No LoRA is selected. Add a row with the node's Add LoRA button, choose a "
                "file in it, and leave its weight above zero."
            )

        sources = []
        for row in selected:
            full_path = folder_paths.get_full_path("loras", row.lora)
            if full_path is None:
                raise ValueError(
                    f"The LoRA `{row.lora}` is not in your LoRA folder. Refresh the list on "
                    f"the node, or pick another file."
                )
            sources.append((full_path, row.weight))

        exact = EXACT_COUNTS.get(mode)
        if exact is not None and len(sources) != exact[0]:
            counted = "1 is" if len(sources) == 1 else f"{len(sources)} are"
            raise ValueError(
                f"{exact[1]} {counted} switched on, change the rows, or pick another mode."
            )
        if mode in NEEDS_TWO and len(sources) < 2:
            raise ValueError(
                f"{mode} mode works out what the later LoRAs add to the first one, so it "
                f"needs at least two. Only {len(sources)} is switched on."
            )

        settings = MergeSettings.read(options)
        device = merge_loras_z.get_device(force_cpu=settings.cpu)
        out_dtype = merge_loras_z.get_dtype(settings.dtype)
        compute_dtype = merge_loras_z.get_compute_dtype(settings.compute_dtype)

        directory = output.lora_directory()
        target, relative = output.resolve_output(directory, output_filename)
        identity = cls._metadata(
            mode, sources, settings, out_dtype, compute_dtype, block_mix_recipe
        )
        if output.built_from(target, identity):
            logger.info("%s already holds this merge, so it is served as it stands", target)
            return cls._from_file(
                model, clip, relative, output_model_strength, output_clip_strength
            )

        # Checked before the merge runs.
        held = output.claimed(target)
        if held is not None:
            raise ValueError(
                f"`{relative}` cannot be written: {held}\n"
                f"  That file is loaded in this ComfyUI session, and a file a loaded model "
                f"reads from cannot be overwritten while it is held.\n"
                f"  Give output_filename a name that is not in use, or restart ComfyUI to "
                f"let go of it. Rows or settings left unchanged would have served the file "
                f"as it stands instead."
            )

        progress = MergeProgress(1, desc="WAS LoRA Merger")
        try:
            loaded, reference_metadata = cls._load(
                sources, device, compute_dtype, settings, mode, progress
            )
            modules = max(1, len({prefix for pairs, _ in loaded for prefix in pairs}))
            progress.set_total((modules * 2) + 5)
            progress.update_absolute(0)

            merged = cls._merge(
                mode, block_mix_recipe, loaded, settings, compute_dtype,
                lambda current, total: progress.update_absolute(
                    min(modules, round((current / max(total, 1)) * modules))
                ),
            )

            state, metadata = merge_loras_z.build_state_dict(
                merged,
                metadata_ref=reference_metadata,
                dtype=out_dtype,
                device=device,
                progress_cb=lambda stage, current, total, message=None: progress.update_absolute(
                    modules + min(modules, round((current / max(total, 1)) * modules))
                ),
            )
            cls._report("merged", merged, 1.0, compute_dtype, settings)
            progress.update_absolute((modules * 2) + 1)

            metadata.update(identity)
            if mode == "obfuscate":
                for key in ("mode", "merged_from", "include_patterns", "exclude_patterns"):
                    metadata.pop(key, None)

            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                save_file(state, str(target), metadata=metadata)
            except (PermissionError, OSError) as refused:
                raise RuntimeError(
                    f"The merged LoRA could not be written to `{relative}`: {refused}\n"
                    f"  Something took hold of that file while the merge ran, which on "
                    f"Windows is usually a loader reading it.\n"
                    f"  Give output_filename a name that is not in use, or restart ComfyUI "
                    f"to let go of it. The merge itself finished, so only the save was lost."
                ) from refused
            logger.info("saved the merged LoRA to %s", target)

            progress.update_absolute(progress.total)
        finally:
            progress.close()

        if model is not None:
            import comfy.sd

            # Applied from the tensors in hand rather than from the saved file.
            model, patched = comfy.sd.load_lora_for_models(
                model, clip, state, output_model_strength,
                output_clip_strength if clip is not None else 0.0,
            )
            if clip is not None:
                clip = patched
        return io.NodeOutput(model, clip, relative)

    @classmethod
    def _from_file(cls, model, clip, relative, model_strength, clip_strength):
        """Answer with a merged LoRA that is already on disk.

        Args:
            model: The model to apply it to, or ``None``.
            clip: The clip to apply it to, or ``None``.
            relative: The file's name inside the LoRA directory.
            model_strength: How strongly it is applied to the model.
            clip_strength: How strongly it is applied to the clip.

        Returns:
            The node's three outputs.
        """
        if model is None:
            return io.NodeOutput(model, clip, relative)

        from nodes import LoraLoader

        model, patched = LoraLoader().load_lora(
            model, clip, relative, model_strength,
            clip_strength if clip is not None else 0.0,
        )
        return io.NodeOutput(model, patched if clip is not None else clip, relative)

    @classmethod
    def _load(cls, sources, device, compute_dtype, settings, mode, progress):
        """Read every selected LoRA off disk.

        Args:
            sources: ``(path, weight)`` per selected row, in row order.
            device: Where the tensors are loaded to.
            compute_dtype: Precision the summary statistics are worked out in.
            settings: The merge settings, for the module filters.
            mode: The merge mode, named in the log line only.
            progress: Bar to drive while the files are read.

        Returns:
            ``(loaded, metadata)``, a ``(pairs, weight)`` entry per file, and the first
            file's metadata, which the merged file inherits its key naming style from.
        """
        from ....modules.lora import merge_loras_z

        loaded = []
        reference_metadata = {}
        for index, (path, weight) in enumerate(sources):
            pairs, metadata = merge_loras_z.load_lora_pairs(
                path,
                device=device,
                progress_cb=lambda stage, current, total, message=None: progress.update_absolute(current),
            )
            if index == 0:
                reference_metadata = metadata if isinstance(metadata, dict) else {}
            loaded.append((pairs, weight))
            cls._report("input", pairs, weight, compute_dtype, settings, path=path, mode=mode)
        return loaded, reference_metadata

    @classmethod
    def _merge(cls, mode, block_mix_recipe, loaded, settings, compute_dtype, on_progress):
        """Run one merge mode over the loaded LoRAs.

        Args:
            mode: The chosen merge mode.
            block_mix_recipe: Routing recipe, read by ``block-mix`` only.
            loaded: ``(pairs, weight)`` per file.
            settings: The merge settings.
            compute_dtype: Precision the arithmetic runs in, or ``None`` for automatic.
            on_progress: ``(current, total)`` callback driving the progress bar.

        Returns:
            The merged pairs, one per module.
        """
        from ....modules.lora import merge_loras_z

        def progress_cb(stage, current, total, message=None):
            on_progress(current, total)

        common = {
            "include_patterns": settings.include_list,
            "exclude_patterns": settings.exclude_list,
            "explicit_compute_dtype": compute_dtype,
            "progress_cb": progress_cb,
        }
        ranked = {"rank_value": settings.rank, "auto_rank_threshold": settings.auto_rank_threshold}

        if mode == "add":
            return merge_loras_z.merge_mode_add(
                loaded, settings.include_list, settings.exclude_list, compute_dtype,
                progress_cb=progress_cb,
            )
        if mode == "block-mix":
            return cls._block_mix(loaded, block_mix_recipe, settings, common, ranked)
        if mode == "add-diff":
            return merge_loras_z.merge_mode_add_diff(loaded, **ranked, **common)
        if mode == "add-orth":
            return merge_loras_z.merge_mode_add_orth(loaded, **ranked, **common)
        if mode == "diff-export":
            return merge_loras_z.merge_mode_diff_export(loaded, **ranked, **common)
        if mode == "moe":
            return merge_loras_z.merge_mode_moe(
                loaded,
                moe_temperature=settings.moe_temperature,
                moe_hard=settings.moe_hard,
                **ranked,
                **common,
            )
        if mode == "obfuscate":
            return merge_loras_z.merge_mode_obfuscate(loaded, **common)
        if mode == "rebase":
            return merge_loras_z.merge_mode_rebase(loaded[0], **ranked, **common)
        return merge_loras_z.merge_mode_svd(
            loaded,
            preserve_norm=settings.preserve_norm,
            cap_mult=settings.cap,
            **ranked,
            **common,
        )

    @classmethod
    def _block_mix(cls, loaded, recipe, settings, common, ranked):
        """Route each module to one of two LoRAs, or blend them per role.

        Args:
            loaded: ``(pairs, weight)`` for LoRA A and LoRA B.
            recipe: Which modules come from which side.
            settings: The merge settings, for the preset, method and mix ratios.
            common: Filters, compute dtype and progress callback shared by every mode.
            ranked: Rank and automatic-rank threshold.

        Returns:
            The merged pairs.
        """
        from ....modules.lora import merge_loras_z

        try:
            routing = merge_loras_z.block_mix_routing_report(
                a_pairs=loaded[0][0],
                b_pairs=loaded[1][0],
                preset=settings.block_mix_preset,
                recipe=recipe,
                include_patterns=settings.include_list,
                exclude_patterns=settings.exclude_list,
            )
            logger.info("block-mix routing:\n%s", json.dumps(routing, indent=2))
        except Exception as error:
            logger.warning("the block-mix routing report could not be built: %s", error)

        shared = {
            "method": settings.block_mix_method,
            "preset": settings.block_mix_preset,
            "recipe": recipe,
            **ranked,
            **common,
        }
        if settings.block_mix_weighted:
            return merge_loras_z.merge_mode_block_mix_weighted(
                loaded,
                concept_mix=settings.block_mix_concept_mix,
                style_mix=settings.block_mix_style_mix,
                **shared,
            )
        return merge_loras_z.merge_mode_block_mix(loaded, **shared)

    @classmethod
    def _report(cls, stage, pairs, weight, compute_dtype, settings, path=None, mode=None):
        """Log what a set of LoRA pairs contains.

        Args:
            stage: ``"input"`` for a source file, ``"merged"`` for the result.
            pairs: The pairs to summarise.
            weight: Weight the pairs are counted at.
            compute_dtype: Precision the statistics are worked out in.
            settings: The merge settings, for the module filters.
            path: Source file, for an input report.
            mode: Merge mode, for an input report.
        """
        from ....modules.lora import merge_loras_z

        try:
            stats = merge_loras_z.summarize_pairs(
                pairs,
                weight=weight,
                explicit_compute_dtype=compute_dtype,
                include_patterns=settings.include_list,
                exclude_patterns=settings.exclude_list,
            )
        except Exception as error:
            logger.warning("the %s report could not be built: %s", stage, error)
            return
        report = {"stats": stats}
        if path is not None:
            report = {"path": path, "weight": float(weight), "mode": mode, "stats": stats}
        logger.info("%s report:\n%s", stage, json.dumps(report, indent=2))

    @classmethod
    def _metadata(cls, mode, sources, settings, out_dtype, compute_dtype, block_mix_recipe=""):
        """Build the metadata block saved inside the merged file.

        Args:
            mode: The merge mode used.
            sources: ``(path, weight)`` per source file.
            settings: The merge settings.
            out_dtype: Precision the tensors were saved in.
            compute_dtype: Precision the arithmetic ran in, or ``None`` for automatic.
            block_mix_recipe: Which modules block-mix took from which LoRA.

        Returns:
            Metadata keys to add to the ones the merge produced. Every value is a string,
            which is all the safetensors header holds.
        """
        note = "Created by WAS Merge Loras"
        return {
            "merged_from": str([(path, weight) for path, weight in sources] if mode != "obfuscate" else None),
            "mode": mode,
            "block_mix_recipe": str(block_mix_recipe) if mode == "block-mix" else "None",
            "rank": str(settings.rank),
            "dtype": str(out_dtype).replace("torch.", ""),
            "compute_dtype": (
                str(compute_dtype).replace("torch.", "")
                if compute_dtype is not None
                else "auto(bf16-if-mixed)"
            ),
            "preserve_norm": str(settings.preserve_norm),
            "cap_mult": str(settings.cap_mult) if settings.cap_mult_enable else "None",
            "include_patterns": str(settings.include_list),
            "exclude_patterns": str(settings.exclude_list),
            "moe_temperature": str(settings.moe_temperature),
            "moe_hard": str(settings.moe_hard),
            "auto_rank_threshold": str(settings.auto_rank_threshold),
            "tool": "WAS Merge Loras",
            "note": note if mode == "obfuscate" else f"{note} - Merging Mode: {mode}",
        }

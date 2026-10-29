"""Apply any number of LoRAs to a model and CLIP from one growing list of rows."""

from __future__ import annotations

import os

from comfy_api.latest import io

from ....modules.compat.sockets import require_input
from ....modules.compat.types import LIST, LORA_STACK, NUMBER
from ....modules.io import picker
from ....modules.log import get_logger
from ...legacy.loaders.lora_loader import NONE_OPTION, lora_names

logger = get_logger("nodes.extras.lora")

#: How many rows the node offers, and how many name outputs it answers. Neither a schema's
#: inputs nor its outputs can grow, so every row is declared up front and the interface draws
#: only as many as are in use. A row past this one is still counted into the joined names.
MAX_ROWS = 26

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
    "How strongly this row counts: 1.0 = as trained; below 1 weakens it; above 1 pushes past "
    "it; negative reverses it; 0 leaves the row out. A list, as Number Sequence's values, sets "
    "it per sampling step, times strength."
)

#: Tooltip carried by each per-row name output, filled in with that row's number.
NAME_TOOLTIP = (
    "File name of applied LoRA {slot}, without its folder or its extension, the same name "
    "that appears in the names output. Empty when the stack applies fewer LoRAs than that."
)


def name_slots(names: list[str]) -> list[str]:
    """The per-row name outputs, one string per slot.

    Args:
        names: The applied LoRA names, in the order they were applied.

    Returns:
        Exactly :data:`MAX_ROWS` strings, cut to that ceiling and padded with empty ones.
    """
    filled = list(names[:MAX_ROWS])
    return filled + [""] * (MAX_ROWS - len(filled))


class PowerLoraLoader(io.ComfyNode):
    """Stack several LoRAs onto a model in one node, each with its own strength and switch."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASPowerLoraLoader",
            display_name="Power LoRA Loader",
            search_aliases=[
                "WASPowerLoraLoader",
                "Power LoRA Loader",
                "power lora",
                "multi lora",
                "lora stack",
                "load loras",
            ],
            category="WAS Suite/LoRA",
            description=(
                "Apply any number of LoRAs in one node. Each row names a file, carries its "
                "own strength and has a switch, and a list on a row's weight or on strength "
                "sets it at every sampling step. The lora_stack output hands the rows to Power "
                "LoRA Merger, which bakes them into a single file."
            ),
            inputs=[
                io.Model.Input(
                    "model",
                    optional=True,
                    tooltip=(
                        "The diffusion model the LoRAs are applied to, in row order. Needed "
                        "only where the model output is wired; lora_stack alone needs none."
                    ),
                ),
                io.Clip.Input(
                    "clip",
                    optional=True,
                    tooltip=(
                        "The text encoder the LoRAs are applied to. Optional: left "
                        "unconnected the model alone is patched, which is what a workflow "
                        "encoding its prompt elsewhere wants."
                    ),
                ),
                io.Boolean.Input(
                    "fuse",
                    default=False,
                    optional=True,
                    tooltip=(
                        "`false` = one patch per row; `true` = every row joined into one "
                        "patch, cheaper on a long stack, near-identical results within float "
                        "rounding. A row whose format cannot be joined is carried through, "
                        "scaled."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input(
                        "strength",
                        default=1.0,
                        min=-10.0,
                        max=10.0,
                        step=0.01,
                        optional=True,
                    ),
                    [io.Float, NUMBER, LIST],
                    optional=True,
                    tooltip=(
                        "Multiplies every row's strength; 1.0 leaves the rows as set. A list, as "
                        "Number Sequence's values, sets it per sampling step, stretched to fit; "
                        "the text encoder takes the first value, and steps away from it run "
                        "slower."
                    ),
                ),
                io.Boolean.Input(
                    "lora_1_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                picker.path_input(io.Combo.Input(
                    "lora_1", options=lora_names(), default=NONE_OPTION,
                    optional=True, tooltip=ROW_NAME_TIP,
                ), model=True),
                io.MultiType.Input(
                    io.Float.Input("lora_1_weight", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
                    [io.Float, NUMBER, LIST], optional=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_2_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                picker.path_input(io.Combo.Input(
                    "lora_2", options=lora_names(), default=NONE_OPTION,
                    optional=True, tooltip=ROW_NAME_TIP,
                ), model=True),
                io.MultiType.Input(
                    io.Float.Input("lora_2_weight", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
                    [io.Float, NUMBER, LIST], optional=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_3_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                picker.path_input(io.Combo.Input(
                    "lora_3", options=lora_names(), default=NONE_OPTION,
                    optional=True, tooltip=ROW_NAME_TIP,
                ), model=True),
                io.MultiType.Input(
                    io.Float.Input("lora_3_weight", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
                    [io.Float, NUMBER, LIST], optional=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_4_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                picker.path_input(io.Combo.Input(
                    "lora_4", options=lora_names(), default=NONE_OPTION,
                    optional=True, tooltip=ROW_NAME_TIP,
                ), model=True),
                io.MultiType.Input(
                    io.Float.Input("lora_4_weight", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
                    [io.Float, NUMBER, LIST], optional=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_5_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                picker.path_input(io.Combo.Input(
                    "lora_5", options=lora_names(), default=NONE_OPTION,
                    optional=True, tooltip=ROW_NAME_TIP,
                ), model=True),
                io.MultiType.Input(
                    io.Float.Input("lora_5_weight", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
                    [io.Float, NUMBER, LIST], optional=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_6_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                picker.path_input(io.Combo.Input(
                    "lora_6", options=lora_names(), default=NONE_OPTION,
                    optional=True, tooltip=ROW_NAME_TIP,
                ), model=True),
                io.MultiType.Input(
                    io.Float.Input("lora_6_weight", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
                    [io.Float, NUMBER, LIST], optional=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_7_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                picker.path_input(io.Combo.Input(
                    "lora_7", options=lora_names(), default=NONE_OPTION,
                    optional=True, tooltip=ROW_NAME_TIP,
                ), model=True),
                io.MultiType.Input(
                    io.Float.Input("lora_7_weight", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
                    [io.Float, NUMBER, LIST], optional=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_8_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                picker.path_input(io.Combo.Input(
                    "lora_8", options=lora_names(), default=NONE_OPTION,
                    optional=True, tooltip=ROW_NAME_TIP,
                ), model=True),
                io.MultiType.Input(
                    io.Float.Input("lora_8_weight", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
                    [io.Float, NUMBER, LIST], optional=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_9_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                picker.path_input(io.Combo.Input(
                    "lora_9", options=lora_names(), default=NONE_OPTION,
                    optional=True, tooltip=ROW_NAME_TIP,
                ), model=True),
                io.MultiType.Input(
                    io.Float.Input("lora_9_weight", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
                    [io.Float, NUMBER, LIST], optional=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_10_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                picker.path_input(io.Combo.Input(
                    "lora_10", options=lora_names(), default=NONE_OPTION,
                    optional=True, tooltip=ROW_NAME_TIP,
                ), model=True),
                io.MultiType.Input(
                    io.Float.Input("lora_10_weight", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
                    [io.Float, NUMBER, LIST], optional=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_11_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                picker.path_input(io.Combo.Input(
                    "lora_11", options=lora_names(), default=NONE_OPTION,
                    optional=True, tooltip=ROW_NAME_TIP,
                ), model=True),
                io.MultiType.Input(
                    io.Float.Input("lora_11_weight", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
                    [io.Float, NUMBER, LIST], optional=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_12_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                picker.path_input(io.Combo.Input(
                    "lora_12", options=lora_names(), default=NONE_OPTION,
                    optional=True, tooltip=ROW_NAME_TIP,
                ), model=True),
                io.MultiType.Input(
                    io.Float.Input("lora_12_weight", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
                    [io.Float, NUMBER, LIST], optional=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_13_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                picker.path_input(io.Combo.Input(
                    "lora_13", options=lora_names(), default=NONE_OPTION,
                    optional=True, tooltip=ROW_NAME_TIP,
                ), model=True),
                io.MultiType.Input(
                    io.Float.Input("lora_13_weight", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
                    [io.Float, NUMBER, LIST], optional=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_14_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                picker.path_input(io.Combo.Input(
                    "lora_14", options=lora_names(), default=NONE_OPTION,
                    optional=True, tooltip=ROW_NAME_TIP,
                ), model=True),
                io.MultiType.Input(
                    io.Float.Input("lora_14_weight", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
                    [io.Float, NUMBER, LIST], optional=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_15_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                picker.path_input(io.Combo.Input(
                    "lora_15", options=lora_names(), default=NONE_OPTION,
                    optional=True, tooltip=ROW_NAME_TIP,
                ), model=True),
                io.MultiType.Input(
                    io.Float.Input("lora_15_weight", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
                    [io.Float, NUMBER, LIST], optional=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_16_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                picker.path_input(io.Combo.Input(
                    "lora_16", options=lora_names(), default=NONE_OPTION,
                    optional=True, tooltip=ROW_NAME_TIP,
                ), model=True),
                io.MultiType.Input(
                    io.Float.Input("lora_16_weight", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
                    [io.Float, NUMBER, LIST], optional=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_17_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                picker.path_input(io.Combo.Input(
                    "lora_17", options=lora_names(), default=NONE_OPTION,
                    optional=True, tooltip=ROW_NAME_TIP,
                ), model=True),
                io.MultiType.Input(
                    io.Float.Input("lora_17_weight", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
                    [io.Float, NUMBER, LIST], optional=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_18_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                picker.path_input(io.Combo.Input(
                    "lora_18", options=lora_names(), default=NONE_OPTION,
                    optional=True, tooltip=ROW_NAME_TIP,
                ), model=True),
                io.MultiType.Input(
                    io.Float.Input("lora_18_weight", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
                    [io.Float, NUMBER, LIST], optional=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_19_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                picker.path_input(io.Combo.Input(
                    "lora_19", options=lora_names(), default=NONE_OPTION,
                    optional=True, tooltip=ROW_NAME_TIP,
                ), model=True),
                io.MultiType.Input(
                    io.Float.Input("lora_19_weight", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
                    [io.Float, NUMBER, LIST], optional=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_20_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                picker.path_input(io.Combo.Input(
                    "lora_20", options=lora_names(), default=NONE_OPTION,
                    optional=True, tooltip=ROW_NAME_TIP,
                ), model=True),
                io.MultiType.Input(
                    io.Float.Input("lora_20_weight", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
                    [io.Float, NUMBER, LIST], optional=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_21_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                picker.path_input(io.Combo.Input(
                    "lora_21", options=lora_names(), default=NONE_OPTION,
                    optional=True, tooltip=ROW_NAME_TIP,
                ), model=True),
                io.MultiType.Input(
                    io.Float.Input("lora_21_weight", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
                    [io.Float, NUMBER, LIST], optional=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_22_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                picker.path_input(io.Combo.Input(
                    "lora_22", options=lora_names(), default=NONE_OPTION,
                    optional=True, tooltip=ROW_NAME_TIP,
                ), model=True),
                io.MultiType.Input(
                    io.Float.Input("lora_22_weight", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
                    [io.Float, NUMBER, LIST], optional=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_23_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                picker.path_input(io.Combo.Input(
                    "lora_23", options=lora_names(), default=NONE_OPTION,
                    optional=True, tooltip=ROW_NAME_TIP,
                ), model=True),
                io.MultiType.Input(
                    io.Float.Input("lora_23_weight", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
                    [io.Float, NUMBER, LIST], optional=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_24_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                picker.path_input(io.Combo.Input(
                    "lora_24", options=lora_names(), default=NONE_OPTION,
                    optional=True, tooltip=ROW_NAME_TIP,
                ), model=True),
                io.MultiType.Input(
                    io.Float.Input("lora_24_weight", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
                    [io.Float, NUMBER, LIST], optional=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_25_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                picker.path_input(io.Combo.Input(
                    "lora_25", options=lora_names(), default=NONE_OPTION,
                    optional=True, tooltip=ROW_NAME_TIP,
                ), model=True),
                io.MultiType.Input(
                    io.Float.Input("lora_25_weight", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
                    [io.Float, NUMBER, LIST], optional=True, tooltip=ROW_WEIGHT_TIP,
                ),
                io.Boolean.Input(
                    "lora_26_enabled", default=True, optional=True,
                    socketless=True, tooltip=ROW_ON_TIP,
                ),
                picker.path_input(io.Combo.Input(
                    "lora_26", options=lora_names(), default=NONE_OPTION,
                    optional=True, tooltip=ROW_NAME_TIP,
                ), model=True),
                io.MultiType.Input(
                    io.Float.Input("lora_26_weight", default=1.0, min=-10.0, max=10.0, step=0.01, optional=True),
                    [io.Float, NUMBER, LIST], optional=True, tooltip=ROW_WEIGHT_TIP,
                ),
            ],
            outputs=[
                io.Model.Output(
                    display_name="model",
                    tooltip="The model with every switched-on row applied, in order.",
                ),
                io.Clip.Output(
                    display_name="clip",
                    tooltip=(
                        "The text encoder with the same rows applied. The input passes "
                        "through untouched when no CLIP was connected."
                    ),
                ),
                LORA_STACK.Output(
                    display_name="lora_stack",
                    tooltip=(
                        "The switched-on rows as a LoRA stack, each at its strength times "
                        "strength, a list counting at its first value, for the stack_data input "
                        "of Power LoRA Merger. Empty when every row is off."
                    ),
                ),
                io.String.Output(
                    display_name="names",
                    tooltip=(
                        "The LoRAs that were applied, one per line, as name and strength, or "
                        "under a list as name, `per step` and the strength at step 1. Empty when "
                        "every row is off."
                    ),
                ),
                # One per row, after the four fixed outputs: a link is kept by position.
                io.String.Output(display_name="name_1", tooltip=NAME_TOOLTIP.format(slot=1)),
                io.String.Output(display_name="name_2", tooltip=NAME_TOOLTIP.format(slot=2)),
                io.String.Output(display_name="name_3", tooltip=NAME_TOOLTIP.format(slot=3)),
                io.String.Output(display_name="name_4", tooltip=NAME_TOOLTIP.format(slot=4)),
                io.String.Output(display_name="name_5", tooltip=NAME_TOOLTIP.format(slot=5)),
                io.String.Output(display_name="name_6", tooltip=NAME_TOOLTIP.format(slot=6)),
                io.String.Output(display_name="name_7", tooltip=NAME_TOOLTIP.format(slot=7)),
                io.String.Output(display_name="name_8", tooltip=NAME_TOOLTIP.format(slot=8)),
                io.String.Output(display_name="name_9", tooltip=NAME_TOOLTIP.format(slot=9)),
                io.String.Output(display_name="name_10", tooltip=NAME_TOOLTIP.format(slot=10)),
                io.String.Output(display_name="name_11", tooltip=NAME_TOOLTIP.format(slot=11)),
                io.String.Output(display_name="name_12", tooltip=NAME_TOOLTIP.format(slot=12)),
                io.String.Output(display_name="name_13", tooltip=NAME_TOOLTIP.format(slot=13)),
                io.String.Output(display_name="name_14", tooltip=NAME_TOOLTIP.format(slot=14)),
                io.String.Output(display_name="name_15", tooltip=NAME_TOOLTIP.format(slot=15)),
                io.String.Output(display_name="name_16", tooltip=NAME_TOOLTIP.format(slot=16)),
                io.String.Output(display_name="name_17", tooltip=NAME_TOOLTIP.format(slot=17)),
                io.String.Output(display_name="name_18", tooltip=NAME_TOOLTIP.format(slot=18)),
                io.String.Output(display_name="name_19", tooltip=NAME_TOOLTIP.format(slot=19)),
                io.String.Output(display_name="name_20", tooltip=NAME_TOOLTIP.format(slot=20)),
                io.String.Output(display_name="name_21", tooltip=NAME_TOOLTIP.format(slot=21)),
                io.String.Output(display_name="name_22", tooltip=NAME_TOOLTIP.format(slot=22)),
                io.String.Output(display_name="name_23", tooltip=NAME_TOOLTIP.format(slot=23)),
                io.String.Output(display_name="name_24", tooltip=NAME_TOOLTIP.format(slot=24)),
                io.String.Output(display_name="name_25", tooltip=NAME_TOOLTIP.format(slot=25)),
                io.String.Output(display_name="name_26", tooltip=NAME_TOOLTIP.format(slot=26)),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model=None, clip=None, fuse=False, strength=1.0, **rows) -> io.NodeOutput:
        """Apply every switched-on row.

        Args:
            model: The model to patch, or None where only the stack is read.
            clip: The text encoder to patch, or None to patch the model alone.
            fuse: Join the rows into one patch before applying, rather than one each.
            strength: Multiplier over every row's own strength, or a list of them, one per
                sampling step.
            rows: The row widgets the interface sends, read by ``rows_from_inputs``. A row's
                file may be a full path inside a LoRA folder, and its weight a list, one per
                sampling step.

        Returns:
            The patched model and CLIP, the LoRA stack, what was applied, and one name per row.

        Raises:
            ValueError: The model output is wired but nothing is connected to the model input,
                or a strength or weight list holds no number.
            PathNotAllowed: A row's file is a full path outside the LoRA folders.
        """
        import folder_paths

        from ....modules.lora import rows as lora_rows
        from ....modules.lora import schedule as lora_schedule
        from ....modules.lora import stack as lora_stack
        from ...legacy.loaders.lora_loader import lora_state_dict

        if model is None and lora_stack.consumed(cls.hidden.prompt, cls.hidden.unique_id, 0):
            require_input(model, "Power LoRA Loader", "model", "model", "checkpoint loader", "MODEL")

        scale = lora_schedule.strength_list(strength) or [float(strength)]
        selected = lora_rows.rows_from_inputs(rows)
        weights = [lora_rows.weight_list(row.weight) for row in selected]
        scheduled = len(set(scale)) > 1 or any(len(set(listed)) > 1 for listed in weights)
        start = scale[0]

        # Every row at step 1: its first weight times the first strength.
        stack = [
            (row.lora, listed[0] * start, listed[0] * start)
            for row, listed in zip(selected, weights)
            if listed[0] * start != 0.0
        ]
        shown = (
            [(row.lora, listed) for row, listed in zip(selected, weights)]
            if scheduled
            else [(name, [weight]) for name, weight, _ in stack]
        )
        if not shown:
            logger.debug("no lora row is switched on, the model passes through")
            return io.NodeOutput(model, clip, [], "", *name_slots([]))

        names = [os.path.splitext(os.path.basename(name))[0] for name, _ in shown]
        if scheduled:
            text = "\n".join(
                f"{name} per step, {listed[0] * start:g} at step 1" for name, (_, listed) in zip(names, shown)
            )
        else:
            text = "\n".join(f"{name} {listed[0]:g}" for name, (_, listed) in zip(names, shown))
        if model is None:
            return io.NodeOutput(model, clip, stack, text, *name_slots(names))

        states = {
            name: lora_state_dict(folder_paths.get_full_path_or_raise("loras", name))
            for name, _ in shown
        }
        model, clip = cls.apply(
            model, clip, [(states[name], weight) for name, weight, _ in stack], fuse,
            [os.path.splitext(os.path.basename(name))[0] for name, _, _ in stack],
        )
        if scheduled:
            model = lora_schedule.attach(model, [(states[name], listed) for name, listed in shown], scale)
            logger.debug("applied %d lora(s) with per-step strengths", len(shown))
        else:
            logger.debug("applied %d lora(s)", len(stack))
        return io.NodeOutput(model, clip, stack, text, *name_slots(names))

    @staticmethod
    def apply(model, clip, loaded: list, fuse: bool, names: list | None = None):
        """Patch a model and text encoder with LoRAs, each at its own strength.

        Args:
            model: The model to patch.
            clip: The text encoder to patch, or None.
            loaded: ``(state_dict, strength)`` per LoRA, in row order.
            fuse: Join the LoRAs into one patch before applying.
            names: One name per LoRA, for the message about parts that do not fit.

        Returns:
            The patched model and text encoder.
        """
        from ....modules.lora import fit

        names = list(names or [])
        names += [f"LoRA {index + 1}" for index in range(len(names), len(loaded))]
        if not loaded:
            return model, clip
        if fuse and len(loaded) > 1:
            from ....modules.lora.fuse import fuse_state_dicts

            combined, paired, carried = fuse_state_dicts(loaded)
            logger.debug("fused %d module(s), carried %d key(s)", paired, carried)
            # Strength 1.0: every row's own weight is already folded into the factors.
            return fit.load_lora_for_models(
                model, clip, combined, 1.0, 1.0 if clip is not None else 0.0, " + ".join(names)
            )
        for (state, weight), name in zip(loaded, names):
            # Each row patches the answer of the one before it, so the stack composes in the
            # order the rows are drawn rather than every row patching the original.
            model, clip = fit.load_lora_for_models(
                model, clip, state, weight, weight if clip is not None else 0.0, name
            )
        return model, clip

"""Reduce several booleans to one."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.logic.compare import REDUCTIONS, counted_conditions, reduce_booleans

#: Tooltip on every condition. One wording for all of them: the letter is already on the socket.
SLOT_TIP = (
    "One condition, `true` or `false`, wired from a test. An unwired slot counts only when "
    "ticked `true`; unticked, it is left out, so the reduction runs over what is wired."
)


class BooleanReduce(io.ComfyNode):
    """Combine any number of booleans into one."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASBooleanReduce",
            display_name="Boolean Reduce",
            search_aliases=[
                "WASBooleanReduce",
                "Boolean Reduce",
                "all",
                "any",
                "and or",
                "combine booleans",
            ],
            category="WAS Suite/Logic/Boolean",
            description=(
                "Combine any number of conditions into one answer: all of them, any of "
                "them, none of them, exactly one, or a majority. Logic Comparison AND and "
                "OR take two, so four conditions need three of them chained; this takes "
                "them all at once."
            ),
            inputs=[
                io.Combo.Input(
                    "reduction",
                    options=list(REDUCTIONS),
                    default="all",
                    tooltip=(
                        "How the conditions combine. With 3 wired and 2 true: `all` = false, "
                        "`any` = true, `none` = false, `exactly one` = false, `majority` = "
                        "true."
                    ),
                ),
                io.Boolean.Input(
                    "condition_a", default=False, optional=True, tooltip=SLOT_TIP,
                ),
                io.Boolean.Input(
                    "condition_b", default=False, optional=True, tooltip=SLOT_TIP,
                ),
                io.Boolean.Input(
                    "condition_c", default=False, optional=True, tooltip=SLOT_TIP,
                ),
                io.Boolean.Input(
                    "condition_d", default=False, optional=True, tooltip=SLOT_TIP,
                ),
                io.Boolean.Input(
                    "condition_e", default=False, optional=True, tooltip=SLOT_TIP,
                ),
                io.Boolean.Input(
                    "condition_f", default=False, optional=True, tooltip=SLOT_TIP,
                ),
                io.Boolean.Input(
                    "condition_g", default=False, optional=True, tooltip=SLOT_TIP,
                ),
                io.Boolean.Input(
                    "condition_h", default=False, optional=True, tooltip=SLOT_TIP,
                ),
                io.Boolean.Input(
                    "condition_i", default=False, optional=True, tooltip=SLOT_TIP,
                ),
                io.Boolean.Input(
                    "condition_j", default=False, optional=True, tooltip=SLOT_TIP,
                ),
                io.Boolean.Input(
                    "condition_k", default=False, optional=True, tooltip=SLOT_TIP,
                ),
                io.Boolean.Input(
                    "condition_l", default=False, optional=True, tooltip=SLOT_TIP,
                ),
                io.Boolean.Input(
                    "condition_m", default=False, optional=True, tooltip=SLOT_TIP,
                ),
                io.Boolean.Input(
                    "condition_n", default=False, optional=True, tooltip=SLOT_TIP,
                ),
                io.Boolean.Input(
                    "condition_o", default=False, optional=True, tooltip=SLOT_TIP,
                ),
                io.Boolean.Input(
                    "condition_p", default=False, optional=True, tooltip=SLOT_TIP,
                ),
                io.Boolean.Input(
                    "condition_q", default=False, optional=True, tooltip=SLOT_TIP,
                ),
                io.Boolean.Input(
                    "condition_r", default=False, optional=True, tooltip=SLOT_TIP,
                ),
                io.Boolean.Input(
                    "condition_s", default=False, optional=True, tooltip=SLOT_TIP,
                ),
                io.Boolean.Input(
                    "condition_t", default=False, optional=True, tooltip=SLOT_TIP,
                ),
                io.Boolean.Input(
                    "condition_u", default=False, optional=True, tooltip=SLOT_TIP,
                ),
                io.Boolean.Input(
                    "condition_v", default=False, optional=True, tooltip=SLOT_TIP,
                ),
                io.Boolean.Input(
                    "condition_w", default=False, optional=True, tooltip=SLOT_TIP,
                ),
                io.Boolean.Input(
                    "condition_x", default=False, optional=True, tooltip=SLOT_TIP,
                ),
                io.Boolean.Input(
                    "condition_y", default=False, optional=True, tooltip=SLOT_TIP,
                ),
                io.Boolean.Input(
                    "condition_z", default=False, optional=True, tooltip=SLOT_TIP,
                ),
            ],
            outputs=[
                io.Boolean.Output(
                    display_name="boolean",
                    tooltip="The combined answer. Wire it to any switch's boolean.",
                ),
                io.Int.Output(
                    display_name="true_count",
                    tooltip="How many connected conditions are true.",
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="Conditions counted: the wired ones, and any unwired slot ticked `true`.",
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, reduction="all", **conditions) -> io.NodeOutput:
        """Combine the wired conditions.

        Args:
            reduction: How they combine.
            conditions: The condition slots the graph wired.

        Returns:
            The combined answer, how many were true, and how many were counted.

        Raises:
            ValueError: The reduction is unknown.
        """
        values = [value for _, value in counted_conditions(conditions, cls.hidden.prompt, cls.hidden.unique_id)]
        return io.NodeOutput(reduce_booleans(values, reduction), sum(values), len(values))

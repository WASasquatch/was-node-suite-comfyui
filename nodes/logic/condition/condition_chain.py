"""Pick the first condition that holds, as a number a switch can read."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.logic.compare import MAX_CONDITIONS, counted_conditions

#: Tooltip on every condition. One wording for all of them: the letter is already on the socket.
SLOT_TIP = (
    "One condition, `true` or `false`, tested in slot order. The first `true` decides the "
    "answer. An unwired slot is skipped unless ticked `true`, so the numbering follows what is wired."
)


class ConditionChain(io.ComfyNode):
    """Answer the position of the first condition that holds."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASConditionChain",
            display_name="Condition Chain",
            search_aliases=[
                "WASConditionChain",
                "Condition Chain",
                "if else",
                "elseif",
                "case",
                "branch",
                "first true",
            ],
            category="WAS Suite/Logic/Boolean",
            description=(
                "Test conditions in order and answer the position of the first one that "
                "holds. Wire that into an index switch and the pair reads as if, else if, "
                "else: condition_a picks input_a, condition_b picks input_b, and nothing "
                "matching picks whichever slot the fallback names."
            ),
            inputs=[
                io.Int.Input(
                    "fallback",
                    default=0,
                    min=-1,
                    max=MAX_CONDITIONS - 1,
                    tooltip=(
                        "Answer when no condition holds; this is the `else`. -1 answers -1 "
                        "and sets matched to false, which an index switch reads as its last "
                        "slot."
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
                io.Int.Output(
                    display_name="index",
                    tooltip=(
                        "Position of the first true condition, counting connected slots from "
                        "0, or the fallback when none held."
                    ),
                ),
                io.Boolean.Output(
                    display_name="matched",
                    tooltip="true when a condition held, false when the fallback answered.",
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="Conditions counted: the wired ones, and any unwired slot ticked `true`. The index runs 0..count-1.",
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, fallback=0, **conditions) -> io.NodeOutput:
        """Answer the first condition that holds.

        Args:
            fallback: Answer when none holds.
            conditions: The condition slots the graph wired.

        Returns:
            The position, whether a condition held, and how many are connected.
        """
        counted = counted_conditions(conditions, cls.hidden.prompt, cls.hidden.unique_id)
        for position, (_, value) in enumerate(counted):
            if value:
                return io.NodeOutput(position, True, len(counted))
        return io.NodeOutput(int(fallback), False, len(counted))

"""The entry point of a condition-driven loop."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules.compat.types import DICT, WAS_LOOP
from ....modules.logic.loop_expand import MAX_ITERATIONS

#: Iterations a While Loop runs before stopping itself, unless its widget says fewer. Lower
#: than the hard ceiling, so a condition that never goes false stops after a hundred
#: iterations rather than ten thousand.
DEFAULT_MAX_ITERATIONS = 100

FIRST_IN_TIP = (
    "The value slot 1 starts with; any type. Read it back out of value_1 and hand the "
    "changed one to While Loop Close's value_1."
)
MORE_IN_TIP = (
    "The value slot {{n}} starts with; any type. A new slot appears as this one is wired."
)
FIRST_OUT_TIP = (
    "Slot 1 as this iteration receives it; any type. The starting value on iteration 1, then "
    "whatever While Loop Close was given last iteration."
)
MORE_OUT_TIP = (
    "Slot {{n}} as this iteration receives it; any type."
)


class WhileLoopOpen(io.ComfyNode):
    """Emit one iteration's carried values and counters."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        templates = [io.MatchType.Template(f"while_loop_value_{n}") for n in range(1, 9)]
        return io.Schema(
            node_id="WASWhileLoopOpen",
            display_name="While Loop Open",
            search_aliases=[
                'WASWhileLoopOpen',
                "While Loop Open", "While Loop Start", "while loop", "repeat until",
                "condition", "loop", "open loop", "accumulate",
            ],
            category="WAS Suite/Logic/Loop",
            description=(
                "Open a loop that runs until a condition says stop. The body runs at least "
                "once; While Loop Close reads the condition after each iteration."
            ),
            inputs=[
                io.Int.Input(
                    "max_iterations",
                    default=DEFAULT_MAX_ITERATIONS,
                    min=1,
                    max=MAX_ITERATIONS,
                    tooltip=(
                        "Safety limit; INT, 1 to 10000. Stops the loop even if the condition "
                        "stays true, so a mistake cannot run forever."
                    ),
                ),
                io.Int.Input(
                    "start",
                    default=0,
                    min=0,
                    max=MAX_ITERATIONS,
                    tooltip=(
                        "The index the first iteration reads, as `0`, or `2` to resume a run "
                        "at its third item; INT. Counts up by 1 from here."
                    ),
                ),
                io.MatchType.Input(
                    "value_1", template=templates[0], optional=True,
                    tooltip=FIRST_IN_TIP,
                ),
                io.MatchType.Input(
                    "value_2", template=templates[1], optional=True,
                    tooltip=MORE_IN_TIP.format(n=2),
                ),
                io.MatchType.Input(
                    "value_3", template=templates[2], optional=True,
                    tooltip=MORE_IN_TIP.format(n=3),
                ),
                io.MatchType.Input(
                    "value_4", template=templates[3], optional=True,
                    tooltip=MORE_IN_TIP.format(n=4),
                ),
                io.MatchType.Input(
                    "value_5", template=templates[4], optional=True,
                    tooltip=MORE_IN_TIP.format(n=5),
                ),
                io.MatchType.Input(
                    "value_6", template=templates[5], optional=True,
                    tooltip=MORE_IN_TIP.format(n=6),
                ),
                io.MatchType.Input(
                    "value_7", template=templates[6], optional=True,
                    tooltip=MORE_IN_TIP.format(n=7),
                ),
                io.MatchType.Input(
                    "value_8", template=templates[7], optional=True,
                    tooltip=MORE_IN_TIP.format(n=8),
                ),
            ],
            outputs=[
                WAS_LOOP.Output(
                    display_name="iterator",
                    tooltip=(
                        "Identifies the loop and where it is up to; WAS_LOOP. Wire it "
                        "straight to While Loop Close's iterator input, and nothing else."
                    ),
                ),
                io.Int.Output(
                    display_name="index",
                    tooltip=(
                        "The loop's counter; INT. Starts at start, 0 by default, and counts "
                        "up by 1 each iteration, for reading a position in a list."
                    ),
                ),
                DICT.Output(
                    display_name="metadata",
                    tooltip=(
                        "This iteration's counters as one value; DICT. Read them with "
                        "Loop Metadata: current_iteration, index, and the frames collected "
                        "so far."
                    ),
                ),
                io.MatchType.Output(
                    template=templates[0], display_name="value_1",
                    tooltip=FIRST_OUT_TIP,
                ),
                io.MatchType.Output(
                    template=templates[1], display_name="value_2",
                    tooltip=MORE_OUT_TIP.format(n=2),
                ),
                io.MatchType.Output(
                    template=templates[2], display_name="value_3",
                    tooltip=MORE_OUT_TIP.format(n=3),
                ),
                io.MatchType.Output(
                    template=templates[3], display_name="value_4",
                    tooltip=MORE_OUT_TIP.format(n=4),
                ),
                io.MatchType.Output(
                    template=templates[4], display_name="value_5",
                    tooltip=MORE_OUT_TIP.format(n=5),
                ),
                io.MatchType.Output(
                    template=templates[5], display_name="value_6",
                    tooltip=MORE_OUT_TIP.format(n=6),
                ),
                io.MatchType.Output(
                    template=templates[6], display_name="value_7",
                    tooltip=MORE_OUT_TIP.format(n=7),
                ),
                io.MatchType.Output(
                    template=templates[7], display_name="value_8",
                    tooltip=MORE_OUT_TIP.format(n=8),
                ),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def execute(
        cls,
        max_iterations=DEFAULT_MAX_ITERATIONS,
        start=0,
        value_1=None,
        value_2=None,
        value_3=None,
        value_4=None,
        value_5=None,
        value_6=None,
        value_7=None,
        value_8=None,
    ) -> io.NodeOutput:
        # A widget's `max` bounds only what can be typed into it; a value arriving on a wire is
        # not held to it, so every ceiling is enforced again here.
        from ....modules.logic import loop_meta

        first = max(0, min(int(start), MAX_ITERATIONS))
        iterator = {
            "start_id": str(cls.hidden.unique_id),
            "mode": "condition",
            "max_iterations": max(1, min(int(max_iterations), MAX_ITERATIONS)),
            "start_index": first,
            "index": first,
            "iteration": 1,
        }
        metadata = loop_meta.build(
            mode="condition",
            current_iteration=1,
            index=first,
            limit=iterator["max_iterations"],
        )
        return io.NodeOutput(
            iterator,
            first,
            metadata,
            value_1, value_2, value_3, value_4, value_5, value_6, value_7, value_8,
        )

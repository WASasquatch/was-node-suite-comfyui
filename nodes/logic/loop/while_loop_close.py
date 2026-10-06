"""The exit point of a condition-driven loop, and what re-runs its body."""

from __future__ import annotations

from comfy_api.latest import io, ui

from ....modules.compat.types import DICT, NUMBER, WAS_LOOP

NODE_NAME = "While Loop Close"

FIRST_IN_TIP = (
    "Slot 1's value at the end of this iteration; any type. It goes back to While Loop Open's "
    "value_1 for the next one."
)
MORE_IN_TIP = (
    "Slot {{n}}'s value at the end of this iteration; any type."
)
FIRST_OUT_TIP = (
    "Slot 1's result; any type. The last value it held, or every value it held once "
    "accumulate is on, batched where they batch and a LIST where they do not."
)
MORE_OUT_TIP = (
    "Slot {{n}}'s result; any type. Every value it held while accumulate is on, otherwise "
    "the last one."
)


class WhileLoopClose(io.ComfyNode):
    """Finish one iteration and expand into the next until the loop is done."""

    # A node with nothing wired after it inside the loop, a Save Image with no further use for
    # its output, runs once on the first iteration and never again; wire it through a spare
    # value slot to make it part of the loop instead.
    #
    # `metadata` is declared before the value slots, not after them. The canvas draws only the
    # slots in use, so an output declared past them sits at a lower index there than in the
    # schema, and a link records the drawn index: the backend would answer with whichever value
    # slot happens to hold that index instead.

    @classmethod
    def define_schema(cls) -> io.Schema:
        templates = [io.MatchType.Template(f"while_loop_close_value_{n}") for n in range(1, 9)]
        return io.Schema(
            node_id="WASWhileLoopClose",
            display_name="While Loop Close",
            search_aliases=[
                'WASWhileLoopClose',
                "While Loop Close", "While Loop End", "while loop", "repeat until",
                "condition", "loop", "close loop", "accumulate", "collect",
            ],
            category="WAS Suite/Logic/Loop",
            description=(
                "Finish one iteration of a While Loop and run another while continue_loop is "
                "true, up to max_iterations. Only nodes wired back to here run again."
            ),
            inputs=[
                WAS_LOOP.Input(
                    "iterator",
                    tooltip=(
                        "Identifies the loop and where it is up to; WAS_LOOP. Wired "
                        "straight from While Loop Open's iterator output, and nothing else."
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
                io.MultiType.Input(
                    io.Boolean.Input("continue_loop", default=False),
                    [io.Boolean, NUMBER, io.Int, io.Float],
                    tooltip=(
                        "true runs another iteration, false stops; BOOLEAN, or a NUMBER where "
                        "any non-zero value means true. Read after each iteration, so the body "
                        "always runs at least once. Wire it from Logic Compare Numbers, Text "
                        "Compare, or any Logic node."
                    ),
                ),
                io.Boolean.Input(
                    "accumulate",
                    default=False,
                    tooltip=(
                        "Collect every iteration's values; BOOLEAN. On, each value output "
                        "carries everything that slot received, batched where images, masks or "
                        "latents batch and a LIST where they do not."
                    ),
                ),
            ],
            outputs=[
                DICT.Output(
                    display_name="metadata",
                    tooltip=(
                        "What the finished loop did, as one value; DICT. Read it with Loop "
                        "Metadata: iterations completed, frames collected, and why it "
                        "stopped."
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
            hidden=[io.Hidden.unique_id, io.Hidden.dynprompt],
            enable_expand=True,
            is_output_node=True,
        )

    @classmethod
    def execute(
        cls,
        iterator,
        accumulate=False,
        continue_loop=False,
        value_1=None,
        value_2=None,
        value_3=None,
        value_4=None,
        value_5=None,
        value_6=None,
        value_7=None,
        value_8=None,
    ) -> io.NodeOutput:
        """Collect this iteration, then either finish the loop or expand into the next.

        Raises:
            ValueError: The loop asks for a frame target with accumulate off, so there is
                nothing to count it against.
        """
        from ....modules.logic import loop_accumulate, loop_expand, loop_meta, loop_readout

        completed = int(iterator.get("iteration", 1))
        slots = [
            ("value_1", value_1),
            ("value_2", value_2),
            ("value_3", value_3),
            ("value_4", value_4),
            ("value_5", value_5),
            ("value_6", value_6),
            ("value_7", value_7),
            ("value_8", value_8),
        ]
        names = [name for name, _ in slots]
        max_iterations = int(iterator["max_iterations"])
        limit = max_iterations
        # A NUMBER arrives as an int or a float, where any non-zero value is true, which is what
        # `bool` already means for both. The socket accepts both types so a numeric comparison
        # can drive the loop without a converter node in between.
        keep_going = bool(continue_loop)

        accumulated = iterator.get("accumulated")
        if accumulate:
            for name, value in slots:
                if value is None:
                    continue
                accumulated = loop_accumulate.append(accumulated, name, value)
        frames = loop_accumulate.total_count(accumulated, names)

        # Filed under the original Close node so every iteration lands on the node the user can
        # see, rather than on the clone that happens to be running this iteration.
        readout_id = iterator.get("end_id") or str(cls.hidden.unique_id)
        counts = {"iteration": completed, "limit": max_iterations}
        if accumulate:
            counts["frames"] = frames

        if not keep_going:
            reason = f"continue_loop was false after {completed} iteration(s)."
        elif completed >= max_iterations:
            reason = (
                f"Stopped at the max_iterations limit of {max_iterations}; continue_loop was "
                f"still true."
            )
        else:
            reason = None

        if reason is not None:
            results = []
            kinds = set()
            for name, value in slots:
                result, kind = loop_accumulate.finalize(
                    loop_accumulate.collected(accumulated, name), accumulate, value,
                )
                results.append(result)
                if value is not None or result is not None:
                    kinds.add(kind)
            metadata = loop_meta.build(
                mode=iterator.get("mode", "iterations"),
                current_iteration=completed,
                index=int(iterator.get("index", 0)),
                iterations_completed=completed,
                limit=limit,
                accumulated_count=frames,
                accumulated_as=_one_kind(kinds),
                stopped_reason=reason,
            )
            loop_readout.publish_iteration(readout_id, reason, counts, slots)
            return io.NodeOutput(metadata, *results, ui=ui.PreviewText(reason))

        # The first iteration runs as the real node, so its own id is the stable one every later
        # iteration needs; a later iteration already carries it in the flow token and passes it
        # on rather than substituting its own ephemeral id.
        end_id = iterator.get("end_id") or str(cls.hidden.unique_id)
        next_flow = {
            "start_id": iterator["start_id"],
            "end_id": end_id,
            "mode": "condition",
            "max_iterations": max_iterations,
            "start_index": int(iterator.get("start_index", 0)),
            "index": int(iterator.get("start_index", 0)) + completed,
            "iteration": completed + 1,
            "accumulated": accumulated,
        }
        next_metadata = loop_meta.build(
            mode=iterator.get("mode", "iterations"),
            current_iteration=completed + 1,
            index=next_flow["index"],
            iterations_completed=completed,
            limit=limit,
            accumulated_count=frames,
            accumulated_as=loop_meta.FINAL,
        )
        # Keyed by the slot While Loop Open declares each output at: iterator, index,
        # metadata, then value_1 onward.
        next_values = {
            0: next_flow,
            1: next_flow["index"],
            2: next_metadata,
            3: value_1,
            4: value_2,
            5: value_3,
            6: value_4,
            7: value_5,
            8: value_6,
            9: value_7,
            10: value_8,
        }

        running = f"Running iteration {completed}, limit {max_iterations}."
        loop_readout.publish_iteration(readout_id, running, counts, slots)

        graph, end_node = loop_expand.clone_iteration(
            cls.hidden.dynprompt, iterator["start_id"], end_id, next_values,
            "WASWhileLoopClose",
        )
        return io.NodeOutput(
            end_node.out(0),
            end_node.out(1),
            end_node.out(2),
            end_node.out(3),
            end_node.out(4),
            end_node.out(5),
            end_node.out(6),
            end_node.out(7),
            end_node.out(8),
            expand=graph,
            ui=ui.PreviewText(running),
        )


def _one_kind(kinds: set) -> str:
    """How the slots left the loop, when they all left it the same way.

    Args:
        kinds: The kind each wired slot reported.

    Returns:
        The single kind, or ``loop_meta.LIST`` where slots differ, since a mixed report is
        better read as "not one batch" than as whichever slot happened to come first.
    """
    from ....modules.logic import loop_meta

    if len(kinds) == 1:
        return next(iter(kinds))
    if not kinds:
        return loop_meta.FINAL
    return loop_meta.LIST

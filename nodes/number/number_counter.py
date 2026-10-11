"""A per-node counter that advances once per prompt."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import NUMBER

#: ``node id -> its current count``, for the lifetime of the process. v2 kept this on the
#: node instance ComfyUI caches per graph node, so a count lasted as long as the server ran
#: and started over on the next one. ``execute`` runs on a per-execution clone of the class,
#: which is discarded afterwards and cannot hold it, and a module attribute is what has the
#: same lifetime. Nothing is written to disk: a count keyed on a graph node id would
#: otherwise outlive the workflow that made it and be picked up by whatever numbered a node
#: the same way next.
_counters: dict[str, float] = {}


class NumberCounter(io.ComfyNode):
    """Step a stored counter and emit it.

    The count is held for as long as the server runs.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Number Counter",
            display_name="Number Counter",
            search_aliases=["Number Counter", "counter", "increment", "decrement", "index"],
            category="WAS Suite/Number",
            description=(
                "Emit a number that moves on every prompt, to number a batch or sweep a "
                "setting over a run of images. Each copy counts on its own, can freeze or "
                "loop at a limit, and starts over when ComfyUI restarts."
            ),
            inputs=[
                io.Combo.Input(
                    "number_type",
                    options=["integer", "float"],
                    tooltip=(
                        "Whether the count is kept whole. `integer` begins at a whole start "
                        "and emits whole numbers; `float` allows fractions, so a step of "
                        "0.25 counts 0.25, 0.5, 0.75."
                    ),
                ),
                io.Combo.Input(
                    "mode",
                    options=[
                        "increment",
                        "decrement",
                        "increment_to_stop",
                        "decrement_to_stop",
                        "reset_after_stop",
                    ],
                    tooltip=(
                        "`increment`/`decrement` = run on forever; '_to_stop' modes = freeze "
                        "on the first value reaching stop, so start 0, step 3, stop 10 holds "
                        "at 12; `reset_after_stop` = jumps back to start + step."
                    ),
                ),
                io.Float.Input(
                    "start",
                    default=0,
                    min=-18446744073709551615,
                    max=18446744073709551615,
                    step=0.01,
                    tooltip=(
                        "Where the count begins, and where a reset sends it back to. The "
                        "first prompt already applies one step, so an incrementing counter "
                        "from a start of 0 with a step of 1 first emits 1 rather than 0."
                    ),
                ),
                io.Float.Input(
                    "stop",
                    default=0,
                    min=-18446744073709551615,
                    max=18446744073709551615,
                    step=0.01,
                    tooltip=(
                        "The limit the '_to_stop' and `reset_after_stop` modes watch for; "
                        "`increment` and `decrement` ignore it. Left at 0, an "
                        "`increment_to_stop` counter is already at its limit and never "
                        "moves."
                    ),
                ),
                io.Float.Input(
                    "step",
                    default=1,
                    min=0,
                    max=99999,
                    step=0.01,
                    tooltip=(
                        "How far the count moves each prompt. 1 counts 1, 2, 3; 10 counts "
                        "10, 20, 30; 0 holds the count still. Always positive, `decrement` "
                        "is what subtracts it."
                    ),
                ),
                io.MultiType.Input(
                    "reset_bool",
                    [NUMBER, io.Int, io.Float],
                    optional=True,
                    tooltip=(
                        "Send 1 or more here to put the count back to start before this "
                        "prompt's step; 0, or nothing connected, leaves it running. The "
                        "value is rounded first, so 0.6 also resets. The NUMBER output of "
                        "Logic Boolean fits this socket."
                    ),
                ),
            ],
            outputs=[
                NUMBER.Output(
                    display_name="number",
                    tooltip=(
                        "The count after this prompt's step, whole when number_type is "
                        "`integer`."
                    ),
                ),
                io.Float.Output(
                    display_name="float",
                    tooltip=(
                        "The count as a float. This one keeps a fraction even when "
                        "number_type is `integer`, which a fractional step can produce."
                    ),
                ),
                io.Int.Output(
                    display_name="int",
                    tooltip=(
                        "The count as a whole number, cut off rather than rounded, so 2.9 "
                        "leaves here as 2."
                    ),
                ),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        """NaN, so the counter advances on every prompt rather than serving a cached step."""
        return float("NaN")

    @classmethod
    def execute(cls, number_type, mode, start, stop, step, reset_bool=0) -> io.NodeOutput:
        key = str(cls.hidden.unique_id)

        counter = int(start) if number_type == "integer" else start
        if key in _counters:
            counter = _counters[key]

        if round(reset_bool) >= 1:
            counter = start

        if mode == "increment":
            counter += step
        elif mode == "decrement":
            counter -= step
        elif mode == "increment_to_stop":
            counter = counter + step if counter < stop else counter
        elif mode == "decrement_to_stop":
            counter = counter - step if counter > stop else counter
        elif mode == "reset_after_stop":
            counter = counter + step if counter < stop else start + step

        _counters[key] = counter

        result = int(counter) if number_type == "integer" else float(counter)
        return io.NodeOutput(result, float(counter), int(counter))

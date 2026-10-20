"""Build a list of numbers from a range or typed values, shaped by a curve and looped."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import LIST, NUMBER
from ...modules.number import sequence
from ...modules.util.numbers import as_number, split_values, whole

#: Display name, written into the messages this node raises.
NAME = "Number Sequence"

#: The key the node's run output carries its values under, read by its interface.
UI_KEY = "was_number_sequence"


class NumberSequence(io.ComfyNode):
    """Build a list of numbers through keyframes, shaped by a curve, repeated and looped."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASNumberSequence",
            display_name="Number Sequence",
            search_aliases=[
                "WASNumberSequence",
                "Number Sequence",
                "schedule",
                "strength schedule",
                "float list",
                "list of floats",
                "number list",
                "keyframes",
                "curve",
                "ramp",
                "fade",
                "loop",
                "ping pong",
                "wave",
                "pulse",
                "per step",
            ],
            category="WAS Suite/Number/Operations",
            description=(
                "Build a list of numbers from a start and end or from typed values, shaped by a "
                "curve, repeated and looped, and drawn on the node where its points can be "
                "dragged. The values output is a per-step strength for Power LoRA Loader; the "
                "other outputs run the graph once per number."
            ),
            inputs=[
                io.Combo.Input(
                    "mode",
                    options=["range", "values"],
                    tooltip=(
                        "`range` = from start to end; `values` = through the numbers typed in "
                        "values, evenly spaced, as `1, 1, 0` for a strength that holds and "
                        "then fades."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input("start", default=1.0, min=-1e9, max=1e9, step=0.01),
                    [io.Float, NUMBER, io.Int],
                    tooltip="The first value in `range` mode, as `1.0` for a strength that starts full.",
                ),
                io.MultiType.Input(
                    io.Float.Input("end", default=0.0, min=-1e9, max=1e9, step=0.01),
                    [io.Float, NUMBER, io.Int],
                    tooltip=(
                        "The last value in `range` mode. Below start counts down, as `0.0` "
                        "for a fade-out."
                    ),
                ),
                io.MultiType.Input(
                    io.String.Input(
                        "values",
                        default="1, 1, 0",
                        multiline=True,
                        placeholder="1, 1, 0.5, 0 (commas, spaces or one to a line)",
                    ),
                    [io.String, LIST, NUMBER, io.Float, io.Int],
                    tooltip=(
                        "The values `values` mode passes through, separated by commas, spaces "
                        "or lines, as `1, 1, 0.5, 0`. A LIST can be wired in instead."
                    ),
                ),
                io.Int.Input(
                    "count",
                    default=20,
                    min=0,
                    max=sequence.MAX_COUNT,
                    step=1,
                    tooltip=(
                        "Entries in the list; set it to the sampler's steps for a per-step "
                        "strength. 0 = one entry per value: start and end, or each typed "
                        "value."
                    ),
                ),
                io.Combo.Input(
                    "curve",
                    options=list(sequence.CURVE_NAMES),
                    tooltip=(
                        "`linear` = straight; `ease_in_*` = starts slow; `ease_out_*` = "
                        "finishes slow; `ease_in_out_*` = both; `*_back`, `*_elastic` = "
                        "overshoot; `*_bounce` = settles in hops; `hold` = each value kept "
                        "until the next."
                    ),
                ),
                io.Combo.Input(
                    "curve_on",
                    options=list(sequence.CURVE_TARGETS),
                    tooltip=(
                        "`transitions` = each move from one value to the next follows the "
                        "curve; `values` = the values themselves are bent between the lowest "
                        "and highest, which also shapes a list typed one entry per step."
                    ),
                ),
                io.Int.Input(
                    "repeat",
                    default=1,
                    min=1,
                    max=sequence.MAX_REPEAT,
                    step=1,
                    tooltip=(
                        "How many times the shape plays across the list, as `4` for four "
                        "pulses. The entries are shared out, so 20 over 4 repeats is 5 each."
                    ),
                ),
                io.Combo.Input(
                    "loop",
                    options=list(sequence.LOOPS),
                    tooltip=(
                        "How each repeat follows the last. `restart` = from the first value "
                        "again, a sawtooth; `ping_pong` = backwards every other time, a wave; "
                        "`continue` = on from where the last ended, a staircase."
                    ),
                ),
                io.Boolean.Input(
                    "reverse",
                    default=False,
                    tooltip="`true` = the finished list runs back to front, so a fade-in becomes a fade-out.",
                ),
                io.Int.Input(
                    "decimals",
                    default=6,
                    min=0,
                    max=12,
                    step=1,
                    tooltip=(
                        "Decimal places each value is rounded to, which keeps "
                        "0.30000000000000004 out of a filename. 0 = whole numbers."
                    ),
                ),
            ],
            outputs=[
                LIST.Output(
                    display_name="values",
                    tooltip=(
                        "The whole list on one wire: a per-step strength for Power LoRA "
                        "Loader, or a list for Number List Plot and the other list nodes."
                    ),
                ),
                NUMBER.Output(
                    is_output_list=True,
                    tooltip=(
                        "One NUMBER per entry. A node reading it runs once for each, for a "
                        "series of renders."
                    ),
                ),
                io.Float.Output(
                    display_name="floats",
                    is_output_list=True,
                    tooltip="The same entries as decimals, one per run.",
                ),
                io.Int.Output(
                    display_name="ints",
                    is_output_list=True,
                    tooltip=(
                        "The same entries as whole numbers, one per run, cut off rather than "
                        "rounded, so 7.6 arrives as 7."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many entries the list holds, which is how many runs the per-run outputs drive.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        mode="range",
        start=1.0,
        end=0.0,
        values="1, 1, 0",
        count=20,
        curve="linear",
        curve_on="transitions",
        repeat=1,
        loop="restart",
        reverse=False,
        decimals=6,
    ) -> io.NodeOutput:
        """Build the list.

        Returns:
            The list on one wire, the same entries one per run as NUMBER, FLOAT and INT, and
            how many there are.

        Raises:
            ValueError: A typed value is not a number, there are no values, or the settings
                ask for more entries than the node makes.
        """
        from ...modules.interface.channel import executing_workflow_id

        keys = cls.keyframes(mode, start, end, values)
        count = int(count)
        repeat = int(repeat)
        length = sequence.length_of(len(keys), count, repeat, loop)
        if length > sequence.MAX_COUNT:
            raise ValueError(
                f"{NAME} would hold {length:,} entries ({len(keys)} values over {repeat} "
                f"repeat(s)), and holds at most {sequence.MAX_COUNT:,}. Set count above 0, or "
                f"lower repeat."
            )
        if loop == "restart" and length < repeat:
            raise ValueError(
                f"{NAME} shares {length} entr{'y' if length == 1 else 'ies'} between {repeat} "
                f"repeats, which leaves some repeats empty. Raise count to {repeat} or more, "
                f"or lower repeat."
            )

        built = sequence.build(keys, count, curve, curve_on, repeat, loop, bool(reverse))
        rounded = [round(value, int(decimals)) for value in built]
        numbers = [whole(value) for value in rounded]
        report = {"workflow": executing_workflow_id() or "", "values": list(rounded)}
        # A list of its own per slot, so a node that edits the one it was handed does not
        # change what the other slots emit.
        return io.NodeOutput(
            rounded,
            list(rounded),
            list(rounded),
            numbers,
            len(rounded),
            ui={UI_KEY: [report]},
        )

    @staticmethod
    def keyframes(mode: str, start, end, values) -> list[float]:
        """The values one cycle passes through.

        Args:
            mode: ``range`` for start and end, ``values`` for the typed or wired values.
            start: First value in ``range`` mode.
            end: Last value in ``range`` mode.
            values: Text or a list, read in ``values`` mode.

        Returns:
            At least one finite number.

        Raises:
            ValueError: An entry is not a number, there are none, or there are more than
                :data:`~modules.number.sequence.MAX_COUNT`.
        """
        if mode != "values":
            return [float(start), float(end)]
        entries = split_values(values)
        if not entries:
            raise ValueError(
                f"{NAME} is in `values` mode and its values box is empty. Type the values it "
                f"passes through, as `1, 1, 0`, or switch mode to `range`."
            )
        if len(entries) > sequence.MAX_COUNT:
            raise ValueError(
                f"{NAME} was given {len(entries):,} values and passes through at most "
                f"{sequence.MAX_COUNT:,}. Shorten the list."
            )
        keys = []
        for position, entry in enumerate(entries, 1):
            number = as_number(entry)
            if number is None:
                raise ValueError(
                    f"{NAME} could not read value {position}, {str(entry)[:40]!r}, as a "
                    f"number. Correct it in the values box, as `0.5`."
                )
            keys.append(number)
        return keys

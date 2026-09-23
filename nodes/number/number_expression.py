"""Work out a written expression over a set of numbers."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import NUMBER
from ...modules.number.expression import ON_ERROR, ExpressionError, as_int, evaluate

logger = log.get_logger("nodes.number")

#: The names a formula may use, one per numeric input.
SLOTS = tuple("abcdefghijklmnopqrstuvwx")


class NumberExpression(io.ComfyNode):
    """Evaluate an arithmetic expression over its numeric inputs."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASNumberExpression",
            display_name="Number Expression",
            search_aliases=[
                "WASNumberExpression",
                "Number Expression",
                "math",
                "formula",
                "calculator",
                "clamp",
                "round",
                "floor",
                "ceil",
                "abs",
                "min",
                "max",
                "sqrt",
                "lerp",
            ],
            category="WAS Suite/Number/Operations",
            description=(
                "Work out a whole formula over up to 24 numbers in one node, such as "
                "`(a * b) / 2 + c`, `clamp(a, 0, 1)` or `round(a / b, 2)`. Rounding, "
                "clamping, interpolation, roots, logarithms and trigonometry are all there, "
                "with pi, e and tau as constants, and the expression box lists them. "
                "Comparisons and `and`, `or` work too, so `a if a > b else b` picks the "
                "larger and the boolean output carries the answer. Only arithmetic is read: "
                "a name, an attribute or a call that is not on the list is refused by name "
                "before anything runs. The box takes several lines, joined into one, and "
                "`#` starts a comment. Slots a to x are sockets taking a whole number, a "
                "decimal, a NUMBER or a true or false as 1 or 0; unwired counts as 0, so a "
                "fixed number goes in the formula."
            ),
            inputs=[
                io.String.Input(
                    "expression",
                    default="a + b",
                    multiline=True,
                    placeholder="Eg: (a * b) / 2 + c",
                    tooltip=(
                        "The formula, over `a` to `x`. Eg: `(a * b) / 2 + c`. Functions: "
                        "min max abs round floor ceil sqrt clamp lerp sign log log2 log10 exp "
                        "sin cos tan atan2 hypot degrees radians, plus pi, e and tau. `a > b` "
                        "comes out as 1 or 0; `#` starts a comment."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input("a", optional=True,
                                   force_input=True),
                    [io.Float, NUMBER, io.Int, io.Boolean],
                    optional=True,
                    tooltip=(
                        "Wire the number `a` stands for, as `INT`, `FLOAT`, `NUMBER` or `BOOLEAN`. An "
                        "unwired slot counts as `0`, `true` counts as `1`, and a slot the "
                        "expression never names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input("b", optional=True,
                                   force_input=True),
                    [io.Float, NUMBER, io.Int, io.Boolean],
                    optional=True,
                    tooltip=(
                        "Wire the number `b` stands for. `a / b` with nothing wired to `b` divides "
                        "by `0` and stops the run unless on_error is set to zero."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input("c", optional=True,
                                   force_input=True),
                    [io.Float, NUMBER, io.Int, io.Boolean],
                    optional=True,
                    tooltip=(
                        "Wire the number `c` stands for. Handy as the offset in `(a * b) / 2 + c`."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input("d", optional=True,
                                   force_input=True),
                    [io.Float, NUMBER, io.Int, io.Boolean],
                    optional=True,
                    tooltip=(
                        "Wire the number `d` stands for. The fourth value, free for a limit such "
                        "as `clamp(a, c, d)`."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input("e", optional=True,
                                   force_input=True),
                    [io.Float, NUMBER, io.Int, io.Boolean],
                    optional=True,
                    tooltip=(
                        "Wire the number `e` stands for, as `INT`, `FLOAT`, `NUMBER` or "
                        "`BOOLEAN`. An unwired slot counts as `0`, and a slot the expression "
                        "never names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input("f", optional=True,
                                   force_input=True),
                    [io.Float, NUMBER, io.Int, io.Boolean],
                    optional=True,
                    tooltip=(
                        "Wire the number `f` stands for, as `INT`, `FLOAT`, `NUMBER` or "
                        "`BOOLEAN`. An unwired slot counts as `0`, and a slot the expression "
                        "never names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input("g", optional=True,
                                   force_input=True),
                    [io.Float, NUMBER, io.Int, io.Boolean],
                    optional=True,
                    tooltip=(
                        "Wire the number `g` stands for, as `INT`, `FLOAT`, `NUMBER` or "
                        "`BOOLEAN`. An unwired slot counts as `0`, and a slot the expression "
                        "never names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input("h", optional=True,
                                   force_input=True),
                    [io.Float, NUMBER, io.Int, io.Boolean],
                    optional=True,
                    tooltip=(
                        "Wire the number `h` stands for, as `INT`, `FLOAT`, `NUMBER` or "
                        "`BOOLEAN`. An unwired slot counts as `0`, and a slot the expression "
                        "never names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input("i", optional=True,
                                   force_input=True),
                    [io.Float, NUMBER, io.Int, io.Boolean],
                    optional=True,
                    tooltip=(
                        "Wire the number `i` stands for, as `INT`, `FLOAT`, `NUMBER` or "
                        "`BOOLEAN`. An unwired slot counts as `0`, and a slot the expression "
                        "never names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input("j", optional=True,
                                   force_input=True),
                    [io.Float, NUMBER, io.Int, io.Boolean],
                    optional=True,
                    tooltip=(
                        "Wire the number `j` stands for, as `INT`, `FLOAT`, `NUMBER` or "
                        "`BOOLEAN`. An unwired slot counts as `0`, and a slot the expression "
                        "never names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input("k", optional=True,
                                   force_input=True),
                    [io.Float, NUMBER, io.Int, io.Boolean],
                    optional=True,
                    tooltip=(
                        "Wire the number `k` stands for, as `INT`, `FLOAT`, `NUMBER` or "
                        "`BOOLEAN`. An unwired slot counts as `0`, and a slot the expression "
                        "never names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input("l", optional=True,
                                   force_input=True),
                    [io.Float, NUMBER, io.Int, io.Boolean],
                    optional=True,
                    tooltip=(
                        "Wire the number `l` stands for, as `INT`, `FLOAT`, `NUMBER` or "
                        "`BOOLEAN`. An unwired slot counts as `0`, and a slot the expression "
                        "never names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input("m", optional=True,
                                   force_input=True),
                    [io.Float, NUMBER, io.Int, io.Boolean],
                    optional=True,
                    tooltip=(
                        "Wire the number `m` stands for, as `INT`, `FLOAT`, `NUMBER` or "
                        "`BOOLEAN`. An unwired slot counts as `0`, and a slot the expression "
                        "never names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input("n", optional=True,
                                   force_input=True),
                    [io.Float, NUMBER, io.Int, io.Boolean],
                    optional=True,
                    tooltip=(
                        "Wire the number `n` stands for, as `INT`, `FLOAT`, `NUMBER` or "
                        "`BOOLEAN`. An unwired slot counts as `0`, and a slot the expression "
                        "never names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input("o", optional=True,
                                   force_input=True),
                    [io.Float, NUMBER, io.Int, io.Boolean],
                    optional=True,
                    tooltip=(
                        "Wire the number `o` stands for, as `INT`, `FLOAT`, `NUMBER` or "
                        "`BOOLEAN`. An unwired slot counts as `0`, and a slot the expression "
                        "never names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input("p", optional=True,
                                   force_input=True),
                    [io.Float, NUMBER, io.Int, io.Boolean],
                    optional=True,
                    tooltip=(
                        "Wire the number `p` stands for, as `INT`, `FLOAT`, `NUMBER` or "
                        "`BOOLEAN`. An unwired slot counts as `0`, and a slot the expression "
                        "never names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input("q", optional=True,
                                   force_input=True),
                    [io.Float, NUMBER, io.Int, io.Boolean],
                    optional=True,
                    tooltip=(
                        "Wire the number `q` stands for, as `INT`, `FLOAT`, `NUMBER` or "
                        "`BOOLEAN`. An unwired slot counts as `0`, and a slot the expression "
                        "never names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input("r", optional=True,
                                   force_input=True),
                    [io.Float, NUMBER, io.Int, io.Boolean],
                    optional=True,
                    tooltip=(
                        "Wire the number `r` stands for, as `INT`, `FLOAT`, `NUMBER` or "
                        "`BOOLEAN`. An unwired slot counts as `0`, and a slot the expression "
                        "never names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input("s", optional=True,
                                   force_input=True),
                    [io.Float, NUMBER, io.Int, io.Boolean],
                    optional=True,
                    tooltip=(
                        "Wire the number `s` stands for, as `INT`, `FLOAT`, `NUMBER` or "
                        "`BOOLEAN`. An unwired slot counts as `0`, and a slot the expression "
                        "never names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input("t", optional=True,
                                   force_input=True),
                    [io.Float, NUMBER, io.Int, io.Boolean],
                    optional=True,
                    tooltip=(
                        "Wire the number `t` stands for, as `INT`, `FLOAT`, `NUMBER` or "
                        "`BOOLEAN`. An unwired slot counts as `0`, and a slot the expression "
                        "never names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input("u", optional=True,
                                   force_input=True),
                    [io.Float, NUMBER, io.Int, io.Boolean],
                    optional=True,
                    tooltip=(
                        "Wire the number `u` stands for, as `INT`, `FLOAT`, `NUMBER` or "
                        "`BOOLEAN`. An unwired slot counts as `0`, and a slot the expression "
                        "never names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input("v", optional=True,
                                   force_input=True),
                    [io.Float, NUMBER, io.Int, io.Boolean],
                    optional=True,
                    tooltip=(
                        "Wire the number `v` stands for, as `INT`, `FLOAT`, `NUMBER` or "
                        "`BOOLEAN`. An unwired slot counts as `0`, and a slot the expression "
                        "never names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input("w", optional=True,
                                   force_input=True),
                    [io.Float, NUMBER, io.Int, io.Boolean],
                    optional=True,
                    tooltip=(
                        "Wire the number `w` stands for, as `INT`, `FLOAT`, `NUMBER` or "
                        "`BOOLEAN`. An unwired slot counts as `0`, and a slot the expression "
                        "never names is ignored."
                    ),
                ),
                io.MultiType.Input(
                    io.Float.Input("x", optional=True,
                                   force_input=True),
                    [io.Float, NUMBER, io.Int, io.Boolean],
                    optional=True,
                    tooltip=(
                        "Wire the number `x` stands for, as `INT`, `FLOAT`, `NUMBER` or "
                        "`BOOLEAN`. An unwired slot counts as `0`, and a slot the expression "
                        "never names is ignored."
                    ),
                ),
                io.Int.Input(
                    "decimals",
                    default=6,
                    min=0,
                    max=15,
                    step=1,
                    optional=True,
                    tooltip=(
                        "Decimal places a fractional answer is rounded to. 6 = 0.333333, "
                        "2 = 0.33, 0 = whole, so 3.7 comes out 4.0. It also clears the trailing "
                        "0.0000000001 that decimal arithmetic leaves behind. A whole answer is "
                        "untouched."
                    ),
                ),
                io.Combo.Input(
                    "on_error",
                    options=list(ON_ERROR),
                    default="error",
                    optional=True,
                    tooltip=(
                        "What a refused or impossible expression does. `error` = stop the run "
                        "and name the cause, `zero` = log it and answer 0. Pick `zero` where a "
                        "division by zero is expected on some frames of a batch."
                    ),
                ),
            ],
            outputs=[
                NUMBER.Output(
                    display_name="number",
                    tooltip=(
                        "The answer on the NUMBER wire, whole where it came out whole. A "
                        "comparison answers 1 or 0."
                    ),
                ),
                io.Float.Output(
                    display_name="float",
                    tooltip="The same answer as a decimal, so 7 leaves here as 7.0.",
                ),
                io.Int.Output(
                    display_name="int",
                    tooltip=(
                        "The same answer with its fraction cut off rather than rounded, so 3.9 "
                        "leaves here as 3. Held to the range a whole-number socket carries."
                    ),
                ),
                io.Boolean.Output(
                    display_name="boolean",
                    tooltip=(
                        "false when the answer is 0, true for anything else. Wire it to a "
                        "switch to branch on `a > b`."
                    ),
                ),
                io.String.Output(
                    display_name="text",
                    tooltip=(
                        "The answer written out, as `4.5` or `7`. Feed it to a filename prefix "
                        "or a text join."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        expression="a + b",
        decimals=6,
        on_error="error",
        **extra,
    ) -> io.NodeOutput:
        """Work out the expression and answer it in five forms.

        Args:
            expression: The formula, over ``a`` to ``x``.
            decimals: Decimal places a fractional answer is rounded to.
            on_error: ``error`` to stop the run, ``zero`` to answer 0.
            extra: The numeric slots ``a`` to ``x``, whatever the expression names.

        Returns:
            The answer as a number, a float, an int, a boolean and text.

        Raises:
            ExpressionError: The expression was refused or could not be worked out, and
                on_error is ``error``.
        """
        slots = {}
        for name in SLOTS:
            given = extra.get(name, 0.0)
            slots[name] = int(given) if isinstance(given, bool) else given

        try:
            value = evaluate(expression, slots)
        except ExpressionError as refused:
            if on_error != "zero":
                raise
            logger.warning("Number Expression answered 0 instead: %s", refused)
            value = 0

        if isinstance(value, bool):
            value = int(value)
        elif isinstance(value, float):
            value = round(value, int(decimals))

        return io.NodeOutput(value, float(value), as_int(value), value != 0, str(value))

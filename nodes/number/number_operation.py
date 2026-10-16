"""Arithmetic and comparison between two NUMBERs."""

from __future__ import annotations

import operator

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import NUMBER
from ...modules.number.expression import carried, power

logger = log.get_logger("nodes.number")

#: Operation name -> the callable that computes it. Comparisons emit ``int`` rather than
#: ``bool``: the result travels on a NUMBER wire, where ``True`` would render as "True"
#: in every downstream text node.
OPERATIONS = {
    "addition": operator.add,
    "subtraction": operator.sub,
    "division": operator.truediv,
    "floor division": operator.floordiv,
    "multiplication": operator.mul,
    "exponentiation": power,
    "modulus": operator.mod,
    "greater-than": lambda a, b: int(a > b),
    "greater-than or equals": lambda a, b: int(a >= b),
    "less-than": lambda a, b: int(a < b),
    "less-than or equals": lambda a, b: int(a <= b),
    "equals": lambda a, b: int(a == b),
    "does not equal": lambda a, b: int(a != b),
}


class NumberOperation(io.ComfyNode):
    """Apply one arithmetic or comparison operation to two numbers.

    Comparisons emit 1 or 0. Division and modulus by zero raise, as they do in Python.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Number Operation",
            display_name="Number Operation",
            search_aliases=["Number Operation", "math", "arithmetic", "compare"],
            category="WAS Suite/Number/Operations",
            description=(
                "Combine two numbers with one operation. The seven arithmetic operations "
                "emit the result; the six comparisons emit 1 when they hold and 0 when "
                "they do not. `exponentiation` raises A to the power of B, so 2 and 10 give "
                "1024, and `modulus` is the remainder, so 7 and 2 give 1."
            ),
            inputs=[
                io.MultiType.Input(
                    "number_a",
                    [NUMBER, io.Int, io.Float],
                    tooltip=(
                        "The left-hand value: what is divided, raised to a power or compared "
                        "against number_b."
                    ),
                ),
                io.MultiType.Input(
                    "number_b",
                    [NUMBER, io.Int, io.Float],
                    tooltip=(
                        "The right-hand value: what number_a is divided by, raised to, or "
                        "measured against. A value of 0 with `division`, `floor division` or "
                        "`modulus` stops with a division error."
                    ),
                ),
                io.Combo.Input(
                    "operation",
                    options=[
                        "addition",
                        "subtraction",
                        "division",
                        "floor division",
                        "multiplication",
                        "exponentiation",
                        "modulus",
                        "greater-than",
                        "greater-than or equals",
                        "less-than",
                        "less-than or equals",
                        "equals",
                        "does not equal",
                    ],
                    tooltip=(
                        "What to do with the two values. `division` keeps the fraction, so "
                        "7 over 2 is 3.5, while `floor division` throws it away and gives "
                        "3. `exponentiation` takes exponents up to 64 and bases up to 1e15, "
                        "either side of zero."
                    ),
                ),
            ],
            outputs=[
                NUMBER.Output(
                    tooltip=(
                        "The result of the operation, or 1/0 when a comparison was chosen. "
                        "A result past 309 digits, or infinite, stops with an error."
                    ),
                ),
                io.Float.Output(tooltip="The same result, on a FLOAT socket."),
                io.Int.Output(
                    tooltip=(
                        "The same result as a whole number, cut off rather than rounded, so "
                        "3.5 leaves here as 3."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, number_a, number_b, operation="addition") -> io.NodeOutput:
        compute = OPERATIONS.get(operation)
        if compute is None:
            logger.error("Invalid number operation selected: %s", operation)
            return io.NodeOutput(number_a, number_a, int(number_a))

        result = carried(compute(number_a, number_b))
        return io.NodeOutput(result, result, int(result))

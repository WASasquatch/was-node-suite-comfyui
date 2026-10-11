"""Pick between two numbers by comparing them."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import NUMBER


def _is_prime(n) -> bool:
    """Whether ``n`` is prime, by trial division over 6k +/- 1.

    Args:
        n: Value to test. Anything below 2 is not prime.

    Returns:
        True when ``n`` has no divisor other than 1 and itself.
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _compare(number_a, number_b, comparison: str) -> bool | None:
    """Evaluate one of the node's comparisons.

    ``divisible by`` and ``factor of`` are the same test, ``number_b % number_a == 0``,
    as they were in v2.

    Args:
        number_a: Left operand.
        number_b: Right operand.
        comparison: Name of the comparison, from the node's combo.

    Returns:
        The outcome of the comparison, or None when ``comparison`` names none of them.

    Raises:
        ZeroDivisionError: A modulo comparison was asked for with ``number_a`` zero.
    """
    if comparison == "and":
        return number_a != 0 and number_b != 0
    if comparison == "or":
        return number_a != 0 or number_b != 0
    if comparison == "greater-than":
        return number_a > number_b
    if comparison == "greater-than or equals":
        return number_a >= number_b
    if comparison == "less-than":
        return number_a < number_b
    if comparison == "less-than or equals":
        return number_a <= number_b
    if comparison == "equals":
        return number_a == number_b
    if comparison == "does not equal":
        return number_a != number_b
    if comparison in ("divisible by", "factor of"):
        return number_b % number_a == 0
    if comparison == "if A odd":
        return number_a % 2 != 0
    if comparison == "if A even":
        return number_a % 2 == 0
    if comparison == "if A prime":
        return _is_prime(number_a)
    return None


class NumberInputCondition(io.ComfyNode):
    """Compare two numbers and emit either the outcome or the number that satisfied it."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="Number Input Condition",
            display_name="Number Input Condition",
            search_aliases=["Number Input Condition", "compare numbers", "number condition"],
            category="WAS Suite/Logic/Switch",
            description=(
                "Test two numbers against each other and emit either 1/0 for the outcome or "
                "the number that won the test, so a value can be picked without a separate "
                "switch node. Tests cover ordering, equality, and/or, divisibility, odd, even "
                "and prime."
            ),
            inputs=[
                io.MultiType.Input(
                    "number_a",
                    [NUMBER, io.Int, io.Float],
                    tooltip=(
                        "The left-hand value, and the only one the 'if A' tests look at. "
                        "When return_boolean is 'false' this is what comes out if the test "
                        "holds."
                    ),
                ),
                io.MultiType.Input(
                    "number_b",
                    [NUMBER, io.Int, io.Float],
                    tooltip=(
                        "The right-hand value. When return_boolean is 'false' this is what "
                        "comes out if the test fails. The 'if A odd', 'if A even' and 'if A "
                        "prime' tests ignore it."
                    ),
                ),
                io.Combo.Input(
                    "return_boolean",
                    ["false", "true"],
                    tooltip=(
                        "What the outputs carry. 'false' passes number_a through when the "
                        "test holds and number_b when it does not; 'true' reports the "
                        "outcome itself as 1 or 0."
                    ),
                ),
                io.Combo.Input(
                    "comparison",
                    [
                        "and",
                        "or",
                        "greater-than",
                        "greater-than or equals",
                        "less-than",
                        "less-than or equals",
                        "equals",
                        "does not equal",
                        "divisible by",
                        "if A odd",
                        "if A even",
                        "if A prime",
                        "factor of",
                    ],
                    tooltip=(
                        "'and' = neither number is 0; 'or' = at least one is not; ordering "
                        "tests compare A against B; 'divisible by' and 'factor of' = B "
                        "divides evenly by A, and an A of 0 stops the prompt; 1 or below is "
                        "not prime."
                    ),
                ),
            ],
            outputs=[
                NUMBER.Output(
                    tooltip=(
                        "The outcome: 1 or 0 when return_boolean is 'true', otherwise "
                        "whichever of the two numbers the test picked."
                    ),
                ),
                io.Float.Output(tooltip="The same result as a float, so 1 leaves here as 1.0."),
                io.Int.Output(
                    tooltip=(
                        "The same result as a whole number, cut off rather than rounded, so "
                        "2.9 leaves here as 2."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, number_a, number_b, return_boolean, comparison) -> io.NodeOutput:
        matched = _compare(number_a, number_b, comparison)
        if return_boolean == "true":
            result = 0 if matched is None else int(matched)
        else:
            result = number_a if matched is None or matched else number_b
        return io.NodeOutput(result, float(result), int(result))

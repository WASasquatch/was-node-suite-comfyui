"""Working out an arithmetic expression written as text.

The text is parsed and walked node by node. Numbers, :data:`VARIABLE_NAMES`,
:data:`CONSTANTS`, :data:`FUNCTIONS`, :data:`OPERATORS` and comparisons are computed;
anything else raises :class:`ExpressionError`.
"""

from __future__ import annotations

import ast
import math
import operator

__all__ = [
    "CONSTANTS",
    "FUNCTIONS",
    "INT_LIMIT",
    "MAX_EXPRESSION_CHARS",
    "MAX_POW_BASE",
    "MAX_POW_EXPONENT",
    "MAX_ROUND_DIGITS",
    "ON_ERROR",
    "VARIABLE_NAMES",
    "ExpressionError",
    "as_int",
    "carried",
    "evaluate",
    "power",
]


class ExpressionError(ValueError):
    """An expression that could not be read, is not allowed, or would not compute."""


#: Characters an expression may run to, once comments and line breaks are folded out.
MAX_EXPRESSION_CHARS = 1024

#: Largest exponent ``**`` accepts, either side of zero.
MAX_POW_EXPONENT = 64

#: Largest base ``**`` accepts, either side of zero.
MAX_POW_BASE = 1e15

#: Most decimal places ``round`` accepts, either side of zero.
MAX_ROUND_DIGITS = 400

#: Widest whole number an INT socket carries.
INT_LIMIT = 2**63 - 1

#: Characters of an expression a message quotes back.
MAX_SHOWN_CHARS = 80

#: Digits a whole number may run to before a message gives its size in place of its value.
MAX_SHOWN_DIGITS = 40

#: The variables an expression may name, in the order a node declares them.
VARIABLE_NAMES: tuple[str, ...] = tuple("abcdefghijklmnopqrstuvwx")

#: What a refused or impossible expression does.
ON_ERROR: tuple[str, ...] = ("error", "zero")

#: Names standing for a fixed value.
CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}


def _clamp(value, low, high):
    """Hold a value between two bounds."""
    return low if value < low else high if value > high else value


def _lerp(start, end, position):
    """Mix two values, ``position`` of the way from the first to the second."""
    return start + (end - start) * position


def _sign(value):
    """-1 below zero, 0 at zero, 1 above it."""
    return (value > 0) - (value < 0)


def _round(value, digits=None):
    """Round to a number of decimal places, or to a whole number where none is given.

    Args:
        value: The number rounded.
        digits: Decimal places, negative for tens, hundreds and up, or None.

    Returns:
        The rounded value.

    Raises:
        ValueError: ``digits`` is past :data:`MAX_ROUND_DIGITS` either side of zero.
    """
    if digits is None:
        return round(value)
    places = int(digits)
    if abs(places) > MAX_ROUND_DIGITS:
        raise ValueError(
            f"the places have to be from -{MAX_ROUND_DIGITS} to {MAX_ROUND_DIGITS}, and "
            f"{_value_word(places)} was given; use a count inside that range"
        )
    return round(value, places)


#: Function name -> the callable, the fewest arguments and the most, ``None`` for no
#: ceiling.
FUNCTIONS: dict[str, tuple] = {
    "abs": (abs, 1, 1),
    "atan2": (math.atan2, 2, 2),
    "ceil": (math.ceil, 1, 1),
    "clamp": (_clamp, 3, 3),
    "cos": (math.cos, 1, 1),
    "degrees": (math.degrees, 1, 1),
    "exp": (math.exp, 1, 1),
    "floor": (math.floor, 1, 1),
    "hypot": (math.hypot, 2, 2),
    "lerp": (_lerp, 3, 3),
    "log": (math.log, 1, 2),
    "log10": (math.log10, 1, 1),
    "log2": (math.log2, 1, 1),
    "max": (max, 2, None),
    "min": (min, 2, None),
    "radians": (math.radians, 1, 1),
    "round": (_round, 1, 2),
    "sign": (_sign, 1, 1),
    "sin": (math.sin, 1, 1),
    "sqrt": (math.sqrt, 1, 1),
    "tan": (math.tan, 1, 1),
}

#: The two-sided operators that are computed. ``**`` is absent: it goes through
#: :func:`power`, which bounds it.
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

#: The one-sided operators that are computed.
UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

#: The comparisons that are computed.
COMPARISONS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}

#: How an operator is written in a message, computed or not.
OPERATOR_WORDS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "//",
    ast.Mod: "%",
    ast.Pow: "**",
    ast.UAdd: "+",
    ast.USub: "-",
    ast.MatMult: "@",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.BitXor: "^",
    ast.LShift: "<<",
    ast.RShift: ">>",
    ast.Invert: "~",
    ast.Not: "not",
    ast.And: "and",
    ast.Or: "or",
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.In: "in",
    ast.NotIn: "not in",
    ast.Is: "is",
    ast.IsNot: "is not",
}

#: How a refused kind of expression is named in a message.
NODE_WORDS = {
    "Attribute": "Attribute access, such as `a.real`,",
    "Subscript": "Indexing, such as `a[0]`,",
    "Slice": "A slice, such as `a[1:2]`,",
    "Lambda": "A lambda",
    "ListComp": "A list comprehension",
    "SetComp": "A set comprehension",
    "DictComp": "A dict comprehension",
    "GeneratorExp": "A generator expression",
    "List": "A list, such as `[1, 2]`,",
    "Tuple": "A tuple, such as `(1, 2)`,",
    "Set": "A set, such as `{1, 2}`,",
    "Dict": "A dict, such as `{1: 2}`,",
    "Starred": "A `*` in front of an argument",
    "JoinedStr": "An f-string",
    "NamedExpr": "An assignment with `:=`",
    "Await": "`await`",
    "Yield": "`yield`",
    "YieldFrom": "`yield from`",
    "Compare": "A comparison",
    "Call": "A call",
    "Name": "A name",
}


def _allowed() -> str:
    """The wording every rejection ends with, naming what may be written instead."""
    return (
        "An expression may hold numbers, the values "
        + f"{VARIABLE_NAMES[0]} to {VARIABLE_NAMES[-1]}"
        + ", the constants "
        + ", ".join(CONSTANTS)
        + ", the operators + - * / // % **, comparisons, and the functions "
        + ", ".join(sorted(FUNCTIONS))
        + "."
    )


def _operator_word(node) -> str:
    """How one operator is written in a message."""
    return OPERATOR_WORDS.get(type(node), type(node).__name__)


def _node_word(node) -> str:
    """How one kind of parsed node is named in a message."""
    return NODE_WORDS.get(type(node).__name__, type(node).__name__)


def _digits(value: int) -> int:
    """About how many digits a whole number is written with.

    Args:
        value: A whole number.

    Returns:
        The digit count, within one of the exact figure.
    """
    return int(value.bit_length() * math.log10(2)) + 1


def _value_word(value) -> str:
    """A value as it appears in a message.

    Args:
        value: A number the walk produced or was given.

    Returns:
        The value written out, or its size where it has more than
        :data:`MAX_SHOWN_DIGITS` digits.
    """
    # Writing an int past this width out in full raises rather than returning a string.
    if type(value) is int and _digits(value) > MAX_SHOWN_DIGITS:
        return f"a {_digits(value)} digit number"
    return str(value)


def _written(values, separator: str = ", ") -> str:
    """A run of values as they appear in a message."""
    return separator.join(_value_word(value) for value in values)


def _shown(text: str) -> str:
    """An expression cut down to the opening of it, for a message."""
    return text if len(text) <= MAX_SHOWN_CHARS else text[:MAX_SHOWN_CHARS] + "..."


def _arity_word(low: int, high: int | None) -> str:
    """How many arguments a function takes, as words."""
    if high is None:
        return f"{low} arguments or more"
    if low == high:
        return f"{low} argument" if low == 1 else f"{low} arguments"
    return f"{low} or {high} arguments"


def _clean(expression) -> str:
    """Fold text typed over several lines into one.

    Args:
        expression: The expression as typed.

    Returns:
        The lines joined with a space, with ``#`` comments and blank lines dropped.
    """
    lines = []
    for line in str(expression).splitlines():
        body = line.split("#", 1)[0].strip()
        if body:
            lines.append(body)
    return " ".join(lines)


def _number(value, name: str):
    """Read a socket value as a number.

    Args:
        value: What the socket carried.
        name: The variable the value stands for, for the message where it cannot be read.

    Returns:
        The value as an int or a float, with true and false read as 1 and 0.

    Raises:
        ExpressionError: The value does not read as a number.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return value
    try:
        text = str(value).strip()
        return int(text) if text.lstrip("+-").isdigit() else float(text)
    except (TypeError, ValueError) as unreadable:
        raise ExpressionError(
            f"`{name}` was given {value!r}, which is not a number. Wire a whole number, a "
            f"decimal or a NUMBER to it, or leave it unconnected to use its own widget."
        ) from unreadable


def power(base, exponent):
    """Raise one value to another, within bounds.

    Args:
        base: The value raised.
        exponent: The power it is raised to.

    Returns:
        The result as an int or a float.

    Raises:
        ExpressionError: Either side is past its bound, or the result is not a real number.
    """
    if abs(exponent) > MAX_POW_EXPONENT:
        raise ExpressionError(
            f"`{_written((base, exponent), ' ** ')}` asks for an exponent of "
            f"{_value_word(exponent)}, and the largest allowed is {MAX_POW_EXPONENT}. A "
            f"larger power can take minutes to work out and holds up the queue, so write the "
            f"result you want instead."
        )
    if abs(base) > MAX_POW_BASE:
        raise ExpressionError(
            f"`{_written((base, exponent), ' ** ')}` raises {_value_word(base)} to a power, "
            f"and the largest base allowed is {MAX_POW_BASE:g}. Scale the value down before "
            f"raising it."
        )
    try:
        result = base**exponent
    except ZeroDivisionError as error:
        raise ExpressionError(
            f"`{_written((base, exponent), ' ** ')}` divides by zero: 0 cannot be raised to "
            f"a negative power."
        ) from error
    except OverflowError as error:
        raise ExpressionError(
            f"`{_written((base, exponent), ' ** ')}` is too large to work out as a decimal."
        ) from error
    if isinstance(result, complex):
        raise ExpressionError(
            f"`{_written((base, exponent), ' ** ')}` has no real answer, because "
            f"{_value_word(base)} is negative and {_value_word(exponent)} is a fraction. "
            f"Take the root of abs({_value_word(base)}) instead."
        )
    return result


def _constant(node):
    """The value of a literal.

    Args:
        node: A parsed constant.

    Returns:
        The number it holds.

    Raises:
        ExpressionError: The literal is not a number.
    """
    if type(node.value) in (int, float):
        return node.value
    if isinstance(node.value, bool):
        raise ExpressionError(
            f"`{node.value}` is not a number. Write 1 for true and 0 for false."
        )
    raise ExpressionError(
        f"{node.value!r} is not a number. Only numbers such as 2, -1 and 0.5 may be written "
        f"into an expression."
    )


def _variable(node, names: dict):
    """The value a name stands for.

    Args:
        node: A parsed name.
        names: Name -> value for every variable and constant in scope.

    Returns:
        The value.

    Raises:
        ExpressionError: The name is not one of the variables or constants.
    """
    if node.id in names:
        return names[node.id]
    raise ExpressionError(f"`{node.id}` is not a name an expression knows. {_allowed()}")


def _unary(node, names: dict):
    """The value of a one-sided operation.

    Args:
        node: A parsed unary operation.
        names: Name -> value for every variable and constant in scope.

    Returns:
        The result.

    Raises:
        ExpressionError: The operator is not `+` or `-`.
    """
    compute = UNARY_OPERATORS.get(type(node.op))
    if compute is None:
        raise ExpressionError(
            f"`{_operator_word(node.op)}` is not allowed in front of a value. Only + and - "
            f"are, as in `-a`."
        )
    return compute(_visit(node.operand, names))


def _binary(node, names: dict):
    """The value of a two-sided operation.

    Args:
        node: A parsed binary operation.
        names: Name -> value for every variable and constant in scope.

    Returns:
        The result.

    Raises:
        ExpressionError: The operator is not one of `+ - * / // % **`, or it could not be
            computed.
    """
    word = _operator_word(node.op)
    if not isinstance(node.op, ast.Pow) and type(node.op) not in OPERATORS:
        raise ExpressionError(
            f"`{word}` is not allowed between two values. The operators are + - * / // % **."
        )
    left = _visit(node.left, names)
    right = _visit(node.right, names)
    if isinstance(node.op, ast.Pow):
        return power(left, right)
    try:
        return OPERATORS[type(node.op)](left, right)
    except ZeroDivisionError as error:
        raise ExpressionError(
            f"`{_value_word(left)} {word} {_value_word(right)}` divides by zero. Guard it "
            f"with a comparison, as in `a / b if b != 0 else 0`, or set on_error to zero."
        ) from error
    except OverflowError as error:
        raise ExpressionError(
            f"`{_value_word(left)} {word} {_value_word(right)}` is too large to work out."
        ) from error


def _boolean(node, names: dict):
    """The value of an `and` or `or` chain, stopping at the operand that settles it.

    Args:
        node: A parsed boolean operation.
        names: Name -> value for every variable and constant in scope.

    Returns:
        The operand that settled the chain, or the last one.
    """
    settles = isinstance(node.op, ast.Or)
    result = None
    for operand in node.values:
        result = _visit(operand, names)
        if bool(result) is settles:
            return result
    return result


def _compare(node, names: dict):
    """Whether a comparison holds, including a chain such as `0 < a < 10`.

    Args:
        node: A parsed comparison.
        names: Name -> value for every variable and constant in scope.

    Returns:
        True when every link of the chain holds.

    Raises:
        ExpressionError: A link uses `in`, `is` or another comparison that is not offered.
    """
    left = _visit(node.left, names)
    for op, right_node in zip(node.ops, node.comparators):
        compute = COMPARISONS.get(type(op))
        if compute is None:
            raise ExpressionError(
                f"`{_operator_word(op)}` is not allowed as a comparison. The comparisons are "
                f"== != < <= > >=."
            )
        right = _visit(right_node, names)
        if not compute(left, right):
            return False
        left = right
    return True


def _call(node, names: dict):
    """The value of a function call.

    Args:
        node: A parsed call.
        names: Name -> value for every variable and constant in scope.

    Returns:
        What the function answered.

    Raises:
        ExpressionError: The function is not offered, is called with the wrong number or
            shape of arguments, or could not work the values out.
    """
    if not isinstance(node.func, ast.Name):
        raise ExpressionError(
            f"{_node_word(node.func)} cannot be called. Call a function by name, as in "
            f"`sqrt(a)`. {_allowed()}"
        )
    name = node.func.id
    entry = FUNCTIONS.get(name)
    if entry is None:
        raise ExpressionError(f"`{name}` is not a function an expression knows. {_allowed()}")
    compute, low, high = entry
    if node.keywords:
        raise ExpressionError(
            f"`{name}` takes plain arguments only, so write `{name}(a, b)` rather than naming "
            f"them."
        )
    arguments = [_visit(argument, names) for argument in node.args]
    if len(arguments) < low or (high is not None and len(arguments) > high):
        raise ExpressionError(
            f"`{name}` takes {_arity_word(low, high)} and was given {len(arguments)}."
        )
    try:
        return compute(*arguments)
    except ZeroDivisionError as error:
        raise ExpressionError(
            f"`{name}({_written(arguments)})` divides by zero."
        ) from error
    except (ValueError, OverflowError, TypeError) as error:
        raise ExpressionError(
            f"`{name}({_written(arguments)})` could not be worked out: {error}."
        ) from error


def _visit(node, names: dict):
    """The value of one parsed node.

    Args:
        node: The parsed node.
        names: Name -> value for every variable and constant in scope.

    Returns:
        The value, as an int, a float or a bool.

    Raises:
        ExpressionError: The node is of a kind that is not allowed, or its value could not
            be worked out.
    """
    if isinstance(node, ast.Expression):
        return _visit(node.body, names)
    if isinstance(node, ast.Constant):
        return _constant(node)
    if isinstance(node, ast.Name):
        return _variable(node, names)
    if isinstance(node, ast.UnaryOp):
        return _unary(node, names)
    if isinstance(node, ast.BinOp):
        return _binary(node, names)
    if isinstance(node, ast.BoolOp):
        return _boolean(node, names)
    if isinstance(node, ast.Compare):
        return _compare(node, names)
    if isinstance(node, ast.IfExp):
        if _visit(node.test, names):
            return _visit(node.body, names)
        return _visit(node.orelse, names)
    if isinstance(node, ast.Call):
        return _call(node, names)
    raise ExpressionError(f"{_node_word(node)} is not allowed in an expression. {_allowed()}")


def carried(value):
    """A computed value, checked as something a graph can carry.

    Args:
        value: What an expression or an operation answered.

    Returns:
        The same value.

    Raises:
        ExpressionError: The value is infinite, is not a number, or has too many digits.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExpressionError(
                f"The result came out as {value}, which no downstream node can use. "
                f"Check for a division by a very small number."
            )
        return value
    if isinstance(value, int):
        try:
            float(value)
        except OverflowError as error:
            raise ExpressionError(
                f"The result has about {_digits(value)} digits, which is too many to travel "
                f"on a number wire. The widest a number wire carries is 309 digits."
            ) from error
        return value
    raise ExpressionError(f"The result came out as {value!r}, which is not a number.")


def as_int(value) -> int:
    """Cut a value down to a whole number a graph can carry.

    Args:
        value: A finite number.

    Returns:
        The value with its fraction removed, held to the range of a 64-bit whole number.
    """
    whole = int(value)
    return max(-INT_LIMIT - 1, min(INT_LIMIT, whole))


def evaluate(expression, variables: dict | None = None):
    """Work out an arithmetic expression written as text.

    Args:
        expression: The expression, which may run over several lines and hold ``#``
            comments.
        variables: Value for each of :data:`VARIABLE_NAMES`. A missing or ``None`` entry
            counts as 0.

    Returns:
        The value as an int or a float, or as a bool where the expression is a comparison.

    Raises:
        ExpressionError: The expression is empty, too long, could not be parsed, holds
            something that is not allowed, or could not be worked out.
    """
    text = _clean(expression)
    if not text:
        raise ExpressionError(
            "The expression is empty. Type one, such as `(a * b) / 2 + c`."
        )
    if len(text) > MAX_EXPRESSION_CHARS:
        raise ExpressionError(
            f"The expression is {len(text)} characters long and the limit is "
            f"{MAX_EXPRESSION_CHARS}. Split it across two nodes."
        )

    given = variables or {}
    names = dict(CONSTANTS)
    for name in VARIABLE_NAMES:
        value = given.get(name)
        # An unwired letter that is also a constant, `e`, keeps the constant's value.
        if value is None and name in CONSTANTS:
            continue
        names[name] = 0 if value is None else _number(value, name)

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as error:
        raise ExpressionError(
            f"`{_shown(text)}` could not be read as an expression: {error.msg}. Check the "
            f"brackets and the operators."
        ) from error
    except (MemoryError, RecursionError) as error:
        raise ExpressionError(
            "The expression is nested too deeply to read. Split it across two nodes."
        ) from error
    except ValueError as error:
        raise ExpressionError(
            f"`{_shown(text)}` could not be read as an expression: {error}. Retype it, or "
            f"paste it from somewhere that carries no hidden characters."
        ) from error

    try:
        return carried(_visit(tree, names))
    except RecursionError as error:
        raise ExpressionError(
            "The expression is nested too deeply to work out. Split it across two nodes."
        ) from error

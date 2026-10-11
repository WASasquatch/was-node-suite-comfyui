"""Whatever is on a wire, written out on the node it is connected to.

The rendering, the value's type and its size publish under the node's own id, and the value
passes through unchanged.
"""

from __future__ import annotations

import json
import math
from collections.abc import Collection, Mapping
from itertools import islice

from comfy_api.latest import io, ui

from ...modules import log
from ...modules.interface import run_result
from ...modules.interface.batch_report import memory_of, readable_bytes

logger = log.get_logger("nodes.utility")

#: What the published rendering is called, which is the name a panel asks for.
BODY_NAME = "value"

#: The summary line for a run handed nothing, which is drawn in the warning colour.
NOTHING = "nothing arrived on the value input"

#: How many levels of a container are written out. Past this a nested container is named and
#: its contents are left to the count beside the name.
MAX_DEPTH = 6

#: Entries listed from one container. The rest are counted on a line of their own. A
#: rendering longer than the channel carries is cut by its own limit rather than this one.
MAX_ROWS = 64

#: Keys named in the one-line summary of a mapping. The rest are left to the count beside it.
MAX_NAMED = 8

#: Values written into the node's own text output from one socket carrying several. The rest
#: are counted under them.
MAX_LIST_ROWS = 64

#: Characters of a string written into the one line naming a value. A longer one is named by
#: its length there, and written out in full in the rendering.
SHORT_TEXT = 60


class DisplayAny(io.ComfyNode):
    """Write the value on the socket onto the node, and pass it on unchanged."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        template = io.MatchType.Template("display_any")
        return io.Schema(
            node_id="WASDisplayAny",
            display_name="Display Any",
            search_aliases=[
                "WASDisplayAny", "Display Any",
                "show value",
                "preview any",
                "inspect",
                "debug any",
                "what is this",
            ],
            category="WAS Suite/Utilities",
            description=(
                "Show whatever is connected on the node: text and numbers as they are, a list "
                "or dictionary as its data, and an image, mask or latent as its shape, type, "
                "device and value range. Several values on one socket each get a box to "
                "select and copy. The value passes through unchanged."
            ),
            inputs=[
                io.MatchType.Input(
                    "value",
                    template=template,
                    tooltip=(
                        "Anything at all: text, a number, an image, a mask, a latent, a "
                        "model. Whatever connects here first decides the type of the node, "
                        "and the output then carries that same type."
                    ),
                ),
            ],
            outputs=[
                io.MatchType.Output(
                    template=template,
                    display_name="value",
                    is_output_list=True,
                    tooltip=(
                        "The same value that came in, unchanged, on a socket carrying its "
                        "type. Leave it unwired to end a branch on the node, or wire it on "
                        "to watch a value partway down a chain."
                    ),
                ),
            ],
            is_input_list=True,
            is_output_node=True,
        )

    @classmethod
    def execute(cls, value=None) -> io.NodeOutput:
        # Under is_input_list every value on the socket arrives at once, so a socket
        # carrying several is written out entry by entry rather than a run per entry.
        values = value if isinstance(value, list) else [] if value is None else [value]
        single = values[0] if len(values) == 1 else None
        subject = single if len(values) == 1 else values
        rendered = _written(single) if len(values) == 1 else _list(values)
        _publish(subject, rendered, len(values))
        return io.NodeOutput(values, ui=ui.PreviewText(_shown(rendered)))


def _written(value) -> str:
    """A value written out, or what stopped it being written out.

    Args:
        value: The value the node was handed.

    Returns:
        The rendering, or a line naming the value and what stopped it being written out.
    """
    try:
        return _render(value)
    except Exception as error:
        return f"{type(value).__name__} could not be written out ({error})"


def _list(values: list) -> str:
    """Every value on the socket, one after another with a blank line between them.

    Args:
        values: What the socket delivered.

    Returns:
        The values as they are, so the whole rendering can be copied and is the values
        themselves rather than a numbered list of them.
    """
    written = [_written(item) for item in islice(values, MAX_LIST_ROWS)]
    if len(values) > MAX_LIST_ROWS:
        written.append(
            f"{run_result.ELLIPSIS} {_plural(len(values) - MAX_LIST_ROWS, 'more value')}"
        )
    return "\n\n".join(written) if written else "empty"


def _bodies(values: list) -> list:
    """One body per value, so each is drawn in a box of its own.

    Args:
        values: What the socket delivered.

    Returns:
        A body per value, named by its position out of the total. The name is drawn above
        the box, so what the box holds is the value and nothing else.
    """
    total = len(values)
    return [
        run_result.body(f"{index + 1} / {total}", _written(item))
        for index, item in enumerate(islice(values, run_result.MAX_BODIES))
    ]


def _publish(value, rendered: str, arity: int = 1) -> None:
    """Report what arrived to the node's own interface.

    Args:
        value: The value the node was handed, or the list of them.
        rendered: That value written out.
        arity: How many values the socket delivered.
    """
    try:
        if not run_result.watching():
            return
        empty = arity == 0
        counts = _counts(value) if arity == 1 else {"values": arity}
        facts = _facts(value) if arity == 1 else {"type": _list_type(value)}
        bodies = (
            [run_result.body(BODY_NAME, rendered)] if arity <= 1 else _bodies(value)
        )
        run_result.publish(
            status=run_result.WARNING if empty else run_result.OK,
            summary=NOTHING if empty else _summary(value, arity),
            counts={} if empty else counts,
            facts={} if empty else facts,
            bodies=bodies,
        )
    except Exception as error:
        logger.debug("Display Any did not report what it was handed (%s)", error)


def _summary(value, arity: int) -> str:
    """The line naming what arrived, whether that is one value or several."""
    if arity == 1:
        return _describe(value)
    return f"{_plural(arity, 'value')} arrived"


def _list_type(values) -> str:
    """The type every value on the socket shares, or how many types they are between them."""
    names = {_type_name(item) for item in values}
    if len(names) == 1:
        return names.pop()
    return f"{_plural(len(names), 'type')}: {', '.join(sorted(names)[:4])}"


def _shown(rendered: str) -> str:
    """The rendering as the node draws it, cut to the length a result carries.

    Args:
        rendered: The value written out.

    Returns:
        The whole rendering, or its first :data:`run_result.MAX_BODY_CHARS` characters with
        the whole length stated under them.
    """
    if len(rendered) <= run_result.MAX_BODY_CHARS:
        return rendered
    return (
        f"{rendered[:run_result.MAX_BODY_CHARS]}\n"
        f"{run_result.ELLIPSIS} {len(rendered):,} characters in all"
    )


def _counts(value) -> dict:
    """The numbers drawn as figures, inside :data:`run_result.MAX_COUNTS`.

    Args:
        value: The value the node was handed.

    Returns:
        A mapping of name to number, empty for a value with no number worth a figure.
    """
    if isinstance(value, bool) or value is None:
        return {}
    if isinstance(value, (int, float)) and _measurable(value):
        return {"value": value}
    if isinstance(value, str):
        return {"characters": len(value), "lines": value.count("\n") + 1}
    if _is_tensor(value):
        sizes = _sizes(value)
        return {"frames": sizes[0]} if len(sizes) >= 3 else {}
    if isinstance(value, Mapping):
        return {"keys": len(value)}
    if _is_sized(value):
        return {"items": len(value)}
    return {}


def _facts(value) -> dict:
    """The rows drawn under the figures, inside :data:`run_result.MAX_FACTS`.

    Args:
        value: The value the node was handed.

    Returns:
        A mapping of name to text, in the order they are drawn.
    """
    facts = {"type": _type_name(value)}
    if _is_tensor(value):
        facts["shape"] = _shape(value)
        facts["dtype"] = _dtype(value)
        facts["device"] = str(getattr(value, "device", ""))
        facts["range"] = _span(value)
        memory = memory_of(value)
        facts["memory"] = readable_bytes(memory) if memory else ""
    elif isinstance(value, Mapping):
        facts["keys"] = ", ".join(str(name) for name in islice(value, MAX_NAMED))
    elif isinstance(value, (bytes, bytearray, memoryview)):
        facts["length"] = f"{len(value):,} bytes"
    return {name: text for name, text in facts.items() if text}


def _render(value, depth: int = 0) -> str:
    """A value written out.

    Args:
        value: The value to write out.
        depth: How many containers this one sits inside.

    Returns:
        The rendering: the text itself for a string, the data itself for a container, and
        the line naming it for anything else.
    """
    if isinstance(value, str):
        return value
    if _is_tensor(value):
        span = _span(value)
        return f"{_tensor(value)}\nrange {span}" if span else _tensor(value)
    if _is_container(value):
        return _written_out(value)
    return _describe(value)


def _serialised(value, depth: int = 0, seen: tuple = ()) -> object:
    """A value as the JSON-safe data it holds.

    Args:
        value: The value to convert.
        depth: How many containers this one sits inside.
        seen: Ids of the containers it sits inside, so a cycle stops.

    Returns:
        The same data in types JSON writes. Anything JSON has no form for, a tensor or an
        object, becomes the line that names it.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _describe(value)
    if _is_tensor(value) or not _is_container(value):
        return _describe(value)
    if depth >= MAX_DEPTH or id(value) in seen:
        return _describe(value)

    inside = (*seen, id(value))
    if isinstance(value, Mapping):
        kept = {
            str(name): _serialised(item, depth + 1, inside)
            for name, item in islice(value.items(), MAX_ROWS)
        }
        left = len(value) - len(kept)
        if left > 0:
            kept[run_result.ELLIPSIS] = _plural(left, "more entry", "more entries")
        return kept

    kept = [_serialised(item, depth + 1, inside) for item in islice(value, MAX_ROWS)]
    left = len(value) - len(kept)
    if left > 0:
        kept.append(f"{run_result.ELLIPSIS} {_plural(left, 'more entry', 'more entries')}")
    return kept


def _written_out(value) -> str:
    """A container written out as the data it holds.

    Args:
        value: A mapping, or any other sized collection.

    Returns:
        Its entries as JSON, set in two spaces a level. An empty one answers ``empty``.
    """
    if not len(value):
        return "empty"
    return json.dumps(_serialised(value), indent=2, ensure_ascii=False, default=_describe)


def _describe(value) -> str:
    """One line naming what a value is, and its size or its contents where they are short.

    Args:
        value: The value to name.

    Returns:
        A line such as ``torch.Tensor 1x512x512x3 float32 cuda:0`` or ``dict, 3 keys``.
    """
    if value is None:
        return "None"
    if isinstance(value, (bool, int, float, complex)):
        return f"{_type_name(value)} {value}"
    if isinstance(value, str):
        if len(value) <= SHORT_TEXT and "\n" not in value:
            return f'str "{value}"'
        return f"str, {len(value):,} characters"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"{_type_name(value)}, {len(value):,} bytes"
    if _is_tensor(value):
        return _tensor(value)
    if isinstance(value, Mapping):
        return f"{_type_name(value)}, {_plural(len(value), 'key')}"
    if _is_sized(value):
        return f"{_type_name(value)}, {_plural(len(value), 'item')}"
    return _type_name(value)


def _tensor(value) -> str:
    """A tensor named by its class, its shape, its element type and where it is held."""
    parts = [_type_name(value), _shape(value), _dtype(value), str(getattr(value, "device", ""))]
    return " ".join(part for part in parts if part)


def _sizes(value) -> tuple[int, ...]:
    """The lengths of a tensor's axes, empty for a single number and for anything unreadable."""
    try:
        return tuple(int(size) for size in value.shape)
    except Exception:
        return ()


def _shape(value) -> str:
    """A tensor's axes as ``1x512x512x3``, or what a tensor holding one number is."""
    sizes = _sizes(value)
    return "x".join(str(size) for size in sizes) if sizes else "scalar"


def _dtype(value) -> str:
    """A tensor's element type."""
    # The torch prefix is dropped from the spelling.
    return str(getattr(value, "dtype", "")).removeprefix("torch.")


def _span(value) -> str:
    """The lowest and highest values a tensor holds, or nothing where they cannot be read."""
    try:
        lowest, highest = float(value.min()), float(value.max())
    except Exception:
        return ""
    return f"{lowest:.4g} to {highest:.4g}"


def _type_name(value) -> str:
    """A value's class, with the module it comes from unless that module is the language's."""
    kind = type(value)
    module = getattr(kind, "__module__", "")
    name = getattr(kind, "__qualname__", "") or kind.__name__
    return name if module in ("", "builtins") else f"{module}.{name}"


def _measurable(number) -> bool:
    """Whether a number is one a figure can carry, which an infinity and a huge integer are not."""
    try:
        return math.isfinite(number)
    except OverflowError:
        return False


def _is_tensor(value) -> bool:
    """Whether a value carries a shape and an element type, as an image or a latent does."""
    return hasattr(value, "shape") and hasattr(value, "dtype")


def _is_sized(value) -> bool:
    """Whether a value's entries can be counted and walked without consuming it."""
    return isinstance(value, Collection) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    )


def _is_container(value) -> bool:
    """Whether a value is listed entry by entry rather than written out on one line."""
    return not _is_tensor(value) and (isinstance(value, Mapping) or _is_sized(value))


def _plural(count: int, word: str, many: str = "") -> str:
    """A count and its word, with the word's plural where the count is not one.

    Args:
        count: The number of them.
        word: What one of them is called.
        many: What several of them are called. Left out, ``word`` with an ``s``.

    Returns:
        Text such as ``3 keys``.
    """
    return f"{count:,} {word}" if count == 1 else f"{count:,} {many or word + 's'}"

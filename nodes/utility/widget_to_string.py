"""Reading a named widget off another node of the running graph, as text.

The graph arrives on the hidden dynprompt, so a value is read as the prompt queued it
rather than as the canvas draws it.
"""

from __future__ import annotations

import json

from comfy_api.latest import io, ui

from ...modules import log

logger = log.get_logger("nodes.utility")

#: How many nodes or widget names a refusal lists before it counts the rest.
MAX_NAMED = 8

#: Stands for an input a node does not declare, so a widget holding None is told from one
#: that is not there.
_ABSENT = object()


class WidgetToString(io.ComfyNode):
    """Read a named widget on another node of the graph and answer its value as a string."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASWidgetToString",
            display_name="Widget to String",
            search_aliases=[
                "WASWidgetToString", "Widget to String",
                "widget value",
                "read widget",
                "node widget",
                "seed to string",
                "sampler name",
            ],
            category="WAS Suite/Utilities",
            description=(
                "Read one widget off another node in the graph as text, so a filename, "
                "caption or log line can carry the seed, sampler, steps or prompt from where "
                "it is set. Give the node's id and the widget's name; nothing is wired to it."
            ),
            inputs=[
                io.String.Input(
                    "node_id",
                    default="",
                    multiline=False,
                    tooltip=(
                        "The id of the node to read, as ComfyUI draws it on the node's "
                        "badge: 12. Turn the badge on under Settings if no number is "
                        "showing. Inside a subgraph the local number is enough, and a full "
                        "path such as 12:3 works too."
                    ),
                ),
                io.String.Input(
                    "widget_name",
                    default="",
                    multiline=False,
                    tooltip=(
                        "The widget to read, spelled exactly as its node spells it: seed, "
                        "steps, cfg, sampler_name, ckpt_name, text. An input filled by a "
                        "wire holds no widget value, and asking for one says which node "
                        "feeds it."
                    ),
                ),
                io.String.Input(
                    "default",
                    default="",
                    multiline=False,
                    optional=True,
                    tooltip=(
                        "Text to answer when the node or widget cannot be found, keeping the "
                        "prompt running with found `false`. Empty = a miss stops the prompt, "
                        "naming what was looked for and which nodes carry that widget."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="string",
                    tooltip=(
                        "What the widget holds, written out: 42, 8.0, true, dpmpp_2m, or a "
                        "whole prompt. Feed it to a filename prefix, a text join, or "
                        "anything else taking a STRING."
                    ),
                ),
                io.Boolean.Output(
                    display_name="found",
                    tooltip=(
                        "true when the widget was read, false when default stood in for "
                        "it. Wire it into a switch where a stand-in must be handled "
                        "differently, since the string alone cannot tell the two apart."
                    ),
                ),
            ],
            hidden=[io.Hidden.unique_id, io.Hidden.dynprompt],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        """NaN, so the widget is read again on every prompt rather than served from a cache."""
        return float("NaN")

    @classmethod
    def execute(cls, node_id, widget_name, default="") -> io.NodeOutput:
        hidden = getattr(cls, "hidden", None)
        graph = getattr(hidden, "dynprompt", None)
        try:
            value = _read(
                graph,
                str(node_id).strip(),
                str(widget_name).strip(),
                getattr(hidden, "unique_id", None),
            )
        except ValueError as refusal:
            if not default:
                raise
            logger.warning("%s Answering the default instead.", refusal)
            return io.NodeOutput(default, False, ui=ui.PreviewText(default))
        text = _as_text(value)
        return io.NodeOutput(text, True, ui=ui.PreviewText(text))


def _read(graph, wanted: str, name: str, unique_id):
    """The value a named widget holds on a named node of the graph.

    Args:
        graph: The run's ``comfy_execution.graph.DynamicPrompt``, or None outside a prompt.
        wanted: The node id to read, stripped of surrounding space.
        name: The widget's name, spelled as its own node spells it.
        unique_id: The reading node's own graph id, which names the subgraph searched first.

    Returns:
        The widget's value, with whatever type the queued prompt stored it as.

    Raises:
        ValueError: No prompt is running, either field is empty, the graph holds no such
            node, that node carries no such widget, or the input is filled by a wire.
    """
    if graph is None:
        raise ValueError(
            "Widget to String ran outside a prompt, so there is no graph to read. Queue "
            "the workflow rather than calling the node on its own."
        )
    if not wanted:
        raise ValueError(
            "Widget to String was given no node_id. Type the id ComfyUI draws on the badge "
            "of the node to read, such as 12."
        )
    if not name:
        raise ValueError(
            "Widget to String was given no widget_name. Type the name of the widget to "
            "read, such as seed, steps or text."
        )

    key = _resolve(graph, wanted, unique_id)
    if key is None:
        raise ValueError(
            f'Widget to String found no node "{wanted}" in this graph. '
            f"{_carrying(graph, name)} Turn on the node id badge in ComfyUI's settings to "
            f"read an id off the canvas, or fill in default to answer text instead of "
            f"stopping."
        )

    inputs = graph.get_node(key).get("inputs") or {}
    if name not in inputs:
        raise ValueError(
            f'{_label(graph, key)} has no widget named "{name}". {_widgets(inputs)} Check '
            f"the spelling, or fill in default to answer text instead of stopping."
        )

    value = inputs[name]
    if _is_link(value):
        raise ValueError(
            f'Input "{name}" on {_label(graph, key)} is filled by a wire from node '
            f"{value[0]}, so it carries no widget value of its own. Read a widget on node "
            f"{value[0]} instead, or fill in default to answer text instead of stopping."
        )
    return value


def _resolve(graph, wanted: str, unique_id) -> str | None:
    """The graph's own key for a typed node id, or None when no node answers to it."""
    for key in _keys(wanted, unique_id):
        if graph.has_node(key):
            return key
    return None


def _keys(wanted: str, unique_id) -> list[str]:
    """The keys a typed node id could mean, the reading node's own subgraph first.

    Args:
        wanted: The node id as typed.
        unique_id: The reading node's graph id, a colon joined path inside a subgraph.

    Returns:
        The candidate keys, in the order they are tried. A bare number typed inside a
        subgraph is offered against that subgraph's path before the top level.
    """
    here = str(unique_id or "")
    if ":" in here and ":" not in wanted:
        return [f"{here.rsplit(':', 1)[0]}:{wanted}", wanted]
    return [wanted]


def _label(graph, key: str) -> str:
    """A node named by its graph id and its title, as ``12 (KSampler)``."""
    node = graph.get_node(key)
    meta = node.get("_meta") or {}
    return f"{key} ({meta.get('title') or node.get('class_type') or 'untitled'})"


def _carrying(graph, name: str) -> str:
    """One sentence naming the nodes that do carry a widget called ``name``."""
    found = [key for key in _sorted_ids(graph) if _has_widget(graph, key, name)]
    if not found:
        return f'No node in this graph carries a widget named "{name}".'
    listed = ", ".join(_label(graph, key) for key in found[:MAX_NAMED])
    return f'Nodes carrying a widget named "{name}": {listed}{_rest(len(found))}.'


def _has_widget(graph, key: str, name: str) -> bool:
    """Whether a node holds a widget of this name rather than an input filled by a wire."""
    value = (graph.get_node(key).get("inputs") or {}).get(name, _ABSENT)
    return value is not _ABSENT and not _is_link(value)


def _widgets(inputs: dict) -> str:
    """One sentence listing the widget names a node does carry."""
    names = [name for name, value in inputs.items() if not _is_link(value)]
    if not names:
        return "Every input on it is filled by a wire, so it carries no widget at all."
    return f"Its widgets are: {', '.join(names[:MAX_NAMED])}{_rest(len(names))}."


def _rest(total: int) -> str:
    """What is appended to a list cut at :data:`MAX_NAMED`, empty when nothing was cut."""
    return f", and {total - MAX_NAMED} more" if total > MAX_NAMED else ""


def _sorted_ids(graph) -> list:
    """Every node id in the graph, the plain numeric ones in numeric order."""
    return sorted(graph.all_node_ids(), key=_order)


def _order(key) -> tuple:
    """A sort key putting numeric node ids in numeric order ahead of subgraph paths."""
    text = str(key)
    return (0, int(text), "") if text.isdigit() else (1, 0, text)


def _is_link(value) -> bool:
    """Whether a stored input is a wire from another node rather than a widget value."""
    from comfy_execution.graph_utils import is_link

    return is_link(value)


def _as_text(value) -> str:
    """A widget value written out as a string.

    Args:
        value: Whatever the queued prompt stored for that widget.

    Returns:
        The text itself for a string, ``true`` or ``false`` for a boolean, the digits for a
        number, an empty string for nothing, and JSON for anything else.
    """
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, default=str)

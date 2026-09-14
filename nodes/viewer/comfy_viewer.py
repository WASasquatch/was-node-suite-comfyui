"""An embedded viewer that renders whatever is wired into it."""

from __future__ import annotations

import hashlib
import json

from comfy_api.latest import io

from ...modules import log

REQUIRES = "viewer"

logger = log.get_logger("nodes.viewer")

#: Separator joining several list items into the single string the frontend renders, and
#: splitting them back out again. Long enough not to occur in content by accident.
LIST_SEPARATOR = "\n---LIST_SEPARATOR---\n"

#: How much of the content is written to the log. The viewer is routinely handed whole
#: documents and whole image tensors, and neither belongs in a log line.
LOG_EXCERPT = 256


def to_string(value) -> str:
    """One input value as text the viewer can render.

    Args:
        value: Anything arriving on the ``content`` socket, a string, a number, or an
            object a view knows how to draw.

    Returns:
        ``value`` as a string: itself when it already is one, its JSON form when it
        serialises, and ``repr``-ish text when it does not. Never raises.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value)
    except (TypeError, ValueError) as error:
        logger.debug("%s could not be serialised as JSON (%s)", type(value).__name__, error)
    try:
        return str(value)
    except Exception as error:
        logger.warning("a %s value could not be rendered (%s)", type(value).__name__, error)
        return "Content exists but could not be serialized."


def has_content(items) -> bool:
    """Whether the list holds an item that is neither ``None`` nor an empty string."""
    # An explicit walk, not :func:`any`: a truth test on an image tensor raises rather than
    # answering.
    for item in items or ():
        if item is None:
            continue
        if isinstance(item, str) and not item:
            continue
        return True
    return False


def input_hash(values) -> str:
    """A digest of the content wired in, used to spot when a cached view went stale."""
    combined = "".join(to_string(value) for value in values or () if value is not None)
    if not combined:
        return ""
    return hashlib.md5(combined.encode("utf-8", errors="replace"), usedforsecurity=False).hexdigest()


def content_hash(source: str) -> str:
    """A cheap change marker for content the frontend has already been sent."""
    return f"{len(source)}_{hash(source) & 0xFFFFFFFF}"


def as_list(value) -> list:
    """One socket's value as a list, whatever arity it arrived with."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def expand_lists(items: list) -> list:
    """One item per entry, opening any LIST value out into the entries it carries.

    Args:
        items: What one socket delivered, already a list.

    Returns:
        The same items, with every list or tuple of text and numbers replaced by its own
        entries, so a LIST is drawn as the entries it holds rather than as one object.
    """
    opened = []
    for item in items:
        if isinstance(item, (list, tuple)) and all(
            entry is None or isinstance(entry, (str, int, float, bool)) for entry in item
        ):
            opened.extend("" if entry is None else entry for entry in item)
            continue
        opened.append(item)
    return opened


def excluded_indexes(viewer_meta) -> list:
    """The list item indexes the user has unticked, read out of the frontend's metadata."""
    if not viewer_meta:
        return []
    raw = viewer_meta[0] if isinstance(viewer_meta, list) else viewer_meta
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(parsed, dict):
        return []
    excluded = parsed.get("excluded")
    return excluded if isinstance(excluded, list) else []


def cached_output(view_state, current: str):
    """A view's own stored output, when it is still answerable for the current input.

    Args:
        view_state: The ``view_state`` widget, as the list this node receives.
        current: The digest of the content wired in right now.

    Returns:
        The parsed output, or ``None`` when there is none, when it cannot be read, or when
        the input has moved on since it was stored.
    """
    if not has_content(view_state):
        return None
    raw = to_string(view_state[0]) if len(view_state) == 1 else view_state[0]
    try:
        state = json.loads(raw) if raw else {}
    except (TypeError, ValueError):
        return None
    if not isinstance(state, dict):
        return None

    stored = state.get("_input_hash", "")
    if current and stored and stored != current:
        logger.info("the wired content changed, so the stored view output was discarded")
        return None
    if current and not stored:
        return None

    from ...modules.viewer.parsers import parse_output

    for key, value in state.items():
        if not key.endswith("_output") or not value:
            continue
        parsed = parse_output(value, logger)
        if parsed:
            return parsed
    return None


class WASComfyViewer(io.ComfyNode):
    """Render anything wired into it, in a sandboxed frame, and pass it on."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASComfyViewer",
            display_name="Content Viewer",
            search_aliases=[
                'WASComfyViewer',
                "Content Viewer",
                "viewer",
                "markdown",
                "html",
                "preview",
                "omni viewer",
                "code",
            ],
            category="WAS Suite/View",
            description=(
                "Display anything wired in, in an embedded frame, and pass it on unchanged: "
                "Markdown with Mermaid diagrams and KaTeX maths, HTML, SVG, documents on a "
                "DOC wire, syntax-highlighted code, collapsible JSON and YAML, CSV tables, "
                "coloured terminal logs, an image canvas with layers, brushes and blend "
                "modes, and an inspector for tensors. The view is picked from the content, "
                "and the dropdown beside it changes to any other. An edit made in the node "
                "is saved with the workflow and goes downstream, and a list arrives as "
                "numbered panes with a tick box each, so entries can be dropped before they "
                "go on."
            ),
            inputs=[
                # A wildcard socket carries no widget, so it is link-only already and takes
                # none of the widget keywords the string inputs below do.
                io.AnyType.Input(
                    "content",
                    optional=True,
                    tooltip=(
                        "Whatever should be displayed. Text, a list of strings, a parsed "
                        "object or an image batch all work: the view that recognises the "
                        "content is the one that draws it. A list arrives as one numbered "
                        "container per item, each with its own tick box and copy button."
                    ),
                ),
                # The three below carry state the viewer's own frontend writes and reads
                # back. They are socketless, leaving that state to the frontend alone. The
                # content input above is the one to wire into.
                io.String.Input(
                    "manual_content",
                    default="",
                    optional=True,
                    socketless=True,
                    tooltip=(
                        "What was typed into the node with the Edit button. Written by the "
                        "viewer rather than by hand, and saved with the workflow, so an edit "
                        "survives a reload and takes precedence over the wired content."
                    ),
                ),
                io.String.Input(
                    "viewer_meta",
                    default='{"lastInputHash": "", "excluded": []}',
                    optional=True,
                    socketless=True,
                    tooltip=(
                        "Which list items are unticked, as JSON. Written by the viewer; "
                        "unticked items are shown but left out of the output."
                    ),
                ),
                io.String.Input(
                    "view_state",
                    default="{}",
                    optional=True,
                    socketless=True,
                    tooltip=(
                        "Which view is selected and anything that view has stored, as JSON. "
                        "Written by the viewer, and what lets a composited canvas or a "
                        "rendered frame survive a reload without re-running the graph."
                    ),
                ),
                io.Boolean.Input(
                    "hold_for_edit",
                    default=False,
                    optional=True,
                    socketless=True,
                    tooltip=(
                        "`false` shows the content and carries on. `true` stops the run "
                        "here until Continue is pressed, and sends on whatever was edited. "
                        "Set from Pause Workflow on the viewer's own bar."
                    ),
                ),
                io.Float.Input(
                    "hold_timeout",
                    default=600.0,
                    min=0.0,
                    max=86400.0,
                    step=10.0,
                    optional=True,
                    socketless=True,
                    tooltip=(
                        "Seconds to wait before carrying on with what arrived. 600 is ten "
                        "minutes, 0 waits with no limit. The whole queue holds still."
                    ),
                ),
            ],
            outputs=[
                io.AnyType.Output(
                    display_name="content",
                    is_output_list=True,
                    tooltip=(
                        "What is on display, as a list. Edits replace the wired content, and "
                        "unticked list items are dropped, so this is what was shown rather "
                        "than what arrived."
                    ),
                ),
            ],
            is_input_list=True,
            is_output_node=True,
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    def fingerprint_inputs(cls, hold_for_edit=False, **kwargs):
        """Never cached while holding, cached as before otherwise.

        Args:
            hold_for_edit: Whether the run stops at this node.

        Returns:
            A value that never matches the last one while held, and a constant otherwise.
        """
        held = hold_for_edit[0] if isinstance(hold_for_edit, list) else hold_for_edit
        return float("NaN") if held else False

    @classmethod
    def execute(cls, content=None, manual_content=None, viewer_meta=None, view_state=None,
                hold_for_edit=False, hold_timeout=600.0):
        content = expand_lists(as_list(content))
        manual_content = as_list(manual_content)
        # Every input arrives as a list under is_input_list, including the two settings.
        holding = bool(as_list(hold_for_edit)[0]) if as_list(hold_for_edit) else False
        seconds = float(as_list(hold_timeout)[0]) if as_list(hold_timeout) else 600.0
        excluded = excluded_indexes(viewer_meta)
        current = input_hash(content)

        logger.debug(
            "viewer content=%s manual=%s excluded=%s state=%s",
            [to_string(item)[:LOG_EXCERPT] for item in content],
            [to_string(item)[:LOG_EXCERPT] for item in manual_content],
            excluded,
            str(view_state)[:LOG_EXCERPT] if view_state else None,
        )

        stored = None if holding else cached_output(view_state, current)
        if stored is not None:
            return io.NodeOutput(
                stored["output_values"],
                ui={
                    "text": (stored["display_text"],),
                    "source_content": (stored["display_text"],),
                    "content_hash": (stored["content_hash"],),
                },
            )

        # Views ship for Markdown with LaTeX and Mermaid, HTML, SVG, highlighted code, JSON
        # and YAML trees, CSV tables, ANSI logs, image canvases and object inspectors.
        from ...modules.viewer.parsers import handle_all_inputs

        handled = handle_all_inputs(content, logger)
        if handled:
            logger.debug("content claimed by the %s view", handled.get("parser_name", "unknown"))
            display = handled["display_content"]
            if holding:
                from ...modules.interface.pause import wait_for_resume
                from ...modules.viewer.parsers.canvas_parser import CanvasParser

                _, edited = wait_for_resume(
                    str(cls.hidden.unique_id), timeout=seconds,
                    message="edit on the canvas, then send the output",
                    kind="canvas" if CanvasParser.detect_input(content) else "text",
                    content=display,
                )
                if edited and CanvasParser.detect_output(edited):
                    composed = CanvasParser.parse_output(edited, logger)
                    if composed:
                        return cls._output(
                            composed["display_text"], display, composed["content_hash"],
                            composed["output_values"], current,
                        )
            return cls._output(display, display, handled["content_hash"], handled["output_values"], current)

        source = LIST_SEPARATOR.join(to_string(item) for item in content) if content else ""

        if holding:
            from ...modules.interface.pause import wait_for_resume

            shown = (
                LIST_SEPARATOR.join(to_string(item) for item in manual_content)
                if has_content(manual_content) else source
            )
            _, edited = wait_for_resume(
                str(cls.hidden.unique_id), timeout=seconds,
                message="edit the content, then Continue", kind="text", content=shown,
            )
            if edited:
                manual_content = [edited]

        if has_content(manual_content):
            combined = (
                to_string(manual_content[0])
                if len(manual_content) == 1
                else LIST_SEPARATOR.join(to_string(item) for item in manual_content)
            )
            values = combined.split(LIST_SEPARATOR) if LIST_SEPARATOR in combined else [combined]
        elif has_content(content):
            values = [to_string(item) for item in content]
        else:
            return cls._output("", "", "empty_0", [""], current)

        kept = [value for index, value in enumerate(values) if index not in excluded] or [""]
        return cls._output(
            LIST_SEPARATOR.join(values), source, content_hash(source), kept, current
        )

    @classmethod
    def _output(cls, display, source, digest, values, current) -> io.NodeOutput:
        """One return, so every path reports the same four fields to the frontend."""
        return io.NodeOutput(
            values,
            ui={
                "text": (display,),
                "source_content": (source,),
                "content_hash": (digest,),
                "input_hash": (current,),
            },
        )

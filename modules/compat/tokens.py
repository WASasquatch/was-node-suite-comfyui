"""Token expansion for every node's string inputs.

Tokens resolve in each ``STRING`` input and in each string of a list input. A backslash keeps
a bracketed run as text, and an input in :data:`LITERAL_INPUTS` arrives as written.
"""

from __future__ import annotations

import functools
import inspect
import re
from collections.abc import Mapping

from .. import log

__all__ = ["LITERAL_INPUTS", "apply", "expand", "text_inputs"]

logger = log.get_logger("tokens")

#: A run marked as text, and the same run without its backslash.
UNESCAPE = re.compile(r"\\(\[[^\]]*\])")

#: A value shaped like a Windows path, where a separator before a token reads as a mark.
WINDOWS_PATH = re.compile(r"^\\\\|[A-Za-z]:\\")

#: ``io_type`` of the inputs this expands. ``COMBO`` widgets are not expanded.
STRING_TYPE = "STRING"

#: Set on a class once :func:`apply` has looked at it, so a class collected twice is
#: wrapped once.
MARKER = "_was_token_expansion"

#: Node id -> the inputs on it that reach ``execute`` exactly as written: token definitions,
#: validated identifiers, colour strings, state written by a node's own frontend, keys of a
#: listing the pack built, stored library text, and addresses fetched from another host.
LITERAL_INPUTS = {
    "WASComfyViewer": ("manual_content", "viewer_meta", "view_state"),
    "WASImageCurves": ("curve_points",),
    "WASLayersArrange": ("arrangement",),
    "WASCurveToNumbers": ("curve_points",),
    "Mask Rect Area": ("drawn_mask",),
    "Mask Rect Area (Advanced)": ("drawn_mask",),
    "Mask Threshold Region": ("drawn_mask",),
    "Mask Fill Holes": ("drawn_mask",),
    "Mask Dominant Region": ("drawn_mask",),
    "Mask Minority Region": ("drawn_mask",),
    "Text Add Tokens": ("tokens",),
    "Text Add Token by Input": ("token_name", "token_value"),
    "BLIP Model Loader": ("blip_model", "vqa_model_id"),
    "CLIPSeg Model Loader": ("model",),
    "Diffusers Hub Model Down-Loader": ("repo_id", "revision"),
    "Hex to HSL": ("hex_color",),
    "HSL to Hex": ("hsl_color",),
    "WASNSImageTileExtract": ("border_color",),
    "WASNSImageTileShuffle": ("border_color",),
    "WASImageDrawText": ("text_color", "stroke_color", "background_color"),
    "WASDrawImageBounds": ("color",),
    "WASImagePaletteMap": ("palette",),
    "WASNoodleSoupPick": ("picked",),
    "WASSaveSystemPrompt": ("system_prompt", "model_family", "description"),
    "WASZipSave": ("files",),
    "WASLoadImagesFromZIP": ("pad_color",),
    "WASThreeTextureURL": ("url",),
    "WASAudioEditor": ("cuts",),
    "WASAudioEqualizer": ("bands",),
    "WASPikaH3FlexSampling": ("knots",),
    "WASBoundingBoxEditor": ("layout",),
    "WASStringListSections": ("lists",),
    "WASLLMLoadWebPage": ("url",),
    "WASLLMStructuredOutput": ("schema",),
    "WASLLMOpenAI": ("model",),
    "WASLLMAnthropic": ("model",),
    "WASLLMGemini": ("model",),
    "WASLLMProvider": ("model",),
    "WASLLMQueryEngineTool": ("name",),
    "WASLLMRetrieverTool": ("name",),
    "WASLLMAgent": ("name", "can_handoff_to"),
    "WASLLMRunAgent": ("root_agent",),
    "WASLLMChatSession": ("conversation",),
    "WASLLMMCPTools": ("allowed_tools", "prefix"),
    "WASLLMLoadTable": ("table_name", "sheet"),
    "WASLLMTableToDocuments": ("table",),
    "WASLLMVectorStore": ("collection",),
    "WASLLMIndexNode": ("target",),
    "WASLLMToDocuments": ("title",),
}


def text_inputs(schema) -> tuple[str, ...]:
    """The inputs on one schema whose value arrives with its tokens expanded.

    Args:
        schema: The node's ``io.Schema``.

    Returns:
        Input ids, in schema order. Empty when the node takes no string input, or when
        every string input it takes is listed in :data:`LITERAL_INPUTS`.
    """
    literal = LITERAL_INPUTS.get(schema.node_id, ())
    return tuple(
        spec.id
        for spec in (schema.inputs or ())
        if getattr(spec, "io_type", None) == STRING_TYPE and spec.id not in literal
    )


def expand(values: Mapping, names) -> dict:
    """A copy of ``values`` with the tokens in ``names`` replaced.

    Args:
        values: The keyword arguments bound for one ``execute`` call.
        names: Input ids that expand, from :func:`text_inputs`.

    Returns:
        ``values`` as a plain dict. A list keeps its length and order, each string in it
        expanded. A value with no string holding ``[`` is the same object passed in.
    """
    pending = [name for name in names if _holds_token(values.get(name))]
    if not pending:
        return dict(values)
    table = _table()
    if table is None:
        return dict(values)
    expanded = dict(values)
    for name in pending:
        value = expanded[name]
        if isinstance(value, list):
            expanded[name] = [
                _expand_text(table, item, name) if _expandable(item) else item for item in value
            ]
        else:
            expanded[name] = _expand_text(table, value, name)
    return expanded


def _expand_text(table, text: str, name: str) -> str:
    """One string with its tokens replaced, or as written when expansion raises.

    Args:
        table: The :class:`TextTokens` for this execution.
        text: The value to expand.
        name: The input it came from, for the line logged on a failure.

    Returns:
        The expanded text, or ``text`` unchanged.
    """
    try:
        return _unescape(table.parseTokens(text), name)
    except Exception as error:
        logger.warning(
            "the tokens in %s could not be expanded (%s), so it was left as written",
            name, error,
        )
        return text


def _unescape(text: str, name: str) -> str:
    """Drop the backslash from every run marked as text.

    Args:
        text: One value, with its tokens already replaced.
        name: The input it came from, for the line logged about a path.

    Returns:
        ``text`` with one backslash removed from each marked run.
    """
    if "\\" not in text:
        return text
    marked = UNESCAPE.findall(text)
    if marked and WINDOWS_PATH.search(text):
        logger.warning(
            "a backslash before %s in %s marks it as text, so it was not expanded and the "
            "backslash was dropped. Write the path with forward slashes to expand it, as in "
            "C:/renders/%s/out.png",
            marked[0], name, marked[0],
        )
    return UNESCAPE.sub(r"\1", text)


def apply(node_cls, schema=None) -> None:
    """Wrap ``node_cls.execute`` so its string inputs arrive with tokens expanded.

    Args:
        node_cls: A node class the loader has collected.
        schema: Its schema, already built. Left out, it is built here.
    """
    if getattr(node_cls, MARKER, False):
        return
    try:
        _wrap(node_cls, schema)
    except Exception as error:
        logger.debug("%s was left unwrapped (%s)", getattr(node_cls, "__name__", node_cls), error)


def _wrap(node_cls, schema=None) -> None:
    """Wrap ``execute`` on one class. :func:`apply` catches what this raises."""
    names = text_inputs(schema if schema is not None else node_cls.GET_SCHEMA())
    setattr(node_cls, MARKER, True)
    if not names:
        return
    declared = node_cls.__dict__.get("execute")
    if not isinstance(declared, classmethod):
        # An execute() inherited from a shared base, or none at all, is left unwrapped.
        logger.debug("%s does not declare execute(), so its tokens are not expanded", node_cls.__name__)
        return
    function = declared.__func__

    if inspect.iscoroutinefunction(function):

        @functools.wraps(function)
        async def execute(cls, *args, **kwargs):
            return await function(cls, *(args), **expand(kwargs, names))

    else:

        @functools.wraps(function)
        def execute(cls, *args, **kwargs):
            return function(cls, *(args), **expand(kwargs, names))

    node_cls.execute = classmethod(execute)


def _expandable(value) -> bool:
    """Is this value a string that could hold a token?"""
    return isinstance(value, str) and "[" in value


def _holds_token(value) -> bool:
    """Is this value, or any item of it when it is a list, a string that could hold a token?"""
    if isinstance(value, list):
        return any(_expandable(item) for item in value)
    return _expandable(value)


def _table():
    """A :class:`TextTokens` for this execution, or ``None`` if one cannot be built."""
    try:
        from ..state.tokens import TextTokens

        return TextTokens()
    except Exception as error:
        logger.warning(
            "the token table is unavailable (%s), so [tokens] were left as written", error
        )
        return None

"""Token expansion for the string inputs of every node in the pack.

Built-in and custom tokens resolve in any ``STRING`` input. A backslash keeps a bracketed
run as text, and an input in :data:`LITERAL_INPUTS` arrives as written.
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

#: ``io_type`` of the inputs this expands. ``COMBO`` widgets are also strings and are
#: deliberately not included: their value is one of a fixed option list, and a token in it
#: could only ever produce a value the node rejects.
STRING_TYPE = "STRING"

#: Set on a class once :func:`apply` has looked at it, so a class collected twice is
#: wrapped once. Wrapping twice would be harmless, expansion is idempotent, but it would
#: put a second frame in every traceback raised from a node body.
MARKER = "_was_token_expansion"

#: Node id -> the inputs on it that must reach ``execute`` exactly as the user wrote them.
#:
#: Five kinds of field are listed, and nothing else is:
#:
#: * the token-definition widgets. `Text Add Tokens` reads ``[name]: value`` lines and
#:   `Text Add Token by Input` takes a name and a value, so expanding either would define
#:   a token named after some earlier token's value, or freeze a value that is meant to be
#:   stored as written.
#: * validated identifiers. A repository id is checked against a grammar and then joined
#:   onto the models directory; a git revision names a ref. An expanded ``[time]`` passes
#:   that grammar and turns a typo into a lookup for a repository that cannot exist, which
#:   is a worse error than the one the raw value produces.
#: * colour strings. ``#RRGGBB`` and ``hsl(...)`` are parsed as colours, and digits from a
#:   token parse as a different colour rather than as a mistake.
#: * machine state written by a node's own frontend. The viewer's three widgets carry JSON
#:   and edited content the frontend round-trips, so a value that came back changed would
#:   drift from what is on screen. ``drawn_mask``, on every mask node that takes a brush,
#:   carries a base64 PNG behind a header, and a token in either would make the run decode
#:   a different picture from the one on the node.
#: * keys of a listing the pack built. The archive selection holds one menu label per line,
#:   each ending in ``[input]``, ``[output]`` or ``[temp]``, and a custom token by one of
#:   those names would rewrite the tag, after which the label names no file at all. The
#:   terminology pick list is the same: each line names a stored terminology and one of its
#:   words, and a rewritten name matches nothing in the pantry.
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
    "WASImageTileExtract": ("border_color",),
    "WASImageTileShuffle": ("border_color",),
    "WASImageDrawText": ("text_color", "stroke_color", "background_color"),
    "WASDrawImageBounds": ("color",),
    "WASImagePaletteMap": ("palette",),
    "WASNoodleSoupPick": ("picked",),
    "WASZipSave": ("files",),
    "WASLoadImagesFromZIP": ("pad_color",),
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
        ``values`` as a plain dict. Values that are not strings, and strings holding no
        ``[``, are carried over untouched, so a node whose text has no token is handed
        the identical object it would have been handed with none of this in place.
    """
    pending = [name for name in names if _expandable(values.get(name))]
    if not pending:
        return dict(values)
    table = _table()
    if table is None:
        return dict(values)
    expanded = dict(values)
    for name in pending:
        try:
            expanded[name] = _unescape(table.parseTokens(expanded[name]), name)
        except Exception as error:
            # One unparseable value must not take the execution with it: the node is handed
            # what the user wrote, which is what it would have been handed before this.
            logger.warning(
                "the tokens in %s could not be expanded (%s), so it was left as written",
                name, error,
            )
    return expanded


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


def apply(node_cls) -> None:
    """Wrap ``node_cls.execute`` so its string inputs arrive with tokens expanded.

    Args:
        node_cls: A node class the loader has collected.
    """
    if getattr(node_cls, MARKER, False):
        return
    try:
        _wrap(node_cls)
    except Exception as error:
        logger.debug("%s was left unwrapped (%s)", getattr(node_cls, "__name__", node_cls), error)


def _wrap(node_cls) -> None:
    """Do the wrapping. Split out so :func:`apply` owns the one guard around all of it."""
    names = text_inputs(node_cls.GET_SCHEMA())
    setattr(node_cls, MARKER, True)
    if not names:
        return
    declared = node_cls.__dict__.get("execute")
    if not isinstance(declared, classmethod):
        # Inherited from a shared base, or absent. Wrapping the inherited function here
        # would bind one node's input names onto every sibling that shares the base.
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

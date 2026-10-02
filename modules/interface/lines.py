"""A text file's lines, as JSON a node interface can draw.

``GET /was/interface/api/text_lines?file=<menu label>`` answers the lines of a listed text
file, or 404. An answer holds at most :data:`MAX_LINES` lines of :data:`MAX_LINE_CHARS`
characters.
"""

from __future__ import annotations

import codecs
import os
import random
import threading

from .. import log
from ..util import sandbox, text_files
from .channel import NO_STORE

__all__ = [
    "MAX_BYTES",
    "MAX_LINES",
    "MAX_LINE_CHARS",
    "MAX_TOTAL_LINES",
    "ROUTE",
    "lines_payload",
    "register_routes",
]

logger = log.get_logger("interface.lines")

#: The one route serving a listed text file's lines.
ROUTE = "/was/interface/api/text_lines"

#: Lines in one answer, and the largest ``limit`` a request can ask for. More than fits a
#: node-sized panel at any zoom, and it holds an answer near 200 KB.
MAX_LINES = 500

#: Characters of one line that are returned. A longer line is cut and flagged, which is
#: wider than a node can draw and stops one minified JSON line filling the answer.
MAX_LINE_CHARS = 400

#: Lines counted at all. Past this the answer says it was truncated and counts this many.
MAX_TOTAL_LINES = 20000

#: Bytes read from the file. Read as one bounded prefix rather than line by line, so a 50 MB
#: minified JSON costs this and not its whole size, and one request's work is bounded
#: whatever it names.
MAX_BYTES = 4 * 1024 * 1024

#: Answered for any label that is not in the listing, so the route says nothing about
#: whether a path it does not list exists.
UNLISTED = "that file is not one this pack lists"

#: Answered for a listed file that could not be opened or read.
UNREADABLE = "that file could not be read"

#: Query values read as false, and as true. Anything else leaves the parameter at its
#: default, since a malformed request is answered rather than refused.
FALSE = frozenset({"0", "false", "no", "off"})
TRUE = frozenset({"1", "true", "yes", "on"})

#: Serializes the file reads, so several requests arriving together cost one
#: :data:`MAX_BYTES` buffer at a time rather than one each. The route runs on the server's
#: thread while a prompt may be running.
_lock = threading.Lock()

_registered = False


def lines_payload(label, start=0, limit=MAX_LINES, skip_comments=True, seed=None) -> dict | None:
    """One listed text file's lines, as the object the route answers with.

    Args:
        label: The exact combo label the listing offers. Anything not in it answers None.
        start: First line of the window, counting every line in the file, comment lines
            included. Clamped into the file.
        limit: How many lines the window holds, clamped to :data:`MAX_LINES`.
        skip_comments: Whether the reading node is dropping comment lines, which decides
            the ``index`` each line carries and what ``kept`` counts.
        seed: A seed to report the random draw for, or None to report none.

    Returns:
        ``{"total", "kept", "start", "lines", "random_index", "clipped", "lossy",
        "truncated", "revision"}``, or None when the label is not listed, the path is
        outside every permitted read root, or the file could not be read. ``total`` counts
        every line, ``kept`` counts the ones the node will index, and each line carries the
        ``index`` the node will use for it, or None where the node drops it.
    """
    path = text_files.resolve(label if isinstance(label, str) else "")
    if path is None:
        return None
    try:
        # A label names a file inside a listed root, each a permitted read root; the gate is
        # here, not in the listing.
        resolved = sandbox.resolve_read(path)
    except sandbox.PathNotAllowed as error:
        logger.debug("%s refused %s (%s)", ROUTE, path, error)
        return None
    try:
        with _lock:
            stat = os.stat(resolved)
            with open(resolved, "rb") as handle:
                data = handle.read(MAX_BYTES + 1)
    except OSError as error:
        logger.debug("%s could not read %s (%s)", ROUTE, resolved, error)
        return None

    over_budget = len(data) > MAX_BYTES
    text, lossy = _decode(data[:MAX_BYTES], final=not over_budget)
    lines = text_files.split_lines(text)
    # The last entry of an over-budget read stopped at the byte budget rather than at a
    # line break, so it is part of a line. It is kept and flagged rather than dropped: a
    # file that is one enormous line would otherwise answer with no lines at all.
    partial = over_budget and bool(lines) and not text.endswith("\n")
    truncated = over_budget or len(lines) > MAX_TOTAL_LINES
    if len(lines) > MAX_TOTAL_LINES:
        lines = lines[:MAX_TOTAL_LINES]
        partial = False
    return _payload(lines, stat, start, limit, skip_comments, seed, lossy, truncated, partial)


def register_routes() -> bool:
    """Register the route serving a listed text file's lines.

    Returns:
        True when the route was registered. False when it was registered already, or when
        the server could not be reached, in which case an interface asking for a file's
        lines gets a failed request.
    """
    global _registered
    if _registered:
        return False
    try:
        from aiohttp import web
        from server import PromptServer

        @PromptServer.instance.routes.get(ROUTE)
        async def get_text_lines(request):
            label = request.query.get("file")
            payload = lines_payload(
                label,
                start=_int(request.query.get("start"), 0),
                limit=_int(request.query.get("limit"), MAX_LINES),
                skip_comments=_flag(request.query.get("skip_comments"), True),
                seed=_int(request.query.get("seed"), None),
            )
            if payload is not None:
                return web.json_response(payload, headers=NO_STORE)
            # A label nobody listed, one refused by the containment layer and one whose
            # file has since gone all answer 404. The two messages separate a menu that has
            # moved on from a file that has, and neither carries a path.
            listed = text_files.resolve(label if isinstance(label, str) else "") is not None
            return web.Response(
                status=404, text=UNREADABLE if listed else UNLISTED, headers=NO_STORE
            )

    except Exception as error:
        logger.warning(
            "%s was not registered (%s: %s), so a node interface asking for a text file's "
            "lines gets a failed request",
            ROUTE, type(error).__name__, error,
        )
        logger.debug("%s could not be registered", ROUTE, exc_info=True)
        return False
    _registered = True
    logger.debug("%s is serving text file lines", ROUTE)
    return True


def _decode(data: bytes, final: bool) -> tuple[str, bool]:
    """Bytes as text, and whether anything in them was not UTF-8.

    Args:
        data: The prefix that was read.
        final: Whether ``data`` is the whole file. False holds back a character the byte
            budget cut in half, so a truncated read of a valid file is not reported as one
            that is not UTF-8.

    Returns:
        ``(text, lossy)``. On ``lossy`` the text carries a replacement character wherever
        the bytes were not UTF-8, which the node reading the same file will stop on.
    """
    try:
        text = codecs.getincrementaldecoder(text_files.ENCODING)("strict").decode(data, final)
    except UnicodeDecodeError:
        text = codecs.getincrementaldecoder(text_files.ENCODING)("replace").decode(data, final)
        return text_files.normalize_newlines(text), True
    return text_files.normalize_newlines(text), False


def _payload(lines, stat, start, limit, skip_comments, seed, lossy, truncated, partial) -> dict:
    """Build the answer out of the lines already read.

    Args:
        lines: Every line of the file, already bounded.
        stat: The file's ``os.stat_result``, which the revision is built from.
        start: Requested first line of the window.
        limit: Requested window size.
        skip_comments: Whether the reading node drops comment lines.
        seed: A seed to report the random draw for, or None.
        lossy: Whether the bytes were not UTF-8 and were decoded with replacements.
        truncated: Whether the file is longer than the answer covers.
        partial: Whether the last line is only the part of it that was read.

    Returns:
        The answer object :func:`lines_payload` describes.
    """
    total = len(lines)
    comments = [skip_comments and text_files.is_comment(line) for line in lines]
    # The index each line will carry in the node, worked out here so the numbering cannot
    # drift when comments are dropped, and so the comment test is never re-derived in
    # another language: str.strip() and trimStart() disagree on which characters are space.
    indices: list[int | None] = []
    kept = 0
    for comment in comments:
        indices.append(None if comment else kept)
        kept += not comment

    window = max(0, min(_whole(limit, MAX_LINES), MAX_LINES))
    first = max(0, min(_whole(start, 0), total))
    clipped = 0
    rows = []
    for offset in range(first, min(first + window, total)):
        text = lines[offset]
        cut = len(text) > MAX_LINE_CHARS or (partial and offset == total - 1)
        clipped += cut
        rows.append(
            {
                "text": text[:MAX_LINE_CHARS] if cut else text,
                "index": indices[offset],
                "comment": comments[offset],
                "clipped": cut,
            }
        )

    return {
        "total": total,
        "kept": kept,
        "start": first,
        "lines": rows,
        # A private generator rather than the shared random module: seeding that one from
        # the server's thread would move a draw inside a node part way through a run. It
        # answers the index random.seed(seed) then random.choice(lines) selects.
        "random_index": random.Random(seed).randrange(kept) if seed is not None and kept else None,
        "clipped": clipped,
        "lossy": lossy,
        "truncated": truncated,
        "revision": f"{stat.st_size}:{stat.st_mtime_ns}",
    }


def _whole(value, default: int) -> int:
    """A value as an integer, or ``default`` when it is not one."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _int(value, default):
    """A query value as an integer, or ``default`` when it is missing or not one."""
    if value is None:
        return default
    return _whole(str(value).strip(), default)


def _flag(value, default: bool) -> bool:
    """A query value as a boolean, or ``default`` when it is missing or not one."""
    if value is None:
        return default
    text = str(value).strip().casefold()
    if text in FALSE:
        return False
    if text in TRUE:
        return True
    return default

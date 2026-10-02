"""What a node's last run did, as JSON that node's interface can draw.

``GET /was/interface/api/run_result?node_id=<id>`` answers a status, counts, facts, bodies and
sampled items, or 204. ``GET /was/interface/api/run_result_page`` answers a range of one body's
lines. :data:`MAX_ENTRIES` nodes hold a result.
"""

from __future__ import annotations

import json
import math
import re
import threading
import zlib
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from itertools import islice

from .. import log
from .channel import (
    NO_STORE,
    REFUSAL_HEADER,
    executing_class_type,
    executing_node_id,
    node_key,
    watching,
)

__all__ = [
    "CONTEXT_CHARS",
    "ELLIPSIS",
    "ERROR",
    "MAX_BODIES",
    "MAX_BODY_CHARS",
    "MAX_BYTES",
    "MAX_COUNTS",
    "MAX_ENTRIES",
    "MAX_FACTS",
    "MAX_INPUTS",
    "MAX_ITEMS",
    "MAX_KEPT_BODY_CHARS",
    "MAX_KEPT_NODE_CHARS",
    "MAX_KEPT_TOTAL_CHARS",
    "MAX_LABEL_CHARS",
    "MAX_MARKS",
    "MAX_PAGE_CHARS",
    "MAX_PAGE_LINES",
    "MAX_SUMMARY_CHARS",
    "MAX_TALLIES",
    "MAX_TEXT_CHARS",
    "OK",
    "PAGE_ROUTE",
    "ROUTE",
    "STATUSES",
    "WARNING",
    "WINDOW_LEAD",
    "body",
    "excerpt",
    "given",
    "page",
    "publish",
    "register_routes",
    "result",
    "watching",
    "window",
]

logger = log.get_logger("interface.run_result")

#: The one route serving what a node published, keyed by a ``node_id`` query parameter.
ROUTE = "/was/interface/api/run_result"

#: The route serving a range of lines from one body of what a node published, keyed by the same
#: ``node_id`` and by the body's place in the report.
PAGE_ROUTE = "/was/interface/api/run_result_page"

#: Answered for a node that has published nothing, and for a report filed under its id by a node
#: of another kind.
NO_REPORT = "that node has not published a run result in this session"

#: The run did what the node is for.
OK = "ok"

#: The run finished and produced something a person would want to see before using it: no
#: match, nothing written, every entry skipped.
WARNING = "warning"

#: The run finished but part of what it was asked to do did not happen.
ERROR = "error"

#: The statuses a result may carry. A result carrying anything else is refused whole, since the
#: status is what an interface draws the rest of the result inside.
STATUSES = (OK, WARNING, ERROR)

#: How many nodes hold a result at once. This process outlives every prompt, so the store is
#: bounded rather than left to grow with the graph, and a result is small enough that more nodes
#: are held than the picture store holds.
MAX_ENTRIES = 32

#: Named numbers one result carries, such as how many entries were found and how many were
#: written. The ones past this are dropped and ``truncated`` carries ``"counts"``.
MAX_COUNTS = 8

#: Named numbers one result carries as a breakdown rather than as a headline, such as one total
#: per pattern of a search. A tally names the thing it counts, so two of them may be named alike
#: and the order they were given in is the order they are drawn. The ones past this are dropped,
#: ``tallies_total`` still counting every one the run made, and ``truncated`` carries
#: ``"tallies"``.
MAX_TALLIES = 8

#: Named strings one result carries, such as which input a value arrived on. The ones past this
#: are dropped and ``truncated`` carries ``"facts"``.
MAX_FACTS = 8

#: Values one result names as the ones a run was handed, such as the text it searched. The
#: ones past this are dropped and ``truncated`` carries ``"inputs"``.
MAX_INPUTS = 8

#: Sample rows one result carries. A node with more hands over the first ones and states the
#: whole number in ``items_total``, and ``truncated`` carries ``"items"``.
MAX_ITEMS = 8

#: Characters of a count name, a fact name, a fact value or an item's note. A longer one is cut
#: to this and ``truncated`` carries ``"text"``.
MAX_LABEL_CHARS = 64

#: Characters of an item's own text, ellipses included. A longer one is cut to this, its
#: ``mark`` is clamped into what is left, and ``truncated`` carries ``"text"``.
MAX_TEXT_CHARS = 240

#: Characters of the summary line. A longer one is cut to this and ``truncated`` carries
#: ``"summary"``.
MAX_SUMMARY_CHARS = 200

#: Bodies one result carries, each a text with its own marked spans, such as what a run read
#: and what it wrote, or one entry of a socket carrying many. The ones past this are dropped
#: and ``truncated`` carries ``"bodies"``.
MAX_BODIES = 16

#: Characters of the piece of a body a result carries. A longer body publishes a window of this
#: many instead, ``offset`` saying where the window opens and ``length`` how long the whole text
#: is, ``whole`` reads False, and ``truncated`` carries ``"body_text"``.
MAX_BODY_CHARS = 8000

#: Marked spans one body carries. The ones past this are dropped, ``marks_total`` still counting
#: every span the whole text holds, and ``truncated`` carries ``"marks"``.
MAX_MARKS = 64

#: Characters :func:`window` keeps before the index it opens on, so a window onto a long body
#: reads as the text leading into that span rather than starting on it.
WINDOW_LEAD = 80

#: Bytes of one result, measured as the route serialises it. Items are dropped from the end
#: until the rest fits, then bodies, each drop naming itself in ``truncated``. A result of
#: plain text sits inside this with every other bound at its limit: 16 KB of bodies, 2 KB of
#: sample rows and 3 KB of tallies. Text that escapes to six bytes a character reaches it,
#: and the bodies past the fit are dropped and named.
MAX_BYTES = 64 * 1024

#: Lines of one body a page answers, and the largest ``count`` a request may name. A larger
#: number is served as this many.
MAX_PAGE_LINES = 500

#: Characters of one page. Lines are answered until the next would pass this, so a page of long
#: lines holds fewer than :data:`MAX_PAGE_LINES` of them, and a single line longer than this is
#: cut to it. Either answer reads ``clipped`` True.
MAX_PAGE_CHARS = 64 * 1024

#: Characters of one body held for paging beside the report it was cut for. A longer body holds
#: this many, back to the last line break in them, and the lines past that are not reachable:
#: a page says so by counting fewer in ``held`` than in ``total``.
MAX_KEPT_BODY_CHARS = 512 * 1024

#: Characters held for paging across one report. Its bodies are held in the order they were
#: published until the next would pass this, and the rest hold nothing.
MAX_KEPT_NODE_CHARS = 1024 * 1024

#: Characters held for paging across every node. Publishing drops the least recently read node's
#: held text until the new report fits, and the report that text belonged to stays, answering as
#: a body nothing is held for.
MAX_KEPT_TOTAL_CHARS = 8 * 1024 * 1024

#: Characters of the text kept on each side of a marked span by :func:`excerpt`, so a sample
#: reads as a phrase rather than as the match alone.
CONTEXT_CHARS = 48

#: What marks a cut inside an item's text. Written into the text rather than left implicit, so
#: an interface draws the answer as it stands and the ``mark`` still points at the right
#: characters.
ELLIPSIS = "..."

#: A UTF-16 half with no pair, which has no UTF-8 encoding. A browser writes U+FFFD in place
#: of one, so :func:`given` measures the same bytes on both sides by doing the same.
_LONE_SURROGATE = re.compile("[\ud800-\udfff]")

# A node publishes on the thread running the prompt and the route answers on the server's, so
# every read and write of the store below goes through this.
_lock = threading.Lock()

#: Node id to the result that node last published, most recently used last.
_results: OrderedDict[str, dict] = OrderedDict()

#: Node id to the lines of each body of that result, in the same order the report carries them,
#: as ``{"bodies": [[line, ...], ...], "chars": int}``. Held beside the report rather than in it,
#: so what a first draw fetches is the report alone, and dropped with the report it belongs to.
_pages: OrderedDict[str, dict] = OrderedDict()

#: Counts every result stored in this process, so an interface polling the route can tell a
#: fresh answer from the one it already drew.
_run = 0

_registered = False


def publish(
    status=OK,
    summary="",
    counts=None,
    tallies=None,
    tallies_total=None,
    facts=None,
    items=None,
    items_total=None,
    bodies=None,
    inputs=None,
    node_id=None,
) -> bool:
    """Store what a run did, for the publishing node's own interface to fetch.

    Never raises, and never touches the values it is given.

    Args:
        status: One of :data:`STATUSES`. Anything else refuses the whole result.
        summary: One line saying what the run did, written for the person running the pack.
        counts: Named numbers, as a mapping of name to a finite int or float.
        tallies: A breakdown, as a sequence of mappings each carrying ``name`` and ``value``.
            One mapping is read as a sequence of one. Names may repeat, so a run naming two of
            its parts alike reports both.
        tallies_total: How many the breakdown held before the node took a sample. Left out, the
            number handed over stands in for it, and an iterator is read no further than one
            past :data:`MAX_TALLIES`.
        facts: Named strings, as a mapping of name to a string, an int or a float.
        items: Sample rows, as a sequence of strings or of the mappings :func:`excerpt`
            builds. One string or one mapping is read as a sequence of one.
        items_total: How many rows there were before the node took a sample. Left out, the
            number handed over stands in for it, and an iterator is read no further than one
            row past :data:`MAX_ITEMS`.
        bodies: Texts with spans marked, as a sequence of the mappings :func:`body` builds.
            One mapping is read as a sequence of one. The ``source`` of each is held for
            :func:`page` and left out of the result.
        inputs: The values the run was handed, as a sequence of the mappings :func:`given`
            builds. One mapping is read as a sequence of one.
        node_id: The publishing node's graph id. Left out, the id of the node ComfyUI is
            executing is read from its execution context.

    Returns:
        True when a result was stored. False when no browser is connected, when the status is
        not one of :data:`STATUSES`, when no node id could be found, and when the result could
        not be built, each of which costs the readout and nothing else.
    """
    global _run
    if not watching():
        return False
    try:
        if status not in STATUSES:
            logger.debug(
                "a run result was published with status %r, which is not one of %s",
                status, ", ".join(STATUSES),
            )
            return False
        key = node_key(node_id if node_id is not None else executing_node_id())
        if key is None:
            logger.debug("a run result was published with no node id to file it under")
            return False
        payload, sources = _payload(
            status, summary, counts, tallies, tallies_total, facts, items, items_total,
            bodies, inputs,
        )
        declared = executing_class_type(node_id)
        with _lock:
            _run += 1
            payload["run"] = _run
            # Graph ids are handed out again after a graph is cleared, so a new node can carry
            # an id an unrelated one published under. The class is stored beside the report and
            # the route refuses to hand it to a node of another kind.
            if declared:
                payload["node_type"] = declared
            # Fitted with the counter already written in, so what is measured is the answer
            # the route sends rather than one field short of it.
            if not _fit(payload):
                return False
            # Reinserted rather than assigned, so publishing again makes that node the most
            # recent and the oldest is the one evicted.
            _results.pop(key, None)
            _results[key] = payload
            # The bodies the fit dropped are dropped from what is held too, so a page index is
            # the body's place in the report that was sent.
            _keep(key, sources[: len(payload["bodies"])])
            while len(_results) > MAX_ENTRIES:
                evicted, _ = _results.popitem(last=False)
                _pages.pop(evicted, None)
        return True
    except Exception as error:
        logger.debug("no run result was published for node %s (%s)", node_id, error)
        return False


def result(node_id) -> dict | None:
    """The result a node published on its last run.

    Args:
        node_id: A node's graph id, as a string or an integer. Anything else, including a
            missing or malformed query value, answers None.

    Returns:
        The stored result object, or None when that node has published nothing in this
        session.
    """
    key = node_key(node_id)
    if key is None:
        return None
    with _lock:
        payload = _results.get(key)
        if payload is not None:
            _results.move_to_end(key)
            # The two stores are ordered alike, so the text dropped for the ceiling belongs to
            # the report that is about to be evicted rather than to one being read.
            if key in _pages:
                _pages.move_to_end(key)
    return payload


def page(node_id, body=0, start=0, count=MAX_PAGE_LINES, node_type=None) -> dict | None:
    """A range of lines from one body of the result a node published on its last run.

    Args:
        node_id: A node's graph id, as a string or an integer. Anything else, including a
            missing or malformed query value, answers None.
        body: Which body of the report, counting from zero in the order it carries them.
        start: The first line wanted, counting from zero. Past the last line answers no lines.
        count: How many lines to answer, held to :data:`MAX_PAGE_LINES`.
        node_type: The asking node's class. A report filed by a node of another kind answers
            None. Left out, whatever is filed under the id is read.

    Returns:
        ``{"name", "start", "lines", "total", "held", "clipped", "run"}``: the body's name, the
        line the page opens on, the lines themselves, how many lines the whole body holds, how
        many of them are still held for paging, whether a line was cut to fit the page, and the
        run the report was published on. A body the report does not carry, a range past the end
        and a body nothing is held for each answer a page of no lines. None when that node has
        published nothing in this session and when what it published was published by a node of
        another kind.
    """
    payload = result(node_id)
    if payload is None or _mismatched(payload, node_type):
        return None
    try:
        index = _index(body, 0)
        first = _index(start, 0)
        wanted = min(_index(count, MAX_PAGE_LINES), MAX_PAGE_LINES)
        blocks = payload.get("bodies") or []
        block = blocks[index] if index < len(blocks) else None
        with _lock:
            entry = _pages.get(node_key(node_id))
            kept = (entry or {}).get("bodies") or []
            held = kept[index] if index < len(kept) else []
            lines, clipped = _fit_page(held[first:first + wanted])
        stated = _index(block.get("lines"), 0) if isinstance(block, Mapping) else 0
        return {
            "name": block.get("name", "") if isinstance(block, Mapping) else "",
            "start": first,
            "lines": lines,
            "total": max(stated, len(held)),
            "held": len(held),
            "clipped": clipped,
            "run": _index(payload.get("run"), 0),
        }
    except Exception as error:
        logger.debug("no page of node %s was built (%s)", node_id, error)
        return _blank_page()


def excerpt(source, start, end, note=None, context=CONTEXT_CHARS) -> dict:
    """One span of a longer text as a sample row, with the characters around it.

    Args:
        source: The text the span was found in.
        start: Index of the span's first character.
        end: Index one past its last character. A span of no width marks a position.
        note: A short label for the row, such as the line the span sits on.
        context: Characters kept on each side of the span.

    Returns:
        A row for :func:`publish`: ``text`` carrying the span and its surroundings, ``mark``
        as ``[start, end]`` inside that text, ``note``, and ``clipped``, which is True when
        the row stands for more than it holds. :data:`ELLIPSIS` marks every cut inside
        ``text``, and the whole of it is at most :data:`MAX_TEXT_CHARS` characters.
    """
    text = source if isinstance(source, str) else str(source)
    room = max(0, int(context))
    first = max(0, min(int(start), len(text)))
    last = max(first, min(int(end), len(text)))
    # The span keeps whatever the two context windows leave, so the row is within the bound
    # before it reaches the normaliser and the mark cannot be clamped off the end of it.
    span_room = max(0, MAX_TEXT_CHARS - 2 * room)
    span = text[first:last]
    clipped = len(span) > span_room
    if clipped:
        span = span[: max(0, span_room - len(ELLIPSIS))] + ELLIPSIS
    before = text[max(0, first - room):first]
    if first > len(before):
        clipped = True
        if room > len(ELLIPSIS):
            before = ELLIPSIS + before[len(ELLIPSIS):]
    after = text[last:last + room]
    if last + len(after) < len(text):
        clipped = True
        if room > len(ELLIPSIS):
            after = after[: len(after) - len(ELLIPSIS)] + ELLIPSIS
    return {
        "text": before + span + after,
        "mark": [len(before), len(before) + len(span)],
        "note": note,
        "clipped": clipped,
    }


def window(length, anchor=0) -> tuple[int, int]:
    """Which characters of a body a result carries.

    Args:
        length: Characters in the whole text.
        anchor: An index the window has to hold, such as where its first marked span starts.

    Returns:
        ``(start, stop)`` into the whole text: all of a text inside :data:`MAX_BODY_CHARS`,
        and otherwise :data:`MAX_BODY_CHARS` characters opening :data:`WINDOW_LEAD` before
        ``anchor`` and held inside the text at both ends.
    """
    size = max(0, int(length))
    if size <= MAX_BODY_CHARS:
        return 0, size
    start = max(0, min(int(anchor) - WINDOW_LEAD, size - MAX_BODY_CHARS))
    return start, start + MAX_BODY_CHARS


def body(name, source, marks=(), marks_total=None, start=0) -> dict:
    """One text with spans marked, as much of it as a result carries.

    Args:
        name: What the text is, such as ``"before"``.
        source: The whole text.
        marks: Spans in it, each a pair of indices into the whole text, in the order they
            occur. A pair of equal indices marks a position.
        marks_total: How many spans the whole text holds. Left out, the number of spans
            handed over stands in for it.
        start: Where the piece to carry opens, which is :func:`window`'s first answer.

    Returns:
        A body for :func:`publish`: ``name``, ``text`` as the piece carried, ``marks``
        clipped to that piece and rebased into it, ``marks_total``, ``offset`` as ``start``,
        ``length`` as the whole text's, ``lines`` as how many lines the whole text holds,
        ``source`` as the whole text, which is held for :func:`page` and never sent, and
        ``whole``, True when ``text`` is all of it.
    """
    text = source if isinstance(source, str) else str(source)
    first = max(0, min(int(start), len(text)))
    piece = text[first:first + MAX_BODY_CHARS]
    stop = first + len(piece)
    try:
        given = list(islice(iter(marks), MAX_MARKS)) if marks is not None else []
    except TypeError:
        given = []
    kept = []
    for span in given:
        try:
            begin, end = int(span[0]), int(span[1])
        except (IndexError, KeyError, TypeError, ValueError):
            continue
        end = max(begin, end)
        if begin == end:
            # A span of no width marks a position, so one sitting at either edge of the piece
            # is kept where a span clipped to nothing is dropped.
            if not first <= begin <= stop:
                continue
            low = high = begin
        else:
            low, high = max(begin, first), min(end, stop)
            if low >= high:
                continue
        kept.append([low - first, high - first])
    stated = isinstance(marks_total, int) and not isinstance(marks_total, bool)
    return {
        "name": name,
        "text": piece,
        "marks": kept,
        "marks_total": max(marks_total if stated else len(given), len(kept)),
        "offset": first,
        "length": len(text),
        "lines": _lines_in(text),
        "source": text,
        "whole": first == 0 and stop == len(text),
    }


def given(name, value, linked=None) -> dict:
    """One value a run was handed, named rather than carried.

    Args:
        name: What the value filled, such as the input's name as the schema spells it.
        value: The value itself, which is read and not kept.
        linked: True when it arrived from another node's output, False when it is what the
            widget beside that input holds, None when neither is known.

    Returns:
        An entry for :func:`publish`: ``name``, ``bytes`` as the length of the value's UTF-8
        encoding, ``checksum`` as the CRC-32 of those bytes in eight hexadecimal characters,
        and ``linked`` when it was given. Two different values answer alike only when they
        share both the length and the checksum.
    """
    text = value if isinstance(value, str) else str(value)
    try:
        data = text.encode("utf-8")
    except UnicodeEncodeError:
        data = _LONE_SURROGATE.sub("\ufffd", text).encode("utf-8")
    entry = {"name": name, "bytes": len(data), "checksum": f"{zlib.crc32(data):08x}"}
    if linked is not None:
        entry["linked"] = bool(linked)
    return entry


def register_routes() -> bool:
    """Register the routes serving published run results and their pages.

    Returns:
        True when the routes were registered. False when they were registered already, or when
        the server could not be reached, in which case an interface asking for a result gets a
        failed request.
    """
    global _registered
    if _registered:
        return False
    try:
        from aiohttp import web
        from server import PromptServer

        @PromptServer.instance.routes.get(ROUTE)
        async def get_run_result(request):
            payload = result(request.query.get("node_id"))
            if payload is not None and _mismatched(payload, request.query.get("node_type")):
                payload = None
            if payload is None:
                return web.Response(status=204, headers={**NO_STORE, REFUSAL_HEADER: NO_REPORT})
            return web.json_response(payload, headers=NO_STORE)

        @PromptServer.instance.routes.get(PAGE_ROUTE)
        async def get_run_result_page(request):
            answer = page(
                request.query.get("node_id"),
                body=_query_int(request.query.get("body"), 0),
                start=_query_int(request.query.get("start"), 0),
                count=_query_int(request.query.get("count"), MAX_PAGE_LINES),
                node_type=request.query.get("node_type"),
            )
            if answer is None:
                return web.Response(status=204, headers={**NO_STORE, REFUSAL_HEADER: NO_REPORT})
            return web.json_response(answer, headers=NO_STORE)

    except Exception as error:
        logger.warning(
            "%s was not registered (%s: %s), so a node interface asking what its node's last "
            "run did gets a failed request",
            ROUTE, type(error).__name__, error,
        )
        logger.debug("%s could not be registered", ROUTE, exc_info=True)
        return False
    _registered = True
    logger.debug("%s and %s are serving run results", ROUTE, PAGE_ROUTE)
    return True


def _payload(
    status, summary, counts, tallies, tallies_total, facts, items, items_total, bodies, inputs
) -> tuple[dict, list[str]]:
    """Build one result out of what a node handed over, inside every bound but the byte one.

    Args:
        status: One of :data:`STATUSES`, already checked.
        summary: The summary line.
        counts: Named numbers, or None.
        tallies: A breakdown, as a sequence of ``name`` and ``value`` mappings, or None.
        tallies_total: How many the breakdown held, or None for the number handed over.
        facts: Named strings, or None.
        items: Sample rows, or None.
        items_total: How many rows there were, or None for the number handed over.
        bodies: Texts with spans marked, or None.
        inputs: The values the run was handed, or None.

    Returns:
        ``(payload, sources)``: the object the route answers with, its ``run`` still to be
        written, and the whole text of each body in it, in the order it carries them.
    """
    truncated = []
    line, cut = _string(summary, MAX_SUMMARY_CHARS)
    if cut:
        truncated.append("summary")

    numbers, dropped = _named(counts, MAX_COUNTS, numeric=True)
    if dropped:
        truncated.append("counts")
    breakdown, kept, tally_total = _tallies(tallies, tallies_total)
    if kept < tally_total:
        truncated.append("tallies")
    strings, dropped = _named(facts, MAX_FACTS, numeric=False)
    if dropped:
        truncated.append("facts")

    rows, held, total = _rows(items, items_total)
    if held < total:
        truncated.append("items")

    handed, dropped = _inputs(inputs)
    if dropped:
        truncated.append("inputs")

    blocks, dropped = _bodies(bodies)
    if dropped:
        truncated.append("bodies")
    if any(not block["whole"] for block in blocks):
        truncated.append("body_text")
    if any(len(block["marks"]) < block["marks_total"] for block in blocks):
        truncated.append("marks")

    # Taken off every row rather than stopping at the first that answers: the marker rides on
    # the row itself, and one left behind would go out in the answer.
    cuts = [row.pop("_cut")
            for group in (rows, numbers, breakdown, strings, blocks, handed) for row in group]
    if any(cuts):
        truncated.append("text")
    # Taken off the same way, so the whole text a body was cut from rides no further than here.
    sources = [block.pop("_source") for block in blocks]

    return {
        "status": status,
        "summary": line,
        "counts": numbers,
        "tallies": breakdown,
        "tallies_total": tally_total,
        "facts": strings,
        "inputs": handed,
        "bodies": blocks,
        "items": rows,
        "items_total": total,
        "truncated": truncated,
        "run": 0,
    }, sources


def _fit(payload: dict) -> bool:
    """Drop sample rows, and then bodies, until the result is inside :data:`MAX_BYTES`.

    Args:
        payload: The result, changed in place.

    Returns:
        True when it fits, False when it does not fit with neither of those left in it.
    """
    # Measured the way the route sends it: json_response escapes every non-ASCII character,
    # so the serialised length is the number of bytes on the wire.
    while len(json.dumps(payload)) > MAX_BYTES:
        # Sample rows go before bodies: a row stands for one match, where a body is the text
        # the whole readout is about.
        for part in ("items", "bodies"):
            if payload[part]:
                payload[part].pop()
                if part not in payload["truncated"]:
                    payload["truncated"].append(part)
                break
        else:
            return False
    return True


def _keep(key: str, sources) -> None:
    """Hold the lines of one report's bodies, replacing whatever that node held before.

    Args:
        key: The node's store key.
        sources: The whole text of each body the report carries, in its order.
    """
    bodies = []
    chars = 0
    for source in sources:
        text = source if isinstance(source, str) else ""
        if len(text) > MAX_KEPT_BODY_CHARS:
            # Cut back to a line break, so a half line is never served as a whole one.
            edge = text.rfind("\n", 0, MAX_KEPT_BODY_CHARS + 1)
            text = text[:edge] if edge > 0 else ""
        if chars + len(text) > MAX_KEPT_NODE_CHARS:
            text = ""
        chars += len(text)
        bodies.append(text.split("\n") if text else [])
    _pages.pop(key, None)
    if not chars:
        return
    _pages[key] = {"bodies": bodies, "chars": chars}
    while sum(entry["chars"] for entry in _pages.values()) > MAX_KEPT_TOTAL_CHARS:
        dropped, _ = _pages.popitem(last=False)
        if dropped == key:
            return


def _fit_page(lines) -> tuple[list[str], bool]:
    """One page's lines, held inside :data:`MAX_PAGE_CHARS`.

    Args:
        lines: The lines the range named, in order.

    Returns:
        ``(lines, clipped)``, ``clipped`` True when a line was cut to fit and when the page
        stopped short of the range to stay inside the bound.
    """
    kept = []
    room = MAX_PAGE_CHARS
    for line in lines:
        if len(line) > room:
            if not kept:
                return [line[:room]], True
            return kept, True
        kept.append(line)
        room -= len(line)
    return kept, False


def _blank_page() -> dict:
    """A page carrying no lines, for a body there is nothing to answer for."""
    return {
        "name": "",
        "start": 0,
        "lines": [],
        "total": 0,
        "held": 0,
        "clipped": False,
        "run": 0,
    }


def _mismatched(payload: dict, node_type) -> bool:
    """Whether a stored result was published by a node of another kind.

    Args:
        payload: The stored result.
        node_type: The class the asking node names, or None.

    Returns:
        True when both kinds are known and they differ, False when either is unknown, which is
        the answer for a result published outside a queued prompt.
    """
    asked = (node_type or "").strip() if isinstance(node_type, str) else ""
    held = payload.get("node_type") or ""
    return bool(asked and held and asked != held)


def _lines_in(text: str) -> int:
    """How many lines a text holds, an empty one holding none."""
    return text.count("\n") + 1 if text else 0


def _query_int(value, default: int) -> int:
    """A query value as a whole number, or ``default`` when it is not one."""
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _named(values, cap: int, numeric: bool) -> tuple[list[dict], bool]:
    """Named values as payload rows, in the order they were given.

    Args:
        values: A mapping of name to value, or None.
        cap: How many rows are kept.
        numeric: True to keep finite numbers and drop everything else, False to keep
            strings, ints and floats and write each as text.

    Returns:
        ``(rows, dropped)``, each row ``{"name", "value"}`` and ``dropped`` True when a name
        or a value was unusable, or when there were more than ``cap`` of them.
    """
    if not isinstance(values, Mapping):
        return [], values is not None
    rows = []
    dropped = False
    for name, value in values.items():
        label, name_cut = _string(name, MAX_LABEL_CHARS)
        if numeric:
            kept, value_cut = _number(value), False
        else:
            kept, value_cut = _string(value, MAX_LABEL_CHARS)
            kept = kept or None
        if not label or kept is None:
            dropped = True
            continue
        if len(rows) >= cap:
            dropped = True
            break
        rows.append({"name": label, "value": kept, "_cut": name_cut or value_cut})
    return rows, dropped


def _tallies(tallies, tallies_total) -> tuple[list[dict], int, int]:
    """A breakdown as payload rows, in the order it was given, and how many rows it held.

    Args:
        tallies: A sequence of ``name`` and ``value`` mappings, one of them, or None.
        tallies_total: How many rows the breakdown held before the node took a sample, or None.

    Returns:
        ``(rows, kept, total)``, ``kept`` counting the rows kept and ``total`` how many the node
        says there were, which is never below ``kept``.
    """
    if tallies is None:
        given = []
    elif isinstance(tallies, Mapping):
        given = [tallies]
    elif isinstance(tallies, Sequence) and not isinstance(tallies, str):
        given = list(tallies)
    else:
        try:
            given = list(islice(iter(tallies), MAX_TALLIES + 1))
        except TypeError:
            given = []
    rows = []
    for value in given[:MAX_TALLIES]:
        row = _tally(value)
        if row is not None:
            rows.append(row)
    kept = len(rows)
    counted = isinstance(tallies_total, int) and not isinstance(tallies_total, bool)
    return rows, kept, max(tallies_total if counted else len(given), kept)


def _tally(value) -> dict | None:
    """One row of a breakdown, or None when it does not name a number."""
    if not isinstance(value, Mapping):
        return None
    name, cut = _string(value.get("name"), MAX_LABEL_CHARS)
    number = _number(value.get("value"))
    if not name or number is None:
        return None
    return {"name": name, "value": number, "_cut": cut}


def _rows(items, items_total) -> tuple[list[dict], int, int]:
    """Sample rows as payload rows, and how many there were.

    Args:
        items: A sequence of strings or mappings, one of either, or None.
        items_total: How many rows there were before the node took a sample, or None.

    Returns:
        ``(rows, held, total)``, ``held`` counting the rows kept and ``total`` how many the
        node says there were, which is never below ``held``.
    """
    if items is None:
        given = []
    elif isinstance(items, (str, Mapping)):
        given = [items]
    elif isinstance(items, Sequence):
        given = items
    else:
        try:
            # An iterator is read one row past the bound rather than drained, so a node
            # handing over a generator of 400000 rows builds no list of them here.
            given = list(islice(iter(items), MAX_ITEMS + 1))
        except TypeError:
            given = []
    rows = []
    for value in given[:MAX_ITEMS]:
        row = _row(value)
        if row is not None:
            rows.append(row)
    held = len(rows)
    counted = isinstance(items_total, int) and not isinstance(items_total, bool)
    return rows, held, max(items_total if counted else len(given), held)


def _inputs(inputs) -> tuple[list[dict], bool]:
    """The values a run was handed as payload rows, in the order they were given.

    Args:
        inputs: A sequence of the mappings :func:`given` builds, one of them, or None.

    Returns:
        ``(handed, dropped)``, ``dropped`` True when an entry was unusable or when there were
        more than :data:`MAX_INPUTS` of them.
    """
    if inputs is None:
        entries = []
    elif isinstance(inputs, Mapping):
        entries = [inputs]
    elif isinstance(inputs, Sequence) and not isinstance(inputs, str):
        entries = list(inputs)
    else:
        try:
            entries = list(islice(iter(inputs), MAX_INPUTS + 1))
        except TypeError:
            entries = []
    handed = []
    dropped = len(entries) > MAX_INPUTS
    for value in entries[:MAX_INPUTS]:
        entry = _input(value)
        if entry is None:
            dropped = True
            continue
        handed.append(entry)
    return handed, dropped


def _input(value) -> dict | None:
    """One value a run was handed, or None when it is not named well enough to compare."""
    if not isinstance(value, Mapping):
        return None
    name, name_cut = _string(value.get("name"), MAX_LABEL_CHARS)
    checksum, sum_cut = _string(value.get("checksum"), MAX_LABEL_CHARS)
    if not name or not checksum:
        return None
    linked = value.get("linked")
    return {
        "name": name,
        "linked": linked if isinstance(linked, bool) else None,
        "bytes": _index(value.get("bytes"), 0),
        "checksum": checksum,
        "_cut": name_cut or sum_cut,
    }


def _bodies(bodies) -> tuple[list[dict], bool]:
    """Texts with spans marked as payload bodies, in the order they were given.

    Args:
        bodies: A sequence of the mappings :func:`body` builds, one of them, or None.

    Returns:
        ``(blocks, dropped)``, ``dropped`` True when a body was unusable or when there were
        more than :data:`MAX_BODIES` of them.
    """
    if bodies is None:
        given = []
    elif isinstance(bodies, Mapping):
        given = [bodies]
    elif isinstance(bodies, Sequence) and not isinstance(bodies, str):
        given = list(bodies)
    else:
        try:
            given = list(islice(iter(bodies), MAX_BODIES + 1))
        except TypeError:
            given = []
    blocks = []
    dropped = len(given) > MAX_BODIES
    for value in given[:MAX_BODIES]:
        block = _block(value)
        if block is None:
            dropped = True
            continue
        blocks.append(block)
    return blocks, dropped


def _block(value) -> dict | None:
    """One body of text, or None when it is not one."""
    if not isinstance(value, Mapping):
        return None
    name, name_cut = _string(value.get("name"), MAX_LABEL_CHARS)
    if not name:
        return None
    text, text_cut = _string(value.get("text"), MAX_BODY_CHARS)
    offset = _index(value.get("offset"), 0)
    length = max(_index(value.get("length"), len(text)), offset + len(text))
    whole = offset == 0 and len(text) == length
    source = value.get("source")
    # A body handed over without its whole text is pageable over the piece it carries, and only
    # where that piece is the whole of it.
    held = source if isinstance(source, str) else (text if whole else "")
    marks = []
    spans = value.get("marks")
    if isinstance(spans, (list, tuple)):
        for span in spans[:MAX_MARKS]:
            mark = _mark(span, len(text))
            if mark is not None:
                marks.append(mark)
    return {
        "name": name,
        "text": text,
        "marks": marks,
        "marks_total": max(_index(value.get("marks_total"), len(marks)), len(marks)),
        "offset": offset,
        "length": length,
        "lines": max(_index(value.get("lines"), _lines_in(held)), _lines_in(text)),
        "whole": whole,
        "_cut": name_cut or text_cut,
        "_source": held,
    }


def _row(value) -> dict | None:
    """One sample row, or None when it carries nothing to draw."""
    if isinstance(value, Mapping):
        text, cut = _string(value.get("text"), MAX_TEXT_CHARS)
        note, note_cut = _string(value.get("note"), MAX_LABEL_CHARS)
        mark = _mark(value.get("mark"), len(text))
        clipped = bool(value.get("clipped")) or cut
    else:
        text, cut = _string(value, MAX_TEXT_CHARS)
        note, note_cut, mark, clipped = "", False, None, cut
    if not text:
        return None
    return {"text": text, "mark": mark, "note": note, "clipped": clipped, "_cut": cut or note_cut}


def _mark(mark, length: int) -> list[int] | None:
    """A span inside a row's text, clamped into it, or None when it is not a span."""
    if isinstance(mark, (list, tuple)) and len(mark) == 2:
        try:
            first = max(0, min(int(mark[0]), length))
            last = max(first, min(int(mark[1]), length))
        except (TypeError, ValueError):
            return None
        return [first, last]
    return None


def _index(value, default: int) -> int:
    """A value as a character index of at least zero, or ``default`` when it is not one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return max(0, int(value)) if math.isfinite(value) else default


def _string(value, cap: int) -> tuple[str, bool]:
    """A value as text no longer than ``cap``, and whether it was cut.

    Args:
        value: A string, an int or a float. Anything else, None included, answers ``""``.
        cap: Characters kept.

    Returns:
        ``(text, cut)``.
    """
    if isinstance(value, str):
        text = value
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        text = str(value)
    else:
        return "", False
    return (text[:cap], True) if len(text) > cap else (text, False)


def _number(value):
    """A value as a finite number, or None when it is not one.

    Args:
        value: An int or a float. A bool, a string and a value no browser can read as a
            number, ``nan`` and both infinities, all answer None.

    Returns:
        The number, or None.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    # json.dumps writes nan and inf as bare NaN and Infinity, which no JSON parser in a
    # browser reads, so one of those would cost the whole readout rather than one row.
    return value if math.isfinite(value) else None

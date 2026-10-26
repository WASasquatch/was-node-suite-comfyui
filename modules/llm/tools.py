"""Tools an agent calls: the built-in function tools, query engine and retriever tools.

Network tools check ``features.network`` when called and answer a refusal as text. A fetched
address is connected only when every address its host resolves to is public.
"""

import ast
import asyncio
import calendar
import json
import math
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Annotated
from urllib.parse import quote

from .. import deps, log
from ..number import expression
from ..util import sandbox
from . import FEATURE, documents, runtime, wires

__all__ = [
    "BRAVE_KEY",
    "KINDS",
    "MAX_DESCRIPTION_CHARS",
    "TAVILY_KEY",
    "brave_tools",
    "calculator_tools",
    "chain_of",
    "check_name",
    "datetime_tools",
    "describe",
    "document_options",
    "document_stamp",
    "document_tools",
    "names",
    "page_text",
    "query_engine_tool",
    "retriever_tool",
    "sources_in",
    "tavily_tools",
    "web_page_tools",
    "wikipedia_tools",
]

logger = log.get_logger("llm.tools")

#: The built-in tools LLM Tool offers, as the words on its menu.
KINDS = (
    "calculator",
    "date and time",
    "wikipedia",
    "web page",
    "web search (Tavily)",
    "web search (Brave)",
    "document reader",
)

#: The environment variable holding the Tavily key.
TAVILY_KEY = "TAVILY_API_KEY"

#: The environment variable holding the Brave Search key.
BRAVE_KEY = "BRAVE_API_KEY"

#: Bytes of a document file read at most.
MAX_DOCUMENT_BYTES = 32 * 1024 * 1024

#: Characters of text a document yields at most.
MAX_DOCUMENT_CHARS = 4_000_000

#: Characters of a tool description a provider accepts.
MAX_DESCRIPTION_CHARS = 1024

#: A tool or agent name: a letter, then letters, digits and underscores, 64 at most.
NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")

#: Names llama-index keeps for itself.
RESERVED_NAMES = frozenset({"handoff"})

#: A Wikipedia language code, such as ``en``, ``simple`` or ``zh-yue``.
LANGUAGE = re.compile(r"[a-z]{2,12}(-[a-z]{2,10})?")

#: A UTC offset such as ``+05:30``, ``UTC-3`` or ``GMT+1``.
OFFSET = re.compile(r"(?:utc|gmt)?\s*([+-])(\d{1,2})(?::?(\d{2}))?", re.IGNORECASE)

#: A time zone database key such as ``Europe/London``.
ZONE_KEY = re.compile(r"[A-Za-z][A-Za-z0-9_+\-]{0,30}(/[A-Za-z0-9_+\-]{1,30}){0,2}")

#: Bytes of an API answer read at most.
MAX_API_BYTES = 2 * 1024 * 1024

#: Seconds an API call waits at most.
API_TIMEOUT = 20.0

#: Redirects an API call follows at most, each on the same host.
MAX_API_REDIRECTS = 3

#: The User-Agent every request carries.
USER_AGENT = "WAS-Node-Suite-ComfyUI (https://github.com/WASasquatch/was-node-suite-comfyui)"

#: Days, hours and minutes a date may move by at most.
MAX_SHIFT_DAYS = 3_650_000

#: Years a date may move by at most.
MAX_SHIFT_YEARS = 9000

#: Characters of a search query or title sent at most.
MAX_QUERY_CHARS = 400

#: How the expression evaluator's messages name its variables, left out of the calculator's.
VARIABLE_WORDS = f"the values {expression.VARIABLE_NAMES[0]} to {expression.VARIABLE_NAMES[-1]}, "

_wired_classes: dict | None = None


def check_name(name, node: str, field: str) -> str:
    """A tool or agent name, checked to be one a model can call.

    Args:
        name: The widget value.
        node: The node's display name, for the message.
        field: The input's name, for the message.

    Returns:
        The name, stripped.

    Raises:
        ValueError: The name is empty, malformed or reserved.
    """
    text = str(name or "").strip()
    if not NAME.fullmatch(text):
        raise ValueError(
            f"{node} needs its {field} to be a letter followed by letters, digits or "
            f"underscores, 64 characters at most, such as `search_manuals`. "
            f"{text[:40]!r} is not one."
        )
    if text.lower() in RESERVED_NAMES:
        raise ValueError(f"{node} cannot use {text!r} as its {field}; it is kept for handing off between agents. Pick another name.")
    return text


def chain_of(tools, node: str) -> tuple:
    """The tools arriving on an ``LLM_TOOLS`` input, checked.

    Args:
        tools: What arrived, a tuple or ``None``.
        node: The node's display name, for the message.

    Returns:
        The tools in chain order, empty for ``None``.

    Raises:
        ValueError: An item is not a llama-index tool.
    """
    runtime.core()
    from llama_index.core.tools import BaseTool

    chained = wires.chain(tools)
    if not wires.every(chained, BaseTool):
        raise ValueError(
            f"{node} was handed something other than tools on its tools input. Connect LLM Tool, "
            f"LLM Query Engine Tool or LLM Retriever Tool to it."
        )
    return chained


def names(tools) -> list[str]:
    """The name of every tool in a chain, in order."""
    return [str(getattr(tool.metadata, "name", "") or "") for tool in tools or ()]


def _refusal(what: str) -> str | None:
    """The text handed back when ``features.network`` is off, or ``None`` when it is on."""
    try:
        runtime.require_network(what)
    except RuntimeError as refused:
        return str(refused)
    return None


def _clip(text: str, limit: int) -> str:
    """Text cut to a number of characters, the cut marked."""
    text = str(text or "")
    return text if len(text) <= limit else text[: max(0, limit - 20)].rstrip() + "\n[cut off here]"


def _function_tool(fn, name: str, description: str):
    """A llama-index function tool around a callable."""
    runtime.core()
    from llama_index.core.tools import FunctionTool

    return FunctionTool.from_defaults(fn=fn, name=name, description=_clip(description, MAX_DESCRIPTION_CHARS))


# ---------------------------------------------------------------------- calculator


def calculator_tools() -> list:
    """The calculator, as one function tool.

    Returns:
        ``[calculate]``.
    """

    def calculate(
        expression: Annotated[str, "The arithmetic to work out, such as (12.5 * 4) / 3 + 2 ** 8 or sqrt(2) * pi."],
    ) -> str:
        """Work out an arithmetic expression."""
        return _calculate(expression)

    functions = ", ".join(sorted(expression.FUNCTIONS))
    description = (
        "Works out arithmetic exactly. Takes numbers, + - * / // % **, comparisons, the "
        f"constants pi, e and tau and the functions {functions}. Powers are limited to an "
        f"exponent of {expression.MAX_POW_EXPONENT}. Letters are not variables."
    )
    return [_function_tool(calculate, "calculate", description)]


def _calculate(text) -> str:
    """The answer to an expression, or the error it met, as text for the model."""
    written = str(text or "")
    folded = " ".join(line.split("#", 1)[0] for line in written.splitlines()).strip()
    if 0 < len(folded) <= expression.MAX_EXPRESSION_CHARS:
        try:
            tree = ast.parse(folded, mode="eval")
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            tree = None
        if tree is not None:
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id in expression.VARIABLE_NAMES and node.id != "e":
                    return (
                        f"Error: `{node.id}` is a letter, and the calculator takes numbers only. "
                        f"Put the value itself in place of `{node.id}`."
                    )
    try:
        value = expression.evaluate(written, {"e": math.e})
    except expression.ExpressionError as error:
        return "Error: " + str(error).replace(VARIABLE_WORDS, "")
    if isinstance(value, float):
        shown = f"{value:.15g}"
    else:
        shown = str(value)
    return f"{folded[:200]} = {shown}"


# ---------------------------------------------------------------------- date and time


def datetime_tools(default_zone: str = "UTC") -> list:
    """The date and time tools.

    Args:
        default_zone: The zone used when the model names none, such as ``UTC`` or ``+02:00``.

    Returns:
        ``[current_time, date_difference, shift_date]``.

    Raises:
        ValueError: ``default_zone`` names no zone this machine knows.
    """
    zone_text = str(default_zone or "UTC").strip() or "UTC"
    zone, problem = _zone(zone_text)
    if zone is None:
        raise ValueError(f"LLM Tool cannot use {zone_text[:64]!r} as its timezone: {problem}")

    def current_time(
        timezone_name: Annotated[str, "A time zone such as Europe/London, a UTC offset such as +05:30, UTC or local. Empty for the default."] = "",
    ) -> str:
        """The current date and time in a zone."""
        chosen = zone
        if str(timezone_name or "").strip():
            chosen, message = _zone(str(timezone_name))
            if chosen is None:
                return f"Error: {message}"
        now = datetime.now(chosen)
        return f"{now.isoformat(timespec='seconds')} ({now.strftime('%A %d %B %Y, %H:%M')}, {_zone_label(now)})"

    def date_difference(
        start: Annotated[str, "The earlier date or date and time, such as 2026-03-01 or 2026-03-01T09:30."],
        end: Annotated[str, "The later date or date and time, in the same form."],
    ) -> str:
        """The time between two dates."""
        first, message = _moment(start, zone)
        if first is None:
            return f"Error: {message}"
        second, message = _moment(end, zone)
        if second is None:
            return f"Error: {message}"
        gap = second - first
        seconds = gap.total_seconds()
        days = seconds / 86400.0
        return (
            f"From {first.isoformat(timespec='minutes')} to {second.isoformat(timespec='minutes')} is "
            f"{days:.4g} days ({gap.days} whole days and {int(abs(seconds) % 86400 // 3600)} hours; "
            f"{days / 7.0:.4g} weeks; {seconds / 3600.0:.6g} hours)."
        )

    def shift_date(
        date: Annotated[str, "The starting date or date and time, such as 2026-03-01 or 2026-03-01T09:30."],
        years: Annotated[int, "Whole years to add, negative to go back."] = 0,
        months: Annotated[int, "Whole months to add, negative to go back."] = 0,
        weeks: Annotated[float, "Weeks to add, negative to go back."] = 0,
        days: Annotated[float, "Days to add, negative to go back."] = 0,
        hours: Annotated[float, "Hours to add, negative to go back."] = 0,
        minutes: Annotated[float, "Minutes to add, negative to go back."] = 0,
    ) -> str:
        """A date moved by an amount of time."""
        moment, message = _moment(date, zone)
        if moment is None:
            return f"Error: {message}"
        try:
            whole_years, whole_months = int(years or 0), int(months or 0)
            span = timedelta(
                weeks=float(weeks or 0), days=float(days or 0),
                hours=float(hours or 0), minutes=float(minutes or 0),
            )
        except OverflowError:
            return "Error: that moves the date outside the years 1 to 9999."
        except (TypeError, ValueError):
            return "Error: years and months take whole numbers; weeks, days, hours and minutes take numbers."
        if abs(whole_years) > MAX_SHIFT_YEARS or abs(whole_months) > MAX_SHIFT_YEARS * 12 or abs(span.days) > MAX_SHIFT_DAYS:
            return "Error: that moves the date outside the years 1 to 9999."
        try:
            moved = _add_months(moment, whole_years * 12 + whole_months) + span
        except (ValueError, OverflowError):
            return "Error: that moves the date outside the years 1 to 9999."
        return f"{moved.isoformat(timespec='minutes')} ({moved.strftime('%A %d %B %Y, %H:%M')})"

    zone_word = _zone_label(datetime.now(zone))
    return [
        _function_tool(
            current_time, "current_time",
            f"Tells the current date, time and weekday. Without a zone it answers in {zone_word}.",
        ),
        _function_tool(
            date_difference, "date_difference",
            "Works out the days, weeks and hours between two dates or dates and times written as "
            "YYYY-MM-DD or YYYY-MM-DDTHH:MM.",
        ),
        _function_tool(
            shift_date, "shift_date",
            "Moves a date forward or back by years, months, weeks, days, hours and minutes, and "
            "tells the weekday it lands on. A month past a shorter month's end lands on its last day.",
        ),
    ]


def _zone(name: str):
    """A tzinfo for a zone name, or ``(None, message)``.

    Args:
        name: ``UTC``, ``local``, an offset such as ``+05:30`` or a key such as ``Asia/Tokyo``.

    Returns:
        ``(tzinfo, "")`` or ``(None, message)``.
    """
    text = str(name or "").strip()[:64]
    lowered = text.lower()
    if lowered in ("", "utc", "z", "gmt", "zulu"):
        return timezone.utc, ""
    if lowered == "local":
        return datetime.now().astimezone().tzinfo, ""
    offset = OFFSET.fullmatch(text)
    if offset:
        hours, minutes = int(offset.group(2)), int(offset.group(3) or 0)
        if hours > 14 or minutes > 59:
            return None, f"{text!r} is not an offset a clock uses; offsets run from -14:00 to +14:00."
        sign = -1 if offset.group(1) == "-" else 1
        return timezone(sign * timedelta(hours=hours, minutes=minutes)), ""
    if not ZONE_KEY.fullmatch(text):
        return None, f"{text!r} is not a time zone. Use a name such as Europe/London, an offset such as +05:30, UTC or local."
    try:
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    except ImportError:
        return None, "this machine has no time zone database. Use an offset such as +05:30, UTC or local."
    try:
        return ZoneInfo(text), ""
    except (ZoneInfoNotFoundError, ValueError, OSError):
        pass
    # A machine without a system zone database, such as Windows, reads zones from tzdata.
    try:
        deps.require("tzdata", feature=FEATURE)
    except deps.DependencyError:
        return None, (
            f"no time zone database on this machine knows {text!r}. Use an offset such as "
            f"+05:30, UTC or local; installing requirements/llm.txt adds the zone names."
        )
    try:
        return ZoneInfo(text), ""
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return None, f"{text!r} is not a time zone the zone database knows. Use a name such as Europe/London."


def _zone_label(moment: datetime) -> str:
    """A zone as words for the model, such as ``UTC+02:00 (CEST)``."""
    offset = moment.utcoffset() or timedelta(0)
    total = int(offset.total_seconds() // 60)
    sign = "-" if total < 0 else "+"
    label = f"UTC{sign}{abs(total) // 60:02d}:{abs(total) % 60:02d}"
    name = moment.tzname()
    return f"{label} ({name})" if name and name != label and not name.startswith(("UTC", "+", "-")) else label


def _moment(text, zone):
    """A datetime read from ISO text, or ``(None, message)``.

    Args:
        text: ``YYYY-MM-DD``, ``YYYY-MM-DDTHH:MM`` and the forms ``fromisoformat`` reads.
        zone: The zone given to a time that names none.

    Returns:
        ``(datetime, "")`` or ``(None, message)``.
    """
    written = str(text or "").strip()[:64]
    if written.lower() in ("now", "today"):
        moment = datetime.now(zone)
        return (moment if written.lower() == "now" else moment.replace(hour=0, minute=0, second=0, microsecond=0)), ""
    try:
        moment = datetime.fromisoformat(written.replace("Z", "+00:00"))
    except ValueError:
        return None, f"{written!r} is not a date. Write it as YYYY-MM-DD or YYYY-MM-DDTHH:MM."
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=zone)
    return moment, ""


def _add_months(moment: datetime, months: int) -> datetime:
    """A datetime moved by whole months, the day held to the target month's length."""
    if not months:
        return moment
    index = moment.year * 12 + (moment.month - 1) + months
    year, month = divmod(index, 12)
    day = min(moment.day, calendar.monthrange(year, month + 1)[1]) if 1 <= year <= 9999 else moment.day
    return moment.replace(year=year, month=month + 1, day=day)


# ---------------------------------------------------------------------- web access


def _snippet(markup: str) -> str:
    """A search snippet or excerpt as plain text, its markup removed."""
    from ..document import plaintext

    try:
        return " ".join(plaintext.to_plaintext(str(markup or ""), links="text only").split())
    except Exception as error:
        logger.debug("a snippet did not lay out as text (%s)", error)
        return " ".join(re.sub(r"<[^>]{0,200}>", " ", str(markup or "")).split())


async def _read_capped(response, limit: int) -> tuple[bytes, bool]:
    """The body of a streamed response, at most ``limit`` bytes, and whether it went on."""
    chunks: list[bytes] = []
    held = 0
    async for chunk in response.aiter_bytes():
        chunks.append(chunk)
        held += len(chunk)
        if held >= limit:
            return b"".join(chunks)[:limit], True
    return b"".join(chunks), False


async def _get_json(url: str, *, params=None, headers=None, body=None) -> dict:
    """A JSON answer from a fixed API address, at most :data:`MAX_API_BYTES`.

    Raises:
        ValueError: The answer is an error status or not JSON.
        httpx.HTTPError: The request failed.
    """
    httpx = deps.require("httpx", feature=FEATURE)
    sent = {"User-Agent": USER_AGENT, "Accept": "application/json", **(headers or {})}
    async with httpx.AsyncClient(timeout=API_TIMEOUT, follow_redirects=False) as client:
        method = "POST" if body is not None else "GET"
        request = client.build_request(method, url, params=params, headers=sent, json=body)
        host = request.url.host
        for _ in range(MAX_API_REDIRECTS + 1):
            response = await client.send(request, stream=True)
            try:
                if response.next_request is None:
                    data, cut = await _read_capped(response, MAX_API_BYTES)
                    if response.status_code >= 400:
                        raise ValueError(f"the service answered HTTP {response.status_code}")
                    break
                request = response.next_request
                if request.url.host != host or request.url.scheme != "https":
                    raise ValueError("the service sent the request on to another address")
            finally:
                await response.aclose()
        else:
            raise ValueError("the service redirected too many times")
    if cut:
        raise ValueError("the answer was too large to read")
    try:
        parsed = json.loads(data.decode("utf-8", errors="replace"))
    except ValueError:
        raise ValueError("the answer was not JSON") from None
    return parsed if isinstance(parsed, dict) else {}


def _failed(what: str, error: BaseException) -> str:
    """A failed call as text for the model."""
    return f"Error: {what} failed: {runtime.describe_error(error)}"


def page_text(url: str, max_bytes: int, timeout: float) -> tuple[str, str, str, bool]:
    """The main text of a public web page, read by the document loader's fetcher.

    Args:
        url: An ``http`` or ``https`` address.
        max_bytes: Bytes of the body read at most.
        timeout: Seconds the whole fetch may take.

    Returns:
        ``(final address, title, text, cut off)``.

    Raises:
        ValueError: The address is refused, or the answer is not HTML or text.
        RuntimeError: The server could not be reached or answered with an error.
    """
    try:
        fetched = documents.fetch(str(url or ""), timeout, max_bytes)
        made, title = documents.page_documents(fetched, True, False)
    except (ValueError, RuntimeError) as error:
        # The fetcher's messages are handed on naming the tool rather than the loader node.
        raise type(error)(str(error).replace("LLM Load Web Page", "The web page tool")) from None
    return fetched.url, title, made[0].text if made else "", fetched.truncated


def web_page_tools(max_kilobytes: int = 1024, max_characters: int = 8000, timeout: float = 20.0) -> list:
    """The web page reader, as one function tool.

    Args:
        max_kilobytes: Kilobytes of a page read at most.
        max_characters: Characters of text handed to the model at most.
        timeout: Seconds a fetch takes at most.

    Returns:
        ``[fetch_web_page]``.
    """
    limit = max(1, int(max_kilobytes)) * 1024
    characters = max(1, int(max_characters))
    seconds = max(1.0, float(timeout))

    async def fetch_web_page(
        url: Annotated[str, "The full http or https address of the page, such as https://example.com/article."],
    ) -> str:
        """The text of a public web page."""
        refused = _refusal("The web page tool")
        if refused:
            return refused
        runtime.check_interrupt()
        try:
            final, title, text, cut = await asyncio.to_thread(page_text, str(url or ""), limit, seconds)
        except Exception as error:
            return f"Error: {runtime.describe_error(error)}"
        logger.debug("the web page tool read %d characters", len(text))
        note = "\n[the page went on past what was read]" if cut else ""
        heading = f"{final}\n{title}" if title else final
        return f"{heading}\n\n{_clip(text, characters)}{note}"

    description = (
        "Reads the main text of a public web page at an http or https address. Returns at most "
        f"{characters} characters; scripts, menus and markup are removed."
    )
    return [_function_tool(fetch_web_page, "fetch_web_page", description)]


def wikipedia_tools(language: str = "en", results: int = 3, max_characters: int = 4000) -> list:
    """Wikipedia search and page summary, as two function tools.

    Args:
        language: The Wikipedia language code, such as ``en`` or ``de``.
        results: Pages a search lists.
        max_characters: Characters of an answer handed to the model at most.

    Returns:
        ``[wikipedia_search, wikipedia_page]``.

    Raises:
        ValueError: ``language`` is not a language code.
    """
    code = str(language or "en").strip().lower()
    if not LANGUAGE.fullmatch(code):
        raise ValueError(f"LLM Tool needs a Wikipedia language code such as `en`, `de` or `simple`, not {code[:20]!r}.")
    count = max(1, int(results))
    characters = max(1, int(max_characters))
    base = f"https://{code}.wikipedia.org"

    async def wikipedia_search(
        query: Annotated[str, "What to look for, such as a person, place, event or idea."],
    ) -> str:
        """Pages matching a search."""
        refused = _refusal("The wikipedia tool")
        if refused:
            return refused
        runtime.check_interrupt()
        try:
            data = await _get_json(
                f"{base}/w/rest.php/v1/search/page",
                params={"q": str(query or "")[:MAX_QUERY_CHARS], "limit": count},
            )
        except Exception as error:
            return _failed("The Wikipedia search", error)
        pages = data.get("pages") if isinstance(data.get("pages"), list) else []
        if not pages:
            return "No Wikipedia page matched. Try other words."
        lines = []
        for page in pages[:count]:
            if not isinstance(page, dict):
                continue
            title = str(page.get("title") or "")
            about = str(page.get("description") or "")
            excerpt = _snippet(page.get("excerpt") or "")
            lines.append(f"- {title}" + (f" ({about})" if about else "") + (f": {excerpt}" if excerpt else ""))
        return _clip("Wikipedia pages:\n" + "\n".join(lines) + "\nCall wikipedia_page with a title to read its summary.", characters)

    async def wikipedia_page(
        title: Annotated[str, "The exact page title, such as one wikipedia_search listed."],
    ) -> str:
        """The summary of one page."""
        refused = _refusal("The wikipedia tool")
        if refused:
            return refused
        runtime.check_interrupt()
        name = str(title or "").strip()[:MAX_QUERY_CHARS].replace(" ", "_")
        if not name:
            return "Error: give the title of a page."
        try:
            data = await _get_json(f"{base}/api/rest_v1/page/summary/{quote(name, safe='')}", params={"redirect": "true"})
        except Exception as error:
            return _failed("Reading the Wikipedia page", error)
        heading = str(data.get("title") or name)
        extract = str(data.get("extract") or "")
        about = str(data.get("description") or "")
        address = ""
        urls = data.get("content_urls")
        if isinstance(urls, dict) and isinstance(urls.get("desktop"), dict):
            address = str(urls["desktop"].get("page") or "")
        if not extract:
            return f"The page {heading!r} has no summary. Try wikipedia_search for another title."
        text = f"{heading}" + (f" ({about})" if about else "") + (f"\n{address}" if address else "") + f"\n\n{extract}"
        return _clip(text, characters)

    return [
        _function_tool(
            wikipedia_search, "wikipedia_search",
            f"Searches the {code} Wikipedia and lists up to {count} matching pages with a short excerpt each.",
        ),
        _function_tool(
            wikipedia_page, "wikipedia_page",
            f"Reads the summary of one {code} Wikipedia page by its title.",
        ),
    ]


def _key(variable: str, service: str) -> str:
    """An API key from the environment.

    Raises:
        ValueError: The variable is not set.
    """
    value = os.environ.get(variable, "").strip()
    if not value:
        raise ValueError(
            f"The {service} search tool needs an API key, and {variable} is not set. Set "
            f"{variable} in the environment ComfyUI starts in, then restart ComfyUI."
        )
    runtime.hide(value)
    return value


def _search_lines(rows, count: int, title: str, address: str, text: str) -> list[str]:
    """Search results as one entry each for the model."""
    lines = []
    for row in rows[:count]:
        if not isinstance(row, dict):
            continue
        lines.append(f"- {row.get(title) or ''}\n  {row.get(address) or ''}\n  {_snippet(row.get(text) or '')}")
    return lines


def tavily_tools(results: int = 5, max_characters: int = 6000) -> list:
    """Web search through Tavily, as one function tool.

    Args:
        results: Results a search lists.
        max_characters: Characters handed to the model at most.

    Returns:
        ``[tavily_search]``.

    Raises:
        ValueError: :data:`TAVILY_KEY` is not set.
    """
    _key(TAVILY_KEY, "Tavily")
    count = max(1, int(results))
    characters = max(1, int(max_characters))

    async def tavily_search(
        query: Annotated[str, "What to search the web for."],
    ) -> str:
        """Web results for a query."""
        refused = _refusal("The Tavily search tool")
        if refused:
            return refused
        runtime.check_interrupt()
        try:
            key = _key(TAVILY_KEY, "Tavily")
            data = await _get_json(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {key}"},
                body={"query": str(query or "")[:MAX_QUERY_CHARS], "max_results": count, "search_depth": "basic"},
            )
        except Exception as error:
            return _failed("The web search", error)
        rows = data.get("results") if isinstance(data.get("results"), list) else []
        if not rows:
            return "The web search found nothing. Try other words."
        return _clip("Web results:\n" + "\n".join(_search_lines(rows, count, "title", "url", "content")), characters)

    return [
        _function_tool(
            tavily_search, "tavily_search",
            f"Searches the web and lists up to {count} results, each with its address and a passage. "
            "Read a result in full with fetch_web_page where that tool is offered.",
        )
    ]


def brave_tools(results: int = 5, max_characters: int = 6000) -> list:
    """Web search through Brave Search, as one function tool.

    Args:
        results: Results a search lists.
        max_characters: Characters handed to the model at most.

    Returns:
        ``[brave_search]``.

    Raises:
        ValueError: :data:`BRAVE_KEY` is not set.
    """
    _key(BRAVE_KEY, "Brave")
    count = max(1, int(results))
    characters = max(1, int(max_characters))

    async def brave_search(
        query: Annotated[str, "What to search the web for."],
    ) -> str:
        """Web results for a query."""
        refused = _refusal("The Brave search tool")
        if refused:
            return refused
        runtime.check_interrupt()
        try:
            key = _key(BRAVE_KEY, "Brave")
            data = await _get_json(
                "https://api.search.brave.com/res/v1/web/search",
                headers={"X-Subscription-Token": key},
                params={"q": str(query or "")[:MAX_QUERY_CHARS], "count": count},
            )
        except Exception as error:
            return _failed("The web search", error)
        web = data.get("web") if isinstance(data.get("web"), dict) else {}
        rows = web.get("results") if isinstance(web.get("results"), list) else []
        if not rows:
            return "The web search found nothing. Try other words."
        return _clip("Web results:\n" + "\n".join(_search_lines(rows, count, "title", "url", "description")), characters)

    return [
        _function_tool(
            brave_search, "brave_search",
            f"Searches the web and lists up to {count} results, each with its address and a short "
            "description. Read a result in full with fetch_web_page where that tool is offered.",
        )
    ]


# ---------------------------------------------------------------------- document reader


def document_options() -> list[str]:
    """The document reader's menu: every file the document loader reads, or its empty entry."""
    return documents.file_options()


def document_stamp(label: str):
    """The picked document's label, size and time, or ``NaN`` where it is not there."""
    return documents.file_fingerprint(label)


def _read_document(label: str) -> tuple[str, str]:
    """The text of the one document a menu label names, and its file name.

    Raises:
        ValueError: Nothing is picked, the label names no file that is there, or the file
            holds no text.
        PathNotAllowed: The file resolved outside every permitted read root.
    """
    chosen = str(label or "").strip()
    if not chosen or chosen == documents.NO_FILES:
        raise ValueError(
            "LLM Tool's document reader needs a document picked. Put a text, Markdown, HTML, "
            "PDF, Word, OpenDocument, EPUB or PowerPoint file in ComfyUI's input folder and pick it from the document menu."
        )
    try:
        found, _root = documents.picked_file(chosen)
    except sandbox.PathNotAllowed:
        raise
    except ValueError:
        raise ValueError(f"LLM Tool cannot find the document {chosen[:200]!r}. Pick it again from the menu.") from None
    shown = os.path.basename(found.relative)
    try:
        texts, _details = documents.read_file(found.path, found.extension, MAX_DOCUMENT_BYTES)
    except documents.Unreadable as error:
        raise ValueError(f"LLM Tool cannot read the document {shown!r}: {error}. Pick another file.") from None
    text = "\n\n".join(part for part in texts if part.strip())[:MAX_DOCUMENT_CHARS]
    return text, shown


def document_tools(label: str, name: str = "read_document", max_characters: int = 8000) -> list:
    """A reader for the one picked document, as one function tool.

    Args:
        label: The menu label of the picked file.
        name: The tool's name.
        max_characters: Characters one part holds.

    Returns:
        ``[reader]``.

    Raises:
        ValueError: Nothing is picked, the name is malformed, or the file holds no text.
        PathNotAllowed: The file resolved outside every permitted read root.
    """
    tool_name = check_name(name, "LLM Tool", "name")
    text, shown = _read_document(label)
    size = max(1, int(max_characters))
    parts = max(1, math.ceil(len(text) / size))

    def read_document(
        part: Annotated[int, f"Which part to read, from 1 to {parts}."] = 1,
    ) -> str:
        """One part of the picked document."""
        try:
            index = int(part)
        except (TypeError, ValueError):
            index = 1
        index = min(max(index, 1), parts)
        piece = text[(index - 1) * size: index * size]
        return f"{shown}, part {index} of {parts}:\n\n{piece}"

    description = (
        f"Reads the document {shown!r}: {len(text)} characters in {parts} part"
        f"{'s' if parts != 1 else ''} of up to {size}. Start with part 1 and read further parts "
        "when the answer needs them."
    )
    return [_function_tool(read_document, tool_name, description)]


# ---------------------------------------------------------------------- engine tools


def _classes() -> dict:
    """The query engine and retriever tool classes that name the models they call."""
    global _wired_classes
    if _wired_classes is not None:
        return _wired_classes
    runtime.core()
    from llama_index.core.tools import QueryEngineTool, RetrieverTool

    class WiredQueryEngineTool(QueryEngineTool):
        """A query engine tool carrying the wires it calls."""

        def __init__(self, *args, depends=(), **kwargs):
            super().__init__(*args, **kwargs)
            self.depends = tuple(depends)

        def models(self) -> tuple:
            return wires.models_in(*self.depends)

    class WiredRetrieverTool(RetrieverTool):
        """A retriever tool carrying the wires it calls."""

        def __init__(self, *args, depends=(), **kwargs):
            super().__init__(*args, **kwargs)
            self.depends = tuple(depends)

        def models(self) -> tuple:
            return wires.models_in(*self.depends)

    _wired_classes = {"query": WiredQueryEngineTool, "retriever": WiredRetrieverTool}
    return _wired_classes


def _description(text, node: str) -> str:
    """A tool description, checked and cut to :data:`MAX_DESCRIPTION_CHARS`.

    Raises:
        ValueError: The description is empty.
    """
    written = " ".join(str(text or "").split())
    if not written:
        raise ValueError(
            f"{node} needs a description saying what the tool answers, such as `Answers "
            f"questions about the camera manuals.` The model reads it to decide when to call the tool."
        )
    return written[:MAX_DESCRIPTION_CHARS]


def query_engine_tool(engine: wires.QueryEngine, name: str, description: str):
    """A query engine tool that names the models its engine calls.

    Args:
        engine: The query engine wire.
        name: The tool's name.
        description: What it answers, read by the model.

    Returns:
        A llama-index ``QueryEngineTool``.
    """
    from llama_index.core.tools import ToolMetadata

    tool_name = check_name(name, "LLM Query Engine Tool", "name")
    metadata = ToolMetadata(name=tool_name, description=_description(description, "LLM Query Engine Tool"))
    return _classes()["query"](query_engine=engine.engine, metadata=metadata, depends=(engine,))


def retriever_tool(retriever: wires.Retriever, name: str, description: str):
    """A retriever tool that names the models its retriever calls.

    Args:
        retriever: The retriever wire.
        name: The tool's name.
        description: What it finds, read by the model.

    Returns:
        A llama-index ``RetrieverTool``.
    """
    from llama_index.core.tools import ToolMetadata

    tool_name = check_name(name, "LLM Retriever Tool", "name")
    metadata = ToolMetadata(name=tool_name, description=_description(description, "LLM Retriever Tool"))
    return _classes()["retriever"](retriever=retriever.retriever, metadata=metadata, depends=(retriever,))


def sources_in(raw_output) -> list:
    """The retrieved chunks a tool's raw output carries.

    Args:
        raw_output: A query engine response, a list of scored chunks, or anything else.

    Returns:
        llama-index ``NodeWithScore`` objects, empty when it carries none.
    """
    from llama_index.core.schema import NodeWithScore

    found = getattr(raw_output, "source_nodes", None)
    if isinstance(found, (list, tuple)):
        return [item for item in found if isinstance(item, NodeWithScore)]
    if isinstance(raw_output, (list, tuple)):
        return [item for item in raw_output if isinstance(item, NodeWithScore)]
    return []


def describe(tools) -> list[str]:
    """One line per tool, its name and the opening of its description, for a panel."""
    rows = []
    for tool in tools or ():
        metadata = tool.metadata
        rows.append(f"{metadata.name}: {str(metadata.description or '')[:120]}")
    return rows

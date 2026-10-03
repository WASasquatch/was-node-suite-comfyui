"""A document written as a ``.pdf`` file through xhtml2pdf.

:func:`build_html` wraps the content in a page whose ``@page`` rule carries the size and the
margins, and writes the metadata as ``<meta>`` tags. Every picture becomes a ``data:`` URL.
"""

from __future__ import annotations

import base64
import re
from html import escape
from io import BytesIO
from typing import TYPE_CHECKING

from ... import deps, log
from ..summary import keywords_text
from . import FEATURE, Page, blocks

if TYPE_CHECKING:  # pragma: no cover - imported for the annotation alone
    from ..container import Document

__all__ = ["BASE_FONT", "MONOSPACE_FONT", "build_html", "write"]

logger = log.get_logger("document.export.pdf")

#: Fonts named in the base stylesheet. Only the fonts built into PDF itself are used, since
#: xhtml2pdf embeds nothing it has not been given a font file for.
BASE_FONT = "Helvetica"
MONOSPACE_FONT = "Courier"

#: Base text size and line spacing of the page.
BASE_FONT_POINTS = 11.0
LINE_HEIGHT = 1.35

#: Colour a link is drawn in, and the grey a rule, a table border and a header cell use.
LINK_COLOR = "#0563c1"
RULE_COLOR = "#808080"
HEADER_FILL = "#eeeeee"

_BODY = re.compile(r"<body\b[^>]*>(.*?)</body\s*>", re.IGNORECASE | re.DOTALL)
_STYLE = re.compile(r"<style\b[^>]*>.*?</style\s*>", re.IGNORECASE | re.DOTALL)
_HEAD = re.compile(r"<head\b[^>]*>.*?</head\s*>", re.IGNORECASE | re.DOTALL)
_HTML_TAGS = re.compile(r"</?(?:html|body)\b[^>]*>", re.IGNORECASE)
_IMAGE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
_SOURCE = re.compile(
    r"""(\bsrc\s*=\s*)("[^"]*"|'[^']*'|[^\s>]+)""", re.IGNORECASE | re.VERBOSE
)
_ALT = re.compile(r"""\balt\s*=\s*("([^"]*)"|'([^']*)'|([^\s>]+))""", re.IGNORECASE)
_LINK = re.compile(r"<link\b[^>]*>", re.IGNORECASE)
_IMPORT = re.compile(r"@import\b[^;]*;?", re.IGNORECASE)
_URL = re.compile(r"""url\(\s*(?!["']?\s*data:)[^)]*\)""", re.IGNORECASE)
_STYLE_ATTRIBUTE = re.compile(r"""(\bstyle\s*=\s*)("[^"]*"|'[^']*')""", re.IGNORECASE)

#: What a request for a file is answered with: an empty inline file, which opens nothing.
NOTHING = "data:text/plain;base64,"


def write(document: "Document", page: Page) -> bytes:
    """Write a document as a ``.pdf`` file.

    Args:
        document: The document to convert.
        page: The page to lay it out on.

    Returns:
        The file's bytes.

    Raises:
        DependencyError: xhtml2pdf is missing or unusable.
        ValueError: xhtml2pdf could not lay the document out, with what it reported.
    """
    deps.require("xhtml2pdf", feature=FEATURE)
    from xhtml2pdf import pisa

    markup = build_html(document, page)
    buffer = BytesIO()
    refusing = _refuse_every_fetch()
    try:
        result = pisa.CreatePDF(
            src=markup,
            dest=buffer,
            encoding="utf-8",
            link_callback=_never_fetch,
            **({"resource_policy": refusing} if refusing is not None else {}),
        )
    except Exception as error:
        raise ValueError(
            f"xhtml2pdf could not lay this document out as a PDF: {error}. The document "
            f"itself is unharmed; exporting it as .docx or .odt, or saving it as .wasdoc, "
            f"does not go through xhtml2pdf."
        ) from error
    if getattr(result, "err", 0):
        raise ValueError(
            f"xhtml2pdf reported {result.err} error(s) laying this document out as a PDF. "
            f"Its own messages are in the ComfyUI console, above this one."
        )
    return buffer.getvalue()


def build_html(document: "Document", page: Page) -> str:
    """Build the HTML page xhtml2pdf is given.

    Args:
        document: The document to convert.
        page: The page to lay it out on.

    Returns:
        A whole HTML document: a head carrying the metadata, the base stylesheet and the
        document's own style blocks, and a body holding its content with every picture
        turned into a ``data:`` URL. No ``<link>``, ``@import`` or ``url()`` other than a
        ``data:`` one is left in it.
    """
    content, styles = _split(document.content or "")
    metadata = document.metadata
    head = [
        '<meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>',
        f"<title>{escape(metadata.title)}</title>" if metadata.title else "",
        _meta("author", metadata.author),
        _meta("subject", metadata.description),
        _meta("keywords", keywords_text(metadata.keywords)),
        f"<style>{_stylesheet(page)}</style>",
        *(_inert_css(style) for style in styles),
    ]
    language = f' lang="{escape(metadata.language, quote=True)}"' if metadata.language else ""
    body = _inert_markup(_with_data_urls(content, document.assets))
    return (
        f"<!DOCTYPE html><html{language}><head>{''.join(part for part in head if part)}"
        f"</head><body>{body}</body></html>"
    )


def _split(markup: str) -> tuple[str, list[str]]:
    """The document's body content, and its own style blocks, separated.

    Args:
        markup: ``content.html``, whether it is a fragment or a whole page.

    Returns:
        ``(body, styles)``. A page's ``<head>`` is dropped, keeping its style blocks, so a
        document's own CSS is written after the base stylesheet and wins over it.
    """
    styles = _STYLE.findall(markup)
    body = _BODY.search(markup)
    content = body.group(1) if body else _HEAD.sub("", markup)
    content = _STYLE.sub("", content)
    return _HTML_TAGS.sub("", content), styles


def _stylesheet(page: Page) -> str:
    """The base stylesheet, which the document's own CSS is free to override."""
    return (
        f"@page {{ size: {page.width:.2f}pt {page.height:.2f}pt; "
        f"margin: {page.margin:.2f}pt; }}"
        f"body {{ font-family: {BASE_FONT}; font-size: {BASE_FONT_POINTS:.1f}pt; "
        f"line-height: {LINE_HEIGHT}; }}"
        "h1, h2, h3, h4, h5, h6 { font-weight: bold; margin: 8pt 0 4pt 0; }"
        "h1 { font-size: 20pt; } h2 { font-size: 17pt; } h3 { font-size: 14pt; }"
        "h4 { font-size: 12pt; } h5 { font-size: 11pt; } h6 { font-size: 10pt; }"
        "p { margin: 0 0 6pt 0; }"
        "ul, ol { margin: 0 0 6pt 16pt; }"
        "table { border-collapse: collapse; margin: 0 0 6pt 0; }"
        f"th, td {{ border: 0.5pt solid {RULE_COLOR}; padding: 3pt; "
        "vertical-align: top; }"
        f"th {{ background-color: {HEADER_FILL}; font-weight: bold; }}"
        "caption { font-style: italic; padding-bottom: 2pt; }"
        f"pre, code, kbd, samp, tt {{ font-family: {MONOSPACE_FONT}; }}"
        "pre { white-space: pre; margin: 0 0 6pt 0; }"
        f"blockquote {{ margin: 0 0 6pt 18pt; border-left: 2pt solid {RULE_COLOR}; "
        "padding-left: 6pt; }"
        f"hr {{ border: none; border-top: 0.5pt solid {RULE_COLOR}; margin: 6pt 0; }}"
        f"a {{ color: {LINK_COLOR}; }}"
        "img { max-width: 100%; }"
    )


def _meta(name: str, value: str) -> str:
    """One ``<meta>`` tag, or an empty string where there is nothing to write."""
    if not value:
        return ""
    return f'<meta name="{name}" content="{escape(value, quote=True)}"/>'


def _with_data_urls(content: str, assets) -> str:
    """Rewrite every picture so it carries its own bytes, or is replaced by its words.

    Args:
        content: The body markup.
        assets: The document's embedded files, keyed relative to ``assets/``.

    Returns:
        The markup with each ``src`` replaced by a ``data:`` URL, and each picture whose
        file the document does not carry replaced by its description in square brackets, so
        nothing points anywhere xhtml2pdf would have to fetch.
    """
    dropped = 0

    def replace(match: "re.Match[str]") -> str:
        nonlocal dropped
        tag = match.group(0)
        source = _SOURCE.search(tag)
        raw = _unquoted(source.group(2)) if source else ""
        if raw.lower().startswith("data:"):
            return tag
        picture = blocks.resolve(raw, assets)
        if not picture.data:
            dropped += 1
            return f"<span>[{escape(_alt(tag) or raw or 'picture')}]</span>"
        payload = base64.b64encode(picture.data).decode("ascii")
        return (
            tag[: source.start(2)]
            + f'"data:{picture.media_type};base64,{payload}"'
            + tag[source.end(2):]
        )

    rewritten = _IMAGE.sub(replace, content)
    if dropped:
        logger.warning(
            "%d picture(s) were replaced by their description in the exported PDF. A picture "
            "is drawn when the document carries the file itself; one named by a web address "
            "is never fetched.",
            dropped,
        )
    return rewritten


def _alt(tag: str) -> str:
    """A picture's description, read out of its own tag."""
    match = _ALT.search(tag)
    if not match:
        return ""
    return match.group(2) or match.group(3) or match.group(4) or ""


def _unquoted(value: str) -> str:
    """One attribute value with its quotes taken off."""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1].strip()
    return value


def _never_fetch(uri: str, rel: str) -> str:
    """Answer xhtml2pdf's request for a file with an empty inline file, so nothing is opened.

    Args:
        uri: What the markup or the stylesheet named.
        rel: What it was named relative to.

    Returns:
        :data:`NOTHING`, which xhtml2pdf reads in place of ``uri``.
    """
    logger.debug("the PDF export did not open %r, which the document referred to", uri)
    return NOTHING


def _refuse_every_fetch():
    """xhtml2pdf's resource policy refusing every network and local read.

    Returns:
        The policy, or ``None`` on an xhtml2pdf without resource policies.
    """
    try:
        from xhtml2pdf.config.resources import ResourceAccessPolicy
    except ImportError:
        return None
    return ResourceAccessPolicy(allow_remote=False, base_dir=None)


def _inert_css(css: str) -> str:
    """CSS with every ``@import`` removed and every ``url()`` other than a ``data:`` one emptied."""
    return _URL.sub("none", _IMPORT.sub("", css))


def _inert_markup(markup: str) -> str:
    """Markup with its ``<link>`` tags removed and its ``style`` attributes made inert."""
    markup = _LINK.sub("", markup)
    return _STYLE_ATTRIBUTE.sub(
        lambda match: match.group(1) + _inert_css(match.group(2)), markup
    )

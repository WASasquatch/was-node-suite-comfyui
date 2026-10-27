"""Files, folders, text, web pages and DOC documents read into llama-index documents.

Metadata names a file by its folder label and its path inside that folder, such as
``notes [input]`` and ``week 1/monday.md``. Sizes are bytes and times are UTC.
"""

from __future__ import annotations

import hashlib
import html
import http.client
import ipaddress
import json
import math
import os
import re
import socket
import ssl
import time
import zlib
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urljoin, urlsplit

from .. import deps, log
from ..document import markup, plaintext, tree
from ..interface import run_result
from ..util import file_listing, sandbox
from . import FEATURE, runtime, wires

__all__ = [
    "DEFAULT_EXTENSIONS",
    "FILE_EXTENSIONS",
    "Loaded",
    "MAX_EXAMINED",
    "MAX_FETCH_REDIRECTS",
    "MAX_TOTAL_BYTES",
    "Unreadable",
    "clean_metadata",
    "doc_documents",
    "fetch",
    "file_documents",
    "file_fingerprint",
    "file_options",
    "fingerprint_folder",
    "folder_root",
    "picked_file",
    "load_folder",
    "make_document",
    "page_documents",
    "parse_extensions",
    "publish",
    "text_document",
    "to_text",
    "walk",
]

logger = log.get_logger("llm.documents")

#: Extensions read as HTML.
MARKUP_EXTENSIONS = (".html", ".htm", ".xhtml")

#: The PDF extension.
PDF = ".pdf"

#: Word processor formats, read through the document library.
OFFICE_EXTENSIONS = (".docx", ".odt")

#: Packaged formats read straight out of their zip: an e-book and a slide deck.
EPUB = ".epub"
PPTX = ".pptx"
PACKAGE_EXTENSIONS = (EPUB, PPTX)

#: Bytes one chapter or slide part is read to at most.
MAX_PART_BYTES = 16 * 1024 * 1024

#: Chapters or slides read from one file at most.
MAX_PARTS = 2000

#: Extensions read as text, offered by the single-file menu.
TEXT_EXTENSIONS = (
    ".txt", ".md", ".markdown", ".rst", ".csv", ".tsv", ".json", ".jsonl", ".yaml", ".yml",
    ".xml", ".toml", ".ini", ".cfg", ".log", ".tex", ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".c", ".h", ".cpp", ".hpp", ".cs", ".go", ".rs", ".rb", ".php", ".sh", ".ps1",
    ".bat", ".sql", ".css", ".lua", ".kt", ".swift", ".r",
)

#: Every extension the single-file menu lists.
FILE_EXTENSIONS = TEXT_EXTENSIONS + MARKUP_EXTENSIONS + (PDF,) + OFFICE_EXTENSIONS + PACKAGE_EXTENSIONS

#: Entries the single-file menu offers, and what it shows when there are none.
MAX_FILE_OPTIONS = 2000
NO_FILES = "No documents"

#: The extensions a folder load reads when none are typed.
DEFAULT_EXTENSIONS = (
    ".txt, .md, .rst, .html, .htm, .csv, .tsv, .json, .jsonl, .yaml, .yml, .xml, .py, .js, "
    ".ts, .c, .cpp, .h, .java, .cs, .go, .rs, .sh, .sql, .pdf, .docx, .odt, .epub, .pptx"
)

#: Import name and pip name of the package each extension needs.
READER_PACKAGES = {".docx": "python-docx", ".odt": "odfdo", PDF: "pypdf"}

#: Most extensions one folder load filters on.
MAX_EXTENSIONS = 64

#: One typed extension: a dot and up to 15 letters, digits, ``_``, ``+`` or ``-``.
_EXTENSION = re.compile(r"^\.[a-z0-9_+\-]{1,15}$")

#: Directory entries a folder walk examines before it stops.
MAX_EXAMINED = 200000

#: Bytes one folder load reads across every file.
MAX_TOTAL_BYTES = 512 * 1024 * 1024

#: Bytes one picked file may hold.
MAX_FILE_BYTES = 256 * 1024 * 1024

#: Pages one PDF is read to.
MAX_PDF_PAGES = 5000

#: Characters one file contributes.
MAX_FILE_CHARS = 64 * 1024 * 1024

#: Bytes of a file examined for a NUL byte that marks it as binary.
BINARY_PROBE = 8192

#: Keys and characters one metadata mapping keeps.
MAX_METADATA_KEYS = 64
MAX_METADATA_KEY_CHARS = 64
MAX_METADATA_CHARS = 2000

#: Characters of a title kept.
MAX_TITLE_CHARS = 300

#: Metadata keys the embedding never reads.
EMBED_HIDDEN = (
    "folder", "relative_path", "extension", "size", "modified", "source", "page_count",
    "url", "fetched", "content_type", "created", "language",
)

#: Metadata keys the language model never reads.
LLM_HIDDEN = (
    "extension", "size", "modified", "source", "page_count", "fetched", "content_type",
    "created",
)

#: A value naming a place on disk: a drive, a network share, a ``file:`` address or a rooted path.
_PATH_LIKE = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]|\\\\[^\\\s]+\\|(?i:\bfile:)|^/[^\s/]+/")

#: The title element of an HTML file.
_TITLE = re.compile(r"<title[^>]*>(.{0,2000}?)</title\s*>", re.IGNORECASE | re.DOTALL)

#: Characters of the opening of a file searched for its title.
TITLE_WINDOW = 65536

#: Characters of each document's text a run result carries as a sample.
SAMPLE_CHARS = 4000

#: Documents a run result shows a sample of.
SAMPLES = 3

#: Most redirects a web page fetch follows.
MAX_FETCH_REDIRECTS = 5

#: Bytes of a page's body read at most, whatever a caller asks for.
MAX_FETCH_BYTES = 20 * 1024 * 1024

#: Longest web address accepted.
MAX_URL_CHARS = 2048

#: Bytes read from a socket at a time.
READ_CHUNK = 65536

#: Content types read as HTML, and the ones read as plain text.
HTML_TYPES = ("text/html", "application/xhtml+xml")
TEXT_TYPES = ("text/", "application/json", "application/xml", "application/ld+json")

#: Elements removed from a fetched page before its text is read.
STRIPPED_ELEMENTS = (
    "script", "style", "noscript", "template", "iframe", "svg", "canvas", "object", "embed",
    "link", "meta",
)

#: Elements removed from a fetched page when only its main text is wanted.
BOILERPLATE_ELEMENTS = ("nav", "header", "footer", "aside", "form", "button", "dialog", "menu")

#: Characters of main text below which the whole page is read instead.
MIN_READABLE_CHARS = 200

#: What a fetch announces itself as.
USER_AGENT = "Mozilla/5.0 (compatible; WAS-Node-Suite document loader)"

#: Heading elements and their levels.
HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

#: Elements a section split reads through to reach the headings inside them.
SECTION_CONTAINERS = frozenset(
    {"div", "section", "article", "main", "body", "html", "header", "footer", "span"}
)


class Unreadable(ValueError):
    """A file that holds nothing to read, with a short reason.

    Attributes:
        reason: A few words for a count, such as ``"not text"``.
    """

    def __init__(self, reason: str, message: str = ""):
        super().__init__(message or reason)
        self.reason = reason


@dataclass
class Found:
    """One file a folder walk found.

    Attributes:
        relative: Path below the folder, spelled with ``/``.
        path: Absolute path on disk.
        size: Size in bytes.
        mtime: Modification time, in seconds.
    """

    relative: str
    path: str
    size: int
    mtime: float

    @property
    def extension(self) -> str:
        """The lowercased suffix, with its dot."""
        return os.path.splitext(self.relative)[1].lower()


@dataclass
class Loaded:
    """What a folder load read and what it passed over.

    Attributes:
        items: llama-index ``Document`` objects, in path order.
        formats: Extension to files read.
        skipped: Reason to files passed over.
        matched: Files whose extension matched.
        read: Files read.
    """

    items: list = field(default_factory=list)
    formats: Counter = field(default_factory=Counter)
    skipped: Counter = field(default_factory=Counter)
    matched: int = 0
    read: int = 0


# ---------------------------------------------------------------------- documents


def make_document(text: str, metadata: Mapping[str, Any], doc_id: str):
    """One llama-index document with the house metadata exclusions.

    Args:
        text: The document's text.
        metadata: Its metadata, already cleaned.
        doc_id: The id it is stored under.

    Returns:
        A ``Document``.
    """
    runtime.core()
    from llama_index.core import Document

    kept = dict(metadata)
    return Document(
        text=text,
        metadata=kept,
        id_=doc_id,
        excluded_embed_metadata_keys=[key for key in EMBED_HIDDEN if key in kept],
        excluded_llm_metadata_keys=[key for key in LLM_HIDDEN if key in kept],
    )


def clean_metadata(values: Any) -> dict[str, Any]:
    """Metadata a document can store, read from a mapping a workflow supplied.

    Args:
        values: A mapping, or anything else, which reads as empty.

    Returns:
        At most :data:`MAX_METADATA_KEYS` pairs, keys as text. Text, whole numbers, finite
        numbers, booleans and ``None`` are kept, text cut to :data:`MAX_METADATA_CHARS`; any
        other value is kept as its JSON text, cut the same way.
    """
    if not isinstance(values, Mapping):
        return {}
    kept: dict[str, Any] = {}
    for key, value in values.items():
        if len(kept) >= MAX_METADATA_KEYS:
            break
        name = str(key).strip()[:MAX_METADATA_KEY_CHARS]
        if not name:
            continue
        if value is None or isinstance(value, bool) or isinstance(value, int):
            kept[name] = value
        elif isinstance(value, float):
            kept[name] = value if math.isfinite(value) else str(value)
        elif isinstance(value, str):
            kept[name] = value[:MAX_METADATA_CHARS]
        else:
            try:
                written = json.dumps(value, default=str, ensure_ascii=False)
            except (TypeError, ValueError, RecursionError):
                written = str(type(value).__name__)
            kept[name] = written[:MAX_METADATA_CHARS]
    return kept


def text_document(text: str, title: str, metadata: Any):
    """One document from typed text.

    Args:
        text: The text.
        title: Its title, empty for none.
        metadata: Extra pairs from a DICT wire, or ``None``.

    Returns:
        A ``Document`` whose id is taken from its title and text, so the same text keeps one id.

    Raises:
        ValueError: The text is empty.
    """
    body = str(text or "")
    if not body.strip():
        raise ValueError(
            "LLM Text to Document was given no text. Type or wire the text the document holds."
        )
    values = clean_metadata(metadata)
    name = str(title or "").strip()[:MAX_TITLE_CHARS]
    if name:
        values["title"] = name
    values.setdefault("source", "text")
    return make_document(body, values, "text:" + _digest(name, body))


# ---------------------------------------------------------------------- reading files


def parse_extensions(text: str) -> tuple[str, ...]:
    """The extensions a folder load filters on.

    Args:
        text: Extensions separated by commas or spaces, such as ``.txt, .md pdf``.

    Returns:
        Each lowercased with its dot, in the order typed, each once.

    Raises:
        ValueError: None are given, more than :data:`MAX_EXTENSIONS`, or one is not an extension.
    """
    found: list[str] = []
    for token in re.split(r"[\s,;]+", str(text or "")):
        word = token.strip().lower()
        if not word:
            continue
        if not word.startswith("."):
            word = "." + word
        if not _EXTENSION.match(word):
            raise ValueError(
                f"`{token[:40]}` is not a file extension. Type extensions separated by commas, "
                f"such as `.txt, .md, .pdf`."
            )
        if word not in found:
            found.append(word)
    if not found:
        raise ValueError(
            "No extension was given, so no file would be read. Type the ones to read, such as "
            "`.txt, .md, .pdf`."
        )
    if len(found) > MAX_EXTENSIONS:
        raise ValueError(
            f"{len(found)} extensions were given and at most {MAX_EXTENSIONS} are read. Keep the "
            f"ones the folder holds."
        )
    return tuple(found)


def read_file(path: str, extension: str, max_bytes: int, keep_html: bool = False) -> tuple[list[str], dict]:
    """The text one file holds, one entry per page for a PDF and one entry otherwise.

    Args:
        path: The file, already contained by the caller.
        extension: Its lowercased suffix.
        max_bytes: Most bytes read.
        keep_html: Keep an HTML file's markup rather than reading it as text.

    Returns:
        ``(texts, metadata)``: the text of each page, and ``title``, ``author`` or
        ``page_count`` where the file states them.

    Raises:
        Unreadable: The file is too large, binary, empty, encrypted or damaged.
        DependencyError: The package its format needs is missing.
    """
    try:
        size = os.stat(path).st_size
    except OSError as error:
        raise Unreadable("unreadable", f"it could not be read ({error.strerror or type(error).__name__})") from None
    if size > max_bytes:
        raise Unreadable("too large", f"it holds {_megabytes(size)} and the ceiling is {_megabytes(max_bytes)}")
    try:
        with open(path, "rb") as handle:
            data = handle.read(max_bytes + 1)
    except OSError as error:
        raise Unreadable("unreadable", f"it could not be read ({error.strerror or type(error).__name__})") from None
    if len(data) > max_bytes:
        raise Unreadable("too large", f"it grew past {_megabytes(max_bytes)} while it was read")

    details: dict[str, Any] = {}
    if extension == PDF:
        pages, details = _pdf_pages(data)
        texts = pages
    elif extension in OFFICE_EXTENSIONS:
        text, details = _office_text(data, extension)
        texts = [text]
    elif extension == EPUB:
        texts, details = _epub_text(data)
    elif extension == PPTX:
        texts, details = _slides_text(data)
    else:
        decoded = _decoded(data)
        if decoded is None:
            raise Unreadable("not text", "it holds binary data rather than text")
        if extension in MARKUP_EXTENSIONS:
            title = _html_title(decoded)
            if title:
                details["title"] = title
            texts = [markup.clean(decoded)[0] if keep_html else plaintext.to_plaintext(decoded)]
        else:
            if extension in (".md", ".markdown"):
                title = _markdown_title(decoded)
                if title:
                    details["title"] = title
            texts = [decoded]
    texts = _capped(texts)
    if not any(text.strip() for text in texts):
        raise Unreadable("empty", "it holds no text")
    return texts, details


def file_documents(found: Found, folder: str, max_bytes: int, keep_html: bool = False,
                   page_per_document: bool = False) -> list:
    """The documents one file makes.

    Args:
        found: The file.
        folder: The label of the folder it was found in, such as ``notes [input]``.
        max_bytes: Most bytes read.
        keep_html: Keep an HTML file's markup.
        page_per_document: One document per PDF page rather than one per file.

    Returns:
        ``Document`` objects, never empty.

    Raises:
        Unreadable: The file holds nothing to read.
        DependencyError: The package its format needs is missing.
    """
    texts, details = read_file(found.path, found.extension, max_bytes, keep_html)
    base = {
        "file_name": os.path.basename(found.relative),
        "folder": folder,
        "relative_path": found.relative,
        "extension": found.extension,
        "size": int(found.size),
        "modified": _stamp(found.mtime),
        "source": "file",
    }
    base.update(details)
    doc_id = f"{folder}:{found.relative}"
    if found.extension == PDF and page_per_document:
        made = []
        for number, text in enumerate(texts, 1):
            if text.strip():
                made.append(make_document(text, {**base, "page": number}, f"{doc_id}#page={number}"))
        return made
    return [make_document("\n\n".join(text for text in texts if text.strip()), base, doc_id)]


def walk(root: Path, extensions: tuple[str, ...], recursive: bool) -> tuple[list[Found], bool]:
    """Every file under a folder whose extension is listed, links and junctions passed over.

    Args:
        root: The folder, already resolved inside a permitted read root.
        extensions: Lowercased suffixes to keep.
        recursive: Descend into subfolders.

    Returns:
        ``(files sorted by relative path, stopped)``, ``stopped`` True where
        :data:`MAX_EXAMINED` entries were examined with more left.
    """
    wanted = tuple(extensions)
    found: list[Found] = []
    examined = 0
    queue: deque[tuple[str, int, str]] = deque([(str(root), 0, "")])
    while queue:
        parent, depth, prefix = queue.popleft()
        try:
            listing = os.scandir(parent)
        except OSError:
            continue
        with listing:
            for entry in listing:
                examined += 1
                if examined > MAX_EXAMINED:
                    found.sort(key=lambda row: row.relative.casefold())
                    return found, True
                # A dotted name is tool state, and a link or a junction is not followed.
                if entry.name.startswith(".") or file_listing.linked(entry):
                    continue
                try:
                    if entry.is_dir(follow_symlinks=False):
                        if recursive and depth < file_listing.MAX_DEPTH:
                            queue.append((entry.path, depth + 1, f"{prefix}{entry.name}/"))
                        continue
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    if not entry.name.lower().endswith(wanted):
                        continue
                    info = entry.stat(follow_symlinks=False)
                except OSError:
                    continue
                found.append(
                    Found(f"{prefix}{entry.name}", entry.path, max(0, int(info.st_size)), info.st_mtime)
                )
    found.sort(key=lambda row: row.relative.casefold())
    return found, False


def load_folder(root: Path, folder: str, extensions: tuple[str, ...], recursive: bool,
                max_files: int, max_bytes: int, keep_html: bool) -> Loaded:
    """Read every listed file under a folder into documents, passing over what cannot be read.

    Args:
        root: The folder, already resolved inside a permitted read root.
        folder: Its menu label, such as ``notes [input]``.
        extensions: Lowercased suffixes to read.
        recursive: Descend into subfolders.
        max_files: Most files read.
        max_bytes: Most bytes one file may hold.
        keep_html: Keep an HTML file's markup.

    Returns:
        What was read, and the reasons files were passed over.
    """
    from ..io import picker

    # A wired path is recorded as the label the menu lists, never as a path on this machine.
    folder = picker.folder_label(folder)
    files, stopped = walk(root, extensions, recursive)
    loaded = Loaded(matched=len(files))
    if stopped:
        logger.info(
            "LLM Load Documents stopped walking %s after %d entries", folder, MAX_EXAMINED
        )
    if len(files) > max_files:
        loaded.skipped["past max_files"] = len(files) - max_files
        files = files[:max_files]
    total = 0
    for found in files:
        runtime.check_interrupt()
        if total + found.size > MAX_TOTAL_BYTES:
            loaded.skipped[f"past {_megabytes(MAX_TOTAL_BYTES)} in all"] += 1
            continue
        try:
            made = file_documents(found, folder, max_bytes, keep_html)
        except Unreadable as error:
            loaded.skipped[error.reason] += 1
            logger.debug("LLM Load Documents passed over %s: %s", found.relative, error)
            continue
        except deps.DependencyError:
            loaded.skipped[f"needs {READER_PACKAGES.get(found.extension, 'a package')}"] += 1
            continue
        except Exception as error:
            if type(error).__name__ == "InterruptProcessingException":
                raise
            loaded.skipped["unreadable"] += 1
            logger.debug("LLM Load Documents could not read %s: %s", found.relative, error)
            continue
        total += found.size
        loaded.items.extend(made)
        loaded.formats[found.extension] += 1
        loaded.read += 1
    return loaded


def file_options() -> list[str]:
    """The single-file menu: every readable file under the read roots, or :data:`NO_FILES`."""
    try:
        found = file_listing.labels(FILE_EXTENSIONS, file_listing.ROOTS, MAX_FILE_OPTIONS)
    except Exception as error:
        logger.debug("the file listing could not be read: %s", error)
        found = []
    return list(found) or [NO_FILES]


def picked_file(label: str) -> tuple[Found, str]:
    """The file one single-file menu label names, and the label of the root it sits in.

    Args:
        label: A label :func:`file_options` offered, such as ``notes/a.md [input]``.

    Returns:
        ``(file, root label)``, the file's relative path taken below its root.

    Raises:
        ValueError: Nothing was picked, or the label names no file that is there.
        PathNotAllowed: The file resolved outside every permitted read root.
    """
    chosen = str(label or "").strip()
    if not chosen or chosen == NO_FILES:
        raise ValueError(
            "LLM Load Document has no file picked. Put a document in ComfyUI's input folder, "
            "refresh the page and pick it from the file menu."
        )
    entry = file_listing.find(chosen, FILE_EXTENSIONS, file_listing.ROOTS, paths=True)
    if entry is None:
        raise ValueError(
            f"`{chosen}` names no file LLM Load Document can find. It may have been moved or "
            f"deleted; refresh the page and pick it again."
        )
    resolved = sandbox.resolve_read(entry.path)
    if not resolved.is_file():
        raise ValueError(f"`{chosen}` is not a file. Pick a document from the menu.")
    return Found(entry.relative, str(resolved), entry.size, entry.mtime), entry.tag


def file_fingerprint(label: str) -> str | float:
    """The picked file's label, size and time, or ``NaN`` where it is not there."""
    try:
        found, root = picked_file(label)
        stamp = os.stat(found.path)
    except (OSError, ValueError):
        return float("NaN")
    return f"{root}|{found.relative}|{stamp.st_size}|{stamp.st_mtime_ns}"


def folder_root(label: str) -> Path | None:
    """The folder one folder menu label names.

    Args:
        label: A label :func:`modules.io.picker.folders` offered, such as ``notes [input]``.

    Returns:
        The resolved folder, or ``None`` where the label names no root that is there.

    Raises:
        PathNotAllowed: The label names another machine, or a folder outside every read root.
    """
    from ..io import picker

    if sandbox.names_another_host(label or ""):
        raise sandbox.PathNotAllowed(
            f"`{label}` names another machine. Pick a folder from the menu; a share added under "
            f"paths.allow_read in config.yaml is listed there under its own name."
        )
    return picker.resolve_folder(label)


def fingerprint_folder(root: Path | None, extensions: tuple[str, ...], recursive: bool, max_files: int) -> str | float:
    """A digest of the files a folder load would read, compared between runs.

    Args:
        root: The folder, or ``None`` where it names nothing.
        extensions: Lowercased suffixes read.
        recursive: Descend into subfolders.
        max_files: Most files read.

    Returns:
        A hex digest of each file's path, size and time, or ``NaN`` where the folder is not there.
    """
    if root is None or not root.is_dir():
        return float("NaN")
    files, _stopped = walk(root, extensions, recursive)
    written = "\n".join(f"{row.relative}|{row.size}|{row.mtime}" for row in files[:max_files])
    return hashlib.sha256(written.encode("utf-8", "replace")).hexdigest()


def _decoded(data: bytes) -> str | None:
    """A file's bytes as text, or ``None`` for binary data."""
    if data.startswith(b"\xef\xbb\xbf"):
        return data[3:].decode("utf-8", errors="replace")
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    if b"\x00" in data[:BINARY_PROBE]:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def _capped(texts: list[str]) -> list[str]:
    """Page texts cut once they pass :data:`MAX_FILE_CHARS` together."""
    kept, used = [], 0
    for text in texts:
        if used >= MAX_FILE_CHARS:
            break
        piece = text[: MAX_FILE_CHARS - used]
        kept.append(piece)
        used += len(piece)
    return kept


#: Form XObjects of one page whose decoded copies are released, and how deep forms nest.
MAX_PAGE_FORMS = 256
MAX_FORM_DEPTH = 4


def _page_streams(value, depth: int = 0):
    """The content streams one PDF page decodes for its text: its contents and their form XObjects."""
    from pypdf.generic import ArrayObject, EncodedStreamObject

    if value is None or depth > MAX_FORM_DEPTH:
        return
    value = value.get_object() if hasattr(value, "get_object") else value
    if isinstance(value, ArrayObject):
        for item in value[:MAX_PAGE_FORMS]:
            yield from _page_streams(item, depth + 1)
    elif isinstance(value, EncodedStreamObject):
        yield value
        yield from _form_streams(value.get("/Resources"), depth + 1)


def _form_streams(resources, depth: int):
    """The form XObjects a resource dictionary names, and the streams inside them."""
    if resources is None or depth > MAX_FORM_DEPTH:
        return
    resources = resources.get_object()
    forms = resources.get("/XObject") if hasattr(resources, "get") else None
    if forms is None:
        return
    forms = forms.get_object()
    for key in list(forms.keys())[:MAX_PAGE_FORMS]:
        item = forms[key].get_object()
        if hasattr(item, "get") and item.get("/Subtype") == "/Form":
            yield from _page_streams(item, depth + 1)


def _release_page(page) -> None:
    """Drop the decoded copies of a read PDF page's streams, so pages do not pile up in memory."""
    try:
        streams = [*_page_streams(page.get("/Contents")), *_form_streams(page.get("/Resources"), 0)]
        for stream in streams:
            stream.decoded_self = None
    except Exception as error:
        logger.debug("a PDF page's decoded streams were not released: %s", error)


def _pdf_pages(data: bytes) -> tuple[list[str], dict]:
    """The text of each page of a PDF, and its title and page count.

    Raises:
        Unreadable: The PDF is encrypted, damaged or longer than :data:`MAX_PDF_PAGES`.
        DependencyError: pypdf is missing.
    """
    import io as streams

    pypdf = deps.require("pypdf", feature=FEATURE)
    try:
        reader = pypdf.PdfReader(streams.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise Unreadable("encrypted", "it is a PDF locked with a password")
        count = len(reader.pages)
    except Unreadable:
        raise
    except Exception as error:
        raise Unreadable("damaged", f"it is not a readable PDF ({runtime.describe_error(error)})") from None
    if count > MAX_PDF_PAGES:
        raise Unreadable("too many pages", f"it has {count} pages and at most {MAX_PDF_PAGES} are read")
    pages: list[str] = []
    for page in reader.pages:
        runtime.check_interrupt()
        try:
            pages.append(page.extract_text() or "")
        except Exception as error:
            logger.debug("a PDF page could not be read: %s", error)
            pages.append("")
        _release_page(page)
    details: dict[str, Any] = {"page_count": count}
    try:
        info = reader.metadata
        title = _kept_text(getattr(info, "title", "") if info else "")
        author = _kept_text(getattr(info, "author", "") if info else "")
    except Exception:
        title = author = ""
    if title:
        details["title"] = title
    if author:
        details["author"] = author
    return pages, details


def _office_text(data: bytes, extension: str) -> tuple[str, dict]:
    """The text of a ``.docx`` or ``.odt`` file, and its title and author.

    Raises:
        Unreadable: The file is not that format, or unpacks past the document bounds.
        DependencyError: python-docx or odfdo is missing.
    """
    from ..document import container, office

    try:
        document = office.read(data, extension)
    except container.DocumentError as error:
        raise Unreadable("damaged", str(error).splitlines()[0][:300]) from None
    details: dict[str, Any] = {}
    title = _kept_text(document.metadata.title)
    author = _kept_text(document.metadata.author)
    if title:
        details["title"] = title
    if author:
        details["author"] = author
    return plaintext.to_plaintext(document.content), details


def _package(data: bytes, kind: str):
    """An e-book's or a slide deck's zip, opened once its unpacked size is within bounds.

    Raises:
        Unreadable: The bytes are not a zip, or unpack past the document bound.
    """
    import zipfile
    from io import BytesIO

    from ..document import container

    try:
        archive = zipfile.ZipFile(BytesIO(data))
    except (zipfile.BadZipFile, OSError, ValueError):
        raise Unreadable("damaged", f"it is not a readable {kind}") from None
    declared = sum(max(int(info.file_size), 0) for info in archive.infolist())
    if declared > container.MAX_UNPACKED_BYTES:
        archive.close()
        raise Unreadable("too large", f"it unpacks to {_megabytes(declared)}, past {_megabytes(container.MAX_UNPACKED_BYTES)}")
    return archive


def _part(archive, name: str) -> bytes | None:
    """One part of a package, or ``None`` where it is absent or larger than :data:`MAX_PART_BYTES`."""
    try:
        info = archive.getinfo(name)
    except KeyError:
        return None
    if info.file_size > MAX_PART_BYTES:
        return None
    with archive.open(info) as stream:
        data = stream.read(MAX_PART_BYTES + 1)
    return data if len(data) <= MAX_PART_BYTES else None


def _local(tag) -> str:
    """An XML tag without its namespace."""
    return str(tag).rsplit("}", 1)[-1]


def _epub_text(data: bytes) -> tuple[list[str], dict]:
    """The text of each chapter of an EPUB, in reading order, and its title and author."""
    import posixpath

    from ..document import office

    details: dict[str, Any] = {}
    with _package(data, "e-book") as archive:
        found = office._parsed(archive, "META-INF/container.xml") if "META-INF/container.xml" in archive.namelist() else None
        root_path = ""
        for element in found.iter() if found is not None else ():
            if _local(element.tag) == "rootfile":
                root_path = str(element.get("full-path") or "")
                break
        opf = office._parsed(archive, root_path) if root_path in archive.namelist() else None
        if opf is None:
            raise Unreadable("damaged", "its package file could not be read")
        base = posixpath.dirname(root_path)
        manifest = {}
        order = []
        for element in opf.iter():
            name = _local(element.tag)
            if name == "item" and "html" in str(element.get("media-type") or ""):
                manifest[element.get("id")] = posixpath.normpath(posixpath.join(base, str(element.get("href") or "")))
            elif name == "itemref":
                order.append(element.get("idref"))
            elif name == "title" and element.text and "title" not in details:
                details["title"] = _kept_text(element.text)
            elif name == "creator" and element.text and "author" not in details:
                details["author"] = _kept_text(element.text)
        texts = []
        for idref in [item for item in order if item in manifest][:MAX_PARTS]:
            runtime.check_interrupt()
            raw = _part(archive, manifest[idref])
            decoded = _decoded(raw) if raw else None
            if decoded:
                texts.append(plaintext.to_plaintext(decoded))
    details["chapters"] = len(texts)
    return texts, details


def _slides_text(data: bytes) -> tuple[list[str], dict]:
    """The text of each slide of a PowerPoint deck, in slide order, and its title."""
    from ..document import office

    details: dict[str, Any] = {}
    with _package(data, "slide deck") as archive:
        names = [name for name in archive.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)]
        names.sort(key=lambda name: int(re.search(r"(\d+)\.xml$", name).group(1)))
        texts = []
        for number, name in enumerate(names[:MAX_PARTS], 1):
            runtime.check_interrupt()
            root = office._parsed(archive, name)
            if root is None:
                continue
            lines = []
            for paragraph in root.iter():
                if _local(paragraph.tag) == "p":
                    line = "".join(item.text or "" for item in paragraph.iter() if _local(item.tag) == "t").strip()
                    if line:
                        lines.append(line)
            if lines:
                texts.append(f"Slide {number}\n" + "\n".join(lines))
        if "docProps/core.xml" in archive.namelist():
            core = office._parsed(archive, "docProps/core.xml")
            for element in core.iter() if core is not None else ():
                if _local(element.tag) == "title" and element.text:
                    details["title"] = _kept_text(element.text)
    details["slides"] = len(names)
    return texts, details


def _html_title(text: str) -> str:
    """The title element of an HTML file, or an empty string."""
    found = _TITLE.search(text[:TITLE_WINDOW])
    return _kept_text(html.unescape(found.group(1))) if found else ""


def _markdown_title(text: str) -> str:
    """The first level one heading of a Markdown file, or an empty string."""
    for line in text[:TITLE_WINDOW].splitlines()[:200]:
        if line.startswith("# "):
            return _kept_text(line[2:MAX_TITLE_CHARS * 2].strip().rstrip("#").strip())
    return ""


def _kept_text(value: Any) -> str:
    """A title or author on one line, cut to :data:`MAX_TITLE_CHARS`, empty where it names a path."""
    text = " ".join(str(value or "").split())[:MAX_TITLE_CHARS]
    return "" if _PATH_LIKE.search(text) else text


# ---------------------------------------------------------------------- DOC documents


def doc_documents(document, split_sections: bool) -> list:
    """The documents one DOC makes: one, or one per heading.

    Args:
        document: A ``modules.document.container.Document``.
        split_sections: One document per heading rather than one for the whole.

    Returns:
        ``Document`` objects carrying the DOC's title, author, description, language,
        keywords and dates. Empty where the DOC holds no text.
    """
    record = document.metadata
    base: dict[str, Any] = {"source": "doc"}
    for key in ("title", "author", "description", "language", "created", "modified"):
        value = _kept_text(getattr(record, key, ""))
        if value:
            base[key] = value
    keywords = ", ".join(str(word) for word in record.keywords)[:MAX_METADATA_CHARS]
    if keywords:
        base["keywords"] = keywords
    identity = "doc:" + _digest(record.title, document.content)
    if not split_sections:
        text = plaintext.to_plaintext(document.content)
        return [make_document(text, base, identity)] if text.strip() else []
    made = []
    for index, (heading, level, fragment) in enumerate(_sections(document.content)):
        text = plaintext.to_plaintext(fragment)
        if not text.strip():
            continue
        values = {**base, "section": heading[:MAX_TITLE_CHARS], "section_level": level,
                  "section_index": index}
        made.append(make_document(text, values, f"{identity}#section={index}"))
    return made


def _sections(content: str) -> list[tuple[str, int, str]]:
    """``(heading, level, markup)`` for each heading of a document and what follows it.

    Args:
        content: Document markup.

    Returns:
        The part before the first heading as heading ``""`` at level 0 where it holds anything,
        then one entry per heading.
    """
    root = tree.parse(content)
    sections: list[list] = [["", 0, []]]
    for node in _flattened(root.children):
        if isinstance(node, tree.Element) and node.tag in HEADINGS:
            sections.append([_inner_text(node), HEADINGS[node.tag], [node]])
        else:
            sections[-1][2].append(node)
    return [
        (heading, level, "".join(_serialized(node) for node in nodes))
        for heading, level, nodes in sections
        if nodes
    ]


def _flattened(children: list):
    """Top-level nodes, reading through containers that hold a heading."""
    for child in children:
        if (
            isinstance(child, tree.Element)
            and child.tag in SECTION_CONTAINERS
            and _holds_heading(child)
        ):
            yield from _flattened(child.children)
        else:
            yield child


def _holds_heading(node) -> bool:
    """Whether a heading element sits anywhere inside an element."""
    for child in node.children:
        if isinstance(child, tree.Element) and (child.tag in HEADINGS or _holds_heading(child)):
            return True
    return False


def _inner_text(node) -> str:
    """An element's text on one line."""
    parts: list[str] = []

    def collect(item) -> None:
        for child in item.children:
            if isinstance(child, str):
                parts.append(child)
            else:
                collect(child)

    collect(node)
    return " ".join("".join(parts).split())


def _serialized(node) -> str:
    """A tree node written back as markup."""
    if isinstance(node, str):
        return html.escape(node, quote=False)
    attributes = "".join(
        f' {name}="{html.escape(value, quote=True)}"' for name, value in node.attrs.items()
    )
    if node.tag in tree.VOID_ELEMENTS:
        return f"<{node.tag}{attributes}>"
    inner = "".join(_serialized(child) for child in node.children)
    return f"<{node.tag}{attributes}>{inner}</{node.tag}>"


# ---------------------------------------------------------------------- web pages


@dataclass
class Fetched:
    """One web response.

    Attributes:
        url: The address the page was read from, after redirects.
        status: The HTTP status.
        content_type: The media type, lowercased, without parameters.
        charset: The character set the response names, or ``""``.
        data: The body, at most the byte ceiling.
        truncated: True where the body was longer than the ceiling.
        redirects: Redirects followed.
    """

    url: str
    status: int
    content_type: str
    charset: str
    data: bytes
    truncated: bool
    redirects: int


class _PinnedHTTP(http.client.HTTPConnection):
    """An HTTP connection made to one checked address."""

    def __init__(self, host: str, address: str, port: int, timeout: float):
        super().__init__(host, port, timeout=timeout)
        self._address = address

    def connect(self) -> None:
        """Connect to the checked address."""
        self.sock = socket.create_connection((self._address, self.port), self.timeout)


class _PinnedHTTPS(http.client.HTTPSConnection):
    """An HTTPS connection made to one checked address, verified against the host name."""

    def __init__(self, host: str, address: str, port: int, timeout: float):
        super().__init__(host, port, timeout=timeout, context=ssl.create_default_context())
        self._address = address

    def connect(self) -> None:
        """Connect to the checked address and verify the certificate for the host name."""
        raw = socket.create_connection((self._address, self.port), self.timeout)
        self.sock = self._context.wrap_socket(raw, server_hostname=self.host)


def fetch(url: str, timeout: float, max_bytes: int) -> Fetched:
    """Read one web page, refusing any address that resolves to this machine or a private network.

    Args:
        url: An ``http`` or ``https`` address.
        timeout: Seconds the whole fetch may take.
        max_bytes: Most bytes of the body read.

    Returns:
        The response.

    Raises:
        ValueError: The address is not ``http`` or ``https``, names a private, loopback,
            link-local or reserved address, or redirects too often.
        RuntimeError: The server could not be reached or answered with an error.
    """
    max_bytes = max(1, min(int(max_bytes), MAX_FETCH_BYTES))
    current = _checked_url(url)
    deadline = time.monotonic() + max(1.0, float(timeout))
    for redirects in range(MAX_FETCH_REDIRECTS + 1):
        runtime.check_interrupt()
        parts = urlsplit(current)
        host = parts.hostname or ""
        port = parts.port or (443 if parts.scheme == "https" else 80)
        addresses = _public_addresses(host, port)
        connection, response = _request(parts, host, port, addresses, deadline)
        try:
            if response.status in (301, 302, 303, 307, 308):
                location = response.getheader("Location", "")
                if not location:
                    raise RuntimeError(
                        f"LLM Load Web Page: {host} answered {response.status} with no address to "
                        f"go to."
                    )
                current = _checked_url(urljoin(current, location.strip()))
                continue
            if response.status >= 400:
                raise RuntimeError(
                    f"LLM Load Web Page: {host} answered {response.status} "
                    f"{' '.join(str(response.reason or '').split())[:80]}. "
                    f"Check the address in a browser."
                )
            content_type, charset = _content_type(response.getheader("Content-Type", ""))
            data, truncated = _body(response, max_bytes, deadline)
            data, cut = _decompressed(data, response.getheader("Content-Encoding", ""), max_bytes)
            return Fetched(current, response.status, content_type, charset, data,
                           truncated or cut, redirects)
        finally:
            connection.close()
    raise ValueError(
        f"LLM Load Web Page followed {MAX_FETCH_REDIRECTS} redirects without reaching a page. "
        f"Give the address the page finally lives at."
    )


def page_documents(fetched: Fetched, readable: bool, keep_html: bool) -> tuple[list, str]:
    """The document one fetched page makes.

    Args:
        fetched: The response.
        readable: Read the page's main text rather than all of it.
        keep_html: Keep the chosen part's markup rather than its text.

    Returns:
        ``([Document], title)``.

    Raises:
        ValueError: The response is not HTML or text, or holds no text.
        DependencyError: beautifulsoup4 is missing.
    """
    kind = fetched.content_type
    title = ""
    if kind in HTML_TYPES or (not kind and fetched.data.lstrip()[:1] == b"<"):
        text, title = _page_text(fetched.data, fetched.charset, readable, keep_html)
    elif any(kind.startswith(prefix) for prefix in TEXT_TYPES):
        text = fetched.data.decode(_codec(fetched.charset), errors="replace")
    else:
        raise ValueError(
            f"LLM Load Web Page reads HTML and text, and {urlsplit(fetched.url).hostname} sent "
            f"{kind or 'an unnamed type'}. Download the file into ComfyUI's input folder and use "
            f"LLM Load Document instead."
        )
    if not text.strip():
        raise ValueError(
            f"LLM Load Web Page found no text on {urlsplit(fetched.url).hostname}. The page may "
            f"build its text with scripts, which are not run; try readable_text false, or "
            f"another address."
        )
    metadata: dict[str, Any] = {
        "url": fetched.url,
        "fetched": _stamp(time.time()),
        "content_type": kind,
        "source": "web",
    }
    if title:
        metadata["title"] = title
    if len(text) > MAX_FILE_CHARS:
        text = text[:MAX_FILE_CHARS]
    return [make_document(text, metadata, fetched.url)], title


def _checked_url(url: str) -> str:
    """A web address, once its scheme and host are acceptable.

    Raises:
        ValueError: The address is empty, too long, not ``http`` or ``https``, or names no host.
    """
    text = str(url or "").strip()
    if not text:
        raise ValueError("LLM Load Web Page was given no address. Type one, such as https://example.com/page.")
    if len(text) > MAX_URL_CHARS or any(ord(char) < 33 for char in text):
        raise ValueError(
            "LLM Load Web Page was given an address holding spaces, control characters or more "
            f"than {MAX_URL_CHARS} characters. Give the address as a browser shows it."
        )
    try:
        parts = urlsplit(text)
        parts.port
    except ValueError as error:
        raise ValueError(f"LLM Load Web Page cannot read that address ({error}).") from None
    if parts.scheme.lower() not in ("http", "https"):
        raise ValueError(
            f"LLM Load Web Page reads http and https addresses, and this one is "
            f"`{parts.scheme or 'missing a scheme'}`. Start it with https://."
        )
    if not parts.hostname:
        raise ValueError("LLM Load Web Page was given an address with no host. Give a whole address, such as https://example.com/page.")
    return text


def _public_addresses(host: str, port: int) -> list[str]:
    """Every address a host resolves to, once each is a public one.

    Raises:
        ValueError: Any address is loopback, private, link-local, multicast, reserved or otherwise
            not public.
        RuntimeError: The host does not resolve.
    """
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        literal = None
    if literal is not None:
        resolved = [str(literal)]
    else:
        try:
            infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        except (OSError, UnicodeError) as error:
            raise RuntimeError(f"LLM Load Web Page could not find the host {host} ({error}).") from None
        resolved = []
        for info in infos:
            address = str(info[4][0]).split("%", 1)[0]
            if address not in resolved:
                resolved.append(address)
    if not resolved:
        raise RuntimeError(f"LLM Load Web Page could not find the host {host}.")
    for address in resolved:
        if not _is_public(address):
            raise ValueError(
                f"LLM Load Web Page refuses {host}: it resolves to {address}, an address on this "
                f"machine or a private network. Only public web addresses are fetched."
            )
    return resolved


def _is_public(address: str) -> bool:
    """Whether an address is a public unicast one."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address):
        for inner in (ip.ipv4_mapped, ip.sixtofour, (ip.teredo or (None, None))[1]):
            if inner is not None and not _is_public(str(inner)):
                return False
    return not (
        ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_multicast
        or ip.is_reserved or ip.is_unspecified or not ip.is_global
    )


def _request(parts, host: str, port: int, addresses: list[str], deadline: float):
    """Send one GET to the first checked address that answers.

    Returns:
        ``(connection, response)``.

    Raises:
        RuntimeError: No address could be reached in time.
    """
    target = parts.path or "/"
    if parts.query:
        target += "?" + parts.query
    shown_host = host if ":" not in host else f"[{host}]"
    default = 443 if parts.scheme == "https" else 80
    headers = {
        "Host": shown_host if port == default else f"{shown_host}:{port}",
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "close",
    }
    last: Exception | None = None
    for address in addresses:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        kind = _PinnedHTTPS if parts.scheme == "https" else _PinnedHTTP
        connection = kind(host, address, port, remaining)
        try:
            connection.request("GET", target, headers=headers)
            return connection, connection.getresponse()
        except (OSError, http.client.HTTPException, ssl.SSLError) as error:
            connection.close()
            last = error
    detail = f" ({runtime.describe_error(last)})" if last else ""
    raise RuntimeError(
        f"LLM Load Web Page could not reach {host} in time{detail}. Check the address and the "
        f"connection, or raise timeout."
    )


def _content_type(header: str) -> tuple[str, str]:
    """The media type and character set a Content-Type header names."""
    kind, _, rest = str(header or "").partition(";")
    charset = ""
    for part in rest.split(";"):
        name, _, value = part.strip().partition("=")
        if name.lower() == "charset":
            charset = value.strip().strip("\"'")[:40]
    return kind.strip().lower(), charset


def _codec(charset: str) -> str:
    """A codec Python knows for a character set name, UTF-8 where it knows none."""
    import codecs

    try:
        return codecs.lookup(charset).name if charset else "utf-8"
    except LookupError:
        return "utf-8"


def _body(response, max_bytes: int, deadline: float) -> tuple[bytes, bool]:
    """A response body read to ``max_bytes``, and whether more was left."""
    chunks: list[bytes] = []
    held = 0
    while held <= max_bytes:
        runtime.check_interrupt()
        if time.monotonic() > deadline:
            raise RuntimeError("LLM Load Web Page ran out of time while reading the page. Raise timeout.")
        try:
            chunk = response.read(min(READ_CHUNK, max_bytes + 1 - held))
        except (OSError, http.client.HTTPException) as error:
            raise RuntimeError(f"LLM Load Web Page lost the connection ({runtime.describe_error(error)}).") from None
        if not chunk:
            break
        chunks.append(chunk)
        held += len(chunk)
    data = b"".join(chunks)
    return data[:max_bytes], len(data) > max_bytes


def _decompressed(data: bytes, encoding: str, max_bytes: int) -> tuple[bytes, bool]:
    """A gzip or deflate body inflated to at most ``max_bytes``, and whether it was cut."""
    name = str(encoding or "").strip().lower()
    if name not in ("gzip", "x-gzip", "deflate"):
        return data, False
    inflater = zlib.decompressobj(47 if name != "deflate" else zlib.MAX_WBITS)
    try:
        out = inflater.decompress(data, max_bytes)
    except zlib.error:
        try:
            out = zlib.decompressobj(-zlib.MAX_WBITS).decompress(data, max_bytes)
        except zlib.error as error:
            raise RuntimeError(f"LLM Load Web Page could not unpack the page ({error}).") from None
        return out, len(out) >= max_bytes
    return out, bool(inflater.unconsumed_tail)


def _page_text(data: bytes, charset: str, readable: bool, keep_html: bool) -> tuple[str, str]:
    """A page's text, or its cleaned markup, and its title.

    Raises:
        DependencyError: beautifulsoup4 is missing.
    """
    bs4 = deps.require("bs4", feature=FEATURE)
    soup = bs4.BeautifulSoup(data, "html.parser", from_encoding=charset or None)
    title = _kept_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
    for element in soup(list(STRIPPED_ELEMENTS)):
        element.decompose()
    for comment in soup.find_all(string=lambda value: isinstance(value, bs4.Comment)):
        comment.extract()
    whole = soup.body or soup
    chosen = whole
    if readable:
        chosen = soup.find("main") or soup.find("article") or soup.find(attrs={"role": "main"}) or whole
        for element in chosen(list(BOILERPLATE_ELEMENTS)):
            element.decompose()
    fragment = str(chosen)
    if keep_html:
        return markup.clean(fragment)[0], title
    links = "text only" if readable else plaintext.LINK_MODES[0]
    text = plaintext.to_plaintext(fragment, links=links)
    if readable and chosen is not whole and len(text) < MIN_READABLE_CHARS:
        for element in whole(list(BOILERPLATE_ELEMENTS)):
            element.decompose()
        text = plaintext.to_plaintext(str(whole), links=links)
    return text, title


# ---------------------------------------------------------------------- text out


def to_text(items, separator: str, include_metadata: bool, max_chars: int) -> tuple[str, int, bool]:
    """Documents joined into one string.

    Args:
        items: llama-index nodes.
        separator: Written between two documents.
        include_metadata: Write each document's metadata above its text, as a language model reads it.
        max_chars: Most characters in the result.

    Returns:
        ``(text, documents included, cut)``, ``cut`` True where the text stopped at ``max_chars``.
    """
    runtime.core()
    from llama_index.core.schema import MetadataMode

    mode = MetadataMode.LLM if include_metadata else MetadataMode.NONE
    parts: list[str] = []
    used = 0
    for item in items:
        piece = item.get_content(metadata_mode=mode)
        join = len(separator) if parts else 0
        if used + join + len(piece) > max_chars:
            room = max_chars - used - join
            if room > 0:
                parts.append(piece[:room])
            return separator.join(parts), len(parts), True
        parts.append(piece)
        used += join + len(piece)
    return separator.join(parts), len(parts), False


# ---------------------------------------------------------------------- reporting


def publish(summary: str, documents: wires.Documents, added: list, formats: Counter | None = None,
            skipped: Counter | None = None, facts: Mapping[str, Any] | None = None,
            status: str = run_result.OK) -> None:
    """Publish what a loader read: counts, formats, reasons and a sample of the text.

    Args:
        summary: One line saying what was read.
        documents: Everything the node hands on.
        added: The documents this node made.
        formats: Extension to files read.
        skipped: Reason to files passed over.
        facts: Further named values.
        status: A run result status.
    """
    characters = sum(len(getattr(item, "text", "") or "") for item in added)
    counts = {"documents": len(added), "characters": characters}
    if len(documents) != len(added):
        counts["with chain"] = len(documents)
    if skipped:
        counts["skipped"] = sum(skipped.values())
    shown: dict[str, Any] = {}
    if formats:
        listed = Counter(formats).most_common()
        shown["formats"] = ", ".join(f"{name.lstrip('.')} {count}" for name, count in listed)
    if skipped:
        listed = Counter(skipped).most_common()
        shown["skipped"] = ", ".join(f"{reason} {count}" for reason, count in listed)
    shown.update(facts or {})
    bodies = []
    for index, item in enumerate(added[:SAMPLES]):
        meta = getattr(item, "metadata", {}) or {}
        name = meta.get("title") or meta.get("file_name") or meta.get("section") or f"document {index + 1}"
        if "page" in meta:
            name = f"{name}, page {meta['page']}"
        bodies.append(run_result.body(str(name), (getattr(item, "text", "") or "")[:SAMPLE_CHARS]))
    run_result.publish(status=status, summary=summary, counts=counts, facts=shown, bodies=bodies)


# ---------------------------------------------------------------------- helpers


def _digest(*parts: str) -> str:
    """A short hex digest of some text."""
    joined = "\x1f".join(str(part) for part in parts)
    return hashlib.sha1(joined.encode("utf-8", "replace")).hexdigest()[:16]


def _stamp(seconds: float) -> str:
    """A time in seconds as UTC, ``2026-10-25T09:30:00Z``."""
    try:
        return datetime.fromtimestamp(float(seconds), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OverflowError, OSError, ValueError):
        return ""


def _megabytes(count: int) -> str:
    """A byte count as megabytes, for a message."""
    return f"{count / (1024 * 1024):.1f} MB"

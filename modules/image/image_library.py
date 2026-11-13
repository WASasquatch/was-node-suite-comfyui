"""Captioned picture libraries, embedded once and ranked against a prompt by meaning.

Captions come from a JSON manifest, sidecar text files, saved prompts or file names. Vectors
are kept per folder in ``image_library/`` in the pack's state directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .. import log
from ..constants import ALLOWED_EXT
from ..util.kept import Kept

__all__ = [
    "CAPTION_SOURCES",
    "DEFAULT_TEXT_MODEL",
    "Entry",
    "KEYWORDS",
    "DEFAULT_PICTURE_MODEL",
    "Library",
    "Match",
    "NO_PICTURE_MODEL",
    "build",
    "load_picture",
    "picture_choices",
    "pieces",
    "read",
    "search",
    "stamp",
    "text_choices",
]

logger = log.get_logger("image.library")

#: The config key the library nodes load under.
FEATURE = "features.preprocessors"

#: Where captions come from: a manifest when the folder has one, else each picture's own.
CAPTION_SOURCES = ("auto", "manifest", "sidecar text", "picture prompt", "file name")

#: The text model choice that ranks by shared words, with no model.
KEYWORDS = "keywords only"

#: The picture model choice that embeds no pictures.
NO_PICTURE_MODEL = "none"

#: The text model a new node starts on.
DEFAULT_TEXT_MODEL = "google/embeddinggemma-2"

#: Text models offered for download beside the folders in models/LLM.
TEXT_SUGGESTED = (
    DEFAULT_TEXT_MODEL,
    "BAAI/bge-small-en-v1.5",
    "unsloth/embeddinggemma-300m",
    "intfloat/multilingual-e5-small",
    "Alibaba-NLP/gte-modernbert-base",
    "Snowflake/snowflake-arctic-embed-m-v1.5",
    "BAAI/bge-base-en-v1.5",
    "Qwen/Qwen3-Embedding-0.6B",
)

#: The picture model a new node starts on.
DEFAULT_PICTURE_MODEL = "google/embeddinggemma-2"

#: Picture models offered for download beside the folders in models/LLM.
PICTURE_SUGGESTED = (
    DEFAULT_PICTURE_MODEL,
    "google/siglip2-base-patch16-224",
    "google/siglip2-so400m-patch14-384",
    "openai/clip-vit-large-patch14",
)

#: The task an instruction-following text model is told it is doing.
TASK = "Given an image generation prompt, retrieve the reference picture whose description matches it"

#: Sub-folder of the pack's state directory the vectors are kept in.
CACHE_FOLDER = "image_library"

#: Bumped when the cache layout changes, so old files are passed over.
CACHE_VERSION = 1

#: Largest vector cache read back, in bytes.
MAX_CACHE_BYTES = 2 * 1024 ** 3

#: Folders below the library searched for manifests: the library itself and ``data``.
MANIFEST_FOLDERS = ("", "data")

#: Largest manifest read, in bytes, and how many are examined.
MAX_MANIFEST_BYTES = 32 * 1024 * 1024
MAX_MANIFESTS = 24

#: Most records one manifest yields.
MAX_RECORDS = 200_000

#: Most pictures one library holds.
MAX_FILES = 100_000

#: Most folder entries examined while reading one library.
MAX_EXAMINED = 400_000

#: Most characters of one caption kept.
MAX_CAPTION_CHARS = 1200

#: Most characters of a prompt read, and most pieces it is split into.
MAX_PROMPT_CHARS = 8000
MAX_PIECES = 48

#: Share of the ranking the picture model decides for a picture captioned only by its file name.
FILE_NAME_PICTURE_WEIGHT = 0.9

#: How many of a prompt's best matching pieces a caption's score is the mean of.
BEST_PIECES = 3

#: Largest picture file opened for an output, in bytes.
MAX_PICTURE_BYTES = 256 * 1024 * 1024

#: Texts and pictures embedded together.
TEXT_BATCH = 64
PICTURE_BATCH = 32
GEMMA_PICTURE_BATCH = 8

#: Manifest keys whose text names the picture.
NAME_KEYS = frozenset({
    "names", "name", "title", "caption", "captions", "description", "prompt", "text", "keywords",
    "tags", "label", "ポーズ名", "姿势名称", "名称", "名前",
})

#: Manifest keys never read into a caption.
SKIP_KEYS = frozenset({
    "id", "no", "number", "index", "schema", "hash", "sha1", "sha256", "md5", "url", "href",
    "link", "search", "検索", "检索", "ポーズ番号", "姿势编号", "width", "height", "size",
})

#: Words in a manifest key, or a path, that mark a thumbnail.
THUMB_MARKS = ("thumb", "サムネ", "缩略", "preview", "small")

#: Extensions tried in place of a manifest picture that is not on disk.
SWAPPED_EXTENSIONS = (".webp", ".png", ".jpg", ".jpeg")

#: Sidecar caption extensions, in the order read, and the most bytes of one read.
SIDECARS = (".txt", ".caption")
MAX_SIDECAR_BYTES = 64 * 1024

#: Language keys a multilingual name is read under, in order.
LANGUAGES = ("en", "english", "en_us", "en-us")

#: A short code such as ``P001`` or ``SPR-0001``, never read into a caption.
_CODE = re.compile(r"^[A-Za-z]{0,4}[-_]?\d{1,6}$")

#: Sentence ends and clause breaks a prompt is split at.
_SENTENCES = re.compile(r"[.!?;\n\r。！？；]+")
_CLAUSES = re.compile(r"[,，、|]+")

#: Words a keyword ranking drops.
_STOPWORDS = frozenset(
    "a an and are as at be by for from has her his in into is it its of on or the their to with "
    "while one two".split()
)

_WORD = re.compile(r"\w+")

#: A camera view word ending a file name, such as the ``_front`` of ``P001_front.png``.
_VIEW = re.compile(
    r"[_\-\s](front|back|rear|side|left|right|high|low|top|diag|diagonal|three[_\-]?quarter|profile)$",
    re.IGNORECASE,
)

#: The camera views a picture or a prompt can name.
VIEWS = ("front", "side", "diag", "high", "low", "back")

#: File name view words, as the view they name.
_FILE_VIEWS = {
    "front": "front", "back": "back", "rear": "back", "side": "side", "left": "side", "right": "side",
    "profile": "side", "high": "high", "top": "high", "low": "low", "diag": "diag", "diagonal": "diag",
}

#: Words in a prompt that ask for a camera view.
_PROMPT_VIEWS = (
    ("front", re.compile(r"\bfront view\b|\bfrom the front\b|\bfront[- ]facing\b|\bfacing (?:the )?(?:camera|viewer)\b", re.I)),
    ("side", re.compile(r"\bside view\b|\bfrom the side\b|\bprofile\b|\bside[- ]on\b", re.I)),
    ("diag", re.compile(r"\bthree[- ]quarters?\b|\b3/4\b|\bdiagonal\b|\bangled view\b", re.I)),
    ("high", re.compile(r"\bhigh[- ]angle\b|\bfrom above\b|\bbird'?s[- ]eye\b|\boverhead\b|\btop[- ]down\b", re.I)),
    ("low", re.compile(r"\blow[- ]angle\b|\bfrom below\b|\bworm'?s[- ]eye\b", re.I)),
    ("back", re.compile(r"\bfrom behind\b|\bback view\b|\brear view\b|\bfacing away\b", re.I)),
)


# ---------------------------------------------------------------------- menus


def _text_choices() -> list[str]:
    """Keyword ranking, the text models in models/LLM, then the suggested ones not on disk."""
    from ..llm import embeddings

    found = embeddings._local_choices()
    local = [label for label in found if label not in embeddings.SUGGESTED]
    ordered = [*TEXT_SUGGESTED, *local, *found]
    return [KEYWORDS, *dict.fromkeys(ordered)]


def _picture_choices() -> list[str]:
    """No picture model, the CLIP and SigLIP folders in models/LLM, then suggested ones."""
    from ..llm import pictures

    return [NO_PICTURE_MODEL, *dict.fromkeys([*PICTURE_SUGGESTED, *pictures._choices()])]


_text_menu = Kept("picture library text models", _text_choices, [KEYWORDS, *TEXT_SUGGESTED])
_picture_menu = Kept("picture library picture models", _picture_choices, [NO_PICTURE_MODEL, *PICTURE_SUGGESTED])


def text_choices() -> list[str]:
    """What the text model menu offers, as a worker thread last read it."""
    return list(_text_menu.get())


def picture_choices() -> list[str]:
    """What the picture model menu offers, as a worker thread last read it."""
    return list(_picture_menu.get())


# ---------------------------------------------------------------------- reading


@dataclass(frozen=True)
class Entry:
    """One picture in a library.

    Attributes:
        key: Path of the picture below the library folder, ``/`` separated.
        path: The picture on disk.
        preview: The smaller copy panels draw and pictures are embedded from, or ``path``.
        caption: The text it is found by.
        name: Its file name without the extension.
        size: Its size in bytes.
        modified: Its modification time in nanoseconds.
        group: What it is one view of, such as a pose shot from five angles; its key otherwise.
        view: The camera view its file name ends in, one of :data:`VIEWS`, or ``""``.
        source: Where its caption came from: ``manifest``, ``sidecar text``, ``picture prompt`` or
            ``file name``.
    """

    key: str
    path: str
    preview: str
    caption: str
    name: str
    size: int
    modified: int
    group: str = ""
    view: str = ""
    source: str = ""


@dataclass
class Unit:
    """One folder's share of a library, indexed and cached on its own.

    Attributes:
        folder: The folder on disk.
        label: Its path below the library folder, ``.`` for the library folder itself.
        source: Where its captions came from: ``manifest <name>`` or a caption source.
        entries: Its pictures, in key order.
    """

    folder: str
    label: str
    source: str
    entries: list[Entry] = field(default_factory=list)


@dataclass
class Read:
    """What reading a library found.

    Attributes:
        units: The folders read, each indexed on its own, the library folder first.
        entries: Every unit's pictures, in unit order.
        skipped: Reason to pictures passed over.
        captioned: Count by caption source.
        found: Pictures matched before ``max_files`` was applied.
    """

    units: list[Unit] = field(default_factory=list)
    entries: list[Entry] = field(default_factory=list)
    skipped: Counter = field(default_factory=Counter)
    captioned: Counter = field(default_factory=Counter)
    found: int = 0

    @property
    def source(self) -> str:
        """Where the captions came from, in a few words."""
        if len(self.units) == 1:
            unit = self.units[0]
            return unit.source if unit.label == "." else f"{unit.source} in {unit.label}"
        manifests = sum(1 for unit in self.units if unit.source.startswith("manifest"))
        folders = len(self.units) - manifests
        parts = [
            f"{count} {word}{'' if count == 1 else 's'}"
            for count, word in ((manifests, "manifest"), (folders, "folder")) if count
        ]
        return " and ".join(parts) or "nothing"


def _view_group(key: str) -> str:
    """The group a picture belongs to: its path with a trailing camera view word dropped."""
    stem, _extension = os.path.splitext(key)
    base = _VIEW.sub("", stem)
    return base if base and base != stem else key


def _file_view(key: str) -> str:
    """The camera view a picture's file name ends in, ``""`` for none."""
    found = _VIEW.search(os.path.splitext(key)[0])
    if not found:
        return ""
    word = found.group(1).lower().replace("-", "").replace("_", "")
    return "diag" if word.startswith("three") else _FILE_VIEWS.get(word, "")


def prompt_views(prompt: str) -> set[str]:
    """The camera views a prompt asks for, such as ``{"side"}`` for ``side view``."""
    text = str(prompt or "")[:MAX_PROMPT_CHARS]
    return {view for view, pattern in _PROMPT_VIEWS if pattern.search(text)}


def _clean(text: str) -> str:
    """Text on one line with runs of space closed up, cut to :data:`MAX_CAPTION_CHARS`."""
    return " ".join(str(text).split())[:MAX_CAPTION_CHARS]


def _is_picture_path(value) -> bool:
    """Whether a manifest value names a picture file by a relative path."""
    if not isinstance(value, str) or "://" in value:
        return False
    return value.strip().lower().endswith(ALLOWED_EXT)


def _language_text(value: dict) -> str:
    """The English text of a ``{language: text}`` mapping, or its first text."""
    for key in LANGUAGES:
        if isinstance(value.get(key), str) and value[key].strip():
            return value[key]
    return next((text for text in value.values() if isinstance(text, str) and text.strip()), "")


def _text_of(value) -> str:
    """The caption text a manifest value carries, ``""`` for none."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return _language_text(value).strip()
    if isinstance(value, list) and value and all(isinstance(item, str) for item in value[:64]):
        return ", ".join(item.strip() for item in value[:64] if item.strip())
    return ""


def _tables(data) -> dict[str, dict[str, str]]:
    """Lookup tables a manifest carries: lists of ``{"key", "names"}`` keyed by the list's name."""
    found: dict[str, dict[str, str]] = {}
    if not isinstance(data, dict):
        return found
    for name, value in data.items():
        if not isinstance(value, list) or not value or len(value) > 10_000:
            continue
        if not all(isinstance(item, dict) and isinstance(item.get("key"), str) for item in value):
            continue
        table = {}
        for item in value:
            label = item.get("names") or item.get("name") or item.get("label") or item.get("title")
            text = _text_of(label)
            if text:
                table[item["key"]] = text
        if table:
            found[str(name).lower()] = table
    return found


def _table_for(key: str, tables: dict) -> dict | None:
    """The lookup table a record key's values are named in, by the table's plural name."""
    low = key.lower()
    for name in (low, f"{low}s", f"{low}es", f"{low[:-1]}ies" if low.endswith("y") else ""):
        if name and name in tables:
            return tables[name]
    return None


def _records(data) -> list[tuple[dict, str]]:
    """The longest list of records a manifest holds, nested records spread into their parents' groups.

    Returns:
        ``(record, group)`` pairs, ``group`` empty for a record with nothing nested.
    """
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        lists = [value for value in data.values() if isinstance(value, list) and value and isinstance(value[0], dict)]
        rows = max(lists, key=len) if lists else []
    else:
        rows = []
    flat: list[tuple[dict, str]] = []
    for number, row in enumerate(rows):
        if len(flat) >= MAX_RECORDS:
            break
        if not isinstance(row, dict):
            continue
        nested = next(
            (value for value in row.values() if isinstance(value, list) and value and all(isinstance(item, dict) for item in value)),
            None,
        )
        if nested is None:
            flat.append((row, ""))
            continue
        parent = {key: value for key, value in row.items() if value is not nested}
        group = f"{number}:{_text_of(row.get('id'))}"
        for child in nested[:MAX_RECORDS - len(flat)]:
            flat.append(({**parent, **child}, group))
    return flat


def _located(relative: str, bases: list[str], root: Path) -> str | None:
    """A manifest's relative picture path found inside the library, another extension tried, or ``None``."""
    from ..util import sandbox

    text = relative.strip().replace("\\", "/")
    while text.startswith("./"):
        text = text[2:]
    if not text or sandbox.leaves_folder(text) is not None:
        return None
    parts = text.split("/")
    stem, extension = os.path.splitext(parts[-1])
    for base in bases:
        names = [parts[-1]] + [stem + swapped for swapped in SWAPPED_EXTENSIONS if swapped != extension.lower()]
        for name in names:
            candidate = os.path.join(base, *parts[:-1], name)
            if not os.path.isfile(candidate):
                continue
            if not sandbox.contains(root, Path(os.path.realpath(candidate))):
                logger.debug("a manifest picture leading outside the library was passed over")
                continue
            return candidate
    return None


def _manifest_entries(data, manifest_dir: str, root: Path) -> tuple[list[Entry], Counter]:
    """The entries a parsed manifest describes, with the reasons records were passed over."""
    tables = _tables(data)
    bases = list(dict.fromkeys([str(root), manifest_dir]))
    entries: list[Entry] = []
    skipped: Counter = Counter()
    seen: set[str] = set()
    for record, group in _records(data):
        pictures = [(key, value) for key, value in record.items() if _is_picture_path(value)]
        if not pictures:
            skipped["no picture named"] += 1
            continue
        full = [value for key, value in pictures if not any(mark in f"{key}/{value}".lower() for mark in THUMB_MARKS)]
        thumbs = [value for key, value in pictures if value not in full]
        path = next((found for found in (_located(value, bases, root) for value in full) if found), None)
        preview = next((found for found in (_located(value, bases, root) for value in thumbs) if found), None)
        path = path or preview
        if path is None:
            skipped["picture missing"] += 1
            continue
        key = os.path.relpath(path, str(root)).replace("\\", "/")
        if key in seen:
            continue
        seen.add(key)
        names, others = [], []
        for name, value in record.items():
            low = str(name).lower()
            if _is_picture_path(value) or low in SKIP_KEYS:
                continue
            text = _text_of(value)
            if not text or _CODE.match(text) or "/" in text and _is_picture_path(text):
                continue
            table = _table_for(str(name), tables)
            if table is not None and isinstance(value, str):
                text = table.get(value, text)
            if low in NAME_KEYS:
                names.append(text)
            else:
                others.append(f"{name}: {text}")
        caption = _clean(". ".join(dict.fromkeys(names + others)))
        stat = os.stat(path)
        entries.append(Entry(
            key=key,
            path=path,
            preview=preview or path,
            caption=caption or _file_caption(path),
            name=os.path.splitext(os.path.basename(path))[0],
            size=int(stat.st_size),
            modified=int(stat.st_mtime_ns),
            group=group or _view_group(key),
            view=_file_view(key),
            source="manifest" if caption else "file name",
        ))
    return entries, skipped


def _manifests(root: Path) -> list[tuple[str, str]]:
    """JSON files a library may be described by: ``(label, path)``, in the library and ``data``."""
    from ..util.file_listing import linked

    found = []
    for below in MANIFEST_FOLDERS:
        folder = os.path.join(str(root), below) if below else str(root)
        try:
            with os.scandir(folder) as listing:
                for entry in listing:
                    if len(found) >= MAX_MANIFESTS:
                        break
                    if not entry.name.lower().endswith(".json") or linked(entry):
                        continue
                    if entry.is_file(follow_symlinks=False) and entry.stat().st_size <= MAX_MANIFEST_BYTES:
                        found.append((f"{below}/{entry.name}" if below else entry.name, entry.path))
        except OSError:
            continue
    return sorted(found, key=lambda pair: pair[0].casefold())


def _best_manifest(root: Path, parsed: set) -> tuple[str, list[Entry], Counter]:
    """The manifest that places the most full-size pictures, its entries and what it passed over.

    Args:
        root: The folder the manifest describes.
        parsed: Manifests already read, by path; each read here is added.
    """
    best = ("", [], Counter())
    best_rank = (0, 0, 0)
    for label, path in _manifests(root):
        held = _held(path)
        if held in parsed:
            continue
        parsed.add(held)
        try:
            with open(path, encoding="utf-8-sig") as handle:
                data = json.load(handle)
        except (OSError, ValueError, RecursionError):
            continue
        entries, skipped = _manifest_entries(data, os.path.dirname(path), root)
        latin = sum(1 for entry in entries if _latin(entry.caption))
        rank = (sum(1 for entry in entries if entry.preview != entry.path), latin, len(entries))
        if entries and rank > best_rank:
            best, best_rank = (label, entries, skipped), rank
    return best


def _latin(text: str) -> bool:
    """Whether most of a text's letters are Latin script."""
    letters = [char for char in text if char.isalpha()]
    return bool(letters) and sum(char.isascii() for char in letters) * 2 > len(letters)


def _file_caption(path: str) -> str:
    """A picture's file name as words, its counters and separators dropped."""
    stem = os.path.splitext(os.path.basename(path))[0]
    words = re.sub(r"[_\-.]+", " ", stem)
    words = re.sub(r"(?:\s+\d+)+\s*$", "", words).strip()
    return _clean(words or stem)


def _prompt_caption(path: str) -> str:
    """The prompt ComfyUI or A1111 saved in a picture's header, ``""`` for none."""
    from PIL import Image

    from ..llm.pictures import prompt_of

    try:
        with Image.open(path) as opened:
            info = dict(opened.info)
    except Exception as error:
        logger.debug("a picture header could not be read (%s)", type(error).__name__)
        return ""
    return _clean(prompt_of(info)[0])


def _sidecar_caption(name: str, sidecars: dict[str, str]) -> str:
    """The caption saved beside a picture as ``.txt`` or ``.caption``, ``""`` for none.

    Args:
        name: The picture's file name.
        sidecars: Caption files in the picture's folder, by lowercased name, links left out.
    """
    stem = os.path.splitext(name)[0]
    for suffix in SIDECARS:
        path = sidecars.get(f"{stem}{suffix}".casefold())
        if path is None:
            continue
        try:
            with open(path, "rb") as handle:
                text = handle.read(MAX_SIDECAR_BYTES).decode("utf-8-sig", errors="replace")
        except OSError as error:
            logger.debug("a caption file could not be read (%s)", error.strerror)
            continue
        if text.strip():
            return _clean(text)
    return ""


def _held(path: str) -> str:
    """A path in the form two spellings of one file share."""
    return os.path.normcase(os.path.normpath(path))


def _folders(root: Path, recursive: bool, result: Read):
    """The library folder, then each folder below it when recursive, links and junctions passed over.

    Yields:
        ``(folder, label, pictures, sidecars)``: the folder, its path below the library folder
        (``.`` for the library folder), the picture files directly in it sorted by name, and its
        caption files by lowercased name.
    """
    from collections import deque

    from ..util.file_listing import MAX_DEPTH, linked

    examined = 0
    queue = deque([(str(root), ".", 0)])
    while queue:
        folder, label, depth = queue.popleft()
        pictures, below, sidecars = [], [], {}
        try:
            with os.scandir(folder) as listing:
                for entry in listing:
                    examined += 1
                    if examined > MAX_EXAMINED:
                        result.skipped["folder too large to walk"] += 1
                        break
                    if entry.name.startswith(".") or linked(entry):
                        continue
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if recursive and depth < MAX_DEPTH:
                                below.append(entry.name)
                        elif entry.is_file(follow_symlinks=False):
                            low = entry.name.lower()
                            if low.endswith(ALLOWED_EXT):
                                pictures.append(entry)
                            elif low.endswith(SIDECARS):
                                sidecars[entry.name.casefold()] = entry.path
                    except OSError:
                        continue
        except OSError:
            continue
        pictures.sort(key=lambda entry: entry.name.casefold())
        yield folder, label, pictures, sidecars
        if examined > MAX_EXAMINED:
            return
        for name in sorted(below, key=str.casefold):
            queue.append((os.path.join(folder, name), name if label == "." else f"{label}/{name}", depth + 1))


def _relative(path: str, root: Path) -> str:
    """A path below the library folder, ``/`` separated."""
    return os.path.relpath(path, str(root)).replace(os.sep, "/")


def _loose_entry(entry: os.DirEntry, root: Path, captions: str, sidecars: dict[str, str]) -> tuple[Entry, str]:
    """A picture no manifest names, captioned from its own sources, and the source used."""
    caption, source = "", ""
    if captions in ("auto", "sidecar text"):
        caption, source = _sidecar_caption(entry.name, sidecars), "sidecar text"
    if not caption and captions in ("auto", "picture prompt"):
        caption, source = _prompt_caption(entry.path), "picture prompt"
    if not caption:
        caption, source = _file_caption(entry.path), "file name"
    info = entry.stat(follow_symlinks=False)
    key = _relative(entry.path, root)
    return Entry(
        key=key,
        path=entry.path,
        preview=entry.path,
        caption=caption,
        name=os.path.splitext(entry.name)[0],
        size=int(info.st_size),
        modified=int(info.st_mtime_ns),
        group=_view_group(key),
        view=_file_view(key),
        source=source,
    ), source


def read(root: Path, captions: str, recursive: bool, max_files: int) -> Read:
    """Every picture in a library folder with its caption, one unit per folder read.

    Args:
        root: The folder, already resolved inside a permitted read root.
        captions: One of :data:`CAPTION_SOURCES`.
        recursive: Read every folder below it too.
        max_files: Most pictures kept.

    Returns:
        What was read.

    Raises:
        ValueError: ``captions`` is ``manifest`` and no folder read holds one that names a picture.
    """
    from ..llm.runtime import check_interrupt

    result = Read()
    limit = max(1, min(int(max_files), MAX_FILES))
    claimed: set[str] = set()
    parsed: set[str] = set()
    kept = 0

    def take(entries: list[Entry]) -> list[Entry]:
        nonlocal kept
        result.found += len(entries)
        room = max(0, limit - kept)
        if len(entries) > room:
            result.skipped["past max_files"] += len(entries) - room
            entries = entries[:room]
        kept += len(entries)
        return entries

    for folder, label, pictures, sidecars in _folders(root, recursive, result):
        check_interrupt()
        if captions in ("auto", "manifest"):
            name, entries, skipped = _best_manifest(Path(folder), parsed)
            if entries:
                result.skipped.update(skipped)
                for entry in entries:
                    claimed.add(_held(entry.path))
                    claimed.add(_held(entry.preview))
                rekeyed = sorted(
                    (
                        Entry(
                            key=_relative(entry.path, root), path=entry.path, preview=entry.preview,
                            caption=entry.caption, name=entry.name, size=entry.size,
                            modified=entry.modified, group=f"{label}|{entry.group}", view=entry.view,
                            source=entry.source,
                        )
                        for entry in entries
                    ),
                    key=lambda entry: entry.key.casefold(),
                )
                taken = take(rekeyed)
                if taken:
                    result.units.append(Unit(folder, label, f"manifest {name}", taken))
                    result.captioned["manifest"] += len(taken)
        if captions == "manifest":
            continue
        loose = [entry for entry in pictures if _held(entry.path) not in claimed]
        if not loose:
            continue
        made_entries = []
        processed = 0
        for entry in loose:
            if kept + len(made_entries) >= limit:
                break
            if processed % 256 == 0:
                check_interrupt()
            processed += 1
            try:
                made, source = _loose_entry(entry, root, captions, sidecars)
            except OSError:
                result.skipped["unreadable"] += 1
                continue
            made_entries.append(made)
            result.captioned[source] += 1
        left = len(loose) - processed
        if left:
            result.skipped["past max_files"] += left
            result.found += left
        taken = take(made_entries)
        if taken:
            result.units.append(Unit(folder, label, "each picture" if captions == "auto" else captions, taken))
    if captions == "manifest" and not result.units:
        raise ValueError(
            "Image Library Search found no JSON manifest naming pictures in this folder or its "
            "data folder. Set captions to auto to caption each picture from a text file beside "
            "it, its saved prompt or its file name."
        )
    result.entries = [entry for unit in result.units for entry in unit.entries]
    return result


def stamp(root: Path, recursive: bool) -> str:
    """A fingerprint of everything a library reads: its pictures, captions and manifests.

    Args:
        root: The folder, already resolved inside a permitted read root.
        recursive: Descend into sub-folders.

    Returns:
        A hex digest that changes when any of those files is added, removed or changed.
    """
    from ..llm.documents import walk

    files, _stopped = walk(root, (*ALLOWED_EXT, *SIDECARS, ".json"), recursive or bool(_manifests(root)))
    digest = hashlib.sha1()
    for found in files:
        digest.update(f"{found.relative}\0{found.size}\0{found.mtime}\n".encode("utf-8", "surrogatepass"))
    return digest.hexdigest()


# ---------------------------------------------------------------------- models


@dataclass
class TextModel:
    """The model captions and prompts are embedded with, or keyword ranking.

    Attributes:
        label: Its menu name.
        folder: Its checkpoint folder, ``None`` for keyword ranking.
        pooling: How token states become one vector.
        query: Placed before every prompt piece.
        text: Placed before every caption.
        family: The checkpoint's ``model_type``.
    """

    label: str
    folder: str | None = None
    pooling: str = "mean"
    query: str = ""
    text: str = ""
    family: str = ""

    @property
    def keywords(self) -> bool:
        """Whether this ranks by shared words rather than a model."""
        return self.folder is None

    @property
    def signature(self) -> str:
        """What a cached caption vector depends on besides the caption."""
        return f"{self.label}|{self.pooling}|{self.text}"

    def encode(self, texts: list[str], query: bool):
        """Unit vectors for texts, as a float32 CPU tensor."""
        from ..llm import embeddings
        from ..model import embedding_gemma2

        prefix = self.query if query else self.text
        if self.family == embedding_gemma2.MODEL_TYPE:
            return embedding_gemma2.text_vectors(self.folder, list(texts), prefix, TEXT_BATCH)
        return embeddings.text_vectors(
            self.folder, [prefix + text for text in texts], self.pooling, True, 512, TEXT_BATCH,
        )


@dataclass
class PictureModel:
    """The model pictures and prompts are embedded with: CLIP, SigLIP or EmbeddingGemma 2.

    Attributes:
        label: Its menu name.
        folder: Its checkpoint folder.
        family: ``clip``, ``siglip``, ``siglip2`` or ``embedding_gemma2``.
    """

    label: str
    folder: str
    family: str

    def encode_texts(self, texts: list[str]):
        """Unit vectors for texts in the pictures' space."""
        from ..llm import pictures
        from ..model import embedding_gemma2

        if self.family == embedding_gemma2.MODEL_TYPE:
            return embedding_gemma2.text_vectors(self.folder, list(texts), embedding_gemma2.QUERY, TEXT_BATCH)
        return pictures.text_vectors(self.folder, self.family, texts, PICTURE_BATCH)

    def encode_pictures(self, paths: list[str]):
        """Unit vectors for picture files."""
        from ..llm import pictures
        from ..model import embedding_gemma2

        if self.family == embedding_gemma2.MODEL_TYPE:
            return embedding_gemma2.picture_vectors(self.folder, paths, GEMMA_PICTURE_BATCH, opener=_rgb)
        return pictures.picture_vectors(self.folder, paths, PICTURE_BATCH, opener=_rgb)


def _folder_of(choice: str, suggested: tuple, download: tuple | None, what: str, node: str) -> tuple[str, str]:
    """The checkpoint folder a model menu entry names, fetched when it is a suggested one."""
    from ..io import picker
    from ..llm import embeddings
    from ..model import model_directories

    model_directories(embeddings.FOLDER)
    label = picker.model_name(choice, (embeddings.FOLDER,), folder=True).replace("\\", "/").strip()
    folder = embeddings._local_folder(label)
    if folder is None:
        if label not in suggested:
            raise ValueError(
                f"{node} cannot find the {what} `{label[:80]}` in models/LLM. Pick one from the menu, "
                f"or put a Hugging Face checkpoint folder holding config.json in models/LLM."
            )
        folder = embeddings._fetched(label, download) if download else embeddings._fetched(label)
    embeddings._require_safetensors(folder, label, node)
    return label, folder


def text_model(choice: str, node: str) -> TextModel:
    """The text model a menu entry names.

    Raises:
        ValueError: It names nothing usable, or a CLIP or SigLIP model.
        ModelUnavailable: A suggested model is not here and ``features.network`` is off.
    """
    from ..llm import embeddings

    if str(choice).strip() == KEYWORDS:
        return TextModel(KEYWORDS)
    label, folder = _folder_of(choice, (*TEXT_SUGGESTED, *embeddings.SUGGESTED), None, "text model", node)
    if embeddings.model_type(folder) in embeddings.PICTURE_TYPES:
        raise ValueError(
            f"{node}: {label} is a picture model. Pick it as picture_model, and a text model such "
            f"as {DEFAULT_TEXT_MODEL} as text_model."
        )
    from ..model import embedding_gemma2

    if embeddings.model_type(folder) == embedding_gemma2.MODEL_TYPE:
        return TextModel(label, folder, "mean", embedding_gemma2.QUERY, embedding_gemma2.TEXT, embedding_gemma2.MODEL_TYPE)
    settings = embeddings.folder_settings(folder)
    pooling = settings["pooling"] or ("last token" if settings["decoder"] else "cls" if "bge" in label.lower() else "mean")
    family_query, family_text = embeddings.family_instructions(label)
    if "qwen3-embedding" in label.lower():
        query = f"Instruct: {TASK}\nQuery:"
    else:
        query = settings["query"] or family_query
    return TextModel(label, folder, pooling, query, settings["text"] or family_text)


def picture_model(choice: str, node: str) -> PictureModel | None:
    """The picture model a menu entry names, ``None`` for :data:`NO_PICTURE_MODEL`.

    Raises:
        ValueError: It names nothing usable, or a model that does not embed pictures.
        ModelUnavailable: A suggested model is not here and ``features.network`` is off.
    """
    from ..llm import embeddings, pictures

    if str(choice).strip() == NO_PICTURE_MODEL:
        return None
    label, folder = _folder_of(
        choice, (*PICTURE_SUGGESTED, *pictures.SUGGESTED), pictures.DOWNLOAD_FILES, "picture model", node,
    )
    from ..model import embedding_gemma2

    family = embeddings.model_type(folder)
    if family not in embeddings.PICTURE_TYPES and family != embedding_gemma2.MODEL_TYPE:
        raise ValueError(
            f"{node}: {label} is a `{family or 'unknown'}` model, which does not embed pictures. Pick "
            f"EmbeddingGemma 2, a CLIP or a SigLIP model such as {PICTURE_SUGGESTED[0]}, or none."
        )
    return PictureModel(label, folder, family)


# ---------------------------------------------------------------------- vectors


def _cache_file(folder: str, kind: str, signature: str) -> Path:
    """Where one folder's vectors from one model are kept: a hex name in the state directory."""
    from ..config import state_file

    held = f"{CACHE_VERSION}|{_held(folder)}|{kind}|{signature}"
    name = hashlib.sha1(held.encode("utf-8", "surrogatepass")).hexdigest()[:24]
    return state_file(CACHE_FOLDER) / f"{name}-{kind}.npz"


def _load_vectors(path: Path) -> dict:
    """Fingerprint to vector, as a cache file holds them, empty where there is none to read."""
    import numpy as np

    try:
        if not path.is_file() or path.stat().st_size > MAX_CACHE_BYTES:
            return {}
        with np.load(path, allow_pickle=False) as held:
            keys, vectors = held["keys"], held["vectors"]
        if keys.ndim != 1 or vectors.ndim != 2 or len(keys) != len(vectors):
            return {}
        return {str(key): vectors[row] for row, key in enumerate(keys)}
    except Exception as error:
        logger.debug("a picture library cache could not be read (%s)", type(error).__name__)
        return {}


def _save_vectors(path: Path, rows: dict) -> None:
    """Write fingerprint to vector rows to a cache file, replacing it in one step."""
    import numpy as np

    scratch = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(rows), dtype=str)
        vectors = np.stack(list(rows.values())).astype(np.float16) if rows else np.zeros((0, 0), np.float16)
        with open(scratch, "wb") as handle:
            np.savez(handle, keys=keys, vectors=vectors)
        os.replace(scratch, path)
    except OSError as error:
        logger.warning("Image Library Search could not save its vectors (%s)", error.strerror)
        try:
            scratch.unlink()
        except OSError:
            pass


def _cached(path: Path, fingerprints: list[str], make, what: str) -> tuple:
    """Vectors for every fingerprint, made only for those the cache file lacks.

    Args:
        path: The cache file.
        fingerprints: One per entry, in entry order.
        make: Called with the entry positions to embed, answering their vectors as a tensor,
            or ``None`` to read the cache alone.
        what: What is embedded, for the progress log.

    Returns:
        ``(float32 tensor of unit rows in entry order, rows made, rows reused, rows missing)``.
        The tensor is ``None`` where ``make`` is ``None`` and a row is missing.
    """
    import numpy as np
    import torch

    held = _load_vectors(path)
    wanted = list(dict.fromkeys(fingerprints))
    missing = [key for key in wanted if key not in held]
    if missing and make is None:
        return None, 0, len(wanted) - len(missing), len(missing)
    if missing:
        first: dict[str, int] = {}
        for position, key in enumerate(fingerprints):
            first.setdefault(key, position)
        started = time.monotonic()
        made = make([first[key] for key in missing]).numpy().astype(np.float16)
        held.update(zip(missing, made))
        logger.info("Image Library Search embedded %d %s in %.1fs", len(missing), what, time.monotonic() - started)
    kept = {key: held[key] for key in wanted}
    if make is not None and (missing or len(kept) != len(held)):
        _save_vectors(path, kept)
    widths = {row.shape[0] for row in kept.values()}
    if len(widths) > 1:
        raise RuntimeError("the cached vectors disagree on their length; delete the image_library cache folder")
    if not fingerprints:
        return torch.zeros((0, 0)), len(missing), 0, 0
    matrix = torch.from_numpy(np.stack([kept[key] for key in fingerprints]).astype(np.float32))
    return torch.nn.functional.normalize(matrix, p=2, dim=-1), len(missing), len(wanted) - len(missing), 0


def _fingerprint(*parts) -> str:
    """A short digest of the parts."""
    return hashlib.sha1("\0".join(str(part) for part in parts).encode("utf-8", "surrogatepass")).hexdigest()[:24]


def _rgb(path: str):
    """A picture file as an RGB PIL picture, transparency laid over white."""
    from PIL import Image

    from .sizing import open_bytes

    with open(path, "rb") as handle:
        data = handle.read(MAX_PICTURE_BYTES + 1)
    if len(data) > MAX_PICTURE_BYTES:
        raise ValueError(f"a library picture is larger than {MAX_PICTURE_BYTES // (1024 * 1024)} MB")
    picture = open_bytes(data)
    if picture.getextrema()[3][0] < 255:
        backdrop = Image.new("RGBA", picture.size, (255, 255, 255, 255))
        picture = Image.alpha_composite(backdrop, picture)
    return picture.convert("RGB")


def load_picture(path: str):
    """A library picture as a one-frame ``IMAGE`` tensor at its own size.

    Raises:
        ValueError: The file is past :data:`MAX_PICTURE_BYTES`.
        NotAnImage: It is not a picture.
        ImageTooLarge: Its header declares more pixels than one picture may hold.
    """
    from ..convert.tensors import pil2tensor

    return pil2tensor(_rgb(path))


def _terms(text: str) -> list[str]:
    """The words a keyword ranking counts, lowercased, short plural endings dropped."""
    found = []
    for word in _WORD.findall(str(text).lower()):
        if word in _STOPWORDS:
            continue
        if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        found.append(word)
    return found


@dataclass
class Library:
    """A read and embedded picture library: its pictures, vectors and models.

    Attributes:
        label: The folder as its menu names it.
        source: Where the captions came from.
        entries: The pictures.
        text_model: How captions are ranked.
        text: Caption vectors, one row per entry, ``None`` for keyword ranking.
        keywords: The keyword ranker, ``None`` where a model ranks.
        picture_model: The picture model, or ``None``.
        pictures: Picture vectors, one row per entry, or ``None``.
        figures: What building it did, for its report.
        states: Folder label to ``(index state, picture count)``, for every folder read.
        shortened: Caption vectors cut to a length, by that length.
    """

    label: str
    source: str
    entries: list[Entry]
    text_model: TextModel
    text: object = None
    keywords: object = None
    picture_model: PictureModel | None = None
    pictures: object = None
    figures: dict = field(default_factory=dict)
    states: dict = field(default_factory=dict)
    shortened: dict = field(default_factory=dict)

    def find(self, key: str) -> Entry | None:
        """The entry a picture's key names, or ``None``."""
        return next((entry for entry in self.entries if entry.key == key), None)


#: Index states a folder is reported in.
INDEXED = "indexed"
PARTLY = "partly indexed"
NOT_INDEXED = "not indexed"


def build(label: str, read_result: Read, text: TextModel, picture: PictureModel | None, embed: bool = True) -> Library:
    """Embed a library's captions and pictures folder by folder, reusing every cached vector.

    Args:
        label: The library folder as its menu names it.
        read_result: What :func:`read` found.
        text: The text model.
        picture: The picture model, or ``None``.
        embed: Make missing vectors. Off, a folder whose vectors are not all cached is left out.

    Returns:
        The library, its ``states`` naming each folder's index state.
    """
    import torch

    figures = {"embedded captions": 0, "reused captions": 0, "embedded pictures": 0, "reused pictures": 0}
    states: dict[str, tuple[str, int]] = {}
    text_blocks, picture_blocks, kept_units = [], [], []
    for unit in read_result.units:
        complete = True
        partial = False
        text_block = picture_block = None
        if not text.keywords:
            captions = [entry.caption for entry in unit.entries]
            prints = [_fingerprint(text.signature, caption) for caption in captions]
            text_block, made, reused, missing = _cached(
                _cache_file(unit.folder, "text", f"{text.signature}|{unit.source}"), prints,
                (lambda positions, captions=captions: text.encode([captions[at] for at in positions], query=False))
                if embed else None,
                "captions",
            )
            figures["embedded captions"] += made
            figures["reused captions"] += reused
            complete = complete and not missing
            partial = partial or bool(reused)
        if picture is not None:
            prints = [
                _fingerprint(picture.label, _relative(entry.path, Path(unit.folder)), entry.size, entry.modified)
                for entry in unit.entries
            ]
            picture_block, made, reused, missing = _cached(
                _cache_file(unit.folder, "picture", picture.label), prints,
                (lambda positions, unit=unit: picture.encode_pictures([unit.entries[at].preview for at in positions]))
                if embed else None,
                "pictures",
            )
            figures["embedded pictures"] += made
            figures["reused pictures"] += reused
            complete = complete and not missing
            partial = partial or bool(reused)
        if not complete:
            states[unit.label] = (PARTLY if partial else NOT_INDEXED, len(unit.entries))
            continue
        states[unit.label] = (INDEXED, len(unit.entries))
        kept_units.append(unit)
        if text_block is not None:
            text_blocks.append(text_block)
        if picture_block is not None:
            picture_blocks.append(picture_block)
    entries = [entry for unit in kept_units for entry in unit.entries]
    library = Library(label, read_result.source, entries, text, picture_model=picture)
    if text.keywords:
        from ..llm.bm25 import Okapi

        library.keywords = Okapi([_terms(f"{entry.caption} {entry.name}") for entry in entries])
    else:
        library.text = torch.cat(text_blocks) if text_blocks else None
    if picture is not None:
        library.pictures = torch.cat(picture_blocks) if picture_blocks else None
    figures["folders"] = len(kept_units)
    library.figures = figures
    library.states = states
    return library


def select(found: Read, folders: str) -> Read:
    """The part of a read whose folders a selection names.

    Args:
        found: What :func:`read` found.
        folders: Folder labels below the library folder, one per line, ``.`` for the library
            folder itself. Empty keeps every folder.

    Returns:
        A read holding only those folders.
    """
    wanted = {line.strip().strip("/") for line in str(folders or "").splitlines() if line.strip()}
    if not wanted:
        return found
    units = [unit for unit in found.units if unit.label in wanted]
    return Read(
        units=units,
        entries=[entry for unit in units for entry in unit.entries],
        skipped=found.skipped,
        captioned=found.captioned,
        found=found.found,
    )


# ---------------------------------------------------------------------- searching


@dataclass(frozen=True)
class Match:
    """One ranked picture.

    Attributes:
        entry: The picture.
        score: What it was ranked by.
        text_score: How close its caption is, or ``None``.
        picture_score: How close the picture itself is, or ``None``.
    """

    entry: Entry
    score: float
    text_score: float | None
    picture_score: float | None


def pieces(prompt: str) -> list[str]:
    """A prompt and the sentences and runs of up to three clauses it is made of.

    Args:
        prompt: The prompt or description.

    Returns:
        At most :data:`MAX_PIECES` distinct texts, the whole prompt first.
    """
    text = " ".join(str(prompt or "")[:MAX_PROMPT_CHARS].split())
    if not text:
        return []
    found = [text]
    windows = []
    for sentence in _SENTENCES.split(text):
        clauses = [clause.strip() for clause in _CLAUSES.split(sentence) if clause.strip()]
        if not clauses:
            continue
        found.append(", ".join(clauses))
        for size in (3, 2, 1):
            for start in range(0, max(0, len(clauses) - size + 1)):
                windows.append(", ".join(clauses[start:start + size]))
    unique = list(dict.fromkeys(piece for piece in found + windows if len(piece) > 2))
    return unique[:MAX_PIECES]


def _allowed(entries: list[Entry], wanted: str):
    """Which entries a filter keeps: every plain word present, no ``-word`` present."""
    import torch

    words = str(wanted or "").lower().split()
    need = [word for word in words if not word.startswith("-")]
    refuse = [word[1:] for word in words if word.startswith("-") and len(word) > 1]
    keep = []
    for entry in entries:
        haystack = f"{entry.caption} {entry.key}".lower()
        keep.append(all(word in haystack for word in need) and not any(word in haystack for word in refuse))
    return torch.tensor(keep, dtype=torch.bool)


def _standard(scores):
    """Scores moved to mean 0 and spread 1 across the library."""
    spread = scores.std()
    return (scores - scores.mean()) / spread if float(spread) > 1e-6 else scores - scores.mean()


#: What the view setting offers: the prompt's view words, any view, or one view.
VIEW_CHOICES = ("auto", "any", *VIEWS)


def _shortened(matrix, dimensions: int):
    """Rows cut to their first ``dimensions`` numbers and scaled to length 1, or unchanged for 0."""
    import torch

    if not dimensions or dimensions >= matrix.shape[-1]:
        return matrix
    return torch.nn.functional.normalize(matrix[..., :dimensions], p=2, dim=-1)


def search(
    library: Library, prompt: str, top_k: int, picture_weight: float, wanted: str = "", distinct: bool = True,
    view: str = "auto", dimensions: int = 0,
) -> tuple[list[Match], int]:
    """Rank a library against a prompt.

    Args:
        library: The library.
        prompt: A description of the picture wanted, or a whole generation prompt.
        top_k: Most matches answered.
        picture_weight: Share of the ranking the picture model decides, 0 to 1; at least
            :data:`FILE_NAME_PICTURE_WEIGHT` for a picture captioned only by its file name.
        wanted: Filter words: each plain word must be in a caption or path, no ``-word`` may be.
        distinct: Keep only one picture of each group, such as one angle of each pose.
        view: ``auto`` for the prompt's view words, ``any`` for none, or one of :data:`VIEWS`.
        dimensions: Caption vector length compared, 0 for the model's own.

    Within a group the pictures whose camera view the prompt names come first, so the prompt's
    view words choose the angle and never the pose.

    Returns:
        ``(matches best first, entries the filter kept)``.

    Raises:
        ValueError: The prompt is empty.
    """
    import torch

    parts = pieces(prompt)
    if not parts:
        raise ValueError("Image Library Search was given an empty prompt. Describe the picture wanted, or wire a prompt in.")
    entries = library.entries
    keep = _allowed(entries, wanted)
    if not entries or not bool(keep.any()):
        return [], int(keep.sum()) if entries else 0
    text_scores = picture_scores = None
    if library.keywords is not None:
        raw = torch.from_numpy(library.keywords.scores(_terms(prompt))).float()
        best = float(raw.max()) if raw.numel() else 0.0
        text_scores = raw / best if best > 0 else raw
    elif library.text is not None:
        width = max(0, int(dimensions))
        queries = _shortened(library.text_model.encode(parts, query=True), width)
        held = library.shortened.get(width)
        if held is None:
            held = library.shortened[width] = _shortened(library.text, width)
        text_scores = (queries @ held.T).topk(min(BEST_PIECES, len(parts)), dim=0).values.mean(dim=0)
    if library.pictures is not None and library.picture_model is not None:
        queries = library.picture_model.encode_texts(parts)
        picture_scores = (queries @ library.pictures.T).topk(min(BEST_PIECES, len(parts)), dim=0).values.mean(dim=0)
    weight = max(0.0, min(1.0, float(picture_weight)))
    if picture_scores is None or weight <= 0.0:
        ranking = text_scores
    elif text_scores is None or weight >= 1.0:
        ranking = picture_scores
    else:
        weights = torch.tensor(
            [max(weight, FILE_NAME_PICTURE_WEIGHT) if entry.source == "file name" else weight for entry in entries],
            dtype=torch.float32,
        )[keep]
        ranking = (1.0 - weights) * _standard(text_scores[keep]) + weights * _standard(picture_scores[keep])
        full = torch.full((len(entries),), float("-inf"))
        full[keep] = ranking
        ranking = full
    ranking = ranking.masked_fill(~keep, float("-inf"))
    count = min(max(1, int(top_k)), int(keep.sum()))
    views = prompt_views(prompt) if view == "auto" else set() if view not in VIEWS else {view}
    groups: dict[str, list[int]] = {}
    for at in torch.argsort(ranking, descending=True)[:int(keep.sum())].tolist():
        groups.setdefault(entries[at].group or entries[at].key, []).append(at)
    order: list[int] = []
    copies: set[tuple[str, int]] = set()
    for members in groups.values():
        if views:
            members = sorted(members, key=lambda at: entries[at].view not in views)
        members = [at for at in members if (entries[at].name, entries[at].size) not in copies]
        for at in members[:1] if distinct else members:
            copies.add((entries[at].name, entries[at].size))
            order.append(at)
            if len(order) >= count:
                break
        if len(order) >= count:
            break
    matches = [
        Match(
            entries[at],
            float(ranking[at]),
            None if text_scores is None else float(text_scores[at]),
            None if picture_scores is None else float(picture_scores[at]),
        )
        for at in order
    ]
    return matches, int(keep.sum())

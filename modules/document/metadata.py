"""The metadata a document carries, and how ``meta.json`` spells it.

:class:`Metadata` is frozen. Timestamps are UTC in the :data:`STAMP_FORMAT` spelling, and
word and character counts are not fields here.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping

from .. import log

__all__ = [
    "GENERATOR",
    "STAMP_FORMAT",
    "Metadata",
    "from_dict",
    "now_stamp",
    "touched",
    "with_stamps",
]

logger = log.get_logger("document.metadata")

#: How a timestamp is written: seconds resolution, UTC, with the ``Z`` that says so. Every
#: export target takes a date in this shape or one derived from it, and it sorts as text.
STAMP_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

#: What :func:`with_stamps` records as having produced a document. A version number is not
#: part of it: the container states its own version, and a support question about a file is
#: answered by that rather than by the build that happened to write it.
GENERATOR = "WAS Node Suite"


@dataclass(frozen=True)
class Metadata:
    """The authored metadata of one document.

    Attributes:
        title: What the document is called. The name an export carries as its title, and
            the one a file manager shows in a document column.
        description: A sentence or two saying what the document is. Written to the
            description field of every export format.
        author: Who wrote it.
        copyright: The rights statement, such as ``"(c) 2026 A. Name, CC BY 4.0"``. Free
            text rather than a licence identifier.
        language: BCP 47 tag for the language the document is written in, such as ``"en"``
            or ``"pt-BR"``. Decides hyphenation and spell checking in an exported file, and
            is what a screen reader picks a voice from.
        keywords: Search terms, in the order they were given. Every export format has a
            keyword field, and it is the one metadata slot a desktop search engine reads.
        created: When the document was first made, in :data:`STAMP_FORMAT`.
        modified: When its content last changed, in :data:`STAMP_FORMAT`.
        generator: What produced it, :data:`GENERATOR` for a document this pack made.
        custom: Any further pairs of text the author wants carried. All three export
            formats have a user-defined property table, so a pair put here survives an
            export instead of being dropped.
    """

    title: str = ""
    description: str = ""
    author: str = ""
    copyright: str = ""
    language: str = ""
    keywords: tuple[str, ...] = ()
    created: str = ""
    modified: str = ""
    generator: str = ""
    custom: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Both collections are normalized here rather than at every call site, and both end
        # up read-only, so a consumer handed a document cannot reach into its metadata.
        object.__setattr__(self, "keywords", _as_keywords(self.keywords))
        object.__setattr__(self, "custom", MappingProxyType(_as_custom(self.custom)))

    def __reduce__(self) -> tuple:
        """Rebuild through :func:`from_dict`.

        Returns:
            That function and the dictionary :meth:`to_dict` writes, which carries every
            field. Copying and pickling both come here.
        """
        return (from_dict, (self.to_dict(),))

    def to_dict(self) -> dict[str, Any]:
        """The fields as ``meta.json`` writes them.

        Returns:
            A new dictionary in the order the file lists them. ``keywords`` is a list and
            ``custom`` a plain dictionary, so the result is ready for :mod:`json`. The
            container version and the two counts are added by :mod:`.container`, which owns
            both.
        """
        return {
            "title": self.title,
            "description": self.description,
            "author": self.author,
            "copyright": self.copyright,
            "language": self.language,
            "keywords": list(self.keywords),
            "created": self.created,
            "modified": self.modified,
            "generator": self.generator,
            "custom": dict(self.custom),
        }


def now_stamp() -> str:
    """The current UTC time in :data:`STAMP_FORMAT`."""
    return datetime.now(timezone.utc).strftime(STAMP_FORMAT)


def with_stamps(metadata: Metadata) -> Metadata:
    """The record with any empty timestamp and generator filled in.

    Args:
        metadata: The record to complete.

    Returns:
        A record whose ``created``, ``modified`` and ``generator`` are set. A field that
        already holds something is left exactly as it is, so reading a document and writing
        it back never rewrites its history.
    """
    stamp = now_stamp()
    return replace(
        metadata,
        created=metadata.created or stamp,
        modified=metadata.modified or stamp,
        generator=metadata.generator or GENERATOR,
    )


def touched(metadata: Metadata) -> Metadata:
    """The record with ``modified`` set to now.

    Args:
        metadata: The record whose document has just changed.

    Returns:
        A new record. ``created`` is untouched, and is filled in when it was empty, so a
        document that was never stamped gets both stamps at its first edit.
    """
    stamp = now_stamp()
    return replace(metadata, created=metadata.created or stamp, modified=stamp)


def from_dict(payload: Mapping[str, Any]) -> Metadata:
    """Read a record out of what ``meta.json`` held.

    Args:
        payload: The decoded JSON object. Keys it does not define are ignored: the fields
            of this record are the whole of the format, and further pairs of an author's
            own belong under ``custom``.

    Returns:
        The record. A missing field reads as empty, a number or a boolean is read as its
        text, and a value of a shape that cannot be read as text at all reads as empty and
        is logged.
    """
    return Metadata(
        title=_as_text(payload.get("title")),
        description=_as_text(payload.get("description")),
        author=_as_text(payload.get("author")),
        copyright=_as_text(payload.get("copyright")),
        language=_as_text(payload.get("language")),
        keywords=_as_keywords(payload.get("keywords")),
        created=_as_text(payload.get("created")),
        modified=_as_text(payload.get("modified")),
        generator=_as_text(payload.get("generator")),
        custom=_as_custom(payload.get("custom")),
    )


def _as_text(value: Any) -> str:
    """One metadata value as text.

    Args:
        value: Whatever the file held.

    Returns:
        The string itself, the text of a number or a boolean, or an empty string for
        anything else, which is reported at debug level.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (bool, int, float)):
        return str(value)
    logger.debug(
        "a metadata field holds a %s, which is not text, and was read as empty",
        type(value).__name__,
    )
    return ""


def _as_keywords(value: Any) -> tuple[str, ...]:
    """Keywords as a tuple of non-empty strings.

    Args:
        value: A list of keywords, or one string holding them separated by commas, which
            is how every export format spells the same field.

    Returns:
        The keywords in the order given, each stripped, with blanks dropped.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        entries = value.split(",")
    elif isinstance(value, (list, tuple, set, frozenset)):
        entries = list(value)
    else:
        logger.debug(
            "keywords hold a %s rather than a list or a comma-separated string, and were "
            "read as none at all",
            type(value).__name__,
        )
        return ()
    found = []
    for entry in entries:
        word = _as_text(entry).strip()
        if word:
            found.append(word)
    return tuple(found)


def _as_custom(value: Any) -> dict[str, str]:
    """The user-defined pairs as a plain dictionary of text.

    Args:
        value: A mapping of names to values.

    Returns:
        A new dictionary, every key and value read as text. Anything that is not a mapping
        reads as empty and is reported at debug level.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        logger.debug(
            "the custom metadata holds a %s rather than a mapping of names to values, and "
            "was read as empty",
            type(value).__name__,
        )
        return {}
    return {str(name): _as_text(entry) for name, entry in value.items()}

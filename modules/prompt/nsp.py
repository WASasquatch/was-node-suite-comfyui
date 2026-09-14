"""Noodle Soup Prompts terminology substitution.

A noodle is a terminology name in a delimiter, ``__animals__`` by default, replaced with a
random entry of that term. Terms live in the state database, one record per entry.
"""

from __future__ import annotations

import base64
import json
import random
import time
import zlib
from collections.abc import Mapping
from pathlib import Path

from . import state_path
from .. import log
from ..state import store as store_module

__all__ = [
    "PANTRY_FILE",
    "bundled_pantry",
    "SOURCE",
    "YOURS",
    "add_entries",
    "add_term",
    "delete_term",
    "entry_mark",
    "ensure_pantry",
    "export_pantry",
    "generation",
    "import_pantry",
    "local_counts",
    "local_entries",
    "nsp_parse",
    "pantry",
    "pantry_file",
    "remove_entries",
    "search_entries",
    "set_term_entries",
    "term_entries",
    "term_page",
    "terms",
]

logger = log.get_logger("prompt.nsp")

#: The file name a pantry is imported from and exported to when no other is named.
PANTRY_FILE = "nsp_pantry.json"

#: Meta key whose value changes with every pantry write.
GENERATION = "nsp:generation"

#: Meta key holding the published pantry the stored one was last merged with.
UPSTREAM = "nsp:upstream"

#: Meta key holding the entries removed by hand, which no refresh puts back.
REMOVED = "nsp:removed"

#: Record field naming where an entry came from. An entry the published pantry supplied
#: carries no field at all.
SOURCE = "source"

#: The :data:`SOURCE` value on an entry added from a node or brought in from a file.
YOURS = "yours"

#: The fields written on an entry added from a node or brought in from a file.
MINE = {SOURCE: YOURS}

_terms: dict[str, int] | None = None
_generation: str | None = None
_published: tuple[str, dict[str, set[str]] | None] | None = None


def pantry_file() -> Path:
    """The pantry file's path, ``<config dir>/nsp_pantry.json``.

    Returns:
        The path a pantry is imported from and exported to by default.
    """
    return state_path(PANTRY_FILE)


def _database():
    """The shared state database."""
    return store_module.shared_store()


def terms() -> dict[str, int]:
    """Every term and how many entries it has, in pantry order.

    Returns:
        ``{term: entries}``, a term with no entries included at 0.
    """
    global _terms, _generation
    database = _database()
    generation = database.meta(GENERATION, "")
    if _terms is None or generation != _generation:
        _terms = database.record_counts(store_module.NSP)
        _generation = generation
    return dict(_terms)


def generation(default: str = "") -> str:
    """The stamp that changes with every pantry write.

    Args:
        default: Answered when the store holds no stamp.

    Returns:
        The stamp, which is what a cache of anything read from the pantry is keyed on.
    """
    return _database().meta(GENERATION, default) or default


def term_entries(term: str) -> list[str]:
    """Every entry of one term, in order, repeats included."""
    return _database().record_names(store_module.NSP, term)


def term_page(term: str, start: int = 0, limit: int = 500) -> list[tuple[str, bool]]:
    """A window of one term's entries, each with whether it is the reader's own.

    Args:
        term: The terminology name.
        start: How many entries to pass over first.
        limit: At most this many entries.

    Returns:
        ``[(entry, mine)]`` in pantry order, empty for a start past the end.
    """
    name = str(term)
    sets = _published_sets()
    published = None if sets is None else sets.get(name, set())
    return [
        (record.name, _mine(record.name, record.fields, published))
        for record in _database().record_page(store_module.NSP, name, start, limit)
    ]


def entry_mark(term: str, entry: str) -> bool | None:
    """Whether one entry of one term is the reader's own.

    Space around a stored entry is ignored.

    Args:
        term: The terminology name.
        entry: The entry.

    Returns:
        True for an entry added from a node or a file, False for one the published pantry
        supplied, and None when the term does not hold it.
    """
    name = str(term)
    record = _database().find_record(store_module.NSP, name, str(entry))
    if record is None:
        record = _padded_record(name, str(entry))
    if record is None:
        return None
    sets = _published_sets()
    published = None if sets is None else sets.get(name, set())
    return _mine(record.name, record.fields, published)


def _padded_record(term: str, entry: str):
    """The record of one term whose text differs from an entry only by space around it.

    Args:
        term: The terminology name.
        entry: The entry.

    Returns:
        The record, or None when the term holds no such entry.
    """
    wanted = entry.strip()
    if not wanted:
        return None
    for record in _database().records(store_module.NSP, term):
        if record.name != wanted and record.name.strip() == wanted:
            return record
    return None


def search_entries(
    needle: str, start: int = 0, limit: int = 500, ceiling: int = 10_000
) -> tuple[list[tuple[str, str, bool]], int]:
    """The entries whose text holds ``needle``, across every term.

    Args:
        needle: Text matched anywhere in an entry, case-insensitively for ASCII.
        start: How many matches to pass over first.
        limit: At most this many matches returned.
        ceiling: How many matches are counted at all.

    Returns:
        ``([(term, entry, mine)], total)`` in pantry order. ``total`` is ``ceiling`` when
        there are at least that many matches.
    """
    rows, total = _database().record_matches(store_module.NSP, needle, start, limit, ceiling)
    sets = _published_sets()
    found = []
    for record in rows:
        published = None if sets is None else sets.get(record.category, set())
        found.append(
            (record.category, record.name, _mine(record.name, record.fields, published))
        )
    return found, total


def pantry() -> dict[str, list[str]]:
    """The whole stored pantry.

    Returns:
        ``{term: [entry, ...]}``, in pantry order.
    """
    return {term: [name for name, _fields in rows] for term, rows in _records().items()}


def _records() -> dict[str, list[tuple[str, dict | None]]]:
    """The whole stored pantry with each entry's fields, as ``{term: [(entry, fields)]}``."""
    grouped = _database().dump_records(store_module.NSP)
    return {
        term: [(record.name, record.fields) for record in records]
        for term, records in grouped.items()
    }


def _term_records(term: str) -> list[tuple[str, dict | None]]:
    """One term's entries with their fields, as ``[(entry, fields)]``."""
    return [
        (record.name, record.fields)
        for record in _database().records(store_module.NSP, str(term))
    ]


def _mine(name: str, fields, published) -> bool:
    """Whether one entry is the reader's own rather than the published pantry's.

    Args:
        name: The entry.
        fields: The entry's record fields.
        published: The entries the published pantry supplied for that term, or None when
            no published pantry has been recorded.

    Returns:
        True for an entry added from a node, brought in from a file, or absent from the
        published pantry the store was last merged with.
    """
    if (fields or {}).get(SOURCE) == YOURS:
        return True
    return False if published is None else name not in published


def _published_sets() -> dict[str, set[str]] | None:
    """The published pantry recorded by the last refresh, as a set per term, or None."""
    # Held against the store's version stamp, so a run of reads costs one decompression.
    global _published
    stamp = generation()
    if _published is not None and _published[0] == stamp:
        return _published[1]
    snapshot = _snapshot()
    built = None if snapshot is None else {
        term: set(entries) for term, entries in snapshot.items()
    }
    _published = (stamp, built)
    return built


def local_entries(term: str) -> list[str]:
    """One term's entries that did not come from the published pantry, in order.

    Args:
        term: The terminology name.

    Returns:
        Every entry added from a node, imported from a file, or absent from the published
        pantry the store was last merged with.
    """
    name = str(term)
    sets = _published_sets()
    published = None if sets is None else sets.get(name, set())
    return [entry for entry, fields in _term_records(name) if _mine(entry, fields, published)]


def local_counts() -> dict[str, int]:
    """How many entries of each term did not come from the published pantry.

    Returns:
        ``{term: entries}``, in pantry order, a term with none included at 0.
    """
    sets = _published_sets()
    counts = {}
    for term, rows in _records().items():
        published = None if sets is None else sets.get(term, set())
        counts[term] = sum(1 for entry, fields in rows if _mine(entry, fields, published))
    return counts


# Writing


def _write(work, label: str) -> bool:
    """Run one pantry change and its version stamp in a single transaction.

    Args:
        work: Called with the connection inside the transaction.
        label: What the change is, named in the log line when it fails.

    Returns:
        True when the transaction committed.
    """

    def transaction(connection):
        work(connection)
        store_module.write_meta(connection, GENERATION, str(time.time_ns()))

    return _database().write(transaction, label)


def _removed() -> dict[str, list[str]]:
    """The entries removed by hand, as ``{term: [entry, ...]}``."""
    try:
        stored = json.loads(_database().meta(REMOVED) or "{}")
    except ValueError:
        return {}
    if not isinstance(stored, Mapping):
        return {}
    return {
        str(term): [str(entry) for entry in entries]
        for term, entries in stored.items()
        if isinstance(entries, list)
    }


def _snapshot() -> dict[str, list[str]] | None:
    """The published pantry the stored one was last merged with, or None when unknown."""
    packed = _database().meta(UPSTREAM)
    if not packed:
        return None
    try:
        return json.loads(zlib.decompress(base64.b64decode(packed)).decode("utf-8"))
    except (ValueError, zlib.error) as error:
        logger.warning(
            "the record of the published pantry could not be read (%s), so a refresh adds "
            "and updates terms without retiring any",
            error,
        )
        return None


def _pack(pantry_data: Mapping[str, list[str]]) -> str:
    """A published pantry as one compressed line of text."""
    body = json.dumps(dict(pantry_data), separators=(",", ":")).encode("utf-8")
    return base64.b64encode(zlib.compress(body, 9)).decode("ascii")


def _change(term: str, added=0, removed=0, skipped=0, saved=False) -> dict:
    """One edit's counts, read back from the store after it committed.

    Args:
        term: The terminology the edit named.
        added: Entries stored.
        removed: Entries dropped.
        skipped: Entries the edit left alone, already there or not there at all.
        saved: Whether the write committed.

    Returns:
        ``term``, ``added``, ``removed``, ``skipped``, ``total`` entries the term holds
        now, ``local`` of them, and ``saved``.
    """
    return {
        "term": term,
        "added": added,
        "removed": removed,
        "skipped": skipped,
        "total": len(term_entries(term)),
        "local": len(local_entries(term)),
        "saved": saved,
    }


def set_term_entries(term: str, entries) -> dict:
    """Store one term's entries, replacing whatever it held.

    A new entry is recorded as the reader's own.

    Args:
        term: The terminology name, created when it is not there.
        entries: The entries, in the order a draw should index them.

    Returns:
        The counts :func:`_change` builds.
    """
    name = str(term)
    known = dict(_term_records(name))
    wanted = [str(entry) for entry in entries]
    rows = [(entry, known.get(entry, MINE)) for entry in wanted]
    fresh = sum(1 for entry in wanted if entry not in known)
    dropped = [entry for entry in known if entry not in set(wanted)]
    tombstones = _merge_removed(name, sorted(dropped))

    def work(connection):
        store_module.write_category(connection, store_module.NSP, name, rows)
        store_module.write_meta(connection, REMOVED, json.dumps(tombstones))

    saved = _write(work, f"the {name} terminology")
    return _change(name, added=fresh, removed=len(dropped), saved=saved)


def add_term(term: str) -> dict:
    """Create a term with no entries, leaving one that already exists alone.

    Args:
        term: The terminology name.

    Returns:
        The counts :func:`_change` builds, ``skipped`` at 1 for a term already there.
    """
    name = str(term)
    known = name in _database().record_categories(store_module.NSP)

    def work(connection):
        store_module.write_record_category(connection, store_module.NSP, name)

    saved = _write(work, f"the {name} terminology")
    return _change(name, skipped=1 if known else 0, saved=saved)


def delete_term(term: str) -> dict:
    """Remove a term and every entry in it, and record its entries as removed by hand.

    Args:
        term: The terminology name.

    Returns:
        The counts :func:`_change` builds, ``removed`` counting the entries dropped.
    """
    name = str(term)
    held = term_entries(name)
    tombstones = _merge_removed(name, held)

    def work(connection):
        store_module.drop_record_category(connection, store_module.NSP, name)
        store_module.write_meta(connection, REMOVED, json.dumps(tombstones))

    saved = _write(work, f"deleting the {name} terminology")
    return _change(name, removed=len(held), saved=saved)


def add_entries(term: str, entries) -> dict:
    """Add the entries a term does not hold to the end of it, as the reader's own.

    Args:
        term: The terminology name.
        entries: The entries to add, in the order they should be indexed.

    Returns:
        The counts :func:`_change` builds, ``skipped`` counting the entries already there.
    """
    name = str(term)
    held = _term_records(name)
    present = {entry for entry, _fields in held}
    wanted = [str(entry) for entry in entries]
    fresh: list[str] = []
    for entry in wanted:
        if entry not in present:
            fresh.append(entry)
            present.add(entry)
    if not fresh:
        saved = _write(
            lambda connection: store_module.write_record_category(
                connection, store_module.NSP, name
            ),
            f"the {name} terminology",
        )
        return _change(name, skipped=len(wanted), saved=saved)

    tombstones = _removed()
    remaining = [entry for entry in tombstones.get(name, []) if entry not in set(fresh)]
    if remaining:
        tombstones[name] = remaining
    else:
        tombstones.pop(name, None)
    rows = held + [(entry, MINE) for entry in fresh]

    def work(connection):
        store_module.write_category(connection, store_module.NSP, name, rows)
        store_module.write_meta(connection, REMOVED, json.dumps(tombstones))

    saved = _write(work, f"{len(fresh)} entry(s) in the {name} terminology")
    return _change(name, added=len(fresh), skipped=len(wanted) - len(fresh), saved=saved)


def remove_entries(term: str, entries) -> dict:
    """Remove entries from a term and record them as removed by hand.

    Args:
        term: The terminology name.
        entries: The entries to remove. One the term does not hold is ignored.

    Returns:
        The counts :func:`_change` builds, ``skipped`` counting the entries that were not
        there.
    """
    name = str(term)
    dropping = {str(entry) for entry in entries}
    held = _term_records(name)
    remaining = [(entry, fields) for entry, fields in held if entry not in dropping]
    gone = {entry for entry, _fields in held} & dropping
    tombstones = _merge_removed(name, sorted(gone))

    def work(connection):
        store_module.write_category(connection, store_module.NSP, name, remaining)
        store_module.write_meta(connection, REMOVED, json.dumps(tombstones))

    saved = _write(work, f"{len(dropping)} entry(s) of the {name} terminology")
    return _change(
        name, removed=len(held) - len(remaining), skipped=len(dropping - gone), saved=saved
    )


def _merge_removed(term: str, entries) -> dict[str, list[str]]:
    """The removed-by-hand record with ``entries`` added under ``term``."""
    tombstones = _removed()
    known = tombstones.setdefault(term, [])
    for entry in entries:
        if entry not in known:
            known.append(entry)
    return tombstones


# Filling the pantry


def _validate(payload, where: str) -> dict[str, list[str]]:
    """A downloaded or imported document checked for the shape of a pantry.

    Args:
        payload: The parsed document.
        where: The file or address it came from, named in the error.

    Returns:
        ``{term: [entry, ...]}``.

    Raises:
        ValueError: It is not a mapping of terms to lists of entries.
    """
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"{where} holds a {type(payload).__name__} where a list of terminology names "
            f"was expected, so nothing was changed"
        )
    checked: dict[str, list[str]] = {}
    for term, entries in payload.items():
        if not isinstance(entries, list) or not all(isinstance(e, str) for e in entries):
            raise ValueError(
                f"the '{term}' terminology in {where} is not a list of words, so nothing "
                f"was changed"
            )
        checked[str(term)] = list(entries)
    if not checked:
        raise ValueError(f"{where} holds no terminology at all, so nothing was changed")
    return checked


def bundled_pantry() -> dict[str, list[str]]:
    """The terminology snapshot shipped with the pack.

    Returns:
        ``{term: [entry, ...]}``.

    Raises:
        OSError: The snapshot could not be read.
        ValueError: It does not unpack, or does not hold a pantry.
    """
    from ..data.paths import pantry_seed

    path = pantry_seed()
    try:
        body = zlib.decompress(path.read_bytes())
    except zlib.error as error:
        raise ValueError(f"{path} is not a packed pantry ({error})") from error
    return _validate(json.loads(body.decode("utf-8")), str(path))


def _seed(published: Mapping[str, list[str]]) -> bool:
    """Store a published pantry as it stands and record it as the one merged with."""
    shaped = {term: [(entry, None) for entry in entries] for term, entries in published.items()}
    packed = _pack(published)

    def work(connection):
        store_module.write_records(connection, store_module.NSP, shaped)
        store_module.write_meta(connection, UPSTREAM, packed)

    return _write(work, "the Noodle Soup Prompts pantry")


def ensure_pantry() -> bool:
    """Seed the database from the bundled terminology when it holds none.

    Returns:
        True when the database holds a pantry.

    Raises:
        ValueError: The database holds no pantry and the copy shipped with the pack could
            not be read.
    """
    if terms():
        return True
    try:
        _seed(bundled_pantry())
    except (OSError, ValueError) as error:
        logger.debug("the bundled pantry could not be read (%s)", error)
    else:
        return bool(terms())
    raise ValueError(
        f"no Noodle Soup Prompts terminology is stored and the copy shipped with the pack "
        f"could not be read. Reinstall the pack, or put a copy of {PANTRY_FILE} in "
        f"{pantry_file().parent} and load it with Noodle Soup Pantry Import."
    )


def import_pantry(source=None, replace: bool = False) -> dict:
    """Import a pantry file into the database, recording its entries as the reader's own.

    Args:
        source: The JSON file to read. Defaults to :func:`pantry_file`.
        replace: Store exactly what the file holds, dropping every term not in it.

    Returns:
        ``terms`` and ``entries`` the file held, ``terms_added`` and ``entries_added``
        stored by this import, ``total`` entries the pantry holds now, and ``saved``.

    Raises:
        OSError: The file could not be read.
        ValueError: The file is not a pantry.
    """
    path = Path(source) if source is not None else pantry_file()
    with path.open("r", encoding="utf-8") as handle:
        imported = _validate(json.load(handle), str(path))

    held = _records()
    fresh = 0
    if replace:
        merged = {term: [(entry, MINE) for entry in entries] for term, entries in imported.items()}
        fresh = sum(len(entries) for entries in imported.values())
    else:
        merged = dict(held)
        for term, entries in imported.items():
            known = merged.get(term)
            if known is None:
                merged[term] = [(entry, MINE) for entry in entries]
                fresh += len(entries)
                continue
            present = {entry for entry, _fields in known}
            added = [entry for entry in entries if entry not in present]
            merged[term] = known + [(entry, MINE) for entry in added]
            fresh += len(added)

    def work(connection):
        store_module.write_records(connection, store_module.NSP, merged)

    saved = _write(work, f"the pantry imported from {path.name}")
    total = sum(len(entries) for entries in imported.values())
    logger.info(
        "imported %s term(s) and %s entry(s) from %s; the file is left where it is",
        len(imported),
        total,
        path.name,
    )
    return {
        "terms": len(imported),
        "entries": total,
        "terms_added": len([term for term in imported if term not in held]),
        "entries_added": fresh,
        "total": sum(len(rows) for rows in merged.values()),
        "saved": saved,
    }


def export_pantry(target=None, local_only: bool = False) -> dict:
    """Write the stored pantry out as a JSON file.

    Args:
        target: Where to write. Defaults to :func:`pantry_file`.
        local_only: Write only the entries the published pantry did not supply, which is a
            copy of the additions made here on their own.

    Returns:
        ``terms`` and ``entries`` written, ``path`` written to, and ``saved``, which is
        False when the file could not be written.
    """
    path = Path(target) if target is not None else pantry_file()
    if local_only:
        sets = _published_sets()
        data = {}
        for term, rows in _records().items():
            published = None if sets is None else sets.get(term, set())
            local = [entry for entry, fields in rows if _mine(entry, fields, published)]
            if local:
                data[term] = local
    else:
        data = pantry()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=4)
    except OSError as error:
        logger.error("the pantry could not be written to `%s` (%s).", path, error)
        return {"terms": 0, "entries": 0, "path": str(path), "saved": False}
    total = sum(len(entries) for entries in data.values())
    logger.info("wrote %s term(s) and %s entry(s) to %s", len(data), total, path)
    return {"terms": len(data), "entries": total, "path": str(path), "saved": True}


def _offered(entries, declined, report) -> list[str]:
    """One published term's entries with the ones removed by hand left out.

    Args:
        entries: The published entries.
        declined: The entries removed by hand.
        report: Refresh counts, whose ``entries_declined`` is added to.

    Returns:
        The entries that may be stored.
    """
    refused = set(declined)
    offered = [entry for entry in entries if entry not in refused]
    report["entries_declined"] += len(entries) - len(offered)
    return offered


def nsp_parse(text, seed=0, noodle_key="__", nspterminology=None, pantry_path=None):
    """Replace each noodle in ``text`` with a random pantry entry.

    Args:
        text: The prompt to parse.
        seed: Seed for the shared :mod:`random` module. A seed of exactly ``0`` leaves the
            module's existing state alone, so the parse is not reproducible; any other
            value, positive or negative, seeds it.
        noodle_key: Delimiter placed either side of a terminology name.
        nspterminology: An already-loaded ``{term: [entry, ...]}`` mapping. When supplied,
            nothing is read from the database and nothing is downloaded.
        pantry_path: A pantry JSON file to draw from instead of the database.

    Returns:
        The parsed text. A noodle naming a term the pantry does not have is left as it is.

    Raises:
        ValueError: The database holds no pantry and the copy shipped with the pack could
            not be read.
        OSError: ``pantry_path`` could not be read.
        JSONDecodeError: ``pantry_path`` is not valid JSON.
    """
    if nspterminology is None and pantry_path is not None:
        with open(pantry_path, "r", encoding="utf-8") as handle:
            nspterminology = json.load(handle)

    if nspterminology is not None:
        counts = {term: len(entries) for term, entries in nspterminology.items()}
        return _parse(text, seed, noodle_key, counts, nspterminology)

    ensure_pantry()
    return _parse(text, seed, noodle_key, terms(), None)


def _parse(text, seed, noodle_key, counts, loaded):
    """Substitute every noodle, drawing from a loaded mapping or from the database.

    Args:
        text: The prompt to parse.
        seed: Seed for the shared :mod:`random` module, advanced by one per substitution.
        noodle_key: Delimiter placed either side of a terminology name.
        counts: ``{term: entries}`` in the order the terms are visited.
        loaded: ``{term: [entry, ...]}`` to draw from, or None to draw from the database.

    Returns:
        The parsed text.
    """
    if seed > 0 or seed < 0:
        random.seed(seed)

    database = None if loaded is not None else _database()
    new_text = text
    for term, total in counts.items():
        tkey = f"{noodle_key}{term}{noodle_key}"
        tcount = new_text.count(tkey)
        for _ in range(tcount):
            if not total:
                entry = None
            elif loaded is not None:
                entry = loaded[term][random.randrange(total)]
            else:
                entry = database.record_name_by_position(
                    store_module.NSP, term, random.randrange(total)
                )
            if entry is not None:
                new_text = new_text.replace(tkey, entry, 1)
            seed += 1
            random.seed(seed)

    return new_text

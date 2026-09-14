"""SQLite-backed state for the pack.

One file in the config directory holds every store. ``kv`` carries key/value categories,
``records`` carries ordered named records, both keyed by a store name. Values and record
bodies are JSON text.
"""

from __future__ import annotations

import contextlib
import json
import random
import sqlite3
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

from .. import log

__all__ = [
    "DB_FILE",
    "DEFAULT_CATEGORY",
    "HISTORY",
    "NSP",
    "SCHEMA_VERSION",
    "SETTINGS",
    "STORES",
    "STYLES",
    "Record",
    "StateStore",
    "close_shared_store",
    "drop_record_category",
    "open_store",
    "shared_store",
    "write_category",
    "write_kv",
    "write_meta",
    "write_record_category",
    "write_records",
]

logger = log.get_logger("state.store")

#: The database file, in the pack's config directory.
DB_FILE = "was_state.db"

#: Version of the table layout this module creates and reads.
SCHEMA_VERSION = 1

#: Settings, custom tokens and node cursors.
SETTINGS = "settings"

#: The history lists behind the history combos.
HISTORY = "history"

#: The prompt style library.
STYLES = "styles"

#: Noodle Soup Prompts terminology.
NSP = "nsp"

#: Every store name.
STORES = (SETTINGS, HISTORY, STYLES, NSP)

#: One statement per table a store writes to, each naming its table outright.
_HOLDS_ANYTHING = (
    "SELECT 1 FROM kv WHERE store = ? LIMIT 1",
    "SELECT 1 FROM kv_category WHERE store = ? LIMIT 1",
    "SELECT 1 FROM records WHERE store = ? LIMIT 1",
    "SELECT 1 FROM record_category WHERE store = ? LIMIT 1",
)

#: The category a record store uses when its records form one flat list.
DEFAULT_CATEGORY = ""

#: How long sqlite waits on another connection's write lock, in milliseconds.
BUSY_TIMEOUT_MS = 10_000

#: How long a statement waits for the lock while the file is being opened, in milliseconds.
OPEN_TIMEOUT_MS = 1_000

#: How many times the file is opened again after a lock refused the first attempt.
OPEN_ATTEMPTS = 3

#: How many attempts a statement gets after sqlite reports the database locked.
RETRIES = 8

#: The first pause between attempts, in seconds.
RETRY_DELAY = 0.02

#: The longest pause between attempts, in seconds.
RETRY_CEILING = 0.5

#: How many index rows sqlite samples while gathering query statistics.
ANALYSIS_LIMIT = 400

#: sqlite messages that mean another connection holds the lock.
LOCKED = ("database is locked", "database table is locked", "database is busy")

#: Sort key standing in for a category with no row in its category table.
LAST = 2_147_483_647

#: The tables, created on first connection and left alone afterwards.
SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS meta (
        key   TEXT NOT NULL PRIMARY KEY,
        value TEXT NOT NULL
    ) WITHOUT ROWID
    """,
    """
    CREATE TABLE IF NOT EXISTS kv_category (
        store    TEXT NOT NULL,
        category TEXT NOT NULL,
        ordinal  INTEGER NOT NULL,
        PRIMARY KEY (store, category)
    ) WITHOUT ROWID
    """,
    """
    CREATE TABLE IF NOT EXISTS kv (
        store    TEXT NOT NULL,
        category TEXT NOT NULL,
        key      TEXT NOT NULL,
        ordinal  INTEGER NOT NULL,
        value    TEXT NOT NULL,
        PRIMARY KEY (store, category, key)
    ) WITHOUT ROWID
    """,
    """
    CREATE TABLE IF NOT EXISTS record_category (
        store    TEXT NOT NULL,
        category TEXT NOT NULL,
        ordinal  INTEGER NOT NULL,
        PRIMARY KEY (store, category)
    ) WITHOUT ROWID
    """,
    """
    CREATE TABLE IF NOT EXISTS records (
        store    TEXT NOT NULL,
        category TEXT NOT NULL,
        position INTEGER NOT NULL,
        name     TEXT NOT NULL,
        fields   TEXT,
        PRIMARY KEY (store, category, position)
    ) WITHOUT ROWID
    """,
    "CREATE INDEX IF NOT EXISTS records_by_name ON records (store, category, name)",
)

_KV_ROWS = """
    SELECT kv.category, kv.key, kv.value
    FROM kv LEFT JOIN kv_category AS c
        ON c.store = kv.store AND c.category = kv.category
    WHERE kv.store = ?
    ORDER BY COALESCE(c.ordinal, ?), kv.category, kv.ordinal
"""

_KV_SET = """
    INSERT INTO kv (store, category, key, ordinal, value)
    VALUES (
        ?, ?, ?,
        COALESCE((SELECT MAX(ordinal) + 1 FROM kv WHERE store = ? AND category = ?), 0),
        ?
    )
    ON CONFLICT (store, category, key) DO UPDATE SET value = excluded.value
"""

_KV_ADD_CATEGORY = """
    INSERT OR IGNORE INTO kv_category (store, category, ordinal)
    VALUES (
        ?, ?,
        COALESCE((SELECT MAX(ordinal) + 1 FROM kv_category WHERE store = ?), 0)
    )
"""

_RECORD_ROWS = """
    SELECT r.category, r.name, r.position, r.fields
    FROM records AS r LEFT JOIN record_category AS c
        ON c.store = r.store AND c.category = r.category
    WHERE r.store = ?
    ORDER BY COALESCE(c.ordinal, ?), r.category, r.position
"""

_RECORD_MATCHES = """
    SELECT r.category, r.name, r.position, r.fields
    FROM records AS r LEFT JOIN record_category AS c
        ON c.store = r.store AND c.category = r.category
    WHERE r.store = ? AND r.name LIKE ? ESCAPE '\\'
    ORDER BY COALESCE(c.ordinal, ?), r.category, r.position
    LIMIT ? OFFSET ?
"""

_RECORD_MATCH_COUNT = """
    SELECT COUNT(*) FROM (
        SELECT 1 FROM records
        WHERE store = ? AND name LIKE ? ESCAPE '\\'
        LIMIT ?
    )
"""

_RECORD_ADD_CATEGORY = """
    INSERT OR IGNORE INTO record_category (store, category, ordinal)
    VALUES (
        ?, ?,
        COALESCE((SELECT MAX(ordinal) + 1 FROM record_category WHERE store = ?), 0)
    )
"""

_RECORD_APPEND = """
    INSERT INTO records (store, category, position, name, fields)
    VALUES (
        ?, ?,
        COALESCE(
            (SELECT MAX(position) + 1 FROM records WHERE store = ? AND category = ?), 0
        ),
        ?, ?
    )
"""


def write_kv(
    connection: sqlite3.Connection, store: str, data: Mapping[str, Mapping[str, Any]]
) -> tuple[int, int]:
    """Replace every category and key of one store, on an already open transaction.

    Args:
        connection: A connection inside a write transaction.
        store: One of :data:`STORES`.
        data: ``{category: {key: value}}``. Category and key order are kept.

    Returns:
        ``(categories written, keys written)``.
    """
    rows = []
    categories = []
    for ordinal, (category, entries) in enumerate(data.items()):
        categories.append((store, category, ordinal))
        for index, (key, value) in enumerate(entries.items()):
            rows.append((store, category, key, index, json.dumps(value)))
    connection.execute("DELETE FROM kv WHERE store = ?", (store,))
    connection.execute("DELETE FROM kv_category WHERE store = ?", (store,))
    connection.executemany(
        "INSERT INTO kv_category (store, category, ordinal) VALUES (?, ?, ?)", categories
    )
    connection.executemany(
        "INSERT INTO kv (store, category, key, ordinal, value) VALUES (?, ?, ?, ?, ?)", rows
    )
    return len(categories), len(rows)


def write_records(
    connection: sqlite3.Connection,
    store: str,
    data: Mapping[str, Iterable[tuple[str, Mapping[str, Any] | None]]],
) -> tuple[int, int]:
    """Replace every record of one store, on an already open transaction.

    Args:
        connection: A connection inside a write transaction.
        store: One of :data:`STORES`.
        data: ``{category: [(name, fields), ...]}``. Order is kept and a repeated name
            stays a separate record.

    Returns:
        ``(categories written, records written)``.
    """
    rows = []
    categories = []
    for ordinal, (category, entries) in enumerate(data.items()):
        categories.append((store, category, ordinal))
        for position, (name, fields) in enumerate(entries):
            body = None if fields is None else json.dumps(dict(fields))
            rows.append((store, category, position, name, body))
    connection.execute("DELETE FROM records WHERE store = ?", (store,))
    connection.execute("DELETE FROM record_category WHERE store = ?", (store,))
    connection.executemany(
        "INSERT INTO record_category (store, category, ordinal) VALUES (?, ?, ?)", categories
    )
    connection.executemany(
        "INSERT INTO records (store, category, position, name, fields) VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    return len(categories), len(rows)


def write_category(
    connection: sqlite3.Connection,
    store: str,
    category: str,
    entries: Iterable[tuple[str, Mapping[str, Any] | None]],
) -> int:
    """Replace every record of one category, on an already open transaction.

    Args:
        connection: A connection inside a write transaction.
        store: One of :data:`STORES`.
        category: Category name, created when it is not there.
        entries: ``(name, fields)`` pairs, stored in the order given with positions
            running from 0.

    Returns:
        How many records were written.
    """
    rows = [
        (store, category, position, name, None if fields is None else json.dumps(dict(fields)))
        for position, (name, fields) in enumerate(entries)
    ]
    connection.execute(_RECORD_ADD_CATEGORY, (store, category, store))
    connection.execute(
        "DELETE FROM records WHERE store = ? AND category = ?", (store, category)
    )
    connection.executemany(
        "INSERT INTO records (store, category, position, name, fields) VALUES (?, ?, ?, ?, ?)",
        rows,
    )
    return len(rows)


def write_record_category(connection: sqlite3.Connection, store: str, category: str) -> None:
    """Create an empty record category, on an already open transaction.

    Args:
        connection: A connection inside a write transaction.
        store: One of :data:`STORES`.
        category: Category name. One that already exists is left as it is.
    """
    connection.execute(_RECORD_ADD_CATEGORY, (store, category, store))


def drop_record_category(connection: sqlite3.Connection, store: str, category: str) -> None:
    """Remove a record category and every record in it, on an open transaction.

    Args:
        connection: A connection inside a write transaction.
        store: One of :data:`STORES`.
        category: Category name.
    """
    connection.execute(
        "DELETE FROM records WHERE store = ? AND category = ?", (store, category)
    )
    connection.execute(
        "DELETE FROM record_category WHERE store = ? AND category = ?", (store, category)
    )


def write_meta(connection: sqlite3.Connection, key: str, value: str) -> None:
    """Store one meta value, on an already open transaction.

    Args:
        connection: A connection inside a write transaction.
        key: Meta table key.
        value: Text to store under it, replacing any earlier value.
    """
    connection.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


class Record(NamedTuple):
    """One row of the record table.

    Attributes:
        category: The group the record belongs to, ``""`` for a flat store.
        name: The record's label, unique per category only where the caller keeps it so.
        position: Sort order inside the category, ascending.
        fields: The decoded body, or ``None`` for a record that is only a name.
    """

    category: str
    name: str
    position: int
    fields: dict | None


def _like_escape(text: str) -> str:
    """One search term with the ``LIKE`` metacharacters turned into literals.

    Args:
        text: The term as typed.

    Returns:
        The same term with ``\\``, ``%`` and ``_`` escaped for ``LIKE ... ESCAPE '\\'``.
    """
    for character in ("\\", "%", "_"):
        text = text.replace(character, f"\\{character}")
    return text


def _locked(error: BaseException) -> bool:
    """Whether a sqlite error means another connection holds the lock."""
    text = str(error).lower()
    return any(message in text for message in LOCKED)


def _rollback(connection: sqlite3.Connection) -> None:
    """End an open transaction, discarding it. A closed transaction is left alone."""
    try:
        connection.execute("ROLLBACK")
    except sqlite3.Error:
        pass


def _pause(delay: float) -> float:
    """Wait out one locked attempt and answer how long the next one waits."""
    time.sleep(delay * (1.0 + random.random()))
    return min(delay * 2.0, RETRY_CEILING)


def _optimize(connection: sqlite3.Connection, path) -> None:
    """Gather query statistics for any table that has grown out of step with them.

    Args:
        connection: A connection outside a transaction.
        path: The database file, named in the log line when the statistics are refused.
    """
    try:
        connection.execute(f"PRAGMA analysis_limit = {ANALYSIS_LIMIT}")
        connection.execute("PRAGMA optimize")
    except sqlite3.Error as error:
        logger.debug("the query statistics of %s were not updated (%s)", path, error)


def _has_schema(connection: sqlite3.Connection) -> bool:
    """Whether the tables have already been created in this database."""
    row = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'meta'"
    ).fetchone()
    return row is not None


class StateStore:
    """A sqlite database holding the pack's stores.

    Attributes:
        path: The database file.
        connections: Every connection opened against it, one per thread.
    """

    def __init__(self, path, busy_timeout_ms: int = BUSY_TIMEOUT_MS, retries: int = RETRIES):
        """Open a store. The file and its tables are created on the first statement.

        Args:
            path: Where the database file lives.
            busy_timeout_ms: How long sqlite waits on another connection's write lock.
            retries: Attempts a statement gets after sqlite reports the database locked.
        """
        self.path = Path(path)
        self.connections: list[sqlite3.Connection] = []
        self._busy_timeout_ms = int(busy_timeout_ms)
        self._retries = max(1, int(retries))
        self._local = threading.local()
        self._guard = threading.RLock()
        self._memory: sqlite3.Connection | None = None
        self._ready = False
        self._read_only = False

    @property
    def in_memory(self) -> bool:
        """Whether the file could not be opened and this session is held in memory."""
        return self._memory is not None

    @property
    def read_only(self) -> bool:
        """Whether the file can be read but not written, so every write is refused."""
        return self._read_only

    def close(self) -> None:
        """Close every connection. A later call reopens on demand."""
        with self._guard:
            for connection in self.connections:
                try:
                    connection.close()
                except sqlite3.Error:
                    pass
            self.connections = []
            if self._memory is not None:
                try:
                    self._memory.close()
                except sqlite3.Error:
                    pass
                self._memory = None
            self._local = threading.local()
            self._ready = False
            self._read_only = False

    def write(self, work: Callable[[sqlite3.Connection], Any], label: str) -> bool:
        """Run ``work`` inside one transaction, retrying while the database is locked.

        Args:
            work: Called with the connection between ``BEGIN IMMEDIATE`` and ``COMMIT``.
            label: What the transaction is, named in the log line when it fails.

        Returns:
            True when the transaction committed. A failure is logged and never raised, so
            False means the database is exactly as it was.
        """
        delay = RETRY_DELAY
        for attempt in range(self._retries):
            try:
                with self._session() as connection:
                    # End a transaction an earlier failure left open.
                    if connection.in_transaction:
                        _rollback(connection)
                    connection.execute("BEGIN IMMEDIATE")
                    try:
                        work(connection)
                        connection.execute("COMMIT")
                    except BaseException:
                        _rollback(connection)
                        raise
                    return True
            except sqlite3.OperationalError as error:
                if _locked(error) and attempt < self._retries - 1:
                    delay = _pause(delay)
                    continue
                if "readonly" in str(error).lower():
                    self._read_only = True
                logger.error("%s could not be saved (%s), so nothing was written", label, error)
                return False
            except (sqlite3.Error, TypeError, ValueError) as error:
                logger.error("%s could not be saved (%s), so nothing was written", label, error)
                return False
        return False

    def read(self, work: Callable[[sqlite3.Connection], Any], label: str, default: Any = None) -> Any:
        """Run ``work`` against the database, retrying while it is locked.

        Args:
            work: Called with the connection. Statements outside a transaction see the
                last committed state.
            label: What is being read, named in the log line when it fails.
            default: Returned when the read fails.

        Returns:
            Whatever ``work`` returned, or ``default``. A failure is logged, never raised.
        """
        delay = RETRY_DELAY
        for attempt in range(self._retries):
            try:
                with self._session() as connection:
                    return work(connection)
            except sqlite3.OperationalError as error:
                if _locked(error) and attempt < self._retries - 1:
                    delay = _pause(delay)
                    continue
                logger.error("%s could not be read (%s)", label, error)
                return default
            except (sqlite3.Error, TypeError, ValueError) as error:
                logger.error("%s could not be read (%s)", label, error)
                return default
        return default

    # Connections

    @contextlib.contextmanager
    def _session(self):
        """Yield this thread's connection, serialising callers while held in memory."""
        connection = self._handle()
        if connection is self._memory:
            with self._guard:
                yield connection
        else:
            yield connection

    def _handle(self) -> sqlite3.Connection:
        """This thread's connection, opened and prepared on first use."""
        existing = getattr(self._local, "connection", None)
        if existing is not None:
            return existing
        if self._memory is not None:
            return self._memory
        connection = None
        failure: BaseException | None = None
        delay = RETRY_DELAY
        for attempt in range(OPEN_ATTEMPTS):
            try:
                connection = self._open()
                break
            except (sqlite3.Error, OSError) as error:
                failure = error
                if attempt < OPEN_ATTEMPTS - 1:
                    delay = _pause(delay)
        if connection is None:
            connection = self._open_read_only(failure)
            if connection is None:
                return self._fall_back(failure)
        self._local.connection = connection
        with self._guard:
            self.connections.append(connection)
        return connection

    def _open(self) -> sqlite3.Connection:
        """Connect to the file, apply the pragmas and create the tables once."""
        connection = sqlite3.connect(
            self.path,
            timeout=OPEN_TIMEOUT_MS / 1000.0,
            isolation_level=None,
        )
        try:
            connection.execute(f"PRAGMA busy_timeout = {OPEN_TIMEOUT_MS}")
            # The first statement that needs a lock on the file.
            present = _has_schema(connection)
            self._pragmas(connection)
            with self._guard:
                if not self._ready:
                    if not present and not _has_schema(connection):
                        self._create(connection)
                    self._ready = True
            connection.execute(f"PRAGMA busy_timeout = {self._busy_timeout_ms}")
        except (sqlite3.Error, OSError):
            connection.close()
            raise
        _optimize(connection, self.path)
        return connection

    def optimize(self) -> None:
        """Bring the query statistics up to date with what the tables now hold."""
        connection = self._handle()
        _optimize(connection, self.path)

    def _open_read_only(self, error: BaseException) -> sqlite3.Connection | None:
        """Connect to an existing file that cannot be written, for reading only.

        Args:
            error: What the writable open failed with, named in the warning.

        Returns:
            A connection every write is refused on, or ``None`` when even a read is
            impossible.
        """
        if not self.path.is_file():
            return None
        for parameter in ("mode=ro", "immutable=1"):
            try:
                connection = sqlite3.connect(
                    f"{self.path.as_uri()}?{parameter}",
                    uri=True,
                    timeout=OPEN_TIMEOUT_MS / 1000.0,
                    isolation_level=None,
                )
            except sqlite3.Error:
                continue
            try:
                if _has_schema(connection):
                    with self._guard:
                        if not self._read_only:
                            self._read_only = True
                            logger.warning(
                                "%s can be read but not written (%s), so settings, history "
                                "and styles changed in this session will be lost when "
                                "ComfyUI closes. Check the file's permissions.",
                                self.path,
                                error,
                            )
                    return connection
            except sqlite3.Error:
                pass
            connection.close()
        return None

    def _pragmas(self, connection: sqlite3.Connection) -> None:
        """Set the durability level and the journal mode."""
        try:
            connection.execute("PRAGMA synchronous = FULL")
        except sqlite3.Error as error:
            logger.debug("the durability level of %s was refused (%s)", self.path, error)
        delay = RETRY_DELAY
        for attempt in range(self._retries):
            try:
                connection.execute("PRAGMA journal_mode = WAL").fetchall()
                return
            except sqlite3.Error as error:
                if _locked(error) and attempt < self._retries - 1:
                    delay = _pause(delay)
                    continue
                logger.debug("%s could not be put in WAL mode (%s)", self.path, error)
                return

    def _create(self, connection: sqlite3.Connection) -> None:
        """Create the tables and record the schema version, retrying while locked."""
        delay = RETRY_DELAY
        for attempt in range(self._retries):
            try:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    for statement in SCHEMA:
                        connection.execute(statement)
                    connection.execute(
                        "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
                        (str(SCHEMA_VERSION),),
                    )
                except BaseException:
                    _rollback(connection)
                    raise
                connection.execute("COMMIT")
                return
            except sqlite3.OperationalError as error:
                if _locked(error) and attempt < self._retries - 1:
                    delay = _pause(delay)
                    continue
                raise

    def _fall_back(self, error: BaseException) -> sqlite3.Connection:
        """Keep the session in memory after the file could not be opened."""
        with self._guard:
            if self._memory is None:
                logger.warning(
                    "the state database at %s could not be opened (%s), so this session's "
                    "settings, history and styles are kept in memory and nothing is saved. "
                    "Check that the folder exists, that it can be written to, and that it "
                    "is not on a share or filesystem that refuses file locks.",
                    self.path,
                    error,
                )
                memory = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
                for statement in SCHEMA:
                    memory.execute(statement)
                memory.execute(
                    "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
                self._memory = memory
            return self._memory

    # Bookkeeping

    def meta(self, key: str, default: str | None = None) -> str | None:
        """The value stored in the meta table under ``key``, or ``default``."""

        def work(connection):
            row = connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
            return default if row is None else row[0]

        return self.read(work, f"meta key {key!r}", default)

    def set_meta(self, key: str, value: str) -> bool:
        """Store ``value`` in the meta table under ``key``, replacing any earlier value."""

        def work(connection):
            connection.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

        return self.write(work, f"meta key {key!r}")

    def is_empty(self, store: str) -> bool | None:
        """Whether ``store`` holds no categories, keys or records.

        Args:
            store: One of :data:`STORES`.

        Returns:
            True when nothing is stored under that name, False when something is, and
            ``None`` when the database could not be read.
        """

        def work(connection):
            for statement in _HOLDS_ANYTHING:
                if connection.execute(statement, (store,)).fetchone() is not None:
                    return False
            return True

        return self.read(work, f"the {store} store", None)

    # Key and value stores

    def get(self, store: str, category: str, key: str, default: Any = None) -> Any:
        """The value at ``store``/``category``/``key``, or ``default`` when absent."""

        def work(connection):
            row = connection.execute(
                "SELECT value FROM kv WHERE store = ? AND category = ? AND key = ?",
                (store, category, key),
            ).fetchone()
            return default if row is None else json.loads(row[0])

        return self.read(work, f"{store}/{category}/{key}", default)

    def set(self, store: str, category: str, key: str, value: Any) -> bool:
        """Store ``value``, creating the category when it does not exist.

        Args:
            store: One of :data:`STORES`.
            category: Category name.
            key: Key name.
            value: Anything ``json.dumps`` accepts.

        Returns:
            True when the write committed.
        """

        def work(connection):
            connection.execute(_KV_ADD_CATEGORY, (store, category, store))
            connection.execute(
                _KV_SET, (store, category, key, store, category, json.dumps(value))
            )

        return self.write(work, f"{store}/{category}/{key}")

    def set_many(self, store: str, category: str, values: Mapping[str, Any]) -> bool:
        """Store several keys of one category in a single transaction.

        Args:
            store: One of :data:`STORES`.
            category: Category name, created when it does not exist.
            values: ``{key: value}``, each value anything ``json.dumps`` accepts.

        Returns:
            True when the write committed. Keys not named are left as they are.
        """

        def work(connection):
            connection.execute(_KV_ADD_CATEGORY, (store, category, store))
            for key, value in values.items():
                connection.execute(
                    _KV_SET, (store, category, key, store, category, json.dumps(value))
                )

        return self.write(work, f"{len(values)} key(s) of {store}/{category}")

    def update(self, store: str, category: str, key: str, value: Any) -> bool:
        """Overwrite a key that is already stored, leaving an absent one absent.

        Args:
            store: One of :data:`STORES`.
            category: Category name.
            key: Key name.
            value: Anything ``json.dumps`` accepts.

        Returns:
            True when the write committed, whether or not the key was there.
        """

        def work(connection):
            connection.execute(
                "UPDATE kv SET value = ? WHERE store = ? AND category = ? AND key = ?",
                (json.dumps(value), store, category, key),
            )

        return self.write(work, f"{store}/{category}/{key}")

    def delete(self, store: str, category: str, key: str) -> bool:
        """Remove one key. A key that is not there counts as removed."""

        def work(connection):
            connection.execute(
                "DELETE FROM kv WHERE store = ? AND category = ? AND key = ?",
                (store, category, key),
            )

        return self.write(work, f"deleting {store}/{category}/{key}")

    def has_category(self, store: str, category: str) -> bool:
        """Whether ``category`` exists in ``store``, empty or not."""

        def work(connection):
            row = connection.execute(
                "SELECT 1 FROM kv_category WHERE store = ? AND category = ?",
                (store, category),
            ).fetchone()
            return row is not None

        return bool(self.read(work, f"{store}/{category}", False))

    def has_key(self, store: str, category: str, key: str) -> bool:
        """Whether ``key`` exists inside ``category``."""

        def work(connection):
            row = connection.execute(
                "SELECT 1 FROM kv WHERE store = ? AND category = ? AND key = ?",
                (store, category, key),
            ).fetchone()
            return row is not None

        return bool(self.read(work, f"{store}/{category}/{key}", False))

    def add_category(self, store: str, category: str) -> bool:
        """Create an empty category. An existing one is left as it is."""

        def work(connection):
            connection.execute(_KV_ADD_CATEGORY, (store, category, store))

        return self.write(work, f"the category {store}/{category}")

    def delete_category(self, store: str, category: str) -> bool:
        """Remove a category and every key in it."""

        def work(connection):
            connection.execute(
                "DELETE FROM kv WHERE store = ? AND category = ?", (store, category)
            )
            connection.execute(
                "DELETE FROM kv_category WHERE store = ? AND category = ?", (store, category)
            )

        return self.write(work, f"deleting the category {store}/{category}")

    def categories(self, store: str) -> list[str]:
        """Every category in ``store``, in the order they were created."""

        def work(connection):
            return [
                row[0]
                for row in connection.execute(
                    "SELECT category FROM kv_category WHERE store = ? ORDER BY ordinal",
                    (store,),
                )
            ]

        return self.read(work, f"the categories of {store}", []) or []

    def category(self, store: str, category: str) -> dict[str, Any]:
        """Every key and value of one category, in the order the keys were added."""

        def work(connection):
            return {
                row[0]: json.loads(row[1])
                for row in connection.execute(
                    "SELECT key, value FROM kv WHERE store = ? AND category = ? ORDER BY ordinal",
                    (store, category),
                )
            }

        return self.read(work, f"{store}/{category}", {}) or {}

    def dump(self, store: str) -> dict[str, dict[str, Any]]:
        """The whole store as ``{category: {key: value}}``, empty categories included."""

        def work(connection):
            data: dict[str, dict[str, Any]] = {
                row[0]: {}
                for row in connection.execute(
                    "SELECT category FROM kv_category WHERE store = ? ORDER BY ordinal",
                    (store,),
                )
            }
            for category, key, value in connection.execute(_KV_ROWS, (store, LAST)):
                data.setdefault(category, {})[key] = json.loads(value)
            return data

        return self.read(work, f"the {store} store", {}) or {}

    def replace(self, store: str, data: Mapping[str, Mapping[str, Any]]) -> bool:
        """Replace every category and key of ``store`` in one transaction.

        Args:
            store: One of :data:`STORES`.
            data: ``{category: {key: value}}``. Order is kept.

        Returns:
            True when the write committed.
        """

        def work(connection):
            write_kv(connection, store, data)

        return self.write(work, f"the {store} store")

    # Ordered record stores

    def record_categories(self, store: str) -> list[str]:
        """Every record category in ``store``, in the order they were created."""

        def work(connection):
            return [
                row[0]
                for row in connection.execute(
                    "SELECT category FROM record_category WHERE store = ? ORDER BY ordinal",
                    (store,),
                )
            ]

        return self.read(work, f"the categories of {store}", []) or []

    def records(self, store: str, category: str | None = None) -> list[Record]:
        """Every record, or every record of one category, in order.

        Args:
            store: One of :data:`STORES`.
            category: One category, or ``None`` for all of them.

        Returns:
            :class:`Record` values ordered by category then position.
        """

        def work(connection):
            if category is None:
                rows = connection.execute(_RECORD_ROWS, (store, LAST))
            else:
                rows = connection.execute(
                    "SELECT category, name, position, fields FROM records "
                    "WHERE store = ? AND category = ? ORDER BY position",
                    (store, category),
                )
            return [
                Record(row[0], row[1], row[2], None if row[3] is None else json.loads(row[3]))
                for row in rows
            ]

        return self.read(work, f"the {store} store", []) or []

    def record_names(self, store: str, category: str = DEFAULT_CATEGORY) -> list[str]:
        """Every record name in one category, in order, repeats included."""

        def work(connection):
            return [
                row[0]
                for row in connection.execute(
                    "SELECT name FROM records WHERE store = ? AND category = ? ORDER BY position",
                    (store, category),
                )
            ]

        return self.read(work, f"{store}/{category}", []) or []

    def newest_record_names(
        self, store: str, category: str, limit: int, offset: int = 0
    ) -> list[str]:
        """The last records of a category, newest first, repeats included.

        Args:
            store: One of :data:`STORES`.
            category: Category name.
            limit: At most this many names.
            offset: How many of the newest to pass over first.

        Returns:
            Names in descending position order.
        """

        def work(connection):
            return [
                row[0]
                for row in connection.execute(
                    "SELECT name FROM records WHERE store = ? AND category = ? "
                    "ORDER BY position DESC LIMIT ? OFFSET ?",
                    (store, category, int(limit), int(offset)),
                )
            ]

        return self.read(work, f"{store}/{category}", []) or []

    def record_page(
        self, store: str, category: str, start: int = 0, limit: int = 500
    ) -> list[Record]:
        """A window of one category's records, in order.

        Args:
            store: One of :data:`STORES`.
            category: Category name.
            start: How many records to pass over first.
            limit: At most this many records.

        Returns:
            :class:`Record` values ordered by position. Empty for a start past the end.
        """

        def work(connection):
            return [
                Record(row[0], row[1], row[2], None if row[3] is None else json.loads(row[3]))
                for row in connection.execute(
                    "SELECT category, name, position, fields FROM records "
                    "WHERE store = ? AND category = ? ORDER BY position LIMIT ? OFFSET ?",
                    (store, category, max(0, int(limit)), max(0, int(start))),
                )
            ]

        return self.read(work, f"{store}/{category}", []) or []

    def record_matches(
        self, store: str, needle: str, start: int = 0, limit: int = 500, ceiling: int = 10_000
    ) -> tuple[list[Record], int]:
        """The records of a store whose name holds ``needle``, and how many there are.

        Args:
            store: One of :data:`STORES`.
            needle: Text matched anywhere in a record name, case-insensitively for ASCII.
            start: How many matches to pass over first.
            limit: At most this many matches returned.
            ceiling: How many matches are counted at all. The count stops here.

        Returns:
            ``(records, total)``, ordered by category then position. ``total`` is the match
            count, or ``ceiling`` when there are at least that many.
        """
        pattern = f"%{_like_escape(str(needle))}%"

        def work(connection):
            rows = [
                Record(row[0], row[1], row[2], None if row[3] is None else json.loads(row[3]))
                for row in connection.execute(
                    _RECORD_MATCHES,
                    (store, pattern, LAST, max(0, int(limit)), max(0, int(start))),
                )
            ]
            counted = connection.execute(
                _RECORD_MATCH_COUNT, (store, pattern, max(0, int(ceiling)))
            ).fetchone()
            return rows, 0 if counted is None else int(counted[0])

        return self.read(work, f"the {store} store", ([], 0)) or ([], 0)

    def record_counts(self, store: str) -> dict[str, int]:
        """How many records each category holds, in the order the categories were created.

        Args:
            store: One of :data:`STORES`.

        Returns:
            ``{category: records}``, a category holding none included at 0.
        """

        def work(connection):
            counts = {
                row[0]: 0
                for row in connection.execute(
                    "SELECT category FROM record_category WHERE store = ? ORDER BY ordinal",
                    (store,),
                )
            }
            for category, total in connection.execute(
                "SELECT category, COUNT(*) FROM records WHERE store = ? GROUP BY category",
                (store,),
            ):
                counts[category] = total
            return counts

        return self.read(work, f"the categories of {store}", {}) or {}

    def record_name_by_position(self, store: str, category: str, position: int) -> str | None:
        """The name of the record stored at one position of a category.

        Args:
            store: One of :data:`STORES`.
            category: Category name.
            position: The sort order column, which runs from 0 upward for a category
                written by :meth:`set_records` or :meth:`replace_records`.

        Returns:
            The name, or ``None`` when the category holds no record at that position.
        """

        def work(connection):
            row = connection.execute(
                "SELECT name FROM records WHERE store = ? AND category = ? AND position = ?",
                (store, category, position),
            ).fetchone()
            return None if row is None else row[0]

        return self.read(work, f"{store}/{category}[{position}]", None)

    def find_record(self, store: str, category: str, name: str) -> Record | None:
        """The first record in ``category`` carrying ``name``, or ``None``."""

        def work(connection):
            row = connection.execute(
                "SELECT category, name, position, fields FROM records "
                "WHERE store = ? AND category = ? AND name = ? ORDER BY position LIMIT 1",
                (store, category, name),
            ).fetchone()
            if row is None:
                return None
            return Record(row[0], row[1], row[2], None if row[3] is None else json.loads(row[3]))

        return self.read(work, f"{store}/{category}/{name}", None)

    def add_record_category(self, store: str, category: str) -> bool:
        """Create an empty record category. An existing one is left as it is."""

        def work(connection):
            connection.execute(_RECORD_ADD_CATEGORY, (store, category, store))

        return self.write(work, f"the category {store}/{category}")

    def delete_record_category(self, store: str, category: str) -> bool:
        """Remove a record category and every record in it."""

        def work(connection):
            connection.execute(
                "DELETE FROM records WHERE store = ? AND category = ?", (store, category)
            )
            connection.execute(
                "DELETE FROM record_category WHERE store = ? AND category = ?", (store, category)
            )

        return self.write(work, f"deleting the category {store}/{category}")

    def append_record(
        self, store: str, category: str, name: str, fields: Mapping[str, Any] | None = None
    ) -> bool:
        """Add a record after the last one in its category, repeats allowed.

        Args:
            store: One of :data:`STORES`.
            category: Category name, created when it does not exist.
            name: The record's label.
            fields: The body, or ``None`` for a record that is only a name.

        Returns:
            True when the write committed.
        """

        def work(connection):
            body = None if fields is None else json.dumps(dict(fields))
            connection.execute(_RECORD_ADD_CATEGORY, (store, category, store))
            connection.execute(_RECORD_APPEND, (store, category, store, category, name, body))

        return self.write(work, f"{store}/{category}/{name}")

    def append_records(
        self,
        store: str,
        category: str,
        entries: Iterable[tuple[str, Mapping[str, Any] | None]],
        unique: bool = False,
    ) -> bool:
        """Add records after the last one in a category, all in one transaction.

        Args:
            store: One of :data:`STORES`.
            category: Category name, created when it does not exist.
            entries: ``(name, fields)`` pairs, appended in the order given.
            unique: Remove any record already carrying the name before appending it, so
                the category holds one of each name and the newest is last.

        Returns:
            True when the write committed.
        """
        pairs = list(entries)

        def work(connection):
            connection.execute(_RECORD_ADD_CATEGORY, (store, category, store))
            for name, fields in pairs:
                body = None if fields is None else json.dumps(dict(fields))
                if unique:
                    connection.execute(
                        "DELETE FROM records WHERE store = ? AND category = ? AND name = ?",
                        (store, category, name),
                    )
                connection.execute(
                    _RECORD_APPEND, (store, category, store, category, name, body)
                )

        return self.write(work, f"{len(pairs)} record(s) in {store}/{category}")

    def set_records(
        self,
        store: str,
        category: str,
        entries: Iterable[tuple[str, Mapping[str, Any] | None]],
    ) -> bool:
        """Replace every record of one category in a single transaction.

        Args:
            store: One of :data:`STORES`.
            category: Category name, created when it does not exist.
            entries: ``(name, fields)`` pairs, stored in the order given with positions
                running from 0.

        Returns:
            True when the write committed.
        """
        pairs = list(entries)

        def work(connection):
            write_category(connection, store, category, pairs)

        return self.write(work, f"{len(pairs)} record(s) in {store}/{category}")

    def set_record(
        self, store: str, category: str, name: str, fields: Mapping[str, Any] | None = None
    ) -> bool:
        """Store a record by name, appending it when the name is not already there.

        Args:
            store: One of :data:`STORES`.
            category: Category name, created when it does not exist.
            name: The record's label. Every record carrying it is given the new body.
            fields: The body, or ``None`` for a record that is only a name.

        Returns:
            True when the write committed.
        """

        def work(connection):
            body = None if fields is None else json.dumps(dict(fields))
            connection.execute(_RECORD_ADD_CATEGORY, (store, category, store))
            updated = connection.execute(
                "UPDATE records SET fields = ? WHERE store = ? AND category = ? AND name = ?",
                (body, store, category, name),
            ).rowcount
            if not updated:
                connection.execute(_RECORD_APPEND, (store, category, store, category, name, body))

        return self.write(work, f"{store}/{category}/{name}")

    def rename_record(self, store: str, category: str, name: str, new_name: str) -> bool:
        """Give every record in ``category`` carrying ``name`` the name ``new_name``."""

        def work(connection):
            connection.execute(
                "UPDATE records SET name = ? WHERE store = ? AND category = ? AND name = ?",
                (new_name, store, category, name),
            )

        return self.write(work, f"renaming {store}/{category}/{name}")

    def delete_record(self, store: str, category: str, name: str) -> bool:
        """Remove every record in ``category`` carrying ``name``."""

        def work(connection):
            connection.execute(
                "DELETE FROM records WHERE store = ? AND category = ? AND name = ?",
                (store, category, name),
            )

        return self.write(work, f"deleting {store}/{category}/{name}")

    def replace_records(
        self, store: str, data: Mapping[str, Iterable[tuple[str, Mapping[str, Any] | None]]]
    ) -> bool:
        """Replace every record of ``store`` in one transaction.

        Args:
            store: One of :data:`STORES`.
            data: ``{category: [(name, fields), ...]}``. Category order and record order
                are both kept, and a repeated name is kept as a separate record.

        Returns:
            True when the write committed.
        """

        def work(connection):
            write_records(connection, store, data)

        return self.write(work, f"the {store} store")

    def dump_records(self, store: str) -> dict[str, list[Record]]:
        """The whole record store as ``{category: [Record, ...]}``, empty ones included."""
        grouped: dict[str, list[Record]] = {name: [] for name in self.record_categories(store)}
        for record in self.records(store):
            grouped.setdefault(record.category, []).append(record)
        return grouped


_shared: StateStore | None = None
_shared_guard = threading.Lock()


def open_store(path) -> StateStore:
    """Open a store at a given path, without touching the shared one.

    Args:
        path: Where the database file lives.

    Returns:
        A store of its own, which the caller closes.
    """
    return StateStore(path)


def shared_store() -> StateStore:
    """The store every node uses, opened and migrated on first call.

    Returns:
        The process-wide store at ``<config dir>/was_state.db``.
    """
    global _shared
    with _shared_guard:
        if _shared is None:
            from .. import config
            from . import migration

            path = config.state_file(DB_FILE)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except OSError as error:
                logger.warning(
                    "%s could not be created (%s), so state is kept in memory for this session",
                    path.parent,
                    error,
                )
            store = StateStore(path)
            migration.import_pending(store, path.parent)
            _shared = store
    return _shared


def close_shared_store() -> None:
    """Close the shared store. The next call to :func:`shared_store` reopens it."""
    global _shared
    with _shared_guard:
        if _shared is not None:
            _shared.close()
            _shared = None

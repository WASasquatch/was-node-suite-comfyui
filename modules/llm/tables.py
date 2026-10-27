"""Tables for the LLM nodes: files read into an in-memory SQLite database, and questions asked of it.

A loaded database answers reads only, refusing writes, attaches and pragmas past schema reads,
and stops a query past its time.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import pathlib
import re
import sqlite3
import time
from dataclasses import dataclass

from .. import deps, log
from ..util import file_listing, sandbox
from . import FEATURE, runtime

__all__ = [
    "MAX_ROWS",
    "NO_TABLES",
    "TABLE_EXTENSIONS",
    "Tables",
    "file_options",
    "load_file",
    "picked",
    "query_engine",
    "row_documents",
    "sql_retriever",
    "table_name",
]

logger = log.get_logger("llm.tables")

#: Files a table loader reads.
TABLE_EXTENSIONS = (".csv", ".tsv", ".json", ".jsonl", ".xlsx", ".sqlite", ".sqlite3", ".db")

#: The file menu's entry while no table file is found, never a valid label.
NO_TABLES = "No table files"

#: Files the menu lists at most.
MAX_FILE_OPTIONS = 2000

#: Rows read from one table at most, unless the node asks for fewer.
MAX_ROWS = 1_000_000

#: Columns one table keeps at most.
MAX_COLUMNS = 512

#: Tables one database holds at most.
MAX_TABLES = 64

#: Characters of one cell kept.
MAX_CELL_CHARS = 100_000

#: Bytes of a JSON or JSON Lines table read at most.
MAX_JSON_BYTES = 64 * 1024 * 1024

#: Rows a query hands back at most, and characters of one value in them.
MAX_RESULT_ROWS = 10_000
RESULT_VALUE_CHARS = 300

#: Rows shown on a panel per table.
SAMPLE_ROWS = 5

#: SQLite virtual machine steps between two checks of a query's time.
PROGRESS_STEPS = 10_000

#: A table or column name as the database holds it.
_NAME = re.compile(r"[^a-z0-9_]+")

#: Pragmas a schema read runs, allowed once the database is read only.
_READ_PRAGMAS = frozenset({"table_info", "table_xinfo", "index_list", "index_info", "index_xinfo", "foreign_key_list"})

#: Settings a connection reads, allowed in their reading form only.
_SETTING_READS = frozenset({"read_uncommitted"})


def table_name(text: str, taken=()) -> str:
    """A table or column name: lower case letters, digits and underscores, not starting with a digit.

    Args:
        text: The name as written, such as ``Sales Q3.csv`` or ``Unit Price``.
        taken: Names already used, compared without case; a repeat gains ``_2``, ``_3``.

    Returns:
        The name, ``table`` where nothing usable is left.
    """
    base = _NAME.sub("_", str(text or "").strip().lower()).strip("_")[:60] or "table"
    if base[0].isdigit():
        base = f"t_{base}"
    used = {str(name).lower() for name in taken}
    name, number = base, 2
    while name in used:
        name = f"{base}_{number}"
        number += 1
    return name


@dataclass(eq=False)
class Tables:
    """Tables in one in-memory SQLite database, read only once loaded.

    Attributes:
        connection: The database.
        names: Its tables, in load order.
        sources: Table name to the label of the file it came from.
        rows: Table name to its row count.
    """

    connection: sqlite3.Connection
    names: tuple = ()
    sources: dict = None
    rows: dict = None

    def models(self) -> tuple:
        return ()

    def columns(self, name: str) -> list[tuple[str, str]]:
        """``(column, type)`` of one table."""
        cursor = self.connection.execute(f'SELECT * FROM "{name}" LIMIT 0')
        found = [item[0] for item in cursor.description or ()]
        kinds = {row[1]: row[2] for row in self.connection.execute(f'PRAGMA table_info("{name}")')}
        return [(column, kinds.get(column, "")) for column in found]

    def sample(self, name: str, count: int = SAMPLE_ROWS) -> list[tuple]:
        """The first rows of one table."""
        return list(self.connection.execute(f'SELECT * FROM "{name}" LIMIT ?', (int(count),)))


def _new_connection() -> sqlite3.Connection:
    """An empty in-memory database usable from any thread."""
    return sqlite3.connect(":memory:", check_same_thread=False)


def _copied(previous: Tables | None) -> sqlite3.Connection:
    """A new database holding a copy of another's tables."""
    connection = _new_connection()
    if previous is not None:
        previous.connection.backup(connection)
    return connection


#: Authorizer actions a read-only database allows.
_READ_ACTIONS = frozenset(
    getattr(sqlite3, name) for name in
    ("SQLITE_SELECT", "SQLITE_READ", "SQLITE_FUNCTION", "SQLITE_RECURSIVE", "SQLITE_TRANSACTION", "SQLITE_SAVEPOINT")
    if hasattr(sqlite3, name)
)


def _authorize(action, first, second, database, trigger):
    """Allow reads, functions, transactions and schema pragmas, and deny everything else."""
    if action in _READ_ACTIONS:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_PRAGMA:
        pragma = str(first or "").lower()
        if pragma in _READ_PRAGMAS or (pragma in _SETTING_READS and second is None):
            return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def _lock(connection: sqlite3.Connection) -> None:
    """Make a database answer reads only."""
    connection.set_authorizer(_authorize)


def _typed(values: list) -> tuple[str, list]:
    """A column's SQLite type and its values converted to it, empty text read as NULL."""
    cleaned = [None if value is None or (isinstance(value, str) and not value.strip()) else value for value in values]
    present = [value for value in cleaned if value is not None]

    def whole(value) -> bool:
        if isinstance(value, bool):
            return False
        if isinstance(value, int):
            return True
        return isinstance(value, str) and re.fullmatch(r"\s*[-+]?\d{1,18}\s*", value) is not None

    def number(value) -> bool:
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float)):
            return math.isfinite(float(value))
        if not isinstance(value, str):
            return False
        try:
            return math.isfinite(float(value.replace(",", "") if re.fullmatch(r"\s*[-+]?[\d,]*\.?\d+\s*", value) else value))
        except ValueError:
            return False

    if present and all(whole(value) for value in present):
        return "INTEGER", [None if value is None else int(str(value).strip()) for value in cleaned]
    if present and all(number(value) for value in present):
        return "REAL", [None if value is None else float(str(value).strip().replace(",", "")) for value in cleaned]
    return "TEXT", [None if value is None else str(value)[:MAX_CELL_CHARS] for value in cleaned]


def _store(connection: sqlite3.Connection, name: str, header: list, rows: list) -> int:
    """Write rows into a new table, each column typed by its values.

    Returns:
        Rows written.
    """
    columns = []
    for raw in header[:MAX_COLUMNS]:
        columns.append(table_name(raw if str(raw or "").strip() else f"column_{len(columns) + 1}", columns))
    width = len(columns)
    padded = [list(row[:width]) + [None] * (width - len(row[:width])) for row in rows]
    kinds, values = [], []
    for place in range(width):
        kind, converted = _typed([row[place] for row in padded])
        kinds.append(kind)
        values.append(converted)
    declared = ", ".join(f'"{column}" {kind}' for column, kind in zip(columns, kinds))
    connection.execute(f'CREATE TABLE "{name}" ({declared})')
    marks = ", ".join("?" for _ in columns)
    connection.executemany(f'INSERT INTO "{name}" VALUES ({marks})', list(zip(*values)) if values else [])
    connection.commit()
    return len(padded)


def _delimited(data: bytes, delimiter: str, max_rows: int) -> tuple[list, list]:
    """The header and rows of a CSV or TSV file."""
    text = data.decode("utf-8-sig", errors="replace")
    csv.field_size_limit(MAX_CELL_CHARS)
    reader = csv.reader(io.StringIO(text), delimiter=delimiter)
    header = next(reader, [])
    rows = []
    for row in reader:
        if len(rows) >= max_rows:
            break
        if row:
            rows.append(row)
    return header, rows


def _records(items: list, max_rows: int) -> tuple[list, list]:
    """The header and rows of a list of JSON objects, their keys in first-seen order."""
    header: list = []
    seen: set = set()
    kept = [item for item in items[:max_rows] if isinstance(item, dict)]
    for item in kept:
        for key in item:
            if key not in seen and len(header) < MAX_COLUMNS:
                seen.add(key)
                header.append(key)
    rows = []
    for item in kept:
        row = []
        for key in header:
            value = item.get(key)
            row.append(json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value)
        rows.append(row)
    return [str(key) for key in header], rows


def _json(data: bytes, lines: bool, max_rows: int) -> tuple[list, list]:
    """The header and rows of a JSON array of objects, or of JSON Lines."""
    if len(data) > MAX_JSON_BYTES:
        raise ValueError(f"a JSON table is read up to {MAX_JSON_BYTES // (1024 * 1024)} MB; save a larger one as CSV")
    text = data.decode("utf-8-sig", errors="replace")
    if lines:
        items = []
        for line in text.splitlines():
            if len(items) >= max_rows:
                break
            if line.strip():
                items.append(json.loads(line))
        return _records(items, max_rows)
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        parsed = next((value for value in parsed.values() if isinstance(value, list)), [parsed])
    if not isinstance(parsed, list):
        raise ValueError("the file holds no list of records")
    return _records(parsed, max_rows)


def _sheet(data: bytes, sheet: str, max_rows: int) -> tuple[list, list, str]:
    """The header and rows of one sheet of an .xlsx file, and the sheet's name."""
    openpyxl = deps.require("openpyxl", feature=FEATURE)
    book = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    try:
        wanted = str(sheet or "").strip()
        if wanted and wanted not in book.sheetnames:
            raise ValueError(f"the workbook has no sheet named {wanted!r}; it has {', '.join(book.sheetnames[:12])}")
        chosen = book[wanted] if wanted else book.worksheets[0]
        width = min(int(chosen.max_column or MAX_COLUMNS), MAX_COLUMNS)
        rows_iter = chosen.iter_rows(max_row=int(max_rows) + 1, max_col=width, values_only=True)
        header = [str(value) if value is not None else "" for value in next(rows_iter, ())]
        rows = []
        for row in rows_iter:
            if len(rows) >= max_rows:
                break
            if any(value is not None for value in row):
                rows.append([value.isoformat() if hasattr(value, "isoformat") else value for value in row])
        return header, rows, chosen.title
    finally:
        book.close()


def _sqlite_tables(path: str, connection: sqlite3.Connection, prefix: str, taken: list, max_rows: int) -> dict:
    """Copy every table of a SQLite file into the database, read through a read-only handle.

    Returns:
        New table name to rows copied.
    """
    source = sqlite3.connect(f"{pathlib.Path(os.path.abspath(path)).as_uri()}?mode=ro", uri=True)
    copied: dict = {}
    try:
        names = [row[0] for row in source.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")]
        for raw in names[:MAX_TABLES]:
            runtime.check_interrupt()
            cursor = source.execute(f'SELECT * FROM "{raw.replace(chr(34), chr(34) * 2)}" LIMIT ?', (int(max_rows),))
            header = [item[0] for item in cursor.description or ()]
            rows = [list(row) for row in cursor]
            name = table_name(f"{prefix}_{raw}" if prefix else raw, taken + list(copied))
            copied[name] = _store(connection, name, header, rows)
    finally:
        source.close()
    return copied


def file_options() -> list[str]:
    """The table file menu: every table file under the read roots, or :data:`NO_TABLES`."""
    try:
        found = file_listing.labels(TABLE_EXTENSIONS, file_listing.ROOTS, MAX_FILE_OPTIONS)
    except Exception as error:
        logger.debug("the file listing could not be read: %s", error)
        found = []
    return list(found) or [NO_TABLES]


def picked(label: str, node: str):
    """The file a menu label names: ``(path, size, mtime)``.

    Raises:
        ValueError: Nothing was picked, or the label names no file that is there.
        PathNotAllowed: The file resolved outside every permitted read root.
    """
    chosen = str(label or "").strip()
    if not chosen or chosen == NO_TABLES:
        raise ValueError(
            f"{node} has no file picked. Put a .csv, .xlsx, .json or .sqlite file in ComfyUI's input "
            f"folder, refresh the page and pick it."
        )
    entry = file_listing.find(chosen, TABLE_EXTENSIONS, file_listing.ROOTS, paths=True)
    if entry is None:
        raise ValueError(f"`{chosen[:120]}` names no file {node} can find. Refresh the page and pick it again.")
    resolved = sandbox.resolve_read(entry.path)
    if not resolved.is_file():
        raise ValueError(f"`{chosen[:120]}` is not a file. Pick a table file from the menu.")
    return str(resolved), int(entry.size), float(entry.mtime)


def load_file(label: str, name: str, sheet: str, max_rows: int, max_bytes: int, previous: Tables | None,
              node: str) -> tuple[Tables, list[str]]:
    """A database holding another's tables and those read from one file.

    Args:
        label: A label from :func:`file_options`.
        name: The table's name, blank for one taken from the file's name.
        sheet: The sheet of an .xlsx file, blank for its first.
        max_rows: Rows read from one table at most.
        max_bytes: Largest file read.
        previous: Tables to keep ahead of the new ones, or ``None``.
        node: The node's display name, for messages.

    Returns:
        The new tables, read only, and the names of the tables this file added.

    Raises:
        ValueError: The file is too large, cannot be read, or holds no rows.
    """
    path, size, _mtime = picked(label, node)
    if size > max_bytes:
        raise ValueError(f"{node}: the file is {size // (1024 * 1024)} MB, past max_file_mb. Raise max_file_mb.")
    extension = os.path.splitext(path)[1].lower()
    taken = list(previous.names) if previous is not None else []
    if len(taken) >= MAX_TABLES:
        raise ValueError(f"{node}: the chain already holds {MAX_TABLES} tables, the most one database holds.")
    connection = _copied(previous)
    stem = os.path.splitext(os.path.basename(path))[0]
    added: dict = {}
    try:
        if extension in (".sqlite", ".sqlite3", ".db"):
            added = _sqlite_tables(path, connection, str(name or "").strip(), taken, max_rows)
        else:
            with open(path, "rb") as handle:
                data = handle.read(max_bytes + 1)
            if extension in (".csv", ".tsv"):
                header, rows = _delimited(data, "\t" if extension == ".tsv" else ",", max_rows)
                title = stem
            elif extension in (".json", ".jsonl"):
                header, rows = _json(data, extension == ".jsonl", max_rows)
                title = stem
            else:
                header, rows, sheet_name = _sheet(data, sheet, max_rows)
                title = f"{stem}_{sheet_name}" if sheet else stem
            chosen = table_name(str(name or "").strip() or title, taken)
            added[chosen] = _store(connection, chosen, header, rows)
    except (ValueError, UnicodeError, csv.Error, sqlite3.Error, json.JSONDecodeError) as error:
        connection.close()
        raise ValueError(f"{node} could not read {os.path.basename(path)} ({runtime.describe_error(error)}).") from None
    if not any(added.values()):
        connection.close()
        raise ValueError(f"{node}: {os.path.basename(path)} holds no rows to read.")
    _lock(connection)
    sources = dict(previous.sources or {}) if previous is not None else {}
    rows = dict(previous.rows or {}) if previous is not None else {}
    for table, count in added.items():
        sources[table] = file_listing.label_of(label, TABLE_EXTENSIONS, file_listing.ROOTS)
        rows[table] = count
    made = Tables(connection=connection, names=tuple(taken + list(added)), sources=sources, rows=rows)
    return made, list(added)


def _database_class():
    """A llama-index ``SQLDatabase`` whose answers stop at a row count and a time."""
    runtime.core()
    from llama_index.core.utilities.sql_wrapper import SQLDatabase

    class BoundedDatabase(SQLDatabase):
        """A SQL database answering at most ``max_rows`` rows within ``seconds``."""

        def __init__(self, *args, max_rows: int = 200, seconds: float = 30.0, raw=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._max_rows = int(max_rows)
            self._seconds = float(seconds)
            self._raw = raw

        def run_sql(self, command: str):
            deadline = time.monotonic() + self._seconds
            self._raw.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, PROGRESS_STEPS)
            try:
                cursor = self._raw.execute(str(command).strip().rstrip(";"))
                keys = [item[0] for item in cursor.description or ()]
                rows = cursor.fetchmany(self._max_rows + 1)
            except sqlite3.DatabaseError as error:
                if "interrupted" in str(error).lower():
                    raise NotImplementedError(
                        f"The query ran past {self._seconds:g} seconds and was stopped. Ask for less, or raise timeout_seconds."
                    ) from None
                if "not authorized" in str(error).lower():
                    raise NotImplementedError("Only reading queries are allowed: the tables cannot be changed.") from None
                raise NotImplementedError(f"Statement {command!r} is invalid SQL.\nError: {error}") from None
            finally:
                self._raw.set_progress_handler(None, 0)
            cut = len(rows) > self._max_rows
            rows = rows[: self._max_rows]
            kept = [tuple(self.truncate_word(value, length=RESULT_VALUE_CHARS) for value in row) for row in rows]
            text = str(kept) + (f" (first {self._max_rows} rows)" if cut else "")
            return text, {"result": kept, "col_keys": keys, "cut": cut}

    return BoundedDatabase


def query_engine(tables: Tables, model, synthesize: bool, sample_rows: int, max_rows: int, seconds: float):
    """A llama-index text-to-SQL query engine over the tables.

    Args:
        tables: The tables.
        model: The :class:`~modules.llm.wires.LanguageModel` writing the SQL and the answer.
        synthesize: Write an answer from the rows, else answer the rows themselves.
        sample_rows: Rows of each table shown to the model beside its columns.
        max_rows: Rows one query hands back at most.
        seconds: How long one query may run.

    Returns:
        The ``NLSQLTableQueryEngine``.
    """
    runtime.core()
    import sqlalchemy
    from llama_index.core.query_engine import NLSQLTableQueryEngine
    from sqlalchemy.pool import StaticPool

    raw = tables.connection
    engine = sqlalchemy.create_engine("sqlite://", creator=lambda: raw, poolclass=StaticPool)
    database = _database_class()(
        engine,
        include_tables=list(tables.names),
        sample_rows_in_table_info=int(sample_rows),
        max_rows=int(max_rows),
        seconds=float(seconds),
        raw=raw,
    )
    return NLSQLTableQueryEngine(
        sql_database=database,
        llm=model.llm,
        tables=list(tables.names),
        synthesize_response=bool(synthesize),
    )


def sql_retriever(tables: Tables, model, max_rows: int, seconds: float):
    """A retriever whose model writes a SQL query from each question and hands back the rows.

    Args:
        tables: The tables.
        model: The :class:`~modules.llm.wires.LanguageModel` writing the SQL.
        max_rows: Rows one query hands back at most.
        seconds: How long one query may run.

    Returns:
        The llama-index ``NLSQLRetriever``, each row one retrieved chunk.
    """
    runtime.core()
    import sqlalchemy
    from llama_index.core.retrievers import NLSQLRetriever
    from sqlalchemy.pool import StaticPool

    raw = tables.connection
    engine = sqlalchemy.create_engine("sqlite://", creator=lambda: raw, poolclass=StaticPool)
    database = _database_class()(
        engine, include_tables=list(tables.names), sample_rows_in_table_info=3,
        max_rows=int(max_rows), seconds=float(seconds), raw=raw,
    )
    inner = NLSQLRetriever(database, tables=list(tables.names), llm=model.llm, return_raw=False, handle_sql_errors=True)
    return _rows_class()(inner)


def _rows_class():
    """The retriever writing each row a SQL retriever found as ``column: value`` lines."""
    if "rows" in _CLASSES:
        return _CLASSES["rows"]
    from llama_index.core.base.base_retriever import BaseRetriever
    from llama_index.core.schema import NodeWithScore, TextNode

    class RowsRetriever(BaseRetriever):
        """Rows a text-to-SQL retriever found, each a chunk whose text is its values."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def _retrieve(self, query_bundle):
            found = []
            for item in self.inner.retrieve(query_bundle):
                values = dict(item.node.metadata or {})
                text = item.node.get_content() if not values else "\n".join(f"{key}: {value}" for key, value in values.items())
                node = TextNode(text=text, metadata={"source": "table rows"})
                found.append(NodeWithScore(node=node, score=item.score))
            return found

    _CLASSES["rows"] = RowsRetriever
    return RowsRetriever


_CLASSES: dict = {}


def row_documents(tables: Tables, table: str, rows_per_document: int, max_rows: int) -> list:
    """Table rows as documents, each row written as ``column: value`` lines.

    Args:
        tables: The tables.
        table: One table's name, blank for every table.
        rows_per_document: Rows written into one document.
        max_rows: Rows read from one table at most.

    Returns:
        llama-index ``Document`` objects, each naming its table and first row.

    Raises:
        ValueError: The tables hold no table by that name.
    """
    from .documents import make_document

    wanted = str(table or "").strip()
    if wanted and wanted not in tables.names:
        raise ValueError(f"No table is named {wanted!r}. The tables are {', '.join(tables.names[:20])}.")
    made = []
    for name in [wanted] if wanted else list(tables.names):
        columns = [column for column, _kind in tables.columns(name)]
        cursor = tables.connection.execute(f'SELECT * FROM "{name}" LIMIT ?', (int(max_rows),))
        block, first = [], 1
        for number, row in enumerate(cursor, 1):
            runtime.check_interrupt()
            block.append("\n".join(f"{column}: {value}" for column, value in zip(columns, row) if value is not None))
            if len(block) >= rows_per_document:
                made.append(_rows_document(make_document, name, tables, first, number, block))
                block, first = [], number + 1
        if block:
            made.append(_rows_document(make_document, name, tables, first, first + len(block) - 1, block))
    return made


def _rows_document(make_document, name: str, tables: Tables, first: int, last: int, block: list):
    """One document holding a run of a table's rows."""
    metadata = {
        "table": name,
        "rows": f"{first}-{last}" if last != first else str(first),
        "file_name": re.sub(r"\s*\[[^\]]*\]$", "", str((tables.sources or {}).get(name, "")))[:200],
        "source": "table",
    }
    return make_document("\n\n".join(block), metadata, f"table:{name}:{first}")

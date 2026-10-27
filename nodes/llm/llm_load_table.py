"""Read a spreadsheet, CSV, JSON or SQLite file into tables a language model can query."""

from __future__ import annotations

import os

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM_TABLES
from ...modules.interface import run_result
from ...modules.llm import tables as tabling
from ...modules.llm import wires
from ...modules.io import picker

REQUIRES = "llm"

logger = log.get_logger("llm.tables")

#: The node's name in messages.
NODE = "LLM Load Table"


class LLMLoadTable(io.ComfyNode):
    """Tables read from one file, added to the tables before them."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMLoadTable",
            display_name="LLM Load Table",
            search_aliases=[
                "WASLLMLoadTable",
                "LLM Load Table",
                "SQLDatabase",
                "load csv",
                "load spreadsheet",
                "xlsx",
                "sqlite",
                "text to SQL",
            ],
            category="WAS Suite/LLM/Documents",
            description=(
                "Read a CSV, TSV, Excel sheet, JSON list or SQLite file into tables, so a language "
                "model can answer questions about them with LLM SQL Query Engine. Chain several to "
                "query them together. The panel shows each table's columns and first rows."
            ),
            inputs=[
                picker.path_input(io.Combo.Input(
                    "file", options=tabling.file_options(),
                    tooltip="The file, from ComfyUI's input, output or temp folders: `.csv`, `.tsv`, `.xlsx`, `.json`, `.jsonl`, `.sqlite` or `.db`.",
                )),
                io.String.Input(
                    "table_name", default="",
                    tooltip="The table's name in questions and SQL, such as `sales`. Blank = the file's name. For a SQLite file, placed before each of its tables' names.",
                ),
                io.String.Input(
                    "sheet", default="",
                    tooltip="The sheet of an .xlsx workbook, such as `Q3`. Blank = its first sheet.",
                ),
                io.Int.Input(
                    "max_rows", default=100000, min=1, max=tabling.MAX_ROWS,
                    tooltip="Rows read from one table at most: 100000 = most spreadsheets; rows past it are left out.",
                ),
                io.Int.Input(
                    "max_file_mb", default=64, min=1, max=1024,
                    tooltip="Largest file read, in megabytes: `64` = most exports; a larger file is refused.",
                ),
                LLM_TABLES.Input(
                    "tables", optional=True,
                    tooltip="Tables from an earlier LLM Load Table, kept beside these so one question can join them.",
                ),
            ],
            outputs=[
                LLM_TABLES.Output(
                    display_name="tables",
                    tooltip="Every table so far, read only, for LLM SQL Query Engine, LLM Table to Documents or the next LLM Load Table.",
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, file="", **kwargs) -> str:
        """The file's label, size and time, so an edited file is read again."""
        try:
            path, size, mtime = tabling.picked(file, NODE)
        except Exception:
            return str(file)[:200]
        return f"{file}|{size}|{mtime}"

    @classmethod
    def execute(cls, file, table_name, sheet, max_rows, max_file_mb, tables=None) -> io.NodeOutput:
        """Read the file into tables.

        Args:
            file: A label from the file menu.
            table_name: The table's name, blank for the file's.
            sheet: The workbook sheet, blank for the first.
            max_rows: Rows read from one table at most.
            max_file_mb: Largest file read, in megabytes.
            tables: Tables to keep beside these.

        Returns:
            The tables.
        """
        previous = None
        if tables is not None:
            previous = wires.expect(tables, tabling.Tables, NODE, "tables", "LLM Load Table")
        made, added = tabling.load_file(
            file, table_name, sheet, int(max_rows), int(max_file_mb) * 1024 * 1024, previous, NODE,
        )
        logger.info("%s read %s into %s", NODE, os.path.basename(str(file)), ", ".join(added))
        bodies = []
        for name in added[:4]:
            columns = made.columns(name)
            lines = [", ".join(f"{column} {kind}".strip() for column, kind in columns)]
            lines += [" | ".join("" if value is None else str(value)[:60] for value in row) for row in made.sample(name)]
            bodies.append(run_result.body(name, "\n".join(lines)))
        run_result.publish(
            status=run_result.OK,
            summary=f"read {', '.join(f'{name} ({made.rows[name]} rows)' for name in added)}",
            counts={"tables": len(made.names), "added": len(added), "rows": sum(made.rows[name] for name in added)},
            facts={"file": str(file)[:120], "all tables": ", ".join(made.names)[:200]},
            bodies=bodies,
        )
        return io.NodeOutput(made)

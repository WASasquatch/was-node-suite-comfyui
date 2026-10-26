"""Turn table rows into documents, for searching rows by meaning."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM_DOCUMENTS, LLM_TABLES
from ...modules.interface import run_result
from ...modules.llm import tables as tabling
from ...modules.llm import wires

REQUIRES = "llm"

logger = log.get_logger("llm.tables")

#: The node's name in messages.
NODE = "LLM Table to Documents"


class LLMTableToDocuments(io.ComfyNode):
    """Table rows as documents, each row written as column and value lines."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMTableToDocuments",
            display_name="LLM Table to Documents",
            search_aliases=[
                "WASLLMTableToDocuments",
                "LLM Table to Documents",
                "PandasCSVReader",
                "rows to documents",
                "csv documents",
            ],
            category="WAS Suite/LLM/Documents",
            description=(
                "Turn table rows into documents, each row written as `column: value` lines, so an "
                "index can search them by meaning. Use it for free text in rows; LLM SQL Query Engine "
                "suits counts and sums. The panel shows a sample."
            ),
            inputs=[
                LLM_TABLES.Input(
                    "tables",
                    tooltip="The tables, from LLM Load Table.",
                ),
                io.String.Input(
                    "table", default="",
                    tooltip="Which table, by name, such as `sales`. Blank = every table.",
                ),
                io.Int.Input(
                    "rows_per_document", default=1, min=1, max=1000,
                    tooltip="Rows written into one document: 1 = one row each, best for search; 20 = runs of rows.",
                ),
                io.Int.Input(
                    "max_rows", default=10000, min=1, max=tabling.MAX_ROWS,
                    tooltip="Rows read from one table at most: 10000 = a large sheet.",
                ),
                LLM_DOCUMENTS.Input(
                    "documents", optional=True,
                    tooltip="Documents from another loader. These rows are added after them.",
                ),
            ],
            outputs=[
                LLM_DOCUMENTS.Output(
                    display_name="documents",
                    tooltip="One document per row or run of rows, naming its table and rows, for an LLM index.",
                ),
            ],
        )

    @classmethod
    def execute(cls, tables, table, rows_per_document, max_rows, documents=None) -> io.NodeOutput:
        """Write the rows as documents.

        Args:
            tables: The tables.
            table: One table's name, blank for all.
            rows_per_document: Rows per document.
            max_rows: Rows read from one table.
            documents: Documents to add these after.

        Returns:
            The documents.
        """
        held = wires.expect(tables, tabling.Tables, NODE, "tables", "LLM Load Table")
        previous = None
        if documents is not None:
            previous = wires.expect(documents, wires.Documents, NODE, "documents", "an LLM document loader")
        made = tabling.row_documents(held, table, int(rows_per_document), int(max_rows))
        if not made:
            raise ValueError(f"{NODE}: the tables hold no rows.")
        added = wires.Documents(tuple(made), f"{len(made)} documents from {str(table or 'every table')}")
        result = previous.plus(added) if previous is not None else added
        logger.info("%s wrote %d document(s)", NODE, len(made))
        run_result.publish(
            status=run_result.OK,
            summary=f"wrote {len(made)} document(s)",
            counts={"documents": len(made), **({"with chain": len(result)} if len(result) != len(made) else {})},
            facts={"tables": str(table or ", ".join(held.names))[:200], "rows each": str(int(rows_per_document))},
            bodies=[run_result.body(f"{item.metadata['table']} rows {item.metadata['rows']}", item.text[:2000]) for item in made[:3]],
        )
        return io.NodeOutput(result)

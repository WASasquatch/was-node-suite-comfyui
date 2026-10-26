"""Find table rows by having a language model write a SQL query from each question."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM, LLM_RETRIEVER, LLM_TABLES
from ...modules.interface import run_result
from ...modules.llm import runtime, wires
from ...modules.llm import tables as tabling

REQUIRES = "llm"

logger = log.get_logger("llm.tables")

#: The node's name in messages.
NODE = "LLM SQL Retriever"


class LLMSQLRetriever(io.ComfyNode):
    """A text-to-SQL retriever: each row a query returns is one retrieved chunk."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMSQLRetriever",
            display_name="LLM SQL Retriever",
            search_aliases=[
                "WASLLMSQLRetriever",
                "LLM SQL Retriever",
                "NLSQLRetriever",
                "text to SQL retriever",
                "table rows",
            ],
            category="WAS Suite/LLM/Retrieve",
            description=(
                "Find table rows for a question: the model writes a SQL query and each row it returns "
                "becomes a chunk, so tables join documents in LLM Fusion Retriever, a query engine or an "
                "agent's retriever tool. The tables are only read."
            ),
            inputs=[
                LLM_TABLES.Input(
                    "tables",
                    tooltip="The tables, from LLM Load Table.",
                ),
                LLM.Input(
                    "llm",
                    tooltip="The model that writes the SQL, such as LLM From CLIP or LLM OpenAI.",
                ),
                io.Int.Input(
                    "max_rows", default=50, min=1, max=tabling.MAX_RESULT_ROWS,
                    tooltip="Rows one question hands on at most: 50 = a short context.",
                ),
                io.Int.Input(
                    "timeout_seconds", default=30, min=1, max=600,
                    tooltip="Seconds one query may run before it is stopped: 30 = the usual.",
                ),
            ],
            outputs=[
                LLM_RETRIEVER.Output(
                    display_name="retriever",
                    tooltip="The retriever, for LLM Retrieve, LLM Fusion Retriever, a query engine or LLM Retriever Tool.",
                ),
            ],
        )

    @classmethod
    def execute(cls, tables, llm, max_rows, timeout_seconds) -> io.NodeOutput:
        """Build the retriever.

        Args:
            tables: The tables.
            llm: The model writing the SQL.
            max_rows: Rows one question hands on.
            timeout_seconds: Seconds one query may run.

        Returns:
            The retriever.
        """
        held = wires.expect(tables, tabling.Tables, NODE, "tables", "LLM Load Table")
        model = wires.expect(llm, wires.LanguageModel, NODE, "llm", "LLM From CLIP or LLM OpenAI")
        wires.ready(model)
        with runtime.guarded():
            built = tabling.sql_retriever(held, model, int(max_rows), float(timeout_seconds))
        label = f"SQL over {len(held.names)} table(s)"
        logger.info("%s built over %s", NODE, ", ".join(held.names))
        run_result.publish(
            status=run_result.OK,
            summary=f"Retriever ready: {label}.",
            counts={"tables": len(held.names), "max rows": int(max_rows)},
            facts={"model": model.label, "tables": ", ".join(held.names)[:200], "tables are": "read only"},
        )
        return io.NodeOutput(wires.Retriever(retriever=built, label=label, depends=(model,)))

"""Answer questions about tables by having a language model write and run SQL."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM, LLM_QUERY_ENGINE, LLM_TABLES
from ...modules.interface import run_result
from ...modules.llm import query, runtime, wires
from ...modules.llm import tables as tabling

REQUIRES = "llm"

logger = log.get_logger("llm.tables")

#: The node's name in messages.
NODE = "LLM SQL Query Engine"


class LLMSQLQueryEngine(io.ComfyNode):
    """A text-to-SQL query engine over tables."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMSQLQueryEngine",
            display_name="LLM SQL Query Engine",
            search_aliases=[
                "WASLLMSQLQueryEngine",
                "LLM SQL Query Engine",
                "NLSQLTableQueryEngine",
                "text to SQL",
                "ask a spreadsheet",
                "table question",
            ],
            category="WAS Suite/LLM/Query",
            description=(
                "Answer questions about tables: the model writes a SQL query from each question, runs "
                "it, and answers from the rows. Ask with LLM Query, or give it to an agent with LLM "
                "Query Engine Tool. The tables are only read, never changed."
            ),
            inputs=[
                LLM_TABLES.Input(
                    "tables",
                    tooltip="The tables to ask about, from LLM Load Table.",
                ),
                LLM.Input(
                    "llm",
                    tooltip="The model that writes the SQL and the answer, such as LLM From CLIP or LLM OpenAI.",
                ),
                io.Boolean.Input(
                    "synthesize_response", default=True,
                    tooltip="`true` = an answer in words from the rows; `false` = the rows themselves as the answer.",
                ),
                io.Int.Input(
                    "sample_rows", default=3, min=0, max=20,
                    tooltip="Rows of each table shown to the model beside its columns, so it sees what values look like: 3 = the usual; 0 = columns only.",
                ),
                io.Int.Input(
                    "max_rows", default=200, min=1, max=tabling.MAX_RESULT_ROWS,
                    tooltip="Rows one query hands back at most: 200 = a page of results; more costs more tokens.",
                ),
                io.Int.Input(
                    "timeout_seconds", default=30, min=1, max=600,
                    tooltip="Seconds one query may run before it is stopped: 30 = the usual.",
                ),
            ],
            outputs=[
                LLM_QUERY_ENGINE.Output(
                    display_name="query_engine",
                    tooltip="The engine, for LLM Query, LLM Query Engine Tool or a router.",
                ),
            ],
        )

    @classmethod
    def execute(cls, tables, llm, synthesize_response, sample_rows, max_rows, timeout_seconds) -> io.NodeOutput:
        """Build the engine.

        Args:
            tables: The tables.
            llm: The model.
            synthesize_response: Answer in words, else the rows.
            sample_rows: Rows of each table shown to the model.
            max_rows: Rows one query hands back.
            timeout_seconds: Seconds one query may run.

        Returns:
            The query engine.
        """
        held = wires.expect(tables, tabling.Tables, NODE, "tables", "LLM Load Table")
        model = wires.expect(llm, wires.LanguageModel, NODE, "llm", "LLM From CLIP or LLM OpenAI")
        wires.ready(model)
        with runtime.guarded():
            try:
                engine = tabling.query_engine(
                    held, model, bool(synthesize_response), int(sample_rows), int(max_rows), float(timeout_seconds),
                )
            except Exception as error:
                raise query.failure(NODE, error) from None
        label = f"SQL over {len(held.names)} table(s) on {model.label}"
        wire = wires.QueryEngine(engine=engine, label=label, depends=(model,))
        logger.info("%s built over %s", NODE, ", ".join(held.names))
        run_result.publish(
            status=run_result.OK,
            summary=f"Answers about {', '.join(held.names)[:120]} with {model.label}",
            counts={"tables": len(held.names), "rows": sum((held.rows or {}).values())},
            facts={
                "model": model.label,
                "answer": "in words" if synthesize_response else "the rows",
                "limit": f"{int(max_rows)} rows, {int(timeout_seconds)} s",
                "tables": "read only",
            },
            items=[
                f"{name}: {', '.join(column for column, _kind in held.columns(name))}"[:200]
                for name in held.names[:20]
            ],
        )
        return io.NodeOutput(wire)

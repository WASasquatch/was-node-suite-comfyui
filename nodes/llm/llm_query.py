"""Ask a query engine a question and hand on the answer and the chunks it came from."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM_QUERY_ENGINE, LLM_SOURCES
from ...modules.interface import run_result
from ...modules.llm import query, runtime, wires

REQUIRES = "llm"

logger = log.get_logger("llm.query")


class WASLLMQuery(io.ComfyNode):
    """Ask one question of a query engine."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMQuery",
            display_name="LLM Query",
            search_aliases=[
                "WASLLMQuery",
                "LLM Query",
                "ask documents",
                "RAG",
                "question answering",
                "query engine",
            ],
            category="WAS Suite/LLM/Query",
            is_output_node=True,
            description=(
                "Ask a query engine a question and get its answer with the chunks it was written "
                "from, best first. The answer streams under the node as it is written, and the "
                "panel lists the sources with their scores."
            ),
            inputs=[
                LLM_QUERY_ENGINE.Input(
                    "query_engine",
                    tooltip="The engine to ask, such as from LLM Query Engine, LLM Query Transform or LLM Router Query Engine.",
                ),
                io.String.Input(
                    "question", multiline=True, default="",
                    tooltip="What to ask, such as `What does the contract say about late payment?`",
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="answer",
                    tooltip="The engine's answer, with any thinking a reasoning model wrote removed.",
                ),
                io.String.Output(
                    display_name="sources_text",
                    tooltip="The chunks the answer came from as numbered text, each headed by its score and file name.",
                ),
                LLM_SOURCES.Output(
                    display_name="sources",
                    tooltip="The chunks with their scores, the question and the answer, for the evaluators and LLM Sources to Text.",
                ),
            ],
        )

    @classmethod
    def execute(cls, query_engine, question) -> io.NodeOutput:
        """Ask the question.

        Args:
            query_engine: The engine to ask.
            question: The question.

        Returns:
            The answer, the sources as text and the sources wire.
        """
        node = "LLM Query"
        engine = wires.expect(query_engine, wires.QueryEngine, node, "query_engine", "LLM Query Engine")
        text = str(question or "").strip()
        if not text:
            raise ValueError(f"{node} needs a question. Type one into its question box.")
        wires.ready(engine)
        with runtime.guarded():
            stream = runtime.TextStream(engine.label)
            try:
                found = query.run_query(engine.engine, text, stream)
            except Exception as error:
                raise query.failure(node, error) from None
        answer = found["answer"]
        nodes = found["nodes"]
        listing = query.sources_text(nodes)
        logger.info(
            "LLM Query answered in %.2fs from %d chunk(s), %d characters%s",
            found["seconds"], len(nodes), len(answer), ", streamed" if found["streamed"] else "",
        )
        bodies = [run_result.body("answer", answer)]
        if found["thinking"]:
            bodies.append(run_result.body("thinking", found["thinking"]))
        if found["sql"]:
            bodies.append(run_result.body("sql", found["sql"]))
        if listing:
            bodies.append(run_result.body("sources", listing))
        run_result.publish(
            status=run_result.OK if answer or nodes else run_result.WARNING,
            summary=(
                f"Answered from {len(nodes)} chunk{'s' if len(nodes) != 1 else ''} in {found['seconds']:.1f}s"
                if answer or nodes else "The engine answered nothing and found no chunks"
            ),
            counts={"chunks": len(nodes), "characters": len(answer), "seconds": round(found["seconds"], 2)},
            facts={"engine": engine.label, "streamed": "yes" if found["streamed"] else "no"},
            items=query.source_rows(nodes),
            items_total=len(nodes),
            bodies=bodies,
        )
        sources = wires.Sources(items=nodes, query=text, answer=answer)
        return io.NodeOutput(answer, listing, sources)

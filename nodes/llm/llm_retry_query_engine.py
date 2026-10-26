"""Judge each answer and ask again, the question rewritten, while it fails."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM, LLM_QUERY_ENGINE
from ...modules.interface import run_result
from ...modules.llm import query, runtime, wires

REQUIRES = "llm"

logger = log.get_logger("llm.query")

#: The node's name in messages.
NODE = "LLM Retry Query Engine"


class LLMRetryQueryEngine(io.ComfyNode):
    """A query engine that retries until its answer passes a judge."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMRetryQueryEngine",
            display_name="LLM Retry Query Engine",
            search_aliases=[
                "WASLLMRetryQueryEngine",
                "LLM Retry Query Engine",
                "RetryQueryEngine",
                "RetryGuidelineQueryEngine",
                "self correction",
            ],
            category="WAS Suite/LLM/Query",
            description=(
                "Judge each answer of another query engine and, while it fails, ask again with the "
                "question rewritten from the judge's feedback. With guidelines the answer must follow "
                "them; without, it must fit the question and its sources."
            ),
            inputs=[
                LLM_QUERY_ENGINE.Input(
                    "query_engine",
                    tooltip="The engine whose answers are judged, such as LLM Query Engine.",
                ),
                LLM.Input(
                    "llm",
                    tooltip="The judge, which also rewrites the question, such as LLM OpenAI or LLM From CLIP.",
                ),
                io.String.Input(
                    "guidelines", multiline=True, default="",
                    tooltip="What an answer must do, such as `Name the file it came from. Stay under 60 words.` Blank = it must fit the question and its sources.",
                ),
                io.Int.Input(
                    "max_retries", default=2, min=1, max=5,
                    tooltip="Further attempts after a failing answer: 2 = the usual; each costs a judgement and an answer.",
                ),
            ],
            outputs=[
                LLM_QUERY_ENGINE.Output(
                    display_name="query_engine",
                    tooltip="The retrying engine, for LLM Query or LLM Query Engine Tool.",
                ),
            ],
        )

    @classmethod
    def execute(cls, query_engine, llm, guidelines, max_retries) -> io.NodeOutput:
        """Build the engine.

        Args:
            query_engine: The engine answering.
            llm: The judge.
            guidelines: What an answer must do, or blank.
            max_retries: Further attempts.

        Returns:
            The query engine.
        """
        engine = wires.expect(query_engine, wires.QueryEngine, NODE, "query_engine", "LLM Query Engine")
        model = wires.expect(llm, wires.LanguageModel, NODE, "llm", "LLM OpenAI or LLM From CLIP")
        wires.ready(engine, model)
        with runtime.guarded():
            try:
                built = query.build_retry(engine, model, guidelines, int(max_retries))
            except Exception as error:
                raise query.failure(NODE, error) from None
        judged = "guidelines" if str(guidelines or "").strip() else "fit to the question"
        label = f"retry ({judged}) over {engine.label}"
        logger.info("%s built: %s", NODE, label)
        run_result.publish(
            status=run_result.OK,
            summary=f"Retries up to {int(max_retries)} time(s) while {judged} fails",
            counts={"retries": int(max_retries)},
            facts={"judge": model.label, "engine": engine.label, "judged on": judged},
            bodies=[run_result.body("guidelines", str(guidelines))] if str(guidelines or "").strip() else None,
        )
        return io.NodeOutput(wires.QueryEngine(engine=built, label=label, depends=(engine, model)))

"""Build a query engine that answers questions from a retriever's chunks."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM, LLM_POSTPROCESSORS, LLM_QUERY_ENGINE, LLM_RETRIEVER
from ...modules.interface import run_result
from ...modules.llm import generate, query, runtime, wires

REQUIRES = "llm"

logger = log.get_logger("llm.query")


class WASLLMQueryEngine(io.ComfyNode):
    """A query engine over a retriever, a model and a way of writing the answer."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMQueryEngine",
            display_name="LLM Query Engine",
            search_aliases=[
                "WASLLMQueryEngine",
                "LLM Query Engine",
                "RetrieverQueryEngine",
                "CitationQueryEngine",
                "RAG",
                "question answering",
                "citations",
                "skill",
            ],
            category="WAS Suite/LLM/Query",
            description=(
                "Combine a retriever and a language model into a query engine that answers "
                "questions from indexed documents. Feed it to LLM Query to ask, or to LLM Query "
                "Engine Tool to hand it to an agent. With citations on, answers cite numbered "
                "sources."
            ),
            inputs=[
                LLM_RETRIEVER.Input(
                    "retriever",
                    tooltip="Where the chunks each answer is written from come from, such as LLM Index Retriever over a vector index.",
                ),
                LLM.Input(
                    "llm",
                    tooltip="The model that writes the answer from the chunks, such as LLM OpenAI or LLM From CLIP.",
                ),
                io.Combo.Input(
                    "response_mode", options=list(query.RESPONSE_MODES), default="compact",
                    tooltip=(
                        "`compact` = chunks packed into few calls; `refine` = one call a chunk; "
                        "`tree_summarize` = summarised upward; `simple_summarize` = one call; "
                        "`accumulate` = one answer a chunk, `compact_accumulate` in fewer calls; "
                        "`generation` = chunks ignored; `no_text` = chunks only; `context_only` = "
                        "chunks as the answer."
                    ),
                ),
                io.Boolean.Input(
                    "citations", default=False,
                    tooltip="`true` cuts the chunks into numbered sources and asks the model to cite them, such as `[1]`; `false` answers without citations.",
                ),
                io.Int.Input(
                    "citation_chunk_size", default=512, min=64, max=8192, step=64,
                    tooltip="Tokens in one numbered source while citations is `true`. `256` cites finely, `1024` cites whole passages.",
                ),
                io.String.Input(
                    "qa_prompt", multiline=True, default="", optional=True,
                    tooltip=(
                        "Replaces the question prompt. Must hold `{context_str}` and "
                        "`{query_str}`, such as `Answer from this only:\\n{context_str}\\n"
                        "Question: {query_str}`. Empty uses the built-in prompt."
                    ),
                ),
                LLM_POSTPROCESSORS.Input(
                    "postprocessors", optional=True,
                    tooltip="Filters and rerankers run over the retrieved chunks before the answer is written, in chain order.",
                ),
                io.String.Input(
                    "system_prompt", force_input=True, optional=True,
                    tooltip=(
                        "Standing instructions the model follows in every answer, such as a "
                        "skill from Load System Prompt whose documents feed the retriever, or "
                        "`Answer in British English.` Unconnected = none."
                    ),
                ),
            ],
            outputs=[
                LLM_QUERY_ENGINE.Output(
                    display_name="query_engine",
                    tooltip="The query engine, for LLM Query, LLM Query Transform or LLM Query Engine Tool.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        retriever,
        llm,
        response_mode,
        citations,
        citation_chunk_size,
        qa_prompt="",
        postprocessors=None,
        system_prompt="",
    ) -> io.NodeOutput:
        """Build the query engine.

        Args:
            retriever: Where the chunks come from.
            llm: The model writing answers.
            response_mode: How the answer is written from the chunks.
            citations: Whether answers cite numbered sources.
            citation_chunk_size: Tokens in one numbered source.
            qa_prompt: A question prompt replacing the built-in one.
            postprocessors: Steps run over the chunks first.
            system_prompt: Instructions opening every call to the model.

        Returns:
            The query engine.
        """
        node = "LLM Query Engine"
        source = wires.expect(retriever, wires.Retriever, node, "retriever", "LLM Index Retriever or another LLM retriever")
        model = wires.expect(llm, wires.LanguageModel, node, "llm", "LLM OpenAI or LLM From CLIP")
        steps = wires.chain(postprocessors)
        if not wires.every(steps, wires.Postprocessor):
            raise ValueError(f"{node} was handed something other than postprocessors on its postprocessors input. Connect LLM Postprocessor or LLM Rerank to it.")
        if response_mode not in query.RESPONSE_MODES:
            raise ValueError(f"{node}: response_mode {str(response_mode)[:40]!r} is not one of {', '.join(query.RESPONSE_MODES)}.")
        prompt = query.check_qa_prompt(qa_prompt, node)
        instructions = str(system_prompt or "").strip()
        if instructions:
            try:
                model = model.with_llm(generate.with_system_prompt(model.llm, instructions))
            except ValueError as error:
                raise ValueError(f"{node}: {error}. Leave system_prompt unconnected for this model.") from None
        shows_pictures = bool(source.pictures and model.vision)
        wires.ready(source, model, steps)
        with runtime.guarded():
            try:
                engine = query.build_query_engine(
                    source, model, response_mode, steps, bool(citations), int(citation_chunk_size), prompt,
                    multimodal=shows_pictures,
                )
            except Exception as error:
                raise query.failure(node, error) from None
        label = f"{'cited ' if citations else ''}{response_mode} on {model.label}"
        wire = wires.QueryEngine(engine=engine, label=label, depends=(source, model, *steps))
        unused = []
        if prompt and response_mode in query.NO_QA_MODES:
            unused.append(f"qa_prompt is not read by {response_mode}")
        if citations and response_mode in ("no_text", "context_only"):
            unused.append(f"no answer cites anything under {response_mode}")
        if prompt and shows_pictures:
            unused.append("qa_prompt is not read while pictures are shown to the model")
        facts = {
            "model": model.label,
            "retriever": source.label,
            "response mode": response_mode,
            "citations": f"{int(citation_chunk_size)} tokens a source" if citations else "off",
            "qa prompt": f"{len(prompt)} characters" if prompt else "built-in",
            "system prompt": f"{len(instructions)} characters" if instructions else "none",
        }
        if source.pictures:
            facts["pictures"] = "shown to the model" if shows_pictures else f"named only; {model.label} does not read pictures"
        if steps:
            facts["postprocessors"] = ", ".join(step.label for step in steps)[:64]
        logger.info("LLM Query Engine built %s with %d postprocessor(s)", response_mode, len(steps))
        run_result.publish(
            status=run_result.WARNING if unused else run_result.OK,
            summary="; ".join(unused) if unused else f"Answers with {model.label}, {response_mode}",
            counts={"postprocessors": len(steps)},
            facts=facts,
            bodies=[
                run_result.body(title, text)
                for title, text in (("system prompt", instructions[:4000]), ("qa prompt", prompt))
                if text
            ] or None,
        )
        return io.NodeOutput(wire)

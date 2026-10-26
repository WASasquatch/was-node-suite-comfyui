"""Write titles, summaries, questions and keywords onto chunks with a language model."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM, LLM_DOCUMENTS
from ...modules.interface import run_result
from ...modules.llm import ingest, runtime, wires

REQUIRES = "llm"

logger = log.get_logger("llm.ingest")

#: Characters of each sampled value a run result carries.
SAMPLE_CHARS = 1500


class LLMExtractMetadata(io.ComfyNode):
    """Chunks with model-written titles, summaries, questions and keywords in their metadata."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMExtractMetadata",
            display_name="LLM Extract Metadata",
            search_aliases=[
                "WASLLMExtractMetadata",
                "LLM Extract Metadata",
                "TitleExtractor",
                "SummaryExtractor",
                "QuestionsAnsweredExtractor",
                "KeywordExtractor",
                "metadata extractor",
                "enrich chunks",
            ],
            category="WAS Suite/LLM/Documents",
            description=(
                "Have a language model write a title, summaries, the questions a chunk answers, "
                "keywords and the names it mentions onto each chunk, so an index finds it more often. Every chunk costs "
                "model calls. The panel shows what was written."
            ),
            inputs=[
                LLM_DOCUMENTS.Input(
                    "documents",
                    tooltip="The chunks to enrich, from LLM Split Documents, or whole documents from a loader.",
                ),
                LLM.Input(
                    "llm",
                    tooltip="The model that writes the metadata, such as LLM From CLIP, LLM GGUF Loader or LLM OpenAI.",
                ),
                io.Int.Input(
                    "title",
                    default=5,
                    min=0,
                    max=20,
                    tooltip=(
                        "Chunks of each document its title is written from. `5` = the first five; "
                        "`0` = no title. Every chunk of a document gets the same title."
                    ),
                ),
                io.Combo.Input(
                    "summaries",
                    options=["none", "self", "self and neighbours"],
                    default="none",
                    tooltip=(
                        "`none` = no summary; `self` = a summary of each chunk; `self and "
                        "neighbours` = also the summaries of the chunks before and after it."
                    ),
                ),
                io.Int.Input(
                    "questions",
                    default=3,
                    min=0,
                    max=10,
                    tooltip="Questions each chunk can answer, written onto it. `3` is usual; `0` = none.",
                ),
                io.Int.Input(
                    "keywords",
                    default=0,
                    min=0,
                    max=20,
                    tooltip="Keywords written onto each chunk. `5` is usual; `0` = none.",
                ),
                io.Int.Input(
                    "max_chunks",
                    default=200,
                    min=1,
                    max=10000,
                    tooltip=(
                        "Most chunks enriched in one run; more are refused rather than billed. "
                        "`200` costs about 200 model calls per extractor turned on."
                    ),
                ),
                io.Boolean.Input(
                    "entities", default=False, optional=True,
                    tooltip="`true` writes the people, places, organisations and products each chunk names, such as `Mira, Northwind Films`, as entities; `false` = none.",
                ),
            ],
            outputs=[
                LLM_DOCUMENTS.Output(
                    display_name="documents",
                    tooltip=(
                        "Copies of the chunks with document_title, section_summary, "
                        "questions_this_excerpt_can_answer, excerpt_keywords or entities added, for an "
                        "LLM index."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, documents, llm, title, summaries, questions, keywords, max_chunks, entities=False) -> io.NodeOutput:
        """Run each extractor turned on over the chunks.

        Args:
            documents: The chunks.
            llm: The language model.
            title: Chunks a title is written from, 0 for none.
            summaries: Which summaries to write.
            questions: Questions per chunk, 0 for none.
            keywords: Keywords per chunk, 0 for none.
            max_chunks: Most chunks enriched.
            entities: Write the names each chunk holds.

        Returns:
            The enriched copies.

        Raises:
            ValueError: A wire carries something else, every extractor is off, or there are more
                chunks than max_chunks.
            RuntimeError: A hosted model with features.network off, or the model failed.
        """
        source = wires.expect(
            documents, wires.Documents, "LLM Extract Metadata", "documents", "LLM Split Documents"
        )
        model = wires.expect(
            llm, wires.LanguageModel, "LLM Extract Metadata", "llm", "LLM From CLIP, LLM GGUF Loader or LLM OpenAI"
        )
        wires.ready(model)
        if not source.items:
            raise ValueError("LLM Extract Metadata was handed no chunks. Connect LLM Split Documents.")
        if len(source.items) > int(max_chunks):
            raise ValueError(
                f"LLM Extract Metadata was handed {len(source.items)} chunks and max_chunks is "
                f"{max_chunks}. Raise max_chunks to enrich them all, or split into larger chunks."
            )
        stream = runtime.TextStream(model.label)
        with runtime.guarded():
            enriched, written = ingest.extract(
                list(source.items), model, int(title), str(summaries), int(questions),
                int(keywords), stream, bool(entities),
            )
        stream.finish()
        result = wires.Documents(tuple(enriched), source.origin)
        logger.info(
            "LLM Extract Metadata enriched %d chunk(s) with %s", len(enriched), model.label,
        )
        first = enriched[0].metadata if enriched else {}
        bodies = [
            run_result.body(key, str(first[key])[:SAMPLE_CHARS])
            for key in ingest.EXTRACTED_KEYS
            if first.get(key)
        ]
        counts = {"chunks": len(enriched)}
        counts.update({key.replace("_", " "): count for key, count in written.items()})
        run_result.publish(
            status=run_result.OK,
            summary=f"wrote metadata on {len(enriched)} chunk(s) with {model.label}",
            counts=counts,
            facts={"model": model.label, "summaries": str(summaries)},
            bodies=bodies,
        )
        return io.NodeOutput(result)

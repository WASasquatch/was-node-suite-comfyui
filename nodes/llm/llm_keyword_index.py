"""Map keywords to the chunks holding them in a keyword index."""

from __future__ import annotations

import time

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM, LLM_DOCUMENTS, LLM_INDEX
from ...modules.llm import indexes, runtime, wires

REQUIRES = "llm"

logger = log.get_logger("llm.index")

#: The node's name in messages.
NODE = "LLM Keyword Index"


class LLMKeywordIndex(io.ComfyNode):
    """Map keywords to chunks, named by a model or counted from the text."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMKeywordIndex",
            display_name="LLM Keyword Index",
            search_aliases=[
                "WASLLMKeywordIndex",
                "LLM Keyword Index",
                "KeywordTableIndex",
                "SimpleKeywordTableIndex",
                "keyword table",
                "keywords",
                "RAG",
            ],
            category="WAS Suite/LLM/Index",
            description=(
                "Map keywords to the chunks holding them, for finding passages by the words a "
                "question uses. With a language model connected it names each chunk's keywords; "
                "without one they are counted from the text. The panel lists the keywords found."
            ),
            inputs=[
                LLM_DOCUMENTS.Input(
                    "documents",
                    tooltip=(
                        "What to index, such as from LLM Load Documents, or chunks from LLM Split "
                        "Documents. Whole documents are cut into chunks of chunk_size tokens."
                    ),
                ),
                io.Int.Input(
                    "max_keywords", default=10, min=1, max=100,
                    tooltip="Keywords kept for each chunk. 10 = the main topics; 30 = names and details too.",
                ),
                LLM.Input(
                    "llm", optional=True,
                    tooltip=(
                        "A model that names each chunk's keywords, one generation per chunk. Left "
                        "empty, keywords are the chunk's most frequent words, which is instant."
                    ),
                ),
                io.Int.Input(
                    "chunk_size", default=1024, min=32, max=8192, optional=True, advanced=True,
                    tooltip=(
                        "Tokens in each chunk when whole documents are cut here. 1024 = a few "
                        "paragraphs; 256 = finer matches with less context each. Chunks from LLM "
                        "Split Documents keep their own size."
                    ),
                ),
                io.Int.Input(
                    "chunk_overlap", default=20, min=0, max=2048, optional=True, advanced=True,
                    tooltip=(
                        "Tokens two neighbouring chunks share, so a sentence cut at a border is "
                        "still found whole. 20 = a little; 0 = none. Smaller than chunk_size."
                    ),
                ),
            ],
            outputs=[
                LLM_INDEX.Output(
                    display_name="index",
                    tooltip="The keyword index, for LLM Index Retriever, a query engine or LLM Save Index.",
                ),
            ],
        )

    @classmethod
    def execute(cls, documents, max_keywords, llm=None, chunk_size=1024, chunk_overlap=20) -> io.NodeOutput:
        """Split the documents and map their keywords.

        Args:
            documents: The documents or chunks.
            max_keywords: Keywords kept per chunk.
            llm: The language model naming keywords, or ``None`` to count words.
            chunk_size: Tokens per chunk for whole documents.
            chunk_overlap: Tokens neighbouring chunks share.

        Returns:
            The keyword index.
        """
        docs = wires.expect(documents, wires.Documents, NODE, "documents", "LLM Load Documents")
        model = None
        if llm is not None:
            model = wires.expect(llm, wires.LanguageModel, NODE, "llm", "LLM From CLIP, LLM GGUF Loader or LLM OpenAI")
            wires.ready(model)
        started = time.perf_counter()
        with runtime.guarded(), indexes.failing(NODE):
            prepared = indexes.prepare(docs, NODE, chunk_size, chunk_overlap)
            index = indexes.build_keyword(prepared, model, max_keywords)
        wire = wires.Index(index=index, kind="keyword", llm=model, size=indexes.size_of(index, "keyword"))
        seconds = time.perf_counter() - started
        keywords = len(index.index_struct.table)
        source = model.label if model is not None else "word count"
        logger.info(
            "%s mapped %d keyword(s) across %d chunk(s) by %s in %.2fs",
            NODE, keywords, wire.size, source, seconds,
        )
        indexes.report(wire, seconds, f"mapped {keywords} keywords across {wire.size} chunks", prepared)
        return io.NodeOutput(wire)

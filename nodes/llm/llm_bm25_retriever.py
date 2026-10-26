"""LLM BM25 Retriever: ranks chunks by the words they share with the question."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import LLM_DOCUMENTS, LLM_INDEX, LLM_RETRIEVER
from ...modules.interface import run_result
from ...modules.llm import bm25, runtime, wires

REQUIRES = "llm"


class LLMBM25Retriever(io.ComfyNode):
    """A keyword retriever ranking chunks by Okapi BM25."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMBM25Retriever",
            display_name="LLM BM25 Retriever",
            search_aliases=[
                "WASLLMBM25Retriever",
                "LLM BM25 Retriever",
                "BM25Retriever",
                "keyword search",
                "sparse retrieval",
                "lexical",
                "RAG",
            ],
            category="WAS Suite/LLM/Retrieve",
            description=(
                "Rank chunks by the words they share with the question, as a search engine does, "
                "with no embedding model. It finds exact names, codes and rare terms that "
                "similarity search misses, and pairs well with a vector retriever in LLM Fusion "
                "Retriever."
            ),
            inputs=[
                LLM_DOCUMENTS.Input(
                    "documents", optional=True,
                    tooltip="The chunks to rank, from an LLM splitter. Whole documents work too and are ranked whole.",
                ),
                LLM_INDEX.Input(
                    "index", optional=True,
                    tooltip="An index whose stored chunks are ranked, read once when this node runs. Used with or in place of documents.",
                ),
                io.Int.Input(
                    "top_k", default=10, min=1, max=200,
                    tooltip="Chunks one search hands on: 10 = the usual; 30 = broad recall for fusion or a reranker.",
                ),
                io.Boolean.Input(
                    "stemming", default=True,
                    tooltip="true = word endings are dropped, so `running` matches `runs`; false = words match only as written.",
                ),
                io.Combo.Input(
                    "stopwords", options=list(bm25.LANGUAGES), default="english",
                    tooltip="The language whose common words are ignored, which also picks the stemmer: `english` skips `the`, `and`, `of`; `none` = every word counts.",
                ),
                io.Float.Input(
                    "k1", default=bm25.DEFAULT_K1, min=0.0, max=3.0, step=0.05, optional=True, advanced=True,
                    tooltip="How much repeating a word counts: 1.5 = the usual; 0 = only whether it appears; 3 = repeats weigh heavily.",
                ),
                io.Float.Input(
                    "b", default=bm25.DEFAULT_B, min=0.0, max=1.0, step=0.05, optional=True, advanced=True,
                    tooltip="How much long chunks are held back: 0.75 = the usual; 0 = length ignored; 1 = scaled fully by length.",
                ),
            ],
            outputs=[
                LLM_RETRIEVER.Output(
                    display_name="retriever",
                    tooltip="The retriever, for LLM Retrieve, LLM Fusion Retriever, a query engine or a chat engine.",
                ),
            ],
        )

    @classmethod
    def execute(cls, top_k, stemming, stopwords, documents=None, index=None, k1=bm25.DEFAULT_K1,
                b=bm25.DEFAULT_B) -> io.NodeOutput:
        """Count the chunks and build the retriever.

        Args:
            top_k: Chunks one search hands on.
            stemming: Whether words are stemmed.
            stopwords: The stopword language.
            documents: Chunks to rank.
            index: An index whose chunks are ranked.
            k1: Term frequency saturation.
            b: Length normalisation.

        Returns:
            The retriever.
        """
        node = "LLM BM25 Retriever"
        if documents is None and index is None:
            raise ValueError(
                f"{node} needs chunks to rank. Connect split chunks to documents, or an index to index."
            )
        if documents is not None:
            wires.expect(documents, wires.Documents, node, "documents", "an LLM splitter or LLM Load Documents")
        if index is not None:
            wires.expect(index, wires.Index, node, "index", "an LLM index node")
        if stopwords not in bm25.LANGUAGES:
            raise ValueError(f"stopwords `{str(stopwords)[:32]}` is not one of the listed languages.")
        with runtime.guarded():
            nodes = bm25.corpus(documents, index)
            if not nodes:
                raise ValueError(f"{node} was handed no chunks: the documents and the index are empty.")
            built = bm25.retriever(nodes, top_k, stopwords, bool(stemming), k1, b)
        okapi = built.okapi
        label = f"bm25 top {int(top_k)}"
        retriever = wires.Retriever(retriever=built, label=label, depends=())
        origin = getattr(documents, "origin", "") if documents is not None else ""
        facts = {
            "stopwords": stopwords,
            "stemming": "on" if stemming else "off",
            "k1, b": f"{float(k1):g}, {float(b):g}",
            "average chunk": f"{okapi.average:.0f} terms",
            "counted in": f"{built.seconds:.2f}s",
        }
        if origin:
            facts["from"] = origin[:64]
        if index is not None:
            facts["index"] = index.name or index.kind
        run_result.publish(
            status=run_result.OK,
            summary=f"BM25 over {okapi.size} chunk(s), {okapi.vocabulary} distinct term(s).",
            counts={"chunks": okapi.size, "terms": okapi.vocabulary, "top_k": int(top_k)},
            facts=facts,
        )
        return io.NodeOutput(retriever)

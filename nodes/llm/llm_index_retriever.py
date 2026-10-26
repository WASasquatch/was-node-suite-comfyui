"""LLM Index Retriever: a retriever over a built index, in one of that index's search modes."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM, LLM_EMBEDDING, LLM_INDEX, LLM_RETRIEVER
from ...modules.interface import run_result
from ...modules.llm import retrieval, runtime, wires

REQUIRES = "llm"

logger = log.get_logger("llm.retrieve")

#: Tooltip on every max_keywords setting.
KEYWORDS_TIP = "Most keywords taken from the question: 10 = the usual; 3 = only the strongest terms."

#: Tooltip on every child_branch_factor setting.
BRANCH_TIP = "Branches followed at each level of the tree: 1 = the single best; 2 or 3 = broader, more chunks."

#: Tooltip on every path_depth setting.
DEPTH_TIP = "Steps walked out from each entity found: 1 = direct relations; 2 = relations of relations."


class LLMIndexRetriever(io.ComfyNode):
    """A retriever over a built index in one of its search modes."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        modes = [
            io.DynamicCombo.Option(key="default", inputs=[]),
            io.DynamicCombo.Option(key="mmr", inputs=[
                io.Float.Input(
                    "mmr_threshold", default=0.5, min=0.0, max=1.0, step=0.05,
                    tooltip="1 = plain similarity; 0.5 = balanced; 0 = the most varied chunks. Lower it when the top chunks repeat one another.",
                ),
            ]),
            io.DynamicCombo.Option(key="embedding", inputs=[]),
            io.DynamicCombo.Option(key="llm", inputs=[
                io.Int.Input(
                    "choice_batch_size", default=10, min=1, max=50,
                    tooltip="Chunks or summaries shown to the language model in one prompt while it picks: 10 = the usual; larger = fewer, longer prompts.",
                ),
            ]),
            io.DynamicCombo.Option(key="simple", inputs=[
                io.Int.Input("max_keywords", default=10, min=1, max=50, tooltip=KEYWORDS_TIP),
            ]),
            io.DynamicCombo.Option(key="rake", inputs=[
                io.Int.Input("max_keywords", default=10, min=1, max=50, tooltip=KEYWORDS_TIP),
            ]),
            io.DynamicCombo.Option(key="select_leaf", inputs=[
                io.Int.Input("child_branch_factor", default=1, min=1, max=10, tooltip=BRANCH_TIP),
            ]),
            io.DynamicCombo.Option(key="select_leaf_embedding", inputs=[
                io.Int.Input("child_branch_factor", default=1, min=1, max=10, tooltip=BRANCH_TIP),
            ]),
            io.DynamicCombo.Option(key="root", inputs=[]),
            io.DynamicCombo.Option(key="all_leaf", inputs=[]),
            io.DynamicCombo.Option(key="graph_synonyms", inputs=[
                io.Int.Input("max_keywords", default=10, min=1, max=50, tooltip=KEYWORDS_TIP),
                io.Int.Input("path_depth", default=1, min=1, max=5, tooltip=DEPTH_TIP),
            ]),
            io.DynamicCombo.Option(key="graph_vector", inputs=[
                io.Int.Input("path_depth", default=1, min=1, max=5, tooltip=DEPTH_TIP),
            ]),
            io.DynamicCombo.Option(key="pictures", inputs=[]),
            io.DynamicCombo.Option(key="text", inputs=[]),
        ]
        return io.Schema(
            node_id="WASLLMIndexRetriever",
            display_name="LLM Index Retriever",
            search_aliases=[
                "WASLLMIndexRetriever",
                "LLM Index Retriever",
                "retriever",
                "VectorIndexRetriever",
                "as_retriever",
                "mmr",
                "metadata filters",
                "RAG",
            ],
            category="WAS Suite/LLM/Retrieve",
            description=(
                "Search a built index for the chunks that answer a question: by similarity, by "
                "keywords, by a language model's choice, down a tree or across a graph, with "
                "optional metadata filters. Connect the retriever to LLM Retrieve, LLM Fusion "
                "Retriever or a query engine."
            ),
            inputs=[
                LLM_INDEX.Input(
                    "index",
                    tooltip="The index to search, from an LLM index node such as LLM Vector Index, or a saved one loaded back.",
                ),
                io.Int.Input(
                    "top_k", default=4, min=1, max=200,
                    tooltip="Chunks one search hands on: 4 = a short context; 20 = broad recall for a reranker to narrow. Document summary modes count documents; a tree's root and all_leaf modes hand on everything; an image index's default mode finds this many of each.",
                ),
                io.DynamicCombo.Input(
                    "mode", options=modes, display_name="Mode",
                    tooltip=(
                        "How the index is searched. vector: default, mmr. summary: default, embedding, "
                        "llm. keyword: default, simple, rake. tree: select_leaf, select_leaf_embedding, "
                        "root, all_leaf. document summary: default, embedding, llm. graph: default, "
                        "graph_synonyms, graph_vector. image: default = text and pictures, pictures, text."
                    ),
                ),
                io.String.Input(
                    "metadata_filters", multiline=True, default="",
                    tooltip=(
                        "One filter per line: `file_type = application/pdf`, `page_label != 1`, "
                        "`file_size > 20000`, `file_name in a.txt, b.txt`, `tags contains draft`. "
                        "Quote a value to compare it as text. Blank = no filter."
                    ),
                ),
                io.Combo.Input(
                    "filters_match", options=["all", "any"], default="all", optional=True, advanced=True,
                    tooltip="`all` = a chunk passes every filter line; `any` = passing one line is enough.",
                ),
                LLM.Input(
                    "llm", optional=True,
                    tooltip="A language model for the modes that call one (summary llm, keyword default, tree select_leaf, graph), in place of the index's own.",
                ),
                LLM_EMBEDDING.Input(
                    "embedding", optional=True,
                    tooltip="An embedding model for the modes that compare vectors, in place of the index's own. Use the model the index was built with.",
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
    def execute(cls, index, top_k, mode, metadata_filters="", filters_match="all", llm=None,
                embedding=None) -> io.NodeOutput:
        """Build the retriever.

        Args:
            index: The index wire.
            top_k: Chunks one search hands on.
            mode: The mode and its settings.
            metadata_filters: Filter lines.
            filters_match: ``all`` or ``any``.
            llm: A language model in place of the index's own.
            embedding: An embedding model in place of the index's own.

        Returns:
            The retriever.
        """
        node = "LLM Index Retriever"
        index = wires.expect(index, wires.Index, node, "index", "an LLM index node such as LLM Vector Index")
        if llm is not None:
            wires.expect(llm, wires.LanguageModel, node, "llm", "an LLM model node")
        if embedding is not None:
            wires.expect(embedding, wires.EmbeddingModel, node, "embedding", "LLM Embedding Model")
        wires.ready(index, llm, embedding)
        with runtime.guarded():
            retriever, facts = retrieval.index_retriever(
                index, mode, top_k, metadata_filters, filters_match, llm=llm, embedding=embedding
            )
        if index.name:
            facts["saved as"] = index.name
        logger.info("LLM Index Retriever built a %s retriever", retriever.label)
        run_result.publish(
            status=run_result.OK,
            summary=f"Retriever ready: {retriever.label}.",
            counts={"chunks in index": int(index.size), "top_k": int(top_k)},
            facts=facts,
        )
        return io.NodeOutput(retriever)

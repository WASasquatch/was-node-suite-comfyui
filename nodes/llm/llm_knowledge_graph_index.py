"""Read entities and relations out of documents into a knowledge graph index."""

from __future__ import annotations

import time

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM, LLM_DOCUMENTS, LLM_EMBEDDING, LLM_INDEX
from ...modules.llm import indexes, runtime, wires

REQUIRES = "llm"

logger = log.get_logger("llm.index")

#: The node's name in messages.
NODE = "LLM Knowledge Graph Index"


class LLMKnowledgeGraphIndex(io.ComfyNode):
    """Build a graph of entities and relations from documents."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMKnowledgeGraphIndex",
            display_name="LLM Knowledge Graph Index",
            search_aliases=[
                "WASLLMKnowledgeGraphIndex",
                "LLM Knowledge Graph Index",
                "PropertyGraphIndex",
                "knowledge graph",
                "property graph",
                "triplets",
                "GraphRAG",
                "RAG",
            ],
            category="WAS Suite/LLM/Index",
            description=(
                "Read entities and the relations between them out of documents into a knowledge "
                "graph, for questions about how things connect. Connect an embedding model to "
                "search the graph by meaning too. The panel lists the triplets found."
            ),
            inputs=[
                LLM_DOCUMENTS.Input(
                    "documents",
                    tooltip=(
                        "What to read, such as from LLM Load Documents, or chunks from LLM Split "
                        "Documents. Whole documents are cut into chunks of chunk_size tokens."
                    ),
                ),
                io.DynamicCombo.Input(
                    "extractor",
                    options=[
                        io.DynamicCombo.Option(key="simple", inputs=[
                            io.Int.Input(
                                "max_paths_per_chunk", default=10, min=1, max=100,
                                tooltip="Triplets the model may name in one chunk. 10 = the main facts; 30 = minor ones too.",
                            ),
                        ]),
                        io.DynamicCombo.Option(key="implicit", inputs=[]),
                        io.DynamicCombo.Option(key="dynamic", inputs=[
                            io.Int.Input(
                                "max_triplets_per_chunk", default=10, min=1, max=100,
                                tooltip="Triplets the model may name in one chunk. 10 = the main facts; 30 = minor ones too.",
                            ),
                            io.String.Input(
                                "entity_types", default="",
                                placeholder="Eg: PERSON, PLACE, ORGANIZATION",
                                tooltip=(
                                    "Kinds of entity to start the model with, separated by commas, "
                                    "such as `PERSON, PLACE, ANIMAL`. The model may add others. "
                                    "Empty lets it choose."
                                ),
                            ),
                            io.String.Input(
                                "relation_types", default="",
                                placeholder="Eg: LIVES_IN, WORKS_FOR, PART_OF",
                                tooltip=(
                                    "Kinds of relation to start the model with, separated by "
                                    "commas, such as `LIVES_IN, HUNTS`. The model may add others. "
                                    "Empty lets it choose."
                                ),
                            ),
                        ]),
                    ],
                    tooltip=(
                        "simple = the model names (subject, relation, object) triplets; implicit = "
                        "the links already between chunks, such as next and source, with no model; "
                        "dynamic = the model names typed entities and relations, open schema."
                    ),
                ),
                LLM.Input(
                    "llm", optional=True,
                    tooltip=(
                        "The model that names the triplets, one generation per chunk, for the "
                        "simple and dynamic extractors. Graph search by synonyms asks it later too."
                    ),
                ),
                LLM_EMBEDDING.Input(
                    "embedding", optional=True,
                    tooltip=(
                        "A model that embeds every entity and chunk, such as from LLM Embedding "
                        "Model, so the graph can be searched by meaning. Left empty, it is searched "
                        "by keywords only."
                    ),
                ),
                io.Int.Input(
                    "chunk_size", default=1024, min=32, max=8192, optional=True, advanced=True,
                    tooltip=(
                        "Tokens in each chunk when whole documents are cut here. 1024 = a few "
                        "paragraphs; 256 = more chunks and more generations. Chunks from LLM "
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
                    tooltip="The knowledge graph, for LLM Index Retriever, a query engine or LLM Save Index.",
                ),
            ],
        )

    @classmethod
    async def execute(
        cls, documents, extractor, llm=None, embedding=None, chunk_size=1024, chunk_overlap=20,
    ) -> io.NodeOutput:
        """Split the documents and read their triplets into a graph.

        Args:
            documents: The documents or chunks.
            extractor: The extractor choice and its settings.
            llm: The language model naming triplets, or ``None``.
            embedding: The embedding model, or ``None``.
            chunk_size: Tokens per chunk for whole documents.
            chunk_overlap: Tokens neighbouring chunks share.

        Returns:
            The knowledge graph index.
        """
        docs = wires.expect(documents, wires.Documents, NODE, "documents", "LLM Load Documents")
        model = None
        if llm is not None:
            model = wires.expect(llm, wires.LanguageModel, NODE, "llm", "LLM From CLIP, LLM GGUF Loader or LLM OpenAI")
        embed = None
        if embedding is not None:
            embed = wires.expect(embedding, wires.EmbeddingModel, NODE, "embedding", "LLM Embedding Model")
        wires.ready(model, embed)
        settings = indexes.graph_settings(extractor)
        started = time.perf_counter()
        with runtime.guarded(), indexes.failing(NODE):
            prepared = indexes.prepare(docs, NODE, chunk_size, chunk_overlap)
            index = await indexes.build_graph(prepared, model, embed, settings, NODE)
        wire = wires.Index(
            index=index, kind="graph", embedding=embed, llm=model, size=indexes.size_of(index, "graph"),
        )
        seconds = time.perf_counter() - started
        triplets = len(index.property_graph_store.graph.triplets)
        logger.info(
            "%s found %d triplet(s) in %d chunk(s) with the %s extractor in %.2fs",
            NODE, triplets, wire.size, settings["extractor"], seconds,
        )
        indexes.report(
            wire, seconds, f"found {triplets} triplets in {wire.size} chunks ({settings['extractor']})", prepared,
        )
        return io.NodeOutput(wire)

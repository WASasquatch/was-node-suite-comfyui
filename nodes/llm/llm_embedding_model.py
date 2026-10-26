"""Pick the embedding model the LLM index and retrieval nodes turn text into vectors with."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import LLM_EMBEDDING
from ...modules.interface import run_result
from ...modules.llm import embeddings, providers, runtime

REQUIRES = "llm"

#: The node's name in messages.
NODE = "LLM Embedding Model"

#: The longest text, in tokens, one vector reads.
MAX_LENGTH = 32768

#: The most texts embedded together.
MAX_BATCH = 1024

#: The largest vector length a hosted service may be asked for.
MAX_DIMENSIONS = 8192

#: Where each source's vectors are made, for the panel.
SOURCES = {
    "local": "a Hugging Face folder on this machine",
    "gguf": "a GGUF file on this machine",
    "openai": "OpenAI",
    "gemini": "Google Gemini",
    "provider": "a provider",
}

#: Pooling, normalisation, instruction and length inputs both local sources carry.
LOCAL_INPUTS = [
    io.Combo.Input(
        "pooling", options=embeddings.POOLINGS, default="auto",
        tooltip=(
            "How token vectors become one vector: `auto` = what the model declares or its family "
            "uses; `cls` = the first token, as BGE; `mean` = the average, as e5 and MiniLM; "
            "`last token` = the final token, as Qwen3-Embedding."
        ),
    ),
    io.Boolean.Input(
        "normalize", default=True,
        tooltip=(
            "`true` scales every vector to length 1, which cosine search expects; `false` keeps "
            "the model's raw lengths."
        ),
    ),
    io.String.Input(
        "query_instruction", default="",
        tooltip=(
            "Text placed directly before each search query. Blank = the family's own, such as "
            "`query: ` for e5 or BGE's search sentence; end a typed one with a space."
        ),
    ),
    io.String.Input(
        "text_instruction", default="",
        tooltip=(
            "Text placed directly before each document chunk. Blank = the family's own, such as "
            "`passage: ` for e5; most families use none."
        ),
    ),
    io.Int.Input(
        "max_length", default=512, min=8, max=MAX_LENGTH,
        tooltip=(
            "The most tokens one text is read for; the rest is cut off. `512` suits BGE, e5 and "
            "MiniLM; Qwen3-Embedding reads up to 32768."
        ),
    ),
]


class LLMEmbeddingModel(io.ComfyNode):
    """An embedding model on the LLM embedding wire."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        sources = [
            io.DynamicCombo.Option(
                key="local",
                inputs=[
                    io.Combo.Input(
                        "model", options=embeddings.local_choices(),
                        default=embeddings.SUGGESTED[0],
                        tooltip=(
                            "A Hugging Face folder in models/LLM, or a suggested model such as "
                            "`BAAI/bge-small-en-v1.5`, fetched into models/LLM on first use when "
                            "features.network is on. Safetensors weights only."
                        ),
                    ),
                    *LOCAL_INPUTS,
                    io.Int.Input(
                        "batch_size", default=32, min=1, max=MAX_BATCH,
                        tooltip=(
                            "Texts embedded together. `32` suits most cards; lower it if memory "
                            "runs out on long chunks."
                        ),
                    ),
                ],
            ),
            io.DynamicCombo.Option(
                key="gguf",
                inputs=[
                    io.Combo.Input(
                        "model", options=embeddings.gguf_choices(),
                        tooltip=(
                            "A .gguf embedding model in models/LLM or models/text_encoders, such "
                            "as a bge or nomic-embed GGUF, read through llama.cpp."
                        ),
                    ),
                    *LOCAL_INPUTS,
                ],
            ),
            io.DynamicCombo.Option(
                key="openai",
                inputs=[
                    io.String.Input(
                        "model", default="text-embedding-3-small",
                        tooltip=(
                            "The OpenAI embedding model: `text-embedding-3-small` = 1536 numbers, "
                            "cheap; `text-embedding-3-large` = 3072, more precise. Needs "
                            "OPENAI_API_KEY and features.network."
                        ),
                    ),
                    io.Int.Input(
                        "dimensions", default=0, min=0, max=MAX_DIMENSIONS,
                        tooltip=(
                            "Vector length to ask for: `0` = the model's own; `512` = shorter "
                            "vectors, a smaller index and slightly less precision."
                        ),
                    ),
                ],
            ),
            io.DynamicCombo.Option(
                key="gemini",
                inputs=[
                    io.String.Input(
                        "model", default="gemini-embedding-001",
                        tooltip=(
                            "The Gemini embedding model, such as `gemini-embedding-001`. Needs "
                            "GOOGLE_API_KEY or GEMINI_API_KEY and features.network."
                        ),
                    ),
                    io.Int.Input(
                        "dimensions", default=0, min=0, max=MAX_DIMENSIONS,
                        tooltip=(
                            "Vector length to ask for: `0` = the model's own, 3072 for "
                            "gemini-embedding-001; `768` = a smaller index."
                        ),
                    ),
                ],
            ),
            io.DynamicCombo.Option(
                key="provider",
                inputs=[
                    io.Combo.Input(
                        "provider", options=providers.options(), default="ollama",
                        tooltip=(
                            "Where the model runs, the same list LLM Provider offers: hosted "
                            "services read their own key variable; `ollama`, `lmstudio`, `vllm` "
                            "and `llama.cpp` are servers on this machine."
                        ),
                    ),
                    io.String.Input(
                        "model", default="",
                        tooltip=(
                            "The embedding model id, such as `nomic-embed-text` on Ollama or "
                            "`mistral-embed` on Mistral. Empty = the provider's default, where it "
                            "has one."
                        ),
                    ),
                    io.Int.Input(
                        "dimensions", default=0, min=0, max=MAX_DIMENSIONS,
                        tooltip=(
                            "Vector length to ask for: `0` = the model's own. Only some servers "
                            "take another length."
                        ),
                    ),
                ],
            ),
        ]
        return io.Schema(
            node_id="WASLLMEmbeddingModel",
            display_name="LLM Embedding Model",
            search_aliases=[
                "WASLLMEmbeddingModel",
                "LLM Embedding Model",
                "embedding",
                "embeddings",
                "BaseEmbedding",
                "HuggingFaceEmbedding",
                "OpenAIEmbedding",
                "OllamaEmbedding",
                "bge",
                "vector",
                "RAG",
            ],
            category="WAS Suite/LLM/Models",
            description=(
                "Choose the model that turns text into vectors for indexes and retrieval: a Hugging "
                "Face folder in models/LLM, a .gguf file, or OpenAI, Gemini and other services. "
                "Build an index and search it with the same model. The panel shows the vector "
                "length, device and pooling."
            ),
            inputs=[
                io.DynamicCombo.Input(
                    "source", options=sources,
                    tooltip=(
                        "Where the vectors are made: `local` = a Hugging Face folder on this "
                        "machine; `gguf` = a .gguf file through llama.cpp; `openai` and `gemini` = "
                        "those services; `provider` = another service or a local server."
                    ),
                ),
            ],
            outputs=[
                LLM_EMBEDDING.Output(
                    display_name="embedding",
                    tooltip=(
                        "The embedding model, for the LLM index, retriever and other nodes that "
                        "search text by meaning."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, source) -> io.NodeOutput:
        """Build the embedding model; a local one is loaded and measured.

        Args:
            source: The source choice and its settings.

        Returns:
            The embedding model.

        Raises:
            ValueError: A setting is not valid, or the model is not on this machine.
            RuntimeError: A hosted source with ``features.network`` off or its key unset, or a
                local model that would not load.
        """
        chosen = source if isinstance(source, dict) else {"source": str(source)}
        kind = str(chosen.get("source") or "")
        if kind not in SOURCES:
            raise ValueError(f"{NODE} does not know the source `{kind[:40]}`. Pick one of {', '.join(SOURCES)}.")
        bounded = embeddings.bounded
        dimensions = bounded(chosen.get("dimensions"), 0, 0, MAX_DIMENSIONS)
        details: dict = {}
        host = ""
        with runtime.guarded():
            if kind in ("local", "gguf"):
                settings = {
                    "pooling": str(chosen.get("pooling") or "auto"),
                    "normalize": bool(chosen.get("normalize", True)),
                    "query_instruction": str(chosen.get("query_instruction") or ""),
                    "text_instruction": str(chosen.get("text_instruction") or ""),
                    "max_length": bounded(chosen.get("max_length"), 512, 8, MAX_LENGTH),
                    "node": NODE,
                }
                if kind == "local":
                    batch = bounded(chosen.get("batch_size"), 32, 1, MAX_BATCH)
                    built, details = embeddings.local_embedding(
                        chosen.get("model"), batch_size=batch, **settings,
                    )
                else:
                    built, details = embeddings.gguf_embedding(chosen.get("model"), **settings)
            elif kind == "openai":
                built = embeddings.openai_embedding(chosen.get("model"), dimensions, node=NODE)
                host = "api.openai.com"
            elif kind == "gemini":
                built = embeddings.gemini_embedding(chosen.get("model"), dimensions, node=NODE)
                host = providers.GEMINI_HOST
            else:
                built, target = embeddings.provider_embedding(
                    chosen.get("provider"), chosen.get("model"), dimensions, node=NODE,
                )
                host = target.host
        facts = {
            "model": built.label,
            "source": SOURCES[kind],
            "dimensions": str(built.dimensions) if built.dimensions else "the model's own",
            "text goes": "nowhere, it stays on this machine" if not host
            else f"to {host}" + (", on this machine" if built.local else ", off this machine"),
        }
        if details:
            facts["device"] = details["device"]
            facts["pooling"] = details["pooling"]
            facts["normalized"] = "yes" if chosen.get("normalize", True) else "no"
            used = [name for name in ("query", "text") if details[name]]
            facts["instructions"] = " and ".join(used) if used else "none"
            summary = f"{built.label} makes vectors of {built.dimensions} numbers on {details['device']}."
        elif built.local:
            summary = f"{built.label} is ready. Text stays on this machine."
        else:
            summary = f"{built.label} is ready. Text is sent to {host} to embed."
        run_result.publish(
            status=run_result.OK,
            summary=summary,
            counts={"dimensions": built.dimensions},
            facts=facts,
        )
        return io.NodeOutput(built)

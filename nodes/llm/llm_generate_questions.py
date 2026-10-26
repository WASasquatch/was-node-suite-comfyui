"""Write test questions from documents with a language model."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM, LLM_DOCUMENTS
from ...modules.interface import run_result
from ...modules.llm import evaluation, runtime, wires

REQUIRES = "llm"

logger = log.get_logger("llm.evaluate")


class WASLLMGenerateQuestions(io.ComfyNode):
    """Questions a model writes about each chunk of the documents, one per line."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMGenerateQuestions",
            display_name="LLM Generate Questions",
            search_aliases=[
                "WASLLMGenerateQuestions",
                "LLM Generate Questions",
                "DatasetGenerator",
                "question generation",
                "test questions",
                "eval dataset",
            ],
            category="WAS Suite/LLM/Evaluate",
            is_output_node=True,
            description=(
                "Have a model write questions that the documents answer, a few per chunk, to "
                "test a query engine or an agent with. Outputs the questions one per line and "
                "how many were written."
            ),
            inputs=[
                LLM_DOCUMENTS.Input(
                    "documents",
                    tooltip="The documents or chunks to write questions about, such as from LLM Load Documents. Whole documents are split into chunks of about 1024 tokens first.",
                ),
                LLM.Input(
                    "llm",
                    tooltip="The model that writes the questions, such as LLM OpenAI or LLM GGUF Loader.",
                ),
                io.Int.Input(
                    "questions_per_chunk", default=2, min=1, max=10,
                    tooltip="Questions asked for per chunk. `2` covers a document lightly; `5` tests it closely.",
                ),
                io.Int.Input(
                    "max_chunks", default=20, min=1, max=500,
                    tooltip="Chunks read at most, spread evenly across the documents. One model call per chunk, so `20` is twenty calls.",
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="questions",
                    tooltip="The questions, one per line, such as for a For Each loop feeding LLM Query.",
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="How many questions were written.",
                ),
            ],
        )

    @classmethod
    async def execute(cls, documents, llm, questions_per_chunk, max_chunks) -> io.NodeOutput:
        """Write the questions.

        Args:
            documents: The documents or chunks.
            llm: The model that writes.
            questions_per_chunk: Questions per chunk.
            max_chunks: Chunks read at most.

        Returns:
            The questions and their count.
        """
        node = "LLM Generate Questions"
        found = wires.expect(documents, wires.Documents, node, "documents", "LLM Load Documents")
        model = wires.expect(llm, wires.LanguageModel, node, "llm", "LLM OpenAI or LLM From CLIP")
        wires.ready(model)
        stream = runtime.TextStream(f"questions by {model.label}")
        with runtime.guarded():
            chunks, level = evaluation.chunks_for(found, int(max_chunks))
            if not chunks:
                raise ValueError(f"{node}: the documents hold no text to write questions about. Check what LLM Load Documents read.")
            try:
                pairs = await evaluation.write_questions(chunks, model.llm, int(questions_per_chunk), stream)
            except Exception as error:
                raise RuntimeError(f"{node} failed: {runtime.describe_error(error)}") from None
            finally:
                stream.finish()
        questions = [question for question, _chunk in pairs]
        text = "\n".join(questions)
        logger.info("LLM Generate Questions wrote %d question(s) from %d chunk(s)", len(questions), len(chunks))
        run_result.publish(
            status=run_result.OK if questions else run_result.WARNING,
            summary=(
                f"{len(questions)} question{'s' if len(questions) != 1 else ''} from {len(chunks)} chunk{'s' if len(chunks) != 1 else ''}"
                if questions else "The model wrote no questions it could read. Try a stronger model."
            ),
            counts={"questions": len(questions), "chunks": len(chunks)},
            facts={"model": model.label, "chunks from": "the chunks given" if level == "chunk" else "documents split here"},
            bodies=[run_result.body("questions", text)],
        )
        return io.NodeOutput(text, len(questions))

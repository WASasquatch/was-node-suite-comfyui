"""Measure how often a retriever finds the passage a question was written from."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM, LLM_DOCUMENTS, LLM_RETRIEVER
from ...modules.interface import run_result
from ...modules.llm import evaluation, runtime, wires

REQUIRES = "llm"

logger = log.get_logger("llm.evaluate")


class WASLLMEvaluateRetriever(io.ComfyNode):
    """Writes questions from the documents, asks the retriever each one, and scores what it finds."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMEvaluateRetriever",
            display_name="LLM Evaluate Retriever",
            search_aliases=[
                "WASLLMEvaluateRetriever",
                "LLM Evaluate Retriever",
                "RetrieverEvaluator",
                "hit rate",
                "MRR",
                "retrieval benchmark",
            ],
            category="WAS Suite/LLM/Evaluate",
            is_output_node=True,
            description=(
                "Test a retriever: a model writes questions from each chunk of the documents, the "
                "retriever is asked each one, and the node scores how often and how high it "
                "returns the chunk the question came from."
            ),
            inputs=[
                LLM_RETRIEVER.Input(
                    "retriever",
                    tooltip="The retriever to test, built over an index of the same documents.",
                ),
                LLM_DOCUMENTS.Input(
                    "documents",
                    tooltip="The documents the index was built from. Connect the same chunks the index holds to score by chunk, or the whole documents to score by document.",
                ),
                LLM.Input(
                    "llm",
                    tooltip="The model that writes the test questions, such as LLM OpenAI or LLM GGUF Loader.",
                ),
                io.Int.Input(
                    "questions_per_chunk", default=2, min=1, max=10,
                    tooltip="Questions written per chunk. `2` is a quick test; `5` a closer one.",
                ),
                io.Int.Input(
                    "max_chunks", default=20, min=1, max=500,
                    tooltip="Chunks questions are written from, spread evenly across the documents. One model call per chunk, so `20` is twenty calls.",
                ),
                io.Boolean.Input(
                    "precision", default=False,
                    tooltip="`true` adds precision to the report: the share of what was retrieved that is the source.",
                ),
                io.Boolean.Input(
                    "recall", default=False,
                    tooltip="`true` adds recall to the report: whether the source was retrieved at all, per question.",
                ),
                io.Boolean.Input(
                    "average_precision", default=False,
                    tooltip="`true` adds average precision to the report, which rewards finding the source near the top.",
                ),
                io.Boolean.Input(
                    "ndcg", default=False,
                    tooltip="`true` adds NDCG to the report, which scores how close to first the source was ranked.",
                ),
            ],
            outputs=[
                io.Float.Output(
                    display_name="hit_rate",
                    tooltip="The share of questions whose source was retrieved: `1` = always, `0.5` = half the time. Always measured.",
                ),
                io.Float.Output(
                    display_name="mrr",
                    tooltip="Mean reciprocal rank: `1` = the source came first every time, `0.5` = second on average. Always measured.",
                ),
                io.String.Output(
                    display_name="report",
                    tooltip="Every measure, how the source was matched, and the questions whose source was not found.",
                ),
            ],
        )

    @classmethod
    async def execute(cls, retriever, documents, llm, questions_per_chunk, max_chunks, precision, recall, average_precision, ndcg) -> io.NodeOutput:
        """Write questions, retrieve for each and score the results.

        Args:
            retriever: The retriever to test.
            documents: The documents the index holds.
            llm: The model that writes the questions.
            questions_per_chunk: Questions per chunk.
            max_chunks: Chunks read at most.
            precision: Whether to report precision.
            recall: Whether to report recall.
            average_precision: Whether to report average precision.
            ndcg: Whether to report NDCG.

        Returns:
            The hit rate, the MRR and the report.
        """
        node = "LLM Evaluate Retriever"
        found = wires.expect(retriever, wires.Retriever, node, "retriever", "an LLM retriever node")
        pages = wires.expect(documents, wires.Documents, node, "documents", "LLM Load Documents")
        model = wires.expect(llm, wires.LanguageModel, node, "llm", "LLM OpenAI or LLM From CLIP")
        wires.ready(found, model)
        chosen = {"precision": precision, "recall": recall, "ap": average_precision, "ndcg": ndcg}
        metric_names = ["hit_rate", "mrr", *[name for name, wanted in chosen.items() if wanted]]
        words = {name: word for word, name in evaluation.METRICS}
        stream = runtime.TextStream(f"questions by {model.label}")
        with runtime.guarded():
            chunks, level = evaluation.chunks_for(pages, int(max_chunks))
            if not chunks:
                raise ValueError(f"{node}: the documents hold no text to write questions about. Check what LLM Load Documents read.")
            try:
                pairs = await evaluation.write_questions(chunks, model.llm, int(questions_per_chunk), stream)
                if not pairs:
                    raise ValueError("the model wrote no questions that could be read; try a stronger model")
                stream.note(f"asking {found.label} {len(pairs)} questions")
                scores = await evaluation.score_retriever(found.retriever, pairs, level, metric_names)
            except ValueError as error:
                raise ValueError(f"{node}: {error}") from None
            except Exception as error:
                raise RuntimeError(f"{node} failed: {runtime.describe_error(error)}") from None
            finally:
                stream.finish()

        hit_rate = float(scores.means.get("hit_rate", 0.0))
        mrr = float(scores.means.get("mrr", 0.0))
        matched = "by chunk id" if level == "chunk" else "by the document each chunk came from"
        lines = [
            f"Retriever: {found.label}",
            f"Questions: {scores.questions} from {len(chunks)} chunks, matched {matched}",
            "",
            *[f"{words.get(name, name)}: {scores.means.get(name, 0.0):.4f}" for name in metric_names],
        ]
        missed = [question for question, hit, _rank in scores.rows if hit <= 0.0]
        if missed:
            lines += ["", f"Not found ({len(missed)}):", *[f"- {question}" for question in missed[:50]]]
            if len(missed) > 50:
                lines.append(f"- and {len(missed) - 50} more")
        if scores.questions and hit_rate == 0.0:
            lines += ["", "No retrieved passage matched any source. Connect the same documents the index was built from."]
        report = "\n".join(lines)
        logger.info(
            "LLM Evaluate Retriever asked %d question(s): hit rate %.3f, MRR %.3f",
            scores.questions, hit_rate, mrr,
        )
        run_result.publish(
            status=run_result.OK if hit_rate > 0.0 else run_result.WARNING,
            summary=f"hit rate {hit_rate:.2f}, MRR {mrr:.2f} over {scores.questions} questions",
            counts={words.get(name, name): round(float(scores.means.get(name, 0.0)), 4) for name in metric_names},
            facts={"retriever": found.label[:120], "questions by": model.label, "matched": matched},
            items=[f"not found: {question}" for question in missed[:8]],
            items_total=len(missed),
            bodies=[run_result.body("report", report)],
        )
        return io.NodeOutput(hit_rate, mrr, report)

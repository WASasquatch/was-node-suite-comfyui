"""Score an answer with a judge model or an embedding comparison."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM, LLM_EMBEDDING, LLM_SOURCES
from ...modules.interface import run_result
from ...modules.llm import evaluation, runtime, wires

REQUIRES = "llm"

logger = log.get_logger("llm.evaluate")


class WASLLMEvaluate(io.ComfyNode):
    """Judges one answer and reports a score, a pass and the judge's reasoning."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        kinds = [
            io.DynamicCombo.Option(key="faithfulness", inputs=[]),
            io.DynamicCombo.Option(key="relevancy", inputs=[]),
            io.DynamicCombo.Option(
                key="correctness",
                inputs=[
                    io.String.Input(
                        "reference_answer", multiline=True, default="",
                        tooltip="A known good answer to compare with, such as `The R5 battery lasts about 320 shots.` Empty lets the judge rely on its own knowledge.",
                    ),
                    io.Float.Input(
                        "threshold", default=4.0, min=1.0, max=5.0, step=0.5,
                        tooltip="The score that passes, on the judge's scale of 1 to 5. `4` = relevant and fully correct; `2` passes answers with mistakes.",
                    ),
                ],
            ),
            io.DynamicCombo.Option(
                key="answer relevancy",
                inputs=[
                    io.Float.Input(
                        "threshold", default=0.5, min=0.0, max=1.0, step=0.5,
                        tooltip="The score that passes. `0.5` = on the subject; `1` = on the subject and on the question's angle.",
                    ),
                ],
            ),
            io.DynamicCombo.Option(
                key="context relevancy",
                inputs=[
                    io.Float.Input(
                        "threshold", default=0.5, min=0.0, max=1.0, step=0.05,
                        tooltip="The score that passes. `1` = the passages alone answer the question in full; `0.5` = on the subject but only partly enough.",
                    ),
                ],
            ),
            io.DynamicCombo.Option(
                key="guideline",
                inputs=[
                    io.String.Input(
                        "guidelines", multiline=True,
                        default="The answer should fully answer the question, be accurate, and avoid vague language.",
                        tooltip="The rules the answer is held to, such as `The answer must be under 50 words and name its source.`",
                    ),
                ],
            ),
            io.DynamicCombo.Option(
                key="semantic similarity",
                inputs=[
                    io.String.Input(
                        "reference_answer", multiline=True, default="",
                        tooltip="The known good answer the answer is compared with by meaning, such as `About 320 shots per charge.`",
                    ),
                    io.Float.Input(
                        "threshold", default=0.8, min=0.0, max=1.0, step=0.01,
                        tooltip="The similarity that passes. `1` = the same meaning; `0.8` = close; below `0.5` = mostly different.",
                    ),
                ],
            ),
            io.DynamicCombo.Option(
                key="pairwise",
                inputs=[
                    io.String.Input(
                        "second_answer", multiline=True, default="",
                        tooltip="Another answer to the same question, such as from a second model. Score `1` = answer is better; `0` = second_answer is; `0.5` = a tie.",
                    ),
                ],
            ),
        ]
        return io.Schema(
            node_id="WASLLMEvaluate",
            display_name="LLM Evaluate",
            search_aliases=[
                "WASLLMEvaluate",
                "LLM Evaluate",
                "FaithfulnessEvaluator",
                "RelevancyEvaluator",
                "CorrectnessEvaluator",
                "GuidelineEvaluator",
                "SemanticSimilarityEvaluator",
                "PairwiseComparisonEvaluator",
                "judge",
                "hallucination check",
            ],
            category="WAS Suite/LLM/Evaluate",
            is_output_node=True,
            description=(
                "Score an answer: whether its sources back it up, whether it fits the question, how "
                "correct it is, whether it follows guidelines, how close it is to a reference, or "
                "whether it beats a second answer. Outputs the score, a pass and the judge's reasoning."
            ),
            inputs=[
                io.DynamicCombo.Input(
                    "kind", options=kinds, display_name="Kind",
                    tooltip=(
                        "`faithfulness` = the sources back the answer; `relevancy` = answer and sources fit the "
                        "question; `correctness` = 1 to 5 against a reference; `answer relevancy` / `context "
                        "relevancy` = answer or sources on topic; `guideline` = follows rules; `semantic similarity` = meaning matches."
                    ),
                ),
                io.String.Input(
                    "question", multiline=True, default="",
                    tooltip="What was asked, such as `How long does the R5 battery last?` Empty takes the question the connected sources were retrieved for.",
                ),
                io.String.Input(
                    "answer", multiline=True, default="",
                    tooltip="The answer to judge, such as LLM Query's answer or `About 320 shots per charge.` Empty takes the answer the connected sources carry.",
                ),
                LLM.Input(
                    "llm", optional=True,
                    tooltip="The judge model. A strong model judges best, such as Claude or GPT. Every kind but semantic similarity needs it.",
                ),
                LLM_SOURCES.Input(
                    "sources", optional=True,
                    tooltip="The passages the answer was written from, from LLM Query or LLM Run Agent. Faithfulness, relevancy and context relevancy need them.",
                ),
                LLM_EMBEDDING.Input(
                    "embedding", optional=True,
                    tooltip="The embedding model semantic similarity compares with, from LLM Embedding Model.",
                ),
            ],
            outputs=[
                io.Float.Output(
                    display_name="score",
                    tooltip="The score: 0 or 1 for faithfulness, relevancy and guideline; 1 to 5 for correctness; 0 to 1 for the rest. 0 when the judge's answer could not be read.",
                ),
                io.Boolean.Output(
                    display_name="passing",
                    tooltip="`true` when the answer passed, such as to branch with a switch or to retry a failed answer.",
                ),
                io.String.Output(
                    display_name="feedback",
                    tooltip="The judge's reasoning, or the similarity figure for semantic similarity.",
                ),
            ],
        )

    @classmethod
    async def execute(cls, kind, question, answer, llm=None, sources=None, embedding=None) -> io.NodeOutput:
        """Judge the answer.

        Args:
            kind: What to judge and its settings.
            question: What was asked.
            answer: What was answered.
            llm: The judge model.
            sources: The passages the answer came from.
            embedding: The model semantic similarity compares with.

        Returns:
            The score, whether it passed, and the feedback.
        """
        node = "LLM Evaluate"
        choice = dict(kind or {})
        which = str(choice.get("kind", "faithfulness"))
        if which not in evaluation.KINDS:
            raise ValueError(f"{node}: kind {which[:40]!r} is not one of {', '.join(evaluation.KINDS)}.")
        found = None if sources is None else wires.expect(sources, wires.Sources, node, "sources", "LLM Query")
        asked = str(question or "").strip() or (found.query if found is not None else "")
        said = str(answer or "").strip() or (found.answer if found is not None else "")
        judge = None
        if which in evaluation.NEEDS_JUDGE:
            judge = wires.expect(llm, wires.LanguageModel, node, "llm", "a judge model such as LLM Anthropic or LLM OpenAI")
        embedder = None
        if which == "pairwise" and not str(choice.get("second_answer", "") or "").strip():
            raise ValueError(f"{node}: pairwise compares the answer with a second answer, and second_answer is empty. Wire or type the other answer.")
        if which == "semantic similarity":
            embedder = wires.expect(embedding, wires.EmbeddingModel, node, "embedding", "LLM Embedding Model")
            if not str(choice.get("reference_answer", "") or "").strip():
                raise ValueError(f"{node}: semantic similarity compares the answer with a reference answer, and reference_answer is empty. Type the known good answer.")
        contexts = found.contexts if found is not None else []
        if which in evaluation.NEEDS_CONTEXTS and not contexts:
            raise ValueError(f"{node}: {which} reads the passages the answer came from, and none are connected. Connect LLM Query's sources output to the sources input.")
        if which not in ("faithfulness", "semantic similarity") and not asked:
            raise ValueError(f"{node}: {which} needs the question. Type it into question, or connect sources that carry it.")
        if which != "context relevancy" and not said:
            raise ValueError(f"{node}: there is no answer to judge. Type it into answer, or connect sources that carry it.")
        wires.ready(judge, embedder)
        with runtime.guarded():
            try:
                verdict = await evaluation.judge(
                    which, choice, judge.llm if judge is not None else None, asked, said, contexts,
                    embedder.model if embedder is not None else None,
                )
            except Exception as error:
                raise RuntimeError(f"{node} failed: {runtime.describe_error(error)}") from None
        logger.info("LLM Evaluate judged %s: score %.3f, passing %s", which, verdict.score, verdict.passing)
        facts = {"kind": which, "judge": judge.label if judge is not None else embedder.label}
        if "threshold" in choice:
            facts["threshold"] = str(choice["threshold"])
        if contexts and which in evaluation.NEEDS_CONTEXTS:
            facts["passages"] = str(min(len(contexts), evaluation.MAX_CONTEXTS))
        run_result.publish(
            status=run_result.OK if verdict.readable else run_result.WARNING,
            summary=(
                f"{which}: {verdict.score:.3g}, {'passed' if verdict.passing else 'did not pass'}"
                if verdict.readable else f"{which}: the judge's answer could not be read"
            ),
            counts={"score": round(verdict.score, 4), "passing": int(verdict.passing)},
            facts=facts,
            bodies=[run_result.body("feedback", verdict.feedback)],
        )
        return io.NodeOutput(verdict.score, verdict.passing, verdict.feedback)

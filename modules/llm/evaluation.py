"""Judging answers and retrievers with llama-index's evaluators.

:func:`judge` scores one answer, :func:`write_questions` writes questions from chunks with
the chunk each came from, and :func:`score_retriever` measures how often a retriever finds it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .. import log
from . import runtime, wires

__all__ = [
    "KINDS",
    "METRICS",
    "NEEDS_CONTEXTS",
    "NEEDS_JUDGE",
    "RetrieverScores",
    "Verdict",
    "chunks_for",
    "judge",
    "score_retriever",
    "source_key",
    "write_questions",
]

logger = log.get_logger("llm.evaluate")

#: What LLM Evaluate judges, as the words on its menu.
KINDS = (
    "faithfulness",
    "relevancy",
    "correctness",
    "answer relevancy",
    "context relevancy",
    "guideline",
    "semantic similarity",
    "pairwise",
)

#: The kinds that read the retrieved chunks.
NEEDS_CONTEXTS = frozenset({"faithfulness", "relevancy", "context relevancy"})

#: The kinds that ask a language model.
NEEDS_JUDGE = frozenset(KINDS) - {"semantic similarity"}

#: The retrieval measures offered, as ``(menu word, llama-index metric name)``.
METRICS = (
    ("hit rate", "hit_rate"),
    ("MRR", "mrr"),
    ("precision", "precision"),
    ("recall", "recall"),
    ("average precision", "ap"),
    ("NDCG", "ndcg"),
)

#: Retrieved chunks a judge reads at most.
MAX_CONTEXTS = 32

#: Characters of one retrieved chunk a judge reads at most.
MAX_CONTEXT_CHARS = 20_000

#: Characters of a question, answer or reference handed to a judge at most.
MAX_TEXT_CHARS = 100_000

#: Characters of one written question kept at most.
MAX_QUESTION_CHARS = 1000

#: Characters of one chunk a question writer reads at most.
MAX_CHUNK_CHARS = 16_000

#: Tokens a chunk holds when whole documents arrive and are split.
CHUNK_SIZE = 1024

#: Tokens two neighbouring chunks share.
CHUNK_OVERLAP = 64

_evaluator: type | None = None

#: A numbering, bullet or label in front of a written question.
_LEAD = re.compile(r"^\s*(?:[-*+•]+|\(?\d{1,3}[.):]|q\d{0,3}[.):]|question\s*\d{0,3}[.):])\s*", re.IGNORECASE)


@dataclass
class Verdict:
    """One judgement.

    Attributes:
        score: The score, 0 where the judge's answer could not be read.
        passing: Whether the answer passed.
        feedback: The judge's reasoning, or a note that none could be read.
        readable: False when the judge's answer was not in the expected form.
    """

    score: float = 0.0
    passing: bool = False
    feedback: str = ""
    readable: bool = True


def _clip(text, limit: int) -> str:
    """Text cut to a number of characters."""
    text = str(text or "")
    return text if len(text) <= limit else text[:limit]


async def judge(kind: str, settings: dict, llm, question: str, answer: str, contexts, embedding=None) -> Verdict:
    """Judge one answer.

    Args:
        kind: One of :data:`KINDS`.
        settings: The kind's own settings: ``reference_answer``, ``threshold``, ``guidelines``.
        llm: The llama-index judge LLM, ``None`` for ``semantic similarity``.
        question: What was asked.
        answer: What was answered.
        contexts: Retrieved chunk texts, best first.
        embedding: The llama-index embedding model ``semantic similarity`` compares with.

    Returns:
        The :class:`Verdict`.

    Raises:
        ValueError: The kind is unknown.
    """
    runtime.core()
    from llama_index.core import evaluation

    question = _clip(question, MAX_TEXT_CHARS)
    answer = _clip(answer, MAX_TEXT_CHARS)
    chunks = [_clip(text, MAX_CONTEXT_CHARS) for text in list(contexts or ())[:MAX_CONTEXTS]]
    threshold = float(settings.get("threshold", 0.5))
    reference = _clip(settings.get("reference_answer", ""), MAX_TEXT_CHARS)

    if kind == "faithfulness":
        result = await evaluation.FaithfulnessEvaluator(llm=llm).aevaluate(query=question, response=answer, contexts=chunks)
    elif kind == "relevancy":
        result = await evaluation.RelevancyEvaluator(llm=llm).aevaluate(query=question, response=answer, contexts=chunks)
    elif kind == "correctness":
        evaluator = evaluation.CorrectnessEvaluator(llm=llm, score_threshold=threshold)
        result = await evaluator.aevaluate(query=question, response=answer, reference=reference or None)
    elif kind == "answer relevancy":
        result = await evaluation.AnswerRelevancyEvaluator(llm=llm).aevaluate(query=question, response=answer)
    elif kind == "context relevancy":
        result = await evaluation.ContextRelevancyEvaluator(llm=llm).aevaluate(query=question, contexts=chunks)
    elif kind == "guideline":
        guidelines = _clip(settings.get("guidelines", ""), MAX_TEXT_CHARS).strip() or None
        try:
            result = await evaluation.GuidelineEvaluator(llm=llm, guidelines=guidelines).aevaluate(query=question, response=answer)
        except ValueError as unreadable:
            return Verdict(
                feedback=f"The judge did not answer in the form asked for: {runtime.describe_error(unreadable)}",
                readable=False,
            )
    elif kind == "semantic similarity":
        evaluator = evaluation.SemanticSimilarityEvaluator(embed_model=embedding, similarity_threshold=threshold)
        result = await evaluator.aevaluate(response=answer, reference=reference)
    elif kind == "pairwise":
        second = _clip(settings.get("second_answer", ""), MAX_TEXT_CHARS)
        evaluator = evaluation.PairwiseComparisonEvaluator(llm=llm)
        result = await evaluator.aevaluate(query=question, response=answer, second_response=second)
        readable = not getattr(result, "invalid_result", False) and result.score is not None
        score = float(result.score) if result.score is not None else 0.0
        feedback = str(result.feedback or result.invalid_reason or "").strip()
        return Verdict(score=score, passing=readable and score > 0.5, feedback=feedback, readable=readable)
    else:
        raise ValueError(f"LLM Evaluate does not know the kind {kind!r}. Pick one of {', '.join(KINDS)}.")

    readable = not getattr(result, "invalid_result", False) and result.score is not None
    score = float(result.score) if result.score is not None else 0.0
    if result.passing is not None:
        passing = bool(result.passing)
    else:
        passing = readable and score >= threshold
    feedback = str(result.feedback or result.invalid_reason or "").strip()
    if not readable and not feedback:
        feedback = "The judge's answer held no score that could be read."
    return Verdict(score=score, passing=passing, feedback=feedback, readable=readable)


def chunks_for(documents: wires.Documents, max_chunks: int) -> tuple[list, str]:
    """The chunks questions are written from, spread evenly across the documents.

    Args:
        documents: The documents or chunks connected.
        max_chunks: Chunks taken at most.

    Returns:
        ``(chunks, level)``: ``level`` is ``chunk`` when chunks arrived, so a question's source
        is its chunk, and ``document`` when whole documents arrived and were split here.
    """
    runtime.core()
    from llama_index.core import Document
    from llama_index.core.node_parser import SentenceSplitter

    items = [item for item in documents.items if (getattr(item, "text", "") or "").strip()]
    if any(not isinstance(item, Document) for item in items):
        found, level = items, "chunk"
    else:
        splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        found, level = splitter.get_nodes_from_documents(items), "document"
    count = len(found)
    wanted = max(1, int(max_chunks))
    if count > wanted:
        found = [found[(index * count) // wanted] for index in range(wanted)]
    return found, level


def source_key(chunk, level: str) -> str:
    """The id a retrieved chunk has to carry for a question from ``chunk`` to count as found."""
    if level == "document":
        return str(getattr(chunk, "ref_doc_id", None) or chunk.node_id)
    return str(chunk.node_id)


def _questions(lines) -> list[str]:
    """Questions from the lines a model wrote, numbering and preamble removed."""
    kept = []
    for line in lines or ():
        text = _LEAD.sub("", str(line)).strip().strip("\"'").strip()
        text = " ".join(text.split())[:MAX_QUESTION_CHARS]
        if len(text) >= 8 and text not in kept:
            kept.append(text)
    asked = [text for text in kept if text.endswith("?")]
    return asked or kept


async def write_questions(chunks, llm, per_chunk: int, stream=None) -> list[tuple[str, object]]:
    """Questions about each chunk, written by a model.

    Args:
        chunks: llama-index nodes.
        llm: The llama-index LLM that writes them.
        per_chunk: Questions asked for per chunk.
        stream: A ``runtime.TextStream`` the questions are drawn on, or ``None``.

    Returns:
        ``(question, chunk)`` pairs in chunk order.

    Raises:
        InterruptProcessingException: The queue was cancelled.
    """
    runtime.core()
    from llama_index.core.evaluation.dataset_generation import DEFAULT_QUESTION_GENERATION_PROMPT
    from llama_index.core.prompts import PromptTemplate
    from llama_index.core.schema import MetadataMode

    template = PromptTemplate(DEFAULT_QUESTION_GENERATION_PROMPT)
    count = int(per_chunk)
    ask = (
        f"You are a Teacher/Professor. Your task is to setup {count} questions for an upcoming "
        f"quiz/examination. The questions should be diverse in nature across the document. "
        f"Restrict the questions to the context information provided."
    )
    pairs: list[tuple[str, object]] = []
    for index, chunk in enumerate(chunks):
        runtime.check_interrupt()
        if stream is not None:
            stream.note(f"chunk {index + 1} of {len(chunks)}")
        context = chunk.get_content(metadata_mode=MetadataMode.NONE)[:MAX_CHUNK_CHARS]
        written = await llm.apredict(template, context_str=context, query_str=ask)
        for question in _questions(str(written or "").strip().splitlines())[:count]:
            pairs.append((question, chunk))
            if stream is not None:
                stream.write(question + "\n")
    return pairs


@dataclass
class RetrieverScores:
    """How a retriever did over a set of questions.

    Attributes:
        means: Metric name to its mean over every question.
        rows: ``(question, hit, reciprocal rank)`` per question.
        questions: Questions asked.
    """

    means: dict = field(default_factory=dict)
    rows: list = field(default_factory=list)
    questions: int = 0


def _evaluator_class():
    """A retriever evaluator that can count a chunk from the right document as found."""
    global _evaluator
    if _evaluator is not None:
        return _evaluator
    from llama_index.core.evaluation import RetrieverEvaluator
    from llama_index.core.evaluation.retrieval.base import RetrievalEvalMode

    class SourceRetrieverEvaluator(RetrieverEvaluator):
        """Compares retrieved chunks with the expected source by chunk or by document."""

        document_level: bool = False

        async def _aget_retrieved_ids_and_texts(self, query, mode=RetrievalEvalMode.TEXT):
            found = await self.retriever.aretrieve(query)
            ids, texts, seen = [], [], set()
            for item in found:
                key = item.node.node_id
                if self.document_level:
                    key = getattr(item.node, "ref_doc_id", None) or key
                if key in seen:
                    continue
                seen.add(key)
                ids.append(str(key))
                texts.append(item.node.get_content())
            return ids, texts

    _evaluator = SourceRetrieverEvaluator
    return _evaluator


async def score_retriever(retriever, pairs, level: str, metric_names) -> RetrieverScores:
    """Ask a retriever every question and measure how often it finds the source.

    Args:
        retriever: The llama-index retriever.
        pairs: ``(question, chunk)`` pairs from :func:`write_questions`.
        level: ``chunk`` or ``document``, from :func:`chunks_for`.
        metric_names: llama-index metric names, such as ``hit_rate`` and ``mrr``.

    Returns:
        The :class:`RetrieverScores`.

    Raises:
        InterruptProcessingException: The queue was cancelled.
    """
    from llama_index.core.evaluation import resolve_metrics

    metrics = [metric() for metric in resolve_metrics(list(metric_names))]
    evaluator = _evaluator_class()(metrics=metrics, retriever=retriever, document_level=level == "document")
    totals = {name: 0.0 for name in metric_names}
    scores = RetrieverScores()
    for question, chunk in pairs:
        runtime.check_interrupt()
        result = await evaluator.aevaluate(query=question, expected_ids=[source_key(chunk, level)])
        values = result.metric_vals_dict
        for name in metric_names:
            totals[name] += float(values.get(name, 0.0) or 0.0)
        scores.rows.append((question, float(values.get("hit_rate", 0.0) or 0.0), float(values.get("mrr", 0.0) or 0.0)))
    scores.questions = len(pairs)
    if pairs:
        scores.means = {name: totals[name] / len(pairs) for name in metric_names}
    return scores

"""Query engines answering a question from retrieved chunks, and what a query hands back.

Builders take wire values and return llama-index engines; :func:`run_query` asks one, streaming
the answer where the engine writes through a response synthesizer.
"""

from __future__ import annotations

import logging
import re
import time

from .. import log
from . import messages, runtime, wires

__all__ = [
    "MAX_QA_PROMPT_CHARS",
    "QA_FIELDS",
    "RESPONSE_MODES",
    "SELECTORS",
    "build_query_engine",
    "build_retry",
    "build_router",
    "build_sub_question",
    "build_transform",
    "check_qa_prompt",
    "failure",
    "is_interrupt",
    "query_tools",
    "quiet",
    "retriever_tools",
    "run_query",
    "selector_for",
    "source_name",
    "source_rows",
    "sources_text",
]

logger = log.get_logger("llm.query")

#: How an answer is written from the retrieved chunks, as the widget offers them.
RESPONSE_MODES = (
    "compact",
    "refine",
    "tree_summarize",
    "simple_summarize",
    "accumulate",
    "compact_accumulate",
    "generation",
    "no_text",
    "context_only",
)

#: Response modes that write no answer from the chunks, so a question prompt is never read.
NO_QA_MODES = ("generation", "no_text", "context_only")

#: How a router picks among its engines.
SELECTORS = ("single", "multi")

#: The fields a question prompt holds, each exactly once at least.
QA_FIELDS = ("context_str", "query_str")

#: Characters of a question prompt read at most.
MAX_QA_PROMPT_CHARS = 20000

#: Characters of one chunk shown in the sources text.
SOURCE_TEXT_CHARS = 4000

#: A ``{field}`` in a prompt, matched the way llama-index substitutes one.
_FIELD = re.compile(r"\{([^{}]+)\}")

#: llama-index loggers that write prompt or answer text at INFO.
_CHATTY = (
    "llama_index.core.chat_engine.condense_question",
    "llama_index.core.chat_engine.condense_plus_context",
    "llama_index.core.query_engine.router_query_engine",
)

#: Metadata keys naming where a chunk came from, shortest name first.
_NAME_KEYS = ("file_name", "title", "document_title", "source", "url")


def quiet() -> None:
    """Hold the llama-index loggers that write prompt or answer text at INFO to WARNING."""
    for name in _CHATTY:
        chatty = logging.getLogger(name)
        if chatty.level < logging.WARNING:
            chatty.setLevel(logging.WARNING)


def is_interrupt(error: BaseException) -> bool:
    """Whether an exception is ComfyUI stopping the run."""
    return type(error).__name__ == "InterruptProcessingException"


def failure(node: str, error: BaseException) -> Exception:
    """The exception a node raises for a library or provider failure.

    Args:
        node: The node's display name.
        error: What was raised.

    Returns:
        ``error`` itself for an interrupt, otherwise a ``RuntimeError`` carrying the error's
        text with keys hidden.
    """
    if is_interrupt(error):
        return error
    return RuntimeError(f"{node} failed: {runtime.describe_error(error)}")


def check_qa_prompt(text: str, node: str) -> str:
    """A question prompt, once every ``{field}`` in it is one the engine fills.

    Args:
        text: The prompt as typed, empty for the engine's own.
        node: The node's display name, for the message.

    Returns:
        The prompt, stripped; empty when none was given.

    Raises:
        ValueError: The prompt is too long, misses ``{context_str}`` or ``{query_str}``, or
            holds another field.
    """
    prompt = str(text or "").strip()
    if not prompt:
        return ""
    if len(prompt) > MAX_QA_PROMPT_CHARS:
        raise ValueError(
            f"{node}: qa_prompt is {len(prompt)} characters and at most {MAX_QA_PROMPT_CHARS} "
            f"are read. Shorten it."
        )
    fields = [match.group(1) for match in _FIELD.finditer(prompt)]
    unknown = sorted({field for field in fields if field not in QA_FIELDS})
    if unknown:
        shown = ", ".join("{" + field[:40] + "}" for field in unknown[:5])
        raise ValueError(
            f"{node}: qa_prompt holds {shown}, and only {{context_str}} and {{query_str}} are "
            f"filled in. Remove the other fields, or write the braces as plain words."
        )
    missing = [field for field in QA_FIELDS if field not in fields]
    if missing:
        shown = " and ".join("{" + field + "}" for field in missing)
        raise ValueError(
            f"{node}: qa_prompt needs {shown}, where the retrieved chunks and the question go, "
            f"such as 'Context:\\n{{context_str}}\\nAnswer: {{query_str}}'."
        )
    return prompt


def build_query_engine(
    retriever: wires.Retriever,
    model: wires.LanguageModel,
    response_mode: str = "compact",
    postprocessors: tuple = (),
    citations: bool = False,
    citation_chunk_size: int = 512,
    qa_prompt: str = "",
    multimodal: bool = False,
):
    """A query engine answering from a retriever's chunks.

    Args:
        retriever: Where the chunks come from.
        model: The model writing the answer.
        response_mode: One of :data:`RESPONSE_MODES`.
        postprocessors: Steps run over the chunks first, in order.
        citations: True to cut chunks into numbered sources the answer cites.
        citation_chunk_size: Tokens of one numbered source.
        qa_prompt: A checked question prompt, empty for the engine's own.
        multimodal: Show the model the pictures among the chunks, in place of the question prompt.

    Returns:
        The llama-index query engine.
    """
    runtime.core()
    from llama_index.core.prompts import PromptTemplate
    from llama_index.core.query_engine import CitationQueryEngine, RetrieverQueryEngine
    from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer

    mode = ResponseMode(response_mode)
    template = PromptTemplate(qa_prompt) if qa_prompt else None
    processors = [step.processor for step in postprocessors]
    if citations:
        from llama_index.core.query_engine.citation_query_engine import CITATION_QA_TEMPLATE

        synthesizer = get_response_synthesizer(
            llm=model.llm,
            text_qa_template=template or CITATION_QA_TEMPLATE,
            summary_template=template if mode == ResponseMode.TREE_SUMMARIZE else None,
            response_mode=mode,
            multimodal=multimodal,
        )
        return CitationQueryEngine(
            retriever=retriever.retriever,
            llm=model.llm,
            response_synthesizer=synthesizer,
            citation_chunk_size=int(citation_chunk_size),
            citation_chunk_overlap=min(20, int(citation_chunk_size) // 4),
            node_postprocessors=processors,
            multimodal=multimodal,
        )
    synthesizer = get_response_synthesizer(
        llm=model.llm,
        text_qa_template=template,
        summary_template=template if mode == ResponseMode.TREE_SUMMARIZE else None,
        response_mode=mode,
        multimodal=multimodal,
    )
    return RetrieverQueryEngine(
        retriever=retriever.retriever,
        response_synthesizer=synthesizer,
        node_postprocessors=processors,
    )


def build_transform(engine: wires.QueryEngine, model: wires.LanguageModel, transform: dict):
    """A query engine that rewrites each question before another engine answers it.

    Args:
        engine: The engine answering the rewritten question.
        model: The model rewriting it.
        transform: The transform choice: ``{"transform": "hyde", "include_original": bool}``
            or ``{"transform": "decompose", "num_steps": int}``.

    Returns:
        ``(llama-index engine, label)``.
    """
    runtime.core()
    choice = transform.get("transform", "hyde")
    if choice == "decompose":
        from llama_index.core.indices.query.query_transform.base import StepDecomposeQueryTransform
        from llama_index.core.query_engine import MultiStepQueryEngine
        from llama_index.core.response_synthesizers import get_response_synthesizer

        steps = max(1, min(int(transform.get("num_steps", 3)), 10))
        built = MultiStepQueryEngine(
            query_engine=engine.engine,
            query_transform=StepDecomposeQueryTransform(llm=model.llm, verbose=False),
            response_synthesizer=get_response_synthesizer(llm=model.llm),
            num_steps=steps,
        )
        return built, f"{steps}-step decompose over {engine.label}"
    from llama_index.core.indices.query.query_transform import HyDEQueryTransform
    from llama_index.core.query_engine import TransformQueryEngine

    include = bool(transform.get("include_original", True))
    built = TransformQueryEngine(
        query_engine=engine.engine,
        query_transform=HyDEQueryTransform(llm=model.llm, include_original=include),
    )
    return built, f"HyDE over {engine.label}"


def query_tools(tools, node: str) -> list:
    """The query engine tools on an ``LLM_TOOLS`` wire, checked.

    Args:
        tools: What arrived on the tools input.
        node: The node's display name, for the message.

    Returns:
        The tools, in chain order.

    Raises:
        ValueError: Nothing arrived, an item is not a query engine tool, or two share a name.
    """
    runtime.core()
    from llama_index.core.tools import QueryEngineTool

    items = list(tools) if isinstance(tools, (tuple, list)) else []
    if not items:
        raise ValueError(
            f"{node} needs at least one query engine tool. Connect LLM Query Engine Tool to its "
            f"tools input, one per engine to choose among."
        )
    for item in items:
        if not isinstance(item, QueryEngineTool):
            raise ValueError(
                f"{node} only takes query engine tools, and was handed a "
                f"{type(item).__name__}. Make each tool with LLM Query Engine Tool."
            )
    seen: set[str] = set()
    for item in items:
        name = item.metadata.name or ""
        if name in seen:
            raise ValueError(
                f"{node} was handed two tools named {name[:64]!r}. Give each LLM Query Engine "
                f"Tool its own name."
            )
        seen.add(name)
    return items


def build_sub_question(tools: list, model: wires.LanguageModel):
    """A query engine that splits a question into one sub-question per tool it needs.

    Args:
        tools: Query engine tools, from :func:`query_tools`.
        model: The model writing the sub-questions and the final answer.

    Returns:
        The llama-index query engine.
    """
    runtime.core()
    from llama_index.core.query_engine import SubQuestionQueryEngine
    from llama_index.core.question_gen import LLMQuestionGenerator
    from llama_index.core.response_synthesizers import get_response_synthesizer

    from llama_index.core.question_gen.output_parser import SubQuestionOutputParser

    class TolerantSubQuestionParser(SubQuestionOutputParser):
        """Reads sub-questions a model wrote with doubled braces, as a JSON list or one object."""

        def parse(self, output):
            text = str(output).replace("{{", "{").replace("}}", "}")
            return super().parse(text)

    return SubQuestionQueryEngine(
        question_gen=LLMQuestionGenerator.from_defaults(
            llm=model.llm, output_parser=TolerantSubQuestionParser()
        ),
        response_synthesizer=get_response_synthesizer(llm=model.llm),
        query_engine_tools=tools,
        verbose=False,
        use_async=False,
    )


def _selection_parser():
    """A selection parser that also reads JSON written with the doubled braces of its example."""
    from llama_index.core.output_parsers.selection import SelectionOutputParser

    class TolerantSelectionParser(SelectionOutputParser):
        """Reads a model's choice of engine, single or doubled braces alike."""

        def parse(self, output):
            return super().parse(str(output).replace("{{", "{").replace("}}", "}"))

    return TolerantSelectionParser()


def selector_for(model: wires.LanguageModel, selector: str = "single"):
    """The selector a router picks with: by tool calling where the model offers it, else by its reply.

    Args:
        model: The model choosing.
        selector: ``single`` picks one, ``multi`` may pick several.

    Returns:
        A llama-index selector.
    """
    from llama_index.core.selectors import (
        LLMMultiSelector,
        LLMSingleSelector,
        PydanticMultiSelector,
        PydanticSingleSelector,
    )

    multi = selector == "multi"
    if model.function_calling and getattr(model.llm.metadata, "is_function_calling_model", False):
        try:
            if multi:
                return PydanticMultiSelector.from_defaults(llm=model.llm)
            return PydanticSingleSelector.from_defaults(llm=model.llm)
        except ValueError:
            pass
    if multi:
        return LLMMultiSelector.from_defaults(llm=model.llm, output_parser=_selection_parser())
    return LLMSingleSelector.from_defaults(llm=model.llm, output_parser=_selection_parser())


def retriever_tools(tools, node: str) -> list:
    """The retriever tools on an ``LLM_TOOLS`` wire, checked.

    Raises:
        ValueError: Nothing arrived, an item is not a retriever tool, or two share a name.
    """
    runtime.core()
    from llama_index.core.tools import RetrieverTool

    items = list(tools) if isinstance(tools, (tuple, list)) else []
    if not items:
        raise ValueError(f"{node} needs at least one retriever tool. Connect LLM Retriever Tool to its tools input.")
    names = set()
    for item in items:
        if not isinstance(item, RetrieverTool):
            raise ValueError(
                f"{node} only takes retriever tools, and was handed a {type(item).__name__}. Use LLM "
                f"Retriever Tool, one per retriever to choose among."
            )
        if item.metadata.name in names:
            raise ValueError(f"{node}: two retriever tools are named {item.metadata.name!r}. Give each its own name.")
        names.add(item.metadata.name)
    return items


def build_retry(engine: wires.QueryEngine, model: wires.LanguageModel, guidelines: str, retries: int):
    """A query engine that judges each answer and asks again, rewritten, while the answer fails.

    Args:
        engine: The engine answering.
        model: The judge, which also rewrites the question.
        guidelines: What an answer must do, empty to judge only that it fits the question.
        retries: Further attempts at most.

    Returns:
        The llama-index ``RetryGuidelineQueryEngine`` or ``RetryQueryEngine``.
    """
    runtime.core()
    from llama_index.core.evaluation import GuidelineEvaluator, RelevancyEvaluator
    from llama_index.core.indices.query.query_transform.feedback_transform import FeedbackQueryTransformation
    from llama_index.core.query_engine import RetryGuidelineQueryEngine, RetryQueryEngine

    written = str(guidelines or "").strip()
    if written:
        judge = GuidelineEvaluator(llm=model.llm, guidelines=written[:MAX_QA_PROMPT_CHARS])
        rewriter = FeedbackQueryTransformation(llm=model.llm, resynthesize_query=True)
        return RetryGuidelineQueryEngine(
            engine.engine, judge, resynthesize_query=True, max_retries=int(retries), query_transformer=rewriter,
        )
    return RetryQueryEngine(engine.engine, RelevancyEvaluator(llm=model.llm), max_retries=int(retries))


def build_router(tools: list, model: wires.LanguageModel, selector: str = "single"):
    """A query engine that hands each question to the engine, or engines, best suited.

    Args:
        tools: Query engine tools, from :func:`query_tools`.
        model: The model choosing among them and combining several answers.
        selector: ``single`` picks one engine, ``multi`` may pick several.

    Returns:
        The llama-index query engine.
    """
    runtime.core()
    quiet()
    from llama_index.core.query_engine import RouterQueryEngine
    from llama_index.core.response_synthesizers import TreeSummarize

    chooser = selector_for(model, selector)
    from llama_index.core.prompts.default_prompt_selectors import DEFAULT_TREE_SUMMARIZE_PROMPT_SEL

    return RouterQueryEngine(
        selector=chooser,
        query_engine_tools=tools,
        llm=model.llm,
        summarizer=TreeSummarize(llm=model.llm, summary_template=DEFAULT_TREE_SUMMARIZE_PROMPT_SEL),
        verbose=False,
    )


def _synthesizer(engine):
    """The response synthesizer an engine writes its answer through, or ``None``."""
    from llama_index.core.response_synthesizers import Accumulate

    for _ in range(8):
        if engine is None:
            return None
        synthesizer = getattr(engine, "_response_synthesizer", None)
        if synthesizer is not None and hasattr(synthesizer, "_streaming"):
            return None if isinstance(synthesizer, Accumulate) else synthesizer
        engine = getattr(engine, "_query_engine", None)
    return None


def run_query(engine, question: str, stream: runtime.TextStream | None = None) -> dict:
    """Ask a query engine one question, streaming the answer where the engine allows it.

    Args:
        engine: A llama-index query engine.
        question: The question.
        stream: Where the answer is drawn as it is written, or ``None``.

    Returns:
        ``{"answer", "thinking", "nodes", "seconds", "streamed", "sql"}``: the answer without
        inline thinking, that thinking, the ``NodeWithScore`` chunks it answered from, best
        first, the seconds taken, whether it streamed, and the SQL a table engine ran.
    """
    runtime.core()
    from llama_index.core.base.response.schema import AsyncStreamingResponse, StreamingResponse

    started = time.monotonic()
    synthesizer = _synthesizer(engine) if stream is not None else None
    held = getattr(synthesizer, "_streaming", False) if synthesizer is not None else False
    streamed = False
    try:
        if synthesizer is not None:
            synthesizer._streaming = True
        response = engine.query(question)
        if isinstance(response, StreamingResponse) and response.response_gen is not None:
            streamed = True
            text = ""
            for delta in response.response_gen:
                runtime.check_interrupt()
                text += delta or ""
                stream.write(delta or "")
            response.response_txt = text
        elif isinstance(response, AsyncStreamingResponse):
            text = str(response)
        else:
            text = getattr(response, "response", None)
            text = "" if text is None else str(text)
    finally:
        if synthesizer is not None:
            synthesizer._streaming = held
    if stream is not None:
        if not streamed:
            stream.write(text)
        stream.finish()
    answer, thinking = messages.split_thinking(text)
    return {
        "answer": answer,
        "thinking": thinking,
        "nodes": tuple(getattr(response, "source_nodes", None) or ()),
        "seconds": time.monotonic() - started,
        "streamed": streamed,
        "sql": str((getattr(response, "metadata", None) or {}).get("sql_query", "") or ""),
    }


def source_name(item) -> str:
    """A short name for where a retrieved chunk came from: its file name or title, never a path."""
    node = getattr(item, "node", item)
    metadata = getattr(node, "metadata", None) or {}
    for key in _NAME_KEYS:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            name = value.strip().replace("\\", "/").rsplit("/", 1)[-1] if key != "url" else value
            page = metadata.get("page_label")
            return f"{name[:80]} p.{page}" if isinstance(page, (str, int)) and str(page) else name[:80]
    return ""


def _score(item) -> str:
    """A chunk's score as text, empty when it has none."""
    score = getattr(item, "score", None)
    return f"{score:.3f}" if isinstance(score, (int, float)) else ""


def sources_text(items) -> str:
    """Retrieved chunks as numbered plain text, best first.

    Args:
        items: ``NodeWithScore`` objects.

    Returns:
        One paragraph per chunk headed ``[n] score name``, its text cut to 4000 characters.
    """
    parts = []
    for number, item in enumerate(items or (), start=1):
        head = " ".join(part for part in (f"[{number}]", _score(item), source_name(item)) if part)
        try:
            text = item.node.get_content()
        except Exception:
            text = str(getattr(item.node, "text", "") or "")
        if len(text) > SOURCE_TEXT_CHARS:
            text = text[: SOURCE_TEXT_CHARS - 3] + "..."
        parts.append(f"{head}\n{text.strip()}")
    return "\n\n".join(parts)


def source_rows(items) -> list[str]:
    """One short row per retrieved chunk, best first, for a panel."""
    rows = []
    for number, item in enumerate(items or (), start=1):
        try:
            text = item.node.get_content()
        except Exception:
            text = ""
        text = " ".join(text.split())
        head = " ".join(part for part in (f"{number}.", _score(item), source_name(item)) if part)
        rows.append(f"{head}: {text[:160]}" if text else head)
    return rows

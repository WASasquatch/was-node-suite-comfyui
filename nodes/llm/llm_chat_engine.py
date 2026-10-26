"""Choose how a chat answers: from the model alone, or from retrieved chunks too."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM, LLM_CHAT_ENGINE, LLM_POSTPROCESSORS, LLM_RETRIEVER
from ...modules.interface import run_result
from ...modules.llm import chat, wires

REQUIRES = "llm"

logger = log.get_logger("llm.chat")


class WASLLMChatEngine(io.ComfyNode):
    """How LLM Chat Session answers each message."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMChatEngine",
            display_name="LLM Chat Engine",
            search_aliases=[
                "WASLLMChatEngine",
                "LLM Chat Engine",
                "SimpleChatEngine",
                "ContextChatEngine",
                "CondenseQuestionChatEngine",
                "CondensePlusContextChatEngine",
                "chat with documents",
                "RAG chat",
            ],
            category="WAS Suite/LLM/Chat",
            description=(
                "Set how LLM Chat Session answers: from the model alone, or from chunks a "
                "retriever finds for each message, so a conversation can draw on indexed "
                "documents. The condense modes first rewrite a follow-up into a question that "
                "stands on its own."
            ),
            inputs=[
                LLM.Input(
                    "llm",
                    tooltip="The model that answers, such as LLM OpenAI or LLM From CLIP.",
                ),
                io.Combo.Input(
                    "mode", options=list(chat.CHAT_MODES), default="simple",
                    tooltip=(
                        "`simple` = the model alone; `context` = chunks retrieved for each "
                        "message go in the system prompt; `condense_question` = the follow-up is "
                        "rewritten, then answered from the chunks; `condense_plus_context` = "
                        "rewritten, retrieved, then answered in the conversation."
                    ),
                ),
                LLM_RETRIEVER.Input(
                    "retriever", optional=True,
                    tooltip="Where chunks come from for the context, condense_question and condense_plus_context modes, such as LLM Index Retriever over a vector index.",
                ),
                io.String.Input(
                    "system_prompt", force_input=True, optional=True,
                    tooltip="Instructions behind every answer, such as from Load System Prompt. Not read by condense_question.",
                ),
                LLM_POSTPROCESSORS.Input(
                    "postprocessors", optional=True,
                    tooltip="Filters and rerankers run over the retrieved chunks before each answer, in chain order. Not read by simple.",
                ),
            ],
            outputs=[
                LLM_CHAT_ENGINE.Output(
                    display_name="chat_engine",
                    tooltip="The chat engine, for LLM Chat Session.",
                ),
            ],
        )

    @classmethod
    def execute(cls, llm, mode, retriever=None, system_prompt="", postprocessors=None) -> io.NodeOutput:
        """Describe the chat engine.

        Args:
            llm: The model answering.
            mode: How it answers.
            retriever: Where a context mode finds chunks.
            system_prompt: Instructions behind every answer.
            postprocessors: Steps run over the retrieved chunks.

        Returns:
            The chat engine wire.
        """
        node = "LLM Chat Engine"
        model = wires.expect(llm, wires.LanguageModel, node, "llm", "LLM OpenAI or LLM From CLIP")
        if mode not in chat.CHAT_MODES:
            raise ValueError(f"{node}: mode {str(mode)[:40]!r} is not one of {', '.join(chat.CHAT_MODES)}.")
        source = None
        if mode in chat.CONTEXT_MODES:
            if retriever is None:
                raise ValueError(
                    f"{node}: the {mode} mode searches a retriever for every message and none is "
                    f"connected. Connect LLM Index Retriever, or another LLM retriever, to the retriever input, or set mode "
                    f"to simple to chat with the model alone."
                )
            source = wires.expect(retriever, wires.Retriever, node, "retriever", "LLM Index Retriever or another LLM retriever")
        elif retriever is not None:
            wires.expect(retriever, wires.Retriever, node, "retriever", "LLM Index Retriever or another LLM retriever")
        steps = wires.chain(postprocessors)
        if not wires.every(steps, wires.Postprocessor):
            raise ValueError(f"{node} was handed something other than postprocessors on its postprocessors input. Connect LLM Postprocessor or LLM Rerank to it.")
        prompt = str(system_prompt or "").strip()
        unused = []
        if mode == "simple" and retriever is not None:
            unused.append("the retriever is not searched by simple")
        if mode == "simple" and steps:
            unused.append("postprocessors are not run by simple")
        if mode == "condense_question" and prompt:
            unused.append("the system prompt is not read by condense_question")
        read_prompt = "" if mode == "condense_question" else prompt
        read_steps = () if mode == "simple" else steps
        wires.ready(model, source, read_steps)
        build = chat.make_builder(mode, model, source, read_prompt, read_steps)
        label = f"{mode} chat on {model.label}"
        wire = wires.ChatEngine(build=build, label=label, mode=mode, depends=(model, source, *read_steps))
        facts = {"model": model.label, "mode": mode}
        if source is not None:
            facts["retriever"] = source.label
        if read_prompt:
            facts["system prompt"] = f"{len(read_prompt)} characters"
        if read_steps:
            facts["postprocessors"] = ", ".join(step.label for step in read_steps)[:64]
        logger.info("LLM Chat Engine set up %s chat", mode)
        run_result.publish(
            status=run_result.WARNING if unused else run_result.OK,
            summary="; ".join(unused) if unused else label,
            facts=facts,
        )
        return io.NodeOutput(wire)

"""Add a long-term memory block to the chain LLM Memory reads."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM, LLM_EMBEDDING, LLM_MEMORY_BLOCKS
from ...modules.interface import run_result
from ...modules.llm import storage, wires
from ...modules.llm import memory as memories

REQUIRES = "llm"

logger = log.get_logger("llm.memory")


class WASLLMMemoryBlock(io.ComfyNode):
    """One long-term memory block, appended to a chain."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        kinds = [
            io.DynamicCombo.Option(
                key="static",
                inputs=[
                    io.String.Input(
                        "text", multiline=True, default="",
                        tooltip="What the model is always shown, such as `The user is a lighting artist who prefers short answers.`",
                    ),
                ],
            ),
            io.DynamicCombo.Option(
                key="facts",
                inputs=[
                    io.Int.Input(
                        "max_facts", default=50, min=1, max=500,
                        tooltip="Facts kept before the oldest are condensed together. `50` remembers a long conversation's details; `10` keeps only the essentials.",
                    ),
                ],
            ),
            io.DynamicCombo.Option(
                key="vector",
                inputs=[
                    io.Int.Input(
                        "top_k", default=2, min=1, max=20,
                        tooltip="Older messages brought back for each new one, the most similar first. `2` recalls the closest exchanges; `8` recalls more and costs more tokens.",
                    ),
                ],
            ),
        ]
        return io.Schema(
            node_id="WASLLMMemoryBlock",
            display_name="LLM Memory Block",
            search_aliases=[
                "WASLLMMemoryBlock",
                "LLM Memory Block",
                "StaticMemoryBlock",
                "FactExtractionMemoryBlock",
                "VectorMemoryBlock",
                "long term memory",
            ],
            category="WAS Suite/LLM/Memory",
            description=(
                "Give LLM Memory a long-term block: fixed text shown every message, facts a "
                "model pulls out of older messages, or older messages recalled by similarity. "
                "Chain several and connect the last to LLM Memory."
            ),
            inputs=[
                io.DynamicCombo.Input(
                    "kind", options=kinds, display_name="Kind",
                    tooltip="`static` = fixed text shown every message; `facts` = facts pulled out of messages that leave the recent window; `vector` = older messages recalled when similar to the new one.",
                ),
                io.String.Input(
                    "name", default="core",
                    tooltip="The block's name, different for every block in one memory, such as `profile` or `facts`. A vector block's saved file is named after it.",
                ),
                io.Int.Input(
                    "priority", default=0, min=0, max=100,
                    tooltip="`0` = never cut when the memory is over its token limit; higher numbers are cut first, so `1` goes before `0`.",
                ),
                LLM_MEMORY_BLOCKS.Input(
                    "blocks", optional=True,
                    tooltip="Blocks from earlier LLM Memory Block nodes, kept ahead of this one.",
                ),
                LLM.Input(
                    "llm", optional=True,
                    tooltip="The model a facts block pulls facts out with. Left empty, the conversation's own model does it.",
                ),
                LLM_EMBEDDING.Input(
                    "embedding", optional=True,
                    tooltip="The embedding model a vector block compares messages with, from LLM Embedding Model. Required by vector.",
                ),
            ],
            outputs=[
                LLM_MEMORY_BLOCKS.Output(
                    display_name="blocks",
                    tooltip="The chain with this block appended, for the next LLM Memory Block or LLM Memory.",
                ),
            ],
        )

    @classmethod
    def execute(cls, kind, name, priority, blocks=None, llm=None, embedding=None) -> io.NodeOutput:
        """Append the block.

        Args:
            kind: The block's kind and its settings.
            name: The block's name.
            priority: When it is cut.
            blocks: Earlier blocks.
            llm: The model a facts block extracts with.
            embedding: The model a vector block embeds with.

        Returns:
            The chain.
        """
        node = "LLM Memory Block"
        choice = dict(kind or {})
        which = choice.get("kind", "static")
        if which not in memories.BLOCK_KINDS:
            raise ValueError(f"{node}: kind {str(which)[:40]!r} is not one of {', '.join(memories.BLOCK_KINDS)}.")
        label = storage.check_name(name, "Memory block")
        chained = wires.chain(blocks)
        if not wires.every(chained, memories.BlockSpec):
            raise ValueError(f"{node} was handed something other than memory blocks on its blocks input. Connect another LLM Memory Block to it.")
        if any(block.name == label for block in chained):
            raise ValueError(f"{node}: a block named {label!r} is already in the chain. Give this block another name.")
        model = None if llm is None else wires.expect(llm, wires.LanguageModel, node, "llm", "LLM OpenAI or LLM From CLIP")
        embedder = None if embedding is None else wires.expect(embedding, wires.EmbeddingModel, node, "embedding", "LLM Embedding Model")
        facts = {"kind": which, "name": label, "priority": str(int(priority))}
        if which == "static":
            text = str(choice.get("text", "") or "")
            if not text.strip():
                raise ValueError(f"{node}: a static block shows its text every message, and the text is empty. Type what the model should always know.")
            block = memories.BlockSpec(kind="static", name=label, priority=int(priority), text=text)
            facts["text"] = f"{len(text)} characters"
        elif which == "facts":
            wires.ready(model)
            block = memories.BlockSpec(
                kind="facts", name=label, priority=int(priority), llm=model,
                max_facts=int(choice.get("max_facts", 50)),
            )
            facts["model"] = model.label if model is not None else "the conversation's"
            facts["max facts"] = str(block.max_facts)
        else:
            if embedder is None:
                raise ValueError(f"{node}: a vector block compares messages by embedding and none is connected. Connect LLM Embedding Model to its embedding input.")
            wires.ready(embedder)
            block = memories.BlockSpec(
                kind="vector", name=label, priority=int(priority), embedding=embedder,
                top_k=int(choice.get("top_k", 2)),
            )
            facts["embedding"] = embedder.label
            facts["recalled"] = str(block.top_k)
        result = wires.chain(chained, block)
        logger.info("LLM Memory Block added a %s block; the chain holds %d", which, len(result))
        run_result.publish(
            status=run_result.OK,
            summary=f"{which} block {label!r}, {len(result)} in the chain",
            counts={"blocks": len(result)},
            facts=facts,
        )
        return io.NodeOutput(result)

"""Free the memory a language model and an embedding model hold, passing a value through."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM, LLM_EMBEDDING
from ...modules.interface import run_result
from ...modules.llm import clip_llm, wires

REQUIRES = "llm"

logger = log.get_logger("llm.models")


class WASLLMUnload(io.ComfyNode):
    """Releases a language model and an embedding model between two steps of a graph."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMUnload",
            display_name="LLM Unload",
            search_aliases=[
                "WASLLMUnload",
                "LLM Unload",
                "free memory",
                "unload model",
                "free VRAM",
            ],
            category="WAS Suite/LLM/Models",
            description=(
                "Free the memory a language model and an embedding model hold, then pass a value "
                "on unchanged, so a diffusion step after an LLM step has the VRAM. The panel "
                "shows what was released and how much memory came back."
            ),
            inputs=[
                LLM.Input(
                    "llm", optional=True,
                    tooltip="The language model to release, from LLM From CLIP, LLM GGUF Loader or LLM Transformers Loader. It reloads if used again.",
                ),
                LLM_EMBEDDING.Input(
                    "embedding", optional=True,
                    tooltip="The embedding model to release, from LLM Embedding Model. It reloads if used again.",
                ),
                io.AnyType.Input(
                    "passthrough", optional=True,
                    tooltip="Any value, such as the text an LLM step wrote. Wiring it here makes the release wait for that step.",
                ),
            ],
            outputs=[
                io.AnyType.Output(
                    display_name="passthrough",
                    tooltip="The passthrough value, unchanged, for the step that needs the memory.",
                ),
            ],
        )

    @classmethod
    def execute(cls, llm=None, embedding=None, passthrough=None) -> io.NodeOutput:
        """Release each connected model and hand the passthrough on.

        Args:
            llm: A language model wire.
            embedding: An embedding model wire.
            passthrough: Any value.

        Returns:
            The passthrough value.
        """
        if llm is not None:
            wires.expect(llm, wires.LanguageModel, "LLM Unload", "llm", "an LLM model node")
        if embedding is not None:
            wires.expect(embedding, wires.EmbeddingModel, "LLM Unload", "embedding", "LLM Embedding Model")
        before = clip_llm.free_bytes()
        released, kept, failed = [], [], []
        for model in (llm, embedding):
            if model is None:
                continue
            if model.release is None:
                kept.append(model.label)
                continue
            try:
                model.release()
                released.append(model.label)
            except Exception as error:
                logger.warning("LLM Unload could not release %s: %s", model.label, error)
                failed.append(model.label)
        clip_llm.empty_cache()
        after = clip_llm.free_bytes()
        freed = max(0, after - before) if before is not None and after is not None else 0
        logger.info("LLM Unload released %d model(s), %.2f GB came back", len(released), freed / 2**30)
        facts = {}
        if released:
            facts["released"] = ", ".join(released)
        if kept:
            facts["holds nothing"] = ", ".join(kept)
        if failed:
            facts["failed"] = ", ".join(failed)
        if after is not None:
            facts["free now"] = f"{after / 2**30:.2f} GB"
        if not released and not kept and not failed:
            summary = "nothing was connected to release"
        else:
            summary = f"released {len(released)} model(s), {freed / 2**30:.2f} GB came back"
        run_result.publish(
            status=run_result.ERROR if failed else (run_result.OK if released else run_result.WARNING),
            summary=summary,
            counts={"released": len(released), "freed GB": round(freed / 2**30, 2)},
            facts=facts,
        )
        return io.NodeOutput(passthrough)

"""Answer one prompt with a language model."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM
from ...modules.interface import run_result
from ...modules.llm import generate, runtime, wires
from ...modules.llm import messages as llm_messages

REQUIRES = "llm"

logger = log.get_logger("llm.generate")

#: The node's name in messages.
NODE = "LLM Complete"

#: The nodes that make a language model, for messages.
LLM_HINT = "LLM From CLIP, LLM GGUF Loader or LLM OpenAI"

#: The most tokens the answer may be held to.
MAX_TOKENS = 131072


class LLMComplete(io.ComfyNode):
    """One prompt answered by a language model, streamed onto the node."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASLLMComplete",
            display_name="LLM Complete",
            search_aliases=[
                "WASLLMComplete",
                "LLM Complete",
                "complete",
                "stream_complete",
                "generate text",
                "prompt enhance",
                "expand prompt",
                "ask llm",
            ],
            category="WAS Suite/LLM/Generate",
            is_output_node=True,
            description=(
                "Ask a language model one question and get its answer as text, such as turning a "
                "short idea into a detailed image prompt for CLIP Text Encode. Pictures and sound "
                "can go with the prompt. The answer streams onto the node as it is written."
            ),
            inputs=[
                LLM.Input(
                    "llm",
                    tooltip=(
                        "The model that answers, from LLM From CLIP, LLM GGUF Loader, LLM OpenAI "
                        "or another LLM model node."
                    ),
                ),
                io.String.Input(
                    "prompt", multiline=True, default="",
                    tooltip=(
                        "What to ask, such as `Expand into a detailed photo prompt: a lighthouse "
                        "in a storm`."
                    ),
                ),
                io.Int.Input(
                    "max_tokens", default=0, min=0, max=MAX_TOKENS,
                    tooltip=(
                        "0 = the model's own limit; 300 = about 225 words at most. Lower it to "
                        "keep answers short."
                    ),
                ),
                io.String.Input(
                    "system_prompt", force_input=True, optional=True,
                    tooltip=(
                        "Standing instructions placed before the prompt, such as from Load System "
                        "Prompt: `You write vivid, concise image prompts.`"
                    ),
                ),
                io.Image.Input(
                    "image", optional=True,
                    tooltip=(
                        "Pictures to ask about, for a model that reads them. Every frame of a "
                        "batch is sent, up to 64."
                    ),
                ),
                io.Audio.Input(
                    "audio", optional=True,
                    tooltip=(
                        "Sound to ask about, for a model that hears it. The first clip of a batch "
                        "is sent, up to 10 minutes."
                    ),
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="text",
                    tooltip=(
                        "The answer, thinking removed. Wire it into CLIP Text Encode to use it as "
                        "an image prompt."
                    ),
                ),
                io.String.Output(
                    display_name="thinking",
                    tooltip=(
                        "What a reasoning model thought before answering. Empty for a model that "
                        "does not think aloud."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, llm, prompt, max_tokens, system_prompt=None, image=None, audio=None) -> io.NodeOutput:
        """Ask the model and stream its answer.

        Args:
            llm: The ``LanguageModel`` that answers.
            prompt: What to ask.
            max_tokens: Tokens to write at most, 0 for the model's own limit.
            system_prompt: Standing instructions, or ``None``.
            image: Pictures to ask about, or ``None``.
            audio: Sound to ask about, or ``None``.

        Returns:
            The answer and the thinking before it.

        Raises:
            ValueError: Nothing to ask, no model, or pictures for a model that does not read them.
            RuntimeError: The model failed.
        """
        model = wires.expect(llm, wires.LanguageModel, NODE, "llm", LLM_HINT)
        wires.ready(model)
        prompt = str(prompt or "")
        system = str(system_prompt or "").strip()
        if not prompt.strip() and image is None and not audio:
            raise ValueError(f"{NODE} has nothing to ask. Type a prompt, or connect an image or audio.")
        pictures = int(image.shape[0]) if image is not None and image.ndim == 4 else (1 if image is not None else 0)
        generate.require_vision(model, NODE, pictures)
        with runtime.guarded():
            chat = [llm_messages.message("system", system)] if system else []
            chat.append(llm_messages.message("user", prompt, images=image, audio=audio))
            answer = generate.reply(model, chat, NODE, max_tokens=int(max_tokens or 0))
        logger.info(
            "%s: %s wrote %d characters in %.2fs", NODE, model.label, len(answer.text), answer.seconds
        )
        if not max_tokens:
            limit = "the model's own"
        elif answer.limited:
            limit = str(int(max_tokens))
        else:
            limit = "the model's own; it takes no limit"
        facts = {"model": model.label, "max tokens": limit}
        if system:
            facts["system prompt"] = f"{len(system)} characters"
        if pictures:
            facts["pictures"] = str(pictures)
        if audio:
            facts["sound"] = "1 clip"
        bodies = [run_result.body("text", answer.text)]
        if answer.thinking:
            bodies.append(run_result.body("thinking", answer.thinking))
        empty = not answer.text.strip()
        run_result.publish(
            status=run_result.WARNING if empty or (max_tokens and not answer.limited) else run_result.OK,
            summary=(
                "The model answered nothing." if empty
                else f"Wrote {len(answer.text.split())} words in {answer.seconds:.1f}s."
            ),
            counts={
                "characters": len(answer.text),
                "words": len(answer.text.split()),
                "thinking characters": len(answer.thinking),
                "seconds": round(answer.seconds, 2),
            },
            facts=facts,
            bodies=bodies,
        )
        return io.NodeOutput(answer.text, answer.thinking)

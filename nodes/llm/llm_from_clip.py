"""Turn the language model inside a loaded CLIP into an LLM wire."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM
from ...modules.interface import run_result
from ...modules.llm import clip_llm, runtime, wires

REQUIRES = "llm"

logger = log.get_logger("llm.models")

#: The largest seed the sampler takes.
MAX_SEED = 0xFFFFFFFFFFFFFFFF

#: How each tokenizer family is described on the panel.
FORMATS = {
    clip_llm.CHATML: "ChatML conversation",
    clip_llm.MISTRAL: "Mistral instruction",
    clip_llm.TEMPLATE: "model template, earlier turns as a transcript",
}


class WASLLMFromCLIP(io.ComfyNode):
    """The language model inside a CLIP, on the LLM wire."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        sampling = [
            io.DynamicCombo.Option(
                key="on",
                inputs=[
                    io.Float.Input(
                        "temperature", default=0.7, min=0.01, max=2.0, step=0.01,
                        tooltip="How adventurous each pick is. `0.7` is balanced, `0.2` stays close to the likeliest words, `1.2` wanders.",
                    ),
                    io.Int.Input(
                        "top_k", default=64, min=0, max=1000,
                        tooltip="Pick only among this many likeliest tokens. `0` turns the limit off.",
                    ),
                    io.Float.Input(
                        "top_p", default=0.95, min=0.0, max=1.0, step=0.01,
                        tooltip="Pick only among the likeliest tokens whose chances add up to this. `1.0` turns it off.",
                    ),
                    io.Float.Input(
                        "min_p", default=0.05, min=0.0, max=1.0, step=0.01,
                        tooltip="Drop any token less likely than this fraction of the likeliest one. `0` turns it off.",
                    ),
                    io.Float.Input(
                        "repetition_penalty", default=1.05, min=0.0, max=5.0, step=0.01,
                        tooltip="Above `1.0` makes a token already used less likely again. `1.0` turns it off.",
                    ),
                    io.Int.Input(
                        "seed", default=0, min=0, max=MAX_SEED,
                        tooltip="Sampling seed, such as `0` or `42`. The same seed with the same settings and conversation writes the same text.",
                    ),
                    io.Float.Input(
                        "presence_penalty", default=0.0, min=0.0, max=5.0, step=0.01, optional=True,
                        tooltip="Subtracted from every token already used. `0` turns it off, `0.5` pushes toward new words.",
                    ),
                ],
            ),
            io.DynamicCombo.Option(key="off", inputs=[]),
        ]
        return io.Schema(
            node_id="WASLLMFromCLIP",
            display_name="LLM From CLIP",
            search_aliases=[
                "WASLLMFromCLIP",
                "LLM From CLIP",
                "CustomLLM",
                "local LLM",
                "text encoder LLM",
                "qwen",
                "gemma",
                "ministral",
            ],
            category="WAS Suite/LLM/Models",
            description=(
                "Use the language model inside a loaded CLIP, such as Qwen3 4B or Gemma 3, as "
                "the LLM for chat, retrieval and agents, with no second copy of the weights. "
                "The panel shows the chat format it uses and whether it reads pictures."
            ),
            inputs=[
                io.Clip.Input(
                    "clip",
                    tooltip="A CLIP holding a language model, such as qwen_3_4b from Load CLIP with type `lumina2`, qwen3vl_4b, gemma_3_4b_it or ministral-3-3b.",
                ),
                io.DynamicCombo.Input(
                    "sampling_mode", options=sampling, display_name="Sampling Mode",
                    tooltip="`on` draws each token at random within the limits below. `off` always takes the likeliest token and writes the same text every time.",
                ),
                io.Int.Input(
                    "max_tokens", default=1024, min=1, max=32768,
                    tooltip="The most tokens one reply writes. A token is about three quarters of a word, so `1024` is about 750 words.",
                ),
                io.Boolean.Input(
                    "thinking", default=False, optional=True,
                    tooltip="`true` lets a model that reasons, such as Qwen3, think before answering. The thinking arrives apart from the answer.",
                ),
                io.Boolean.Input(
                    "fast_decode", default=True, optional=True,
                    tooltip="`true` writes on ComfyUI's fixed cache, graph captured decode where the model allows it; `false` runs exactly as core Generate Text.",
                ),
            ],
            outputs=[
                LLM.Output(
                    display_name="llm",
                    tooltip="The model, for LLM Complete, LLM Chat, the query engines and the agents.",
                ),
            ],
        )

    @classmethod
    def execute(cls, clip, sampling_mode, max_tokens, thinking=False, fast_decode=True) -> io.NodeOutput:
        """Wrap the CLIP's language model.

        Args:
            clip: A CLIP holding a language model.
            sampling_mode: The sampling choice and its settings.
            max_tokens: The most tokens one reply writes.
            thinking: Whether a reasoning model thinks first.
            fast_decode: Whether to take the fast decode path.

        Returns:
            The LLM wire.
        """
        if clip is None:
            raise ValueError(
                "LLM From CLIP needs a CLIP. Connect Load CLIP with a language model such as "
                "qwen_3_4b.safetensors."
            )
        if clip_llm.generator_of(clip) is None:
            raise ValueError(
                "This CLIP holds no language model that writes text, such as CLIP-L or T5. Load "
                "one that does, such as qwen_3_4b.safetensors with type lumina2, qwen3vl_4b or "
                "gemma_3_4b_it."
            )
        mode = dict(sampling_mode or {})
        sampled = mode.get("sampling_mode") == "on"
        settings = {
            "on": sampled,
            "temperature": mode.get("temperature", 1.0),
            "top_k": mode.get("top_k", 50),
            "top_p": mode.get("top_p", 1.0),
            "min_p": mode.get("min_p", 0.0),
            "repetition_penalty": mode.get("repetition_penalty", 1.0),
            "presence_penalty": mode.get("presence_penalty", 0.0),
            "seed": mode.get("seed"),
        }
        with runtime.guarded():
            family = clip_llm.family_of(clip)
            name = family.name or "CLIP"
            context = clip_llm.context_of(clip)
            label = f"{name} on CLIP"
            model = clip_llm.clip_model()(
                clip,
                model_label=label,
                context=context,
                max_tokens=int(max_tokens),
                thinking=bool(thinking),
                fast_decode=bool(fast_decode),
                sampling=settings,
            )
            wire = wires.LanguageModel(
                llm=model,
                label=label,
                provider="clip",
                local=True,
                function_calling=False,
                vision=family.vision,
                context_window=context,
                release=lambda: clip_llm.release_clip(clip),
            )
        logger.info("LLM From CLIP wrapped %s (%s, context %d)", name, family.kind, context)
        facts = {
            "model": name,
            "chat format": FORMATS.get(family.kind, family.kind),
            "pictures": "reads them" if family.vision else "text only",
            "thinking": "on" if thinking else "off",
            "sampling": f"on, temperature {settings['temperature']}" if sampled else "off, likeliest token",
            "decode": "fast where the model allows" if fast_decode else "core",
        }
        run_result.publish(
            status=run_result.OK,
            summary=f"{name} is ready as an LLM, {facts['pictures']}",
            counts={"context tokens": context, "max tokens": int(max_tokens)},
            facts=facts,
        )
        return io.NodeOutput(wire)

"""Load a Hugging Face transformers language model as an LLM wire."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM
from ...modules.interface import run_result
from ...modules.io import picker
from ...modules.llm import hf_llm, runtime, wires
from ...modules.model import model_directories

REQUIRES = "llm"

logger = log.get_logger("llm.models")

#: The model menu's only entry when no model folder is found and downloads are off.
NO_MODELS = "put a Hugging Face model folder in models/LLM"

#: The largest seed the sampler takes.
MAX_SEED = 0xFFFFFFFFFFFFFFFF


class WASLLMTransformersLoader(io.ComfyNode):
    """A transformers language model on the LLM wire."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        models = hf_llm.choices()
        return io.Schema(
            node_id="WASLLMTransformersLoader",
            display_name="LLM Transformers Loader",
            search_aliases=[
                "WASLLMTransformersLoader",
                "LLM Transformers Loader",
                "HuggingFaceLLM",
                "transformers",
                "safetensors",
                "local LLM",
                "qwen",
                "smollm",
            ],
            category="WAS Suite/LLM/Models",
            description=(
                "Load a Hugging Face language model folder from models/LLM and run it on this "
                "machine as the LLM for chat, retrieval and agents, its VRAM freed for other "
                "models when they need it. Picture models such as Qwen3-VL read pictures. The "
                "panel shows the model, its size and where it runs."
            ),
            inputs=[
                picker.path_input(io.Combo.Input(
                    "model", options=models or [NO_MODELS],
                    tooltip="A model folder in ComfyUI/models/LLM, such as `Qwen3-4B-Instruct-2507`, or a suggested repository, downloaded on first use when features.network is on.",
                ), folder=True, model=True),
                io.Combo.Input(
                    "dtype", options=hf_llm.DTYPES, default="auto",
                    tooltip="`auto` = bfloat16 or float16 on the GPU, float32 on the CPU; `float32` is exact and twice the memory.",
                ),
                io.Combo.Input(
                    "quantize", options=hf_llm.QUANTIZE, default="none",
                    tooltip="`none` = full weights; `8-bit` about halves the memory; `4-bit` quarters it. Quantized models need an NVIDIA GPU and stay on it.",
                ),
                io.Combo.Input(
                    "attention", options=hf_llm.ATTENTION, default="sdpa",
                    tooltip="`sdpa` = PyTorch's fused attention, faster; `eager` = the plain version, for a model sdpa refuses.",
                ),
                io.Float.Input(
                    "temperature", default=0.7, min=0.0, max=2.0, step=0.01,
                    tooltip="How adventurous each pick is. `0` always takes the likeliest token, `0.7` is balanced, `1.2` wanders.",
                ),
                io.Float.Input(
                    "top_p", default=0.95, min=0.0, max=1.0, step=0.01,
                    tooltip="Pick only among the likeliest tokens whose chances add up to this. `1.0` turns it off.",
                ),
                io.Int.Input(
                    "top_k", default=20, min=0, max=1000,
                    tooltip="Pick only among this many likeliest tokens. `0` turns the limit off.",
                ),
                io.Float.Input(
                    "min_p", default=0.0, min=0.0, max=1.0, step=0.01,
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
                io.Int.Input(
                    "max_tokens", default=1024, min=1, max=32768,
                    tooltip="The most tokens one reply writes. `1024` is about 750 words.",
                ),
                io.Boolean.Input(
                    "thinking", default=False,
                    tooltip="`true` lets a model that reasons, such as SmolLM3 or Qwen3, think before answering. The thinking arrives apart from the answer.",
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
    def execute(
        cls, model, dtype, quantize, attention, temperature, top_p, top_k, min_p,
        repetition_penalty, seed, max_tokens, thinking,
    ) -> io.NodeOutput:
        """Load the model and wrap it.

        Args:
            model: The model's menu name, or a full path to its folder wired in as text.
            dtype: One of the weight types.
            quantize: One of the quantizations.
            attention: One of the attention implementations.
            temperature: Sampling temperature, 0 for the likeliest token.
            top_p: Nucleus sampling share.
            top_k: Candidate count.
            min_p: Relative probability floor.
            repetition_penalty: Penalty on tokens already used.
            seed: Sampling seed.
            max_tokens: The most tokens one reply writes.
            thinking: Whether a reasoning model thinks first.

        Returns:
            The LLM wire.

        Raises:
            PathNotAllowed: A wired path lies outside the LLM folders.
        """
        model_directories(hf_llm.FOLDER)
        model = picker.model_name(model, (hf_llm.FOLDER,), folder=True).replace("\\", "/")
        if model == NO_MODELS or not model:
            raise ValueError(
                "LLM Transformers Loader found no model. Put a Hugging Face model folder, holding "
                "config.json and .safetensors weights, in ComfyUI/models/LLM, or turn "
                "features.network on to download a suggested one."
            )
        for name, value, allowed in (
            ("dtype", dtype, hf_llm.DTYPES),
            ("quantize", quantize, hf_llm.QUANTIZE),
            ("attention", attention, hf_llm.ATTENTION),
        ):
            if value not in allowed:
                raise ValueError(f"{name} `{value}` is not one this node offers. Pick one from the menu.")
        spec = hf_llm.Spec(label=str(model), dtype=dtype, quantize=quantize, attention=attention)
        with runtime.guarded():
            loaded, fresh = hf_llm.load(spec)
            label = f"{spec.short} on transformers"
            llm = hf_llm.hf_model()(
                spec,
                model_label=label,
                context=loaded.context,
                max_tokens=int(max_tokens),
                thinking=bool(thinking),
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                min_p=float(min_p),
                repetition_penalty=float(repetition_penalty),
                seed=int(seed),
            )
            wire = wires.LanguageModel(
                llm=llm,
                label=label,
                provider="transformers",
                local=True,
                function_calling=False,
                vision=loaded.vision,
                context_window=loaded.context,
                release=lambda: hf_llm.release(spec.key),
            )
        backend = loaded.backend
        facts = {
            "model": spec.label,
            "kind": "pictures and text" if loaded.vision else "text",
            "weights": quantize if quantize != "none" else dtype,
            "runs on": str(backend.load_device),
            "attention": attention,
            "loaded": f"{loaded.seconds:.2f} s" if fresh else "already in memory",
        }
        logger.info("LLM Transformers Loader ready: %s on %s", spec.label, backend.load_device)
        run_result.publish(
            status=run_result.OK,
            summary=f"{spec.short} is ready on {backend.load_device}",
            counts={
                "parameters (M)": round(loaded.parameters / 1e6, 1),
                "context tokens": loaded.context,
                "max tokens": int(max_tokens),
            },
            facts=facts,
        )
        return io.NodeOutput(wire)

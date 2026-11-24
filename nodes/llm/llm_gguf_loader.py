"""Load a GGUF language model through llama-cpp-python as an LLM wire."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM
from ...modules.interface import run_result
from ...modules.io import picker
from ...modules.llm import gguf, local_models

REQUIRES = "llm"

logger = log.get_logger("llm.models")

#: The model menu's only entry when no ``.gguf`` file is found.
NO_MODELS = "put a .gguf model in models/LLM or models/text_encoders, or download one in LM Studio"

#: The projector menu's entry for a model read as text only.
NO_PROJECTOR = local_models.NO_PROJECTOR

#: The largest seed llama.cpp takes, one below the value it reads as random.
MAX_SEED = 0xFFFFFFFE


class WASLLMGGUFLoader(io.ComfyNode):
    """A GGUF language model on the LLM wire."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        models = gguf.model_choices()
        return io.Schema(
            node_id="WASLLMGGUFLoader",
            display_name="LLM GGUF Loader",
            search_aliases=[
                "WASLLMGGUFLoader",
                "LLM GGUF Loader",
                "LlamaCPP",
                "llama.cpp",
                "llama-cpp-python",
                "gguf",
                "local LLM",
            ],
            category="WAS Suite/LLM/Models",
            description=(
                "Load a GGUF language model from models/LLM, models/text_encoders or LM Studio's "
                "downloads and run it on this machine as the LLM for chat, retrieval and agents. "
                "An mmproj file lets it read pictures. The panel shows the model, its folder, its "
                "context and whether it runs on the GPU."
            ),
            inputs=[
                picker.path_input(io.Combo.Input(
                    "model", options=models or [NO_MODELS],
                    tooltip=(
                        "The .gguf model file, from ComfyUI/models/LLM, models/text_encoders or the "
                        "folder LM Studio downloads into, such as a Qwen3 4B Instruct Q8_0."
                    ),
                ), model=True),
                picker.path_input(io.Combo.Input(
                    "mmproj", options=[gguf.AUTO_PROJECTOR, NO_PROJECTOR, *gguf.projector_choices()],
                    default=gguf.AUTO_PROJECTOR,
                    tooltip=(
                        "`auto` = the mmproj beside the model made for it; `none` = text only; a "
                        "named mmproj = pictures through that file, such as mmproj-Qwen2.5-VL-7B-f16.gguf."
                    ),
                ), model=True),
                io.Int.Input(
                    "n_ctx", default=8192, min=512, max=262144, step=256,
                    tooltip="The most tokens the model reads at once, conversation and reply together. `8192` suits chat; larger values take more memory.",
                ),
                io.Int.Input(
                    "gpu_layers", default=-1, min=-1, max=999,
                    tooltip="`-1` = every layer on the GPU; `0` = CPU only; `20` = the first 20 layers. Needs a llama-cpp-python build with GPU support.",
                ),
                io.Boolean.Input(
                    "flash_attn", default=False,
                    tooltip="`true` uses flash attention, which saves memory on a long context where the build supports it.",
                ),
                io.Combo.Input(
                    "chat_format", options=gguf.CHAT_FORMATS, default=gguf.AUTO,
                    tooltip="`auto` = the chat template inside the file. A named format such as `chatml` or `llama-3` overrides it; `llava-1-5` or `qwen2.5-vl` places pictures for that family.",
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
                    "top_k", default=40, min=0, max=1000,
                    tooltip="Pick only among this many likeliest tokens. `0` turns the limit off.",
                ),
                io.Float.Input(
                    "min_p", default=0.05, min=0.0, max=1.0, step=0.01,
                    tooltip="Drop any token less likely than this fraction of the likeliest one. `0` turns it off.",
                ),
                io.Float.Input(
                    "repeat_penalty", default=1.1, min=0.0, max=5.0, step=0.01,
                    tooltip="Above `1.0` makes a token already used less likely again. `1.0` turns it off.",
                ),
                io.Int.Input(
                    "seed", default=0, min=0, max=MAX_SEED,
                    tooltip="Sampling seed, such as `0` or `42`. The same seed with the same settings and conversation writes the same text.",
                ),
                io.Int.Input(
                    "max_tokens", default=1024, min=1, max=32768,
                    tooltip="The most tokens one reply writes, within n_ctx. `1024` is about 750 words.",
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
        cls, model, mmproj, n_ctx, gpu_layers, flash_attn, chat_format, temperature, top_p,
        top_k, min_p, repeat_penalty, seed, max_tokens,
    ) -> io.NodeOutput:
        """Load the model and wrap it.

        Args:
            model: The model's menu name, or a full path wired in as text.
            mmproj: The projector's menu name or full path, ``auto`` or ``none``.
            n_ctx: The context length.
            gpu_layers: Layers offloaded to the GPU.
            flash_attn: Whether to use flash attention.
            chat_format: ``auto`` or a llama-cpp-python chat format.
            temperature: Sampling temperature.
            top_p: Nucleus sampling share.
            top_k: Candidate count.
            min_p: Relative probability floor.
            repeat_penalty: Penalty on tokens already used.
            seed: Sampling seed.
            max_tokens: The most tokens one reply writes.

        Returns:
            The LLM wire.

        Raises:
            PathNotAllowed: A wired path lies outside the LLM, text_encoders and LM Studio folders.
        """
        model = gguf.menu_name(model)
        if mmproj not in (gguf.AUTO_PROJECTOR, NO_PROJECTOR):
            mmproj = gguf.menu_name(mmproj)
        wire, report = local_models.from_gguf(
            "LLM GGUF Loader", "" if model == NO_MODELS else model, mmproj, n_ctx, gpu_layers,
            flash_attn, chat_format, temperature, top_p, top_k, min_p, repeat_penalty, seed,
            max_tokens,
        )
        run_result.publish(status=report.status, summary=report.summary, counts=report.counts,
                           facts=report.facts)
        return io.NodeOutput(wire)

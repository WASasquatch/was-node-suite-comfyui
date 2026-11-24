"""Load a local language model from one menu, GGUF or ComfyUI's own files, as an LLM wire."""

from __future__ import annotations

import os

from comfy_api.latest import io

from ...modules import log
from ...modules.compat.types import LLM
from ...modules.interface import run_result
from ...modules.io import picker
from ...modules.llm import gguf, local_models
from ...modules.model import language_model

REQUIRES = "llm"

logger = log.get_logger("llm.models")

NODE_NAME = "LLM Loader"

#: The model menu's only entry when no file is found.
NO_MODELS = "put a language model in models/text_encoders or models/LLM, or download one in LM Studio"

#: The largest seed the sampler takes.
MAX_SEED = 0xFFFFFFFFFFFFFFFF

#: The largest seed llama.cpp takes, one below the value it reads as random.
GGUF_SEED = 0xFFFFFFFE

#: The model last built on ComfyUI's loader, by file, so a changed setting does not read it again.
_built: dict = {}


def _choices() -> tuple[list[str], str | None]:
    """Every language model file the menu lists, and the one most likely to be wanted."""
    files, likely = language_model.offered()
    names = sorted({*files, *gguf.model_choices()}, key=str.lower)
    return names, likely


def _load_clip(name: str):
    """The ``Loaded`` model in one text_encoders file, kept while the file is unchanged."""
    import folder_paths

    path = folder_paths.get_full_path_or_raise(language_model.FOLDER, name)
    stat = os.stat(path)
    key = (path, stat.st_size, stat.st_mtime_ns)
    if _built.get("key") != key:
        _built.clear()
        _built.update(key=key, loaded=language_model.load(name))
    return _built["loaded"]


class WASLLMLoader(io.ComfyNode):
    """A local language model from one menu on the LLM wire."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        files, likely = _choices()
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
            node_id="WASLLMLoader",
            display_name=NODE_NAME,
            search_aliases=[
                "WASLLMLoader",
                NODE_NAME,
                "Load LLM",
                "local LLM",
                "gguf",
                "llama.cpp",
                "joycaption",
                "caption model",
                "vision language model",
            ],
            category="WAS Suite/LLM/Models",
            description=(
                "Load a local language model as the LLM for chat, captions, retrieval and agents, "
                "from one menu of text_encoders files and GGUF models. A GGUF runs on llama.cpp and "
                "reads pictures through the mmproj beside it; any other file runs on ComfyUI's own "
                "loader. The panel shows the model, how it runs and whether it reads pictures."
            ),
            inputs=[
                picker.path_input(io.Combo.Input(
                    "model",
                    options=files or [NO_MODELS],
                    default=likely,
                    tooltip=(
                        "A language model from models/text_encoders, models/LLM or LM Studio's "
                        "downloads, such as qwen3_32b_w4a8.safetensors, qwen3vl_8b.safetensors or "
                        "llama-joycaption-beta-one-hf-llava.Q8_0.gguf. A .gguf runs on llama.cpp."
                    ),
                ), model=True),
                picker.path_input(io.Combo.Input(
                    "mmproj",
                    options=[gguf.AUTO_PROJECTOR, local_models.NO_PROJECTOR, *gguf.projector_choices()],
                    default=gguf.AUTO_PROJECTOR,
                    tooltip=(
                        "For a .gguf model. `auto` = the mmproj beside the model made for it; "
                        "`none` = text only; a named mmproj = pictures through that file."
                    ),
                ), model=True),
                io.DynamicCombo.Input(
                    "sampling_mode", options=sampling, display_name="Sampling Mode",
                    tooltip="`on` draws each token at random within the limits below. `off` always takes the likeliest token and writes the same text every time.",
                ),
                io.Int.Input(
                    "max_tokens", default=1024, min=1, max=32768,
                    tooltip="The most tokens one reply writes. `1024` is about 750 words.",
                ),
                io.Boolean.Input(
                    "thinking", default=False,
                    tooltip="`true` lets a model that reasons, such as Qwen3, think before answering. A .gguf follows the chat template in its file.",
                ),
                io.Int.Input(
                    "n_ctx", default=8192, min=512, max=262144, step=256, advanced=True,
                    tooltip="For a .gguf model: the most tokens it reads at once, conversation and reply together. `8192` suits chat; larger values take more memory.",
                ),
                io.Int.Input(
                    "gpu_layers", default=-1, min=-1, max=999, advanced=True,
                    tooltip="For a .gguf model: `-1` = every layer on the GPU; `0` = CPU only; `20` = the first 20 layers.",
                ),
                io.Boolean.Input(
                    "flash_attn", default=False, advanced=True,
                    tooltip="For a .gguf model: `true` uses flash attention, which saves memory on a long context where the build supports it.",
                ),
                io.Combo.Input(
                    "chat_format", options=gguf.CHAT_FORMATS, default=gguf.AUTO, advanced=True,
                    tooltip="For a .gguf model: `auto` = the chat template inside the file; `chatml` or `llama-3` overrides it; `llava-1-5` or `qwen2.5-vl` places pictures for that family.",
                ),
                io.Boolean.Input(
                    "fast_decode", default=True, advanced=True,
                    tooltip="For other files: `true` writes on ComfyUI's fixed cache, graph captured decode where the model allows it; `false` runs exactly as core Generate Text.",
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
    def execute(cls, model, mmproj, sampling_mode, max_tokens, thinking=False, n_ctx=8192,
                gpu_layers=-1, flash_attn=False, chat_format=gguf.AUTO,
                fast_decode=True) -> io.NodeOutput:
        """Open the model on llama.cpp for a GGUF file, on ComfyUI's loader for any other.

        Args:
            model: The model's menu name, or a full path wired in as text.
            mmproj: The projector's menu name or full path, ``auto`` or ``none``.
            sampling_mode: The sampling choice and its settings.
            max_tokens: The most tokens one reply writes.
            thinking: Whether a reasoning model thinks first.
            n_ctx: The context length of a GGUF model.
            gpu_layers: Layers of a GGUF model offloaded to the GPU.
            flash_attn: Whether a GGUF model uses flash attention.
            chat_format: ``auto`` or a llama-cpp-python chat format.
            fast_decode: Whether a model on ComfyUI's loader takes the fast decode path.

        Returns:
            The LLM wire.

        Raises:
            ValueError: No file is chosen, or the file holds no model that writes text.
            PathNotAllowed: A wired path lies outside the model folders.
        """
        if not model or model == NO_MODELS:
            raise ValueError(
                f"{NODE_NAME} has no model to load. Put a language model in "
                "ComfyUI/models/text_encoders or models/LLM, or download one in LM Studio, then "
                "press R in ComfyUI to refresh the menu."
            )
        settings = local_models.sampling_settings(sampling_mode)
        if local_models.is_gguf(model):
            if mmproj not in (gguf.AUTO_PROJECTOR, local_models.NO_PROJECTOR):
                mmproj = gguf.menu_name(mmproj)
            sampled = settings["on"]
            wire, report = local_models.from_gguf(
                NODE_NAME, gguf.menu_name(model), mmproj, n_ctx, gpu_layers, flash_attn, chat_format,
                temperature=settings["temperature"] if sampled else 0.0,
                top_p=settings["top_p"] if sampled else 1.0,
                top_k=settings["top_k"] if sampled else 1,
                min_p=settings["min_p"] if sampled else 0.0,
                repeat_penalty=settings["repetition_penalty"] if sampled else 1.0,
                seed=int(settings["seed"] or 0) % (GGUF_SEED + 1),
                max_tokens=max_tokens,
                presence_penalty=settings["presence_penalty"] if sampled else 0.0,
            )
            if thinking:
                report.facts["thinking"] = "as the file's chat template writes it"
        else:
            name = picker.model_name(model, (language_model.FOLDER,))
            loaded = _load_clip(name)
            wire, report = local_models.from_clip(
                NODE_NAME, loaded.clip, settings, max_tokens, thinking=thinking,
                fast_decode=fast_decode,
            )
            report.facts["built by"] = loaded.builder
            report.facts["weights"] = loaded.weights
            if loaded.parameters:
                report.counts["parameters (B)"] = round(loaded.parameters / 1e9, 2)
            report.counts["file (GB)"] = round(loaded.file_bytes / 1e9, 2)
        logger.info("%s opened %s", NODE_NAME, wire.label)
        run_result.publish(status=report.status, summary=report.summary, counts=report.counts,
                           facts=report.facts)
        return io.NodeOutput(wire)

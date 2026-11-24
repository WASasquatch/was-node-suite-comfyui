"""Local language models on the LLM wire: a GGUF file through llama.cpp, or a CLIP's model.

Each opener answers the wire and a :class:`Report` for the node's panel.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from . import clip_llm, gguf, runtime, wires

__all__ = ["NO_PROJECTOR", "Report", "from_clip", "from_gguf", "is_gguf", "sampling_settings"]

#: The projector choice for a model read as text only.
NO_PROJECTOR = "none"

#: How each tokenizer family is described on the panel.
FORMATS = {
    clip_llm.CHATML: "ChatML conversation",
    clip_llm.MISTRAL: "Mistral instruction",
    clip_llm.TEMPLATE: "model template, earlier turns as a transcript",
}


@dataclass
class Report:
    """What a node's panel shows about the model it opened.

    Attributes:
        status: ``"ok"`` or ``"warning"``.
        summary: One line.
        counts: Figures by name.
        facts: Text facts by name.
    """

    status: str
    summary: str
    counts: dict = field(default_factory=dict)
    facts: dict = field(default_factory=dict)


def is_gguf(name) -> bool:
    """Whether a menu name or wired path names a ``.gguf`` file."""
    return str(name or "").strip().lower().endswith(".gguf")


def from_gguf(node: str, model: str, mmproj: str, n_ctx: int, gpu_layers: int, flash_attn: bool,
              chat_format: str, temperature: float, top_p: float, top_k: int, min_p: float,
              repeat_penalty: float, seed: int, max_tokens: int,
              presence_penalty: float = 0.0) -> tuple[wires.LanguageModel, Report]:
    """Load a GGUF model through llama.cpp.

    Args:
        node: The node's display name, for messages.
        model: The model's menu name.
        mmproj: The projector's menu name, :data:`gguf.AUTO_PROJECTOR` or :data:`NO_PROJECTOR`.
        n_ctx: The context length.
        gpu_layers: Layers offloaded to the GPU, -1 for every one.
        flash_attn: Whether to use flash attention.
        chat_format: ``auto`` or a llama-cpp-python chat format.
        temperature: Sampling temperature; 0 takes the likeliest token.
        top_p: Nucleus sampling share.
        top_k: Candidate count.
        min_p: Relative probability floor.
        repeat_penalty: Penalty on tokens already used.
        seed: Sampling seed.
        max_tokens: The most tokens one reply writes.
        presence_penalty: Subtracted from every token already used.

    Returns:
        The wire and its report.

    Raises:
        ValueError: No model, an unknown file or format, or a picture format with no projector.
    """
    if chat_format not in gguf.CHAT_FORMATS:
        raise ValueError(f"chat_format `{chat_format}` is not one {node} offers. Pick one from the menu.")
    if not model or not is_gguf(model):
        raise ValueError(
            f"{node} found no .gguf model. Put one in ComfyUI/models/LLM or models/text_encoders, "
            "or download one in LM Studio, then press R in ComfyUI to refresh the menu."
        )
    path = gguf.locate(model)
    if path is None:
        raise ValueError(
            f"`{model}` is not a .gguf file in models/LLM, models/text_encoders or LM Studio's "
            "models folder. Pick one from the menu."
        )
    found = False
    projector, projector_name = "", ""
    if mmproj == gguf.AUTO_PROJECTOR:
        try:
            projector = gguf.projector_for(path, gguf.read_header(path)) or ""
        except (OSError, ValueError):
            projector = ""
        projector_name = os.path.basename(projector)
        found = bool(projector)
    elif mmproj and mmproj != NO_PROJECTOR:
        projector = gguf.locate(mmproj) or ""
        projector_name = mmproj
        if not projector:
            raise ValueError(
                f"The mmproj `{mmproj}` is not a .gguf file in models/LLM, models/text_encoders "
                "or LM Studio's models folder. Pick one from the menu."
            )
    if chat_format in gguf.VISION_FORMATS and not projector:
        raise ValueError(
            f"chat_format {chat_format} places pictures, which needs an mmproj. Put the model's "
            "mmproj file beside it or pick it, or set chat_format to auto."
        )
    llama_cpp = gguf.require()
    offload_ready = bool(llama_cpp.llama_supports_gpu_offload())
    layers = int(gpu_layers) if offload_ready else 0
    spec = gguf.LoadSpec(
        name=model, path=path, projector_name=projector_name, projector=projector,
        n_ctx=int(n_ctx), gpu_layers=layers, flash_attn=bool(flash_attn),
    )
    with runtime.guarded():
        loaded, fresh = gguf.load(spec, seed=int(seed))
        meta = loaded.meta
        title = str(meta.get("general.name") or os.path.splitext(os.path.basename(model))[0])
        label = f"{title} on llama.cpp"
        llm = gguf.gguf_model()(
            spec,
            model_label=label,
            context=int(n_ctx),
            max_tokens=int(max_tokens),
            chat_format=chat_format,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            min_p=float(min_p),
            repeat_penalty=float(repeat_penalty),
            presence_penalty=float(presence_penalty),
            seed=int(seed),
        )
        wire = wires.LanguageModel(
            llm=llm,
            label=label,
            provider="gguf",
            local=True,
            function_calling=False,
            vision=bool(projector) and chat_format not in gguf.TEXT_FORMATS,
            context_window=int(n_ctx),
            release=lambda: gguf.release(spec.key),
        )
    arch = str(meta.get("general.architecture") or "unknown")
    trained = meta.get(f"{arch}.context_length")
    warnings = []
    if int(gpu_layers) != 0 and not offload_ready:
        warnings.append("this llama-cpp-python build cannot use the GPU, so gpu_layers was ignored")
    if isinstance(trained, int) and int(n_ctx) > trained:
        warnings.append(f"n_ctx is past the {trained} tokens the model was trained on")
    if projector and chat_format in gguf.TEXT_FORMATS:
        warnings.append(f"chat_format {chat_format} reads no pictures, so the mmproj is unused")
    if not projector:
        pictures = "text only, no mmproj beside the model" if mmproj == gguf.AUTO_PROJECTOR else "text only"
    else:
        pictures = f"through {projector_name}" + (", found beside the model" if found else "")
    facts = {
        "model": title,
        "folder": gguf.folder_label(path) or "models",
        "architecture": arch,
        "file": f"{os.path.getsize(path) / 2**30:.2f} GB",
        "gpu": ("every layer" if layers < 0 else f"{layers} layers") if layers else "CPU only",
        "chat format": "from the file" if chat_format == gguf.AUTO else chat_format,
        "pictures": pictures,
        "loaded": f"{loaded.seconds:.2f} s" if fresh else "already in memory",
    }
    if warnings:
        facts["note"] = "; ".join(warnings)
    counts = {"context tokens": int(n_ctx), "max tokens": int(max_tokens)}
    if isinstance(trained, int):
        counts["trained context"] = trained
    cache = gguf.cache_bytes(meta, int(n_ctx))
    if cache:
        counts["cache GB"] = round(cache / 2**30, 2)
    report = Report(
        status="warning" if warnings else "ok",
        summary=f"{title} is ready, {facts['gpu']}" + (f" ({warnings[0]})" if warnings else ""),
        counts=counts,
        facts=facts,
    )
    return wire, report


def from_clip(node: str, clip, settings: dict, max_tokens: int, thinking: bool = False,
              fast_decode: bool = True) -> tuple[wires.LanguageModel, Report]:
    """Wrap the language model inside a CLIP.

    Args:
        node: The node's display name, for messages.
        clip: A CLIP holding a language model.
        settings: ``on``, ``temperature``, ``top_k``, ``top_p``, ``min_p``,
            ``repetition_penalty``, ``presence_penalty`` and ``seed``.
        max_tokens: The most tokens one reply writes.
        thinking: Whether a reasoning model thinks first.
        fast_decode: Whether to take the fast decode path.

    Returns:
        The wire and its report.

    Raises:
        ValueError: No CLIP, or one holding no model that writes text.
    """
    if clip is None:
        raise ValueError(
            f"{node} needs a CLIP. Connect Load CLIP with a language model such as "
            "qwen_3_4b.safetensors."
        )
    if clip_llm.generator_of(clip) is None:
        raise ValueError(
            "This CLIP holds no language model that writes text, such as CLIP-L or T5. Load "
            "one that does, such as qwen_3_4b.safetensors with type lumina2, qwen3vl_4b or "
            "gemma_3_4b_it."
        )
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
            sampling=dict(settings),
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
    sampled = bool(settings.get("on"))
    facts = {
        "model": name,
        "chat format": FORMATS.get(family.kind, family.kind),
        "pictures": "reads them" if family.vision else "text only",
        "thinking": "on" if thinking else "off",
        "sampling": f"on, temperature {settings.get('temperature')}" if sampled else "off, likeliest token",
        "decode": "fast where the model allows" if fast_decode else "core",
    }
    report = Report(
        status="ok",
        summary=f"{name} is ready as an LLM, {facts['pictures']}",
        counts={"context tokens": context, "max tokens": int(max_tokens)},
        facts=facts,
    )
    return wire, report


def sampling_settings(mode) -> dict:
    """The sampler settings a ``sampling_mode`` choice holds, likeliest token when off."""
    mode = dict(mode or {})
    return {
        "on": mode.get("sampling_mode") == "on",
        "temperature": mode.get("temperature", 1.0),
        "top_k": mode.get("top_k", 50),
        "top_p": mode.get("top_p", 1.0),
        "min_p": mode.get("min_p", 0.0),
        "repetition_penalty": mode.get("repetition_penalty", 1.0),
        "presence_penalty": mode.get("presence_penalty", 0.0),
        "seed": mode.get("seed"),
    }

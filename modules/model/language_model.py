"""Language models in ComfyUI's text_encoders folders, loaded as a ``CLIP`` that writes text.

Dense Qwen3 files ComfyUI does not read are built on its Qwen3 class, in Hugging Face key
names, plain or ``comfy_quant``, ``down_proj`` optionally in ``parts`` column slices.
"""

from __future__ import annotations

import json
import os
import re
import types
from collections import Counter
from dataclasses import dataclass, field

from .. import log

logger = log.get_logger("model.language_model")

__all__ = ["Loaded", "dense_qwen3", "load", "offered"]

#: The name the encoder and its tokens are filed under.
NAME = "was_qwen3"

#: The folder the files are listed from.
FOLDER = "text_encoders"

#: Qwen's chat turns, and the empty thinking block that answers without reasoning.
SYSTEM_TURN = "<|im_start|>system\n{}<|im_end|>\n"
USER_TURN = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
EMPTY_THINK = "<think>\n\n</think>\n\n"
TURN_START = "<|im_start|>"

#: Qwen's padding token, ``<|endoftext|>``.
PAD = 151643

#: Width of each attention head in every dense Qwen3.
HEAD_WIDTH = 128

#: Positions a dense Qwen3 was trained to read.
CONTEXT = 40960

#: File name fragments the menu offers first, most wanted first.
LIKELY = ("qwen3_32b_w4a8", "qwen3_32b", "qwen3-32b", "qwen3_14b", "qwen3-14b")

#: Prefixes of a vision tower, a hybrid attention stack or a nested language model.
FOREIGN = ("visual.", "model.visual.", "model.language_model.", "thinker.", "mtp.")

#: The child a feed-forward output layer stored in column slices keeps them under.
PARTS = "parts"

#: The column a slice of a split layer starts on is a multiple of this.
PART_ALIGN = 256

#: The end of a column slice's layer name, ``.parts.<index>``.
PART_SUFFIX = re.compile(r"\.parts\.\d+$")

#: Format tags as the panel names them.
FORMAT_NAMES = {
    "int8_tensorwise": "int8",
    "asym_w4a8_int8": "w4a8",
    "w6a8_int8": "w6a8",
    "float8_e4m3fn": "fp8",
    "float8_e5m2": "fp8 e5m2",
    "mxfp8": "mxfp8",
    "nvfp4": "nvfp4",
}

#: ComfyUI's encoders that hold a language model cut short, by detected model name.
NO_HEAD = {
    "QWEN3VL_32B": (
        "{name} is MiniMax H3's prompt encoder: the first 50 of Qwen3-VL-32B's 64 layers with no "
        "output layer, so it cannot write text. Pick a whole language model, such as "
        "qwen3_32b_w4a8.safetensors, or keep this file in Load CLIP for H3."
    ),
}


@dataclass
class Loaded:
    """A language model as ComfyUI holds it, and what was read to build it.

    Attributes:
        clip: The ``CLIP``.
        model: The model's name, such as ``"Qwen3 32B"``.
        builder: ``"WAS-NS"`` where the pack built it, ``"ComfyUI"`` where core did.
        parameters: Weights in the file.
        file_bytes: Size of the file.
        weights: How the weights are stored, such as ``"int8 convrot, 449 layers"``.
        config: The shape the model was built with, empty where core built it.
        head: True when the file carries its own output layer.
    """

    clip: object
    model: str
    builder: str
    parameters: int
    file_bytes: int
    weights: str
    config: dict = field(default_factory=dict)
    head: bool = True


def offered() -> tuple[list[str], str | None]:
    """The files in ComfyUI's text_encoders folders, and the one most likely to be wanted."""
    import folder_paths

    files = list(folder_paths.get_filename_list(FOLDER))
    for fragment in LIKELY:
        found = [name for name in files if fragment in name.lower()]
        if found:
            return files, found[0]
    return files, None


def dense_qwen3(state_dict) -> dict | None:
    """The shape of a dense Qwen3 checkpoint, read from its tensors.

    Args:
        state_dict: Tensors or meta tensors keyed in the Hugging Face layout.

    Returns:
        Keyword arguments for ComfyUI's Qwen3 configuration, or ``None`` when the checkpoint is
        not a dense Qwen3.
    """
    embed = state_dict.get("model.embed_tokens.weight")
    q_norm = state_dict.get("model.layers.0.self_attn.q_norm.weight")
    query = state_dict.get("model.layers.0.self_attn.q_proj.weight")
    key = state_dict.get("model.layers.0.self_attn.k_proj.weight")
    gate = state_dict.get("model.layers.0.mlp.gate_proj.weight")
    if any(value is None for value in (embed, q_norm, query, key, gate)):
        return None
    if "model.layers.0.self_attn.k_proj.bias" in state_dict:
        return None
    if any(name.startswith(FOREIGN) or ".mlp.experts." in name for name in state_dict):
        return None
    if int(q_norm.shape[0]) != HEAD_WIDTH:
        return None
    parts = down_parts(state_dict)
    if parts and int(gate.shape[0]) % parts:
        return None
    layers = sum(
        1 for name in state_dict
        if name.startswith("model.layers.") and name.endswith(".input_layernorm.weight")
    )
    return {
        "vocab_size": int(embed.shape[0]),
        "hidden_size": int(embed.shape[1]),
        "intermediate_size": int(gate.shape[0]),
        "num_hidden_layers": layers,
        "num_attention_heads": int(query.shape[0]) // HEAD_WIDTH,
        "num_key_value_heads": int(key.shape[0]) // HEAD_WIDTH,
        "max_position_embeddings": CONTEXT,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000.0,
        "lm_head": "lm_head.weight" in state_dict or "model.lm_head.weight" in state_dict,
    }


def split_widths(width: int, widest: int) -> list[int]:
    """The column slices a layer of ``width`` inputs is stored in, none wider than ``widest``.

    Args:
        width: The layer's input width.
        widest: The widest slice allowed.

    Returns:
        Equal widths, each a multiple of :data:`PART_ALIGN`, adding up to ``width``; one entry
        when the layer needs no split.
    """
    parts = max(1, -(-width // widest))
    while width % (parts * PART_ALIGN) and parts * PART_ALIGN <= width:
        parts += 1
    return [width // parts] * parts


def down_parts(state_dict) -> int:
    """How many column slices each feed-forward output layer is stored in, 0 for none."""
    prefix = f"model.layers.0.mlp.down_proj.{PARTS}."
    return len({name[len(prefix):].split(".", 1)[0] for name in state_dict if name.startswith(prefix)})


def load(name: str) -> Loaded:
    """Build the language model in one file of ComfyUI's text_encoders folders.

    Args:
        name: The file as the folder lists it.

    Returns:
        The :class:`Loaded` model.

    Raises:
        FileNotFoundError: No text_encoders folder holds ``name``.
        ValueError: The file holds no model that writes text.
    """
    import folder_paths

    path = folder_paths.get_full_path_or_raise(FOLDER, name)
    header = _header(path)
    config = dense_qwen3(header) if header is not None else None
    if config is not None and not _core_reads(header):
        return _build(path, config)
    return _core(path, name, header)


def load_patcher(path: str):
    """Build the model at ``path`` again and answer its patcher, for a copy on another device."""
    header = _header(path)
    return _build(path, dense_qwen3(header)).clip.patcher


def _header(path: str) -> dict | None:
    """A safetensors file's tensors as meta tensors, or ``None`` for any other file."""
    if not path.lower().endswith(".safetensors"):
        return None
    import torch
    from safetensors import safe_open

    dtypes = _safetensors_dtypes()
    out = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for key in handle.keys():
            piece = handle.get_slice(key)
            dtype = dtypes.get(piece.get_dtype(), torch.uint8)
            out[key] = torch.empty(piece.get_shape(), dtype=dtype, device="meta")
    return out


def _core_reads(header: dict) -> bool:
    """Whether ComfyUI's own loader recognises the checkpoint as a Qwen3."""
    import comfy.sd

    found = comfy.sd.detect_te_model(header)
    return found is not None and found.name.startswith("QWEN3")


def _core(path: str, name: str, header: dict | None) -> Loaded:
    """Hand a file to ComfyUI's text encoder loader and check it can write text."""
    import comfy.sd
    import comfy.utils
    import folder_paths

    from . import text_decode

    found = comfy.sd.detect_te_model(header) if header is not None else None
    if found is not None and found.name in NO_HEAD:
        raise ValueError(NO_HEAD[found.name].format(name=name))
    clip = comfy.sd.load_clip(
        ckpt_paths=[path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.STABLE_DIFFUSION,
    )
    transformer, _ = text_decode.decoder_of(clip)
    if transformer is None or not callable(getattr(transformer, "generate", None)):
        raise ValueError(
            f"{name} is a text encoder that conditions images and cannot write text. Pick a "
            "language model, such as a Qwen3, Gemma 3 or Ministral file."
        )
    model = found.name.replace("_", " ").title() if found is not None else os.path.splitext(name)[0]
    parameters = comfy.utils.calculate_parameters(header) if header is not None else 0
    return Loaded(
        clip=clip,
        model=model,
        builder="ComfyUI",
        parameters=int(parameters),
        file_bytes=os.path.getsize(path),
        weights=_weights(path, header) if header is not None else "as stored",
    )


def _build(path: str, config: dict) -> Loaded:
    """Build a dense Qwen3 from ComfyUI's own Qwen3 classes as a ``CLIP``."""
    import comfy.sd
    import comfy.text_encoders.hunyuan_video
    import comfy.utils
    import folder_paths

    state_dict = comfy.utils.load_torch_file(path, safe_load=True)
    weights = _weights(path, state_dict)
    # The output layer's weight, scale and quantization tag move under the model together.
    for key in [key for key in state_dict if key.startswith("lm_head.")]:
        state_dict[f"model.{key}"] = state_dict.pop(key)
    detected = comfy.text_encoders.hunyuan_video.llama_detect(state_dict)
    target = types.SimpleNamespace(
        clip=_encoder(config, detected.get("dtype_llama"), detected.get("llama_quantization_metadata"),
                      down_parts(state_dict)),
        tokenizer=_tokenizer(config["hidden_size"]),
        params={},
    )
    parameters = _qwen3_parameters(config)
    clip = comfy.sd.CLIP(
        target,
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        parameters=parameters,
        tokenizer_data={},
        state_dict=[state_dict],
        model_options={},
    )
    clip.patcher.cached_patcher_init = (load_patcher, (path,))
    model = _qwen3_name(parameters)
    logger.info("built %s, %d layers, %d wide, %s", model, config["num_hidden_layers"],
                config["hidden_size"], weights)
    return Loaded(
        clip=clip,
        model=model,
        builder="WAS-NS",
        parameters=int(parameters),
        file_bytes=os.path.getsize(path),
        weights=weights,
        config=dict(config),
        head=bool(config["lm_head"]),
    )


def _qwen3_parameters(config: dict) -> int:
    """Weights a dense Qwen3 of this shape holds, however they are stored."""
    hidden, inner = config["hidden_size"], config["intermediate_size"]
    query = config["num_attention_heads"] * HEAD_WIDTH
    keys = config["num_key_value_heads"] * HEAD_WIDTH
    block = 2 * hidden * query + 2 * hidden * keys + 3 * hidden * inner + 2 * hidden + 2 * HEAD_WIDTH
    tables = 2 if config["lm_head"] else 1
    return config["num_hidden_layers"] * block + tables * config["vocab_size"] * hidden + hidden


def _qwen3_name(parameters: int) -> str:
    """The published size a dense Qwen3 of this many weights goes by."""
    billions = parameters / 1e9
    for size, low, high in ((14, 13.0, 16.0), (32, 30.0, 34.5)):
        if low <= billions <= high:
            return f"Qwen3 {size}B"
    return f"Qwen3 {billions:.1f}B"


def _weights(path: str, state_dict: dict) -> str:
    """How the layers of a file are stored, such as ``"int8 convrot, 449 layers"``."""
    tags = [name for name in state_dict if name.endswith(".comfy_quant")]
    if not tags:
        floats = Counter(
            str(value.dtype).replace("torch.", "")
            for name, value in state_dict.items() if name.endswith(".weight")
        )
        return floats.most_common(1)[0][0] if floats else "unknown"
    # A layer stored in column slices counts once, under its own name.
    layers = {}
    for name, tag in zip(tags, _tags(path, tags)):
        layer = PART_SUFFIX.sub("", name[: -len(".comfy_quant")])
        layers[layer] = _tag_name(tag)
    formats = Counter(layers.values())
    return ", ".join(f"{fmt}, {count} layers" for fmt, count in formats.most_common())


def _tags(path: str, names: list[str]) -> list[dict]:
    """The ``comfy_quant`` tags of a file, read from the file itself."""
    from safetensors import safe_open

    out = []
    with safe_open(path, framework="pt", device="cpu") as handle:
        for name in names:
            try:
                out.append(json.loads(bytes(handle.get_tensor(name).tolist()).decode("utf-8")))
            except (ValueError, UnicodeDecodeError):
                out.append({})
    return out


def _tag_name(tag: dict) -> str:
    """One ``comfy_quant`` tag as the panel names it."""
    name = FORMAT_NAMES.get(tag.get("format", ""), tag.get("format", "unknown"))
    if tag.get("convrot"):
        name += " convrot"
    if tag.get("group_size"):
        name += f" group {tag['group_size']}"
    return name


def _split_linear():
    """The module a feed-forward output layer stored in column slices runs as."""
    import torch

    class WASSplitLinear(torch.nn.Module):
        """One linear layer held as column slices whose outputs add."""

        def __init__(self, parts):
            super().__init__()
            setattr(self, PARTS, torch.nn.ModuleList(parts))
            self.widths = [part.in_features for part in parts]

        def forward(self, x):
            out = None
            for part, piece in zip(getattr(self, PARTS), x.split(self.widths, dim=-1)):
                y = part(piece)
                out = y if out is None else out + y
            return out

    return WASSplitLinear


def _encoder(config: dict, dtype_llama, quantization, parts: int = 0):
    """The encoder class ComfyUI's ``CLIP`` builds, carrying ``config``."""
    import comfy.sd1_clip
    import comfy.text_encoders.llama

    class WASQwen3Model(comfy.sd1_clip.SDClipModel):
        """One dense Qwen3 on ComfyUI's Qwen3 class, its shape taken from the checkpoint."""

        def __init__(self, device="cpu", layer="hidden", layer_idx=-2, dtype=None,
                     attention_mask=True, model_options={}):
            super().__init__(
                device=device, layer=layer, layer_idx=layer_idx,
                textmodel_json_config=dict(config), dtype=dtype, special_tokens={"pad": PAD},
                layer_norm_hidden_state=False, model_class=comfy.text_encoders.llama.Qwen3_8B,
                enable_attention_masks=attention_mask, return_attention_masks=attention_mask,
                model_options=model_options,
            )
            if parts:
                split = _split_linear()
                width = config["intermediate_size"] // parts
                for block in self.transformer.model.layers:
                    block.mlp.down_proj = split([
                        self.operations.Linear(width, config["hidden_size"], bias=False,
                                               device=device, dtype=dtype)
                        for _ in range(parts)
                    ])

    class WASQwen3TextModel(comfy.sd1_clip.SD1ClipModel):
        """The encoder a ``CLIP`` holds for one dense Qwen3."""

        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                dtype = dtype_llama
            if quantization is not None:
                model_options = {**model_options, "quantization_metadata": quantization}
            super().__init__(device=device, dtype=dtype, name=NAME, clip_model=WASQwen3Model,
                             model_options=model_options)

    return WASQwen3TextModel


def _tokenizer(width: int):
    """The tokenizer class ComfyUI's ``CLIP`` builds, writing Qwen's chat turns."""
    import comfy.sd1_clip
    import comfy.text_encoders.llama
    from transformers import Qwen2Tokenizer

    files = os.path.join(os.path.dirname(os.path.realpath(comfy.text_encoders.llama.__file__)),
                         "qwen25_tokenizer")

    class WASQwen3SDTokenizer(comfy.sd1_clip.SDTokenizer):
        """Qwen's vocabulary, unpadded, with no start or end token added."""

        def __init__(self, embedding_directory=None, tokenizer_data={}):
            super().__init__(
                files, pad_with_end=False, embedding_directory=embedding_directory,
                embedding_size=width, embedding_key=NAME, tokenizer_class=Qwen2Tokenizer,
                has_start_token=False, has_end_token=False, pad_to_max_length=False,
                max_length=99999999, min_length=1, pad_token=PAD, tokenizer_data=tokenizer_data,
            )

    class WASQwen3Tokenizer(comfy.sd1_clip.SD1Tokenizer):
        """A prompt as one Qwen chat turn, thinking off unless asked for."""

        def __init__(self, embedding_directory=None, tokenizer_data={}):
            super().__init__(embedding_directory=embedding_directory,
                             tokenizer_data=tokenizer_data, name=NAME,
                             tokenizer=WASQwen3SDTokenizer)
            self.llama_template = USER_TURN

        def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None,
                                  system_prompt="", thinking=False, skip_template=False,
                                  images=None, image=None, **kwargs):
            if images or image is not None:
                raise ValueError(
                    "This Qwen3 reads text only. Disconnect the image, or use a Qwen3-VL "
                    "model for pictures."
                )
            if skip_template or text.startswith(TURN_START):
                llama_text = text
            else:
                if llama_template is not None:
                    llama_text = llama_template.format(text)
                else:
                    llama_text = self.llama_template.format(text) + ("" if thinking else EMPTY_THINK)
                if system_prompt:
                    llama_text = SYSTEM_TURN.format(system_prompt) + llama_text
            return super().tokenize_with_weights(llama_text, return_word_ids=return_word_ids,
                                                 disable_weights=True, **kwargs)

    return WASQwen3Tokenizer


def _safetensors_dtypes() -> dict:
    """Safetensors dtype names mapped to torch dtypes."""
    import torch

    names = {
        "F64": torch.float64, "F32": torch.float32, "F16": torch.float16,
        "BF16": torch.bfloat16, "I64": torch.int64, "I32": torch.int32, "I16": torch.int16,
        "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool,
    }
    for name, attr in (("F8_E4M3", "float8_e4m3fn"), ("F8_E5M2", "float8_e5m2")):
        if hasattr(torch, attr):
            names[name] = getattr(torch, attr)
    return names

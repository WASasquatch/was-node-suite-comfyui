"""Weight quantization of a loaded diffusion model into ComfyUI's per-layer formats.

:func:`quantize` answers a state dict ComfyUI's diffusion-model loader reads: each chosen linear
layer as Comfy Kitchen stores it, beside a ``comfy_quant`` tag naming its format.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import torch

__all__ = ["ATTENTION", "FORMATS", "LAYERS", "QKV", "SCALES", "Report", "quantize"]


@dataclass(frozen=True)
class Format:
    """One quantization target.

    Attributes:
        tag: The ``format`` written into ``comfy_quant``.
        layout: The registered Comfy Kitchen layout that quantizes it.
        options: Keyword arguments the layout's quantizer takes.
        written: Extra ``comfy_quant`` keys the loader reads back.
        divides: What each input width must divide by, or 1.
        grouped: Whether ``group_size`` applies.
        finer: The menu name a protected layer moves to, or ``None`` to keep its precision.
    """

    tag: str
    layout: str
    options: dict = field(default_factory=dict)
    written: dict = field(default_factory=dict)
    divides: int = 1
    grouped: bool = False
    finer: str | None = None


#: Menu name -> format, the usual choice first.
FORMATS = {
    "int8 convrot": Format(
        "int8_tensorwise", "TensorWiseINT8Layout",
        {"per_channel": True, "convrot": True, "convrot_groupsize": 256},
        {"convrot": True, "convrot_groupsize": 256}, divides=16,
    ),
    "w6a8": Format(
        "w6a8_int8", "AsymW4A8Int8Layout",
        {"bits": 6, "codebook": True, "convrot_groupsize": 256},
        {"convrot": True, "convrot_groupsize": 256}, divides=256, grouped=True, finer="int8 convrot",
    ),
    "w4a8": Format(
        "asym_w4a8_int8", "AsymW4A8Int8Layout",
        {"bits": 4, "codebook": True, "convrot_groupsize": 256},
        {"convrot": True, "convrot_groupsize": 256}, divides=256, grouped=True, finer="int8 convrot",
    ),
    "int8": Format("int8_tensorwise", "TensorWiseINT8Layout", {"per_channel": True}),
    "fp8 e4m3fn": Format("float8_e4m3fn", "TensorCoreFP8E4M3Layout", {"scale": "recalculate"}),
    "mxfp8": Format("mxfp8", "TensorCoreMXFP8Layout", finer="int8 convrot"),
    "nvfp4": Format("nvfp4", "TensorCoreNVFP4Layout", {"scale": "recalculate"}, finer="int8 convrot"),
}

#: Group size each grouped format uses when none is given.
GROUP_DEFAULT = {"w6a8": 32, "w4a8": 16}

#: Hadamard sizes an int8 convrot layer rotates by, the largest its input width divides taken.
ROTATIONS = (256, 64, 16)

#: Rows a grouped format needs; a layer with fewer takes int8 convrot.
GROUPED_ROWS = 64

#: The widest input a grouped format's kernels take on a GPU with 99 KB of shared memory per
#: block; a wider layer takes int8 convrot.
GROUPED_WIDEST = 17408

#: Which linear layers are quantized.
LAYERS = ("like Comfy-Org", "transformer blocks", "every linear layer")

#: How each int8 row's scale is chosen.
SCALES = ("like Comfy-Org", "row maximum", "lowest error")

#: Clip ratios of the row maximum the lowest-error scale is chosen from, smallest first.
CLIPS = torch.linspace(0.55, 1.0, 80).tolist()

#: A path segment Comfy-Org's quantizer leaves unquantized wherever it appears.
COMFY_ORG_SKIPPED = re.compile(
    r"scale_shift|rope|rotary|rel_pos|pos_?embed|embedder|gate_logits|router|routing|logit|"
    r"temperature|(?:^|_)time|temb|t_emb|guidance|register|refiner_blocks|adapter|"
    r"(?:^|_)(?:final|head|proj_out|out_layer)(?:_|$)"
)

#: The narrowest side a layer Comfy-Org's quantizer quantizes has, and its fewest rows.
COMFY_ORG_NARROWEST = 256
COMFY_ORG_ROWS = 8

#: The smallest magnitude a row maximum or scale is held to.
SMALLEST = 1e-30


@dataclass(frozen=True)
class Release:
    """How Comfy-Org's own file of one model in one format was made.

    Attributes:
        quantized: The layers it quantizes, or ``None`` for Comfy-Org's quantizer's own choice.
        searched: Whether each int8 row's scale is the clip with the lowest error.
        lowered: Float32 tensors it writes in bfloat16 except keys ending in one of these, or
            ``None`` to keep every tensor's precision.
        full_precision: The layers it tags to multiply at full precision, or ``None``.
    """

    quantized: re.Pattern | None = None
    searched: bool = False
    lowered: tuple | None = None
    full_precision: re.Pattern | None = None


#: Krea 2's fp8, mxfp8 and nvfp4 releases: every block layer, the text-fusion ones included,
#: with the attention gate and output and the feed-forward output tagged for full precision.
KREA2_FLOAT_RELEASE = Release(
    re.compile(r"^(blocks|txtfusion\.layerwise_blocks|txtfusion\.refiner_blocks)\.\d+\."),
    lowered=(), full_precision=re.compile(r"\.(attn\.gate|attn\.wo|mlp\.down)$"),
)

#: Fibo 2.0's int8 release, and the same layers for the grouped formats: every block layer but the modulation.
FIBO2_INT8_RELEASE = Release(re.compile(r"^(layers|noise_refiner|perceiver\.layers)\.\d+\.(?!adaLN_modulation)"))

#: Model module -> menu name -> how Comfy-Org's own release in that format was made.
COMFY_ORG = {
    "comfy.ldm.fibo2.model": {"int8 convrot": FIBO2_INT8_RELEASE, "w6a8": FIBO2_INT8_RELEASE,
                              "w4a8": FIBO2_INT8_RELEASE},
    "comfy.ldm.krea2.model": {
        "int8 convrot": Release(re.compile(r"^blocks\.\d+\."), searched=True, lowered=(".scale",)),
        "fp8 e4m3fn": KREA2_FLOAT_RELEASE,
        "mxfp8": KREA2_FLOAT_RELEASE,
        "nvfp4": KREA2_FLOAT_RELEASE,
    },
}

#: How attention's query, key and value projections are written.
QKV = ("as loaded", "fused", "separate")

#: Model module -> its separate projection names, for the models built in either form. The
#: fused form is one layer named ``qkv`` beside them.
SPLIT_NAMES = {
    "comfy.ldm.flux3.model": ("q_proj", "k_proj", "v_proj"),
    "comfy.ldm.moss_image.model": ("q", "k", "v"),
}

#: The fused projection's name, and how many projections it holds.
FUSED_NAME = "qkv"
PROJECTIONS = 3

#: What is written for each attention layer's kernel.
ATTENTION = ("as loaded", "like Comfy-Org", "int8 everywhere", "clear")

#: The kernel setting Comfy-Org's quantized files carry.
INT8_ATTENTION = {"attention": "comfy_kitchen_int8"}

#: Model module -> the attention layers Comfy-Org's own quantized release sets to int8.
COMFY_ORG_ATTENTION = {
    "comfy.ldm.flux3.model": re.compile(r"^(content_mode_blocks\.image\.\d+|single_blocks\.\d+)$"),
    "comfy.ldm.fibo2.model": re.compile(r"^layers\.([0-9]|[1-3][0-9])\.attention$"),
}

#: Model module -> key endings renamed back to the release's own names when written.
RELEASE_NAMES = {
    "comfy.ldm.flux3.model": (("_norm.weight", "_norm.scale"),),
}

#: The attribute an attention layer keeps its kernel setting under.
ATTENTION_CHILD = "comfy_attention"

#: A path segment naming a layer that conditions, embeds, normalises or ends the model.
KEPT = re.compile(
    r"^(.*adaln.*|.*modulation.*|.*norm.*|.*emb.*|time.*|timestep.*|txt_in|img_in|.*patch.*|"
    r"final.*|proj_out|lm_head|head)$",
    re.IGNORECASE,
)

#: Layers listed by error in the report.
WORST_SHOWN = 12


@dataclass
class Report:
    """What a quantization did.

    Attributes:
        formats: Layers written in each format, by menu name.
        kept: Layers left at the model's own precision.
        bytes_before: Size of the model's tensors before.
        bytes_after: Size after.
        worst: The largest errors as ``(layer, menu name, relative error)``.
        protected: Layers moved one step finer.
        refitted: Layers whose width did not suit the format and took another.
        qkv: What became of the attention projections, in words.
        projected: Bytes the same layers would take in each format, by menu name.
        attention: What became of the attention kernel settings, in words.
        scales: How the int8 row scales were chosen, in words.
        recipe: Whose layer choice was followed, in words.
    """

    formats: dict = field(default_factory=dict)
    kept: int = 0
    bytes_before: int = 0
    bytes_after: int = 0
    worst: list = field(default_factory=list)
    protected: int = 0
    refitted: int = 0
    qkv: str = "as loaded"
    projected: dict = field(default_factory=dict)
    attention: str = "as loaded"
    scales: str = "row maximum"
    recipe: str = ""


def quantize(patcher, menu_name: str, layers: str, protect: float, group_size: int,
             qkv: str = QKV[0], attention: str = ATTENTION[0], scales: str = SCALES[0],
             progress=None):
    """Quantize a loaded model's linear layers and answer the state dict to rebuild it from.

    Args:
        patcher: The ``ModelPatcher`` a MODEL socket carries. Its patches are merged first.
        menu_name: A key of :data:`FORMATS`.
        layers: One of :data:`LAYERS`.
        protect: Share of the quantized layers, 0 to 1, with the largest error moved one step finer.
        group_size: Group size of a grouped format, 0 for the format's own.
        qkv: One of :data:`QKV`.
        attention: One of :data:`ATTENTION`.
        scales: One of :data:`SCALES`.
        progress: Called with ``(done, total)`` after each layer.

    Returns:
        ``(state_dict, report)``: CPU tensors keyed as the diffusion model names them, and a
        :class:`Report`.

    Raises:
        ValueError: ``menu_name``, ``layers``, ``qkv``, ``attention`` or ``scales`` names nothing
            offered, or the format's layout is not available in this ComfyUI.
    """
    import comfy.model_management
    import comfy.quant_ops

    if menu_name not in FORMATS:
        raise ValueError(f"No quantization format {menu_name!r}. Choose one of: {', '.join(FORMATS)}.")
    if layers not in LAYERS:
        raise ValueError(f"No layer choice {layers!r}. Choose one of: {', '.join(LAYERS)}.")
    if qkv not in QKV:
        raise ValueError(f"No projection choice {qkv!r}. Choose one of: {', '.join(QKV)}.")
    if attention not in ATTENTION:
        raise ValueError(f"No attention choice {attention!r}. Choose one of: {', '.join(ATTENTION)}.")
    if scales not in SCALES:
        raise ValueError(f"No scale choice {scales!r}. Choose one of: {', '.join(SCALES)}.")
    if comfy.quant_ops.get_layout_class(FORMATS[menu_name].layout) is None:
        raise ValueError(
            f"{menu_name} is not available in this ComfyUI. Update ComfyUI and comfy-kitchen, or "
            "pick int8 convrot."
        )
    group = int(group_size) or GROUP_DEFAULT.get(menu_name, 16)

    model = patcher.model.diffusion_model
    release = COMFY_ORG.get(_family(model), {}).get(menu_name) if layers == LAYERS[0] else None
    searched = scales == "lowest error" or (scales == SCALES[0] and release is not None and release.searched)
    lowered = release.lowered if release is not None else None
    tagged = release.full_precision if release is not None else None
    candidates = _candidates(model, layers, release)
    chosen = set(candidates)
    plan, words = _plan(model, qkv)
    report = Report(qkv=words, scales="lowest error" if searched else "row maximum")
    if layers == LAYERS[0]:
        report.recipe = "Comfy-Org's release" if release is not None else "Comfy-Org's quantizer"
    else:
        report.recipe = layers
    regrouped = {source for parts in plan.values() for source, _ in parts}

    # Every output layer, with the source layers it is read from: itself, three fused, or a chunk.
    outputs = {path: [(path, None)] for path in candidates if path not in regrouped}
    for out, parts in plan.items():
        if all(source in chosen for source, _ in parts):
            outputs[out] = parts

    saved = patcher.model_state_dict_for_saving(model, "diffusion_model.")
    state = {}
    for key, value in saved.items():
        report.bytes_before += _bytes(value)
        module = key.rpartition(".")[0]
        if module in regrouped or (module in outputs and not key.endswith(".bias")):
            continue
        state[key] = _lowered(_cpu(value), key, lowered)

    for out, parts in plan.items():
        biases = [saved.get(f"{source}.bias") for source, _ in parts]
        if all(bias is not None for bias in biases):
            state[f"{out}.bias"] = _joined([_cpu(bias) for bias in biases], parts)
        if out not in outputs:
            weight = _joined([_weight(patcher, source) for source, _ in parts], parts)
            state[f"{out}.weight"] = weight.to("cpu", dtype=_stored_dtype(weight))

    errors = {}
    shapes = {}
    total = len(outputs)
    for done, (out, parts) in enumerate(outputs.items(), 1):
        comfy.model_management.throw_exception_if_processing_interrupted()
        weight = _joined([_weight(patcher, source) for source, _ in parts], parts)
        shapes[out] = (int(weight.shape[0]), int(weight.shape[1]))
        full = tagged is not None and tagged.search(out) is not None
        name, tensors, error = _quantize_one(weight, out, menu_name, group, searched, full)
        state.update(tensors)
        errors[out] = (name, error)
        del weight
        if progress is not None:
            progress(done, total)

    if protect > 0 and errors:
        ranked = sorted(errors, key=lambda out: errors[out][1], reverse=True)
        count = int(round(min(1.0, protect) * len(ranked)))
        for out in ranked[:count]:
            name = errors[out][0]
            finer = FORMATS[name].finer if name in FORMATS else None
            for key in [k for k in state if k.startswith(f"{out}.") and not k.endswith(".bias")]:
                del state[key]
            parts = outputs[out]
            weight = _joined([_weight(patcher, source) for source, _ in parts], parts)
            if finer is None:
                kept = weight.to("cpu", dtype=_stored_dtype(weight))
                state[f"{out}.weight"] = _lowered(kept, f"{out}.weight", lowered)
                errors[out] = (None, 0.0)
            else:
                full = tagged is not None and tagged.search(out) is not None
                name, tensors, error = _quantize_one(weight, out, finer, group, searched, full)
                state.update(tensors)
                errors[out] = (name, error)
            report.protected += 1
            del weight

    for out, (name, error) in errors.items():
        if name is None:
            report.kept += 1
        else:
            report.formats[name] = report.formats.get(name, 0) + 1
            if name != menu_name and FORMATS[menu_name].finer != name:
                report.refitted += 1
    report.worst = sorted(
        ((out, name, error) for out, (name, error) in errors.items() if name is not None),
        key=lambda row: row[2], reverse=True,
    )[:WORST_SHOWN]
    report.bytes_after = sum(_bytes(tensor) for tensor in state.values())
    held = sum(
        _bytes(tensor) for key, tensor in state.items()
        if key.rpartition(".")[0] in errors and not key.endswith(".bias")
    )
    rest = report.bytes_after - held
    for name in FORMATS:
        grouping = int(group_size) or GROUP_DEFAULT.get(name, 16)
        report.projected[name] = rest + sum(_layer_bytes(name, n, k, grouping) for n, k in shapes.values())
    report.attention = _attention(model, state, attention)
    state = _release_names(model, state)
    comfy.model_management.soft_empty_cache()
    return state, report


def _family(model) -> str:
    """The module a model's rules are kept under: its own, or the ComfyUI module it stands in for."""
    return getattr(model, "comfy_family", type(model).__module__)


def _attention(model, state: dict, choice: str) -> str:
    """Write each attention layer's kernel setting as asked, and say what was done.

    Args:
        model: The diffusion model.
        state: The state dict being written, changed in place.
        choice: One of :data:`ATTENTION`.

    Returns:
        A short line for the report.
    """
    layers = [
        path.rpartition(".")[0] for path, module in model.named_modules()
        if path.rpartition(".")[2] == ATTENTION_CHILD and type(module).__name__ == "ComfyAttention"
    ]
    keys = {layer: f"{layer}.{ATTENTION_CHILD}.config" for layer in layers}
    if choice == ATTENTION[0]:
        return f"as loaded, {sum(key in state for key in keys.values())} of {len(layers)} set"
    for key in keys.values():
        state.pop(key, None)
    if choice == "clear":
        return f"cleared on {len(layers)}"
    if choice == "int8 everywhere":
        chosen = layers
    else:
        rule = COMFY_ORG_ATTENTION.get(_family(model))
        if rule is None:
            return "no Comfy-Org pattern for this model, none set"
        chosen = [layer for layer in layers if rule.match(layer)]
    encoded = list(json.dumps(INT8_ATTENTION).encode("utf-8"))
    for layer in chosen:
        state[keys[layer]] = torch.tensor(encoded, dtype=torch.uint8)
    return f"int8 on {len(chosen)} of {len(layers)}"


def _release_names(model, state: dict) -> dict:
    """The state dict with keys ComfyUI renamed on loading given their release names again."""
    renames = RELEASE_NAMES.get(_family(model), ())
    if not renames:
        return state
    named = {}
    for key, value in state.items():
        for loaded, released in renames:
            if key.endswith(loaded):
                key = key[: -len(loaded)] + released
                break
        named[key] = value
    return named


def _plan(model, qkv: str):
    """The attention projections to fuse or split, and the words that say what was done.

    Returns:
        ``(plan, words)``: output layer path -> ``[(source path, chunk or None), ...]``, and a
        short line for the report.
    """
    names = SPLIT_NAMES.get(_family(model))
    if qkv == QKV[0]:
        return {}, "as loaded"
    if names is None:
        return {}, "as loaded, this model has one form"
    modules = dict(model.named_modules())
    plan = {}
    if qkv == "fused":
        for path in modules:
            parent, _, leaf = path.rpartition(".")
            if leaf != names[0]:
                continue
            sources = [f"{parent}.{name}" if parent else name for name in names]
            if all(isinstance(modules.get(source), torch.nn.Linear) for source in sources):
                out = f"{parent}.{FUSED_NAME}" if parent else FUSED_NAME
                plan[out] = [(source, None) for source in sources]
        return plan, f"fused in {len(plan)} attentions" if plan else "already fused"
    for path, module in modules.items():
        parent, _, leaf = path.rpartition(".")
        if leaf != FUSED_NAME or not isinstance(module, torch.nn.Linear):
            continue
        for chunk, name in enumerate(names):
            out = f"{parent}.{name}" if parent else name
            plan[out] = [(path, chunk)]
    count = len(plan) // len(names)
    return plan, f"separated in {count} attentions" if plan else "already separate"


def _joined(tensors, parts):
    """Concatenate fused sources, or take a separated source's chunk."""
    if len(parts) > 1:
        return torch.cat(tensors, dim=0)
    chunk = parts[0][1]
    if chunk is None:
        return tensors[0]
    return tensors[0].chunk(PROJECTIONS, dim=0)[chunk].contiguous()


def _cpu(value):
    """A saved value as a CPU tensor of its own."""
    tensor = value.to("cpu") if isinstance(value, torch.Tensor) else value
    return tensor.detach().clone() if isinstance(tensor, torch.Tensor) else tensor


def _candidates(model, layers: str, release: Release | None) -> list[str]:
    """Paths of the linear layers to quantize, in model order."""
    stacks = {path for path, module in model.named_modules() if isinstance(module, torch.nn.ModuleList)}
    chosen = []
    for path, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if not isinstance(module, torch.nn.Linear) or weight is None or weight.dim() != 2:
            continue
        segments = path.split(".")
        if layers == LAYERS[0]:
            if release is not None and release.quantized is not None:
                if release.quantized.search(path) is None:
                    continue
            elif not _comfy_org_quantizes(segments, int(weight.shape[0]), int(weight.shape[1])):
                continue
        elif layers == LAYERS[1]:
            # Inside one member of a block stack, not a stack member of its own.
            if not any(".".join(segments[:end]) in stacks for end in range(1, len(segments) - 1)):
                continue
            if any(KEPT.match(segment) for segment in segments):
                continue
        chosen.append(path)
    return chosen


def _comfy_org_quantizes(segments: list[str], rows: int, width: int) -> bool:
    """Whether Comfy-Org's quantizer quantizes a linear layer at this path and shape."""
    if rows < COMFY_ORG_ROWS or width % ROTATIONS[-1] or min(rows, width) < COMFY_ORG_NARROWEST:
        return False
    # A block index has more of the path after it; a trailing one numbers a sequence's members.
    if not any(segment.isdigit() for segment in segments[:-1]):
        return False
    return not any(COMFY_ORG_SKIPPED.search(segment) for segment in segments)


def _lowered(value, key: str, lowered: tuple | None):
    """A float32 tensor in bfloat16 when the release writes it so, else the value as it is."""
    if lowered is None or not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
        return value
    return value if key.endswith(tuple(lowered)) else value.to(torch.bfloat16)


def _weight(patcher, path: str) -> torch.Tensor:
    """One layer's weight with the patcher's patches merged, as a float tensor on its device."""
    from comfy.quant_ops import QuantizedTensor

    weight = patcher.patch_weight_to_device(
        f"diffusion_model.{path}.weight", device_to=patcher.load_device, return_weight=True
    )
    if isinstance(weight, QuantizedTensor):
        weight = weight.dequantize()
    if weight.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        weight = weight.to(torch.bfloat16)
    return weight


def _stored_scale(value: torch.Tensor) -> torch.Tensor:
    """A power-of-two block scale as the bytes it holds, the rest as they are."""
    power = getattr(torch, "float8_e8m0fnu", None)
    return value.view(torch.uint8) if power is not None and value.dtype == power else value


def _stored_dtype(weight: torch.Tensor):
    """The dtype a kept layer is written in."""
    return weight.dtype if weight.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.bfloat16


def _fits(menu_name: str, rows: int, width: int, group: int) -> bool:
    """Whether a layer of this shape can take the format."""
    target = FORMATS[menu_name]
    if width % target.divides:
        return False
    if target.grouped:
        if rows < GROUPED_ROWS or width > GROUPED_WIDEST or width % group:
            return False
        if target.tag == "w6a8_int8" and (width % 32 or group % 16):
            return False
        if target.tag == "asym_w4a8_int8" and (width % 16 or group < 4 or not (16 % group == 0 or group % 16 == 0)):
            return False
    return True


def _quantize_one(weight: torch.Tensor, path: str, menu_name: str, group: int,
                  searched: bool = False, full_precision: bool = False):
    """Quantize one weight in the format, or the nearest that its shape allows.

    Args:
        weight: The layer's float weight.
        path: The layer's path in the model.
        menu_name: A key of :data:`FORMATS`.
        group: Group size of a grouped format.
        searched: Whether an int8 row's scale is the clip with the lowest error.
        full_precision: Whether the layer is tagged to multiply at full precision.

    Returns:
        ``(menu name, tensors, relative error)``: the layer's weight, scales and ``comfy_quant``
        keyed under its path, on the CPU.
    """
    from comfy.quant_ops import QuantizedTensor

    rows, width = int(weight.shape[0]), int(weight.shape[1])
    for name in (menu_name, "int8 convrot", "int8"):
        if _fits(name, rows, width, group):
            break
    target = FORMATS[name]
    options = dict(target.options)
    written = {"format": target.tag}
    if target.grouped:
        options["group_size"] = group
        written["group_size"] = group
    written.update(target.written)
    if target.tag == "int8_tensorwise":
        rotation = 0
        if target.options.get("convrot"):
            rotation = next(size for size in ROTATIONS if width % size == 0)
            written["convrot_groupsize"] = rotation
        codes, scale, error = _int8_rows(weight, rotation, searched)
        tensors = {f"{path}.weight": codes, f"{path}.weight_scale": scale}
    elif target.tag == "float8_e4m3fn":
        codes, scale, error = _fp8_tensor(weight)
        tensors = {f"{path}.weight": codes, f"{path}.weight_scale": scale}
    else:
        import comfy.model_management

        # Grouped formats start from bfloat16 on the GPU, nvfp4 from the weight's own precision on
        # the GPU, mxfp8 from its own precision on the CPU.
        if target.grouped:
            exact = weight.to(comfy.model_management.get_torch_device(), torch.bfloat16)
        elif target.tag == "nvfp4":
            exact = weight.to(comfy.model_management.get_torch_device())
        else:
            exact = weight.to("cpu")
        quantized = QuantizedTensor.from_float(exact, target.layout, **options)
        restored = quantized.dequantize().float()
        error = float((restored - exact.float()).norm() / exact.float().norm().clamp_min(1e-12))
        tensors = {
            key: _stored_scale(value.detach().to("cpu"))
            for key, value in quantized.state_dict(f"{path}.weight").items()
        }
    if full_precision:
        written["full_precision_matrix_mult"] = True
    tensors[f"{path}.comfy_quant"] = torch.tensor(
        list(json.dumps(written).encode("utf-8")), dtype=torch.uint8
    )
    return name, tensors, error


def _int8_rows(weight: torch.Tensor, rotation: int, searched: bool = False):
    """Row-wise INT8 as Comfy-Org's releases hold it, bit for bit.

    Args:
        weight: The layer's weight, ``(rows, width)``.
        rotation: Hadamard group size, 0 for none.
        searched: Whether each row's scale is the clip with the lowest error.

    Returns:
        ``(codes, scale, relative error)``: int8 ``(rows, width)`` and float32 ``(rows, 1)``,
        both on the CPU.

    Raises:
        ValueError: Comfy Kitchen is not installed in this ComfyUI.
    """
    exact = weight.detach().to("cpu", torch.float32)
    if rotation:
        try:
            from comfy_kitchen.tensor import int8_utils
        except ImportError as error:
            raise ValueError(
                "int8 convrot needs comfy-kitchen, which ships with ComfyUI. Update ComfyUI, or "
                "pick plain int8."
            ) from error
        # Rotated in float32 on the CPU by Comfy Kitchen's own Hadamard blocks.
        blocks = int8_utils._build_hadamard(rotation, device="cpu", dtype=torch.float32)
        exact = int8_utils._rotate_weight(exact, blocks, rotation)
    # Each row's largest magnitude times the float32 reciprocal of 127; codes round half to even.
    largest = exact.abs().amax(dim=-1, keepdim=True).clamp(min=SMALLEST)
    scale = (largest * (1.0 / 127.0)).clamp(min=SMALLEST)
    if searched:
        import comfy.model_management

        scale = _clipped(exact, largest, scale, comfy.model_management.get_torch_device())
    codes = torch.round(exact / scale).clamp(-127, 127).to(torch.int8)
    # The rotation is orthonormal, so the error measured in its basis is the layer's own.
    error = float((codes.float() * scale - exact).norm() / exact.norm().clamp_min(1e-12))
    return codes, scale, error


def _clipped(exact: torch.Tensor, largest: torch.Tensor, scale: torch.Tensor,
             device: torch.device | None = None) -> torch.Tensor:
    """Each row's scale at the clip of :data:`CLIPS` with the lowest mean squared error.

    Args:
        exact: The rotated float32 weight on the CPU, ``(rows, width)``.
        largest: Each row's largest magnitude, ``(rows, 1)``.
        scale: The row-maximum scales, answered for a row no clip improves on.
        device: A GPU to rank the clips on first, or ``None``.

    Returns:
        Float32 scales ``(rows, 1)`` as the CPU's errors rank them; on a tie the smaller clip wins.
    """
    import comfy.model_management

    if device is None or device.type == "cpu":
        lowest = torch.full_like(largest, float("inf"))
        for clip in CLIPS:
            comfy.model_management.throw_exception_if_processing_interrupted()
            trial = (largest * clip * (1.0 / 127.0)).clamp(min=SMALLEST)
            codes = torch.round(exact / trial).clamp_(-127, 127)
            error = codes.mul_(trial).sub_(exact).square_().mean(dim=-1, keepdim=True)
            better = error < lowest
            lowest = torch.where(better, error, lowest)
            scale = torch.where(better, trial, scale)
        return scale

    # Every clip's error on the GPU, summed in another order than the CPU's.
    placed, top = exact.to(device), largest.to(device)
    ranked = torch.empty((len(CLIPS), exact.shape[0]), device=device)
    for index, clip in enumerate(CLIPS):
        comfy.model_management.throw_exception_if_processing_interrupted()
        trial = (top * clip * (1.0 / 127.0)).clamp(min=SMALLEST)
        codes = torch.round(placed / trial).clamp_(-127, 127)
        ranked[index] = codes.mul_(trial).sub_(placed).square_().mean(dim=-1)
    del placed, top, codes
    # The clips within float32 summation error of the GPU's best, which hold the CPU's best.
    width = exact.shape[1]
    reach = 4.0 * width * 2.0 ** -24 + 1e-4
    close = (ranked <= ranked.min(dim=0, keepdim=True).values * (1.0 + reach)).cpu()
    del ranked
    errors = torch.full((len(CLIPS), exact.shape[0]), float("inf"))
    for index, clip in enumerate(CLIPS):
        rows = close[index].nonzero().flatten()
        if not len(rows):
            continue
        # Two rows or more at a time, each summed as in the whole layer.
        picked = rows if len(rows) > 1 else rows.repeat(2)
        trial = (largest[picked] * clip * (1.0 / 127.0)).clamp(min=SMALLEST)
        codes = torch.round(exact[picked] / trial).clamp_(-127, 127)
        error = codes.mul_(trial).sub_(exact[picked]).square_().mean(dim=-1)
        errors[index, rows] = error[: len(rows)]
    chosen = errors.argmin(dim=0)
    clips = torch.tensor(CLIPS, dtype=torch.float32)[chosen].unsqueeze(-1)
    return (largest * clips * (1.0 / 127.0)).clamp(min=SMALLEST)


def _fp8_tensor(weight: torch.Tensor):
    """Tensor-wise float8 e4m3fn as Comfy-Org's releases hold it, bit for bit.

    Args:
        weight: The layer's weight, ``(rows, width)``.

    Returns:
        ``(codes, scale, relative error)``: float8 ``(rows, width)`` and a float32 scalar, both
        on the CPU.
    """
    exact = weight.detach().to("cpu", torch.float32)
    largest = torch.finfo(torch.float8_e4m3fn).max
    # The largest magnitude times the float32 reciprocal of the format's largest value.
    scale = exact.abs().amax() * (1.0 / largest)
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))
    codes = (exact / scale).clamp(-largest, largest).to(torch.float8_e4m3fn)
    error = float((codes.float() * scale - exact).norm() / exact.norm().clamp_min(1e-12))
    return codes, scale, error


def _layer_bytes(menu_name: str, rows: int, width: int, group: int) -> int:
    """Bytes one layer takes in a format, or in the format its width falls back to.

    Args:
        menu_name: A key of :data:`FORMATS`.
        rows: Output width.
        width: Input width.
        group: Group size of a grouped format.

    Returns:
        Bytes of its weight, scales and tag.
    """
    for name in (menu_name, "int8 convrot", "int8"):
        if _fits(name, rows, width, group):
            break
    tag = 80
    if name in ("int8 convrot", "int8"):
        return rows * width + 4 * rows + tag
    if name == "w6a8":
        return rows * (3 * width // 4) + rows * (width // group) + 4 * rows + tag
    if name == "w4a8":
        return rows * (width // 2) + rows * (width // group) + 4 * rows + 64 + tag
    if name == "fp8 e4m3fn":
        return rows * width + 4 + tag
    if name == "mxfp8":
        tall, wide = -(-rows // 32) * 32, -(-width // 32) * 32
        return tall * wide + tall * (wide // 32) + tag
    tall, wide = -(-rows // 16) * 16, -(-width // 16) * 16
    return tall * (wide // 2) + tall * (wide // 16) + 4 + tag


def _bytes(value) -> int:
    """Bytes a tensor holds, 0 for anything else."""
    return value.numel() * value.element_size() if isinstance(value, torch.Tensor) else 0

"""Per-layer block-sparse attention for MiniMax H3 on comfy_kitchen's Sol-Attn kernel.

Each DiT block runs dense on the model's attention backend, or sparse at its own setting under
one of :data:`METHODS`. Sparse blocks compute in INT8.
"""

from __future__ import annotations

import contextlib
import math
import re

from .. import log

__all__ = ["BLOCK", "CHUNK", "CUBE", "HEAD_DIM", "MANUAL", "METHODS", "PRESETS", "SENSITIVE",
           "SINKS", "blocks_of", "describe", "kernel_problem", "parse_plan", "sparse"]

logger = log.get_logger("sampling.h3_sparse_attention")

#: Head width the Sol-Attn kernel takes.
HEAD_DIM = 128

#: Rows in one attention block of the kernel.
BLOCK = 64

#: Tokens projected into the kernel at once.
CHUNK = 4096

#: Frames, rows and columns of video tokens in one VSA tile, 64 tokens in all.
CUBE = (4, 4, 4)

#: VSA tilings kept for reuse, one per latent shape.
TILINGS_KEPT = 4

#: How each method picks the key blocks a query reads exactly: what a layer's number means, as
#: ``(word, smallest, largest)``, in the order the widget offers them.
METHODS = {
    "sol-attn": ("tau", 0.0, 4.0),
    "sla": ("keep percent", 0.5, 95.0),
    "vsa": ("keep percent", 0.5, 95.0),
}

#: Layers of MiniMax H3 left dense by default: the ones its video suffers most for making sparse.
SENSITIVE = "39-42, 49"

#: The layer presets, in the order the widget offers them, each a plan as :func:`parse_plan` reads it.
PRESETS = {
    "balanced": SENSITIVE,
    "careful": "0-1, 37-43, 49",
    "fastest": "",
}

#: The choice that reads a typed plan in place of a preset.
MANUAL = "manual"

#: How the packed conditioning rows are treated, in the order the widget offers them.
SINKS = ["exact_kv_and_rows", "exact_kv", "off"]


def _outside_graph():
    """A context whose allocations stay out of ComfyUI's per-block allocation record."""
    try:
        import comfy.model_prefetch as prefetch
    except ImportError:
        return contextlib.nullcontext()
    pause = getattr(prefetch, "pause_malloc_graph", None)
    return pause() if pause is not None else contextlib.nullcontext()


def parse_plan(text: str, value: float, layers: int, method: str = "sol-attn") -> list[float | None]:
    """Each layer's setting from a written plan.

    Args:
        text: Entries separated by commas, semicolons or lines. ``39-42`` or ``49`` alone runs
            those layers dense; ``0-3: 0.5`` gives them their own setting; ``20: dense`` runs one
            dense. A later entry wins over an earlier one.
        value: Setting of every layer ``text`` does not name.
        layers: Layers in the model.
        method: One of :data:`METHODS`, which sets the range a setting may take.

    Returns:
        One entry per layer: its setting, or ``None`` for dense.

    Raises:
        ValueError: An entry names no layer, a layer the model does not have, or a value that is
            neither ``dense`` nor in the method's range.
    """
    word, low, high = METHODS.get(method, METHODS["sol-attn"])
    plan: list[float | None] = [float(value)] * int(layers)
    for raw in re.split(r"[,;\n]", text or ""):
        entry = raw.strip()
        if not entry:
            continue
        span, _, setting_text = entry.partition(":")
        match = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+)\s*)?", span)
        if match is None:
            raise ValueError(
                f"layer_plan entry `{entry}` names no layer. Write a layer as `49`, a range as "
                f"`39-42`, and a setting after a colon, as `0-3: 0.5` or `20: dense`.")
        first = int(match.group(1))
        last = int(match.group(2)) if match.group(2) is not None else first
        first, last = min(first, last), max(first, last)
        if last >= layers:
            raise ValueError(
                f"layer_plan entry `{entry}` names layer {last}, but the model has layers 0 to "
                f"{layers - 1}.")
        setting_text = setting_text.strip().lower()
        if setting_text in ("", "dense"):
            setting = None
        else:
            try:
                setting = float(setting_text)
            except ValueError:
                setting = math.nan
            if not low <= setting <= high:
                raise ValueError(
                    f"layer_plan entry `{entry}` sets `{setting_text}`. Under {method} a layer "
                    f"takes `dense` or a {word} from {low:g} to {high:g}.")
        for layer in range(first, last + 1):
            plan[layer] = setting
    return plan


def _spans(layers: list[int]) -> str:
    """``[39, 40, 41, 42, 49]`` as ``39-42, 49``."""
    spans, start = [], None
    for index, layer in enumerate(layers):
        if start is None:
            start = layer
        if index + 1 == len(layers) or layers[index + 1] != layer + 1:
            spans.append(str(start) if start == layer else f"{start}-{layer}")
            start = None
    return ", ".join(spans)


def describe(plan: list[float | None], method: str = "sol-attn") -> str:
    """One line naming which layers run dense and which sparse, at what setting.

    Args:
        plan: One entry per layer, as :func:`parse_plan` returns it.
        method: One of :data:`METHODS`.

    Returns:
        The summary.
    """
    word = METHODS.get(method, METHODS["sol-attn"])[0]
    dense = [layer for layer, setting in enumerate(plan) if setting is None]
    settings: dict[float, list[int]] = {}
    for layer, setting in enumerate(plan):
        if setting is not None:
            settings.setdefault(setting, []).append(layer)
    shown = (lambda setting: f"tau {setting:g}") if word == "tau" else (lambda setting: f"keep {setting:g}%")
    parts = [f"{shown(setting)} on layers {_spans(found)}" for setting, found in sorted(settings.items())]
    if dense:
        parts.append(f"dense on layers {_spans(dense)}")
    return f"{method}, {len(plan) - len(dense)} of {len(plan)} layers sparse: " + "; ".join(parts)


def kernel_problem(device=None) -> str | None:
    """Why the Sol-Attn kernel cannot run here, or ``None`` where it can.

    Args:
        device: The device to ask about, or ``None`` for the current CUDA device.

    Returns:
        A sentence naming what is missing.
    """
    try:
        import comfy_kitchen as ck
    except ImportError:
        return "comfy_kitchen is not installed with this ComfyUI"
    if not hasattr(ck, "sol_attn_chunked") or not hasattr(ck, "sol_attn_is_available"):
        return "this comfy_kitchen has no Sol-Attn kernel; update ComfyUI"
    import torch

    if not torch.cuda.is_available():
        return "Sol-Attn runs on NVIDIA GPUs only"
    device = torch.device("cuda", torch.cuda.current_device()) if device is None else device
    if not ck.sol_attn_is_available(device):
        major, minor = torch.cuda.get_device_capability(device)
        return f"Sol-Attn needs an RTX 30-series or newer GPU (sm_80), and this one is sm_{major}{minor}"
    return None


def _tiling(layout, device) -> dict:
    """The VSA tile order: padded 64-row conditioning tiles, then :data:`CUBE` video cubes.

    Returns:
        ``{"rows", "prefix", "source", "place", "lengths"}``: the padded row count, the tiles
        ahead of the video, each padded row's source row (-1 for padding), each source row's
        padded place, and the real rows in every tile.
    """
    import torch

    tiles, prefix = [], 0
    _, frames, height, width, _ = layout.signature
    grid = (int(frames), int(height) // 2, int(width) // 2)
    for start, end, kind in layout.segments:
        rows = torch.arange(start, end, device=device)
        if kind != "video":
            count = -(-(end - start) // BLOCK)
            padded = torch.full((count * BLOCK,), -1, dtype=torch.int64, device=device)
            padded[:end - start] = rows
            tiles.append(padded.view(count, BLOCK))
            prefix += count
            continue
        if grid[0] * grid[1] * grid[2] != end - start:
            raise RuntimeError(
                f"VSA: the video segment has {end - start} rows and the latent grid {grid} "
                f"holds {grid[0] * grid[1] * grid[2]}")
        sizes = [-(-side // cube) * cube for side, cube in zip(grid, CUBE)]
        volume = torch.full(sizes, -1, dtype=torch.int64, device=device)
        volume[:grid[0], :grid[1], :grid[2]] = rows.view(*grid)
        cubes = volume.view(sizes[0] // CUBE[0], CUBE[0], sizes[1] // CUBE[1], CUBE[1],
                            sizes[2] // CUBE[2], CUBE[2]).permute(0, 2, 4, 1, 3, 5).reshape(-1, BLOCK)
        real_first = torch.argsort((cubes < 0).to(torch.int8), dim=1, stable=True)
        tiles.append(torch.gather(cubes, 1, real_first))
    order = torch.cat(tiles)
    source = order.reshape(-1)
    real = source >= 0
    place = torch.empty(layout.seq_len, dtype=torch.int64, device=device)
    place[source[real]] = torch.nonzero(real).flatten()
    return {"rows": int(source.numel()), "prefix": prefix, "source": source, "place": place,
            "lengths": (order >= 0).sum(1).to(torch.int32)}


class _State:
    """What every sparse layer of one patched model shares: settings and the run's statistics."""

    def __init__(self, method: str, high: float, low: float, sinks: str, extra_tokens: int,
                 min_tokens: int):
        self.method = method
        self.high = high
        self.low = low
        self.sinks_mode = sinks
        self.extra_tokens = 0 if method == "vsa" else int(extra_tokens)
        self.min_tokens = int(min_tokens)
        self.stats: dict = {}
        self.refused: set = set()
        self.reported: set = set()
        self.kernels: dict = {}
        self.tilings: dict = {}
        self.rope = None

    def reset(self):
        """Forget the statistics carried between steps and the sizes refused, at the end of a run."""
        self.stats = {}
        self.refused = set()
        self.reported = set()
        self.rope = None

    def report(self, key, level: str, message: str, *values):
        """Log a message once per run."""
        if key not in self.reported:
            self.reported.add(key)
            getattr(logger, level)(message, *values)

    def wanted(self, x, rope_freqs, options, attn, layout) -> bool:
        """Whether one block call runs sparse."""
        tokens = int(x.shape[0])
        if tokens < self.min_tokens or tokens in self.refused or rope_freqs is None:
            return False
        sigmas = (options or {}).get("sigmas")
        if sigmas is not None:
            sigma = float(sigmas.flatten()[0]) if hasattr(sigmas, "flatten") else float(sigmas[0])
            if sigma > self.high or sigma < self.low:
                return False
        if x.device.type != "cuda" or getattr(attn, "head_dim", None) != HEAD_DIM:
            return False
        import torch

        if x.dtype != torch.bfloat16:
            self.report(("dtype", x.dtype), "warning",
                        "H3 Sparse Attention runs dense: the model computes in %s, and Sol-Attn "
                        "takes bfloat16. Load the model with weight_dtype default.", x.dtype)
            return False
        if self.method == "vsa" and getattr(layout, "seq_len", None) != tokens:
            self.report(("layout",), "warning",
                        "H3 Sparse Attention runs dense: VSA needs the H3 layout of the call.")
            return False
        if str(x.device) not in self.kernels:
            self.kernels[str(x.device)] = kernel_problem(x.device)
        problem = self.kernels[str(x.device)]
        if problem is not None:
            self.report(("kernel",), "warning", "H3 Sparse Attention runs dense: %s.", problem)
            return False
        return True

    def sinks(self, layout, tokens: int) -> tuple[tuple[int, int], tuple[int, int]]:
        """Key blocks every query reads exactly, and query blocks run dense, as block spans."""
        if self.sinks_mode == "off" or layout is None or getattr(layout, "seq_len", None) != tokens:
            return (0, 0), (0, 0)
        segments = list(getattr(layout, "segments", ()))
        video = next((start for start, _, kind in segments if kind == "video"), None)
        if not video:
            return (0, 0), (0, 0)
        keys = (0, -(-video // BLOCK))
        if self.sinks_mode == "exact_kv":
            return keys, (0, 0)
        audio = next((start for start, _, kind in segments if kind == "audio"), None)
        return keys, keys if audio is None else (audio // BLOCK, keys[1])

    def tiling(self, layout, device) -> dict:
        """The VSA tiling of a layout, made once per latent shape."""
        key = (tuple(layout.signature), tuple(layout.segments), str(device))
        found = self.tilings.get(key)
        if found is None:
            with _outside_graph():
                found = _tiling(layout, device)
            while len(self.tilings) >= TILINGS_KEPT:
                self.tilings.pop(next(iter(self.tilings)))
            self.tilings[key] = found
        return found

    def tiled_rope(self, rope_freqs, tiling: dict):
        """The rotation table laid out in tile order, made once per step's table."""
        held = self.rope
        if held is not None and held[0] is rope_freqs and held[1] is tiling:
            return held[2]
        with _outside_graph():
            tiled = rope_freqs.new_zeros((1, tiling["rows"]) + tuple(rope_freqs.shape[2:]))
            tiled[0, tiling["place"]] = rope_freqs[0]
        self.rope = (rope_freqs, tiling, tiled)
        return tiled


class _SparseBlock:
    """A block replacement running one H3 block with its attention made sparse."""

    def __init__(self, state: _State, layer: int, setting: float, attn, previous=None):
        self.state = state
        self.layer = int(layer)
        self.setting = float(setting)
        self.attn = attn
        self.previous = previous

    def __call__(self, args, extra_args):
        layout = args.get("layout")
        if self.state.wanted(args["img"], args.get("rope_freqs"), args.get("transformer_options"),
                             self.attn, layout):

            def attention(h, rope_freqs=None, transformer_options={}):
                return self.attention(h, rope_freqs, transformer_options, layout)

            args = {**args, "attention": attention}
        if self.previous is None:
            return extra_args["original_block"](args)
        return self.previous(args, extra_args)

    def attention(self, h, rope_freqs, options, layout):
        """The block's attention through Sol-Attn, or dense where the kernel ran out of memory."""
        import torch

        try:
            return self.sparse(h, rope_freqs, options, layout)
        except torch.OutOfMemoryError:
            tokens = int(h.shape[0])
            self.state.refused.add(tokens)
            self.state.report(("memory", tokens), "warning",
                              "H3 Sparse Attention ran out of memory at %d tokens and runs those "
                              "calls dense from here on. Lower the resolution or the frame count "
                              "to keep them sparse.", tokens)
        return self.attn(h, rope_freqs=rope_freqs, transformer_options=options)

    def sparse(self, h, rope_freqs, options, layout):
        """The block's attention through the chunked Sol-Attn producer."""
        import comfy.model_management
        import comfy_kitchen as ck
        import torch

        attn, state = self.attn, self.state
        tokens, heads = int(h.shape[0]), int(attn.heads)
        q_weight = comfy.model_management.cast_to(attn.q_norm.weight, device=h.device)
        k_weight = comfy.model_management.cast_to(attn.k_norm.weight, device=h.device)
        rows, freqs, tiling, extra, gate = tokens, rope_freqs, None, {}, None
        if state.method == "vsa":
            tiling = state.tiling(layout, h.device)
            rows = tiling["rows"]
            freqs = state.tiled_rope(rope_freqs, tiling)
            keys = queries = (0, tiling["prefix"])
            extra = {"tail": False, "block_len": tiling["lengths"]}
            gate = getattr(attn, "to_gate_compress", None)
            if gate is not None:
                extra["coarse_gate"] = h.new_empty(rows, heads * HEAD_DIM).view(1, rows, heads, HEAD_DIM)
            else:
                state.report(("gate",), "warning",
                             "H3 Sparse Attention: this model has no VSA gate layers, so VSA runs "
                             "without its coarse branch. VSA expects FastH3 weights.")
        else:
            keys, queries = state.sinks(layout, tokens)
        top = 0.0 if state.method == "sol-attn" else self.setting / 100.0
        key = (self.layer, rows, tuple((options or {}).get("uuids", ())))
        held = state.stats.get(key)

        def chunks():
            for start in range(0, rows, CHUNK):
                if tiling is None:
                    yield attn.qkv_proj(h[start:start + CHUNK])
                    continue
                source = tiling["source"][start:start + CHUNK]
                picked = h[source.clamp_min(0)] * (source >= 0).unsqueeze(1).to(h.dtype)
                if gate is not None:
                    extra["coarse_gate"].view(rows, heads * HEAD_DIM)[start:start + picked.shape[0]] = gate(picked)
                yield attn.qkv_proj(picked)

        out, kmean, vscale = ck.sol_attn_chunked(
            chunks, rows, heads, freqs, (q_weight, k_weight),
            kmean=None if held is None else held[0], vscale=None if held is None else held[1],
            tau=self.setting if state.method == "sol-attn" else 1.0, topk_ratio=top,
            token_aug=state.extra_tokens, sink_blocks=list(keys), sink_q=list(queries),
            rope_eps=attn.q_norm.eps, **extra,
        )
        if held is None:
            with _outside_graph():
                held = (torch.empty_like(kmean), torch.empty_like(vscale))
            state.stats[key] = held
        held[0].copy_(kmean)
        held[1].copy_(vscale)
        state.report(("sparse", tokens), "info",
                     "H3 Sparse Attention: %d tokens sparse under %s, conditioning key blocks %s "
                     "exact, query blocks %s dense", tokens, state.method, keys, queries)
        out = out.view(rows, heads * HEAD_DIM)
        if tiling is not None:
            out = out[tiling["place"]]
        return attn.out_proj(out)

    def to(self, device_or_dtype):
        """Pass a device or dtype move on to the replacement this one wraps."""
        if hasattr(self.previous, "to"):
            self.previous = self.previous.to(device_or_dtype)
        return self

    def models(self) -> list:
        """Whatever the wrapped replacement loads."""
        return self.previous.models() if hasattr(self.previous, "models") else []

    def cleanup(self):
        """Forget the run's statistics after sampling."""
        self.state.reset()
        if hasattr(self.previous, "cleanup"):
            self.previous.cleanup()


def blocks_of(model) -> list:
    """The DiT blocks of a MiniMax H3 model.

    Args:
        model: A model patcher.

    Returns:
        The blocks, in order.

    Raises:
        ValueError: The model is not MiniMax H3.
    """
    diffusion = model.get_model_object("diffusion_model")
    blocks = getattr(diffusion, "blocks", None)
    if type(diffusion).__name__ != "MiniMaxH3Model" or blocks is None:
        raise ValueError(
            f"H3 Sparse Attention patches MiniMax H3 models, and this one is "
            f"{type(diffusion).__name__}. Use ComfyUI's Model Sparse Attention for other models.")
    return list(blocks)


def sparse(model, plan: list[float | None], method: str = "sol-attn", start_percent: float = 0.2,
           end_percent: float = 1.0, sinks: str = SINKS[0], extra_tokens: int = 256,
           min_tokens: int = 12288):
    """A clone of a MiniMax H3 model whose planned layers run block-sparse.

    Args:
        model: A MiniMax H3 model patcher.
        plan: One entry per layer, as :func:`parse_plan` returns it.
        method: One of :data:`METHODS`.
        start_percent: Point in the noise schedule sparse attention starts.
        end_percent: Point in the noise schedule it stops.
        sinks: One of :data:`SINKS`, read by ``sol-attn`` and ``sla``.
        extra_tokens: Top-scoring tokens each query block reads beyond its blocks, 0 to 256 in
            steps of 64, read by ``sol-attn`` and ``sla``.
        min_tokens: Calls with fewer tokens run dense.

    Returns:
        The patched clone.

    Raises:
        ValueError: The model is not MiniMax H3, ``method`` is unknown, or ``plan`` does not
            cover its layers.
    """
    if method not in METHODS:
        raise ValueError(f"H3 Sparse Attention has no method `{method}`; pick one of {', '.join(METHODS)}.")
    blocks = blocks_of(model)
    if len(plan) != len(blocks):
        raise ValueError(f"The layer plan covers {len(plan)} layers, and the model has {len(blocks)}.")
    sampling = model.get_model_object("model_sampling")
    high = math.inf if start_percent <= 0.0 else float(sampling.percent_to_sigma(start_percent))
    low = -math.inf if end_percent >= 1.0 else float(sampling.percent_to_sigma(end_percent))
    state = _State(method, high, low, sinks, extra_tokens, min_tokens)
    patched = model.clone()
    for layer, setting in enumerate(plan):
        if setting is None:
            continue
        options = patched.model_options.get("transformer_options", {})
        previous = options.get("patches_replace", {}).get("dit", {}).get(("double_block", layer))
        patched.set_model_patch_replace(_SparseBlock(state, layer, setting, blocks[layer].attn, previous),
                                        "dit", "double_block", layer)
    return patched

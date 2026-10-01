"""Sampling a MiniMax H3 model over long token sequences in less VRAM.

Blocks run all but attention in token slices; resident weights are held to a budget set from the
free VRAM measured inside each model call.
"""

from __future__ import annotations

import contextlib
import weakref

import torch

from .. import log

__all__ = [
    "BLOCK_SLICE",
    "LOW_VRAM_WRAPPER",
    "is_low_vram",
    "low_vram_model",
]

logger = log.get_logger("model.h3_low_vram")

#: Tokens one slice covers when a model block runs over the tokens in slices.
BLOCK_SLICE = 8192

#: Key the low VRAM model's wrappers are registered under.
LOW_VRAM_WRAPPER = "was_h3_low_vram"

#: VRAM set aside for the pass's working memory per token, in bytes.
WORKING_PER_TOKEN = 192 * 1024

#: VRAM set aside for the pass's working memory on top of its tokens, in bytes.
WORKING_FIXED = 3 * 1024 ** 3

#: VRAM kept free beyond the reserve at the fullest point of a model call, in bytes.
WORKING_MARGIN = 768 * 1024 ** 2

#: Key a patched streamed weight is kept under in its module's prefetch record.
PATCHED_KEY = "was_h3_low_vram_patched"

#: Key the run's weight budget is kept under in the transformer options.
BUDGET_KEY = "was_h3_low_vram_budget"


def low_vram_model(model, tokens: int = BLOCK_SLICE):
    """A clone of an H3 model running all but attention in token slices, its weights held in budget.

    Args:
        model: A MiniMax H3 model patcher.
        tokens: Tokens per slice.

    Returns:
        The patched clone, computing the same output at a lower memory peak.
    """
    import comfy.patcher_extension

    patched = model.clone()
    budget = _WeightBudget(weakref.ref(patched.model))
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
                                 LOW_VRAM_WRAPPER, budget.sample)
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                                 LOW_VRAM_WRAPPER, budget.forward)
    network = patched.get_model_object("diffusion_model")
    try:
        from comfy.ldm.minimax.model import _mod_gate, _mod_scale_shift
    except ImportError:
        _mod_gate = _mod_scale_shift = None
    for index, block in enumerate(getattr(network, "blocks", [])):
        if _mod_gate is None:
            key = f"diffusion_model.blocks.{index}.mlp.forward"
            stock = patched.object_patches.get(key, block.mlp.forward)
            patched.add_object_patch(key, _sliced(stock, int(tokens)))
            continue
        key = f"diffusion_model.blocks.{index}.forward"
        stock = patched.object_patches.get(key, block.forward)
        patched.add_object_patch(key, _streamed(block, stock, int(tokens), _mod_scale_shift, _mod_gate))
    return patched


def is_low_vram(model) -> bool:
    """Whether a model patcher already carries the low VRAM setup.

    Args:
        model: A model patcher.

    Returns:
        True where :func:`low_vram_model` has been applied to it or to a model it was cloned from.
    """
    import comfy.patcher_extension

    wrappers = getattr(model, "wrappers", None) or {}
    found = wrappers.get(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, {})
    return bool(found.get(LOW_VRAM_WRAPPER))


def _tokens(x, context) -> int:
    """Tokens an H3 forward runs over: 2x2 video patches, audio steps and text."""
    video = x[0] if isinstance(x, (list, tuple)) else x
    count = int(video.shape[2]) * ((int(video.shape[3]) + 1) // 2) * ((int(video.shape[4]) + 1) // 2)
    if isinstance(x, (list, tuple)) and len(x) > 1:
        count += int(x[1].shape[-1])
    if context is not None and getattr(context, "ndim", 0) >= 2:
        count += int(context.shape[1])
    return count


def _has_patches(module) -> bool:
    """Whether a layer's weights carry patches, such as a LoRA, applied as they are cast."""
    return any((
        getattr(module, "weight_lowvram_function", None) is not None,
        getattr(module, "bias_lowvram_function", None) is not None,
        bool(getattr(module, "weight_function", None)),
        bool(getattr(module, "bias_function", None)),
    ))


def _outside_allocation_graph():
    """A context whose allocations ComfyUI's per-block allocation recording leaves out."""
    try:
        import comfy.model_prefetch as prefetch
    except ImportError:
        return contextlib.nullcontext()
    pause = getattr(prefetch, "pause_malloc_graph", None)
    return pause() if pause is not None else contextlib.nullcontext()


def _patched_once(stock):
    """A weight cast resolver that patches a streamed layer once per prefetch, not once per call.

    Args:
        stock: ComfyUI's ``resolve_cast_module_with_vbar``.

    Returns:
        The resolver, answering repeat calls for the same prefetched layer from the first.
    """

    def resolve(module, *args, **kwargs):
        prefetch = getattr(module, "_prefetch", None)
        if (not isinstance(prefetch, dict) or prefetch.get("resident")
                or prefetch.get("signature") is not None or not _has_patches(module)):
            return stock(module, *args, **kwargs)
        key = (args, tuple(sorted(kwargs.items())))
        held = prefetch.get(PATCHED_KEY)
        if held is not None and held[0] == key:
            return held[1]
        with _outside_allocation_graph():
            result = stock(module, *args, **kwargs)
        prefetch[PATCHED_KEY] = (key, result)
        return result

    return resolve


class _WeightBudget:
    """How much of a dynamically loaded model's weights stays resident during one sampling run.

    Attributes:
        model: A weak reference to the model's ``BaseModel``.
        cap: Resident weight bytes allowed for the current model call.
        calls: Model calls made this run.
        device: The device of the current model call.
        vbar: The model's weight address space on that device.
        low: Fewest free VRAM bytes seen during the current model call.
        tokens: Tokens of the current model call.
        peak: Most tokens of any model call this run.
        learned: The cap the last run settled on, by the most tokens per model call.
    """

    def __init__(self, model):
        self.model = model
        self.cap = None
        self.calls = 0
        self.device = None
        self.vbar = None
        self.low = None
        self.tokens = 0
        self.peak = 0
        self.learned = {}

    def sample(self, executor, *args, **kwargs):
        """Runs a sampling run with an emptied allocator cache and its budget worked out afresh."""
        import comfy.model_management as management
        import comfy.ops as ops

        self.cap, self.calls, self.low, self.tokens, self.peak = None, 0, None, 0, 0
        management.soft_empty_cache(force=True)
        stock = getattr(ops, "resolve_cast_module_with_vbar", None)
        if stock is not None:
            ops.resolve_cast_module_with_vbar = _patched_once(stock)
        try:
            return executor(*args, **kwargs)
        finally:
            if stock is not None:
                ops.resolve_cast_module_with_vbar = stock
            self.cap = None
            self._release()

    def _release(self):
        """Frees this model's resident weights at the end of a run."""
        import comfy.model_management as management

        vbar, self.vbar = self.vbar, None
        if vbar is None:
            return
        try:
            loaded = int(vbar.loaded_size())
            if loaded:
                vbar.free_memory(loaded)
            management.soft_empty_cache()
        except Exception as error:
            logger.warning("H3 low VRAM: could not release the weights after the run: %s", error)

    def forward(self, executor, *args, **kwargs):
        """Runs one model call with the resident weights held to the budget."""
        import comfy.model_management as management

        x = args[0] if args else kwargs.get("x")
        context = args[2] if len(args) > 2 else kwargs.get("context")
        video = x[0] if isinstance(x, (list, tuple)) else x
        vbar = self._vbar(video.device)
        if vbar is None or not hasattr(vbar, "set_watermark_limit"):
            return executor(*args, **kwargs)
        reserve = int(management.extra_reserved_memory())
        tokens = _tokens(x, context)
        if self.cap is None:
            self._release_others(video.device)
            free = _free_vram(video.device)
            working = tokens * WORKING_PER_TOKEN + WORKING_FIXED
            self.cap = self.learned.get(
                tokens, max(0, int(free) + int(vbar.loaded_size()) - working - reserve))
        elif tokens > self.peak:
            self.cap = max(0, self.cap - (tokens - self.peak) * WORKING_PER_TOKEN)
        elif self.low is not None and self.tokens >= self.peak:
            held = min(self.cap, int(vbar.loaded_size()))
            self.cap = max(0, held + int(self.low) - reserve - WORKING_MARGIN)
            self.learned[self.peak] = self.cap
            if self.calls == 1:
                total = int(torch.cuda.get_device_properties(video.device).total_memory)
                logger.info(
                    "H3 low VRAM: %d tokens, %.1f of %.1f GB at the fullest point of the first "
                    "model call, weights held to %.1f GB from the second", self.tokens,
                    (total - self.low) / 1024 ** 3, total / 1024 ** 3, self.cap / 1024 ** 3,
                )
        self.calls += 1
        self.peak = max(self.peak, tokens)
        self.device, self.vbar, self.low, self.tokens = video.device, vbar, None, tokens
        vbar.set_watermark_limit(self.cap)
        options = args[3] if len(args) > 3 else kwargs.get("transformer_options")
        if isinstance(options, dict):
            options[BUDGET_KEY] = self
        _hold(options)
        return executor(*args, **kwargs)

    def hold(self):
        """Evicts the resident weights above the cap and notes the free VRAM left."""
        excess = int(self.vbar.loaded_size()) - self.cap
        if excess > 0:
            self.vbar.free_memory(excess)
        self.measure()

    def measure(self):
        """Notes the free VRAM, keeping the fewest bytes seen during the model call."""
        free = _free_vram(self.device)
        if self.low is None or free < self.low:
            self.low = free

    def _release_others(self, device):
        """Frees the VRAM every other loaded model holds on ``device``, this one kept."""
        import comfy.model_management as management

        mine = self.model()
        others, keep = [], []
        for loaded in list(management.current_loaded_models):
            patcher = loaded.model
            if patcher is None or patcher.model is mine or loaded.device != device:
                keep.append(loaded)
            elif patcher.is_dynamic():
                keep.append(loaded)
                others.append(patcher)
        freed = sum(int(p.partially_unload(p.offload_device, 1e32) or 0) for p in others)
        unloaded = management.free_memory(1e32, device, keep_loaded=keep)
        if freed or unloaded:
            management.soft_empty_cache()
            logger.info("H3 low VRAM: released %.1f GB held by %d other model(s)",
                        freed / 1024 ** 3, len(others) + len(unloaded))

    def _vbar(self, device):
        """The model's weight address space on ``device``, or ``None`` when it has none."""
        model = self.model()
        vbars = getattr(model, "dynamic_vbars", None) if model is not None else None
        if not vbars or getattr(device, "type", None) != "cuda":
            return None
        found = vbars.get(device)
        if found is None:
            found = next((vbar for key, vbar in vbars.items() if torch.device(key) == device), None)
        return found


def _free_vram(device) -> int:
    """Free VRAM on ``device`` as the driver counts it across every process, in bytes."""
    try:
        used = int(torch.cuda.device_memory_used(device))
        return int(torch.cuda.get_device_properties(device).total_memory) - used
    except Exception:
        return int(torch.cuda.mem_get_info(device)[0])


def _hold(transformer_options):
    """Holds the resident weights to the run's budget, when one is set."""
    budget = transformer_options.get(BUDGET_KEY) if isinstance(transformer_options, dict) else None
    if budget is not None:
        budget.hold()


def _measure(transformer_options):
    """Notes the free VRAM against the run's budget, when one is set."""
    budget = transformer_options.get(BUDGET_KEY) if isinstance(transformer_options, dict) else None
    if budget is not None:
        budget.measure()


def _spans(count: int, size: int) -> list[tuple[int, int]]:
    """``(start, stop)`` pairs covering ``count`` tokens in slices of ``size``."""
    return [(start, min(start + size, count)) for start in range(0, count, size)]


def _local(segments, start: int, stop: int) -> list:
    """Modulation segments cut to one slice of tokens and counted from its start.

    Args:
        segments: ``(start, stop, row)`` triples, ``row`` an index or one index per token.
        start: First token of the slice.
        stop: Token after the slice.

    Returns:
        The triples that overlap the slice.
    """
    cut = []
    for first, last, row in segments:
        low, high = max(first, start), min(last, stop)
        if low < high:
            part = row if isinstance(row, int) else row[low - first:high - first]
            cut.append((low - start, high - start, part))
    return cut


class _Deferred:
    """A module forward that hands its input back and keeps it."""

    def __init__(self):
        self.given = None

    def __call__(self, x):
        self.given = x
        return x


@contextlib.contextmanager
def _forward_swapped(module, replacement):
    """A module's forward replaced for the length of the block.

    Args:
        module: A torch module.
        replacement: The callable run in place of its forward.

    Yields:
        The forward the module had.
    """
    had = "forward" in module.__dict__
    stock = module.forward
    module.forward = replacement
    try:
        yield stock
    finally:
        if had:
            module.forward = stock
        else:
            del module.forward


def _sliced(stock, size: int):
    """A forward running ``stock`` over the rows of its input in slices of ``size``."""

    def forward(x):
        if x.shape[0] <= size:
            return stock(x)
        out = None
        for start, stop in _spans(x.shape[0], size):
            part = stock(x[start:stop])
            if out is None:
                out = part.new_empty((x.shape[0],) + tuple(part.shape[1:]))
            out[start:stop] = part
        return out

    return forward


def _streamed(block, stock, size: int, scale_shift, gate):
    """An H3 block forward running all but its attention over the tokens in slices.

    Args:
        block: The block.
        stock: The forward it replaces, run as is for sequences of ``size`` tokens or fewer.
        size: Tokens per slice.
        scale_shift: The model's modulation, ``(h, shift, scale, segments) -> h`` in place.
        gate: The model's gated residual add, ``(x, gate, other, segments) -> x`` in place.

    Returns:
        The forward.
    """

    def forward(x, t_emb, mod_segments, rope_freqs, transformer_options={}, attention=None):
        _hold(transformer_options)
        count = x.shape[0]
        if count <= size:
            return stock(x, t_emb, mod_segments, rope_freqs,
                         transformer_options=transformer_options, attention=attention)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)
        pieces = [(start, stop, _local(mod_segments, start, stop))
                  for start, stop in _spans(count, size)]
        normed = None
        for start, stop, local in pieces:
            part = scale_shift(block.norm1(x[start:stop]), shift_msa, scale_msa, local)
            if normed is None:
                normed = part.new_empty((count,) + tuple(part.shape[1:]))
            normed[start:stop] = part
            del part
        attend = block.attn if attention is None else attention
        heard = _Deferred()
        with _forward_swapped(block.attn.out_proj, heard), \
                _forward_swapped(block.attn.qkv_proj, _sliced(block.attn.qkv_proj.forward, size)):
            mixed = attend(normed, rope_freqs=rope_freqs, transformer_options=transformer_options)
        _measure(transformer_options)
        if heard.given is not None and mixed is not heard.given:
            mixed = attend(normed, rope_freqs=rope_freqs, transformer_options=transformer_options)
            heard.given = None
        del normed
        if heard.given is None:
            gate(x, gate_msa, mixed, mod_segments)
        else:
            for start, stop, local in pieces:
                gate(x[start:stop], gate_msa, block.attn.out_proj(mixed[start:stop]), local)
        del mixed, heard
        for start, stop, local in pieces:
            part = scale_shift(block.norm2(x[start:stop]), shift_mlp, scale_mlp, local)
            gate(x[start:stop], gate_mlp, block.mlp(part), local)
            del part
        return x

    return forward

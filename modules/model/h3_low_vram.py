"""Sampling a MiniMax H3 model over long clips in less VRAM.

Blocks run in token slices, attention optionally by head group and query chunk; resident weights
are held to a budget set from free VRAM measured during each model call.
"""

from __future__ import annotations

import contextlib
import weakref
from typing import NamedTuple

import torch

from .. import log

__all__ = [
    "BLOCK_SLICE",
    "LOW_VRAM_WRAPPER",
    "OPTIONS_KEY",
    "Options",
    "configure",
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

#: Key the low VRAM settings are kept under in the transformer options.
OPTIONS_KEY = "was_h3_low_vram_options"

#: Transformer options key ComfyUI reads an attention override from.
OVERRIDE_KEY = "optimized_attention_override"

#: Query rows per attention tile, the unit token slices and query chunks are aligned to.
TILE_ROWS = 128


class Options(NamedTuple):
    """How a low VRAM model runs its blocks.

    Attributes:
        head_chunks: Head groups attention runs in; 0 or 1 attends every head at once.
        query_chunks: Query chunks attention runs in; 0 or 1 attends every query at once.
        token_slice: Tokens per slice for the norms, projections and feed-forward.
    """

    head_chunks: int = 0
    query_chunks: int = 0
    token_slice: int = BLOCK_SLICE


class _Plan(NamedTuple):
    """How one block call splits its attention.

    Attributes:
        heads: ``(start, stop)`` head ranges, one per head group.
        chunks: ``(start, stop)`` token ranges, one per query chunk.
        asked: The head and query chunk counts the settings asked for.
    """

    heads: list
    chunks: list
    asked: tuple


class _Unsupported(Exception):
    """The block's attention cannot be run in head groups or query chunks."""


def low_vram_model(model, tokens: int = BLOCK_SLICE):
    """A clone of an H3 model running all but attention in token slices, its weights held in budget.

    Args:
        model: A MiniMax H3 model patcher.
        tokens: Tokens per slice when the transformer options carry no :class:`Options`.

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


def configure(model, options: Options):
    """A low VRAM clone of an H3 model that runs its blocks with ``options``.

    Args:
        model: A MiniMax H3 model patcher, with or without the low VRAM setup.
        options: The settings.

    Returns:
        The patched clone.
    """
    patched = model.clone() if is_low_vram(model) else low_vram_model(model)
    settled = Options(max(0, int(options.head_chunks)), max(0, int(options.query_chunks)),
                      max(TILE_ROWS, int(options.token_slice) // TILE_ROWS * TILE_ROWS))
    patched.model_options.setdefault("transformer_options", {})[OPTIONS_KEY] = settled
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


def _options(transformer_options):
    """The :class:`Options` a model call carries, or None."""
    found = transformer_options.get(OPTIONS_KEY) if isinstance(transformer_options, dict) else None
    return found if isinstance(found, Options) else None


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
        settings: The :class:`Options` of the current run, or None.
        reported: What the run's blocks have logged, so each line is logged once.
        learned: The cap the last run settled on, by the most tokens per model call and the settings.
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
        self.settings = None
        self.reported = set()
        self.learned = {}

    def sample(self, executor, *args, **kwargs):
        """Runs a sampling run with an emptied allocator cache and its budget worked out afresh."""
        import comfy.model_management as management
        import comfy.ops as ops

        self.cap, self.calls, self.low, self.tokens, self.peak = None, 0, None, 0, 0
        self.reported = set()
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
        options = args[3] if len(args) > 3 else kwargs.get("transformer_options")
        if isinstance(options, dict):
            options[BUDGET_KEY] = self
        video = x[0] if isinstance(x, (list, tuple)) else x
        vbar = self._vbar(video.device)
        if vbar is None or not hasattr(vbar, "set_watermark_limit"):
            return executor(*args, **kwargs)
        reserve = int(management.extra_reserved_memory())
        tokens = _tokens(x, context)
        settings = _options(options)
        if self.cap is None:
            self._release_others(video.device)
            free = _free_vram(video.device)
            working = tokens * WORKING_PER_TOKEN + WORKING_FIXED
            self.settings = settings
            self.cap = self.learned.get(
                (tokens, settings), max(0, int(free) + int(vbar.loaded_size()) - working - reserve))
        elif tokens > self.peak:
            self.cap = max(0, self.cap - (tokens - self.peak) * WORKING_PER_TOKEN)
        elif self.low is not None and self.tokens >= self.peak:
            held = min(self.cap, int(vbar.loaded_size()))
            self.cap = max(0, held + int(self.low) - reserve - WORKING_MARGIN)
            self.learned[(self.peak, self.settings)] = self.cap
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
        _hold(options)
        return executor(*args, **kwargs)

    def hold(self):
        """Evicts the resident weights above the cap and notes the free VRAM left."""
        if self.cap is None or self.vbar is None:
            return
        excess = int(self.vbar.loaded_size()) - self.cap
        if excess > 0:
            self.vbar.free_memory(excess)
        self.measure()

    def measure(self, extra: int = 0):
        """Notes the free VRAM less ``extra`` bytes, keeping the fewest bytes seen during the model call."""
        if self.cap is None or self.vbar is None:
            return
        free = _free_vram(self.device) - int(extra)
        if self.low is None or free < self.low:
            self.low = free

    def report(self, text: str, warning: bool = False):
        """Logs a line about how the blocks run, once per sampling run."""
        if text in self.reported:
            return
        self.reported.add(text)
        (logger.warning if warning else logger.info)("H3 low VRAM: %s", text)

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


def _measure(transformer_options, extra: int = 0):
    """Notes the free VRAM less ``extra`` bytes against the run's budget, when one is set."""
    budget = transformer_options.get(BUDGET_KEY) if isinstance(transformer_options, dict) else None
    if budget is not None:
        budget.measure(extra)


def _report(transformer_options, text: str, warning: bool = False):
    """Logs a line about how the blocks run through the run's budget, when one is set."""
    budget = transformer_options.get(BUDGET_KEY) if isinstance(transformer_options, dict) else None
    if budget is not None:
        budget.report(text, warning)


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


class _Projected:
    """An H3 attention module whose qkv projection answers a given tensor and whose output projection passes through."""

    def __init__(self, attn, qkv):
        self._attn = attn
        self._qkv = [qkv]

    def __getattr__(self, name):
        if name.startswith("__") or name in ("_attn", "_qkv"):
            raise AttributeError(name)
        return getattr(self._attn, name)

    def qkv_proj(self, x):
        """Hands over the given qkv, once."""
        return self._qkv.pop()

    def out_proj(self, x):
        """Returns ``x`` unchanged."""
        return x


class _Capture:
    """An attention override that keeps the q, k and v it is handed instead of attending.

    Attributes:
        taken: The ``(q, k, v)`` it was handed, or None.
        call: The keyword arguments of that call, less the transformer options.
    """

    def __init__(self):
        self.taken = None
        self.call = None

    def container_function(self, q, k, v, heads, *args, **kwargs):
        """Keeps the tensors three attention containers hold."""
        return self._keep(q.take(), k.take(), v.take(), kwargs)

    def __call__(self, function, q, k, v, heads, *args, **kwargs):
        """Keeps the tensors an attention call was handed."""
        return self._keep(q, k, v, kwargs)

    def _keep(self, q, k, v, kwargs):
        """Keeps one call's tensors and keywords and answers an empty output."""
        self.taken = (q, k, v)
        self.call = {key: value for key, value in kwargs.items()
                     if key not in ("transformer_options", "_inside_attn_wrapper")}
        return q.new_empty((0,))


def _tile_floor(device) -> int:
    """Attention tiles one call is given at least, twice the multiprocessors on CUDA, else 0."""
    if getattr(device, "type", None) != "cuda":
        return 0
    return 2 * int(torch.cuda.get_device_properties(device).multi_processor_count)


def _tiles(rows: int) -> int:
    """Attention tiles covering ``rows`` query rows."""
    return -(-int(rows) // TILE_ROWS)


def _head_ranges(heads: int, groups: int) -> list[tuple[int, int]]:
    """``(start, stop)`` ranges splitting ``heads`` into ``groups`` near-equal groups."""
    ranges, start = [], 0
    for index in range(groups):
        size = heads // groups + (1 if index < heads % groups else 0)
        ranges.append((start, start + size))
        start += size
    return ranges


def _query_ranges(spans, wanted: int, least: int) -> list[tuple[int, int]]:
    """Token ranges grouping whole slices into at most ``wanted`` chunks, none under ``least`` rows.

    Args:
        spans: The block's ``(start, stop)`` token slices.
        wanted: Chunks asked for.
        least: Fewest rows a chunk may have.

    Returns:
        ``(start, stop)`` per chunk.
    """
    count = len(spans)
    wanted = max(1, min(int(wanted), count))
    chunks = [(spans[index * count // wanted][0], spans[(index + 1) * count // wanted - 1][1])
              for index in range(wanted)]
    while len(chunks) > 1 and chunks[-1][1] - chunks[-1][0] < least:
        tail = chunks.pop()
        chunks[-1] = (chunks[-1][0], tail[1])
    return chunks


def _plain_attention(attn) -> bool:
    """Whether a block's attention is the model's own, with no forward patched onto it."""
    try:
        from comfy.ldm.minimax.model import Attention
    except ImportError:
        return False
    return isinstance(attn, Attention) and "forward" not in vars(attn)


def _attention_plan(block, attention, options, x, spans, rope_freqs, transformer_options):
    """Head groups and query chunks one block call attends in.

    Args:
        block: The block.
        attention: The attention a block replacement handed the block, or None.
        options: The model call's :class:`Options`, or None.
        x: The block's input, ``(tokens, hidden)``.
        spans: Its token slices.
        rope_freqs: The rotation table, or None.
        transformer_options: The model call's transformer options.

    Returns:
        A :class:`_Plan`, or None to attend every head and query at once.

    Raises:
        _Unsupported: Head groups or query chunks were asked for and the attention is not the model's own.
    """
    if options is None or (options.head_chunks < 2 and options.query_chunks < 2):
        return None
    attn = block.attn
    if (attention is not None or not _plain_attention(attn)
            or (rope_freqs is not None and rope_freqs.shape[1] != x.shape[0])):
        raise _Unsupported("another node replaced the MiniMax H3 attention")
    overridden = transformer_options.get(OVERRIDE_KEY) is not None
    if overridden and options.head_chunks < 2:
        raise _Unsupported("another node set an attention override, so queries stay whole")
    heads = int(attn.heads)
    floor = _tile_floor(x.device)
    chunks = [(0, x.shape[0])]
    if options.query_chunks >= 2 and not overridden:
        chunks = _query_ranges(spans, options.query_chunks, TILE_ROWS * -(-floor // heads))
    shortest = _tiles(min(stop - start for start, stop in chunks))
    groups = max(1, min(options.head_chunks, heads))
    while groups > 1 and heads // groups * shortest < floor:
        groups -= 1
    if groups < 2 and len(chunks) < 2:
        return None
    return _Plan(_head_ranges(heads, groups), chunks, (options.head_chunks, options.query_chunks))


def _describe(count: int, size: int, plan) -> str:
    """One line naming how a block call ran: its tokens, attention split and slices."""
    slices = f"norms, projections and feed-forward in {-(-count // size)} slices of {size} tokens"
    if plan is None:
        return f"{count} tokens: attention whole; {slices}"
    parts = []
    for done, asked, name in ((len(plan.heads), plan.asked[0], "head"),
                              (len(plan.chunks), plan.asked[1], "query")):
        if asked >= 2:
            parts.append(f"{done} {name} chunk{'' if done == 1 else 's'}"
                         + (f" ({asked} asked)" if done != asked else ""))
    return f"{count} tokens: attention in {' and '.join(parts)}; {slices}"


def _normed_qkv(block, x, start: int, stop: int, local, shift, scale, rope_freqs, scale_shift,
                transformer_options):
    """One token slice's q, k and v, normed and rotated by the model's own attention code.

    Args:
        block: The block.
        x: The block's input, ``(tokens, hidden)``.
        start: First token of the slice.
        stop: Token after the slice.
        local: The modulation segments cut to the slice.
        shift: The attention modulation shift.
        scale: The attention modulation scale.
        rope_freqs: The rotation table for every token, or None.
        scale_shift: The model's modulation, ``(h, shift, scale, segments) -> h`` in place.
        transformer_options: The model call's transformer options.

    Returns:
        ``((q, k, v), call)``: each ``(1, heads, tokens, head_dim)``, and the keyword arguments the
        attention was called with.

    Raises:
        _Unsupported: The attention did not hand them over in that layout.
    """
    attn = block.attn
    part = scale_shift(block.norm1(x[start:stop]), shift, scale, local)
    qkv = attn.qkv_proj(part)
    del part
    stand_in = qkv.new_empty((stop - start, 0))
    view = _Projected(attn, qkv)
    del qkv
    capture = _Capture()
    rope = None if rope_freqs is None else rope_freqs[:, start:stop]
    present, had = OVERRIDE_KEY in transformer_options, transformer_options.get(OVERRIDE_KEY)
    transformer_options[OVERRIDE_KEY] = capture
    try:
        type(attn).forward(view, stand_in, rope_freqs=rope, transformer_options=transformer_options)
    except (AttributeError, TypeError) as error:
        raise _Unsupported(f"the MiniMax H3 attention could not be split: {error}") from error
    finally:
        if present:
            transformer_options[OVERRIDE_KEY] = had
        else:
            transformer_options.pop(OVERRIDE_KEY, None)
    taken, call = capture.taken, capture.call
    if (taken is None or call.get("skip_reshape") is not True or call.get("mask") is not None
            or any(t.ndim != 4 or t.shape[0] != 1 or t.shape[1] != attn.heads
                   or t.shape[2] != stop - start for t in taken)):
        raise _Unsupported("the MiniMax H3 attention did not hand over its q, k and v")
    return taken, call


def _attend(attn, holder: list, heads: int, call: dict, transformer_options):
    """The model's attention over the q, k and v ``holder`` gives up, through ComfyUI's dispatch.

    Args:
        attn: The block's attention module.
        holder: A list holding one ``(q, k, v)`` triple, each ``(1, heads, tokens, head_dim)``; emptied.
        heads: Heads in the triple.
        call: Keyword arguments the model's own attention call was made with.
        transformer_options: The model call's transformer options.

    Returns:
        The attention output, ``(query tokens, heads * head_dim)``.
    """
    from comfy.ldm.minimax import model as minimax

    try:
        from comfy.ldm.modules.attention import AttentionTensorContainer as contain
    except ImportError:
        def contain(tensor):
            return tensor
    q, k, v = (contain(part) for part in holder.pop())
    out = minimax.optimized_attention(q, k, v, heads, preferred_attention=getattr(attn, "comfy_attention", None),
                                      transformer_options=transformer_options, **call)
    return out.squeeze(0)


def _project(block, x, plan, pieces, shift, scale, rope_freqs, scale_shift, transformer_options):
    """Every token's normed q, k and v, kept per head group, or as keys and values when queries are chunked.

    Args:
        block: The block.
        x: The block's input, ``(tokens, hidden)``.
        plan: The block call's :class:`_Plan`.
        pieces: ``(start, stop, segments)`` per token slice.
        shift: The attention modulation shift.
        scale: The attention modulation scale.
        rope_freqs: The rotation table, or None.
        scale_shift: The model's modulation, ``(h, shift, scale, segments) -> h`` in place.
        transformer_options: The model call's transformer options.

    Returns:
        ``(groups, keys, values, call)``: per head group ``(3, tokens, heads, head_dim)`` q, k and v,
        or keys and values ``(tokens, heads, head_dim)``, and the attention call's keywords.

    Raises:
        _Unsupported: The attention did not hand over its q, k and v.
    """
    count = x.shape[0]
    chunked = len(plan.chunks) > 1
    groups, keys, values, call = [None] * len(plan.heads), None, None, None
    for start, stop, local in pieces:
        (q, k, v), call = _normed_qkv(block, x, start, stop, local, shift, scale, rope_freqs,
                                      scale_shift, transformer_options)
        if chunked:
            if keys is None:
                keys = k.new_empty((count, k.shape[1], k.shape[3]))
                values = v.new_empty((count, v.shape[1], v.shape[3]))
            keys[start:stop] = k[0].transpose(0, 1)
            values[start:stop] = v[0].transpose(0, 1)
        else:
            for index, (first, last) in enumerate(plan.heads):
                if groups[index] is None:
                    groups[index] = q.new_empty((3, count, last - first, q.shape[3]))
                for row, tensor in enumerate((q, k, v)):
                    groups[index][row, start:stop] = tensor[0, first:last].transpose(0, 1)
        del q, k, v
    return groups, keys, values, call


def _attend_in_chunks(block, x, plan, pieces, held, shift, scale, gate_msa, rope_freqs, scale_shift,
                      gate, transformer_options):
    """Runs a block's attention in head groups and query chunks and adds it to ``x`` in place.

    Args:
        block: The block.
        x: The block's input, ``(tokens, hidden)``, updated in place.
        plan: The block call's :class:`_Plan`.
        pieces: ``(start, stop, segments)`` per token slice.
        held: What :func:`_project` returned; its group buffers are released as they are used.
        shift: The attention modulation shift.
        scale: The attention modulation scale.
        gate_msa: The attention residual gate.
        rope_freqs: The rotation table, or None.
        scale_shift: The model's modulation, ``(h, shift, scale, segments) -> h`` in place.
        gate: The model's gated residual add, ``(x, gate, other, segments) -> x`` in place.
        transformer_options: The model call's transformer options.
    """
    attn = block.attn
    groups, keys, values, call = held
    dim, width, count = int(attn.head_dim), x.element_size(), x.shape[0]
    first_heads = plan.heads[0][1] - plan.heads[0][0]
    if len(plan.chunks) == 1:
        _measure(transformer_options, 3 * count * first_heads * dim * width)
        outs = []
        for index, (first, last) in enumerate(plan.heads):
            holder = [tuple(part.transpose(0, 1).unsqueeze(0) for part in groups[index].unbind(0))]
            groups[index] = None
            outs.append(_attend(attn, holder, last - first, call, transformer_options))
            if index == 0:
                _measure(transformer_options)
        for start, stop, local in pieces:
            mixed = torch.cat([out[start:stop] for out in outs], dim=-1)
            gate(x[start:stop], gate_msa, attn.out_proj(mixed), local)
            del mixed
        return
    for number, (begin, end) in enumerate(plan.chunks):
        inside = [piece for piece in pieces if begin <= piece[0] < end]
        queries = None
        for start, stop, local in inside:
            (q, k, v), _ = _normed_qkv(block, x, start, stop, local, shift, scale, rope_freqs,
                                       scale_shift, transformer_options)
            del k, v
            if queries is None:
                queries = q.new_empty((end - begin, q.shape[1], q.shape[3]))
            queries[start - begin:stop - begin] = q[0].transpose(0, 1)
            del q
        if number == 0:
            _measure(transformer_options, (end - begin + 2 * count) * first_heads * dim * width)
        mixed = None
        for first, last in plan.heads:
            holder = [tuple(tensor[:, first:last].transpose(0, 1).unsqueeze(0)
                            for tensor in (queries, keys, values))]
            part = _attend(attn, holder, last - first, call, transformer_options)
            if len(plan.heads) == 1:
                mixed = part
            else:
                if mixed is None:
                    mixed = part.new_empty((end - begin, int(attn.heads) * dim))
                mixed[:, first * dim:last * dim] = part
            del part
        del queries
        if number == 0:
            _measure(transformer_options)
        for start, stop, local in inside:
            gate(x[start:stop], gate_msa, attn.out_proj(mixed[start - begin:stop - begin]), local)
        del mixed


def _streamed(block, stock, size: int, scale_shift, gate):
    """An H3 block forward running all but its attention over the tokens in slices.

    Args:
        block: The block.
        stock: The forward it replaces, run as is for sequences of one slice or fewer.
        size: Tokens per slice when the transformer options carry no :class:`Options`.
        scale_shift: The model's modulation, ``(h, shift, scale, segments) -> h`` in place.
        gate: The model's gated residual add, ``(x, gate, other, segments) -> x`` in place.

    Returns:
        The forward, which also runs attention in head groups and query chunks when the options ask.
    """

    def forward(x, t_emb, mod_segments, rope_freqs, transformer_options={}, attention=None):
        _hold(transformer_options)
        options = _options(transformer_options)
        slice_size = options.token_slice if options is not None else size
        count = x.shape[0]
        if count <= slice_size:
            _report(transformer_options, f"{count} tokens: one slice, blocks run unchanged")
            return stock(x, t_emb, mod_segments, rope_freqs,
                         transformer_options=transformer_options, attention=attention)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = block.adaln_proj(t_emb)
        spans = _spans(count, slice_size)
        pieces = [(start, stop, _local(mod_segments, start, stop)) for start, stop in spans]
        held = plan = None
        try:
            plan = _attention_plan(block, attention, options, x, spans, rope_freqs, transformer_options)
            if plan is not None:
                held = _project(block, x, plan, pieces, shift_msa, scale_msa, rope_freqs, scale_shift,
                                transformer_options)
        except _Unsupported as reason:
            plan = None
            _report(transformer_options, f"head_chunks and query_chunks do not apply: {reason}", True)
        _report(transformer_options, _describe(count, slice_size, plan))
        if held is not None:
            _attend_in_chunks(block, x, plan, pieces, held, shift_msa, scale_msa, gate_msa, rope_freqs,
                              scale_shift, gate, transformer_options)
            del held
        else:
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
                    _forward_swapped(block.attn.qkv_proj, _sliced(block.attn.qkv_proj.forward, slice_size)):
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

"""Run a ComfyUI language model's generate on its fixed-cache, graph-captured decode path.

The switches are set on the model for one generation and restored afterwards, and the decode
graphs a generation captured are dropped when it returns.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field

import comfy.model_management
import comfy.model_prefetch

from .. import log

logger = log.get_logger("model.text_decode")

#: The switches on a transformer that select the fast decode path.
FLAGS = ("fixed_kv", "graph_dynamic_vbar_blocks", "prefetch_dynamic_vbars")

#: The generate implementation the switches apply to, by module and qualified name.
GENERATE_IMPL = ("comfy.text_encoders.llama", "BaseGenerate.generate")

#: Decode paths a run can report.
PATH_GRAPH = "graph decode"
PATH_FIXED = "fixed cache decode"
PATH_CORE = "core decode"

#: The tags a reasoning model writes its thinking between.
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

#: Prompt tokens decoded to find a thinking block the chat template left open.
TEMPLATE_TAIL_TOKENS = 4

#: Positional arguments to a transformer's ``generate`` after the embeddings, up to and
#: including ``stop_tokens``.
STOP_TOKENS_ARG = 9


@dataclass
class DecodeReport:
    """What one generation did.

    Attributes:
        path: One of ``PATH_GRAPH``, ``PATH_FIXED`` or ``PATH_CORE``.
        reason: Why the fast path was not taken, empty when it was.
        graph_reason: Why the fast path ran without decode graphs, empty when it had them.
        prompt_tokens: Embedding positions handed to the model, images included.
        new_tokens: Tokens generated, the stop token included.
        seconds: Wall time of the whole generate call.
        load_seconds: Loading the model and embedding the prompt, before the prompt pass.
        prompt_seconds: The prompt pass, up to the first token.
        decode_seconds: Every token after the first. 0 when the first token was not timed.
        switched: True when this run set the switches, False when the model already had them.
    """

    path: str = PATH_CORE
    reason: str = ""
    graph_reason: str = ""
    prompt_tokens: int = 0
    new_tokens: int = 0
    seconds: float = 0.0
    load_seconds: float = 0.0
    prompt_seconds: float = 0.0
    decode_seconds: float = 0.0
    switched: bool = False
    extras: dict = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        """Tokens after the first over the time they took, 0 when that was not timed."""
        if self.decode_seconds <= 0 or self.new_tokens < 2:
            return 0.0
        return (self.new_tokens - 1) / self.decode_seconds


def decoder_of(clip):
    """Find the generating transformer inside a CLIP object.

    Args:
        clip: A ComfyUI ``CLIP``.

    Returns:
        ``(transformer, model)``, the object whose ``generate`` runs and the layer stack under
        it, or ``(None, None)`` where the CLIP holds no language model.
    """
    cond = getattr(clip, "cond_stage_model", None)
    name = getattr(cond, "clip", None)
    inner = getattr(cond, name, None) if isinstance(name, str) else None
    transformer = getattr(inner, "transformer", None)
    model = getattr(transformer, "model", None)
    if transformer is None or model is None:
        return None, None
    return transformer, model


def unsupported(transformer, model) -> str:
    """Say why a transformer cannot take the fast path, or nothing when it can.

    Args:
        transformer: The object whose ``generate`` runs.
        model: The layer stack under it.

    Returns:
        A reason written for the person running the node, empty when the switches apply.
    """
    if transformer is None:
        return "this CLIP holds no language model that generates text"
    method = getattr(type(transformer), "generate", None)
    impl = (getattr(method, "__module__", ""), getattr(method, "__qualname__", ""))
    if impl != GENERATE_IMPL:
        return f"{type(transformer).__name__} generates through its own decoder"
    if not all(hasattr(model, flag) for flag in FLAGS):
        return f"{type(model).__name__} has no fixed-cache decode path"
    if not hasattr(model, "init_kv_cache") or not hasattr(model, "layers"):
        return f"{type(model).__name__} has no fixed-cache decode path"
    return ""


def flash_decode_available(device) -> bool:
    """Whether ComfyUI's flash decode kernel runs on a device.

    Args:
        device: A torch device.

    Returns:
        True when the kernel is present and reports the device as supported.
    """
    try:
        import comfy_kitchen

        probe = getattr(comfy_kitchen, "flash_attention_decode_is_available", None)
        return bool(probe is not None and probe(device))
    except Exception:
        return False


def flash_decode_fits(model, device) -> bool:
    """Whether the flash decode kernel accepts a model's attention shape.

    Args:
        model: The layer stack.
        device: The device the model runs on.

    Returns:
        True when one decode step on a two position cache of the model's shape runs.
    """
    try:
        import comfy_kitchen
        import torch

        config = model.config
        heads, kv_heads, head_dim = config.num_attention_heads, config.num_key_value_heads, config.head_dim
        dtype = torch.bfloat16
        query = torch.zeros((1, 1, heads, head_dim), device=device, dtype=dtype)
        key = torch.zeros((1, 2, kv_heads, head_dim), device=device, dtype=dtype)
        seqlen = torch.full((1,), 2, device=device, dtype=torch.int32)
        comfy_kitchen.flash_attention_decode(query, key, torch.zeros_like(key), seqlen)
        return True
    except Exception as error:
        logger.debug("flash decode refused the model's attention shape: %s", error)
        return False


def smallest_window(model) -> int | None:
    """The narrowest sliding attention window among a model's layers.

    Args:
        model: The layer stack.

    Returns:
        The window in tokens, or None where every layer attends to the whole sequence.
    """
    windows = [
        layer.sliding_attention
        for layer in getattr(model, "layers", [])
        if isinstance(getattr(layer, "sliding_attention", None), int)
        and not isinstance(getattr(layer, "sliding_attention", None), bool)
        and layer.sliding_attention > 0
    ]
    return min(windows) if windows else None


def split_rope(model) -> bool:
    """Whether a model keeps separate rotary tables for its local and global layers.

    Args:
        model: The layer stack.

    Returns:
        True when the config names more than one rotary base, which graph decode cannot hold.
    """
    theta = getattr(getattr(model, "config", None), "rope_theta", None)
    return isinstance(theta, (list, tuple)) and len(theta) > 1


def graph_refusal(clip, model, device) -> str:
    """Say why the fast path would run without decode graphs, or nothing when it would have them.

    Args:
        clip: The CLIP being generated with.
        model: The layer stack.
        device: The device it runs on.

    Returns:
        A reason written for the person running the node, empty when graphs apply.
    """
    if device is None:
        return "the device the model runs on is unknown"
    if split_rope(model):
        return "the model keeps two rotary tables, local and global"
    if getattr(comfy.model_management.args, "disable_cuda_graphs", False):
        return "ComfyUI was started with --disable-cuda-graphs"
    if not comfy.model_prefetch.malloc_graph_enabled(device):
        return "graph allocation is not available on this device"
    if not clip.patcher.is_dynamic():
        return "the model is not loaded with dynamic VRAM"
    return ""


def _synchronize(device) -> None:
    """Wait for queued work on a CUDA device, so a clock read afterwards counts it."""
    try:
        import torch

        if device is not None and getattr(device, "type", "") == "cuda":
            torch.cuda.synchronize(device)
    except Exception as error:
        logger.debug("the device could not be synchronised (%s)", error)


def release_graphs():
    """Drop the decode graphs, prefetch queues and cast buffers a generation left on the device."""
    import comfy.memory_management

    if not getattr(comfy.memory_management, "aimdo_enabled", False):
        return
    import comfy_aimdo.model_vbar

    # Captured graphs hold the finished generation's cache addresses; the next one allocates anew.
    comfy.model_prefetch.cleanup_prefetch_queues()
    comfy.model_management.reset_cast_buffers()
    comfy_aimdo.model_vbar.vbars_reset_watermark_limits()


@contextlib.contextmanager
def switched(model, values: dict):
    """Set switches on a model for the length of a block and put the old values back.

    Args:
        model: The layer stack.
        values: Switch name to value.

    Yields:
        Nothing.
    """
    saved = {name: model.__dict__.get(name, _MISSING) for name in values}
    try:
        for name, value in values.items():
            setattr(model, name, value)
        yield
    finally:
        for name, value in saved.items():
            if value is _MISSING:
                model.__dict__.pop(name, None)
            else:
                setattr(model, name, value)


_MISSING = object()


def generate(clip, tokens, max_length, fast=True, **options):
    """Generate token ids through ``clip.generate`` on the fastest decode path the model allows.

    Args:
        clip: A ComfyUI ``CLIP`` holding a language model.
        tokens: What ``clip.tokenize`` returned.
        max_length: The most tokens to generate.
        fast: False runs the model exactly as ComfyUI would.
        **options: Passed on to ``clip.generate``: sampling settings, seed and ``mtp``.

    Returns:
        ``(token_ids, report)``, the generated ids and a :class:`DecodeReport`.
    """
    report = DecodeReport()
    transformer, model = decoder_of(clip)
    reason = "turned off on the node" if not fast else unsupported(transformer, model)

    already = bool(model is not None and getattr(model, "fixed_kv", False))
    device = getattr(getattr(clip, "patcher", None), "load_device", None)
    if not reason and not already and not flash_decode_available(device):
        reason = "the flash decode kernel is not available on this device"
    if not reason and not already and not flash_decode_fits(model, device):
        reason = (
            f"the flash decode kernel does not take this model's head size of "
            f"{getattr(model.config, 'head_dim', '?')}"
        )

    graph_reason = "" if reason else graph_refusal(clip, model, device)
    graphs = not reason and not graph_reason

    values = {}
    if not reason and not already:
        values = {"fixed_kv": True, "prefetch_dynamic_vbars": graphs, "graph_dynamic_vbar_blocks": graphs}
        report.switched = True

    window = smallest_window(model) if model is not None else None
    original = getattr(transformer, "generate", None) if transformer is not None else None
    sample = getattr(transformer, "sample_token", None) if transformer is not None else None
    marks = {}

    def guarded(embeds=None, *args, **kwargs):
        config = getattr(getattr(transformer, "model", None), "config", None)
        if kwargs.get("stop_tokens") is None and len(args) < STOP_TOKENS_ARG and config is not None \
                and getattr(config, "stop_tokens", None) is None:
            raise ValueError(
                f"{type(config).__name__} names no token that ends an answer, so ComfyUI cannot "
                f"write text with it. Load a text encoder that writes, as qwen3vl_4b with type "
                f"krea2 for pictures or Qwen3 4B with type lumina2 for text")
        marks["entered"] = time.perf_counter()
        # The prompt length is known only here, after images have become embeddings.
        shape = getattr(embeds, "shape", None)
        length = int(shape[-2]) if shape is not None and len(shape) >= 2 else 0
        report.prompt_tokens = length
        limit = kwargs.get("max_length", args[1] if len(args) > 1 else max_length)
        if values and window is not None and length + int(limit) > window:
            report.switched = False
            report.reason = (
                f"{length} prompt tokens plus max_length {limit} passes the model's "
                f"{window} token sliding window"
            )
            with switched(model, {name: False for name in FLAGS}):
                return original(embeds, *args, **kwargs)
        return original(embeds, *args, **kwargs)

    def timed_sample(*args, **kwargs):
        token = sample(*args, **kwargs)
        if "first" not in marks:
            _synchronize(device)
            marks["first"] = time.perf_counter()
        return token

    started = time.perf_counter()
    try:
        with contextlib.ExitStack() as stack:
            if values:
                # Set before the model loads, so each block is registered as a graph unit.
                stack.enter_context(switched(model, values))
            if original is not None:
                stack.enter_context(_shadow(transformer, "generate", guarded))
            if sample is not None:
                stack.enter_context(_shadow(transformer, "sample_token", timed_sample))
            ids = clip.generate(tokens, **options, max_length=max_length)
            finished = time.perf_counter()
    finally:
        # The decode graphs a generation captured are dropped whether or not it finished.
        release_graphs()
    report.seconds = time.perf_counter() - started
    report.new_tokens = len(ids)
    if "entered" in marks:
        report.load_seconds = marks["entered"] - started
        if "first" in marks:
            report.prompt_seconds = marks["first"] - marks["entered"]
            report.decode_seconds = finished - marks["first"]
    if model is not None:
        report.extras["graph_units"] = sum(
            1 for layer in getattr(model, "layers", []) if getattr(layer, "_v_block", None) is not None
        )

    if reason:
        report.reason = reason
        report.path = PATH_CORE
    elif report.reason:
        report.path = PATH_CORE
    elif already or report.switched:
        report.path = PATH_GRAPH if (graphs or getattr(model, "graph_dynamic_vbar_blocks", False)) else PATH_FIXED
    if report.path == PATH_FIXED:
        report.graph_reason = graph_reason
    return ids, report


def figures(report: DecodeReport) -> tuple[str, dict, dict]:
    """What a panel shows of one generation.

    Args:
        report: The report :func:`generate` answered.

    Returns:
        ``(summary, counts, facts)``: one line, the figures, and the rows naming the decode
        path, why it was not faster, and where the time went.
    """
    rate = report.tokens_per_second
    if rate > 0:
        summary = f"{report.new_tokens} tokens, written at {rate:.1f} tokens/s"
    else:
        summary = f"{report.new_tokens} tokens in {report.seconds:.2f} s"
    counts = {"tokens": report.new_tokens, "tokens/s": round(rate, 1), "seconds": round(report.seconds, 2)}
    facts = {"decode": report.path}
    if report.reason:
        facts["why not fast"] = report.reason
    if report.graph_reason:
        facts["no graphs"] = report.graph_reason
    if report.load_seconds >= 0.05:
        facts["load"] = f"{report.load_seconds:.2f} s"
    facts["prompt"] = (
        f"{report.prompt_tokens} tokens in {report.prompt_seconds:.2f} s"
        if report.prompt_seconds > 0 else f"{report.prompt_tokens} tokens"
    )
    if report.decode_seconds > 0:
        facts["writing"] = f"{max(0, report.new_tokens - 1)} tokens in {report.decode_seconds:.2f} s"
    if report.extras.get("graph_units"):
        facts["graph blocks"] = report.extras["graph_units"]
    return summary, counts, facts


def split_reasoning(clip, tokens, prompt: str, text: str) -> tuple[str, str]:
    """Separate a reasoning model's thinking from its answer.

    Args:
        clip: The CLIP that decoded ``text``.
        tokens: What ``clip.tokenize`` returned for the prompt.
        prompt: The prompt as typed.
        text: The decoded generation.

    Returns:
        ``(answer, thinking)``, each stripped. Thinking the length limit cut off before it
        closed is all thinking and no answer.
    """
    head, mark, tail = text.partition(THINK_CLOSE)
    if mark:
        return tail.strip(), head.replace(THINK_OPEN, "", 1).strip()
    if head.lstrip().startswith(THINK_OPEN) or _opens_thinking(clip, tokens, prompt):
        return "", head.replace(THINK_OPEN, "", 1).strip()
    return text.strip(), ""


def _opens_thinking(clip, tokens, prompt: str) -> bool:
    """Whether the prompt, chat template included, ends inside a thinking block."""
    if str(prompt or "").rstrip().endswith(THINK_OPEN):
        return True
    try:
        batch = next(iter(tokens.values())) if isinstance(tokens, dict) else tokens
        ids = [item[0] for item in batch[-1] if isinstance(item, tuple) and isinstance(item[0], int)]
        return clip.decode(ids[-TEMPLATE_TAIL_TOKENS:]).rstrip().endswith(THINK_OPEN)
    except Exception as error:
        logger.debug("the end of the prompt could not be decoded (%s)", error)
        return False


@contextlib.contextmanager
def _shadow(obj, name, value):
    """Put an attribute on one instance for the length of a block.

    Args:
        obj: The instance.
        name: The attribute.
        value: What it holds meanwhile.

    Yields:
        Nothing.
    """
    had = name in obj.__dict__
    saved = obj.__dict__.get(name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if had:
            setattr(obj, name, saved)
        else:
            obj.__dict__.pop(name, None)

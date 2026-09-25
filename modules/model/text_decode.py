"""Run a ComfyUI language model's generate on its fixed-cache, graph-captured decode path.

The switches are set on the model for one generation and restored afterwards.
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


@dataclass
class DecodeReport:
    """What one generation did.

    Attributes:
        path: One of ``PATH_GRAPH``, ``PATH_FIXED`` or ``PATH_CORE``.
        reason: Why the fast path was not taken, empty when it was.
        prompt_tokens: Embedding positions handed to the model, images included.
        new_tokens: Tokens generated.
        seconds: Wall time of the generate call, prompt pass included.
        switched: True when this run set the switches, False when the model already had them.
    """

    path: str = PATH_CORE
    reason: str = ""
    prompt_tokens: int = 0
    new_tokens: int = 0
    seconds: float = 0.0
    switched: bool = False
    extras: dict = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        """Generated tokens over wall time, 0 for an empty run."""
        return self.new_tokens / self.seconds if self.seconds > 0 else 0.0


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

    graphs = bool(
        not reason
        and device is not None
        and not split_rope(model)
        and comfy.model_prefetch.malloc_graph_enabled(device)
        and not getattr(comfy.model_management.args, "disable_cuda_graphs", False)
        and clip.patcher.is_dynamic()
    )

    values = {}
    if not reason and not already:
        values = {"fixed_kv": True, "prefetch_dynamic_vbars": graphs, "graph_dynamic_vbar_blocks": graphs}
        report.switched = True

    window = smallest_window(model) if model is not None else None
    original = getattr(transformer, "generate", None) if transformer is not None else None

    def guarded(embeds=None, *args, **kwargs):
        # The prompt length is known only here, after images have become embeddings.
        length = int(embeds.shape[-2]) if embeds is not None and embeds.ndim >= 2 else 0
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

    started = time.perf_counter()
    with contextlib.ExitStack() as stack:
        if values:
            # Set before the model loads, so each block is registered as a graph unit.
            stack.enter_context(switched(model, values))
            stack.enter_context(_shadow(transformer, "generate", guarded))
        ids = clip.generate(tokens, **options, max_length=max_length)
    report.seconds = time.perf_counter() - started
    report.new_tokens = len(ids)
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
    return ids, report


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

"""LoRAs on MiniMax H3's sound tokens alone, in place of or beside the model's own adapters there.

Rows are positions in core's ``minimax_h3_layout``. qkv, out and fc1 are corrected on sound rows;
fc2's correction follows the feed-forward.
"""

from __future__ import annotations

import copy
import re

import torch

from .. import log

logger = log.get_logger("modules.model.h3_sound_lora")

__all__ = ["KEY", "MODES", "SOUND_SEGMENTS", "Entry", "Stack", "apply", "pairs_from_lora", "stack_of"]

#: Key the injection, wrapper and attachment are filed under on a model.
KEY = "was_h3_sound_lora"

#: ``replace`` drops the model's own adapters on the sound tokens; ``add`` keeps them.
MODES = ("replace", "add")

#: Packed sequence segments holding sound tokens.
SOUND_SEGMENTS = ("audio", "cond_audio", "ref_audio")

#: Projections adapted in each block, by path under the block.
PROJECTIONS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")

#: Attribute on each forward this module installs, holding the forward it wraps.
WRAPPED = "_was_h3_sound_wraps"

_TARGET = re.compile(
    r"^diffusion_model\.(blocks\.\d+\.(?:attn\.qkv_proj|attn\.out_proj|mlp\.fc1|mlp\.fc2))\.weight$")
_MISSING = object()


class Entry:
    """One sound LoRA: its name, strength, mode and low-rank pairs by block projection.

    Attributes:
        name: The file's name.
        strength: Its scale.
        mode: One of :data:`MODES`.
        pairs: ``{path: (down, up, scale)}``, factors on the CPU.
        skipped: How many of the file's adapters lie outside the block projections.
    """

    def __init__(self, name: str, strength: float, mode: str, pairs: dict, skipped: int = 0):
        self.name = name
        self.strength = strength
        self.mode = mode
        self.pairs = pairs
        self.skipped = skipped


class Stack:
    """The sound LoRAs a model carries, in the order they were applied, and the sound clock.

    Attributes:
        entries: :class:`Entry` items, first applied first.
        audio_shift: The audio flow shift the model samples at.
        video_shift: The video flow shift the model samples at.
    """

    def __init__(self, entries: list, audio_shift: float, video_shift: float):
        self.entries = entries
        self.audio_shift = audio_shift
        self.video_shift = video_shift

    @property
    def drop(self) -> bool:
        """Whether the sound tokens drop the model's own adapters."""
        return any(entry.mode == "replace" for entry in self.entries)

    def pairs(self, path: str) -> list:
        """Every entry's ``(down, up, scale)`` on one projection."""
        return [entry.pairs[path] for entry in self.entries if path in entry.pairs]

    def blocks(self) -> list[str]:
        """Paths of the blocks any entry adapts, as ``blocks.0``."""
        found = {path.rsplit(".", 2)[0] for entry in self.entries for path in entry.pairs}
        return sorted(found, key=lambda path: int(path.split(".")[1]))


def stack_of(model) -> Stack | None:
    """The :class:`Stack` a patched model carries, or None."""
    getter = getattr(model, "get_attachment", None)
    return getter(KEY) if getter is not None else None


def _pair(adapter, strength: float, where: str):
    """A plain LoRA adapter as ``(down, up, scale)``, or None for any other adapter."""
    import comfy.weight_adapter

    weights = getattr(adapter, "weights", None)
    if (not isinstance(adapter, comfy.weight_adapter.LoRAAdapter) or weights[3] is not None
            or weights[4] is not None):
        return None
    up, down, alpha = weights[0], weights[1], weights[2]
    return down, up, float(strength) * (float(alpha) / down.shape[0] if alpha is not None else 1.0)


def pairs_from_lora(model, lora: dict, strength: float, name: str) -> tuple[dict, int]:
    """A LoRA's pairs on MiniMax H3's block projections, through core's key mapping.

    Args:
        model: The ``ModelPatcher`` the LoRA is for.
        lora: The LoRA's state dict.
        strength: Its scale.
        name: Its name, for messages.

    Returns:
        ``(pairs, skipped)``: ``{path: (down, up, scale)}``, and how many of the file's adapters
        lie outside the block projections.

    Raises:
        ValueError: An adapter on a block projection is not a plain LoRA pair, a pair does not fit
            the model, or the file adapts none of the block projections.
    """
    import comfy.lora
    import comfy.lora_convert

    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    loaded = comfy.lora.load_lora(comfy.lora_convert.convert_lora(lora), key_map, log_missing=False)
    diffusion = model.get_model_object("diffusion_model")
    pairs, skipped = {}, 0
    for key, adapter in loaded.items():
        match = _TARGET.match(key)
        if match is None:
            skipped += 1
            continue
        path = match.group(1)
        pair = _pair(adapter, strength, path)
        if pair is None:
            raise ValueError(
                f"{name} carries a {type(adapter).__name__} adapter on {path}; the sound tokens take "
                f"plain LoRA pairs only. Use a LoRA without DoRA or LoCon mid weights.")
        module = diffusion.get_submodule(path)
        if pair[0].shape[1] != module.in_features or pair[1].shape[0] != module.out_features:
            raise ValueError(
                f"{name} does not fit this model: {path} is {module.in_features} to "
                f"{module.out_features} wide, and the LoRA is {pair[0].shape[1]} to {pair[1].shape[0]}.")
        pairs[path] = pair
    if not pairs:
        raise ValueError(
            f"{name} adapts none of MiniMax H3's block projections (attn.qkv_proj, attn.out_proj, "
            f"mlp.fc1, mlp.fc2). Choose a MiniMax H3 LoRA.")
    return pairs, skipped


class _State:
    """The running model call's token layout, each projection's row cursor and the rows it covered."""

    def __init__(self):
        self.options: dict | None = None
        self.cursors: dict[str, int] = {}
        self.seen: dict[str, int] = {}
        self._layout = None
        self._rows: tuple[int, list[tuple[int, int]]] = (0, [])

    def sound_rows(self) -> tuple[int, list[tuple[int, int]]]:
        """The packed sequence length and the ``(start, stop)`` spans of its sound tokens.

        Raises:
            RuntimeError: The model call published no token layout.
        """
        layout = (self.options or {}).get("minimax_h3_layout")
        if layout is None:
            raise RuntimeError(
                "H3 Sound LoRA: the MiniMax H3 model gave no token layout to find the sound tokens "
                "by. Update ComfyUI.")
        if layout is not self._layout:
            self._layout = layout
            self._rows = (int(layout.seq_len),
                          [(int(a), int(b)) for a, b, kind in layout.segments if kind in SOUND_SEGMENTS])
        return self._rows

    def spans(self, path: str, count: int) -> tuple[list, list]:
        """The picture and sound rows of a projection's next call, which covers ``count`` rows.

        Returns:
            ``(picture, sound)``: lists of ``(start, stop)`` in the call's own rows.

        Raises:
            RuntimeError: The calls run past the sequence.
        """
        total, sound_spans = self.sound_rows()
        start = self.cursors.get(path, 0)
        stop = start + count
        if stop > total:
            self.cursors[path] = 0
            raise RuntimeError(
                f"H3 Sound LoRA: {path} ran over {stop} of {total} tokens. Another patch reorders "
                f"the tokens in a way the sound rows cannot be followed through; remove that patch.")
        self.cursors[path] = 0 if stop == total else stop
        self.seen[path] = self.seen.get(path, 0) + count
        return _split(sound_spans, start, stop)

    def check(self, paths) -> None:
        """Raise unless every path covered the whole sequence once in this call.

        Raises:
            RuntimeError: A projection was skipped, called around its module, or cut short.
        """
        total = self.sound_rows()[0]
        for path in paths:
            seen = self.seen.get(path, 0)
            if seen != total:
                raise RuntimeError(
                    f"H3 Sound LoRA: {path} covered {seen} of {total} tokens in this step, so the "
                    f"sound tokens were only partly swapped. A memory, caching or attention patch "
                    f"runs it without its own module; remove that patch.")


def _split(spans, start: int, stop: int) -> tuple[list, list]:
    """``(picture, sound)`` row runs of ``[start, stop)``, counted from ``start``."""
    sound = [(max(a, start) - start, min(b, stop) - start) for a, b in spans if a < stop and b > start]
    picture, at = [], 0
    for a, b in sound:
        if a > at:
            picture.append((at, a))
        at = b
    if at < stop - start:
        picture.append((at, stop - start))
    return picture, sound


def _peel(module, forward):
    """The forward beneath core's bypass LoRA hooks on ``module``, and those hooks' adapters.

    Raises:
        ValueError: A hook's adapter computes its own bypass and cannot be applied by row.
    """
    import comfy.weight_adapter
    from comfy.weight_adapter import bypass

    adapters = []
    while True:
        hook = getattr(forward, "__self__", None)
        if (not isinstance(hook, bypass.BypassForwardHook) or hook.module is not module
                or hook.original_forward is None):
            return forward, adapters
        kind = type(hook.adapter)
        custom = getattr(kind, "bypass_forward", None)
        if custom is not None and custom not in (comfy.weight_adapter.WeightAdapterBase.bypass_forward,
                                                  comfy.weight_adapter.WeightAdapterTrainBase.bypass_forward):
            raise ValueError(
                f"A {kind.__name__} adapter on this model runs its own bypass and cannot be kept off "
                f"the sound tokens. Merge that LoRA, or set H3 Sound LoRA's mode to add.")
        adapters.append(hook.adapter)
        forward = hook.original_forward


def _merged(model_patcher, path: str) -> list:
    """The LoRA pairs merged into a projection's weight, as ``(down, up, scale)``.

    Raises:
        ValueError: A merged patch on it is not a plain LoRA pair at full model strength.
    """
    out = []
    for strength, adapter, strength_model, offset, function in model_patcher.patches.get(
            f"diffusion_model.{path}.weight", []):
        pair = _pair(adapter, strength, path)
        if pair is None or offset is not None or function is not None or float(strength_model) != 1.0:
            raise ValueError(
                f"A merged {type(adapter).__name__} patch on {path} cannot be taken off the sound "
                f"tokens. Load that LoRA in bypass mode, or set H3 Sound LoRA's mode to add.")
        out.append(pair)
    return out


def _low(rows: torch.Tensor, x: torch.Tensor, pairs, sign: float) -> None:
    """Add each pair's low-rank update of ``x`` into ``rows`` in place."""
    for down, up, scale in pairs:
        if down.dtype != x.dtype:
            down, up = down.to(x.dtype), up.to(x.dtype)
        rows.addmm_(torch.nn.functional.linear(x, down), up.t(), alpha=sign * scale)


class _Projection:
    """One block projection: the forward it wraps, the model's adapters on it, and the sound pairs.

    Attributes:
        path: The projection's path, as ``blocks.0.attn.qkv_proj``.
        original: The forward installed before this patch, run unchanged on every row.
        adapters: Core bypass adapters under ``original``, taken off the sound rows when dropping.
        merged: Merged pairs, taken off the sound rows when dropping.
        sound: Sound pairs on the compute device.
    """

    def __init__(self, path, original, adapters, merged, sound):
        self.path = path
        self.original = original
        self.adapters = adapters
        self.merged = merged
        self.sound = sound

    def run(self, x, args, kwargs, sound_rows, drop: bool):
        """``original`` on every row, then the sound rows turned to the sound pairs."""
        out = self.original(x, *args, **kwargs)
        if not sound_rows:
            return out
        if not out.is_contiguous():
            out = out.contiguous()
        flat = x.reshape(-1, x.shape[-1])
        rows = out.view(-1, out.shape[-1])
        for a, b in sound_rows:
            self.on_sound(rows[a:b], flat[a:b], drop)
        return out

    def on_sound(self, rows, x, drop: bool) -> None:
        """Turn output rows of sound tokens ``x``, which carry the model's adapters, to the sound pairs.

        Args:
            rows: Output rows, changed in place.
            x: The same rows' input.
            drop: Whether the model's adapters leave the sound tokens.
        """
        if drop:
            for adapter in self.adapters:
                rows -= adapter.h(x, None)
            _low(rows, x, self.merged, -1.0)
        _low(rows, x, self.sound, 1.0)


def _linear(projection: _Projection, state: _State, drop: bool):
    """A forward for a query, key and value or output projection, corrected on the sound rows."""

    def forward(x, *args, **kwargs):
        if state.options is None:
            return projection.original(x, *args, **kwargs)
        _picture, sound_rows = state.spans(projection.path, x.reshape(-1, x.shape[-1]).shape[0])
        return projection.run(x, args, kwargs, sound_rows, drop)

    setattr(forward, WRAPPED, projection.original)
    return forward


def _first(projection: _Projection, held: dict, state: _State, drop: bool, act):
    """A forward for fc1, corrected on the sound rows, keeping their activation for fc2's update."""

    def forward(x, *args, **kwargs):
        if state.options is None:
            return projection.original(x, *args, **kwargs)
        if "sound" not in held:
            raise RuntimeError(
                f"H3 Sound LoRA: {projection.path} ran outside its feed-forward, so fc2's update "
                f"cannot follow it. A memory patch runs the feed-forward without its own module; "
                f"remove that patch.")
        count = x.reshape(-1, x.shape[-1]).shape[0]
        start = held["seen"]
        held["seen"] = start + count
        state.seen[projection.path] = state.seen.get(projection.path, 0) + count
        _picture, sound_rows = _split(held["sound"], start, start + count)
        out = projection.run(x, args, kwargs, sound_rows, drop)
        rows = out.view(-1, out.shape[-1])
        for a, b in sound_rows:
            held["acts"].append((start + a, start + b, act(rows[a:b])))
        return out

    setattr(forward, WRAPPED, projection.original)
    return forward


def _feed_forward(original, path: str, second: _Projection, held: dict, state: _State, drop: bool):
    """A forward for a block's feed-forward that adds fc2's sound update after the fused fc2 runs."""

    def forward(x, *args, **kwargs):
        if state.options is None:
            return original(x, *args, **kwargs)
        _picture, sound_rows = state.spans(path, x.reshape(-1, x.shape[-1]).shape[0])
        held.update(sound=[(a, b) for a, b in sound_rows], seen=0, acts=[])
        try:
            out = original(x, *args, **kwargs)
            acts = held["acts"]
        finally:
            held.clear()
        if not acts:
            return out
        if not out.is_contiguous():
            out = out.contiguous()
        rows = out.view(-1, out.shape[-1])
        for a, b, activation in acts:
            second.on_sound(rows[a:b], activation, drop)
        return out

    setattr(forward, WRAPPED, original)
    return forward


def _compute_dtype(model_patcher) -> torch.dtype:
    """The dtype MiniMax H3 computes its projections in."""
    getter = getattr(model_patcher.model, "get_dtype_inference", None)
    dtype = getter() if getter is not None else None
    return dtype if dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.bfloat16


def _swiglu():
    """ComfyUI's eager SwiGLU, the activation H3's fused fc2 applies, or None where it has none."""
    import comfy.ops

    return getattr(comfy.ops, "INPUT_ACT_EAGER", {}).get("swiglu")


def apply(model, entry: Entry, audio_shift: float):
    """Return a copy of a MiniMax H3 model whose sound tokens take ``entry`` and sample on ``audio_shift``.

    Args:
        model: A ``ModelPatcher`` of MiniMax H3, LoRAs and earlier sound LoRAs allowed.
        entry: The sound LoRA.
        audio_shift: The audio flow shift to sample at.

    Returns:
        ``(model, stack)``: the patched clone and every sound LoRA it carries.

    Raises:
        ValueError: The model is not MiniMax H3, or this ComfyUI has no eager SwiGLU.
    """
    import comfy.model_management
    import comfy.patcher_extension

    act = _swiglu()
    if act is None:
        raise ValueError("H3 Sound LoRA needs a ComfyUI whose MiniMax H3 runs a fused SwiGLU. Update ComfyUI.")
    patched = model.clone()
    sampling = copy.deepcopy(patched.get_model_object("model_sampling"))
    if not hasattr(sampling, "audio_shift"):
        raise ValueError("H3 Sound LoRA takes a MiniMax H3 model.")
    video_shift = float(sampling.shift)
    sampling.set_parameters(shift=video_shift, audio_shift=float(audio_shift),
                            multiplier=getattr(sampling, "multiplier", 1000))
    patched.add_object_patch("model_sampling", sampling)
    options = dict(patched.model_options.get("transformer_options", {}))
    options["minimax_h3_sigma_shift_video"] = video_shift
    options["minimax_h3_sigma_shift_audio"] = float(audio_shift)
    patched.model_options["transformer_options"] = options

    earlier = stack_of(model)
    stack = Stack((list(earlier.entries) if earlier is not None else []) + [entry], float(audio_shift), video_shift)
    drop = stack.drop
    state = _State()
    saved: list = []
    installed: dict = {}

    def install(module, path, wrap):
        current = module.__dict__.get("forward")
        # A wrapper an earlier run of this patch left behind is peeled off first.
        while getattr(current, WRAPPED, None) is not None:
            current = getattr(current, WRAPPED)
            module.forward = current
        before = module.__dict__.get("forward", _MISSING)
        forward = wrap(module.forward)
        module.forward = forward
        saved.append((module, before, forward))
        installed[path] = (module, forward)

    def projection(model_patcher, module, path, device, dtype, original):
        _base, adapters = _peel(module, original)
        merged = _merged(model_patcher, path) if drop else []
        return _Projection(
            path, original, adapters,
            [(down.to(device, dtype), up.to(device, dtype), scale) for down, up, scale in merged],
            [(down.to(device, dtype), up.to(device, dtype), scale) for down, up, scale in stack.pairs(path)])

    def inject(model_patcher):
        diffusion = model_patcher.model.diffusion_model
        device = comfy.model_management.get_torch_device()
        dtype = _compute_dtype(model_patcher)
        for block in stack.blocks():
            for name in ("attn.qkv_proj", "attn.out_proj"):
                path = f"{block}.{name}"
                module = diffusion.get_submodule(path)
                install(module, path, lambda inner, m=module, p=path: _linear(
                    projection(model_patcher, m, p, device, dtype, inner), state, drop))
            mlp = diffusion.get_submodule(f"{block}.mlp")
            held: dict = {}
            fc1_path, fc2_path = f"{block}.mlp.fc1", f"{block}.mlp.fc2"
            # fc2 keeps its own forward and kernel; its update is added after the feed-forward.
            second = projection(model_patcher, mlp.fc2, fc2_path, device, dtype, mlp.fc2.forward)
            install(mlp.fc1, fc1_path, lambda inner, m=mlp.fc1, p=fc1_path, h=held: _first(
                projection(model_patcher, m, p, device, dtype, inner), h, state, drop, act))
            install(mlp, f"{block}.mlp", lambda inner, p=f"{block}.mlp", s=second, h=held: _feed_forward(
                inner, p, s, h, state, drop))

    def eject(model_patcher):
        installed.clear()
        while saved:
            module, forward, mine = saved.pop()
            # A module another patch has already put back keeps what that patch restored.
            if module.__dict__.get("forward") is not mine:
                continue
            if forward is _MISSING:
                module.__dict__.pop("forward", None)
            else:
                module.forward = forward

    def wrapper(executor, x, timestep, context, transformer_options, *args, **kwargs):
        for path, (module, forward) in installed.items():
            if module.__dict__.get("forward") is not forward:
                raise ValueError(
                    f"H3 Sound LoRA: a LoRA or patch was applied to {path} after H3 Sound LoRA, so it "
                    f"would reach the sound tokens. Put H3 Sound LoRA after every LoRA and patch node.")
        state.options = transformer_options
        state.cursors.clear()
        state.seen.clear()
        try:
            out = executor(x, timestep, context, transformer_options, *args, **kwargs)
            state.check(installed)
            return out
        finally:
            state.options = None

    patched.set_injections(KEY, [comfy.patcher_extension.PatcherInjection(inject=inject, eject=eject)])
    patched.remove_wrappers_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, KEY)
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, KEY, wrapper)
    patched.set_attachments(KEY, stack)
    return patched, stack

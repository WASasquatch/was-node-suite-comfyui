"""LoRA strength set per sampling step from a list of values.

LoRAs are merged at the list's first value; each step adds its difference from that value at
forward time, on layers whose patch scales with its strength.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

from .. import log
from ..util.numbers import as_number

__all__ = [
    "INJECTION_KEY",
    "MAX_VALUES",
    "WRAPPER_KEY",
    "attach",
    "step_of",
    "strength_at",
    "strength_list",
]

logger = log.get_logger("lora.schedule")

#: The key the forward hooks are registered under on a model.
INJECTION_KEY = "was_lora_schedule"

#: The key the step reader is registered under on a model.
WRAPPER_KEY = "was_lora_schedule"

#: Most values one strength list holds.
MAX_VALUES = 10000

#: Share of a sigma a level has to fall below to count as the next step.
SIGMA_TOLERANCE = 1e-6


def strength_list(value: Any, node: str = "Power LoRA Loader") -> list[float] | None:
    """A strength list out of what arrived on a strength input.

    Args:
        value: A number, or a list or tuple of numbers or of text holding them.
        node: The display name the exception is written against.

    Returns:
        The values in order, or ``None`` for a single number.

    Raises:
        ValueError: The list is empty, too long, or holds an entry that is not a number.
    """
    if not isinstance(value, (list, tuple)):
        return None
    if not value:
        raise ValueError(
            f"{node} received an empty list on strength, so there is no strength for any step. "
            f"Check the node wired into strength, or disconnect it to type one strength."
        )
    if len(value) > MAX_VALUES:
        raise ValueError(
            f"{node} received {len(value):,} values on strength and takes at most "
            f"{MAX_VALUES:,}, one per sampling step. Shorten the list."
        )
    numbers = []
    for position, entry in enumerate(value, 1):
        number = as_number(entry)
        if number is None:
            raise ValueError(
                f"{node} could not read value {position} of the strength list, "
                f"{str(entry)[:40]!r}, as a number. Every entry has to be a strength, as 0.8."
            )
        numbers.append(number)
    return numbers


def step_of(sigma: Any, sample_sigmas: Any) -> tuple[int, int] | None:
    """The sampling step a model call belongs to.

    Args:
        sigma: The noise level of the call, a number or a tensor.
        sample_sigmas: The sampler's schedule, ending on its last level, or ``None``.

    Returns:
        ``(index, steps)``, the index counted from 0, or ``None`` without a schedule. A call
        between two levels belongs to the step that started above it.
    """
    if sample_sigmas is None:
        return None
    try:
        levels = [float(level) for level in sample_sigmas.flatten().tolist()]
    except AttributeError:
        levels = [float(level) for level in sample_sigmas]
    steps = len(levels) - 1
    if steps < 1:
        return None
    level = float(sigma.max()) if hasattr(sigma, "max") else float(sigma)
    index = next(
        (step for step in range(steps) if levels[step + 1] < level * (1.0 - SIGMA_TOLERANCE)),
        steps - 1,
    )
    return index, steps


def strength_at(values: list[float], index: int, steps: int) -> float:
    """The strength a list gives one step.

    Args:
        values: The strength list.
        index: The step, from 0.
        steps: Steps in the run.

    Returns:
        ``values[index]`` when the list holds one value per step. Otherwise the list is
        stretched across the run and read between its two nearest entries.
    """
    if len(values) == steps or steps <= 1 or len(values) == 1:
        return values[min(max(index, 0), len(values) - 1)]
    place = index * (len(values) - 1) / (steps - 1)
    below = min(int(math.floor(place)), len(values) - 1)
    above = min(below + 1, len(values) - 1)
    return values[below] + (values[above] - values[below]) * (place - below)


class _State:
    """The difference each LoRA's strength at the current step makes from its merged one.

    Attributes:
        rows: One weight list per LoRA.
        strength: The strength list every LoRA's weight is multiplied by.
        node: The display name log lines are written against.
        deltas: Per LoRA, its strength at the current step less its strength at the first.
        told: The step count last reported as stretched.
    """

    def __init__(self, rows: list[list[float]], strength: list[float], node: str):
        self.rows = rows
        self.strength = strength
        self.node = node
        self.deltas = [0.0] * len(rows)
        self.told = None

    def _deltas_at(self, index: int, steps: int) -> list[float]:
        """Every LoRA's difference from its merged strength at one step."""
        scale = strength_at(self.strength, index, steps)
        start = self.strength[0]
        return [strength_at(row, index, steps) * scale - row[0] * start for row in self.rows]

    def wrapper(self, executor, *args, **kwargs):
        """Set the step's difference, then run the model call.

        Args:
            executor: The next wrapper, or the model call itself.
            args: The model call's positional arguments: ``x``, ``t``, ``c_concat``,
                ``c_crossattn``, ``control``, ``transformer_options``.
            kwargs: The model call's keyword arguments.

        Returns:
            What the model call answers.
        """
        sigma = args[1] if len(args) > 1 else kwargs.get("t")
        options = args[5] if len(args) > 5 else kwargs.get("transformer_options")
        found = step_of(sigma, (options or {}).get("sample_sigmas")) if sigma is not None else None
        if found is None:
            self.deltas = [0.0] * len(self.rows)
        else:
            index, steps = found
            lengths = {len(values) for values in (self.strength, *self.rows) if len(values) > 1}
            if lengths - {steps} and self.told != steps:
                self.told = steps
                logger.info(
                    "%s: a list of %s value(s) is stretched across %d sampling step(s)",
                    self.node, " and ".join(str(length) for length in sorted(lengths - {steps})), steps,
                )
            self.deltas = self._deltas_at(index, steps)
        return executor(*args, **kwargs)


class _Layer:
    """Every scheduled LoRA part on one layer, added to its output by a forward hook.

    Attributes:
        module: The layer.
        parts: ``(adapter, lora, offset)`` per part: the LoRA's place in the state's lists, and
            the weight slice it patches or ``None``.
    """

    def __init__(self, module, state: _State, device, dtype):
        self.module = module
        self.state = state
        self.device = device
        self.dtype = dtype
        self.parts: list[tuple[Any, int, tuple | None]] = []
        self.stored: list[Any] = []
        self.handle = None

    def inject(self) -> None:
        """Move the parts to the compute device and hook the layer."""
        if self.handle is not None:
            return
        self.stored = []
        for adapter, _, _ in self.parts:
            self.stored.append(adapter.weights)
            adapter.weights = _moved(adapter.weights, self.device, self.dtype)
        held = getattr(self.module, "_was_schedule_hooks", 0)
        if held == 0:
            self.module._was_schedule_force = getattr(self.module, "comfy_force_forward", False)
        self.module._was_schedule_hooks = held + 1
        self.module.comfy_force_forward = True
        self.handle = self.module.register_forward_hook(self.hook)

    def eject(self) -> None:
        """Unhook the layer and hand the parts their own weights back."""
        if self.handle is None:
            return
        self.handle.remove()
        self.handle = None
        for (adapter, _, _), weights in zip(self.parts, self.stored):
            adapter.weights = weights
        self.stored = []
        held = max(0, getattr(self.module, "_was_schedule_hooks", 1) - 1)
        self.module._was_schedule_hooks = held
        if held == 0:
            self.module.comfy_force_forward = getattr(self.module, "_was_schedule_force", False)

    def hook(self, module, args, output):
        """Add each part at the current step's difference.

        Args:
            module: The layer.
            args: The layer's positional inputs, the first being ``x``.
            output: What the layer answered.

        Returns:
            The output with every part added, or ``None`` to keep it as it was.
        """
        import torch

        deltas = self.state.deltas
        if not args or not torch.is_tensor(output) or not torch.is_tensor(args[0]):
            return None
        x = args[0]
        result = output
        owned = False
        for adapter, lora, offset in self.parts:
            delta = deltas[lora]
            if delta == 0.0:
                continue
            adapter.multiplier = delta
            part = adapter.h(x, output)
            if offset is None:
                result = result + part
                owned = True
                continue
            if not owned and torch.is_grad_enabled():
                result = result.clone()
            owned = True
            axis = 1 if getattr(adapter, "is_conv", False) else result.ndim - 1
            result.narrow(axis, offset[1], offset[2]).add_(part)
        return result


def _moved(weights, device, dtype):
    """A LoRA's tensors on the compute device, floating ones in the compute dtype."""
    import torch

    def move(value):
        if not torch.is_tensor(value):
            return value
        if dtype is not None and value.is_floating_point():
            return value.to(device=device, dtype=dtype)
        return value.to(device=device)

    if isinstance(weights, (list, tuple)):
        moved = [move(value) for value in weights]
        return tuple(moved) if isinstance(weights, tuple) else moved
    return move(weights)


def _scales_linearly(patch) -> bool:
    """Whether a patch's forward-time part is its strength times a fixed term."""
    try:
        from comfy.weight_adapter.loha import LoHaAdapter
        from comfy.weight_adapter.lokr import LoKrAdapter
        from comfy.weight_adapter.lora import LoRAAdapter
    except ImportError:
        return False
    weights = getattr(patch, "weights", None)
    if not isinstance(weights, (list, tuple)):
        return False
    if isinstance(patch, LoRAAdapter):
        return len(weights) > 5 and weights[4] is None and weights[5] is None
    if isinstance(patch, LoHaAdapter):
        return len(weights) > 7 and weights[7] is None
    if isinstance(patch, LoKrAdapter):
        return len(weights) > 8 and weights[8] is None
    return False


def _shape_adapter(adapter, module) -> None:
    """Give an adapter the layer facts its forward-time part reads."""
    from comfy.weight_adapter.bypass import get_module_type_info

    info = get_module_type_info(module)
    adapter.is_conv = info["is_conv"]
    adapter.conv_dim = info["conv_dim"]
    adapter.kernel_size = info["kernel_size"]
    adapter.in_channels = info["in_channels"]
    adapter.out_channels = info["out_channels"]
    adapter.kw_dict = (
        {
            "stride": info["stride"],
            "padding": info["padding"],
            "dilation": info["dilation"],
            "groups": info["groups"],
        }
        if info["is_conv"]
        else {}
    )


def attach(
    model,
    loras: Iterable[tuple[dict, list[float]]],
    strength: list[float],
    node: str = "Power LoRA Loader",
):
    """A copy of a model whose LoRAs, already merged at their first values, follow weight lists.

    Args:
        model: The model with each LoRA merged at its first weight times the first strength.
        loras: ``(state_dict, weights)`` per LoRA, ``weights`` the row's weight list, one entry
            for a fixed weight.
        strength: The strength list every row's weight is multiplied by, one entry for a fixed
            strength.
        node: The display name log lines are written against.

    Returns:
        The copy, or ``model`` itself where no LoRA part can follow the lists.

    Raises:
        RuntimeError: This ComfyUI has no forward-time LoRA support to schedule with.
    """
    import comfy.lora
    import comfy.lora_convert
    import comfy.model_management
    import comfy.utils
    import torch
    from comfy.patcher_extension import PatcherInjection, WrappersMP

    from . import fit

    try:
        import comfy.weight_adapter.bypass  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            f"{node} takes a strength list through ComfyUI's forward-time LoRA support, which "
            f"this ComfyUI does not have. Update ComfyUI, or wire single strengths."
        ) from error

    device = comfy.model_management.get_torch_device()
    dtype = None
    try:
        dtype = model.model.get_dtype_inference()
    except Exception:  # noqa: BLE001
        dtype = None
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        dtype = None

    entries = [(lora, [float(value) for value in weights]) for lora, weights in loras]
    state = _State([weights for _, weights in entries], [float(value) for value in strength], node)
    key_map = comfy.lora.model_lora_keys_unet(model.model, {})
    layers: dict[str, _Layer] = {}
    held = 0
    for position, (lora, weights) in enumerate(entries):
        if not any(weights) or not any(strength):
            continue
        loaded = comfy.lora.load_lora(comfy.lora_convert.convert_lora(lora), key_map, log_missing=False)
        for key, patch in loaded.items():
            name, offset = (key, None) if isinstance(key, str) else (key[0], key[1])
            function = key[2] if not isinstance(key, str) and len(key) > 2 else None
            # A part that does not fit was already left out of the merge at step 1.
            if not fit.fits(model, name, patch, offset):
                continue
            if (
                function is not None
                or not name.endswith(".weight")
                or (offset is not None and offset[0] != 0)
                or not _scales_linearly(patch)
            ):
                held += 1
                continue
            path = name[: -len(".weight")]
            layer = layers.get(path)
            if layer is None:
                try:
                    module = comfy.utils.get_attr(model.model, path)
                except AttributeError:
                    held += 1
                    continue
                if not isinstance(getattr(module, "weight", None), torch.Tensor):
                    held += 1
                    continue
                layer = layers[path] = _Layer(module, state, device, dtype)
            _shape_adapter(patch, layer.module)
            layer.parts.append((patch, position, offset))

    if held:
        logger.info(
            "%s: %d LoRA part(s) cannot follow a list and stay at their strength at step 1",
            node, held,
        )
    if not layers:
        logger.warning(
            "%s: no part of these LoRAs can follow a list, so every step runs at its strength "
            "at step 1", node,
        )
        return model

    patched = model.clone()
    every = list(layers.values())

    def inject(_patcher) -> None:
        for layer in every:
            layer.inject()

    def eject(_patcher) -> None:
        for layer in reversed(every):
            layer.eject()

    existing = patched.get_injections(INJECTION_KEY) or []
    patched.set_injections(INJECTION_KEY, [*existing, PatcherInjection(inject=inject, eject=eject)])
    patched.add_wrapper_with_key(WrappersMP.APPLY_MODEL, WRAPPER_KEY, state.wrapper)
    logger.debug("%s: %d layer(s) follow per-step strengths", node, len(every))
    return patched

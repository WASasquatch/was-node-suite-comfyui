"""Noise schedules laid out in a model's own time, so its flow shift moves the steps.

:func:`tangent` is RES4LYF's ``bong_tangent`` read through the model's curve; :func:`flow_shift`
sets a Flux model's shift; :func:`handoff` joins a shifted curve onto the model's own.
"""

from __future__ import annotations

import copy
import math

import torch

__all__ = [
    "REFERENCE_AREA",
    "TANGENT",
    "TANGENT_DEFAULTS",
    "flow_shift",
    "handoff",
    "native_sampling",
    "register",
    "shift_factor",
    "tangent",
    "tangent_fractions",
]

#: The tangent schedule's name in every scheduler menu.
TANGENT = "shifted_tangent"

#: The curve's settings when a scheduler menu picks it.
TANGENT_DEFAULTS = {
    "start": 1.0, "middle": 0.5, "end": 0.0,
    "pivot_1": 0.6, "pivot_2": 0.6, "slope_1": 0.2, "slope_2": 0.2,
}

#: Picture area in pixels a resolution-scaled shift is measured against.
REFERENCE_AREA = 1024 * 1024


def _stage(count: int, slope: float, pivot: float, start: float, end: float) -> list[float]:
    """``count`` values of one arctangent stage, ``start`` down to ``end``."""
    top = ((2 / math.pi) * math.atan(-slope * (0 - pivot)) + 1) / 2
    bottom = ((2 / math.pi) * math.atan(-slope * ((count - 1) - pivot)) + 1) / 2
    span = top - bottom
    scale = start - end
    return [
        ((((2 / math.pi) * math.atan(-slope * (x - pivot)) + 1) / 2) - bottom) * (1 / span) * scale + end
        for x in range(count)
    ]


def tangent_fractions(steps: int, start: float = 1.0, middle: float = 0.5, end: float = 0.0,
                      pivot_1: float = 0.6, pivot_2: float = 0.6, slope_1: float = 0.2,
                      slope_2: float = 0.2) -> list[float]:
    """The two-stage arctangent curve as fractions of the model's time.

    Args:
        steps: Sampling steps.
        start: Fraction the first step starts from.
        middle: Fraction where the second stage takes over.
        end: Fraction the last step ends at.
        pivot_1: Share of the steps where the first stage turns.
        pivot_2: Share of the steps where the second stage turns.
        slope_1: Steepness of the first stage.
        slope_2: Steepness of the second stage.

    Returns:
        ``steps + 1`` fractions, ``start`` down to ``end``.
    """
    total = steps + 2
    midpoint = int((total * pivot_1 + total * pivot_2) / 2)
    first_pivot, second_pivot = int(total * pivot_1), int(total * pivot_2)
    first_slope, second_slope = slope_1 / (total / 40), slope_2 / (total / 40)
    second_count = total - midpoint
    first_count = total - second_count
    first = _stage(first_count, first_slope, first_pivot, start, middle)[:-1]
    return first + _stage(second_count, second_slope, second_pivot - first_count, middle, end)


def tangent(model_sampling, steps: int, **settings) -> torch.Tensor:
    """Sigmas of the tangent curve read through a model's sampling curve.

    Args:
        model_sampling: The model's ``model_sampling`` object.
        steps: Sampling steps.
        **settings: Any of :data:`TANGENT_DEFAULTS`.

    Returns:
        ``steps + 1`` float32 sigmas; the last is 0 when ``end`` is 0.
    """
    import comfy.model_sampling

    fractions = tangent_fractions(steps, **{**TANGENT_DEFAULTS, **settings})
    sampling = model_sampling
    if isinstance(sampling, comfy.model_sampling.CONST):
        # A flow model's time runs from 0 to its timestep at sigma 1, read in double precision.
        top = _read(sampling.timestep, 1.0)
        sigmas = [_read(sampling.sigma, f * top) if f > 0 else 0.0 for f in fractions]
    else:
        first = float(sampling.timestep(sampling.sigma_max))
        last = float(sampling.timestep(sampling.sigma_min))
        sigmas = [float(sampling.sigma(torch.tensor(last + f * (first - last)))) for f in fractions]
        if fractions[-1] <= 0:
            sigmas[-1] = 0.0
    return torch.FloatTensor(sigmas)


def _read(curve, value: float) -> float:
    """One point of a sampling curve as a Python float, from a float where the curve takes one."""
    try:
        return float(curve(value))
    except (TypeError, AttributeError, RuntimeError):
        return float(curve(torch.tensor(value)))


def register() -> None:
    """Add :data:`TANGENT` to every scheduler menu, once."""
    import comfy.samplers

    if TANGENT in comfy.samplers.SCHEDULER_HANDLERS:
        return
    comfy.samplers.SCHEDULER_HANDLERS[TANGENT] = comfy.samplers.SchedulerHandler(tangent)
    if TANGENT not in comfy.samplers.SCHEDULER_NAMES:
        comfy.samplers.SCHEDULER_NAMES.append(TANGENT)


def flow_shift(model, shift: float, area: int = 0):
    """A clone of a flow model whose sampling curve has the given shift.

    Args:
        model: A model patcher sampling with Flux's time shift or SD3's flow shift.
        shift: The shift as a plain factor, 0 for the model's own.
        area: Picture area in pixels to scale the shift by its square root against
            :data:`REFERENCE_AREA`, 0 to leave it as given.

    Returns:
        ``(model, factor)``: the patched clone, or ``model`` itself when nothing changes, and the
        shift it samples with.

    Raises:
        ValueError: The model samples with no flow shift.
    """
    import comfy.model_sampling as sampling_module

    current = model.get_model_object("model_sampling")
    sampling = _unjoined(current)
    if not _joinable(sampling):
        raise ValueError(
            f"Flow Shift sets the shift of a flow model such as FLUX, MOSS-Image or SD3, and this "
            f"model samples with {type(sampling).__name__}, which has no shift to set. Leave Flow "
            f"Shift out for it."
        )
    flux = isinstance(sampling, sampling_module.ModelSamplingFlux)
    own = native_sampling(model)
    same_family = _joinable(own) and isinstance(own, sampling_module.ModelSamplingFlux) == flux
    factor = shift if shift > 0 else shift_factor(own if same_family else sampling)
    if area > 0:
        # The shift grows with the square root of the area, as in the SD3 paper's resolution rule.
        factor *= math.sqrt(area / REFERENCE_AREA)
    if current is sampling and math.isclose(shift_factor(sampling), factor, rel_tol=1e-9):
        return model, factor

    if flux:
        class FluxShift(sampling_module.ModelSamplingFlux, sampling_module.CONST):
            pass

        shifted = FluxShift(model.model.model_config)
        shifted.set_parameters(shift=math.log(factor))
    else:
        class FlowShift(sampling_module.ModelSamplingDiscreteFlow, sampling_module.CONST):
            pass

        shifted = FlowShift(model.model.model_config)
        shifted.set_noise_scale(getattr(sampling, "noise_scale", 1.0))
        shifted.set_parameters(shift=factor, multiplier=sampling.multiplier)
    patched = model.clone()
    patched.add_object_patch("model_sampling", shifted)
    return patched, factor


def native_sampling(model):
    """A model's own sampling curve, as its config sets it, under any patch.

    Args:
        model: A model patcher.

    Returns:
        A new ``model_sampling`` object, or the patched one where none can be built.
    """
    import comfy.model_base

    base = getattr(model, "model", None)
    try:
        return comfy.model_base.model_sampling(base.model_config, base.model_type)
    except Exception:
        return model.get_model_object("model_sampling")


def shift_factor(sampling) -> float | None:
    """A flow sampling curve's shift as a plain factor.

    Args:
        sampling: A ``model_sampling`` object.

    Returns:
        The factor, or None for a curve with no flow shift.
    """
    import comfy.model_sampling

    sampling = _unjoined(sampling)
    if isinstance(sampling, comfy.model_sampling.ModelSamplingFlux):
        return math.exp(float(sampling.shift))
    if isinstance(sampling, comfy.model_sampling.ModelSamplingDiscreteFlow):
        return float(sampling.shift)
    return None


class _Handoff:
    """A flow sampling curve following a whole-frame curve, then the model's own."""

    def sigma(self, timestep):
        """Sigma at a timestep, on the whole-frame curve above the join and the model's own below."""
        join = self.__dict__.get("was_handoff")
        if join is None:
            return super().sigma(timestep)
        frame, own, span, own_from, frame_from = join
        tensor = torch.is_tensor(timestep)
        t = torch.as_tensor(timestep, dtype=torch.float64)
        share = t / span
        stretch = (1.0 - frame_from) / (1.0 - own_from)
        upper = frame.sigma((frame_from + (share - own_from).clamp(min=0.0) * stretch) * span)
        lower = own.sigma(t)
        joined = torch.where(share >= own_from, torch.as_tensor(upper), torch.as_tensor(lower))
        return joined.to(timestep.dtype) if tensor else float(joined)

    def percent_to_sigma(self, percent):
        """Sigma at a share of the schedule, read along the joined curve."""
        join = self.__dict__.get("was_handoff")
        if join is None:
            return super().percent_to_sigma(percent)
        if percent <= 0.0:
            return 1.0
        if percent >= 1.0:
            return 0.0
        return float(self.sigma((1.0 - percent) * join[2]))


_JOINED_TYPES: dict = {}


def _unjoined(sampling):
    """The whole-frame curve of a joined curve; any other curve as it is."""
    join = getattr(sampling, "__dict__", {}).get("was_handoff")
    return join[0] if join is not None else sampling


def _joinable(sampling) -> bool:
    """Whether a sampling curve is a flow curve with a shift that can be joined."""
    import comfy.model_sampling as sampling_module

    audio = getattr(sampling_module, "ModelSamplingAV", ())
    families = (sampling_module.ModelSamplingFlux, sampling_module.ModelSamplingDiscreteFlow)
    return (isinstance(sampling, families) and isinstance(sampling, sampling_module.CONST)
            and not (audio and isinstance(sampling, audio)))


def _at(curve, share: float, span: float) -> float:
    """One point of a curve at a share of its time, in double precision."""
    return float(curve.sigma(torch.tensor(share * span, dtype=torch.float64)))


def handoff(frame, own, start: float):
    """A sampling curve following ``frame`` over the first ``start`` of the model's time, then ``own``.

    Args:
        frame: The curve the whole-frame steps follow, such as a shifted one.
        own: The model's own curve, followed from the join to the end.
        start: Share of the model's time, 0 to 1, laid along ``frame``.

    Returns:
        The joined ``model_sampling`` object, meeting ``own`` at the sigma ``own`` has at
        ``start``; ``own`` when ``start`` is 0; None when the two curves are the same or
        cannot be joined.
    """
    frame = _unjoined(frame)
    if not (_joinable(frame) and _joinable(own)):
        return None
    span = float(frame.timestep(1.0))
    if not math.isclose(span, float(own.timestep(1.0)), rel_tol=1e-9):
        return None
    if all(math.isclose(_at(frame, u, span), _at(own, u, span), abs_tol=1e-7)
           for u in (0.1, 0.3, 0.5, 0.7, 0.9)):
        return None
    if start <= 0.0:
        return own
    if start >= 1.0:
        return frame
    own_from = 1.0 - start
    target = _at(own, own_from, span)
    low, high = 0.0, 1.0
    for _ in range(64):
        middle = (low + high) / 2
        if _at(frame, middle, span) < target:
            low = middle
        else:
            high = middle
    frame_from = (low + high) / 2

    kind = type(frame)
    if kind not in _JOINED_TYPES:
        _JOINED_TYPES[kind] = type(f"Handoff{kind.__name__}", (_Handoff, kind), {})
    joined = copy.deepcopy(frame)
    joined.__class__ = _JOINED_TYPES[kind]
    joined.was_handoff = (frame, own, span, own_from, frame_from)
    count = len(frame.sigmas)
    grid = torch.arange(1, count + 1, 1, dtype=torch.float32) / count * span
    joined.register_buffer("sigmas", joined.sigma(grid).to(frame.sigmas.dtype))
    return joined

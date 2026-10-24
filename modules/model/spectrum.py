"""Spectrum feature forecasting (Han et al., CVPR 2026): sampling steps that skip a diffusion
transformer's blocks through ``patches_replace["dit"]``, the last block's output taken from a ridge
Chebyshev fit over the steps that ran in full.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field

import torch

from .. import log
from .adaptive_detail import EXTRA_PASS, SCHEDULE_SIGMA
from .sampling_window import steps_inside

logger = log.get_logger("modules.model.spectrum")

__all__ = [
    "COORDINATES", "FEATURES", "KEY", "LOG_SNR", "MIN_STEPS", "OUTPUT", "RESIDUAL", "SIGMA",
    "STEP", "TARGETS",
    "Settings", "apply", "block_keys", "chebyshev_weights", "schedule", "window",
]

#: Key the holder, the wrappers and the block patches are filed under.
KEY = "was_spectrum"

#: What is forecast: the last block's output, its change over the step's own block input, or the
#: whole model's output.
FEATURES = "block output"
RESIDUAL = "block residual"
OUTPUT = "model output"
TARGETS = (FEATURES, RESIDUAL, OUTPUT)

#: The coordinate the polynomial is fitted over: the step index, the noise level, or the
#: log signal-to-noise ratio.
STEP = "step"
SIGMA = "sigma"
LOG_SNR = "log-snr"
COORDINATES = (STEP, SIGMA, LOG_SNR)

#: Fewest steps a run is forecast in; a shorter run samples every step in full.
MIN_STEPS = 10

#: Diffusion model classes whose forward consults ``patches_replace["dit"]``, with each block list
#: in the order the forward runs it and the hook name of its blocks.
HOOKED = {
    "MOSSImage": (("layers", "single_block"),),
    "Fibo2": (("layers", "single_block"),),
    "QwenImage21Transformer2DModel": (("transformer_blocks", "single_block"),),
    "QwenImageTransformer2DModel": (("transformer_blocks", "double_block"),),
    "Flux": (("double_blocks", "double_block"), ("single_blocks", "single_block")),
    "WanModel": (("blocks", "double_block"),),
    "MiniMaxH3Model": (("blocks", "double_block"),),
}

#: Rows sampled from a feature to measure a forecast's error.
ERROR_SAMPLE_ROWS = 512


@dataclass
class Settings:
    """How a run is scheduled and forecast.

    Attributes:
        start_percent: Sampling percent where forecasting starts; every step before it runs in full.
        end_percent: Sampling percent where forecasting ends; every step after it runs in full.
        interval: Steps from one full run to the next inside the window.
        growth: Steps added to the interval after each full run.
        taper: Whether the interval narrows again toward ``end_percent``, by the same steps it
            widened by after ``start_percent``.
        tolerance: Disagreement between the Chebyshev and the linear forecast, relative to the
            Chebyshev one, above which a scheduled forecast runs in full instead; ``0`` never
            checks.
        degree: Highest Chebyshev degree fitted.
        ridge: Ridge weight of the fit.
        history: Most full runs a forecast is fitted over.
        target: One of :data:`TARGETS`.
        coordinate: One of :data:`COORDINATES`.
    """

    start_percent: float = 0.3
    end_percent: float = 0.95
    interval: float = 2.0
    growth: float = 2.0
    taper: bool = True
    tolerance: float = 0.0
    degree: int = 4
    ridge: float = 0.1
    history: int = 8
    target: str = FEATURES
    coordinate: str = STEP


def window(total: int, settings: Settings, sigmas=None, model_sampling=None) -> list[bool]:
    """Which steps of a run lie between ``start_percent`` and ``end_percent``.

    Args:
        total: Steps in the run.
        settings: The schedule.
        sigmas: The run's noise levels, one more than ``total``.
        model_sampling: The model's sampling object, which turns a percent into a noise level;
            without it, or without ``sigmas``, a percent is a share of the steps.

    Returns:
        One flag per step, True where the step may be forecast.
    """
    return steps_inside(total, settings.start_percent, settings.end_percent, sigmas, model_sampling)


def _span_steps(first: int, stop: int, settings: Settings) -> set[int]:
    """The steps run in full inside one stretch of forecastable steps, ``first`` to ``stop - 1``."""
    lower, upper = first - 1, stop
    gap = max(1.0, float(settings.interval))
    growth = max(0.0, float(settings.growth))
    full = set()
    while True:
        step = max(1, math.floor(gap))
        room = upper - lower
        if room <= step:
            return full
        if not settings.taper:
            lower += step
            full.add(lower)
        elif room <= 2 * step:
            full.add(lower + room // 2)
            return full
        elif room <= 3 * step:
            full.update((lower + room // 3, lower + (2 * room) // 3))
            return full
        else:
            lower += step
            upper -= step
            full.update((lower, upper))
        gap += growth


def schedule(inside: list[bool], settings: Settings) -> set[int]:
    """The steps a schedule runs in full.

    Args:
        inside: One flag per step, True where the step may be forecast, from :func:`window`.
        settings: The schedule.

    Returns:
        The step indices, counting from 0, run in full.
    """
    total = len(inside)
    full = {step for step in range(total) if not inside[step]}
    step = 0
    while step < total:
        if not inside[step]:
            step += 1
            continue
        stop = step
        while stop < total and inside[stop]:
            stop += 1
        full |= _span_steps(step, stop, settings)
        step = stop
    return full


def chebyshev_weights(times, target: float, degree: int, ridge: float) -> list[float]:
    """Weights that turn the stored features into the ridge Chebyshev forecast at ``target``.

    Args:
        times: Coordinates of the stored features, each in ``[-1, 1]``.
        target: Coordinate of the forecast, in ``[-1, 1]``.
        degree: Highest Chebyshev degree; lowered to one less than the number of features.
        ridge: Ridge weight.

    Returns:
        One weight per stored feature; their weighted sum is the forecast.
    """
    count = len(times)
    if count == 0:
        return []
    if count == 1:
        return [1.0]
    order = max(0, min(int(degree), count - 1))

    def basis(tau):
        row = [1.0]
        if order >= 1:
            row.append(tau)
        for _ in range(2, order + 1):
            row.append(2.0 * tau * row[-1] - row[-2])
        return row

    design = torch.tensor([basis(float(t)) for t in times], dtype=torch.float64)
    point = torch.tensor(basis(float(target)), dtype=torch.float64)
    normal = design.T @ design + float(ridge) * torch.eye(order + 1, dtype=torch.float64)
    solved = torch.linalg.solve(normal, point)
    return (design @ solved).tolist()


def block_keys(diffusion) -> list[tuple[str, int]]:
    """The ``patches_replace["dit"]`` keys of a diffusion model's blocks, in the order they run.

    Args:
        diffusion: The diffusion model.

    Returns:
        The keys, or an empty list for a model whose forward offers no block hooks.
    """
    for name, lists in HOOKED.items():
        if not any(cls.__name__ == name for cls in type(diffusion).__mro__):
            continue
        keys = []
        for attribute, hook in lists:
            blocks = getattr(diffusion, attribute, None)
            if blocks is None:
                return []
            keys += [(hook, index) for index in range(len(blocks))]
        return keys
    return []


@dataclass
class _History:
    """The full runs one batch of conditionings has been fitted over."""

    times: list = field(default_factory=list)
    features: list = field(default_factory=list)

    def add(self, time_: float, feature: torch.Tensor, limit: int) -> None:
        self.times.append(float(time_))
        self.features.append(feature)
        while len(self.times) > max(2, int(limit)):
            self.times.pop(0)
            self.features.pop(0)

    def disagreement(self, target: float, settings: Settings) -> float:
        """How far the Chebyshev forecast at ``target`` is from the line through the last two features."""
        rows = _rows(self.features[-1])
        flat = [f.reshape(-1, f.shape[-1])[rows].float() for f in self.features[-2:]]
        t0, t1 = self.times[-2:]
        slope = (target - t1) / (t1 - t0) if t1 != t0 else 0.0
        linear = flat[1] + (flat[1] - flat[0]) * slope
        spectral = self.forecast(target, settings, rows)
        return _relative_error(linear, spectral)

    def forecast(self, target: float, settings: Settings, rows=None) -> torch.Tensor:
        weights = chebyshev_weights(self.times, target, settings.degree, settings.ridge)
        pick = (lambda f: f) if rows is None else (lambda f: f.reshape(-1, f.shape[-1])[rows])
        first = pick(self.features[0])
        out = torch.zeros(first.shape, dtype=torch.float32, device=first.device)
        for weight, feature in zip(weights, self.features):
            out.add_(pick(feature), alpha=float(weight))
        return out


def _rows(feature: torch.Tensor) -> slice:
    """A stride over a feature's rows, for measuring an error cheaply."""
    rows = feature.reshape(-1, feature.shape[-1]).shape[0]
    return slice(0, rows, max(1, rows // ERROR_SAMPLE_ROWS))


def _relative_error(forecast: torch.Tensor, actual: torch.Tensor) -> float:
    actual = actual.float()
    norm = float(actual.norm())
    return float((forecast.float() - actual).norm()) / norm if norm > 0 else 0.0


@dataclass
class _Call:
    """What one model call does, read by the block patches."""

    mode: str = "pass"
    forecast: torch.Tensor | None = None
    first: torch.Tensor | None = None
    last: torch.Tensor | None = None
    order: list = field(default_factory=list)


class _Holder:
    """A run's schedule, histories and record, cloned for each sampling run."""

    def __init__(self, settings: Settings, keys, node_id=None):
        self.settings = settings
        self.keys = list(keys)
        self.node_id = node_id
        self.reset()

    def clone(self) -> "_Holder":
        return _Holder(copy.copy(self.settings), self.keys, self.node_id)

    def reset(self) -> None:
        self.sigmas: list[float] = []
        self.total = 0
        self.full: set[int] = set()
        self.gated: list[int] = []
        self.decided: dict[int, bool] = {}
        self.histories: dict = {}
        self.seen: set = set()
        self.shared = False
        self.verified = False
        self.residual = None
        self.output_ok = True
        self.call = _Call()
        self.ran_full: list[int] = []
        self.forecast_steps: list[int] = []
        self.errors: list[float] = []
        self.started = 0.0

    def prepare(self, sigmas, model_sampling=None) -> None:
        self.reset()
        self.sigmas = [float(s) for s in sigmas]
        self.total = max(0, len(self.sigmas) - 1)
        inside = window(self.total, self.settings, self.sigmas, model_sampling)
        self.full = (schedule(inside, self.settings) if self.total >= MIN_STEPS
                     else set(range(self.total)))
        self.flow = bool(self.sigmas) and max(self.sigmas) <= 1.0 + 1e-6
        self.started = time.perf_counter()

    def position(self, sigma: float) -> float:
        """The fractional step index of a noise level on this run's schedule."""
        values = self.sigmas
        for index in range(len(values) - 1):
            high, low = values[index], values[index + 1]
            if min(high, low) <= sigma <= max(high, low):
                span = high - low
                return index + ((high - sigma) / span if span else 0.0)
        return 0.0 if sigma >= values[0] else float(self.total)

    def log_snr(self, sigma: float) -> float:
        """The log signal-to-noise ratio of a noise level, for a flow or a variance-exploding model."""
        if self.flow:
            sigma = min(max(sigma, 1e-4), 1.0 - 1e-4)
            return 2.0 * math.log((1.0 - sigma) / sigma)
        return -2.0 * math.log(max(sigma, 1e-6))

    def coordinate(self, sigma: float) -> float:
        """A noise level as the ``[-1, 1]`` coordinate the fit is taken over."""
        if self.settings.coordinate in (SIGMA, LOG_SNR) and self.sigmas:
            measure = (lambda s: s) if self.settings.coordinate == SIGMA else (lambda s: -self.log_snr(s))
            nonzero = [s for s in self.sigmas if s > 0] or self.sigmas
            top, bottom = measure(nonzero[0]), measure(nonzero[-1])
            span = top - bottom
            return 2.0 * (top - measure(sigma)) / span - 1.0 if span else 0.0
        return 2.0 * self.position(sigma) / max(1, self.total) - 1.0

    def full_step(self, step: int, history, coordinate: float) -> bool:
        """Whether a step runs in full: scheduled, or a forecast past the tolerance, decided once per step."""
        if step not in self.decided:
            decision = step in self.full or step >= self.total
            tolerance = float(self.settings.tolerance)
            if not decision and tolerance > 0 and len(history.times) >= 2:
                if history.disagreement(coordinate, self.settings) > tolerance:
                    decision = True
                    self.gated.append(step)
            self.decided[step] = decision
        return self.decided[step]

    def ran(self, step: int, error: float | None) -> None:
        """Record a full step."""
        if step in self.ran_full:
            return
        self.ran_full.append(step)
        if error is not None:
            self.errors.append(error)

    def report(self) -> dict:
        """What the run did, as the node's report."""
        full = set(self.ran_full)
        skipped = set(self.forecast_steps)
        pattern = "".join("F" if s in full else ("." if s in skipped else "?") for s in range(self.total))
        return {
            "total": self.total,
            "full": len(full),
            "forecast": len(skipped),
            "ratio": self.total / max(1, len(full)) if self.total else 1.0,
            "pattern": pattern,
            "seconds": time.perf_counter() - self.started if self.started else 0.0,
            "mean_error": sum(self.errors) / len(self.errors) if self.errors else None,
            "max_error": max(self.errors) if self.errors else None,
            "shared": self.shared,
            "gated": len(self.gated),
        }


def _options_of(args, kwargs):
    """The ``transformer_options`` a diffusion model call was handed."""
    options = kwargs.get("transformer_options")
    if isinstance(options, dict):
        return options
    for value in reversed(args):
        if isinstance(value, dict) and (KEY in value or "sigmas" in value):
            return value
    return None


def _first_tensor(value):
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)) and value and torch.is_tensor(value[0]):
        return value[0]
    return None


def _branch(options, x) -> tuple:
    """The key one batch of conditionings keeps its history under."""
    uuids = tuple(str(u) for u in options.get("uuids", ()) or ())
    halves = tuple(options.get("cond_or_uncond", ()) or ())
    return uuids, halves, tuple(x.shape)


def _diffusion_wrapper(executor, *args, **kwargs):
    options = _options_of(args, kwargs)
    holder = options.get(KEY) if options else None
    x = _first_tensor(args[0]) if args else None
    sigmas = options.get("sigmas") if options else None
    if holder is None or x is None or sigmas is None or not holder.sigmas or options.get(EXTRA_PASS):
        return executor(*args, **kwargs)
    settings = holder.settings
    sigma = float(options[SCHEDULE_SIGMA]) if SCHEDULE_SIGMA in options else float(sigmas.flatten()[0])
    step = min(max(0, holder.total - 1), int(math.floor(holder.position(sigma) + 1e-6)))
    coordinate = holder.coordinate(sigma)
    branch = _branch(options, x)
    if (branch, round(sigma, 7)) in holder.seen:
        holder.shared = True
    holder.seen.add((branch, round(sigma, 7)))
    history = holder.histories.setdefault(branch, _History())
    feature_level = settings.target != OUTPUT and bool(holder.keys)
    forecastable = (not holder.shared and len(history.times) >= 2
                    and not holder.full_step(step, history, coordinate))

    if forecastable and feature_level and holder.verified:
        holder.call = _Call(mode="forecast", forecast=history.forecast(coordinate, settings))
        options["prefetch_dynamic_vbars"] = False
        try:
            out = executor(*args, **kwargs)
            used = holder.call.mode == "used"
        finally:
            holder.call = _Call()
        if used:
            if step not in holder.forecast_steps:
                holder.forecast_steps.append(step)
            return out
        holder.keys = []
        holder.ran(step, None)
        return out
    if forecastable and not feature_level and holder.output_ok:
        forecast = history.forecast(coordinate, settings)
        if settings.target == RESIDUAL:
            forecast = forecast + x.float()
        if step not in holder.forecast_steps:
            holder.forecast_steps.append(step)
        return forecast.to(x.dtype)

    holder.call = _Call(mode="full")
    try:
        out = executor(*args, **kwargs)
        call = holder.call
    finally:
        holder.call = _Call()
    if feature_level:
        if not holder.verified:
            # The hooks ran in the table's order and the last one ran last: forecasts may start.
            holder.verified = call.order == holder.keys
            if not holder.verified:
                logger.info("Spectrum: this model's blocks did not run as listed; every step runs in full")
                holder.keys = []
        feature = call.last
        if feature is not None:
            if holder.residual is None:
                holder.residual = (settings.target == RESIDUAL and call.first is not None
                                   and call.first.shape == feature.shape)
            feature = feature - call.first if holder.residual else feature.clone()
    else:
        produced = out if torch.is_tensor(out) else None
        if produced is None:
            holder.output_ok = False
            feature = None
        else:
            feature = produced.float() - x.float() if settings.target == RESIDUAL else produced.clone()
    error = None
    if feature is not None:
        if len(history.times) >= 2:
            rows = _rows(feature)
            predicted = history.forecast(coordinate, settings, rows)
            error = _relative_error(predicted, feature.reshape(-1, feature.shape[-1])[rows])
        history.add(coordinate, feature, settings.history)
    holder.ran(step, error)
    return out


def _block_patch(key, last, previous):
    """A ``patches_replace["dit"]`` patch for one block: passes it on, skips it, or forecasts."""

    def run(args, extra):
        return previous(args, extra) if previous is not None else extra["original_block"](args)

    def patch(args, extra):
        options = args.get("transformer_options") or {}
        holder = options.get(KEY)
        call = holder.call if holder is not None else None
        if call is None or call.mode in ("pass", "unmatched"):
            return run(args, extra)
        if call.mode == "full":
            call.order.append(key)
            if call.first is None and holder.settings.target == RESIDUAL:
                call.first = args["img"].clone()
            out = run(args, extra)
            if last:
                call.last = out["img"]
            return out
        if not last:
            return {name: args[name] for name in ("img", "txt") if name in args}
        img = args["img"]
        forecast = call.forecast
        if forecast is None or tuple(forecast.shape) != tuple(img.shape):
            call.mode = "unmatched"
            return run(args, extra)
        if holder.residual:
            img.add_(forecast.to(img.dtype))
        else:
            img.copy_(forecast.to(img.dtype))
        call.mode = "used"
        out = {"img": img}
        if "txt" in args:
            out["txt"] = args["txt"]
        return out

    return patch


def _sample_wrapper(executor, *args, **kwargs):
    import comfy.model_patcher

    guider = executor.class_obj
    original = guider.model_options
    guider.model_options = comfy.model_patcher.create_model_options_clone(original)
    options = guider.model_options.setdefault("transformer_options", {})
    holder = options[KEY].clone()
    options[KEY] = holder
    try:
        model_sampling = guider.model_patcher.get_model_object("model_sampling")
    except Exception:
        model_sampling = None
    holder.prepare(args[3], model_sampling)
    try:
        return executor(*args, **kwargs)
    finally:
        try:
            _publish(holder)
        finally:
            holder.reset()
            guider.model_options = original


def _publish(holder: _Holder) -> None:
    """Log the run and file it as the node's report."""
    report = holder.report()
    level = "block features" if holder.keys and holder.settings.target != OUTPUT else "model output"
    if report["total"] and report["total"] < MIN_STEPS:
        level = f"none: runs under {MIN_STEPS} steps sample every step"
    logger.info(
        "Spectrum: %d of %d steps in full, %d forecast at the %s, %.2fx fewer passes, %.1fs",
        report["full"], report["total"], report["forecast"], level, report["ratio"], report["seconds"],
    )
    if holder.node_id is None:
        return
    try:
        from ..interface import run_result

        facts = {
            "steps": report["pattern"] or "-",
            "forecast at": level,
            "sampled in": f"{report['seconds']:.1f} s",
        }
        if report["mean_error"] is not None:
            facts["forecast error"] = f"{report['mean_error']:.3f} mean, {report['max_error']:.3f} worst"
        if report["gated"]:
            facts["checked"] = f"{report['gated']} forecast(s) ran in full past the tolerance"
        if report["shared"]:
            facts["note"] = "the sampler calls the model twice a step; all ran in full"
        run_result.publish(
            summary=(f"{report['full']} of {report['total']} steps in full, "
                     f"{report['ratio']:.2f}x fewer transformer passes"),
            counts={"full steps": report["full"], "forecast steps": report["forecast"]},
            facts=facts,
            node_id=holder.node_id,
        )
    except Exception as error:
        logger.debug("the Spectrum report was not published (%s)", error)


def apply(model, settings: Settings, node_id=None):
    """A copy of a model whose sampling forecasts some steps instead of running them.

    Args:
        model: The ``ModelPatcher`` to patch.
        settings: The schedule and the fit.
        node_id: The node to file each run's report under, or None.

    Returns:
        ``(model, keys)``: the patched copy, and the block hooks it uses, empty where the model
        is forecast at its output.
    """
    import comfy.patcher_extension

    patched = model.clone()
    diffusion = patched.get_model_object("diffusion_model")
    keys = block_keys(diffusion) if settings.target != OUTPUT else []
    replaces = patched.model_options.get("transformer_options", {}).get("patches_replace", {}).get("dit", {})
    for index, key in enumerate(keys):
        previous = replaces.get(key)
        patched.set_model_patch_replace(_block_patch(key, index == len(keys) - 1, previous), "dit", key[0], key[1])
    patched.model_options.setdefault("transformer_options", {})[KEY] = _Holder(settings, keys, node_id)
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, KEY, _sample_wrapper)
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, KEY, _diffusion_wrapper)
    return patched, keys

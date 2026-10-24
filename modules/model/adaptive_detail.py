"""Adaptive Detail: a sampling patch that tells the model a lower noise level on the steps where
detail forms, and can add second-order passes where fine detail changes fastest, through a
``PREDICT_NOISE`` wrapper.
"""

from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .. import log
from .sampling_window import steps_inside

logger = log.get_logger("modules.model.adaptive_detail")

__all__ = ["EXTRA_PASS", "KEY", "SCHEDULE_SIGMA", "Settings", "apply", "detail_band"]

#: Key the holder and the wrappers are filed under.
KEY = "was_adaptive_detail"

#: ``transformer_options`` flag set while an extra pass runs.
EXTRA_PASS = "was_extra_pass"

#: ``transformer_options`` key holding the schedule's noise level while the model is told a lower one.
SCHEDULE_SIGMA = "was_schedule_sigma"

#: Relative difference within which a call's noise level is taken as a step of the schedule.
SIGMA_MATCH = 1e-4

#: Most recent step measurements the trend of the remaining steps is fitted over.
TREND_SPAN = 6


@dataclass
class Settings:
    """How hard a run is pushed toward detail, where, and how many extra passes it gets.

    Attributes:
        detail: Share by which the noise level the model is told is lowered inside the window.
        start_percent: Sampling percent where the push and the extra passes may start.
        end_percent: Sampling percent where the push stops.
        extra: Second-order passes as a share of the run's steps, placed from ``start_percent``
            to the last step.
    """

    detail: float = 0.04
    start_percent: float = 0.3
    end_percent: float = 0.8
    extra: float = 0.0


def detail_band(tensor: torch.Tensor) -> torch.Tensor:
    """The fine-detail band of a latent: the latent less a 3 by 3 box blur over its last two axes."""
    if tensor.ndim < 3 or min(tensor.shape[-2:]) < 3:
        return tensor.float()
    flat = tensor.float().reshape(-1, 1, *tensor.shape[-2:])
    blurred = F.avg_pool2d(flat, 3, stride=1, padding=1, count_include_pad=False)
    return (flat - blurred).reshape(tensor.shape)


def _euler(x, denoised, sigma: float, sigma_next: float):
    return x + (sigma_next - sigma) * (x - denoised) / sigma


def _landing_prediction(x, landed, sigma: float, sigma_next: float):
    """The prediction whose Euler step from ``x`` at ``sigma`` lands on ``landed`` at ``sigma_next``."""
    return x - (landed - x) * sigma / (sigma_next - sigma)


class _Holder:
    """A run's schedule, budget and record, cloned for each sampling run."""

    def __init__(self, settings: Settings, node_id=None):
        self.settings = settings
        self.node_id = node_id
        self.reset()

    def clone(self) -> "_Holder":
        return _Holder(copy.copy(self.settings), self.node_id)

    def reset(self) -> None:
        self.sigmas: list[float] = []
        self.total = 0
        self.pushing: list[bool] = []
        self.refinable: list[bool] = []
        self.budget = 0
        self.previous = None
        self.rates: dict[int, float] = {}
        self.evaluated: set[int] = set()
        self.pushed: list[int] = []
        self.refined: list[int] = []
        self.between = False
        self.started = 0.0

    def prepare(self, sigmas, model_sampling=None) -> None:
        self.reset()
        self.sigmas = [float(s) for s in sigmas]
        self.total = max(0, len(self.sigmas) - 1)
        settings = self.settings
        self.pushing = steps_inside(self.total, settings.start_percent, settings.end_percent,
                                    self.sigmas, model_sampling)
        if self.total:
            self.pushing[-1] = False
        self.refinable = steps_inside(self.total, settings.start_percent, 1.0, self.sigmas,
                                      model_sampling)
        self.budget = min(int(round(max(0.0, float(settings.extra)) * self.total)),
                          sum(self.refinable))
        self.started = time.perf_counter()

    def step_of(self, sigma: float):
        """The schedule step a noise level starts, or None for a level between steps."""
        for index in range(self.total):
            value = self.sigmas[index]
            if abs(value - sigma) <= SIGMA_MATCH * max(abs(value), 1e-8):
                return index
        return None

    def push(self, step: int) -> float:
        """The factor the noise level the model is told is multiplied by at a step: 1 for none."""
        amount = max(0.0, float(self.settings.detail))
        if amount <= 0 or not self.pushing[step] or self.sigmas[step + 1] <= 0:
            return 1.0
        return 1.0 - min(amount, 0.9)

    def observe(self, step: int, velocity: torch.Tensor) -> None:
        """Record how fast the fine detail of the velocity changed since the previous step."""
        detail = detail_band(velocity)
        if self.previous is not None and self.previous[0] == step - 1:
            span = abs(self.sigmas[step] - self.sigmas[step - 1])
            if span > 0:
                self.rates[step] = float((detail - self.previous[1]).norm()) / span
        self.previous = (step, detail)

    def _error(self, step: int, rate: float) -> float:
        """Estimated fine-detail error of an Euler step: its length squared times the rate."""
        length = self.sigmas[step] - self.sigmas[step + 1]
        return length * length * rate

    def _trend(self, step: int, ahead: list[int]):
        """Rates of the steps ahead, from a log-linear fit over the most recent rates."""
        known = sorted((s, r) for s, r in self.rates.items() if s <= step and r > 0)[-TREND_SPAN:]
        if len(known) < 3:
            return [self.rates[step]] * len(ahead)
        xs = [float(s) for s, _ in known]
        ys = [math.log(r) for _, r in known]
        mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
        spread = sum((x - mean_x) ** 2 for x in xs)
        if spread <= 0:
            return [self.rates[step]] * len(ahead)
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / spread
        return [math.exp(mean_y + slope * (s - mean_x)) for s in ahead]

    def choose(self, step: int) -> bool:
        """Whether a step gets an extra pass: refinable, within budget, and among the largest
        estimated errors the remaining budget can cover."""
        slots = self.budget - len(self.refined)
        if self.between or slots <= 0 or step >= self.total or not self.refinable[step]:
            return False
        ahead = [s for s in range(step + 1, self.total) if self.refinable[s]]
        if slots > len(ahead):
            return True
        rate = self.rates.get(step)
        if rate is None:
            return False
        mine = self._error(step, rate)
        larger = sum(1 for s, r in zip(ahead, self._trend(step, ahead)) if self._error(s, r) > mine)
        return larger < slots

    def report(self) -> dict:
        """What the run did, as the node's report."""
        pushed, refined = set(self.pushed), set(self.refined)

        def mark(step):
            if step in pushed and step in refined:
                return "*"
            if step in refined:
                return "+"
            if step in pushed:
                return "d"
            return "." if step in self.evaluated else "-"

        window = [s for s, flag in enumerate(self.pushing) if flag]
        return {
            "total": self.total,
            "pushed": len(pushed),
            "refined": len(refined),
            "budget": self.budget,
            "pattern": "".join(mark(s) for s in range(self.total)),
            "window": (window[0], window[-1]) if window else None,
            "seconds": time.perf_counter() - self.started if self.started else 0.0,
            "between": self.between,
        }


def _second_order(executor, holder: _Holder, x, timestep, model_options, seed, denoised, step: int):
    """The prediction of a refined step: a Heun step, or two half steps when it ends at zero noise."""
    sigma, sigma_next = holder.sigmas[step], holder.sigmas[step + 1]
    options = model_options.setdefault("transformer_options", {})
    options[EXTRA_PASS] = True
    try:
        if sigma_next > 0:
            ahead = _euler(x, denoised, sigma, sigma_next)
            corrected = executor(ahead, torch.full_like(timestep, sigma_next), model_options, seed)
            slope = ((x - denoised) / sigma + (ahead - corrected.float()) / sigma_next) / 2
            landed = x + (sigma_next - sigma) * slope
        else:
            middle = sigma / 2
            halfway = _euler(x, denoised, sigma, middle)
            landed = executor(halfway, torch.full_like(timestep, middle), model_options, seed).float()
    finally:
        options.pop(EXTRA_PASS, None)
    return _landing_prediction(x, landed, sigma, sigma_next)


def _predict_wrapper(executor, x, timestep, model_options=None, seed=None):
    options = (model_options or {}).get("transformer_options") or {}
    holder = options.get(KEY)
    if holder is None or not holder.sigmas or not torch.is_tensor(timestep):
        return executor(x, timestep, model_options, seed)
    sigma = float(timestep.flatten()[0])
    step = holder.step_of(sigma)
    if step is None or step in holder.evaluated:
        if sigma > 0:
            holder.between = True
        return executor(x, timestep, model_options, seed)
    holder.evaluated.add(step)
    factor = holder.push(step)
    if factor < 1.0:
        holder.pushed.append(step)
        shared = model_options.setdefault("transformer_options", {})
        shared[SCHEDULE_SIGMA] = sigma
        try:
            denoised = executor(x, timestep * factor, model_options, seed)
        finally:
            shared.pop(SCHEDULE_SIGMA, None)
    else:
        denoised = executor(x, timestep, model_options, seed)
    current = holder.sigmas[step]
    if not torch.is_tensor(denoised) or current <= 0 or holder.sigmas[step + 1] == current:
        return denoised
    holder.observe(step, (x.float() - denoised.float()) / current)
    if not holder.choose(step):
        return denoised
    holder.refined.append(step)
    refined = _second_order(executor, holder, x.float(), timestep, model_options, seed,
                            denoised.float(), step)
    return refined.to(denoised.dtype)


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
    logger.info(
        "Adaptive Detail: %d of %d steps pushed toward detail, %d refined with an extra pass, %.1fs",
        report["pushed"], report["total"], report["refined"], report["seconds"],
    )
    if holder.node_id is None:
        return
    try:
        from ..interface import run_result

        first_last = report["window"]
        facts = {
            "steps": report["pattern"] or "-",
            "key": "d = pushed, + = extra pass, * = both, . = plain",
            "detail": (f"{holder.settings.detail:.2f} on steps {first_last[0] + 1} to {first_last[1] + 1}"
                       if first_last and report["pushed"] else "off"),
            "extra": f"{report['refined']} of {report['budget']} passes",
            "sampled in": f"{report['seconds']:.1f} s",
        }
        summary = (f"{report['pushed']} of {report['total']} steps pushed toward detail, "
                   f"{report['refined']} refined with an extra pass")
        warn = report["between"] and not report["pushed"] and not report["refined"]
        if warn:
            summary = "nothing changed: the sampler calls the model more than once per step"
            facts["fix"] = "use euler, ddim, uni_pc or a multistep sampler"
        run_result.publish(
            status=run_result.WARNING if warn else run_result.OK,
            summary=summary,
            counts={"pushed": report["pushed"], "refined": report["refined"], "steps": report["total"]},
            facts=facts,
            node_id=holder.node_id,
        )
    except Exception as error:
        logger.debug("the Adaptive Detail report was not published (%s)", error)


def apply(model, settings: Settings, node_id=None):
    """A copy of a model whose sampling is pushed toward detail where detail forms.

    Args:
        model: The ``ModelPatcher`` to patch.
        settings: The push, the window and the extra passes.
        node_id: The node to file each run's report under, or None.

    Returns:
        The patched copy.
    """
    import comfy.patcher_extension

    patched = model.clone()
    patched.model_options.setdefault("transformer_options", {})[KEY] = _Holder(settings, node_id)
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, KEY, _sample_wrapper)
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.PREDICT_NOISE, KEY, _predict_wrapper)
    return patched

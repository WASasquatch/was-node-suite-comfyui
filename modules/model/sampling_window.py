"""The steps of a sampling run that fall between a start and an end sampling percent."""

from __future__ import annotations

from .. import log

logger = log.get_logger("modules.model.sampling_window")

__all__ = ["WINDOW_SLACK", "steps_inside"]

#: Sampling percent a step may fall short of a window bound by and still count as on it.
WINDOW_SLACK = 0.005


def steps_inside(total: int, start_percent: float, end_percent: float, sigmas=None,
                 model_sampling=None) -> list[bool]:
    """Which steps of a run lie between ``start_percent`` and ``end_percent``.

    Args:
        total: Steps in the run.
        start_percent: Sampling percent where the window opens.
        end_percent: Sampling percent where the window closes.
        sigmas: The run's noise levels, one more than ``total``.
        model_sampling: The model's sampling object, which turns a percent into a noise level;
            without it, or without ``sigmas``, a percent is a share of the steps.

    Returns:
        One flag per step, True where the step lies inside the window.
    """
    total = max(0, int(total))
    start, end = float(start_percent), float(end_percent)
    if model_sampling is not None and sigmas is not None and len(sigmas) > total:
        try:
            high = float(model_sampling.percent_to_sigma(start - WINDOW_SLACK))
            low = float(model_sampling.percent_to_sigma(end - WINDOW_SLACK))
            return [low < float(sigmas[step]) <= high for step in range(total)]
        except Exception as error:
            logger.debug("the sampling window falls back to step shares (%s)", error)
    return [start <= step / total < end for step in range(total)] if total else []

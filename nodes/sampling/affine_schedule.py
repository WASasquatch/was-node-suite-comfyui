"""The curve that decides how strong an affine is at each step of a sampler run."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ...modules.compat.types import DICT
from ...modules.interface import preview, run_result
from ...modules.latent import affine
from ...modules.util.easing import EASING_NAMES

#: Size of the plot the node draws of its own curve, in pixels.
PLOT_WIDTH = 480
PLOT_HEIGHT = 200

#: Colours the plot is drawn in, as 0.0 to 1.0 RGB.
PLOT_BACKGROUND = (0.11, 0.11, 0.13)
PLOT_GRID = (0.22, 0.22, 0.26)
PLOT_LINE = (0.40, 0.78, 1.00)

#: How many grid lines the plot draws each way, not counting the border.
PLOT_DIVISIONS = 4


def plot(values: list[float]) -> torch.Tensor:
    """Draw a strength curve as an image.

    Args:
        values: One strength per step, each 0.0 to 1.0.

    Returns:
        A ``[1, PLOT_HEIGHT, PLOT_WIDTH, 3]`` image tensor.
    """
    from PIL import Image, ImageDraw

    canvas = Image.new("RGB", (PLOT_WIDTH, PLOT_HEIGHT), _rgb(PLOT_BACKGROUND))
    pen = ImageDraw.Draw(canvas)

    for n in range(1, PLOT_DIVISIONS):
        x = round(n * (PLOT_WIDTH - 1) / PLOT_DIVISIONS)
        y = round(n * (PLOT_HEIGHT - 1) / PLOT_DIVISIONS)
        pen.line([(x, 0), (x, PLOT_HEIGHT - 1)], fill=_rgb(PLOT_GRID))
        pen.line([(0, y), (PLOT_WIDTH - 1, y)], fill=_rgb(PLOT_GRID))
    pen.rectangle([0, 0, PLOT_WIDTH - 1, PLOT_HEIGHT - 1], outline=_rgb(PLOT_GRID))

    if len(values) > 1:
        span = len(values) - 1
        points = [
            (
                round(i * (PLOT_WIDTH - 1) / span),
                round((1.0 - min(max(v, 0.0), 1.0)) * (PLOT_HEIGHT - 1)),
            )
            for i, v in enumerate(values)
        ]
        pen.line(points, fill=_rgb(PLOT_LINE), width=2, joint="curve")

    pixels = torch.frombuffer(bytearray(canvas.tobytes()), dtype=torch.uint8)
    pixels = pixels.view(PLOT_HEIGHT, PLOT_WIDTH, 3).float() / 255.0
    return pixels.unsqueeze(0)


def _rgb(colour: tuple[float, float, float]) -> tuple[int, int, int]:
    """Turn a 0.0 to 1.0 colour into the 0 to 255 one a drawing takes.

    Args:
        colour: Three channels on a 0.0 to 1.0 scale.

    Returns:
        The same colour as three whole numbers.
    """
    return tuple(int(round(channel * 255)) for channel in colour)


class AffineSchedule(io.ComfyNode):
    """Shape the per-step strength curve the Affine samplers follow."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAffineSchedule",
            display_name="Affine Schedule",
            search_aliases=[
                "WASAffineSchedule",
                "Affine Schedule",
                "affine curve",
                "step schedule",
                "sampling ramp",
            ],
            category="WAS Suite/Sampling",
            description=(
                "Decide how strong an affine is at each step of a sampler run: where it "
                "starts, where it stops, where it peaks and how it eases in and out. The "
                "curve is drawn on the node so the shape can be read before the run. Wire "
                "it into the affine_schedule socket of any Affine sampler."
            ),
            inputs=[
                io.Float.Input(
                    "start",
                    default=0.2,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How far into the run the affine begins, as a share of the steps. "
                        "0.0 = from the first step, 0.2 = a fifth of the way in, once the "
                        "composition has settled."
                    ),
                ),
                io.Float.Input(
                    "end",
                    default=0.8,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Where it stops, on the same scale. 0.8 leaves the last fifth "
                        "untouched, which lets the sampler resolve the detail cleanly; 1.0 "
                        "carries it to the final step."
                    ),
                ),
                io.Float.Input(
                    "bias",
                    default=0.5,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Where the peak sits between start and end. 0.5 = the middle, 0.1 = "
                        "hits hard early then fades, 0.9 = builds slowly to the end."
                    ),
                ),
                io.Float.Input(
                    "exponent",
                    default=1.0,
                    min=0.0,
                    max=10.0,
                    step=0.05,
                    tooltip=(
                        "Bends the whole curve towards zero. 1.0 = as the easing draws it, "
                        "2.0 = a narrower peak, 0.5 = a broad plateau. 0.0 pins every active "
                        "step at full strength."
                    ),
                ),
                io.Float.Input(
                    "start_offset",
                    default=0.0,
                    min=-1.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Strength held before the curve begins. 0.0 = nothing until start, "
                        "0.3 = a third of the affine from the very first step."
                    ),
                ),
                io.Float.Input(
                    "end_offset",
                    default=0.0,
                    min=-1.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Strength held after the curve ends. 0.0 = nothing after end, 0.3 = a "
                        "third of the affine carried to the last step."
                    ),
                ),
                io.Combo.Input(
                    "curve",
                    options=list(EASING_NAMES),
                    default="ease_in_out_sine",
                    tooltip=(
                        "How the strength travels between nothing and the peak. "
                        "'ease_in_out_sine' is a smooth swell; 'linear' is a plain ramp; the "
                        "'expo' and 'quint' curves stay near zero then rush; 'back' and "
                        "'elastic' overshoot, which pushes past the scale that was asked for."
                    ),
                ),
                io.Int.Input(
                    "preview_steps",
                    default=20,
                    min=2,
                    max=1000,
                    optional=True,
                    tooltip=(
                        "How many steps the drawn curve is sampled at. 20 matches a default "
                        "run; set it to the step count of the sampler this feeds and the "
                        "plot is exactly what that run will do. The schedule itself is "
                        "unchanged either way."
                    ),
                ),
            ],
            outputs=[
                DICT.Output(
                    display_name="affine_schedule",
                    tooltip="The curve, for the affine_schedule socket of any Affine sampler.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls, start, end, bias, exponent, start_offset, end_offset, curve, preview_steps=20
    ) -> io.NodeOutput:
        schedule = {
            "start": float(start),
            "end": float(end),
            "bias": float(bias),
            "exponent": float(exponent),
            "start_offset": float(start_offset),
            "end_offset": float(end_offset),
            "curve": str(curve),
        }
        _report(schedule, int(preview_steps))
        return io.NodeOutput(schedule)


def _report(schedule: dict, steps: int) -> None:
    """Publish the curve and its figures, for the node's own panel.

    Args:
        schedule: The schedule the node built.
        steps: How many steps to sample it at.
    """
    try:
        values = affine.step_schedule(steps, schedule)
        preview.publish_output(plot(values))
        if not run_result.watching() or not values:
            return
        active = [i for i, v in enumerate(values) if v > 1e-8]
        peak = max(range(len(values)), key=values.__getitem__)
        run_result.publish(
            status=run_result.OK if active else run_result.WARNING,
            summary=(
                f"{len(active)} of {steps} steps carry the affine, peaking at step {peak}"
                if active
                else f"0 of {steps} steps carry the affine"
            ),
            counts={
                "active steps": len(active),
                "first step": active[0] if active else 0,
                "peak step": peak,
                "last step": active[-1] if active else 0,
                "peak strength": round(values[peak], 3),
            },
            facts={"curve": str(schedule["curve"]), "sampled at": f"{steps} steps"},
        )
    except Exception:
        run_result.publish(status=run_result.WARNING, summary="the schedule plot could not be drawn")

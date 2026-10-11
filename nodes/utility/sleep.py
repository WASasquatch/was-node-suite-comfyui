"""A timed pause in a graph, handing whatever is wired through it back unchanged.

Times are in seconds.
"""

from __future__ import annotations

import math
import time

from comfy_api.latest import io

#: Longest single pause taken between checks for a cancelled run.
TICK = 0.05

#: Longest wait the node offers.
MAX_SECONDS = 3600.0


def wait(seconds: float) -> float:
    """Pause for a time, stopping the moment the run is cancelled.

    Args:
        seconds: How long to pause for, clamped to ``0`` to :data:`MAX_SECONDS`.

    Returns:
        Seconds actually spent paused.

    Raises:
        InterruptProcessingException: The run was cancelled during the pause.
    """
    import comfy.model_management
    import comfy.utils

    wanted = max(0.0, min(float(seconds), MAX_SECONDS))
    steps = max(1, math.ceil(wanted / TICK))
    progress = comfy.utils.ProgressBar(steps)
    started = time.monotonic()
    while True:
        comfy.model_management.throw_exception_if_processing_interrupted()
        elapsed = time.monotonic() - started
        left = wanted - elapsed
        if left <= 0.0:
            break
        progress.update_absolute(int(elapsed / TICK))
        time.sleep(min(TICK, left))
    progress.update_absolute(steps)
    return time.monotonic() - started


class Sleep(io.ComfyNode):
    """Hold a run still for a set time, then pass its input on unchanged."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        template = io.MatchType.Template("sleep_passthrough")
        return io.Schema(
            node_id="WASSleep",
            display_name="Sleep",
            search_aliases=[
                "WASSleep",
                "Sleep",
                "wait",
                "delay",
                "pause",
                "throttle",
                "rate limit",
            ],
            category="WAS Suite/Utilities",
            description=(
                "Wait a set number of seconds, then hand whatever is wired in straight back "
                "out, to pace a rate-limited web service, a folder still being written or a "
                "busy loop. It runs on every queue, so everything below it runs again too; "
                "Cancel stops the wait within a twentieth of a second."
            ),
            inputs=[
                io.MatchType.Input(
                    "passthrough",
                    template=template,
                    optional=True,
                    tooltip=(
                        "Anything: an image, a model, text, a number. It comes back out "
                        "unchanged once the wait is over, putting the delay in the middle of "
                        "a chain. Leave it unwired to wait on its own."
                    ),
                ),
                io.Float.Input(
                    "seconds",
                    default=1.0,
                    min=0.0,
                    max=MAX_SECONDS,
                    step=0.1,
                    tooltip=(
                        "How long to wait. 0 = no wait; 0.5 = half a second; 60 = a minute; "
                        "3600 = an hour, the most on offer. Match it to the limit being "
                        "respected, such as 1.2 for a service allowing 50 calls a minute."
                    ),
                ),
            ],
            outputs=[
                io.MatchType.Output(
                    template=template,
                    display_name="passthrough",
                    tooltip=(
                        "The value that came in, unchanged, on a socket carrying its type. "
                        "Nothing wired to it starts until the wait is over. Empty when "
                        "nothing was wired into passthrough."
                    ),
                ),
                io.Float.Output(
                    display_name="slept",
                    tooltip=(
                        "Seconds actually spent waiting, measured rather than repeated back, "
                        "so 1.0 comes out as 1.001 or so. Wire it to Text Concatenate or "
                        "Number Operation to record how long a run was paced for."
                    ),
                ),
            ],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs) -> float:
        """NaN never equals itself, so the wait is taken again on every prompt."""
        return float("NaN")

    @classmethod
    def execute(cls, passthrough=None, seconds=1.0) -> io.NodeOutput:
        """Wait, then answer what arrived and how long the wait ran for.

        Args:
            passthrough: Any value, handed back unchanged.
            seconds: How long to wait for.

        Returns:
            The value that came in, and the seconds spent waiting.

        Raises:
            InterruptProcessingException: The run was cancelled during the wait.
        """
        return io.NodeOutput(passthrough, wait(seconds))

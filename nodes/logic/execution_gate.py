"""A gate that decides whether a branch of the graph runs at all."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules import log

logger = log.get_logger("nodes.logic.execution_gate")

#: Stands in for a socket nothing is wired to, which None cannot say on a lazy input.
UNWIRED = object()


class ExecutionGate(io.ComfyNode):
    """Pass a value on while a switch is on, and end the graph there while it is off."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        match = io.MatchType.Template("was_execution_gate")
        return io.Schema(
            node_id="WASExecutionGate",
            display_name="Execution Gate",
            search_aliases=[
                "WASExecutionGate",
                "Execution Gate",
                "gate",
                "conditional execution",
                "block execution",
                "skip branch",
                "stop branch",
                "bypass",
            ],
            category="WAS Suite/Logic",
            description=(
                "Pass a value of any type on only while a switch is on. Switched off, the "
                "branch feeding the gate and every node after it are skipped while the rest "
                "of the graph finishes; bypass_downstream also draws those nodes bypassed."
            ),
            inputs=[
                io.Boolean.Input(
                    "open",
                    default=True,
                    tooltip=(
                        "Whether the branch runs. `true` passes value on. `false` skips "
                        "everything wired into value and every node after the gate."
                    ),
                ),
                io.MatchType.Input(
                    "value",
                    template=match,
                    lazy=True,
                    optional=True,
                    tooltip=(
                        "What to pass on, of any type. The first connection fixes the "
                        "type. Nothing wired here is evaluated while the gate is closed."
                    ),
                ),
                io.Boolean.Input(
                    "bypass_downstream",
                    default=False,
                    optional=True,
                    tooltip=(
                        "`true` = every node after the gate is drawn bypassed while open is "
                        "off; `false` = they stay as they are. Read only when open is set on "
                        "the gate: with open wired, the run skips them."
                    ),
                ),
                io.String.Input(
                    "closed_message",
                    default="",
                    optional=True,
                    tooltip=(
                        "A line written to the console while the gate is closed, such as "
                        "`no face found, nothing to upscale`. Empty writes nothing. The "
                        "branch is skipped either way and the run finishes."
                    ),
                ),
            ],
            outputs=[
                io.MatchType.Output(
                    template=match,
                    display_name="output",
                    tooltip=(
                        "The value, while the gate is open. Nothing downstream of it runs "
                        "while the gate is closed."
                    ),
                ),
            ],
        )

    @classmethod
    def check_lazy_status(
        cls, open=True, value=UNWIRED, bypass_downstream=False, closed_message=""
    ) -> list[str]:
        """Which inputs to evaluate before the body runs.

        Args:
            open: Whether the branch runs.
            value: The wired value, None while it is wired but not yet evaluated.
            bypass_downstream: Whether the canvas draws the nodes after the gate as bypassed.
            closed_message: What a closed gate reports.

        Returns:
            ``["value"]`` where the gate is open and the value is still to be worked out,
            an empty list otherwise.
        """
        if open and value is None:
            return ["value"]
        return []

    @classmethod
    def execute(
        cls, open=True, value=UNWIRED, bypass_downstream=False, closed_message=""
    ) -> io.NodeOutput:
        """Pass the value on, or stop the graph.

        Args:
            open: Whether the branch runs.
            value: The wired value.
            bypass_downstream: Whether the canvas draws the nodes after the gate as bypassed.
            closed_message: What a closed gate reports.

        Returns:
            The value while open, otherwise an output that skips every node after it.
        """
        if not open:
            from comfy_execution.graph_utils import ExecutionBlocker

            message = str(closed_message or "").strip()
            if message:
                logger.info("Execution Gate closed: %s", message)
            # A blocker without a message skips its readers instead of failing the run.
            return io.NodeOutput(ExecutionBlocker(None))
        return io.NodeOutput(None if value is UNWIRED else value)

"""Let a value through only when a condition holds, stopping the rest of the branch."""

from __future__ import annotations

from comfy_api.latest import io

from ....modules import log

logger = log.get_logger("nodes.logic.condition.any_gate")


class AnyGate(io.ComfyNode):
    """Pass a value on when a condition holds, and block the branch when it does not."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASAnyGate",
            display_name="Any Gate",
            search_aliases=[
                "WASAnyGate",
                "Any Gate",
                "gate",
                "block",
                "conditional save",
                "stop branch",
            ],
            category="WAS Suite/Logic/Boolean",
            description=(
                "Pass a value on only when a condition holds. When it does not, everything "
                "downstream is skipped, which is the one way to stop a save or a preview "
                "from running. A switch chooses between two branches; this one stops a "
                "branch outright."
            ),
            inputs=[
                io.MatchType.Input(
                    "value",
                    template=io.MatchType.Template("any_gate"),
                    lazy=True,
                    tooltip=(
                        "What to pass on. It is only worked out when the gate opens, so a "
                        "closed gate also skips the work behind it."
                    ),
                ),
                io.Boolean.Input(
                    "open",
                    default=True,
                    tooltip=(
                        "true lets the value through; false stops every node downstream. "
                        "Wire it from Compare, Boolean Reduce or any test."
                    ),
                ),
                io.String.Input(
                    "message",
                    default="",
                    multiline=False,
                    optional=True,
                    tooltip=(
                        "A line written to the console while the gate is closed, such as "
                        "`no face found`. Empty writes nothing. The branch is skipped either "
                        "way and the run finishes."
                    ),
                ),
            ],
            outputs=[
                io.MatchType.Output(
                    template=io.MatchType.Template("any_gate"),
                    display_name="value",
                    tooltip="The value when the gate is open. Nothing runs downstream when it is not.",
                ),
            ],
        )

    @classmethod
    def check_lazy_status(cls, open=True, value=None, message="") -> list[str]:
        """Ask for the value only when the gate is open.

        Args:
            open: Whether the gate lets the value through.
            value: The value, present once it has been evaluated.
            message: Written to the console while the gate is closed.

        Returns:
            ``["value"]`` while the gate is open and the value has not arrived.
        """
        if open and value is None:
            return ["value"]
        return []

    @classmethod
    def execute(cls, open=True, value=None, message="") -> io.NodeOutput:
        """Pass the value on, or block the branch.

        Args:
            open: Whether the gate lets the value through.
            value: What to pass on.
            message: Written to the console while the gate is closed.

        Returns:
            The value, or an ``ExecutionBlocker`` that skips everything downstream.
        """
        from comfy_execution.graph_utils import ExecutionBlocker

        if open:
            return io.NodeOutput(value)
        note = str(message or "").strip()
        if note:
            logger.info("Any Gate closed: %s", note)
        # A blocker without a message skips its readers instead of failing the run.
        return io.NodeOutput(ExecutionBlocker(None))

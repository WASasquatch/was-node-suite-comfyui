"""List the graph's execution gates with a switch on each."""

from __future__ import annotations

from comfy_api.latest import io


class ExecutionGateControlboard(io.ComfyNode):
    """A panel of the graph's execution gates, each switching that gate open or closed."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASExecutionGateControlboard",
            display_name="Execution Gate Controlboard",
            search_aliases=[
                "WASExecutionGateControlboard",
                "Execution Gate Controlboard",
                "Execution Gateway Controlboard",
                "gate controlboard",
                "gate switches",
                "control panel",
                "toggle gates",
                "enable disable branches",
            ],
            category="WAS Suite/Logic",
            description=(
                "List every Execution Gate and Any Gate in the graph, subgraphs included, "
                "each with a switch that opens or closes it. One place to turn whole branches "
                "of a workflow on and off. The switches are the gates' own open widgets, so "
                "they survive a save, an undo and a copy. The node reads nothing and answers "
                "nothing."
            ),
            inputs=[],
            outputs=[],
        )

    @classmethod
    def execute(cls) -> io.NodeOutput:
        """Answer nothing.

        Returns:
            An empty result.
        """
        return io.NodeOutput()

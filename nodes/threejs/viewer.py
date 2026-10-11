"""Render a Three.js scene on the node."""

from __future__ import annotations

import json

from comfy_api.latest import io

from ...modules.compat.types import THREE_APP
from ...modules.threejs.spec import refuse_script, require_spec

REQUIRES = "threejs"


class ThreeViewer(io.ComfyNode):
    """Send an app descriptor to the browser and draw it."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeViewer",
            display_name="Three Viewer",
            search_aliases=[
                "WASThreeViewer",
                "Three Viewer",
                "viewer",
                "webgl",
                "preview 3d",
            ],
            category="WAS Suite/Three",
            description=(
                "Draw the scene live on this node, in the browser, with no image output. Drag to "
                "orbit, wheel to zoom, middle-drag or hold shift to pan while Three App's orbit "
                "control is on; Pause stops the animation and Reset Camera returns to the camera "
                "node's values."
            ),
            is_output_node=True,
            inputs=[
                THREE_APP.Input(
                    "app",
                    tooltip="The scene, camera and renderer settings, from Three App.",
                ),
            ],
            outputs=[],
        )

    @classmethod
    def execute(cls, app) -> io.NodeOutput:
        """Hand the descriptor to the browser.

        Raises:
            ValueError: ``app`` is not an app descriptor.
        """
        require_spec(app, "app")
        refuse_script(app, "Three Viewer")
        payload = {"app": app}
        return io.NodeOutput(ui={"three_app": [json.dumps(payload, separators=(",", ":"))]})

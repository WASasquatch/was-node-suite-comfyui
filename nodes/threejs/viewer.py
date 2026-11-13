"""Render a Three.js scene on the node."""

from __future__ import annotations

import json

from comfy_api.latest import io

from ...modules.compat.types import THREE_APP, THREE_CAMERA
from ...modules.model import network_enabled
from ...modules.threejs import viewstate
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
                "viewport",
                "frame the shot",
            ],
            category="WAS Suite/Three",
            description=(
                "A live viewport for the scene: orbit, frame a shot, and pick and move objects "
                "with a gizmo. With override on, that view and those moves replace the "
                "upstream camera and placement, on screen and in the app handed on to Three "
                "Render, so the render matches what was framed here."
            ),
            is_output_node=True,
            inputs=[
                THREE_APP.Input(
                    "app",
                    tooltip="The scene, camera and renderer settings, from Three App.",
                ),
                io.Boolean.Input(
                    "override",
                    default=False,
                    tooltip=(
                        "`false` = the camera and objects follow the nodes upstream; `true` = the "
                        "view framed here and the objects moved here are used instead, and saved "
                        "with the workflow."
                    ),
                ),
                io.String.Input(
                    "view",
                    default="",
                    multiline=False,
                    optional=True,
                    tooltip=(
                        "The framed camera and moved objects as JSON, written by the viewer, as "
                        "`{\"camera\": {\"position\": [0, 1.5, 6], \"target\": [0, 0.8, 0], "
                        "\"fov\": 32}}`. Reset view clears it."
                    ),
                ),
            ],
            outputs=[
                THREE_APP.Output(
                    display_name="app",
                    tooltip=(
                        "The app with this viewer's view in place where override is on, for Three "
                        "Render, Three Path Trace Render or Three Compile."
                    ),
                ),
                THREE_CAMERA.Output(
                    display_name="camera",
                    tooltip="The camera the app above draws from, for another Three App.",
                ),
            ],
        )

    @classmethod
    def execute(cls, app, override=False, view="") -> io.NodeOutput:
        """Hand the descriptor to the browser and answer it with the saved view applied.

        Raises:
            ValueError: ``app`` is not an app descriptor, or the saved view cannot be read.
        """
        require_spec(app, "app")
        refuse_script(app, "Three Viewer")
        try:
            chosen = viewstate.parse(view)
        except ValueError as error:
            raise ValueError(
                f"Three Viewer could not read its saved view: {error}. Press Reset in the viewer "
                f"to clear it."
            ) from None
        moved = bool(override) and bool(chosen["camera"] or chosen["objects"])
        handed = viewstate.apply(app, chosen) if moved else app
        payload = {
            "app": app,
            "network": network_enabled(),
            "view": chosen,
            "override": bool(override),
        }
        return io.NodeOutput(
            handed,
            handed["deps"]["camera"],
            ui={"three_app": [json.dumps(payload, separators=(",", ":"))]},
        )

"""Path trace a Three.js scene on the node, a sample at a time."""

from __future__ import annotations

import json

from comfy_api.latest import io

from ...modules.compat.types import THREE_APP
from ...modules.threejs.spec import refuse_script, require_spec

REQUIRES = "threejs"


class ThreePathTraceViewer(io.ComfyNode):
    """Send an app descriptor to the browser and trace it on the node."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreePathTraceViewer",
            display_name="Three Path Trace Viewer",
            search_aliases=[
                "WASThreePathTraceViewer",
                "Three Path Trace Viewer",
                "path tracing preview",
                "ray traced viewer",
                "interactive path tracer",
                "progressive render",
            ],
            category="WAS Suite/Three",
            description=(
                "Trace the scene on this node and keep adding samples to the same picture, so it "
                "starts noisy and cleans up in place while nothing is touched. Moving the camera "
                "starts it over, which is how the framing for a Three Path Trace Render is found "
                "without waiting on a render each time. The animation is held still to begin "
                "with, since a moving scene starts every frame over and never settles; Play runs "
                "it anyway and the picture stays grainy. Drag inside to orbit, wheel to zoom, "
                "middle-drag or hold shift to pan. Nothing is rendered on the server and no "
                "image comes out of this node."
            ),
            is_output_node=True,
            inputs=[
                THREE_APP.Input(
                    "app",
                    tooltip="The scene, camera and renderer settings, from Three App.",
                ),
                io.Int.Input(
                    "max_samples",
                    default=512,
                    min=1,
                    max=100000,
                    tooltip=(
                        "Samples per pixel to stop at, so a settled picture stops drawing on the "
                        "GPU. 512 is clean for most scenes, 4096 for caustics."
                    ),
                ),
                io.Int.Input(
                    "bounces",
                    default=5,
                    min=1,
                    max=30,
                    tooltip=(
                        "How many surfaces one path may hit. 1 is direct light only, 5 suits "
                        "most scenes, 12 for a room lit through a doorway."
                    ),
                ),
                io.Int.Input(
                    "transmissive_bounces",
                    default=10,
                    min=0,
                    max=60,
                    tooltip=(
                        "Extra bounces allowed inside glass, on top of `bounces`. 10 carries a "
                        "path through a few panes; 0 turns glass black inside."
                    ),
                ),
                io.Float.Input(
                    "filter_glossy",
                    default=0.05,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Roughens sharp reflections to settle the bright speckle they leave. "
                        "0.0 is exact, 0.05 clears most speckle, 0.5 visibly blurs highlights."
                    ),
                ),
                io.Int.Input(
                    "tiles",
                    default=3,
                    min=1,
                    max=16,
                    tooltip=(
                        "Splits each pass into this many tiles across and down, so the node "
                        "stays draggable while it traces. 3 gives nine tiles."
                    ),
                ),
                io.Int.Input(
                    "texture_size",
                    default=1024,
                    min=16,
                    max=8192,
                    tooltip=(
                        "Size every texture in the scene is fitted to for tracing. 1024 suits "
                        "most work; 2048 keeps fine detail in a close-up, at more memory."
                    ),
                ),
            ],
            outputs=[],
        )

    @classmethod
    def execute(
        cls, app, max_samples, bounces, transmissive_bounces, filter_glossy, tiles,
        texture_size,
    ) -> io.NodeOutput:
        """Hand the descriptor and the tracer settings to the browser.

        Raises:
            ValueError: ``app`` is not an app descriptor.
        """
        require_spec(app, "app")
        refuse_script(app, "Three Path Trace Viewer")
        payload = {
            "app": app,
            "trace": {
                "samples": int(max_samples),
                "bounces": int(bounces),
                "transmissiveBounces": int(transmissive_bounces),
                "filterGlossy": float(filter_glossy),
                "tiles": int(tiles),
                "textureSize": int(texture_size),
            },
        }
        return io.NodeOutput(ui={"three_app": [json.dumps(payload, separators=(",", ":"))]})

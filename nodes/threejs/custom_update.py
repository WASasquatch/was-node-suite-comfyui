"""Hand-written JavaScript run on an object every frame."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_OBJECT
from ...modules.threejs.spec import create_spec, require_spec

REQUIRES = "threejs"

DEFAULT_BODY = "object.rotation.y += delta * 0.5;"


class ThreeCustomUpdate(io.ComfyNode):
    """Run a JavaScript body on an object each frame."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeCustomUpdate",
            display_name="Three Custom Update",
            search_aliases=[
                "WASThreeCustomUpdate",
                "Three Custom Update",
                "custom update",
                "javascript",
                "per frame",
            ],
            category="WAS Suite/Three",
            description=(
                "Run a short JavaScript body on an object every drawn frame, for motion Three "
                "Animate Transform cannot express. The code runs in the browser with the reach "
                "of any extension, so only load a workflow carrying it from a trusted source."
            ),
            inputs=[
                THREE_OBJECT.Input(
                    "object",
                    tooltip="The object the code moves. It is wrapped rather than changed.",
                ),
                io.String.Input(
                    "javascript",
                    default=DEFAULT_BODY,
                    multiline=True,
                    tooltip=(
                        "A body run each frame, as `object.position.y = Math.sin(time) * 0.5;`. "
                        "In scope: `object`; `time`, seconds since start; `delta`, seconds "
                        "since the last frame; `THREE`; `ctx`, kept between frames. Nothing is "
                        "returned."
                    ),
                ),
            ],
            outputs=[
                THREE_OBJECT.Output(
                    display_name="animated",
                    tooltip="The moving object, for Three Group or the root socket on Three Scene.",
                ),
            ],
        )

    @classmethod
    def execute(cls, object, javascript) -> io.NodeOutput:
        """Carry the code to the browser.

        Raises:
            ValueError: The input is not an object descriptor.
        """
        require_spec(object, "object")
        return io.NodeOutput(
            create_spec(
                "object",
                "CustomUpdateGroup",
                params={"javascript": javascript},
                children=[object],
            )
        )

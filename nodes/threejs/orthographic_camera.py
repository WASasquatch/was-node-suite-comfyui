"""A view with no perspective convergence."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_CAMERA, THREE_TRACK
from ...modules.threejs.spec import compact_deps, create_spec

REQUIRES = "threejs"


class ThreeOrthographicCamera(io.ComfyNode):
    """Build an orthographic camera descriptor."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeOrthographicCamera",
            display_name="Three Orthographic Camera",
            search_aliases=[
                "WASThreeOrthographicCamera",
                "Three Orthographic Camera",
                "orthographic",
                "isometric",
                "elevation",
            ],
            category="WAS Suite/Three",
            description=(
                "A camera with no perspective, so parallel lines stay parallel and an object "
                "is the same size at any distance, as in an elevation drawing or an isometric "
                "game. Framing is set with view height rather than by moving closer."
            ),
            inputs=[
                io.Float.Input(
                    "view_height",
                    default=6.0,
                    min=0.0001,
                    max=1000000.0,
                    step=0.01,
                    tooltip=(
                        "How many scene units fit top to bottom; the width follows the "
                        "viewer's shape. 6.0 frames a unit cube with room around it."
                    ),
                ),
                io.Float.Input(
                    "near",
                    default=0.1,
                    min=-1000000.0,
                    max=1000000.0,
                    step=0.001,
                    tooltip=(
                        "Nearest distance drawn. 0.1 suits most scenes, and unlike a perspective "
                        "camera this may be negative."
                    ),
                ),
                io.Float.Input(
                    "far",
                    default=1000.0,
                    min=0.0001,
                    max=1000000000.0,
                    step=1.0,
                    tooltip="Furthest distance drawn. 1000.0 covers most scenes; anything beyond is clipped.",
                ),
                io.Float.Input(
                    "position_x",
                    default=3.0,
                    min=-1000000.0,
                    max=1000000.0,
                    step=0.01,
                    tooltip="Where the camera sits along X. With 3.0, 2.0 and 5.0 the view reads as isometric.",
                ),
                io.Float.Input(
                    "position_y",
                    default=2.0,
                    min=-1000000.0,
                    max=1000000.0,
                    step=0.01,
                    tooltip="Where the camera sits along Y. 2.0 looks slightly down on the target.",
                ),
                io.Float.Input(
                    "position_z",
                    default=5.0,
                    min=-1000000.0,
                    max=1000000.0,
                    step=0.01,
                    tooltip=(
                        "Where the camera sits along Z. 5.0 is the default; only the direction "
                        "matters here, not the distance."
                    ),
                ),
                io.Float.Input(
                    "target_x",
                    default=0.0,
                    min=-1000000.0,
                    max=1000000.0,
                    step=0.01,
                    tooltip="X of the point the camera looks at, and orbits around. 0.0 is the world origin.",
                ),
                io.Float.Input(
                    "target_y",
                    default=0.0,
                    min=-1000000.0,
                    max=1000000.0,
                    step=0.01,
                    tooltip="Y of the point the camera looks at. 0.0 is the origin, 1.6 about head height.",
                ),
                io.Float.Input(
                    "target_z",
                    default=0.0,
                    min=-1000000.0,
                    max=1000000.0,
                    step=0.01,
                    tooltip="Z of the point the camera looks at, and orbits around. 0.0 is the world origin.",
                ),
                THREE_TRACK.Input(
                    "track",
                    optional=True,
                    tooltip=(
                        "Aim or follow an object instead of a fixed point, from Three Track. "
                        "Wired, it overrides the target below."
                    ),
                ),
            ],
            outputs=[
                THREE_CAMERA.Output(
                    display_name="camera",
                    tooltip="The camera, for the camera socket on Three App.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        view_height,
        near,
        far,
        position_x,
        position_y,
        position_z,
        target_x,
        target_y,
        target_z,
        track=None,
    ) -> io.NodeOutput:
        """Describe the camera.

        Raises:
            ValueError: ``far`` is not beyond ``near``, which would leave nothing drawable.
        """
        if far <= near:
            raise ValueError(
                f"Three Orthographic Camera was given far {far} and near {near}. far has to be "
                f"the greater of the two, since it is the back of the range that gets drawn. "
                f"Try near 0.1 and far 1000."
            )
        return io.NodeOutput(
            create_spec(
                "camera",
                "OrthographicCamera",
                params={
                    "viewHeight": float(view_height),
                    "near": float(near),
                    "far": float(far),
                    "position": [float(position_x), float(position_y), float(position_z)],
                    "target": [float(target_x), float(target_y), float(target_z)],
                },
                deps=compact_deps(track=track),
            )
        )

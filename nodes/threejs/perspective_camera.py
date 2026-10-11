"""The view a Three.js scene is rendered from."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_CAMERA, THREE_TRACK
from ...modules.threejs.spec import compact_deps, create_spec

REQUIRES = "threejs"


class ThreePerspectiveCamera(io.ComfyNode):
    """Build a perspective camera descriptor."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreePerspectiveCamera",
            display_name="Three Perspective Camera",
            search_aliases=[
                "WASThreePerspectiveCamera",
                "Three Perspective Camera",
                "camera",
                "perspective",
                "fov",
            ],
            category="WAS Suite/Three",
            description=(
                "A camera with perspective, so distant things are smaller. Position places it "
                "and target is the point it looks at and orbits around; dragging in the viewer "
                "starts from here and Reset Camera returns here."
            ),
            inputs=[
                io.Float.Input(
                    "fov",
                    default=50.0,
                    min=0.1,
                    max=179.0,
                    step=0.1,
                    tooltip=(
                        "Vertical field of view in degrees. 50.0 reads naturally, 20.0 is a "
                        "long lens, 90.0 wide."
                    ),
                ),
                io.Float.Input(
                    "near",
                    default=0.1,
                    min=0.000001,
                    max=1000000.0,
                    step=0.001,
                    tooltip=(
                        "Nearest distance drawn. 0.1 suits a scene a few units across; raising "
                        "it sharpens depth precision."
                    ),
                ),
                io.Float.Input(
                    "far",
                    default=1000.0,
                    min=0.0001,
                    max=1000000000.0,
                    step=1.0,
                    tooltip=(
                        "Furthest distance drawn. 1000.0 covers most scenes. Anything beyond it "
                        "is clipped away."
                    ),
                ),
                io.Float.Input(
                    "position_x",
                    default=3.0,
                    min=-1000000.0,
                    max=1000000.0,
                    step=0.01,
                    tooltip="Where the camera sits along X, in scene units. 3.0 stands it off to the right.",
                ),
                io.Float.Input(
                    "position_y",
                    default=2.0,
                    min=-1000000.0,
                    max=1000000.0,
                    step=0.01,
                    tooltip="Where the camera sits along Y. 2.0 looks slightly down on a unit cube.",
                ),
                io.Float.Input(
                    "position_z",
                    default=5.0,
                    min=-1000000.0,
                    max=1000000.0,
                    step=0.01,
                    tooltip="Where the camera sits along Z. 5.0 frames a unit cube comfortably.",
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
        fov,
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
                f"Three Perspective Camera was given far {far} and near {near}. far has to be "
                f"the greater of the two, since it is the back of the range that gets drawn. "
                f"Try near 0.1 and far 1000."
            )
        return io.NodeOutput(
            create_spec(
                "camera",
                "PerspectiveCamera",
                params={
                    "fov": float(fov),
                    "near": float(near),
                    "far": float(far),
                    "position": [float(position_x), float(position_y), float(position_z)],
                    "target": [float(target_x), float(target_y), float(target_z)],
                },
                deps=compact_deps(track=track),
            )
        )

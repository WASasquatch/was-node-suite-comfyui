"""A scene, a camera and the renderer settings together."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_APP, THREE_CAMERA, THREE_EFFECT, THREE_SCENE
from ...modules.threejs.spec import compact_deps, create_spec, require_spec

REQUIRES = "threejs"

#: Tone maps the renderer offers, in the order the menu lists them.
TONE_MAPPING = ("none", "linear", "reinhard", "cineon", "aces", "agx", "neutral")


class ThreeApp(io.ComfyNode):
    """Build an app descriptor."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeApp",
            display_name="Three App",
            search_aliases=[
                "WASThreeApp",
                "Three App",
                "renderer",
                "tone mapping",
                "orbit",
            ],
            category="WAS Suite/Three",
            description=(
                "Bring a scene and a camera together with the settings the renderer runs under, "
                "for Three Viewer, the Three render nodes and Three Compile. Tone mapping, "
                "exposure, shadows, orbit dragging, effects and the animation's loop length are "
                "set here."
            ),
            inputs=[
                THREE_SCENE.Input(
                    "scene",
                    tooltip="What to draw, from Three Scene.",
                ),
                THREE_CAMERA.Input(
                    "camera",
                    tooltip="Where to draw it from, from either of the Three camera nodes.",
                ),
                io.Boolean.Input(
                    "antialias",
                    default=True,
                    tooltip=(
                        "`true` smooths the edges of shapes; `false` is faster and leaves "
                        "visible stair-stepping."
                    ),
                ),
                io.Boolean.Input(
                    "shadows",
                    default=True,
                    tooltip=(
                        "`true` draws shadows, `false` skips them. A light and a mesh must each opt in "
                        "as well."
                    ),
                ),
                io.Boolean.Input(
                    "orbit_controls",
                    default=True,
                    tooltip=(
                        "`true` lets the viewer be dragged to orbit, wheeled to zoom and middle-dragged "
                        "to pan; `false` pins the camera."
                    ),
                ),
                io.Boolean.Input(
                    "auto_rotate",
                    default=False,
                    tooltip=(
                        "`true` turns the camera around the target on its own, pausing while it is "
                        "dragged; `false` holds still."
                    ),
                ),
                io.Float.Input(
                    "auto_rotate_speed",
                    default=1.0,
                    min=-100.0,
                    max=100.0,
                    step=0.01,
                    tooltip="How fast auto rotate turns. 1.0 is a slow drift, negative turns the other way.",
                ),
                io.Combo.Input(
                    "tone_mapping",
                    options=list(TONE_MAPPING),
                    default="aces",
                    tooltip=(
                        "How brightness above 1.0 is brought into range. 'aces' is filmic, "
                        "'none' clips, 'agx' is gentler."
                    ),
                ),
                io.Float.Input(
                    "exposure",
                    default=1.0,
                    min=0.0,
                    max=100.0,
                    step=0.01,
                    tooltip="Overall brightness into the tone map. 1.0 is neutral, 2.0 one stop up.",
                ),
                io.Float.Input(
                    "loop_seconds",
                    default=4.0,
                    min=0.1,
                    max=3600.0,
                    step=0.1,
                    tooltip=(
                        "How long the whole animation lasts, in seconds. 4.0 = a steady "
                        "turntable; 20.0 = a long move a render captures part of. The viewer "
                        "loops over it, Three Render's strip spans it and `per timeline` "
                        "motion spreads across it."
                    ),
                ),
                io.Float.Input(
                    "pixel_ratio_limit",
                    default=2.0,
                    min=0.25,
                    max=8.0,
                    step=0.25,
                    tooltip=(
                        "Ceiling on how many device pixels back one CSS pixel. 2.0 is sharp, "
                        "1.0 is faster on a large viewer."
                    ),
                ),
                THREE_EFFECT.Input(
                    "effects",
                    optional=True,
                    tooltip=(
                        "The chain of passes the frame is put through, from Three Bloom, Three "
                        "Depth Of Field or Three Antialias. Left unwired the frame is shown as "
                        "it was drawn."
                    ),
                ),
                io.Combo.Input(
                    "shadow_quality",
                    options=["512", "1024", "2048", "4096", "8192"],
                    default="2048",
                    tooltip=(
                        "Pixels a side of the map every shadow is drawn into. 512 is blocky, "
                        "2048 suits most scenes, 8192 costs memory and is for a close crop."
                    ),
                ),
            ],
            outputs=[
                THREE_APP.Output(
                    display_name="app",
                    tooltip="The scene, camera and renderer settings, for Three Viewer.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        scene,
        camera,
        antialias,
        shadows,
        orbit_controls,
        auto_rotate,
        auto_rotate_speed,
        tone_mapping,
        exposure,
        loop_seconds,
        pixel_ratio_limit,
        shadow_quality="2048",
        effects=None,
    ) -> io.NodeOutput:
        """Describe the app.

        Raises:
            ValueError: An input is not a descriptor of the kind its socket takes.
        """
        require_spec(scene, "scene")
        require_spec(camera, "camera")
        if effects is not None:
            require_spec(effects, "effect")
        return io.NodeOutput(
            create_spec(
                "app",
                "ThreeApp",
                params={
                    "antialias": bool(antialias),
                    "shadows": bool(shadows),
                    "orbitControls": bool(orbit_controls),
                    "autoRotate": bool(auto_rotate),
                    "autoRotateSpeed": float(auto_rotate_speed),
                    "toneMapping": tone_mapping,
                    "exposure": float(exposure),
                    "loopSeconds": float(loop_seconds),
                    "pixelRatioLimit": float(pixel_ratio_limit),
                    "shadowMapSize": int(shadow_quality),
                },
                deps=compact_deps(scene=scene, camera=camera, effects=effects),
            )
        )

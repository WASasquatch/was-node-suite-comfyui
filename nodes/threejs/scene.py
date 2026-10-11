"""The root of a Three.js scene graph."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_ENVIRONMENT, THREE_OBJECT, THREE_SCENE
from ...modules.threejs.spec import compact_deps, create_spec, require_spec

REQUIRES = "threejs"


class ThreeScene(io.ComfyNode):
    """Build a scene descriptor."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeScene",
            display_name="Three Scene",
            search_aliases=["WASThreeScene", "Three Scene", "scene", "background", "fog"],
            category="WAS Suite/Three",
            description=(
                "Everything that gets drawn, plus what sits behind it: wire one object into "
                "root, usually a Three Group holding the meshes and the lights. Fog fades "
                "distant objects toward its colour and hides the far edge of a ground plane. A "
                "transparent background lets the page show through."
            ),
            inputs=[
                THREE_OBJECT.Input(
                    "root",
                    optional=True,
                    tooltip=(
                        "The object holding the scene, usually a Three Group. Left unwired the "
                        "scene is empty."
                    ),
                ),
                io.String.Input(
                    "background",
                    default="#111111",
                    multiline=False,
                    tooltip=(
                        "Backdrop colour as hexadecimal. #111111 is near black, #ffffff white. "
                        "Ignored when the mode is transparent."
                    ),
                ),
                io.Combo.Input(
                    "background_mode",
                    options=["color", "transparent"],
                    default="color",
                    tooltip=(
                        "'color' fills the backdrop with the colour above. 'transparent' draws "
                        "no backdrop at all."
                    ),
                ),
                io.Boolean.Input(
                    "fog_enabled",
                    default=False,
                    tooltip=(
                        "`true` fades distant objects toward the fog colour; `false` draws everything "
                        "crisp."
                    ),
                ),
                io.String.Input(
                    "fog_color",
                    default="#111111",
                    multiline=False,
                    tooltip=(
                        "Colour distant objects fade into, as hexadecimal. #111111 matching the "
                        "background hides the horizon."
                    ),
                ),
                io.Float.Input(
                    "fog_near",
                    default=10.0,
                    min=0.0,
                    max=1000000000.0,
                    step=0.1,
                    tooltip="Distance the fade starts at. 10.0 leaves a subject at the origin untouched.",
                ),
                io.Float.Input(
                    "fog_far",
                    default=100.0,
                    min=0.0001,
                    max=1000000000.0,
                    step=0.1,
                    tooltip="Distance objects are fully fog by. 100.0 suits a scene tens of units deep.",
                ),
                THREE_ENVIRONMENT.Input(
                    "environment",
                    optional=True,
                    tooltip=(
                        "Surroundings for every physical material to reflect, from Three "
                        "Environment. Left unwired, metal has nothing to mirror and renders "
                        "nearly black."
                    ),
                ),
            ],
            outputs=[
                THREE_SCENE.Output(
                    display_name="scene",
                    tooltip="The scene, for the scene socket on Three App.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        background,
        background_mode,
        fog_enabled,
        fog_color,
        fog_near,
        fog_far,
        root=None,
        environment=None,
    ) -> io.NodeOutput:
        """Describe the scene.

        Raises:
            ValueError: Fog is on with ``fog_far`` not beyond ``fog_near``, or ``root`` is not
                an object descriptor.
        """
        if fog_enabled and fog_far <= fog_near:
            raise ValueError(
                f"Three Scene has fog on with fog_near {fog_near} and fog_far {fog_far}. "
                f"fog_far has to be the greater of the two, since it is where the fade ends. "
                f"Try fog_near 10 and fog_far 100."
            )
        if root is not None:
            require_spec(root, "object")
        if environment is not None:
            require_spec(environment, "environment")
        return io.NodeOutput(
            create_spec(
                "scene",
                "Scene",
                params={
                    "background": background,
                    "backgroundMode": background_mode,
                    "fogEnabled": bool(fog_enabled),
                    "fogColor": fog_color,
                    "fogNear": float(fog_near),
                    "fogFar": float(fog_far),
                },
                deps=compact_deps(root=root, environment=environment),
            )
        )

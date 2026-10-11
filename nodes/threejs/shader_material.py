"""A surface drawn by hand-written GLSL."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_MATERIAL
from ...modules.threejs.spec import create_spec, parse_json_object

REQUIRES = "threejs"

DEFAULT_VERTEX_SHADER = """varying vec2 vUv;

void main() {
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
"""

DEFAULT_FRAGMENT_SHADER = """uniform float time;
uniform vec3 color;
varying vec2 vUv;

void main() {
    float pulse = 0.65 + 0.35 * sin(time * 2.0 + vUv.y * 10.0);
    gl_FragColor = vec4(color * pulse, 1.0);
}
"""

DEFAULT_UNIFORMS = '{"time": {"type": "float", "value": 0}, "color": {"type": "color", "value": "#6fdcff"}}'


class ThreeShaderMaterial(io.ComfyNode):
    """Build a shader material descriptor."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeShaderMaterial",
            display_name="Three Shader Material",
            search_aliases=[
                "WASThreeShaderMaterial",
                "Three Shader Material",
                "glsl",
                "shader",
                "uniform",
            ],
            category="WAS Suite/Three",
            description=(
                "A surface whose look is written in GLSL rather than set by widgets, untouched "
                "by lights. Time, capture progress, loop position and frame size are filled into "
                "its uniforms every frame, so an animated shader needs no wiring."
            ),
            inputs=[
                io.String.Input(
                    "vertex_shader",
                    default=DEFAULT_VERTEX_SHADER,
                    multiline=True,
                    tooltip=(
                        "GLSL for each vertex. It must set `gl_Position`; the default passes "
                        "`uv` through as `vUv`."
                    ),
                ),
                io.String.Input(
                    "fragment_shader",
                    default=DEFAULT_FRAGMENT_SHADER,
                    multiline=True,
                    tooltip=(
                        "GLSL for each pixel. It must set `gl_FragColor`, or `out vec4` under "
                        "the `glsl3` version."
                    ),
                ),
                io.String.Input(
                    "uniforms_json",
                    default=DEFAULT_UNIFORMS,
                    multiline=True,
                    tooltip=(
                        "Typed values, as `{\"time\": {\"type\": \"float\", \"value\": 0}}`; "
                        "types: float, color, vec2, vec3, vec4. Filled each frame, also "
                        "prefixed `u` as in `uTime`: `time` in seconds; `progress` 0 to 1 over "
                        "the capture; `timeline` 0 to 1 over the loop; `resolution` in pixels."
                    ),
                ),
                io.Boolean.Input(
                    "transparent",
                    default=False,
                    tooltip="`true` blends the shader's alpha with what is behind; `false` draws it solid.",
                ),
                io.Boolean.Input(
                    "depth_write",
                    default=True,
                    tooltip=(
                        "`true` records depth so later objects sort behind; `false` suits glow "
                        "and additive passes."
                    ),
                ),
                io.Boolean.Input(
                    "depth_test",
                    default=True,
                    tooltip="`true` hides the surface behind nearer objects; `false` draws it over everything.",
                ),
                io.Combo.Input(
                    "side",
                    options=["front", "back", "double"],
                    default="front",
                    tooltip=(
                        "Which faces are drawn. 'front' for closed shapes, 'double' for planes "
                        "seen from behind."
                    ),
                ),
                io.Combo.Input(
                    "glsl_version",
                    options=["default", "glsl3"],
                    default="default",
                    tooltip=(
                        "'default' is GLSL 1 with `gl_FragColor`; 'glsl3' is GLSL 3 and wants an "
                        "`out vec4` instead."
                    ),
                ),
            ],
            outputs=[
                THREE_MATERIAL.Output(
                    display_name="material",
                    tooltip="The surface, for the material socket on Three Mesh.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        vertex_shader,
        fragment_shader,
        uniforms_json,
        transparent,
        depth_write,
        depth_test,
        side,
        glsl_version,
    ) -> io.NodeOutput:
        """Describe the surface.

        Raises:
            ValueError: ``uniforms_json`` is not a JSON object.
        """
        return io.NodeOutput(
            create_spec(
                "material",
                "ShaderMaterial",
                params={
                    "vertexShader": vertex_shader,
                    "fragmentShader": fragment_shader,
                    "uniforms": parse_json_object(uniforms_json, "uniforms_json"),
                    "transparent": bool(transparent),
                    "depthWrite": bool(depth_write),
                    "depthTest": bool(depth_test),
                    "side": side,
                    "glslVersion": glsl_version,
                },
            )
        )

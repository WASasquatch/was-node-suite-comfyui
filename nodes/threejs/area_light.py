"""A panel of light with a size, for soft shadows and clean traced lighting."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_OBJECT
from ...modules.threejs.spec import create_spec

REQUIRES = "threejs"

#: The outline light is thrown from, in the order the menu lists them.
SHAPES = ("rectangle", "disc")


class ThreeAreaLight(io.ComfyNode):
    """Build an area light descriptor."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeAreaLight",
            display_name="Three Area Light",
            search_aliases=[
                "WASThreeAreaLight",
                "Three Area Light",
                "rect area light",
                "softbox",
                "panel light",
                "key light",
            ],
            category="WAS Suite/Three",
            description=(
                "Light the scene from a panel rather than a point, so shadows soften and "
                "highlights read as a window or softbox. The panel shines from the face toward "
                "its target and is invisible itself; pair it with an emissive material where "
                "the fitting must show. Only a path traced render casts its shadow."
            ),
            inputs=[
                io.String.Input(
                    "color",
                    default="#ffffff",
                    multiline=False,
                    tooltip="Colour of the light as hexadecimal. `#ffffff` is white, `#ffd9a0` tungsten.",
                ),
                io.Float.Input(
                    "intensity",
                    default=5.0,
                    min=0.0,
                    max=10000.0,
                    step=0.1,
                    tooltip=(
                        "How brightly the panel gives off light. It is spread over the whole "
                        "face, so a wider panel at the same number lights no harder. 5.0 is a "
                        "key light two units across, 40.0 a small bright one."
                    ),
                ),
                io.Float.Input(
                    "width",
                    default=2.0,
                    min=0.001,
                    max=1000.0,
                    step=0.1,
                    tooltip=(
                        "How wide the panel is, in scene units. 0.2 throws a hard shadow, 2.0 a "
                        "soft one, 8.0 an almost shadowless wash."
                    ),
                ),
                io.Float.Input(
                    "height",
                    default=2.0,
                    min=0.001,
                    max=1000.0,
                    step=0.1,
                    tooltip=(
                        "How tall the panel is. 2.0 with a 2.0 width is square; 0.1 by 4.0 is a "
                        "strip light."
                    ),
                ),
                io.Combo.Input(
                    "shape",
                    options=list(SHAPES),
                    default="rectangle",
                    tooltip=(
                        "`rectangle` fills the whole width by height. `disc` rounds it off, for "
                        "the round highlight a dish or a ring light leaves. Three Render draws "
                        "both as a rectangle; only a traced render rounds it."
                    ),
                ),
                io.Float.Input(
                    "position_x",
                    default=0.0,
                    min=-10000.0,
                    max=10000.0,
                    step=0.1,
                    tooltip="Where the panel sits, across. 0.0 is the middle, 3.0 off to one side.",
                ),
                io.Float.Input(
                    "position_y",
                    default=4.0,
                    min=-10000.0,
                    max=10000.0,
                    step=0.1,
                    tooltip="How high the panel sits. 4.0 is above a subject standing at 0.0.",
                ),
                io.Float.Input(
                    "position_z",
                    default=2.0,
                    min=-10000.0,
                    max=10000.0,
                    step=0.1,
                    tooltip="How far forward the panel sits. 2.0 is between the camera and the subject.",
                ),
                io.Float.Input(
                    "target_x",
                    default=0.0,
                    min=-10000.0,
                    max=10000.0,
                    step=0.1,
                    tooltip="Where the face points, across. 0.0 aims at the middle of the scene.",
                ),
                io.Float.Input(
                    "target_y",
                    default=0.0,
                    min=-10000.0,
                    max=10000.0,
                    step=0.1,
                    tooltip="The height it aims at. 0.0 is the floor, 1.5 the head of a figure.",
                ),
                io.Float.Input(
                    "target_z",
                    default=0.0,
                    min=-10000.0,
                    max=10000.0,
                    step=0.1,
                    tooltip="The depth it aims at. 0.0 aims at the middle of the scene.",
                ),
            ],
            outputs=[
                THREE_OBJECT.Output(
                    display_name="object",
                    tooltip="The light, for Three Group or Three Scene.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls, color, intensity, width, height, shape, position_x, position_y, position_z,
        target_x, target_y, target_z,
    ) -> io.NodeOutput:
        """Describe the light."""
        return io.NodeOutput(
            create_spec(
                "object",
                "AreaLight",
                params={
                    "color": color,
                    "intensity": float(intensity),
                    "width": float(width),
                    "height": float(height),
                    "shape": shape,
                    "position": [float(position_x), float(position_y), float(position_z)],
                    "target": [float(target_x), float(target_y), float(target_z)],
                },
            )
        )

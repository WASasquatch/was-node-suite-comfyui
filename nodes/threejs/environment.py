"""The light a scene's physical materials reflect."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_ENVIRONMENT
from ...modules.log import get_logger
from ...modules.threejs import environments, textures
from ...modules.threejs.spec import create_spec
from ...modules.util import file_listing, sandbox

REQUIRES = "threejs"

logger = get_logger("nodes.threejs")

#: Most entries the menu offers.
MAX_OPTIONS = 400

#: First entry of the menu, and its default. A saved workflow stores whatever the menu
#: held, so the value it starts on has to be one every install offers.
NO_FILES = "none"

#: Where the light comes from, in the order the menu lists them.
SOURCES = ("studio room", "image", "file", "none")


def options() -> list[str]:
    """The menu's entries, with :data:`NO_FILES` first."""
    return [NO_FILES] + list(
        file_listing.labels(environments.SUFFIXES, file_listing.ROOTS, MAX_OPTIONS)
    )


class ThreeEnvironment(io.ComfyNode):
    """Light the scene from an image."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeEnvironment",
            display_name="Three Environment",
            search_aliases=[
                "WASThreeEnvironment",
                "Three Environment",
                "ibl",
                "hdri",
                "reflection",
                "image based lighting",
            ],
            category="WAS Suite/Three",
            description=(
                "Light the scene from all around rather than from lamps alone, which is what "
                "gives metal and glass something to reflect. Without one a polished dark "
                "material renders almost black, since there is nothing in the world for it to "
                "mirror. 'studio room' builds a small lit room and needs no file. 'image' "
                "takes an equirectangular picture off the wire. 'file' reads a .hdr or .exr, "
                "which carries real intensities and lights a scene far better than an "
                "ordinary picture. Wire the result into Three Scene."
            ),
            inputs=[
                io.Combo.Input(
                    "source",
                    options=list(SOURCES),
                    default="studio room",
                    tooltip=(
                        "Where the light comes from. `studio room` needs nothing wired. "
                        "`image` reads the image input, `file` reads the menu below, and "
                        "`none` leaves the scene lit by its lamps alone."
                    ),
                ),
                io.Image.Input(
                    "image",
                    optional=True,
                    tooltip=(
                        "An equirectangular picture, twice as wide as it is tall, used when "
                        "source is `image`. A batch is read on its first frame alone."
                    ),
                ),
                io.Combo.Input(
                    "file",
                    options=options(),
                    default=NO_FILES,
                    optional=True,
                    tooltip=(
                        "Which `.hdr` or `.exr` to light from, used when source is `file`. "
                        "The menu lists what is in ComfyUI's input, output and temp folders, "
                        "under `none`."
                    ),
                ),
                io.String.Input(
                    "path",
                    default="",
                    multiline=False,
                    optional=True,
                    tooltip=(
                        "An environment somewhere else, as a full path such as "
                        "`D:/hdri/studio_4k.hdr`. Filled in, it is used instead of the menu."
                    ),
                ),
                io.Float.Input(
                    "intensity",
                    default=1.0,
                    min=0.0,
                    max=20.0,
                    step=0.05,
                    tooltip=(
                        "How strongly the surroundings light the scene. 1.0 is the image as "
                        "it is, 0.3 a hint of fill, 3.0 a bright studio."
                    ),
                ),
                io.Boolean.Input(
                    "as_background",
                    default=False,
                    tooltip=(
                        "`true` draws the environment behind the scene as well as reflecting "
                        "it, `false` keeps Three Scene's own background."
                    ),
                ),
                io.Float.Input(
                    "background_blur",
                    default=0.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "How far the background is blurred when it is drawn. 0.0 is sharp, "
                        "0.3 throws it out of focus behind the subject."
                    ),
                ),
                io.Float.Input(
                    "rotation",
                    default=0.0,
                    min=-360.0,
                    max=360.0,
                    step=1.0,
                    tooltip=(
                        "Turns the surroundings around the scene, in degrees, which moves "
                        "where the highlights fall. 0.0 leaves it as the image was shot."
                    ),
                ),
            ],
            outputs=[
                THREE_ENVIRONMENT.Output(
                    display_name="environment",
                    tooltip="The surroundings, for Three Scene's environment socket.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        source,
        intensity,
        as_background,
        background_blur,
        rotation,
        image=None,
        file="",
        path="",
    ) -> io.NodeOutput:
        """Hold the environment for the browser and describe how it is used.

        Raises:
            ValueError: ``source`` names an input that is not filled in, or the file named
                has no loader.
            PathNotAllowed: The path resolved outside every permitted read root.
        """
        params = {
            "source": source,
            "intensity": float(intensity),
            "asBackground": bool(as_background),
            "backgroundBlur": float(background_blur),
            "rotation": float(rotation),
        }

        if source == "image":
            if image is None:
                raise ValueError(
                    "Three Environment is set to 'image' with nothing wired into image. Wire "
                    "an equirectangular picture in, or choose another source."
                )
            params["url"] = textures.texture_url(image)
            params["format"] = "png"
        elif source == "file":
            chosen = str(path).strip()
            if not chosen:
                if not file or file == NO_FILES:
                    raise ValueError(
                        "Three Environment is set to 'file' with no file chosen. Put a .hdr or "
                        ".exr in ComfyUI's input folder and pick it, give a full path, or "
                        "choose 'studio room', which needs no file."
                    )
                chosen = file_listing.resolve(
                    file, environments.SUFFIXES, file_listing.ROOTS
                ) or file
            resolved = sandbox.resolve_read(chosen)
            url, kind = environments.carried(resolved)
            params["url"] = url
            params["format"] = kind

        return io.NodeOutput(create_spec("environment", "Environment", params=params))

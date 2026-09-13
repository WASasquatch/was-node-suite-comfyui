"""A 3D model file placed in a Three.js scene."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import THREE_OBJECT
from ...modules.log import get_logger
from ...modules.threejs import models
from ...modules.threejs.spec import create_spec
from ...modules.util import file_listing, sandbox

REQUIRES = "threejs"

logger = get_logger("nodes.threejs")

#: Most entries the menu offers.
MAX_OPTIONS = 400

#: What the menu says when it found nothing.
NO_MODELS = "no model files found"


def options() -> list[str]:
    """The menu's entries, or ``[NO_MODELS]`` when there are none."""
    return list(
        file_listing.labels(models.SUFFIXES, file_listing.ROOTS, MAX_OPTIONS)
    ) or [NO_MODELS]


class ThreeLoadModel(io.ComfyNode):
    """Put a model file into the scene."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASThreeLoadModel",
            display_name="Three Load Model",
            search_aliases=[
                "WASThreeLoadModel",
                "Three Load Model",
                "gltf",
                "glb",
                "obj",
                "load 3d model",
            ],
            category="WAS Suite/Three",
            description=(
                "Put a model file into the scene as an object, so a mesh made elsewhere can be "
                "lit, animated and rendered here. It reads .glb, .gltf, .dae, .fbx, .obj, "
                ".3mf, .stl and .ply. The menu lists what is in ComfyUI's input, output and "
                "temp folders; path takes anything else, including the mesh_path a Load 3D "
                "node answers with. A .glb, .gltf, .dae, .fbx or .3mf brings its own "
                "materials, an .obj takes them from a .mtl beside it, and a .stl or .ply "
                "arrives bare and takes the material wired in. Wire nothing and it keeps "
                "whatever it came with."
            ),
            inputs=[
                io.Combo.Input(
                    "file",
                    options=options(),
                    tooltip=(
                        "Which model to place. The menu lists every `.glb`, `.gltf`, `.dae`, "
                        "`.fbx`, `.obj`, `.3mf`, `.stl` and `.ply` in ComfyUI's input, "
                        "output and temp folders."
                    ),
                ),
                io.String.Input(
                    "path",
                    default="",
                    multiline=False,
                    optional=True,
                    tooltip=(
                        "A model somewhere else, as a full path. Wire Load 3D's mesh_path here. "
                        "Filled in, it is used instead of the menu."
                    ),
                ),
                io.Float.Input(
                    "scale",
                    default=1.0,
                    min=0.0001,
                    max=10000.0,
                    step=0.01,
                    tooltip="Multiplies the model's own size. 1.0 leaves it, 0.01 suits a model authored in centimetres.",
                ),
                io.Boolean.Input(
                    "centre",
                    default=True,
                    tooltip=(
                        "`true` moves the model so its middle sits at the origin; `false` keeps "
                        "the coordinates it was saved with."
                    ),
                ),
                io.Boolean.Input(
                    "cast_shadow",
                    default=True,
                    tooltip="`true` lets every mesh in the model throw and receive shadows, `false` neither.",
                ),
            ],
            outputs=[
                THREE_OBJECT.Output(
                    display_name="object",
                    tooltip="The loaded model, for Three Group or the root socket on Three Scene.",
                ),
            ],
        )

    @classmethod
    def execute(cls, file, scale, centre, cast_shadow, path="") -> io.NodeOutput:
        """Hold the model for the browser and describe where it goes.

        Raises:
            ValueError: Neither a menu entry nor a path names a readable model, or its
                suffix has no loader.
            PathNotAllowed: The path resolved outside every permitted read root.
        """
        chosen = str(path).strip()
        if not chosen:
            if not file or file == NO_MODELS:
                raise ValueError(
                    "Three Load Model has no model to place. Put a .glb, .gltf, .dae, .fbx, "
                    ".obj, .3mf, .stl or .ply in ComfyUI's input folder and pick it, or wire "
                    "Load 3D's mesh_path into path."
                )
            chosen = file_listing.resolve(file, models.SUFFIXES, file_listing.ROOTS) or file

        resolved = sandbox.resolve_read(chosen)
        url, kind, sidecars = models.carried(resolved)
        return io.NodeOutput(
            create_spec(
                "object",
                "ModelFile",
                params={
                    "url": url,
                    "format": kind,
                    "scale": float(scale),
                    "centre": bool(centre),
                    "castShadow": bool(cast_shadow),
                    "name": resolved.stem,
                    "resources": sidecars,
                },
            )
        )

"""Read a 3D model file into a model the 3D nodes work on."""

from __future__ import annotations

import os

from comfy_api.latest import io

from ...modules.compat.types import FILE_3D, FILE_3D_FBX, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_OBJ, MODEL_3D
from ...modules.io import picker
from ...modules.log import get_logger
from ...modules.model3d import files, inputs, meta, orient, report

logger = get_logger("nodes.model3d")

#: What the menu says when it found nothing.
NO_MODELS = "no model files found"

#: Where the model's origin goes, against the placement `orient.place` makes.
ORIGINS = {"as saved": "as is", "base": "base", "centre": "centre"}


def options() -> list[str]:
    """The menu's entries, or ``[NO_MODELS]`` when there are none."""
    return picker.labels(files.SUFFIXES) or [NO_MODELS]


def chosen_path(file: str) -> str | None:
    """The file a menu label or a wired path names, contained, or None when nothing is there."""
    if not file or file == NO_MODELS:
        return None
    return picker.resolve(file, files.SUFFIXES)


class Model3DLoad(io.ComfyNode):
    """Read a GLB, glTF, VRM, FBX or OBJ file as a 3D model."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASModel3DLoad",
            display_name="3D Model Load",
            search_aliases=[
                "WASModel3DLoad", "3D Model Load", "load 3d model", "import glb", "load vrm",
                "load fbx", "load obj", "gltf", "tripo",
            ],
            category="WAS Suite/3D/Model",
            description=(
                "Read a GLB, glTF, VRM, FBX or OBJ file with its materials, skeleton, skin, shape "
                "keys and animations, ready for the mesh and character nodes. It can stand the "
                "model on the floor, scale it to a height and turn a rigged character to face "
                "forward."
            ),
            inputs=[
                picker.path_input(io.Combo.Input(
                    "file",
                    options=options(),
                    tooltip=(
                        "Which model to read. The menu lists the newest .glb, .gltf, .vrm, .fbx and "
                        ".obj files in ComfyUI's input, output and temp folders and in any folder "
                        "under paths.allow_read."
                    ),
                )),
                io.Combo.Input(
                    "origin",
                    options=list(ORIGINS),
                    default="as saved",
                    tooltip=(
                        "Where the model sits. `as saved` = the file's own coordinates; `base` = "
                        "centred and standing on the floor at height 0; `centre` = the middle of its "
                        "box at the origin."
                    ),
                ),
                io.Float.Input(
                    "scale_to_height",
                    default=0.0, min=0.0, max=1000.0, step=0.01,
                    tooltip=(
                        "0 = keep the file's size; 1.7 = scaled evenly, about its feet, until it "
                        "stands 1.7 metres tall. Generators deliver models at any size."
                    ),
                ),
                io.Boolean.Input(
                    "face_forward",
                    default=False,
                    tooltip=(
                        "`true` = turn a rigged humanoid about its hips until it faces +Z, as VRM and "
                        "the character nodes expect; `false` = keep the file's facing. A model "
                        "without a skeleton is never turned."
                    ),
                ),
                io.MultiType.Input(
                    "model_3d",
                    types=[FILE_3D, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_FBX, FILE_3D_OBJ],
                    optional=True,
                    tooltip=(
                        "A 3D file from Load 3D or a 3D generator such as Tripo. Wired, it is read "
                        "instead of the menu's file."
                    ),
                ),
            ],
            outputs=[
                MODEL_3D.Output("model_3d", tooltip="The whole model: parts with their materials, and the skeleton, skin, springs, expressions and animations it carries."),
                io.Mesh.Output("mesh", tooltip="The same model as one mesh for core's mesh nodes and Save 3D Model. Parts are merged, their textures packed into one atlas, and the skeleton left out."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def fingerprint_inputs(cls, file, origin="as saved", scale_to_height=0.0, face_forward=False,
                           model_3d=None):
        """The picked file's size and change time, so an edited file is read again."""
        if model_3d is not None:
            return None
        try:
            found = chosen_path(file)
            if found is None:
                return float("NaN")
            status = os.stat(found)
            return (found, status.st_size, status.st_mtime_ns)
        except Exception as error:
            logger.debug("3D Model Load could not stat its file: %s", error)
            return float("NaN")

    @classmethod
    def execute(cls, file, origin="as saved", scale_to_height=0.0, face_forward=False,
                model_3d=None) -> io.NodeOutput:
        """Read the model, then turn, scale and place it as asked.

        Raises:
            ValueError: Nothing names a model, or the file cannot be read as one.
            PathNotAllowed: The picked or wired path lies outside every permitted read root.
        """
        if model_3d is not None:
            model = inputs.to_model(model_3d)
        else:
            found = chosen_path(file)
            if found is None and file and file != NO_MODELS:
                raise ValueError(
                    f"3D Model Load cannot find '{str(file)[:120]}'; it may have been moved or "
                    "deleted. Refresh the node definitions and pick a model the menu lists."
                )
            if found is None:
                raise ValueError(
                    "3D Model Load has no model to read. Put a .glb, .gltf, .vrm, .fbx or .obj file "
                    "in ComfyUI's input folder and pick it, or wire a 3D file into model_3d."
                )
            model = files.load_path(found)
        if not model.parts:
            raise ValueError("the file holds no mesh to work on; pick a model with geometry in it")
        turned = orient.face_forward(model) if face_forward else 0.0
        factor = orient.scale_to_height(model, float(scale_to_height)) if scale_to_height > 0 else 1.0
        orient.place(model, ORIGINS.get(origin, "as is"))

        held = meta.contents(model)
        name = str(model.meta.get("source_name") or "the wired model")
        facts = {"file": name, "format": model.source_format or "?"}
        if turned:
            facts["turned"] = f"{turned:.1f} degrees about Y"
        if factor != 1.0:
            facts["scaled"] = f"x{factor:.4g}"
        summary = f"{name}: {model.triangle_count:,} triangles in {len(model.parts)} part(s)"
        if model.skeleton is not None:
            summary += f", {model.skeleton.count} joints"
        report.publish(model, summary, counts=held, facts=facts)
        hidden = cls.hidden
        wanted = inputs.mesh_wanted(getattr(hidden, "prompt", None), getattr(hidden, "unique_id", None))
        return io.NodeOutput(*inputs.node_output(model, mesh=wanted))

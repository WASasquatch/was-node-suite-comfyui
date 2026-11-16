"""Write a 3D model to ComfyUI's output or input folder as GLB, VRM, FBX, OBJ or glTF."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import FILE_3D, FILE_3D_FBX, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_OBJ, MODEL_3D
from ...modules.log import get_logger
from ...modules.model3d import export, files, inputs, report
from ...modules.util import sandbox

logger = get_logger("nodes.model3d")

#: Formats offered, in menu order.
FORMATS = ("glb", "vrm", "fbx", "obj", "gltf")

#: Texture encodings offered.
TEXTURE_FORMATS = ("keep", "png", "jpeg")

#: Prefix used when the widget is left empty.
DEFAULT_PREFIX = "3d/WAS"

#: Folders a model may be saved in.
FOLDERS = ("output", "input")


class Model3DSave(io.ComfyNode):
    """Write a model to a file in ComfyUI's output or input folder."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASModel3DSave",
            display_name="3D Model Save",
            search_aliases=[
                "WASModel3DSave", "3D Model Save", "save 3d model", "export glb", "export vrm",
                "export fbx", "export obj", "save vrm", "gltf",
            ],
            category="WAS Suite/3D/Model",
            description=(
                "Write a 3D model to ComfyUI's output or input folder as GLB, VRM avatar, FBX, OBJ or glTF, "
                "keeping its skeleton, skin, shape keys and animations where the format holds them. "
                "A VRM is turned to face forward and its humanoid bones are mapped first."
            ),
            inputs=[
                io.MultiType.Input("model_3d", types=[MODEL_3D, io.Mesh, FILE_3D, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_FBX, FILE_3D_OBJ], tooltip="The model to work on: a 3D model from another 3D node, a mesh from core's mesh nodes, or a GLB, glTF, VRM, FBX or OBJ file from Load 3D or a 3D generator such as Tripo."),
                io.Combo.Input(
                    "format",
                    options=list(FORMATS),
                    default="glb",
                    tooltip=(
                        "`glb` = one file with everything inside; `vrm` = a VRM 1.0 avatar, which needs "
                        "a humanoid rig and holds no animation clips; `fbx` = for Blender, Unity and "
                        "Unreal, without animation clips; `obj` = mesh and materials only; `gltf` = "
                        "text glTF."
                    ),
                ),
                io.String.Input(
                    "filename_prefix",
                    default=DEFAULT_PREFIX,
                    tooltip=(
                        "The file's name before its number, below the output folder, such as "
                        "`3d/WAS` or `characters/hero`. Every file of one save shares it."
                    ),
                ),
                io.Boolean.Input(
                    "embed_textures",
                    default=True,
                    tooltip=(
                        "`true` = textures inside the gltf or fbx file; `false` = beside it as image "
                        "files. glb and vrm always hold their textures; obj always writes them "
                        "beside."
                    ),
                ),
                io.Combo.Input(
                    "texture_format",
                    options=list(TEXTURE_FORMATS),
                    default="keep",
                    tooltip=(
                        "`keep` = each texture as it came; `png` = lossless and larger; `jpeg` = smaller "
                        "files. A texture with transparency stays png."
                    ),
                ),
                io.Combo.Input(
                    "folder",
                    options=list(FOLDERS),
                    default="output",
                    tooltip=(
                        "`output` = ComfyUI's output folder; `input` = its input folder, where file "
                        "menus such as Tripo's Import model list it. Those menus show only files "
                        "directly in input, so give a prefix with no folder, such as `hero`."
                    ),
                ),
            ],
            outputs=[
                FILE_3D.Output(
                    "file_3d",
                    tooltip="The saved model file, for core's 3D preview and save nodes or another 3D Model Load.",
                ),
                io.AnyType.Output(
                    "filename",
                    tooltip=(
                        "The saved file's name as ComfyUI's file menus write it, such as hero_00001_.glb, on a "
                        "wire a dropdown accepts. Connect it to a model menu such as Tripo's Import model."
                    ),
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, model_3d, format="glb", filename_prefix=DEFAULT_PREFIX, embed_textures=True,
                texture_format="keep", folder="output") -> io.NodeOutput:
        """Write the model and hand on the main file.

        Raises:
            ValueError: The model is empty, cannot be written in the format, or the format is unknown.
            PathNotAllowed: The prefix leaves the chosen folder.
        """
        import folder_paths
        from comfy_api.latest import Types

        if format not in FORMATS:
            raise ValueError(f"format '{format}' is not one of {', '.join(FORMATS)}")
        model = inputs.to_model(model_3d)
        if not model.parts or model.triangle_count == 0:
            raise ValueError("the model has no triangles to save; check the nodes before 3D Model Save "
                             "kept at least one part")
        before = report.figures(model)
        facts: dict = {"format": format}
        if format == "vrm":
            prepared = export.prepare_vrm(model)
            facts["turned"] = f"{prepared['turned']:.1f} degrees about Y"
            facts["bones guessed"] = prepared["bones guessed"]
            if not model.meta.get("authors"):
                facts["authors"] = "Unknown; set them with 3D Model Meta"
        left_out = export.left_out(model, format)
        if left_out:
            facts["left out"] = ", ".join(left_out)
            logger.warning("3D Model Save left out what a .%s file cannot hold: %s. Save as glb to keep "
                           "everything", format, ", ".join(left_out))
        export.set_texture_format(model, texture_format)

        prefix = str(filename_prefix or "").strip() or DEFAULT_PREFIX
        into_input = folder == "input"
        if into_input:
            reason = sandbox.leaves_folder(prefix)
            if reason is not None:
                raise sandbox.PathNotAllowed(f"the filename prefix {reason}; use a prefix such as `hero`")
            below, _, leaf = prefix.replace("\\", "/").rpartition("/")
            base = folder_paths.get_input_directory()
            if below:
                base = str(sandbox.resolve_save_file(base, below))
            target, name, counter, _, _ = sandbox.save_image_path(leaf or "WAS", base)
            subfolder = below
        else:
            target, name, counter, subfolder, _ = sandbox.save_image_path(prefix, folder_paths.get_output_directory())
        stem = f"{name}_{counter:05}_"
        written = files.save(model, format, target, stem, embed_textures=bool(embed_textures),
                             input_folder=into_input)
        main = written[0]
        shown = f"{subfolder}/{main.name}" if subfolder else main.name
        size = sum(path.stat().st_size for path in written)
        facts["file"] = shown
        facts["folder"] = folder if folder in FOLDERS else "output"
        report.publish(
            model, f"saved {shown}", before=before,
            counts={"files": len(written), "megabytes": round(size / 1048576.0, 2)}, facts=facts,
        )
        results = [{"filename": main.name, "subfolder": subfolder, "type": "input" if into_input else "output"}]
        listed = f"{subfolder}/{main.name}" if subfolder else main.name
        if not into_input:
            listed += " [output]"
        return io.NodeOutput(Types.File3D(str(main), format), listed, ui={"3d": results})

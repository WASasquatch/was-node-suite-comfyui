"""Correct the metalness and roughness of a model's parts, such as hair or painted iron a generator made look like chrome."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import FILE_3D, FILE_3D_FBX, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_OBJ, MODEL_3D
from ...modules.model3d import edit, inputs, materials, report, selection


class Model3DMaterial(io.ComfyNode):
    """Tone down metalness and raise roughness on chosen parts."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASModel3DMaterial",
            display_name="3D Model Material",
            search_aliases=[
                "WASModel3DMaterial", "3D Model Material", "metalness", "roughness", "shiny hair",
                "chrome hair", "material fix", "double sided", "black chrome", "dark metal", "PBR",
            ],
            category="WAS Suite/3D/Model",
            description=(
                "Correct how a model's parts take light: scale metalness down, clear it from dark "
                "texels and give every surface a minimum roughness. For hair, skin, cloth or painted "
                "iron that a 3D generator made look like polished metal or black chrome."
            ),
            inputs=[
                io.MultiType.Input("model_3d", types=[MODEL_3D, io.Mesh, FILE_3D, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_FBX, FILE_3D_OBJ], tooltip="The model to work on: a 3D model from another 3D node, a mesh from core's mesh nodes, or a GLB, glTF, VRM, FBX or OBJ file from Load 3D or a 3D generator such as Tripo."),
                io.Float.Input(
                    "metalness", default=0.2, min=0.0, max=1.0, step=0.05,
                    tooltip="Multiplies the chosen parts' metalness. 0 = no metal at all; 0.2 = a fifth; 1.0 = unchanged.",
                ),
                io.Float.Input(
                    "min_roughness", default=0.5, min=0.0, max=1.0, step=0.05,
                    tooltip="Every texel of the chosen parts is at least this rough. 0.5 = satin, no mirror "
                            "highlights; 0.8 = matte; 0 = unchanged.",
                ),
                io.Combo.Input(
                    "double_sided", options=list(materials.DOUBLE_SIDED), default="keep",
                    tooltip="`on` = draw back faces too, for single-layer cloth and hair cards; `off` = front faces "
                            "only; `keep` = as the material says.",
                ),
                io.String.Input("parts", default="", tooltip="Parts to work on: comma-separated names or labels, wildcards allowed, such as `head, hair*`. Empty works on every part."),
                io.Float.Input(
                    "metal_min_brightness", default=0.4, min=0.0, max=1.0, step=0.05,
                    tooltip="Texels stay metal only where the base colour's brightest channel reaches this; "
                            "darker ones become painted, non-metal surfaces. 0.4 = black iron turns matte, "
                            "brass and steel stay metal; 0 = unchanged.",
                ),
            ],
            outputs=[
                MODEL_3D.Output("model_3d", tooltip="The whole model: parts with their materials, and the skeleton, skin, springs, expressions and animations it carries."),
                io.Mesh.Output("mesh", tooltip="The same model as one mesh for core's mesh nodes and Save 3D Model. Parts are merged, their textures packed into one atlas, and the skeleton left out."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model_3d, metalness=0.2, min_roughness=0.5, double_sided="keep", parts="",
                metal_min_brightness=0.4) -> io.NodeOutput:
        """Adjust the chosen parts' materials and publish what changed.

        Raises:
            ValueError: Nothing is wired, or no part matches ``parts``.
        """
        model = inputs.to_model(model_3d)
        chosen = selection.selected(model, parts)
        if not chosen:
            raise ValueError(f"no part matches '{str(parts)[:80]}'; the model's parts are {edit.part_names(model)}. "
                             "Leave parts empty to change every part.")
        counts = materials.adjust(model, chosen, float(metalness), float(min_roughness),
                                  double_sided if double_sided in materials.DOUBLE_SIDED else "keep",
                                  float(metal_min_brightness))
        report.publish(model, f"{counts['materials changed']} material(s) on {len(chosen)} part(s)",
                       counts=counts, facts={"parts": edit.part_names(model, 8)})
        hidden = cls.hidden
        wanted_mesh = inputs.mesh_wanted(getattr(hidden, "prompt", None), getattr(hidden, "unique_id", None))
        return io.NodeOutput(*inputs.node_output(model, mesh=wanted_mesh))

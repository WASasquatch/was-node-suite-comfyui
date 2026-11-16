"""Cut a 3D model with a plane, keep one side or both as separate parts, and cap the cut."""

from __future__ import annotations

import torch
from comfy_api.latest import io

from ...modules.compat.types import FILE_3D, FILE_3D_FBX, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_OBJ, MODEL_3D
from ...modules.mesh import cut as mesh_cut
from ...modules.model3d import edit, inputs, report, selection

#: Directions the plane may be square to; ``custom`` reads the normal inputs.
AXES = ("X", "Y", "Z", "custom")

#: How ``position`` is read.
POSITION_MODES = ("share of size", "metres")

#: Sides that may be kept.
KEEP = ("both", "above", "below")

#: Cap texture choices as shown, against the cut module's modes.
CAP_TEXTURES = {"edge colour": "edge", "whole texture": "planar"}

#: Furthest coordinate a plane may sit at, in metres.
MAX_COORDINATE = 100000.0


class MeshCut(io.ComfyNode):
    """Cut a model with a plane and close the openings."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASMeshCut",
            display_name="Mesh Cut",
            search_aliases=[
                "WASMeshCut", "Mesh Cut", "bisect", "slice mesh", "plane cut", "split model", "cut in half",
                "cap holes",
            ],
            category="WAS Suite/3D/Mesh",
            description=(
                "Cut a model with a flat plane, such as taking a head off at the neck or halving a model for "
                "printing. Keep the part above, below or both as separate parts, with the cut closed by flat "
                "caps or left open; skin and shape keys carry through."
            ),
            inputs=[
                io.MultiType.Input("model_3d", types=[MODEL_3D, io.Mesh, FILE_3D, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_FBX, FILE_3D_OBJ], tooltip="The model to work on: a 3D model from another 3D node, a mesh from core's mesh nodes, or a GLB, glTF, VRM, FBX or OBJ file from Load 3D or a 3D generator such as Tripo."),
                io.Combo.Input(
                    "axis", options=list(AXES), default="Y",
                    tooltip="Axis the plane is square to. `Y` = a level cut, above is up; `X` = left from right, "
                            "above is the character's left; `Z` = front from back; `custom` = the plane through "
                            "point_x, y, z facing normal_x, y, z.",
                ),
                io.Float.Input(
                    "position", default=0.5, min=-MAX_COORDINATE, max=MAX_COORDINATE, step=0.001,
                    tooltip="Where the plane crosses the axis. As a share of size: 0 = lowest point, 0.5 = middle, "
                            "1 = highest. In metres: the coordinate, such as 1.45 for a neck at 1.45 m. "
                            "Unused with `custom`.",
                ),
                io.Combo.Input(
                    "position_mode", options=list(POSITION_MODES), default="share of size",
                    tooltip="`share of size` = position runs 0 to 1 across the chosen parts' extent along the axis. "
                            "`metres` = position is a coordinate.",
                ),
                io.Combo.Input(
                    "keep", options=list(KEEP), default="both",
                    tooltip="`both` = each cut part becomes <name>_above and <name>_below. `above` = keep the "
                            "side the axis points to. `below` = keep the other side.",
                ),
                io.Boolean.Input(
                    "cap", default=True,
                    tooltip="`true` = close each opening the cut leaves with flat faces, so a closed model stays "
                            "closed. `false` = leave the cut open.",
                ),
                io.Combo.Input(
                    "cap_texture", options=list(CAP_TEXTURES), default="edge colour", advanced=True,
                    tooltip="Texture coordinates of the caps. `edge colour` = each cap takes the texture colour "
                            "at its cut edge. `whole texture` = a flat projection across the whole texture, "
                            "ready for a new unwrap or bake.",
                ),
                io.Float.Input(
                    "point_x", default=0.0, min=-MAX_COORDINATE, max=MAX_COORDINATE, step=0.001, advanced=True,
                    tooltip="With axis `custom`: X of a point on the plane, in metres.",
                ),
                io.Float.Input(
                    "point_y", default=0.0, min=-MAX_COORDINATE, max=MAX_COORDINATE, step=0.001, advanced=True,
                    tooltip="With axis `custom`: Y of a point on the plane, in metres.",
                ),
                io.Float.Input(
                    "point_z", default=0.0, min=-MAX_COORDINATE, max=MAX_COORDINATE, step=0.001, advanced=True,
                    tooltip="With axis `custom`: Z of a point on the plane, in metres.",
                ),
                io.Float.Input(
                    "normal_x", default=0.0, min=-1.0, max=1.0, step=0.01, advanced=True,
                    tooltip="With axis `custom`: X of the direction the plane faces, towards the side kept as "
                            "above. Normalised; 0, 1, 0 = a level cut.",
                ),
                io.Float.Input(
                    "normal_y", default=1.0, min=-1.0, max=1.0, step=0.01, advanced=True,
                    tooltip="With axis `custom`: Y of the direction the plane faces. 0.7 with normal_z 0.7 = a "
                            "cut sloping up towards the front.",
                ),
                io.Float.Input(
                    "normal_z", default=0.0, min=-1.0, max=1.0, step=0.01, advanced=True,
                    tooltip="With axis `custom`: Z of the direction the plane faces.",
                ),
                io.Float.Input(
                    "snap_distance", default=0.00001, min=0.0, max=0.01, step=0.00001, advanced=True,
                    tooltip="Vertices this close to the plane, in metres, move onto it so the cut makes no "
                            "sliver faces. 0 = exact; 0.00001 = a hundredth of a millimetre.",
                ),
                io.String.Input("parts", default="", advanced=True, tooltip="Parts to work on: comma-separated names or labels, wildcards allowed, such as `head, hair*`. Empty works on every part."),
            ],
            outputs=[
                MODEL_3D.Output("model_3d", tooltip="The whole model: parts with their materials, and the skeleton, skin, springs, expressions and animations it carries."),
                io.Mesh.Output("mesh", tooltip="The same model as one mesh for core's mesh nodes and Save 3D Model. Parts are merged, their textures packed into one atlas, and the skeleton left out."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model_3d, axis="Y", position=0.5, position_mode="share of size", keep="both", cap=True,
                cap_texture="edge colour", point_x=0.0, point_y=0.0, point_z=0.0, normal_x=0.0, normal_y=1.0,
                normal_z=0.0, snap_distance=0.00001, parts="") -> io.NodeOutput:
        """Cut the chosen parts and cap them.

        Raises:
            ValueError: Nothing is wired, no part matches ``parts``, the chosen parts have no triangles, the
                custom normal has no length, or nothing is left after the cut.
        """
        model = inputs.to_model(model_3d)
        before = report.figures(model)
        chosen = selection.selected(model, parts)
        if not chosen:
            raise ValueError(f"no part matches '{str(parts)[:80]}'; the model's parts are {edit.part_names(model)}. "
                             "Leave parts empty to cut every part.")
        chosen = [index for index in chosen if model.parts[index].triangle_count]
        if not chosen:
            raise ValueError("the chosen parts have no triangles to cut; wire a model with faces or choose other parts")
        axis = axis if axis in AXES else "Y"
        if axis == "custom":
            point = torch.tensor([point_x, point_y, point_z], dtype=torch.float64)
            normal = torch.tensor([normal_x, normal_y, normal_z], dtype=torch.float64)
            if float(normal.norm()) <= 1e-12:
                raise ValueError("the custom plane's normal is 0, 0, 0; set normal_x, normal_y and normal_z to the "
                                 "direction the plane faces, such as 0, 1, 0 for a level cut")
            normal = normal / normal.norm()
        else:
            column = "XYZ".index(axis)
            normal = torch.zeros(3, dtype=torch.float64)
            normal[column] = 1.0
            coordinate = float(position)
            if position_mode != "metres":
                low = min(float(model.parts[index].vertices[:, column].min()) for index in chosen)
                high = max(float(model.parts[index].vertices[:, column].max()) for index in chosen)
                coordinate = low + (high - low) * float(position)
            point = normal * coordinate
        triangles_before = model.triangle_count
        counts = mesh_cut.cut_model(
            model, chosen, point, normal, keep if keep in KEEP else "both", bool(cap), float(snap_distance),
            CAP_TEXTURES.get(cap_texture, "edge"),
        )
        if model.triangle_count == 0:
            raise ValueError("nothing is left after the cut; the plane lies outside the model on the kept side. "
                             "Move position or keep the other side")
        state = "all openings capped" if cap and not counts["uncapped"] else (
            f"{counts['uncapped']} openings left open" if cap else "cut left open")
        report.publish(
            model, f"{counts['cut_parts']} parts cut, {state}", before=before,
            counts={
                "parts cut": counts["cut_parts"],
                "openings capped": counts["capped"],
                "openings left": counts["uncapped"],
                "parts removed": counts["removed_parts"],
            },
            facts={
                "plane point": ", ".join(f"{value:.4f}" for value in point.tolist()),
                "plane normal": ", ".join(f"{value:.4f}" for value in normal.tolist()),
                "triangles changed": f"{triangles_before:,} to {model.triangle_count:,}",
                "parts": edit.part_names(model, 8),
            },
        )
        hidden = cls.hidden
        wanted = inputs.mesh_wanted(getattr(hidden, "prompt", None), getattr(hidden, "unique_id", None))
        return io.NodeOutput(*inputs.node_output(model, mesh=wanted))

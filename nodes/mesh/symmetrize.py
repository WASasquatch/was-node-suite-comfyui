"""Make a 3D model mirror-symmetric by copying one side onto the other, skin and shape keys included."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import FILE_3D, FILE_3D_FBX, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_OBJ, MODEL_3D
from ...modules.mesh import symmetry
from ...modules.model3d import edit, inputs, report, selection

#: Axes the mirror plane may be square to.
AXES = ("X", "Y", "Z", "auto")

#: Sides that may be kept.
SIDES = ("positive", "negative")


class MeshSymmetrize(io.ComfyNode):
    """Replace one side of a model with the mirror image of the other."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASMeshSymmetrize",
            display_name="Mesh Symmetrize",
            search_aliases=[
                "WASMeshSymmetrize", "Mesh Symmetrize", "mirror mesh", "symmetry", "make symmetric",
                "mirror model", "symmetrise",
            ],
            category="WAS Suite/3D/Mesh",
            description=(
                "Make a model mirror-symmetric: find its mirror plane, then replace one side with the mirror "
                "image of the other, welded at the centre. Textures keep their places, and a rigged model's "
                "skin, joints and left and right shape keys are mirrored by name."
            ),
            inputs=[
                io.MultiType.Input("model_3d", types=[MODEL_3D, io.Mesh, FILE_3D, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_FBX, FILE_3D_OBJ], tooltip="The model to work on: a 3D model from another 3D node, a mesh from core's mesh nodes, or a GLB, glTF, VRM, FBX or OBJ file from Load 3D or a 3D generator such as Tripo."),
                io.Combo.Input(
                    "axis", options=list(AXES), default="auto",
                    tooltip="Axis the mirror plane is square to. `auto` = the best of X, Y, Z and the shape's own "
                            "directions; `X` = left and right of a character facing +Z; `Z` = of one facing +X; "
                            "`Y` = top and bottom.",
                ),
                io.Combo.Input(
                    "keep", options=list(SIDES), default="positive",
                    tooltip="Side copied onto the other. `positive` = the side the axis points to, for X the "
                            "character's left; `negative` = the other side.",
                ),
                io.Combo.Input(
                    "plane", options=list(symmetry.PLANES), default="fit",
                    tooltip="Where the plane sits. `fit` = starts through the middle of the surface, then turns "
                            "and moves until both halves match best; `centre` = through the middle, square to "
                            "the axis; `origin` = through 0, square to the axis.",
                ),
                io.Boolean.Input(
                    "symmetrize_skeleton", default=True, advanced=True,
                    tooltip="`true` = move joints on the replaced side to mirror their partners and centre "
                            "unpaired joints near the plane, so both sides bend alike. `false` = leave the "
                            "skeleton. Skin is mirrored by joint names either way.",
                ),
                io.Float.Input(
                    "seam_distance", default=0.0001, min=0.0, max=0.1, step=0.00001, advanced=True,
                    tooltip="Vertices this close to the plane, in metres, move onto it and are shared by both "
                            "halves. 0.0001 = a tenth of a millimetre; raise it if the centre line shows a crack.",
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
    def execute(cls, model_3d, axis="auto", keep="positive", plane="fit", symmetrize_skeleton=True,
                seam_distance=0.0001, parts="") -> io.NodeOutput:
        """Find the plane on the chosen parts and mirror them across it.

        Raises:
            ValueError: Nothing is wired, no part matches ``parts``, or the chosen parts have no triangles.
        """
        model = inputs.to_model(model_3d)
        before = report.figures(model)
        chosen = selection.selected(model, parts)
        if not chosen:
            raise ValueError(f"no part matches '{str(parts)[:80]}'; the model's parts are {edit.part_names(model)}. "
                             "Leave parts empty to mirror every part.")
        found = symmetry.symmetrize_model(
            model, chosen, axis if axis in AXES else "X", plane if plane in symmetry.PLANES else "fit",
            keep if keep in SIDES else "positive", float(seam_distance), bool(symmetrize_skeleton),
        )
        normal = found["normal"]
        point = found["point"]
        report.publish(
            model, f"Mirrored; the halves differed by {found['error_before'] * 1000:.2f} mm on average",
            before=before,
            counts={
                "faces mirrored": found["mirrored_faces"],
                "seam vertices": found["seam_vertices"],
                "joint pairs": found["joints_by_name"] + found["joints_by_position"],
                "joints moved": found["joints_moved"],
            },
            facts={
                "plane normal": ", ".join(f"{value:.4f}" for value in normal),
                "plane point": ", ".join(f"{value:.4f}" for value in point),
                "asymmetry before": f"{found['error_before'] * 1000:.3f} mm mean",
                "joints paired": f"{found['joints_by_name']} by name, {found['joints_by_position']} by position",
                "shape keys paired": str(found["shape_keys_paired"]),
                "parts removed": str(found["removed_parts"]),
            },
        )
        hidden = cls.hidden
        wanted = inputs.mesh_wanted(getattr(hidden, "prompt", None), getattr(hidden, "unique_id", None))
        return io.NodeOutput(*inputs.node_output(model, mesh=wanted))

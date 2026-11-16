"""Move, turn and scale a whole 3D model, keeping its skeleton, skin, animations and springs in step."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import FILE_3D, FILE_3D_FBX, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_OBJ, MODEL_3D
from ...modules.model3d import inputs, orient, report

#: Where the model's origin goes, against the placement `orient.place` makes.
ORIGINS = ("as is", "base", "centre")

#: Furthest move along one axis, in metres.
MAX_MOVE = 100000.0


class Model3DTransform(io.ComfyNode):
    """Move, rotate and uniformly scale a model, or stand it up, face it forward and size it."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASModel3DTransform",
            display_name="3D Model Transform",
            search_aliases=[
                "WASModel3DTransform", "3D Model Transform", "move model", "rotate model",
                "scale model", "face forward", "resize 3d", "orient",
            ],
            category="WAS Suite/3D/Model",
            description=(
                "Move, rotate and scale a whole model, or turn a rigged character to face forward, "
                "size it to a height and stand it on the floor. The skeleton, skin, animations and "
                "spring bones move with the mesh, so a rigged model still poses correctly."
            ),
            inputs=[
                io.MultiType.Input("model_3d", types=[MODEL_3D, io.Mesh, FILE_3D, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_FBX, FILE_3D_OBJ], tooltip="The model to work on: a 3D model from another 3D node, a mesh from core's mesh nodes, or a GLB, glTF, VRM, FBX or OBJ file from Load 3D or a 3D generator such as Tripo."),
                io.Float.Input(
                    "move_x", default=0.0, min=-MAX_MOVE, max=MAX_MOVE, step=0.01,
                    tooltip="Metres to move along X, the character's left, applied last. 0 = no move.",
                ),
                io.Float.Input(
                    "move_y", default=0.0, min=-MAX_MOVE, max=MAX_MOVE, step=0.01,
                    tooltip="Metres to move along Y, up, applied last. 0 = no move.",
                ),
                io.Float.Input(
                    "move_z", default=0.0, min=-MAX_MOVE, max=MAX_MOVE, step=0.01,
                    tooltip="Metres to move along Z, forward, applied last. 0 = no move.",
                ),
                io.Float.Input(
                    "rotate_x", default=0.0, min=-360.0, max=360.0, step=1.0,
                    tooltip="Degrees to turn about X through the origin; 90 tips the top towards +Z. Turns run X, then Y, then Z.",
                ),
                io.Float.Input(
                    "rotate_y", default=0.0, min=-360.0, max=360.0, step=1.0,
                    tooltip="Degrees to turn about the up axis through the origin; 180 turns a model facing -Z to face +Z.",
                ),
                io.Float.Input(
                    "rotate_z", default=0.0, min=-360.0, max=360.0, step=1.0,
                    tooltip="Degrees to turn about Z through the origin, rolling the model sideways; 90 lays a standing model on its side.",
                ),
                io.Float.Input(
                    "scale", default=1.0, min=0.0001, max=10000.0, step=0.01,
                    tooltip="Uniform size factor about the origin, after the turns. 1.0 = unchanged; 0.01 brings a model made in centimetres to metres.",
                ),
                io.Boolean.Input(
                    "face_forward", default=False,
                    tooltip="`true` = first turn a rigged humanoid about its hips until it faces +Z; `false` = leave its facing. A model without a skeleton is never turned.",
                ),
                io.Float.Input(
                    "scale_to_height", default=0.0, min=0.0, max=1000.0, step=0.01,
                    tooltip="0 = keep the size; 1.7 = after scale, resized evenly about its feet to stand 1.7 metres tall.",
                ),
                io.Combo.Input(
                    "origin", options=list(ORIGINS), default="as is",
                    tooltip="Where the model sits before the move. `as is` = where the turns and scaling left it; `base` = centred, standing at height 0; `centre` = the middle of its box at the origin.",
                ),
            ],
            outputs=[
                MODEL_3D.Output("model_3d", tooltip="The whole model: parts with their materials, and the skeleton, skin, springs, expressions and animations it carries."),
                io.Mesh.Output("mesh", tooltip="The same model as one mesh for core's mesh nodes and Save 3D Model. Parts are merged, their textures packed into one atlas, and the skeleton left out."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model_3d, move_x=0.0, move_y=0.0, move_z=0.0, rotate_x=0.0, rotate_y=0.0,
                rotate_z=0.0, scale=1.0, face_forward=False, scale_to_height=0.0,
                origin="as is") -> io.NodeOutput:
        """Face forward, turn, scale, size to height, place, then move.

        Raises:
            ValueError: Nothing is wired, or the scale is not above 0.
        """
        if not scale > 0:
            raise ValueError("scale must be above 0; use 1.0 to keep the size")
        model = inputs.to_model(model_3d)
        before = report.figures(model)
        counts: dict = {}
        if face_forward:
            counts["turned to face +Z"] = round(orient.face_forward(model), 2)
        if rotate_x or rotate_y or rotate_z or scale != 1.0:
            turn = orient.rotation_matrix(float(rotate_x), float(rotate_y), float(rotate_z))
            orient.transform(model, orient.similarity(turn, float(scale)))
        if scale_to_height > 0:
            counts["height factor"] = round(orient.scale_to_height(model, float(scale_to_height)), 6)
        orient.place(model, origin if origin in ORIGINS else "as is")
        if move_x or move_y or move_z:
            orient.transform(model, orient.similarity(translation=(float(move_x), float(move_y), float(move_z))))
        low, high = model.bounds()
        facts = {
            "low": ", ".join(f"{value:.3f}" for value in low.tolist()),
            "high": ", ".join(f"{value:.3f}" for value in high.tolist()),
        }
        report.publish(model, f"{model.height():.3f} m tall", before=before, counts=counts, facts=facts)
        hidden = cls.hidden
        wanted = inputs.mesh_wanted(getattr(hidden, "prompt", None), getattr(hidden, "unique_id", None))
        return io.NodeOutput(*inputs.node_output(model, mesh=wanted))

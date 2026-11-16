"""Give a standing character a VRM humanoid skeleton and skin weights, without a rigging service."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.character import assemble as character_assemble
from ...modules.character import rig as character_rig
from ...modules.compat.types import FILE_3D, FILE_3D_FBX, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_OBJ, MODEL_3D
from ...modules.model3d import edit, inputs, report


class CharacterRig(io.ComfyNode):
    """Fit a VRM humanoid skeleton to a character and skin it."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASCharacterRig",
            display_name="Character Rig",
            search_aliases=[
                "WASCharacterRig", "Character Rig", "auto rig", "rigging", "skeleton", "skin weights",
                "humanoid", "vrm", "t-pose",
            ],
            category="WAS Suite/3D/Character",
            description=(
                "Rig a standing character with a VRM humanoid skeleton and skin weights, ready to save as a "
                "VRM avatar or pose. Joints, fingers and eyes are fitted to the body's own shape, and the arms "
                "can be turned level into the T-pose VRM expects."
            ),
            inputs=[
                io.MultiType.Input("model_3d", types=[MODEL_3D, io.Mesh, FILE_3D, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_FBX, FILE_3D_OBJ], tooltip="The character to rig: a full figure standing upright, arms held away from the body, legs apart. A skeleton it already has is replaced."),
                io.Combo.Input(
                    "rest_pose", options=list(character_rig.REST_POSES), default="T-pose",
                    tooltip="`T-pose` = turn sloping arms level, as VRM avatars expect; `as modelled` = keep the "
                            "pose the mesh was made in.",
                ),
                io.Int.Input(
                    "detail", default=character_rig.PROXY_TRIANGLES, min=5000, max=300000, step=5000,
                    advanced=True,
                    tooltip="Triangles of the reduced copy the body's joints are fitted to and its weights are "
                            "worked out on. 60000 suits most characters; more follows thin parts closer and takes "
                            "longer. Fingers are always fitted on the full model.",
                ),
                io.Combo.Input(
                    "facing", options=list(character_assemble.FACINGS), default="auto", advanced=True,
                    tooltip="Which way the character faces as loaded. `auto` = as Character Assemble left it, else "
                            "read from a skeleton or the feet; `+Z`, `-Z`, `+X`, `-X` = that direction. The "
                            "character always ends facing +Z.",
                ),
                io.Boolean.Input(
                    "fingers", default=True,
                    tooltip="`true` = fit thumb and finger bones to each hand and share the hand's weights with "
                            "them, so the fingers can curl; joined or mitten fingers get evenly spaced bones. "
                            "`false` = the hand moves as one piece.",
                ),
                io.Boolean.Input(
                    "eyes", default=True,
                    tooltip="`true` = place eye bones and a VRM look-at. Eyeball parts turn with the bones; "
                            "painted or sculpted eyes stay still and the bones set the gaze origin. `false` = no "
                            "eye bones.",
                ),
            ],
            outputs=[
                MODEL_3D.Output("model_3d", tooltip="The whole model: parts with their materials, and the skeleton, skin, springs, expressions and animations it carries."),
                io.Mesh.Output("mesh", tooltip="The same model as one mesh for core's mesh nodes and Save 3D Model. Parts are merged, their textures packed into one atlas, and the skeleton left out."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model_3d, rest_pose="T-pose", detail=character_rig.PROXY_TRIANGLES,
                facing="auto", fingers=True, eyes=True) -> io.NodeOutput:
        """Rig the model and publish the fit.

        Raises:
            ValueError: Nothing is wired, or the model does not read as a standing figure.
        """
        model = inputs.to_model(model_3d)
        before = report.figures(model)
        model, figures = character_rig.rig(
            model, rest_pose=rest_pose if rest_pose in character_rig.REST_POSES else "T-pose",
            proxy_triangles=int(detail),
            facing=facing if facing in character_assemble.FACINGS else "auto",
            fingers=bool(fingers), eyes=bool(eyes), blink=False,
        )
        summary = f"{figures['joints']} joints"
        if figures["finger bones"]:
            summary += f", {figures['finger bones']} finger bones"
        if figures["eye bones"]:
            summary += ", eyes"
        if figures["arms levelled by"]:
            summary += f", arms levelled by {figures['arms levelled by']:.0f} degrees"
        facts = {"facing": figures["facing"], **figures["notes"], "parts": edit.part_names(model, 8)}
        report.publish(
            model, summary, before=before,
            counts={"finger bones": figures["finger bones"], "eye bones": figures["eye bones"],
                    "turned to face +Z": figures["turned to face +Z"], "arms levelled by": figures["arms levelled by"],
                    "proxy triangles": figures["proxy triangles"]},
            facts=facts,
        )
        hidden = cls.hidden
        wanted_mesh = inputs.mesh_wanted(getattr(hidden, "prompt", None), getattr(hidden, "unique_id", None))
        return io.NodeOutput(*inputs.node_output(model, mesh=wanted_mesh))

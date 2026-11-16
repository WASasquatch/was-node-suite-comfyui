"""Give a rigged character's hanging hair spring bones, so it swings in VRM viewers and games."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.character import springs as character_springs
from ...modules.compat.types import FILE_3D, FILE_3D_FBX, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_OBJ, MODEL_3D
from ...modules.model3d import edit, inputs, report


class CharacterSprings(io.ComfyNode):
    """Add VRM spring bones to a rigged character's hanging hair."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASCharacterSprings",
            display_name="Character Springs",
            search_aliases=[
                "WASCharacterSprings", "Character Springs", "spring bones", "hair physics", "secondary motion",
                "vrm springs", "hair bones", "jiggle", "dynamic bones",
            ],
            category="WAS Suite/3D/Character",
            description=(
                "Give a rigged character's hanging hair, such as a ponytail, braids or long locks, and any skirt or "
                "cape part chains of spring bones, with colliders on the body, so they swing and settle in VRM "
                "viewers and games. Hair on the scalp keeps moving with the head."
            ),
            inputs=[
                io.MultiType.Input("model_3d", types=[MODEL_3D, io.Mesh, FILE_3D, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_FBX, FILE_3D_OBJ], tooltip="The rigged character: a humanoid skeleton and skin, from Character Rig or a VRM. Hair below the head joint is what swings."),
                io.Combo.Input(
                    "hair", options=list(character_springs.HAIR_SOURCES), default="auto",
                    tooltip="`auto` = hair found by colour on the head, plus any part labelled `hair`; `hair parts` "
                            "= only parts labelled `hair`; `off` = no chains, colliders only.",
                ),
                io.Int.Input(
                    "joints_per_chain", default=4, min=2, max=12, step=1,
                    tooltip="Segments in each strand's chain. 4 = smooth swing; 8 for long hair that curls as it "
                            "moves.",
                ),
                io.Int.Input(
                    "max_chains", default=16, min=1, max=64, step=1,
                    tooltip="Most strands given chains; the largest are kept. 16 suits braids and a ponytail.",
                ),
                io.Float.Input(
                    "stiffness", default=0.8, min=0.0, max=4.0, step=0.05,
                    tooltip="Pull back to the rest shape. 0.8 = soft hair; 2.0 = stiff, as for a tight braid.",
                ),
                io.Float.Input(
                    "drag", default=0.4, min=0.0, max=1.0, step=0.05,
                    tooltip="Share of its speed a strand loses each step. 0.4 = settles quickly; 0.1 = keeps "
                            "swinging.",
                ),
                io.Float.Input(
                    "gravity", default=0.25, min=0.0, max=2.0, step=0.05,
                    tooltip="Strength of gravity on each strand. 0 = hangs as modelled; 0.5 = falls noticeably "
                            "when the head tilts.",
                ),
                io.Boolean.Input(
                    "colliders", default=True,
                    tooltip="`true` = colliders on the head, neck, chest, upper arms, hips and upper legs, so hair and "
                            "cloth rest on the body instead of passing through it.",
                ),
                io.String.Input(
                    "cloth", default="skirt, cape",
                    tooltip="Parts that swing like cloth, by name or label, wildcards allowed, such as `skirt, cape`. "
                            "Each gets chains round the torso bone it hangs from. Empty = none.",
                ),
            ],
            outputs=[
                MODEL_3D.Output("model_3d", tooltip="The whole model: parts with their materials, and the skeleton, skin, springs, expressions and animations it carries."),
                io.Mesh.Output("mesh", tooltip="The same model as one mesh for core's mesh nodes and Save 3D Model. Parts are merged, their textures packed into one atlas, and the skeleton left out."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model_3d, hair="auto", joints_per_chain=4, max_chains=16, stiffness=0.8, drag=0.4,
                gravity=0.25, colliders=True, cloth="skirt, cape") -> io.NodeOutput:
        """Add the chains and colliders and publish what was made.

        Raises:
            ValueError: The model has no humanoid skeleton or skin, or already has hair chains.
        """
        model = inputs.to_model(model_3d)
        before = report.figures(model)
        model, figures = character_springs.add_springs(
            model, hair=hair if hair in character_springs.HAIR_SOURCES else "auto",
            joints_per_chain=int(joints_per_chain), max_chains=int(max_chains), stiffness=float(stiffness),
            drag=float(drag), gravity=float(gravity), colliders=bool(colliders), cloth=str(cloth or ""))
        summary = f"{figures['chains']} chain{'s' if figures['chains'] != 1 else ''}"
        if figures["colliders"]:
            summary += f", {figures['colliders']} colliders"
        facts = {"parts": edit.part_names(model, 8)}
        if figures.get("collider bones"):
            facts["colliders on"] = figures["collider bones"]
        if figures["notes"]:
            facts["notes"] = "; ".join(figures["notes"])
        report.publish(
            model, summary, before=before,
            counts={"chains": figures["chains"], "joints added": figures["joints added"],
                    "swinging vertices": figures["swinging vertices"], "colliders": figures["colliders"]},
            facts=facts,
        )
        hidden = cls.hidden
        wanted_mesh = inputs.mesh_wanted(getattr(hidden, "prompt", None), getattr(hidden, "unique_id", None))
        return io.NodeOutput(*inputs.node_output(model, mesh=wanted_mesh))

"""Join a character's head and body at the neck into one continuous surface."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.character import neck_join as character_neck_join
from ...modules.compat.types import FILE_3D, FILE_3D_FBX, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_OBJ, MODEL_3D
from ...modules.model3d import edit, inputs, report


class CharacterNeckJoin(io.ComfyNode):
    """Share one ring of vertices between a character's head and body at the neck."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASCharacterNeckJoin",
            display_name="Character Neck Join",
            search_aliases=[
                "WASCharacterNeckJoin", "Character Neck Join", "neck seam", "weld neck", "join head and body",
                "seamless neck", "neck crack", "stitch", "vrm",
            ],
            category="WAS Suite/3D/Character",
            description=(
                "Join a head that Character Assemble set on a body into one continuous surface at the neck, so the "
                "seam cannot open, crease or crack in a render or an animation. Both necks share one ring of "
                "vertices with one shading, and one set of skin weights when the model is rigged."
            ),
            inputs=[
                io.MultiType.Input(
                    "model_3d", types=[MODEL_3D, io.Mesh, FILE_3D, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_FBX, FILE_3D_OBJ],
                    tooltip="The character from Character Assemble, rigged or not. After Character Rig it also "
                            "matches the skin weights across the seam, so place it after the rig; a second pass keeps "
                            "the shape and matches the weights again.",
                ),
                io.Float.Input(
                    "band", default=character_neck_join.BAND, min=0.0, max=character_neck_join.MAX_BAND, step=0.05,
                    tooltip="Height above and below the seam, in neck radii, over which shape, shading and skin "
                            "weights ease into the shared ring. 0 = the ring only; 0.5 = a soft blend; 1.5 for necks "
                            "that met at an angle.",
                ),
                io.Boolean.Input(
                    "merge_parts", default=False,
                    tooltip="`true` = head and body parts drawn with the same material become one part; `false` = "
                            "keep them apart. Merge only after Character Rig and Character Springs, which find the "
                            "head by its label.",
                ),
                io.String.Input(
                    "head_parts", default="head", advanced=True,
                    tooltip="Parts above the seam, by label or name, wildcards allowed, such as `head` or "
                            "`tripo_node_0a7f*`. `head` = the parts Character Assemble labelled head.",
                ),
                io.String.Input(
                    "body_parts", default="body", advanced=True,
                    tooltip="Parts below the seam, by label or name, wildcards allowed, such as `body` or "
                            "`tripo_node_dd15*`. `body` = the parts Character Assemble labelled body.",
                ),
            ],
            outputs=[
                MODEL_3D.Output("model_3d", tooltip="The whole model: parts with their materials, and the skeleton, skin, springs, expressions and animations it carries."),
                io.Mesh.Output("mesh", tooltip="The same model as one mesh for core's mesh nodes and Save 3D Model. Parts are merged, their textures packed into one atlas, and the skeleton left out."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model_3d, band=character_neck_join.BAND, merge_parts=False, head_parts="head",
                body_parts="body") -> io.NodeOutput:
        """Join the neck and publish the seam's figures.

        Raises:
            ValueError: Nothing is wired, no part matches, or the head has no cut neck beside the body's.
        """
        model = inputs.to_model(model_3d)
        before = report.figures(model)
        model, figures = character_neck_join.join_neck(
            model, head_parts=str(head_parts or ""), body_parts=str(body_parts or ""), band=float(band),
            merge_parts=bool(merge_parts))
        summary = (f"{figures['shared vertices']} shared vertices round {figures['ring joined degrees']:.0f} degrees "
                   f"of the neck")
        counts = {
            "shared vertices": figures["shared vertices"],
            "vertices added": figures["head vertices added"] + figures["body vertices added"],
            "gap closed, mm": figures["gap closed max mm"],
            "shading step closed, degrees": figures["normal step closed degrees"],
            "ring joined, degrees": figures["ring joined degrees"],
            "shape eased, mm": figures["shape eased max mm"],
            "neck radius, mm": round(figures["neck radius m"] * 1000, 1),
        }
        if "weight step closed" in figures:
            counts["weight step closed"] = figures["weight step closed"]
        if figures["faces collapsed"]:
            counts["folded faces removed"] = figures["faces collapsed"]
        if figures["parts merged"]:
            counts["parts merged"] = figures["parts merged"]
        facts = {"seam": f"{figures['seam height m']:.3f} m up", "skin weights": figures["weights"],
                 "neck cap": figures["cap"], "parts": edit.part_names(model, 8)}
        if figures["notes"]:
            facts["notes"] = "; ".join(figures["notes"])
        report.publish(model, summary, before=before, counts=counts, facts=facts)
        hidden = cls.hidden
        wanted_mesh = inputs.mesh_wanted(getattr(hidden, "prompt", None), getattr(hidden, "unique_id", None))
        return io.NodeOutput(*inputs.node_output(model, mesh=wanted_mesh))

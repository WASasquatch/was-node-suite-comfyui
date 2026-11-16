"""Choose, combine, divide, rename and label the parts of a 3D model, or add another model's."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import FILE_3D, FILE_3D_FBX, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_OBJ, MODEL_3D
from ...modules.log import get_logger
from ...modules.model3d import edit, inputs, merge, report, selection

logger = get_logger("nodes.model3d")


class Model3DParts(io.ComfyNode):
    """Keep, drop, merge, split, rename and label a model's parts, and add a second model's."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASModel3DParts",
            display_name="3D Model Parts",
            search_aliases=[
                "WASModel3DParts", "3D Model Parts", "split mesh", "separate parts", "merge meshes",
                "combine models", "rename parts", "label parts", "remove parts", "loose parts",
            ],
            category="WAS Suite/3D/Model",
            description=(
                "Edit which parts a model is made of: keep or drop parts by name, merge parts that "
                "share a material, split parts into their separate pieces, rename and label them, "
                "and add the parts of a second model, joining skeletons by joint name."
            ),
            inputs=[
                io.MultiType.Input("model_3d", types=[MODEL_3D, io.Mesh, FILE_3D, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_FBX, FILE_3D_OBJ], tooltip="The model to work on: a 3D model from another 3D node, a mesh from core's mesh nodes, or a GLB, glTF, VRM, FBX or OBJ file from Load 3D or a 3D generator such as Tripo."),
                io.String.Input(
                    "keep", default="",
                    tooltip="Parts of model_3d to keep: comma-separated names or labels, wildcards allowed, such as `body, head*`. Empty keeps every part.",
                ),
                io.String.Input(
                    "drop", default="",
                    tooltip="Parts of model_3d to remove after keep, written the same way, such as `*eyelash*, cape`. Empty removes none.",
                ),
                io.Boolean.Input(
                    "merge_by_material", default=False,
                    tooltip="`true` = parts drawn with the same material become one part, skinned and static parts kept apart; `false` = parts stay as they are.",
                ),
                io.Boolean.Input(
                    "split_shells", default=False,
                    tooltip="`true` = each part becomes one part per separate piece, largest first, named name_1, name_2; `false` = no split. Runs after the merge.",
                ),
                io.String.Input(
                    "rename", default="",
                    tooltip="pattern = new name pairs, comma-separated, such as `tripo_node_* = body`. Several matches are numbered body, body_2. Runs after the split.",
                ),
                io.String.Input(
                    "labels", default="",
                    tooltip="pattern = label pairs giving parts a role the character nodes read, such as `*hair* = hair, body = body`. Runs last.",
                ),
                io.String.Input(
                    "parts", default="", advanced=True,
                    tooltip="Parts to work on: comma-separated names or labels, wildcards allowed, such as `head, hair*`. Empty works on every part.",
                ),
                io.Int.Input(
                    "max_shells", default=256, min=2, max=merge.MAX_SHELLS, advanced=True,
                    tooltip="Most parts one part splits into. 256 = the 255 largest pieces apart and every smaller one together in name_rest.",
                ),
                io.MultiType.Input(
                    "add_model", types=[MODEL_3D, io.Mesh, FILE_3D, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_FBX, FILE_3D_OBJ], optional=True,
                    tooltip="A second model whose parts are added after keep and drop. Both rigged: joints of the same name become one. Only this one rigged: its skeleton is taken. Otherwise its parts stay static.",
                ),
            ],
            outputs=[
                MODEL_3D.Output("model_3d", tooltip="The whole model: parts with their materials, and the skeleton, skin, springs, expressions and animations it carries."),
                io.Mesh.Output("mesh", tooltip="The same model as one mesh for core's mesh nodes and Save 3D Model. Parts are merged, their textures packed into one atlas, and the skeleton left out."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model_3d, keep="", drop="", merge_by_material=False, split_shells=False, rename="",
                labels="", parts="", max_shells=256, add_model=None) -> io.NodeOutput:
        """Apply keep, drop, add, merge, split, rename and labels, in that order.

        Raises:
            ValueError: A pattern leaves no part, parts selects none for merge or split, or a
                rename or label pair is malformed.
        """
        model = inputs.to_model(model_3d)
        before = report.figures(model)
        counts: dict = {}
        facts: dict = {}
        removed = edit.filter_parts(model, keep, drop)
        if removed:
            counts["parts removed"] = removed
        if add_model is not None:
            added = merge.add_model(model, inputs.to_model(add_model))
            counts.update({key: value for key, value in added.items() if value})
        if merge_by_material or split_shells:
            chosen = selection.selected(model, parts)
            if not chosen:
                raise ValueError(f"parts '{str(parts)[:80]}' selects none of the model's parts "
                                 f"({edit.part_names(model)}); use a name or label from that list, or "
                                 "leave parts empty to work on every part")
            if merge_by_material:
                joined = merge.merge_by_material(model, chosen)
                counts["parts merged away"] = joined
            if split_shells:
                targets = {id(model.parts[index]) for index in selection.selected(model, parts)}
                pieces, source = [], []
                split = 0
                for index, part in enumerate(model.parts):
                    divided = merge.split_shells(part, max_shells) if id(part) in targets else [part]
                    split += len(divided) - 1
                    pieces += divided
                    source += [[index]] * len(divided)
                if split:
                    merge.reindex_parts(model, pieces, source)
                counts["parts split off"] = split
        renamed, unmatched = edit.rename_parts(model, rename)
        labelled, unlabelled = edit.label_parts(model, labels)
        if renamed:
            counts["parts renamed"] = renamed
        if labelled:
            counts["parts labelled"] = labelled
        missed = unmatched + unlabelled
        if missed:
            facts["matched nothing"] = ", ".join(missed)[:200]
            logger.warning("3D Model Parts: %d rename or label pattern(s) matched no part; check the "
                           "names on the node's report", len(missed))
        facts["parts"] = edit.part_names(model, 8)
        report.publish(model, f"{len(model.parts)} part(s)", before=before, counts=counts, facts=facts)
        hidden = cls.hidden
        wanted = inputs.mesh_wanted(getattr(hidden, "prompt", None), getattr(hidden, "unique_id", None))
        return io.NodeOutput(*inputs.node_output(model, mesh=wanted))

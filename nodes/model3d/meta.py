"""Set the avatar metadata a VRM file carries: title, authors, licence terms and thumbnail."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import FILE_3D, FILE_3D_FBX, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_OBJ, MODEL_3D
from ...modules.model3d import inputs, meta, report

#: Tooltip ending shared by the yes-or-no permissions.
FLAG_HINT = "`keep` = the model's own setting, `false` when it has none."


class Model3DMeta(io.ComfyNode):
    """Write VRM 1.0 avatar metadata onto a model."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASModel3DMeta",
            display_name="3D Model Meta",
            search_aliases=[
                "WASModel3DMeta", "3D Model Meta", "vrm meta", "avatar licence", "avatar license",
                "author", "thumbnail", "copyright",
            ],
            category="WAS Suite/3D/Model",
            description=(
                "Set the avatar details a VRM file carries: its title, authors, licence terms and "
                "thumbnail. Empty text and keep leave what the model already has. A GLB or glTF "
                "save carries the same fields, without the thumbnail, in its file information."
            ),
            inputs=[
                io.MultiType.Input("model_3d", types=[MODEL_3D, io.Mesh, FILE_3D, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_FBX, FILE_3D_OBJ], tooltip="The model to work on: a 3D model from another 3D node, a mesh from core's mesh nodes, or a GLB, glTF, VRM, FBX or OBJ file from Load 3D or a 3D generator such as Tripo."),
                io.String.Input(
                    "name", default="",
                    tooltip="The avatar's title, such as `Drow Ranger`. Empty keeps the model's own.",
                ),
                io.String.Input(
                    "authors", default="",
                    tooltip="Who made it, comma-separated, such as `Ada, Grace`. Empty keeps the model's own; VRM needs at least one.",
                ),
                io.String.Input(
                    "version", default="",
                    tooltip="The avatar's version, such as `1.0`. Empty keeps the model's own.",
                ),
                io.Combo.Input(
                    "avatar_permission", options=list(meta.AVATAR_PERMISSIONS), default="keep",
                    tooltip=(
                        "Who may act as this avatar. `onlyAuthor` = the authors alone; "
                        "`onlySeparatelyLicensedPerson` = people the authors license; `everyone` = "
                        "anyone. `keep` = the model's own, onlyAuthor when it has none."
                    ),
                ),
                io.Combo.Input(
                    "commercial_usage", options=list(meta.COMMERCIAL_USAGES), default="keep",
                    tooltip=(
                        "Commercial use allowed. `personalNonProfit` = personal, no profit; "
                        "`personalProfit` = individuals may profit; `corporation` = companies too. "
                        "`keep` = the model's own, personalNonProfit when it has none."
                    ),
                ),
                io.Combo.Input(
                    "credit_notation", options=list(meta.CREDIT_NOTATIONS), default="keep",
                    tooltip="`required` = users must credit the authors; `unnecessary` = no credit needed. `keep` = the model's own, required when it has none.",
                ),
                io.Combo.Input(
                    "modification", options=list(meta.MODIFICATIONS), default="keep",
                    tooltip=(
                        "`prohibited` = no edits; `allowModification` = edits for own use; "
                        "`allowModificationRedistribution` = edited copies may be shared. `keep` = the "
                        "model's own, prohibited when it has none."
                    ),
                ),
                io.Combo.Input(
                    "allow_redistribution", options=list(meta.FLAG_OPTIONS), default="keep",
                    tooltip=f"`true` = others may share the unmodified file; `false` = they may not. {FLAG_HINT}",
                ),
                io.String.Input(
                    "copyright_information", default="", advanced=True,
                    tooltip="Copyright line, such as `(c) 2026 Ada`. Empty keeps the model's own.",
                ),
                io.String.Input(
                    "contact_information", default="", advanced=True,
                    tooltip="How to reach the authors, such as `ada@example.com` or a profile page. Empty keeps the model's own.",
                ),
                io.String.Input(
                    "references", default="", advanced=True,
                    tooltip="Works the avatar is based on, comma-separated, such as `https://example.com/sheet`. Empty keeps the model's own.",
                ),
                io.String.Input(
                    "third_party_licenses", default="", advanced=True,
                    tooltip="Licences of third-party material inside the avatar, such as `Font: SIL OFL 1.1`. Empty keeps the model's own.",
                ),
                io.String.Input(
                    "other_license_url", default="", advanced=True,
                    tooltip="The address of a further licence document that applies, such as `https://example.com/licence`. Empty keeps the model's own.",
                ),
                io.Combo.Input(
                    "allow_excessively_violent_usage", options=list(meta.FLAG_OPTIONS), default="keep", advanced=True,
                    tooltip=f"`true` = the avatar may appear in excessively violent content. {FLAG_HINT}",
                ),
                io.Combo.Input(
                    "allow_excessively_sexual_usage", options=list(meta.FLAG_OPTIONS), default="keep", advanced=True,
                    tooltip=f"`true` = the avatar may appear in excessively sexual content. {FLAG_HINT}",
                ),
                io.Combo.Input(
                    "allow_political_or_religious_usage", options=list(meta.FLAG_OPTIONS), default="keep", advanced=True,
                    tooltip=f"`true` = the avatar may be used for political or religious messages. {FLAG_HINT}",
                ),
                io.Combo.Input(
                    "allow_antisocial_or_hate_usage", options=list(meta.FLAG_OPTIONS), default="keep", advanced=True,
                    tooltip=f"`true` = the avatar may be used in antisocial or hateful content. {FLAG_HINT}",
                ),
                io.Image.Input(
                    "thumbnail", optional=True,
                    tooltip="A picture shown for the avatar in VRM apps, ideally a square portrait. Reduced to 1024 pixels on its longest side; unwired keeps the model's own.",
                ),
            ],
            outputs=[
                MODEL_3D.Output("model_3d", tooltip="The whole model: parts with their materials, and the skeleton, skin, springs, expressions and animations it carries."),
                io.Mesh.Output("mesh", tooltip="The same model as one mesh for core's mesh nodes and Save 3D Model. Parts are merged, their textures packed into one atlas, and the skeleton left out."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model_3d, name="", authors="", version="", avatar_permission="keep",
                commercial_usage="keep", credit_notation="keep", modification="keep",
                allow_redistribution="keep", copyright_information="", contact_information="",
                references="", third_party_licenses="", other_license_url="",
                allow_excessively_violent_usage="keep", allow_excessively_sexual_usage="keep",
                allow_political_or_religious_usage="keep", allow_antisocial_or_hate_usage="keep",
                thumbnail=None) -> io.NodeOutput:
        """Write the fields onto a copy of the model.

        Raises:
            ValueError: Nothing is wired, or the thumbnail is not a picture.
        """
        model = inputs.to_model(model_3d)
        written = meta.apply(
            model,
            texts={
                "name": name, "version": version, "copyrightInformation": copyright_information,
                "contactInformation": contact_information, "thirdPartyLicenses": third_party_licenses,
                "otherLicenseUrl": other_license_url,
            },
            lists={"authors": authors, "references": references},
            choices={
                "avatarPermission": avatar_permission, "commercialUsage": commercial_usage,
                "creditNotation": credit_notation, "modification": modification,
            },
            flags={
                "allowRedistribution": allow_redistribution,
                "allowExcessivelyViolentUsage": allow_excessively_violent_usage,
                "allowExcessivelySexualUsage": allow_excessively_sexual_usage,
                "allowPoliticalOrReligiousUsage": allow_political_or_religious_usage,
                "allowAntisocialOrHateUsage": allow_antisocial_or_hate_usage,
            },
        )
        if thumbnail is not None:
            model.extras["vrm_thumbnail"] = meta.thumbnail(thumbnail)
            written.append("thumbnail")
        facts = {key: _shown(model.meta[key]) for key in (
            "name", "authors", "avatarPermission", "commercialUsage", "creditNotation", "modification",
        ) if key in model.meta}
        held = model.extras.get("vrm_thumbnail")
        if held is not None and hasattr(held, "size"):
            facts["thumbnail"] = "{} x {}".format(*held.size)
        report.publish(model, f"{len(written)} meta field(s) written", counts={"fields written": len(written)},
                       facts=facts)
        hidden = cls.hidden
        wanted = inputs.mesh_wanted(getattr(hidden, "prompt", None), getattr(hidden, "unique_id", None))
        return io.NodeOutput(*inputs.node_output(model, mesh=wanted))


def _shown(value) -> str:
    """A meta value as one short line."""
    text = ", ".join(str(item) for item in value) if isinstance(value, list) else str(value)
    return text if len(text) <= 120 else text[:117] + "..."

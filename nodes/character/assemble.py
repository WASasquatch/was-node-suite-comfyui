"""Put a separately generated head, hair and an extra piece on a body, as one character model."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.character import assemble as character_assemble
from ...modules.compat.types import FILE_3D, FILE_3D_FBX, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_OBJ, MODEL_3D
from ...modules.model3d import edit, inputs, report


class CharacterAssemble(io.ComfyNode):
    """Join a body and a head made apart into one model, the head fitted to the neck."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASCharacterAssemble",
            display_name="Character Assemble",
            search_aliases=[
                "WASCharacterAssemble", "Character Assemble", "assemble character", "head on body",
                "attach head", "combine head and body", "avatar", "vrm", "tripo",
            ],
            category="WAS Suite/3D/Character",
            description=(
                "Put a head made separately on a body, with optional hair and one extra piece, as one model "
                "whose parts are labelled body, head and hair. The body is turned to face forward and stood "
                "on the floor, and the head is fitted over the body's own head and its neck joined into the "
                "body's."
            ),
            inputs=[
                io.MultiType.Input(
                    "body", types=[MODEL_3D, io.Mesh, FILE_3D, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_FBX, FILE_3D_OBJ],
                    tooltip="The body: a full figure, or one without a head. A character already assembled may "
                            "be wired again to add more parts.",
                ),
                io.MultiType.Input(
                    "head", types=[MODEL_3D, io.Mesh, FILE_3D, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_FBX, FILE_3D_OBJ],
                    optional=True,
                    tooltip="A head made on its own, ending at the neck. Its size and position are fitted to the "
                            "body's neck.",
                ),
                io.MultiType.Input(
                    "hair", types=[MODEL_3D, io.Mesh, FILE_3D, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_FBX, FILE_3D_OBJ],
                    optional=True,
                    tooltip="Hair made together with the head, in the head's own position. Labelled hair.",
                ),
                io.MultiType.Input(
                    "extra", types=[MODEL_3D, io.Mesh, FILE_3D, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_FBX, FILE_3D_OBJ],
                    optional=True,
                    tooltip="One more piece, such as a hat or a tail, labelled with extra_label and moved with the "
                            "head or the body as extra_follows says.",
                ),
                io.Float.Input(
                    "target_height", default=1.6, min=0.0, max=100.0, step=0.01,
                    tooltip="Height of the finished character in metres. 1.6 = an adult; 0 = keep the body's "
                            "own size.",
                ),
                io.Combo.Input(
                    "facing", options=list(character_assemble.FACINGS), default="auto",
                    tooltip="Which way the body faces as loaded. `auto` = read from a skeleton, or from the feet; "
                            "`+Z`, `-Z`, `+X`, `-X` = that direction. The character always ends facing +Z.",
                ),
                io.Combo.Input(
                    "remove_body_head", options=list(character_assemble.REMOVE_HEAD), default="auto",
                    tooltip="Cut away a head the body already has, at its neck, when a head is wired. `auto` = "
                            "only when the body has one; `yes` = always cut at the neck; `no` = never.",
                ),
                io.Combo.Input(
                    "clear_body_hair", options=list(character_assemble.CLEAR_HAIR), default="auto",
                    tooltip="Remove the body's own hair hanging below the neck when its head is cut away, matched "
                            "by the colour of its crown, and close the opening. `auto` = do it; `off` = keep it.",
                ),
                io.Combo.Input(
                    "head_fit", options=list(character_assemble.HEAD_FITS), default="auto",
                    tooltip="`auto` = fitted over the body's own head, or sized by the neck when the body has "
                            "none; `body's head` and `neck width` = always that; `keep` = move the head as the body "
                            "moved, for parts made together.",
                ),
                io.Float.Input(
                    "head_scale", default=1.0, min=0.1, max=10.0, step=0.01,
                    tooltip="Multiplies the fitted head size. 1.0 = as measured by `head_fit`; 0.9 = a tenth "
                            "smaller.",
                ),
                io.Float.Input(
                    "head_turn", default=0.0, min=-180.0, max=180.0, step=1.0,
                    tooltip="Degrees the head turns about its up axis before fitting, for a head made facing a "
                            "different way from the body. 90 = a quarter turn to the left.",
                ),
                io.Boolean.Input(
                    "hair_follows_head", default=True,
                    tooltip="`true` = hair moves, turns and scales with the head; `false` = with the body.",
                ),
                io.Boolean.Input(
                    "feet_on_ground", default=True,
                    tooltip="`true` = stand the character on the floor at Y = 0, centred; `false` = keep where it is.",
                ),
                io.String.Input(
                    "extra_label", default="extra", advanced=True,
                    tooltip="Label of the extra piece's parts, such as `hat`, `glasses` or `tail`.",
                ),
                io.Combo.Input(
                    "extra_follows", options=list(character_assemble.EXTRA_FOLLOWS), default="head", advanced=True,
                    tooltip="`head` = the extra moves with the head, like a hat; `body` = with the body, like a tail.",
                ),
                io.String.Input(
                    "placement", default="", advanced=True,
                    tooltip="Fine adjustments after the automatic fit, per part, as JSON: "
                            '{"head": {"offset": [0, 0.01, 0], "rotate": [0, 10, 0], "scale": 1.05}}. Offsets in '
                            "metres, rotations in degrees. Empty = none.",
                ),
                io.Combo.Input(
                    "skin_match", options=list(character_assemble.SKIN_MATCHES), default="head to body",
                    tooltip="`head to body` = tone the new head's skin to the body's, leaving hair, eyes and brows, "
                            "and blend the two necks' colours at the seam; `off` = keep both skins as generated.",
                ),
            ],
            outputs=[
                MODEL_3D.Output("model_3d", tooltip="The whole model: parts with their materials, and the skeleton, skin, springs, expressions and animations it carries."),
                io.Mesh.Output("mesh", tooltip="The same model as one mesh for core's mesh nodes and Save 3D Model. Parts are merged, their textures packed into one atlas, and the skeleton left out."),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, body, head=None, hair=None, extra=None, target_height=1.6, facing="auto",
                remove_body_head="auto", clear_body_hair="auto", head_fit="auto", head_scale=1.0, head_turn=0.0, hair_follows_head=True,
                feet_on_ground=True, extra_label="extra", extra_follows="head", placement="",
                skin_match="head to body") -> io.NodeOutput:
        """Assemble the wired pieces and publish the fit.

        Raises:
            ValueError: A piece has no triangles, no neck is found, or placement is invalid.
        """
        model, fit = character_assemble.assemble(
            inputs.to_model(body),
            head=inputs.to_model(head) if head is not None else None,
            hair=inputs.to_model(hair) if hair is not None else None,
            extra=inputs.to_model(extra) if extra is not None else None,
            extra_label=extra_label, extra_follows=extra_follows, target_height=float(target_height),
            facing=facing if facing in character_assemble.FACINGS else "auto",
            remove_body_head=remove_body_head if remove_body_head in character_assemble.REMOVE_HEAD else "auto",
            clear_body_hair=clear_body_hair if clear_body_hair in character_assemble.CLEAR_HAIR else "auto",
            head_fit=head_fit if head_fit in character_assemble.HEAD_FITS else "auto",
            head_scale=float(head_scale), head_turn=float(head_turn), hair_follows_head=bool(hair_follows_head),
            feet_on_ground=bool(feet_on_ground), placement=placement,
            skin_match=skin_match if skin_match in character_assemble.SKIN_MATCHES else "head to body",
        )
        summary = f"{fit['parts']} parts, {fit['height_m']:.2f} m tall"
        if head is not None:
            summary += f"; head x{fit.get('head_scale', 1.0):.2f} on the {fit.get('neck_found', 'neck')}"
        counts = {}
        if "head_scale" in fit:
            counts["head scale"] = fit["head_scale"]
        if "head fit error" in fit:
            counts["fit error, share of head"] = fit["head fit error"]
        if "skin toned by" in fit:
            counts["skin toned by, Lab"] = fit["skin toned by"]
        if "seam toned by" in fit:
            counts["neck seam toned by, Lab"] = fit["seam toned by"]
        facts ={"facing": fit.get("facing", ""), "parts": edit.part_names(model, 8)}
        if "neck_found" in fit:
            facts["neck"] = fit["neck_found"]
        if "head sized by" in fit:
            facts["head placed by"] = fit["head sized by"]
        if fit.get("head joined"):
            facts["neck"] = f"{fit.get('neck_found', 'neck')}, joined"
            counts["seam below the ring, radii"] = fit.get("seam below the ring", 0.0)
            counts["head neck eased by"] = fit["neck eased by"]
            counts["body neck eased by"] = fit.get("body neck eased by", 0.0)
            counts["hair faces kept whole"] = fit.get("head hair faces", 0)
        if "hair faces removed" in fit:
            counts["hair faces removed"] = fit["hair faces removed"]
        if "body faces kept above the seam" in fit:
            counts["body faces kept above the seam"] = fit["body faces kept above the seam"]
        if fit.get("head joined"):
            counts["skin faces cut below the seam"] = fit.get("head faces trimmed", 0)
        if "neck_radius" in fit:
            counts["neck radius m"] = fit["neck_radius"]
        counts["turned to face +Z"] = fit.get("body_turn", 0.0)
        if fit.get("body head removed"):
            facts["body head"] = "cut away at the neck"
        if fit["notes"]:
            facts["notes"] = "; ".join(fit["notes"])
        report.publish(model, summary, counts=counts, facts=facts)
        hidden = cls.hidden
        wanted_mesh = inputs.mesh_wanted(getattr(hidden, "prompt", None), getattr(hidden, "unique_id", None))
        return io.NodeOutput(*inputs.node_output(model, mesh=wanted_mesh))

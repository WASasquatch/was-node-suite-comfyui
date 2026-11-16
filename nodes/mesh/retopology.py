"""Rebuild a 3D model's parts as clean quads that follow its shape, creases and borders."""

from __future__ import annotations

import sys

from comfy_api.latest import io

from ...modules.compat.types import FILE_3D, FILE_3D_FBX, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_OBJ, MODEL_3D
from ...modules.interface.progress import progress_bar
from ...modules.mesh.retopo import pipeline
from ...modules.mesh.retopo.attributes import UV_MODES
from ...modules.model3d import edit, inputs, report

#: Fewest faces a retopology may ask for.
MIN_TARGET_FACES = 24

#: Most relaxation passes.
MAX_SMOOTHING = 20

#: Largest crease angle, in degrees.
MAX_CREASE_ANGLE = 180.0


class MeshRetopology(io.ComfyNode):
    """Rebuild a model's surface as quads laid out along its shape."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASMeshRetopology",
            display_name="Mesh Retopology",
            search_aliases=[
                "WASMeshRetopology", "Mesh Retopology", "retopology", "retopo", "quad remesh", "quadrangulate",
                "quads", "remesh to quads", "auto retopology", "quad mesh", "clean topology", "edge flow",
            ],
            category="WAS Suite/3D/Mesh",
            description=(
                "Rebuild a dense or messy model, such as a scan or a generated character, as an even grid of "
                "quads that follows its shape, creases and open edges. Skin weights and shape keys carry over, "
                "so the result can be rigged, animated, subdivided or unwrapped and baked."
            ),
            inputs=[
                io.MultiType.Input("model_3d", types=[MODEL_3D, io.Mesh, FILE_3D, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_FBX, FILE_3D_OBJ], tooltip="The model to work on: a 3D model from another 3D node, a mesh from core's mesh nodes, or a GLB, glTF, VRM, FBX or OBJ file from Load 3D or a 3D generator such as Tripo."),
                io.Int.Input(
                    "target_faces", default=8000, min=MIN_TARGET_FACES, max=pipeline.MAX_TARGET_FACES, step=100,
                    tooltip="Faces wanted across the chosen parts, shared out by surface area. 2000 suits a "
                            "distant prop; 8000 a game character; 30000 a close-up hero model.",
                ),
                io.Combo.Input(
                    "output", options=list(pipeline.OUTPUTS), default="quads",
                    tooltip="`quads` = quads only, for subdivision and sculpting; `quad dominant` = mostly quads "
                            "with a few triangles, closest to the requested count; `triangles` = the same layout "
                            "cut into triangles, for engines.",
                ),
                io.Float.Input(
                    "crease_angle", default=0.0, min=0.0, max=MAX_CREASE_ANGLE, step=1.0,
                    tooltip="Edges folded at least this many degrees become lines the quads follow, with hard "
                            "shading. 0 = off, for organic models; 30 to 45 for hard-surface models such as "
                            "armour, furniture or vehicles.",
                ),
                io.Combo.Input(
                    "symmetry", options=list(pipeline.SYMMETRIES), default="none",
                    tooltip="`x` = the layout on the -X half mirrors the +X half, joined along x = 0, for a "
                            "character facing +Z centred on the origin; `none` = no mirroring. Parts off "
                            "centre are left unmirrored.",
                ),
                io.Float.Input(
                    "adaptivity", default=0.0, min=0.0, max=1.0, step=0.05,
                    tooltip="0 = even quads everywhere; 0.5 = smaller quads where the surface curves tightly "
                            "or is thin, such as fingers and ears; 1.0 = down to a quarter the size there.",
                ),
                io.Combo.Input(
                    "uvs", options=list(UV_MODES), default="drop",
                    tooltip="`drop` = no texture coordinates or textures, ready for an unwrap and a bake from "
                            "the original; `transfer` = each face keeps its texture from the original, with "
                            "seams where the original's texture pieces meet.",
                ),
                io.Boolean.Input(
                    "keep_borders", default=True, advanced=True,
                    tooltip="`true` = open edges, such as a neck opening or a sleeve end, stay exactly where "
                            "they are, with a row of quads along them; `false` = the layout may round them off.",
                ),
                io.Int.Input(
                    "smoothing", default=3, min=0, max=MAX_SMOOTHING, advanced=True,
                    tooltip="Passes evening out quad sizes along the surface, creases held. 0 = as extracted; "
                            "3 = even; 10 = very even, slightly rounder on tight curves.",
                ),
                io.String.Input("parts", default="", advanced=True, tooltip="Parts to work on: comma-separated names or labels, wildcards allowed, such as `head, hair*`. Empty works on every part."),
            ],
            outputs=[
                MODEL_3D.Output("model_3d", tooltip="The whole model: parts with their materials, and the skeleton, skin, springs, expressions and animations it carries."),
                io.Mesh.Output("mesh", tooltip="The same model as one mesh for core's mesh nodes and Save 3D Model. Parts are merged, their textures packed into one atlas, and the skeleton left out."),
                io.String.Output(
                    "report",
                    tooltip="Faces before and after, the share of quads, irregular vertices, how far corner "
                            "angles stray from 90 degrees and any duplicate, folded or non-manifold faces left, "
                            "per part when there are several.",
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model_3d, target_faces=8000, output="quads", crease_angle=0.0, symmetry="none",
                adaptivity=0.0, uvs="drop", keep_borders=True, smoothing=3, parts="") -> io.NodeOutput:
        """Retopologise the chosen parts, each with its share of the face budget.

        Raises:
            ValueError: Nothing is wired, no part matches ``parts``, the chosen parts have no
                triangles, an option is out of range, or the work would grow too large.
        """
        model = inputs.to_model(model_3d)
        before = report.figures(model)
        bar = progress_bar(100)
        shown = [0]
        management = sys.modules.get("comfy.model_management")
        interrupted = getattr(management, "throw_exception_if_processing_interrupted", None)

        def advance(fraction: float) -> None:
            if interrupted is not None:
                interrupted()
            step = int(round(min(max(fraction, 0.0), 1.0) * 100))
            if step > shown[0]:
                bar.update(step - shown[0])
                shown[0] = step

        model, summary = pipeline.retopologize_model(
            model, parts, int(target_faces), progress=advance,
            output=output if output in pipeline.OUTPUTS else "quads",
            crease_angle=min(max(float(crease_angle), 0.0), MAX_CREASE_ANGLE),
            keep_borders=bool(keep_borders),
            symmetry=symmetry if symmetry in pipeline.SYMMETRIES else "none",
            adaptivity=min(max(float(adaptivity), 0.0), 1.0),
            smoothing=min(max(int(smoothing), 0), MAX_SMOOTHING),
            uvs=uvs if uvs in UV_MODES else "drop",
        )
        figures = summary["figures"]
        polygons = int(figures["polygons"])
        interior = sum(int(figures[f"valence {name}"]) for name in ("3", "4", "5", "6", "7+"))
        irregular = int(figures["irregular vertices"])
        lines = [
            f"Mesh Retopology: {before['triangles']:,} triangles -> {polygons:,} faces "
            f"({100.0 * figures['quad share']:.1f}% quads, {int(figures['triangles']):,} triangles) "
            f"across {len(summary['parts'])} part{'s' if len(summary['parts']) != 1 else ''}",
            f"  {int(figures['vertices']):,} vertices, {irregular:,} irregular "
            f"({100.0 * irregular / max(int(figures['vertices']), 1):.1f}%); valence 3: "
            f"{int(figures['valence 3']):,}, 5: {int(figures['valence 5']):,}, 6 or more: "
            f"{int(figures['valence 6']) + int(figures['valence 7+']):,} of {interior:,} inner vertices",
            f"  corner angles stray {figures['angle error mean']:.1f} degrees from square on average; "
            f"{int(figures['border edges']):,} open edges; {figures['seconds']:.1f} s",
        ]
        flaws = {name: int(figures.get(name, 0)) for name in (
            "duplicate polygons", "folded polygons", "non-manifold edges", "non-manifold vertices", "degenerate polygons")}
        if any(flaws.values()):
            lines.append("  left to check in Mesh Inspect: " + ", ".join(
                f"{count:,} {name}" for name, count in flaws.items() if count))
        if summary["unmirrored"]:
            lines.append(f"  not mirrored, as they are off centre: {', '.join(summary['unmirrored'][:8])}")
        if summary["untextured"]:
            lines.append(f"  textures dropped with the texture coordinates: {', '.join(summary['untextured'][:8])}")
        if len(summary["per_part"]) > 1:
            for name, part_figures in summary["per_part"]:
                lines.append(f"  {name}: {int(part_figures['polygons']):,} faces, "
                             f"{100.0 * part_figures['quad share']:.1f}% quads")
        text = "\n".join(lines)
        report.publish(
            model, f"{before['triangles']:,} triangles -> {polygons:,} faces, "
                   f"{100.0 * figures['quad share']:.1f}% quads",
            before=before,
            counts={"faces": polygons, "quad share %": round(100.0 * figures["quad share"], 2),
                    "irregular vertices": irregular, "angle error deg": figures["angle error mean"],
                    "open edges": int(figures["border edges"]), "seconds": figures["seconds"]},
            facts={"output": output, "symmetry": symmetry, "uvs": uvs, "parts": edit.part_names(model, 8)},
        )
        hidden = cls.hidden
        wanted = inputs.mesh_wanted(getattr(hidden, "prompt", None), getattr(hidden, "unique_id", None))
        return io.NodeOutput(*inputs.node_output(model, mesh=wanted), text)

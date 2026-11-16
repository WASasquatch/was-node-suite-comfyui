"""Reduce the triangle count of a 3D model's parts, keeping textures, normals, skin and shape keys."""

from __future__ import annotations

import math
import sys

import torch
from comfy_api.latest import io

from ...modules.compat.types import FILE_3D, FILE_3D_FBX, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_OBJ, MODEL_3D
from ...modules.interface.progress import progress_bar
from ...modules.mesh import arrays
from ...modules.mesh import decimate as mesh_decimate
from ...modules.model3d import edit, inputs, report, selection

#: What the reduction aims for, as the node shows it.
TARGETS = ("ratio", "triangles", "error")

#: Vertex placement choices, as the node shows them.
PLACEMENTS = ("auto", "optimal", "keep vertices")

#: Reduction methods, as the node shows them.
METHODS = ("edge collapse", "grid cluster")

#: Fewest triangles a reduction aims for.
MIN_TRIANGLES = 4

#: Why a reduction stopped, in words.
STOPS = {
    "target": "reached the target",
    "error": "stopped at max_error; raise it to reduce further",
    "locked": "no vertex left that may move; set borders or seams to slide or cross to reduce further",
    "blocked": "every collapse left would fold or tear the surface",
    "rounds": "stopped at the round limit",
    "grid": "grid cluster",
}


class MeshDecimate(io.ComfyNode):
    """Reduce a model's triangle count while keeping its look and what it carries."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASMeshDecimate",
            display_name="Mesh Decimate",
            search_aliases=[
                "WASMeshDecimate", "Mesh Decimate", "decimate", "simplify mesh", "reduce polygons",
                "polycount", "polygon reduction", "low poly", "lod", "optimize mesh", "reduce triangles",
            ],
            category="WAS Suite/3D/Mesh",
            description=(
                "Cut a model's triangle count while keeping its shape, textures, normals, skin weights and "
                "shape keys, so a dense scan or generated model becomes light enough for games, VRM avatars "
                "or the web. Texture seams and open edges stay where they are unless set otherwise."
            ),
            inputs=[
                io.MultiType.Input("model_3d", types=[MODEL_3D, io.Mesh, FILE_3D, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_FBX, FILE_3D_OBJ], tooltip="The model to work on: a 3D model from another 3D node, a mesh from core's mesh nodes, or a GLB, glTF, VRM, FBX or OBJ file from Load 3D or a 3D generator such as Tripo."),
                io.Combo.Input(
                    "target", options=list(TARGETS), default="ratio",
                    tooltip="`ratio` = keep that share of the triangles; `triangles` = stop at that count; "
                            "`error` = reduce until any further step would move the surface more than max_error.",
                ),
                io.Float.Input(
                    "ratio", default=0.25, min=0.001, max=1.0, step=0.01,
                    tooltip="Share of the triangles kept when target is `ratio`. 0.5 = half; 0.25 = a quarter; "
                            "1.0 = unchanged.",
                ),
                io.Int.Input(
                    "triangles", default=20000, min=MIN_TRIANGLES, max=mesh_decimate.MAX_TRIANGLES,
                    tooltip="Triangles kept across the chosen parts when target is `triangles`. 20000 suits a "
                            "game character; 5000 a background prop.",
                ),
                io.Float.Input(
                    "max_error", default=0.01, min=0.0001, max=1.0, step=0.0001,
                    tooltip="Largest change to the surface when target is `error`, as a share of the model's size. "
                            "0.01 = 1 cm on a 1 m model; 0.001 = 1 mm. `ratio` and `triangles` reach their count "
                            "whatever the change.",
                ),
                io.Combo.Input(
                    "placement", options=list(PLACEMENTS), default="auto",
                    tooltip="Where merged vertices go. `optimal` = where they best fit the surface; `keep vertices` "
                            "= on an original vertex, so skin and shape keys stay exact; `auto` = keep vertices on "
                            "skinned or shape-keyed parts, optimal elsewhere.",
                ),
                io.Combo.Input(
                    "borders", options=list(mesh_decimate.BORDER_MODES), default="slide",
                    tooltip="Open edges, such as a neck opening or a cloak's rim. `slide` = simplified along "
                            "themselves, keeping their outline; `lock` = never moved, so the part still meets "
                            "whatever it is joined to.",
                ),
                io.Combo.Input(
                    "seams", options=list(mesh_decimate.SEAM_MODES), default="slide",
                    tooltip="Where texture coordinates, hard normals or parts split the surface. `slide` = "
                            "simplified only along themselves; `lock` = never moved; `cross` = may be collapsed "
                            "across, texture stretch counted as error, for textures cut into many small islands.",
                ),
                io.Combo.Input(
                    "symmetry", options=list(mesh_decimate.SYMMETRY), default="none",
                    tooltip="`x` = vertices mirrored across x = 0 are reduced together, so a symmetric model "
                            "stays symmetric; `none` = no constraint.",
                ),
                io.Combo.Input(
                    "method", options=list(METHODS), default="edge collapse", advanced=True,
                    tooltip="`edge collapse` = careful reduction keeping seams, borders and the surface's "
                            "topology; `grid cluster` = snaps vertices to a grid for extreme cuts such as distant "
                            "LODs, fast but able to join nearby pieces.",
                ),
                io.String.Input("parts", default="", advanced=True, tooltip="Parts to work on: comma-separated names or labels, wildcards allowed, such as `head, hair*`. Empty works on every part."),
            ],
            outputs=[
                MODEL_3D.Output("model_3d", tooltip="The whole model: parts with their materials, and the skeleton, skin, springs, expressions and animations it carries."),
                io.Mesh.Output("mesh", tooltip="The same model as one mesh for core's mesh nodes and Save 3D Model. Parts are merged, their textures packed into one atlas, and the skeleton left out."),
                io.String.Output(
                    "report",
                    tooltip="Triangles before and after, the largest surface change as a share of the model's "
                            "size, and why the reduction stopped.",
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model_3d, target="ratio", ratio=0.25, triangles=20000, max_error=0.01, placement="auto",
                borders="slide", seams="slide", symmetry="none", method="edge collapse",
                parts="") -> io.NodeOutput:
        """Reduce the chosen parts together, so parts that meet stay joined.

        Raises:
            ValueError: Nothing is wired, no part matches ``parts``, the chosen parts have no
                triangles, or they are larger than one reduction takes.
        """
        model = inputs.to_model(model_3d)
        before = report.figures(model)
        chosen = selection.selected(model, parts)
        if not chosen:
            raise ValueError(f"no part matches '{str(parts)[:80]}'; the model's parts are {edit.part_names(model)}. "
                             "Leave parts empty to decimate every part.")
        chosen = [index for index in chosen if model.parts[index].triangle_count]
        if not chosen:
            raise ValueError("the chosen parts have no triangles to decimate; wire a model with faces or choose "
                             "other parts")
        joint_count = model.skeleton.count if model.skeleton is not None else 0
        skinned = [joint_count > 0 and model.parts[index].weights is not None for index in chosen]
        total = sum(model.parts[index].triangle_count for index in chosen)
        carried = sum(model.parts[index].vertex_count
                      * ((joint_count if is_skinned else 0) + 3 * len(model.parts[index].morphs) + 4)
                      for index, is_skinned in zip(chosen, skinned))
        mesh_decimate.check_size(total, carried)

        if target == "triangles":
            goal = min(int(triangles), total)
        elif target == "error":
            goal = 0
        else:
            goal = min(max(MIN_TRIANGLES, int(math.ceil(total * float(ratio)))), total)
        meshes = [arrays.from_part(model.parts[index], joint_count if is_skinned else 0)
                  for index, is_skinned in zip(chosen, skinned)]
        placements = []
        for index, is_skinned in zip(chosen, skinned):
            if placement in ("optimal", "keep vertices"):
                placements.append(placement)
            else:
                placements.append("keep vertices" if is_skinned or model.parts[index].morphs else "optimal")
        low, high = model.bounds()
        size = float((high - low).max()) or 1.0
        centre = 0.5 * (low + high)
        wanted = set(chosen)
        others = [part.vertices for index, part in enumerate(model.parts) if index not in wanted and part.vertex_count]
        locked = torch.cat(others, 0) if others else None

        bar = progress_bar(100)
        shown = [0]
        interrupted = getattr(sys.modules.get("comfy.model_management"),
                              "throw_exception_if_processing_interrupted", None)

        def advance(fraction: float) -> None:
            if interrupted is not None:
                interrupted()
            step = int(round(min(max(fraction, 0.0), 1.0) * 100))
            if step > shown[0]:
                bar.update(step - shown[0])
                shown[0] = step

        if method == "grid cluster":
            results, stats = mesh_decimate.cluster(
                meshes, target_triangles=goal, max_error=float(max_error),
                symmetry=symmetry if symmetry in mesh_decimate.SYMMETRY else "none",
                size=size, centre=centre, progress=advance,
            )
        else:
            results, stats = mesh_decimate.decimate(
                meshes, target_triangles=goal, max_error=float(max_error) if target == "error" else math.inf,
                placement=placements,
                borders=borders if borders in mesh_decimate.BORDER_MODES else "slide",
                seams=seams if seams in mesh_decimate.SEAM_MODES else "slide",
                symmetry=symmetry if symmetry in mesh_decimate.SYMMETRY else "none",
                locked_points=locked, size=size, centre=centre, progress=advance,
            )
        kept_whole = []
        for index, result in zip(chosen, results):
            part = model.parts[index]
            if result.face_count == 0:
                kept_whole.append(part.name)
                continue
            model.parts[index] = arrays.to_part(part, result)

        after = sum(model.parts[index].triangle_count for index in chosen)
        share = 100.0 * after / max(total, 1)
        reason = STOPS.get(stats.stop, stats.stop)
        if stats.stop in ("error", "locked", "blocked") and goal and method != "grid cluster" and seams != "cross":
            reason += "; seams at `cross` reaches lower counts on textures cut into many islands"
        lines = [
            f"Mesh Decimate: {total:,} -> {after:,} triangles ({share:.1f}%) across {len(chosen)} "
            f"part{'s' if len(chosen) != 1 else ''}",
            f"  largest change {100.0 * stats.error:.3f}% of the model's size; {reason}",
            f"  {stats.collapses:,} collapses in {stats.rounds} rounds, {stats.seconds:.1f} s on {stats.device}",
        ]
        if stats.mirrored >= 0:
            lines.append(f"  {100.0 * stats.mirrored:.1f}% of vertices had a mirror partner")
        if kept_whole:
            lines.append(f"  left unchanged, as reducing them left no faces: {', '.join(kept_whole[:8])}")
        if len(chosen) > 1:
            for index in chosen:
                part = model.parts[index]
                lines.append(f"  {part.name or 'part ' + str(index)}: {part.triangle_count:,} triangles")
        text = "\n".join(lines)
        report.publish(
            model, f"{total:,} -> {after:,} triangles ({share:.1f}%)", before=before,
            counts={"triangles removed": total - after, "largest change %": round(100.0 * stats.error, 4),
                    "rounds": stats.rounds},
            facts={"stopped": reason, "method": method, "parts": edit.part_names(model, 8)},
        )
        hidden = cls.hidden
        wanted_mesh = inputs.mesh_wanted(getattr(hidden, "prompt", None), getattr(hidden, "unique_id", None))
        return io.NodeOutput(*inputs.node_output(model, mesh=wanted_mesh), text)

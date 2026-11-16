"""Repair the parts of a 3D model: weld, remove junk faces and floaters, fix non-manifold spots, orient, fill holes."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import FILE_3D, FILE_3D_FBX, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_OBJ, MODEL_3D
from ...modules.mesh import clean as mesh_clean
from ...modules.model3d import edit, inputs, report, selection
from ...modules.model3d.merge import reindex_parts

#: Choices of the self-intersection step, as the node shows them.
INTERSECTIONS = ("off", "count", "remove small shells")

#: Counts published as the node's own figures, by the name shown.
HEADLINE = (
    ("holes filled", "holes_filled"),
    ("shells removed", "small_shells"),
    ("non-manifold split", "non_manifold_edges"),
    ("vertices welded", "welded"),
)


class MeshClean(io.ComfyNode):
    """Repair a model's parts so they export clean and, where possible, watertight."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASMeshClean",
            display_name="Mesh Clean",
            search_aliases=[
                "WASMeshClean", "Mesh Clean", "repair mesh", "fix mesh", "watertight", "fill holes",
                "non-manifold", "weld vertices", "remove floaters", "merge by distance", "recalculate normals",
            ],
            category="WAS Suite/3D/Mesh",
            description=(
                "Repair a model so it exports clean: weld split vertices, remove broken faces and stray "
                "floating pieces, fix non-manifold spots, turn faces outward and fill small holes. "
                "Textures, skin and shape keys carry through, and the report lists what each step changed."
            ),
            inputs=[
                io.MultiType.Input("model_3d", types=[MODEL_3D, io.Mesh, FILE_3D, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_FBX, FILE_3D_OBJ], tooltip="The model to work on: a 3D model from another 3D node, a mesh from core's mesh nodes, or a GLB, glTF, VRM, FBX or OBJ file from Load 3D or a 3D generator such as Tripo."),
                io.Boolean.Input(
                    "weld", default=True,
                    tooltip="`true` = merge vertices within weld_distance whose texture coordinates, colours, "
                            "normals and skin agree; texture seams stay split. `false` = leave vertices as they are.",
                ),
                io.Float.Input(
                    "weld_distance", default=0.00001, min=0.0, max=1.0, step=0.00001,
                    tooltip="Largest gap between merged vertices, in metres. 0 = exact duplicates only; "
                            "0.00001 = a hundredth of a millimetre, closing rounding cracks; 0.001 = also hairline "
                            "gaps between pieces, collapsing any smaller triangles.",
                ),
                io.Boolean.Input(
                    "remove_degenerate", default=True,
                    tooltip="`true` = remove faces collapsed to a line or a point, and faces repeating another "
                            "face's corners.",
                ),
                io.Boolean.Input(
                    "remove_small_shells", default=True,
                    tooltip="`true` = remove loose pieces with fewer faces than shell_min_faces or less surface "
                            "than shell_min_area, such as the floaters 3D generators leave. The largest piece always stays.",
                ),
                io.Boolean.Input(
                    "fix_non_manifold", default=True,
                    tooltip="`true` = remove small loose flaps hanging off edges, then split edges shared by more "
                            "than two faces and vertices where separate fans of faces touch.",
                ),
                io.Boolean.Input(
                    "orient_faces", default=True,
                    tooltip="`true` = wind every face like its neighbours and turn each piece to face outward. "
                            "`false` = keep the winding as loaded.",
                ),
                io.Boolean.Input(
                    "fill_holes", default=True,
                    tooltip="`true` = close holes up to max_hole_edges with new faces curved to follow the "
                            "surface around them, textured from the border.",
                ),
                io.Int.Input(
                    "max_hole_edges", default=100, min=3, max=1000,
                    tooltip="Longest hole filled, counted in border edges. 100 = the gaps generators leave; "
                            "1000 = also large openings such as a neck or a sleeve.",
                ),
                io.Combo.Input(
                    "self_intersections", options=list(INTERSECTIONS), default="off",
                    tooltip="`off` = skip. `count` = count faces crossing another face, in the report. "
                            "`remove small shells` = also remove crossing pieces smaller than "
                            "intersecting_max_area, such as shapes buried inside the body.",
                ),
                io.Boolean.Input(
                    "recompute_normals", default=False,
                    tooltip="`true` = rebuild normals from the faces, smooth across edges flatter than "
                            "crease_angle and sharp at steeper ones. `false` = keep the model's normals.",
                ),
                io.Float.Input(
                    "crease_angle", default=60.0, min=0.0, max=180.0, step=1.0,
                    tooltip="Angle between faces, in degrees, at which recompute_normals keeps an edge sharp. "
                            "30 = hard surface; 60 = most models; 180 = smooth everywhere.",
                ),
                io.Int.Input(
                    "shell_min_faces", default=8, min=0, max=1000000, advanced=True,
                    tooltip="Loose pieces, and flaps hanging off edges, with fewer faces are removed. "
                            "0 = judge by area only.",
                ),
                io.Float.Input(
                    "shell_min_area", default=0.0001, min=0.0, max=0.5, step=0.0001, advanced=True,
                    tooltip="Loose pieces and flaps with less than this share of the part's surface are removed. "
                            "0.0001 = one hundredth of a percent; 0 = judge by face count only.",
                ),
                io.Float.Input(
                    "intersecting_max_area", default=0.05, min=0.0, max=1.0, step=0.01, advanced=True,
                    tooltip="With self_intersections at `remove small shells`, crossing pieces under this share "
                            "of the part's surface are removed. 0.05 = 5%. The largest piece always stays.",
                ),
                io.Boolean.Input(
                    "remove_unreferenced", default=True, advanced=True,
                    tooltip="`true` = drop vertices no face uses. `false` = keep them.",
                ),
                io.String.Input("parts", default="", advanced=True, tooltip="Parts to work on: comma-separated names or labels, wildcards allowed, such as `head, hair*`. Empty works on every part."),
            ],
            outputs=[
                MODEL_3D.Output("model_3d", tooltip="The whole model: parts with their materials, and the skeleton, skin, springs, expressions and animations it carries."),
                io.Mesh.Output("mesh", tooltip="The same model as one mesh for core's mesh nodes and Save 3D Model. Parts are merged, their textures packed into one atlas, and the skeleton left out."),
                io.String.Output(
                    "report",
                    tooltip="What each step changed, part by part, ending with whether the result is watertight "
                            "and what still keeps it open.",
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model_3d, weld=True, weld_distance=0.00001, remove_degenerate=True, remove_small_shells=True,
                fix_non_manifold=True, orient_faces=True, fill_holes=True, max_hole_edges=100,
                self_intersections="off", recompute_normals=False, crease_angle=60.0, shell_min_faces=8,
                shell_min_area=0.0001, intersecting_max_area=0.05, remove_unreferenced=True,
                parts="") -> io.NodeOutput:
        """Repair every chosen part and report what changed.

        Raises:
            ValueError: Nothing is wired, no part matches ``parts``, or the chosen parts have no triangles.
        """
        model = inputs.to_model(model_3d)
        before = report.figures(model)
        chosen = selection.selected(model, parts)
        if not chosen:
            raise ValueError(f"no part matches '{str(parts)[:80]}'; the model's parts are {edit.part_names(model)}. "
                             "Leave parts empty to clean every part.")
        chosen = [index for index in chosen if model.parts[index].triangle_count]
        if not chosen:
            raise ValueError("the chosen parts have no triangles to clean; wire a model with faces or choose "
                             "other parts")
        settings = mesh_clean.CleanSettings(
            weld=bool(weld), weld_distance=float(weld_distance), degenerate=bool(remove_degenerate),
            unreferenced=bool(remove_unreferenced), shells=bool(remove_small_shells),
            shell_min_faces=int(shell_min_faces), shell_min_area=float(shell_min_area),
            non_manifold=bool(fix_non_manifold), orient=bool(orient_faces), fill=bool(fill_holes),
            max_hole_edges=int(max_hole_edges),
            intersections=self_intersections if self_intersections in INTERSECTIONS else "off",
            intersecting_max_area=float(intersecting_max_area), normals=bool(recompute_normals),
            crease_angle=float(crease_angle),
        )
        totals = dict.fromkeys(mesh_clean.COUNT_NAMES, 0)
        lines, kept, source, emptied = [], [], [], []
        wanted = set(chosen)
        for index, part in enumerate(model.parts):
            if index not in wanted:
                kept.append(part)
                source.append([index])
                continue
            repaired, counts = mesh_clean.clean_part(part, settings)
            for name, value in counts.items():
                totals[name] += value
            if len(chosen) > 1:
                lines.append(f"{part.name or 'part ' + str(index)}: " + "; ".join(mesh_clean.describe(counts)))
            if repaired.triangle_count == 0:
                emptied.append(part.name)
                continue
            kept.append(repaired)
            source.append([index])
        reindex_parts(model, kept, source)
        summary_lines = mesh_clean.describe(totals)
        if emptied:
            summary_lines.insert(-1, f"{len(emptied)} parts left with no faces were removed: {', '.join(emptied[:8])}")
        heading = f"Mesh Clean: {len(chosen)} part{'s' if len(chosen) != 1 else ''}"
        text = "\n".join([heading, *("  " + line for line in summary_lines), *lines])
        whole = mesh_clean.watertight(totals)
        verdict = summary_lines[-1]
        report.publish(
            model, "Watertight" if whole else verdict.replace("Watertight: no; ", "Not watertight: "),
            before=before,
            counts={label: totals[name] for label, name in HEADLINE},
            facts={"watertight": "yes" if whole else "no", "repairs": "; ".join(summary_lines[:-1]),
                   "parts": edit.part_names(model, 8)},
        )
        hidden = cls.hidden
        wanted_mesh = inputs.mesh_wanted(getattr(hidden, "prompt", None), getattr(hidden, "unique_id", None))
        return io.NodeOutput(*inputs.node_output(model, mesh=wanted_mesh), text)

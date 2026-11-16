"""Rebuild the surface of a 3D model's parts with even triangles, keeping its features, texture and skin."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import FILE_3D, FILE_3D_FBX, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_OBJ, MODEL_3D
from ...modules.interface.progress import progress_bar
from ...modules.mesh import remesh as mesh_remesh
from ...modules.mesh import topology
from ...modules.model3d import edit, inputs, report, selection

#: Remeshing modes as the node shows them.
MODES = ("uniform", "adaptive")

#: Ways of setting the triangle size.
SIZE_BY = ("faces", "edge length")

#: Fewest faces given to one part when a face count is shared out.
MIN_PART_FACES = 16

#: Steps the progress bar is divided into per part.
PROGRESS_STEPS = 100

#: Share of a part's vertices held still on seams and corners past which the report suggests freeing seams.
HELD_WARNING = 0.4


class MeshRemesh(io.ComfyNode):
    """Remesh a model's parts towards even triangles on their own surface."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASMeshRemesh",
            display_name="Mesh Remesh",
            search_aliases=[
                "WASMeshRemesh", "Mesh Remesh", "remesh", "isotropic remesh", "retriangulate", "even triangles",
                "adaptive remesh", "uniform mesh", "triangle quality", "resample mesh",
            ],
            category="WAS Suite/3D/Mesh",
            description=(
                "Rebuild a model's surface with even, well shaped triangles of one size, or smaller ones where "
                "the surface bends sharply. Sharp edges, open borders and texture seams stay where they were, "
                "so the texture, skin and shape keys carry over onto the new triangles."
            ),
            inputs=[
                io.MultiType.Input("model_3d", types=[MODEL_3D, io.Mesh, FILE_3D, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_FBX, FILE_3D_OBJ], tooltip="The model to work on: a 3D model from another 3D node, a mesh from core's mesh nodes, or a GLB, glTF, VRM, FBX or OBJ file from Load 3D or a 3D generator such as Tripo."),
                io.Combo.Input(
                    "mode", options=list(MODES), default="uniform",
                    tooltip="`uniform` = every edge close to one length, for sculpting, simulation and clean "
                            "wireframes; `adaptive` = shorter edges where the surface bends sharply and longer on "
                            "flat areas, keeping detail for the same face count.",
                ),
                io.Combo.Input(
                    "size_by", options=list(SIZE_BY), default="faces",
                    tooltip="`faces` = aim for the faces count across the chosen parts, shared out by surface "
                            "area; `edge length` = aim for edge_length on every part, so the count follows the "
                            "model's size.",
                ),
                io.Int.Input(
                    "faces", default=20000, min=MIN_PART_FACES, max=mesh_remesh.MAX_FACES, step=100,
                    tooltip="Triangles wanted across the chosen parts when size_by is `faces`; the result "
                            "usually lands within a few percent. 20000 suits a game character; 100000 a close-up "
                            "or sculpting base.",
                ),
                io.Float.Input(
                    "edge_length", default=0.01, min=0.0001, max=100.0, step=0.0001,
                    tooltip="Edge length wanted when size_by is `edge length`, in the model's units, metres for "
                            "most models. 0.01 = 1 cm edges; halving it gives about four times the faces.",
                ),
                io.Float.Input(
                    "adaptivity", default=0.5, min=0.0, max=1.0, step=0.05,
                    tooltip="How far `adaptive` mode lets edge length follow curvature. 0 = uniform; 0.5 = up "
                            "to 2.5 times shorter in tight curves; 1 = up to 4 times. Ignored in `uniform` mode.",
                ),
                io.Float.Input(
                    "feature_angle", default=45.0, min=0.0, max=180.0, step=1.0,
                    tooltip="Edges where the surface folds by more than this many degrees are kept as sharp "
                            "lines the new triangles follow. 45 = clear creases; 30 keeps softer ones; 180 = "
                            "only open borders and seams. Lines shorter than two edges are ignored.",
                ),
                io.Boolean.Input(
                    "keep_uv_seams", default=True,
                    tooltip="`true` = texture seams stay as edges and each new triangle takes its texture from "
                            "one side, so textures still fit. `false` = remesh across seams; texture coordinates "
                            "and textures are dropped, so follow with Mesh UV Unwrap and Mesh Bake Textures.",
                ),
                io.Int.Input(
                    "iterations", default=8, min=1, max=mesh_remesh.MAX_ITERATIONS, step=1, advanced=True,
                    tooltip="Passes of split, collapse, flip, relax and project. 8 = settled triangles; 4 = "
                            "faster and a little rougher; more rarely helps.",
                ),
                io.String.Input("parts", default="", advanced=True, tooltip="Parts to work on: comma-separated names or labels, wildcards allowed, such as `head, hair*`. Empty works on every part."),
            ],
            outputs=[
                MODEL_3D.Output("model_3d", tooltip="The whole model: parts with their materials, and the skeleton, skin, springs, expressions and animations it carries."),
                io.Mesh.Output("mesh", tooltip="The same model as one mesh for core's mesh nodes and Save 3D Model. Parts are merged, their textures packed into one atlas, and the skeleton left out."),
                io.String.Output(
                    "report",
                    tooltip="Faces before and after, the edge length used, the sharp lines and held vertices "
                            "found, and the time, part by part.",
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model_3d, mode="uniform", size_by="faces", faces=20000, edge_length=0.01, adaptivity=0.5,
                feature_angle=45.0, keep_uv_seams=True, iterations=8, parts="") -> io.NodeOutput:
        """Remesh every chosen part and report what changed.

        Raises:
            ValueError: Nothing is wired, no part matches ``parts``, the chosen parts have no
                triangles, or the size asked for would pass the face limit.
        """
        model = inputs.to_model(model_3d)
        before = report.figures(model)
        chosen = selection.selected(model, parts)
        if not chosen:
            raise ValueError(f"no part matches '{str(parts)[:80]}'; the model's parts are {edit.part_names(model)}. "
                             "Leave parts empty to remesh every part.")
        areas = {}
        for index in chosen:
            part = model.parts[index]
            if part.triangle_count:
                areas[index] = float(topology.face_areas(part.vertices.double(), part.faces.long()).sum())
        chosen = [index for index in chosen if areas.get(index, 0.0) > 0]
        if not chosen:
            raise ValueError("the chosen parts have no triangles with any area to remesh; wire a model with faces "
                             "or choose other parts")
        by_faces = size_by != "edge length"
        total_area = sum(areas[index] for index in chosen)
        if by_faces:
            wanted = {index: max(MIN_PART_FACES, round(int(faces) * areas[index] / total_area)) for index in chosen}
        else:
            expected = sum(mesh_remesh.estimate_faces(areas[index], float(edge_length)) for index in chosen)
            if expected > mesh_remesh.MAX_FACES:
                raise ValueError(
                    f"an edge length of {float(edge_length):g} would give about {expected:,} faces, over the limit "
                    f"of {mesh_remesh.MAX_FACES:,}; raise edge_length, choose fewer parts, or set size_by to faces"
                )
        adaptive = 0.0 if mode != "adaptive" else float(adaptivity)
        joint_count = model.skeleton.count if model.skeleton is not None else 0

        bar = progress_bar(PROGRESS_STEPS * len(chosen))
        shown = [0]
        try:
            from comfy.model_management import throw_exception_if_processing_interrupted
        except ImportError:
            throw_exception_if_processing_interrupted = None

        lines, totals = [], {"faces before": 0, "faces after": 0, "feature lines": 0, "held vertices": 0}
        seconds = 0.0
        crowded = []
        for number, index in enumerate(chosen):
            part = model.parts[index]

            def advance(fraction: float, number=number) -> None:
                if throw_exception_if_processing_interrupted is not None:
                    throw_exception_if_processing_interrupted()
                step = int(round((number + min(max(fraction, 0.0), 1.0)) * PROGRESS_STEPS))
                if step > shown[0]:
                    bar.update(step - shown[0])
                    shown[0] = step

            options = {"faces": wanted[index]} if by_faces else {"length": float(edge_length)}
            new_part, stats = mesh_remesh.remesh_part(
                part, joint_count, adaptivity=adaptive, iterations=int(iterations),
                feature_angle=float(feature_angle), keep_seams=bool(keep_uv_seams), progress=advance, **options,
            )
            model.parts[index] = new_part
            advance(1.0)
            totals["faces before"] += part.triangle_count
            totals["faces after"] += new_part.triangle_count
            totals["feature lines"] += int(stats["feature_chains"])
            totals["held vertices"] += int(stats["held_vertices"])
            seconds += float(stats["seconds"])
            welded = max(int(stats.get("welded_vertices", new_part.vertex_count)), 1)
            if keep_uv_seams and stats.get("regions", 0) and stats["held_vertices"] > HELD_WARNING * welded:
                crowded.append(part.name or f"part {index}")
            lines.append(
                f"  {part.name or 'part ' + str(index)}: {part.triangle_count:,} -> {new_part.triangle_count:,} "
                f"triangles, edge {stats['target_length']:.4g}, {stats['feature_chains']:,} sharp lines, "
                f"{stats['held_vertices']:,} held vertices, {stats['seconds']:.1f} s"
            )

        heading = (f"Mesh Remesh ({mode}): {totals['faces before']:,} -> {totals['faces after']:,} triangles across "
                   f"{len(chosen)} part{'s' if len(chosen) != 1 else ''} in {seconds:.1f} s")
        notes = []
        if crowded:
            notes.append(f"  most vertices of {', '.join(crowded[:6])} sit on texture seams and stay put; turn "
                         "keep_uv_seams off for even triangles, then re-unwrap and bake the textures")
        if not keep_uv_seams:
            notes.append("  texture coordinates and textures were dropped with keep_uv_seams off")
        text = "\n".join([heading, *lines, *notes])
        report.publish(
            model, f"{totals['faces before']:,} -> {totals['faces after']:,} triangles", before=before,
            counts={"faces wanted": int(faces) if by_faces else 0, "sharp lines": totals["feature lines"],
                    "held vertices": totals["held vertices"], "seconds": round(seconds, 2)},
            facts={"mode": mode, "size by": size_by, "seams": "kept" if keep_uv_seams else "dropped",
                   "parts": edit.part_names(model, 8)},
        )
        hidden = cls.hidden
        wanted_mesh = inputs.mesh_wanted(getattr(hidden, "prompt", None), getattr(hidden, "unique_id", None))
        return io.NodeOutput(*inputs.node_output(model, mesh=wanted_mesh), text)

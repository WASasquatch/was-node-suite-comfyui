"""Subdivide the faces of a 3D model's parts, smoothing or keeping the shape, with texture and skin carried."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import FILE_3D, FILE_3D_FBX, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_OBJ, MODEL_3D
from ...modules.interface.progress import progress_bar
from ...modules.mesh import arrays
from ...modules.mesh import subdivide as mesh_subdivide
from ...modules.model3d import edit, inputs, report, selection

#: Schemes as the node shows them, and the name each has in the subdivision module.
SCHEMES = {"loop": "loop", "catmull-clark": "catmull_clark", "midpoint": "midpoint"}


class MeshSubdivide(io.ComfyNode):
    """Subdivide a model's parts one to three levels."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASMeshSubdivide",
            display_name="Mesh Subdivide",
            search_aliases=[
                "WASMeshSubdivide", "Mesh Subdivide", "subdivision surface", "subsurf", "catmull-clark", "loop",
                "smooth subdivide", "add resolution", "quads", "tessellate",
            ],
            category="WAS Suite/3D/Mesh",
            description=(
                "Split every face into four to add resolution or round off a low poly model. The smooth schemes "
                "round the surface, one of them turning it into quads; midpoint keeps the shape exactly. Texture "
                "coordinates, colours, skin and shape keys follow the new faces."
            ),
            inputs=[
                io.MultiType.Input("model_3d", types=[MODEL_3D, io.Mesh, FILE_3D, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_FBX, FILE_3D_OBJ], tooltip="The model to work on: a 3D model from another 3D node, a mesh from core's mesh nodes, or a GLB, glTF, VRM, FBX or OBJ file from Load 3D or a 3D generator such as Tripo."),
                io.Combo.Input(
                    "scheme", options=list(SCHEMES), default="loop",
                    tooltip="`loop` = smooth, keeps triangles; `catmull-clark` = smooth, every face becomes "
                            "quads, the usual choice before sculpting or retopology; `midpoint` = more faces on "
                            "exactly the same shape, for displacement or finer skin weights.",
                ),
                io.Int.Input(
                    "levels", default=1, min=1, max=3, step=1,
                    tooltip="Times to subdivide. Each level gives four times the faces: 1 = x4; 2 = x16; 3 = x64.",
                ),
                io.Float.Input(
                    "crease_angle", default=180.0, min=0.0, max=180.0, step=1.0,
                    tooltip="Edges where the surface folds by more than this many degrees stay sharp. 180 = only "
                            "open borders stay sharp; 60 keeps the hard edges of machines and furniture.",
                ),
                io.Boolean.Input(
                    "smooth_uvs", default=False, advanced=True,
                    tooltip="`false` = texture coordinates stay where they were on every face, so the texture "
                            "does not shift. `true` = smooth them inside each texture island, borders held, which "
                            "suits large hand-made islands.",
                ),
                io.Int.Input(
                    "max_faces", default=mesh_subdivide.MAX_FACES, min=1000, max=mesh_subdivide.LIMIT_FACES,
                    step=1000, advanced=True,
                    tooltip="Most triangles the chosen parts may have afterwards, such as 4000000; the node stops "
                            "with a message before building more. Lower levels or choose fewer parts to stay under it.",
                ),
                io.String.Input("parts", default="", advanced=True, tooltip="Parts to work on: comma-separated names or labels, wildcards allowed, such as `head, hair*`. Empty works on every part."),
            ],
            outputs=[
                MODEL_3D.Output("model_3d", tooltip="The whole model: parts with their materials, and the skeleton, skin, springs, expressions and animations it carries."),
                io.Mesh.Output("mesh", tooltip="The same model as one mesh for core's mesh nodes and Save 3D Model. Parts are merged, their textures packed into one atlas, and the skeleton left out."),
                io.String.Output(
                    "report",
                    tooltip="Faces before and after, quads made, and the time, part by part.",
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model_3d, scheme="loop", levels=1, crease_angle=180.0, smooth_uvs=False,
                max_faces=mesh_subdivide.MAX_FACES, parts="") -> io.NodeOutput:
        """Subdivide every chosen part after checking the result fits under ``max_faces``.

        Raises:
            ValueError: Nothing is wired, no part matches ``parts``, the chosen parts have no
                triangles, or the result would pass ``max_faces``.
        """
        import time

        name = SCHEMES.get(scheme)
        if name is None:
            raise ValueError(f"scheme '{str(scheme)[:40]}' is not one of {', '.join(SCHEMES)}")
        levels = max(1, min(int(levels), 3))
        limit = max(1, min(int(max_faces), mesh_subdivide.LIMIT_FACES))
        model = inputs.to_model(model_3d)
        before = report.figures(model)
        chosen = selection.selected(model, parts)
        if not chosen:
            raise ValueError(f"no part matches '{str(parts)[:80]}'; the model's parts are {edit.part_names(model)}. "
                             "Leave parts empty to subdivide every part.")
        chosen = [index for index in chosen if model.parts[index].triangle_count]
        if not chosen:
            raise ValueError("the chosen parts have no triangles to subdivide; wire a model with faces or choose "
                             "other parts")
        predicted = {index: mesh_subdivide.predicted_faces(arrays.from_part(model.parts[index], 0), name, levels)
                     for index in chosen}
        total = sum(predicted.values())
        if total > limit:
            raise ValueError(
                f"{levels} level{'s' if levels != 1 else ''} of {scheme} would make {total:,} triangles, over "
                f"max_faces of {limit:,}; lower levels, choose fewer parts, or raise max_faces"
            )
        joint_count = model.skeleton.count if model.skeleton is not None else 0
        bar = progress_bar(len(chosen))
        try:
            from comfy.model_management import throw_exception_if_processing_interrupted
        except ImportError:
            throw_exception_if_processing_interrupted = None

        lines = []
        faces_before = faces_after = quads = 0
        started = time.perf_counter()
        for index in chosen:
            if throw_exception_if_processing_interrupted is not None:
                throw_exception_if_processing_interrupted()
            part = model.parts[index]
            begun = time.perf_counter()
            new_part = mesh_subdivide.subdivide_part(
                part, joint_count, name, levels, float(crease_angle), predicted[index], bool(smooth_uvs),
            )
            model.parts[index] = new_part
            bar.update(1)
            made = int((new_part.polygons.sizes == 4).sum()) if new_part.polygons is not None else 0
            faces_before += part.triangle_count
            faces_after += new_part.triangle_count
            quads += made
            lines.append(
                f"  {part.name or 'part ' + str(index)}: {part.triangle_count:,} -> {new_part.triangle_count:,} "
                f"triangles" + (f", {made:,} quads" if made else "") + f", {time.perf_counter() - begun:.2f} s"
            )
        seconds = time.perf_counter() - started
        heading = (f"Mesh Subdivide ({scheme}, {levels} level{'s' if levels != 1 else ''}): {faces_before:,} -> "
                   f"{faces_after:,} triangles across {len(chosen)} part{'s' if len(chosen) != 1 else ''} in "
                   f"{seconds:.2f} s")
        text = "\n".join([heading, *lines])
        report.publish(
            model, f"{faces_before:,} -> {faces_after:,} triangles", before=before,
            counts={"levels": levels, "quads": quads, "seconds": round(seconds, 2)},
            facts={"scheme": scheme, "parts": edit.part_names(model, 8)},
        )
        hidden = cls.hidden
        wanted_mesh = inputs.mesh_wanted(getattr(hidden, "prompt", None), getattr(hidden, "unique_id", None))
        return io.NodeOutput(*inputs.node_output(model, mesh=wanted_mesh), text)

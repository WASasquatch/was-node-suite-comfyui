"""Smooth the surface of a 3D model's parts, keeping borders, sharp edges and volume where asked."""

from __future__ import annotations

from comfy_api.latest import io

from ...modules.compat.types import FILE_3D, FILE_3D_FBX, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_OBJ, MODEL_3D
from ...modules.interface.progress import progress_bar
from ...modules.mesh import smooth as mesh_smooth
from ...modules.model3d import edit, inputs, report, selection


class MeshSmooth(io.ComfyNode):
    """Smooth a model's parts by one of five methods."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="WASMeshSmooth",
            display_name="Mesh Smooth",
            search_aliases=[
                "WASMeshSmooth", "Mesh Smooth", "smooth mesh", "relax", "denoise mesh", "laplacian smooth",
                "taubin", "fair surface", "remove bumps", "scan cleanup",
            ],
            category="WAS Suite/3D/Mesh",
            description=(
                "Smooth a model's surface to take out noise, lumps and faceting. Borders and sharp edges can be "
                "held and the volume restored so it does not shrink, while texture coordinates, skin and shape "
                "keys stay as they were."
            ),
            inputs=[
                io.MultiType.Input("model_3d", types=[MODEL_3D, io.Mesh, FILE_3D, FILE_3D_GLB, FILE_3D_GLTF, FILE_3D_FBX, FILE_3D_OBJ], tooltip="The model to work on: a 3D model from another 3D node, a mesh from core's mesh nodes, or a GLB, glTF, VRM, FBX or OBJ file from Load 3D or a 3D generator such as Tripo."),
                io.Combo.Input(
                    "method", options=list(mesh_smooth.METHODS), default="taubin",
                    tooltip="`laplacian` = average each vertex with its neighbours, shrinks; `taubin` = smooth "
                            "without shrinking; `hc` = smooth while pulled back to the original shape; `implicit` "
                            "= strong, stable smoothing in few steps; `bilateral` = remove scan noise, keep sharp "
                            "edges.",
                ),
                io.Int.Input(
                    "iterations", default=10, min=0, max=mesh_smooth.MAX_ITERATIONS, step=1,
                    tooltip="Smoothing steps. 0 = no change; 10 = light; 50 or more flattens detail. `implicit` "
                            "needs only 1 to 3.",
                ),
                io.Float.Input(
                    "strength", default=0.5, min=0.0, max=1.0, step=0.05,
                    tooltip="Size of each step. 0 = no change; 0.5 = moderate; 1 = strongest. For `bilateral`, "
                            "higher also lets bigger bumps count as noise.",
                ),
                io.Boolean.Input(
                    "keep_borders", default=True,
                    tooltip="`true` = vertices on open borders stay where they are, so a cut neck or sleeve keeps "
                            "its outline and still meets the part beside it. `false` = borders smooth too.",
                ),
                io.Boolean.Input(
                    "keep_features", default=True,
                    tooltip="`true` = edges sharper than feature_angle stay sharp; vertices slide along them or "
                            "hold still. `false` = everything smooths alike.",
                ),
                io.Boolean.Input(
                    "keep_volume", default=False,
                    tooltip="`true` = afterwards, push each closed piece in or out along its normals until it "
                            "holds its earlier volume. Open pieces are left as smoothed.",
                ),
                io.Float.Input(
                    "feature_angle", default=45.0, min=0.0, max=180.0, step=1.0, advanced=True,
                    tooltip="Degrees of fold from which keep_features treats an edge as sharp. 45 = clear "
                            "creases; 30 keeps softer ones; 180 = none.",
                ),
                io.String.Input("parts", default="", advanced=True, tooltip="Parts to work on: comma-separated names or labels, wildcards allowed, such as `head, hair*`. Empty works on every part."),
            ],
            outputs=[
                MODEL_3D.Output("model_3d", tooltip="The whole model: parts with their materials, and the skeleton, skin, springs, expressions and animations it carries."),
                io.Mesh.Output("mesh", tooltip="The same model as one mesh for core's mesh nodes and Save 3D Model. Parts are merged, their textures packed into one atlas, and the skeleton left out."),
                io.String.Output(
                    "report",
                    tooltip="How far vertices moved on average and at most, the volume before and after, and the "
                            "vertices held still, part by part.",
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.unique_id],
        )

    @classmethod
    def execute(cls, model_3d, method="taubin", iterations=10, strength=0.5, keep_borders=True,
                keep_features=True, keep_volume=False, feature_angle=45.0, parts="") -> io.NodeOutput:
        """Smooth every chosen part and report how far it moved.

        Raises:
            ValueError: Nothing is wired, no part matches ``parts``, the chosen parts have no
                triangles, or the method is unknown.
        """
        if method not in mesh_smooth.METHODS:
            raise ValueError(f"method '{str(method)[:40]}' is not one of {', '.join(mesh_smooth.METHODS)}")
        model = inputs.to_model(model_3d)
        before = report.figures(model)
        chosen = selection.selected(model, parts)
        if not chosen:
            raise ValueError(f"no part matches '{str(parts)[:80]}'; the model's parts are {edit.part_names(model)}. "
                             "Leave parts empty to smooth every part.")
        chosen = [index for index in chosen if model.parts[index].triangle_count]
        if not chosen:
            raise ValueError("the chosen parts have no triangles to smooth; wire a model with faces or choose "
                             "other parts")
        bar = progress_bar(len(chosen))
        try:
            from comfy.model_management import throw_exception_if_processing_interrupted
        except ImportError:
            throw_exception_if_processing_interrupted = None

        lines = []
        mean_total = weight = 0.0
        largest = volume_before = volume_after = seconds = 0.0
        held = 0
        for index in chosen:
            if throw_exception_if_processing_interrupted is not None:
                throw_exception_if_processing_interrupted()
            part = model.parts[index]
            new_part, stats = mesh_smooth.smooth_part(
                part, method=method, iterations=int(iterations), strength=float(strength),
                keep_borders=bool(keep_borders), keep_features=bool(keep_features), keep_volume=bool(keep_volume),
                feature_angle=float(feature_angle),
            )
            model.parts[index] = new_part
            bar.update(1)
            mean_total += stats["mean_move"] * part.vertex_count
            weight += part.vertex_count
            largest = max(largest, stats["max_move"])
            volume_before += stats["volume_before"]
            volume_after += stats["volume_after"]
            held += stats["held_vertices"]
            seconds += stats["seconds"]
            lines.append(
                f"  {part.name or 'part ' + str(index)}: moved {stats['mean_move']:.4g} on average, "
                f"{stats['max_move']:.4g} at most; volume {stats['volume_before']:.5g} -> "
                f"{stats['volume_after']:.5g}; {stats['held_vertices']:,} held, "
                f"{stats['line_vertices']:,} sliding along edges"
            )
        mean_move = mean_total / max(weight, 1.0)
        change = 100.0 * (volume_after - volume_before) / abs(volume_before) if volume_before else 0.0
        heading = (f"Mesh Smooth ({method}, {int(iterations)} x {float(strength):g}): {len(chosen)} "
                   f"part{'s' if len(chosen) != 1 else ''}, mean move {mean_move:.4g}, volume {change:+.2f}%, "
                   f"{seconds:.2f} s")
        text = "\n".join([heading, *lines])
        report.publish(
            model, f"mean move {mean_move:.4g}, volume {change:+.2f}%", before=before,
            counts={"mean move": round(mean_move, 6), "largest move": round(largest, 6),
                    "volume change %": round(change, 3), "held vertices": held},
            facts={"method": method, "parts": edit.part_names(model, 8)},
        )
        hidden = cls.hidden
        wanted_mesh = inputs.mesh_wanted(getattr(hidden, "prompt", None), getattr(hidden, "unique_id", None))
        return io.NodeOutput(*inputs.node_output(model, mesh=wanted_mesh), text)
